#nullable enable
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace BotCore
// Ensure BuildMarketHubAsync and AttachLifecycleHandlers are inside the class
// (No structural changes needed, but add explicit region markers for clarity)
{
		public sealed class MarketHubClient : IAsyncDisposable
		{
			private readonly ILogger<MarketHubClient> _log;
			private readonly Func<Task<string?>> _getJwtAsync;

			private HubConnection? _conn;
			private string _contractId = string.Empty;

			private readonly SemaphoreSlim _subLock = new(1, 1);
			private volatile bool _subscribed;
			private volatile bool _firstQuoteLogged;
			private volatile bool _firstTradeLogged;

			private readonly System.Collections.Concurrent.ConcurrentDictionary<string, DateTime> _lastQuoteSeen = new();
			private readonly System.Collections.Concurrent.ConcurrentDictionary<string, DateTime> _lastBarSeen = new();

			// Throttling/backoff fields to reduce log spam during unstable connectivity bursts
			private static DateTime _lastClosedWarnUtc = DateTime.MinValue;
			private static DateTime _lastRebuiltInfoUtc = DateTime.MinValue;
			private static readonly TimeSpan _closedWarnInterval = TimeSpan.FromSeconds(GetEnvInt("MARKET_HUB_CLOSED_WARN_INTERVAL_SECONDS", 10));
			private static readonly TimeSpan _rebuiltInfoInterval = TimeSpan.FromSeconds(GetEnvInt("MARKET_HUB_REBUILT_INFO_INTERVAL_SECONDS", 10));

			private static int GetEnvInt(string name, int def)
			{
				var s = Environment.GetEnvironmentVariable(name);
				return int.TryParse(s, out var v) && v > 0 ? v : def;
			}

			private static bool Concise() => (Environment.GetEnvironmentVariable("APP_CONCISE_CONSOLE") ?? "true").Trim().ToLowerInvariant() is "1" or "true" or "yes";

			public event Action<string, JsonElement>? OnQuote;
			public event Action<string, JsonElement>? OnTrade;
			public event Action<string, JsonElement>? OnDepth;

 		public TimeSpan LastQuoteSeenAge(string contractId)
 			=> DateTime.UtcNow - (_lastQuoteSeen.TryGetValue(contractId, out var t) ? t : DateTime.MinValue);
 		public TimeSpan LastBarSeenAge(string contractId)
 			=> DateTime.UtcNow - (_lastBarSeen.TryGetValue(contractId, out var t) ? t : DateTime.MinValue);
 		public void RecordBarSeen(string contractId)
 			=> _lastBarSeen[contractId] = DateTime.UtcNow;

 		// Helpers for startup warmup checks
 		public bool HasRecentQuote(string contractId, int maxAgeSeconds = 10)
 			=> LastQuoteSeenAge(contractId) <= TimeSpan.FromSeconds(maxAgeSeconds);
 		public bool HasRecentBar(string contractId, string timeframe = "1m", int maxAgeSeconds = 90)
 			=> LastBarSeenAge(contractId) <= TimeSpan.FromSeconds(maxAgeSeconds);

		public MarketHubClient(ILogger<MarketHubClient> log, Func<Task<string?>> getJwtAsync)
		{
			_log = log;
			_getJwtAsync = getJwtAsync;
		}

		public async Task StartAsync(string contractId, CancellationToken ct = default)
		{
			_contractId = contractId ?? throw new ArgumentNullException(nameof(contractId));
			if (_conn is not null) throw new InvalidOperationException("MarketHubClient already started.");

			_conn = await BuildMarketHubAsync();
			AttachLifecycleHandlers(ct);

			await _conn.StartAsync(ct);
			_log.LogInformation("MarketHub connected. State={State}", _conn.State);
			await Task.Delay(200, ct);
			await SubscribeIfConnectedAsync(CancellationToken.None);
		}

		private async Task SubscribeIfConnectedAsync(CancellationToken ct)
		{
			if (_conn is null || string.IsNullOrWhiteSpace(_contractId)) return;
			if (_subscribed) { _log.LogDebug("[MarketHub] Already subscribed to {ContractId}", _contractId); return; }

			await _subLock.WaitAsync(ct);
			try
			{
				if (_subscribed) { _log.LogDebug("[MarketHub] Already subscribed to {ContractId} (post-lock)", _contractId); return; }
				if (_conn.State != HubConnectionState.Connected)
				{
					_log.LogDebug("Skip subscribe: State = {State}", _conn.State);
					return;
				}

				// Try a variety of known subscription methods for compatibility across hub versions
				async Task<bool> TryInvoke(string method, params object[] args)
				{
					try
					{
						await _conn!.InvokeAsync(method, args.Append(ct).ToArray());
						_log.LogInformation("[MarketHub] {Method}({Args}) ok", method, string.Join(",", args));
						return true;
					}
					catch (Exception ex)
					{
						_log.LogDebug("[MarketHub] {Method} unsupported: {Msg}", method, ex.Message);
						return false;
					}
				}

				bool any = false;
				// Quotes / trades / depth
				any |= await TryInvoke("SubscribeContractQuotes", _contractId);
				any |= await TryInvoke("SubscribeQuotes", _contractId);
				any |= await TryInvoke("subscribe", "quote", _contractId);
				any |= await TryInvoke("SubscribeContractTrades", _contractId);
				any |= await TryInvoke("SubscribeTrades", _contractId);
				any |= await TryInvoke("subscribe", "trade", _contractId);
				any |= await TryInvoke("SubscribeContractMarketDepth", _contractId);
				any |= await TryInvoke("SubscribeDepth", _contractId);
				any |= await TryInvoke("subscribe", "depth", _contractId);

				// Bars @ common timeframes (some hubs support these)
				foreach (var tf in new[] { "1m", "5m", "30m" })
				{
					any |= await TryInvoke("SubscribeBars", _contractId, tf);
					any |= await TryInvoke("subscribeBars", _contractId, tf);
				}

				if (any)
				{
					_subscribed = true;
					_log.LogInformation("[MarketHub] Subscribed to {ContractId}", _contractId);
				}
				else
				{
					_subscribed = false;
					_log.LogDebug("[MarketHub] No subscription methods were accepted by the hub (will retry on reconnect)");
				}
			}
			finally
			{
				_subLock.Release();
			}
		}


		private async Task<HubConnection> BuildMarketHubAsync()
		{
			// Prefer explicit RTC_MARKET_HUB, otherwise fall back to TOPSTEPX_RTC_BASE/hubs/market
			var explicitUrl = Environment.GetEnvironmentVariable("RTC_MARKET_HUB");
			string url;
			if (!string.IsNullOrWhiteSpace(explicitUrl))
			{
				url = explicitUrl!.TrimEnd('/');
			}
			else
			{
				var rtcBase = (Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE") ?? "https://rtc.topstepx.com").TrimEnd('/');
				url = $"{rtcBase}/hubs/market";
			}
			string? jwt;
			try { jwt = await _getJwtAsync(); } catch { jwt = null; }
			_log.LogInformation("[MarketHub] Using URL={Url} | JWT length={Len}", url, (jwt?.Length ?? 0));
			var hub = new HubConnectionBuilder()
				.WithUrl(url, o =>
				{
					o.AccessTokenProvider = _getJwtAsync;
					// Allow negotiation and all common transports for compatibility
					o.Transports = HttpTransportType.WebSockets | HttpTransportType.ServerSentEvents | HttpTransportType.LongPolling;
					// Do not skip negotiation; some servers require it
					// o.SkipNegotiation = true;
				})
				.WithAutomaticReconnect(new ExpoRetry())
				.Build();

			hub.ServerTimeout = TimeSpan.FromSeconds(35);
			hub.KeepAliveInterval = TimeSpan.FromSeconds(10);
			hub.HandshakeTimeout = TimeSpan.FromSeconds(12);

			// Helper to forward various payload shapes as JsonElement
			static JsonElement AsJsonElement(object payload)
			{
				if (payload is JsonElement je) return je;
				if (payload is string s)
				{
					try { using var doc = JsonDocument.Parse(s); return doc.RootElement.Clone(); } catch { }
				}
				try { var json = System.Text.Json.JsonSerializer.Serialize(payload); using var doc = JsonDocument.Parse(json); return doc.RootElement.Clone(); } catch { }
				using var empty = JsonDocument.Parse("{}");
				return empty.RootElement.Clone();
			}

			// Known TopstepX gateway-style methods
			hub.On<string, JsonElement>("GatewayQuote", (cid, json) =>
			{
				if (cid == _contractId)
				{
					_lastQuoteSeen[cid] = DateTime.UtcNow;
					if (!_firstQuoteLogged)
					{
						_firstQuoteLogged = true;
						if (!Concise()) _log.LogInformation("[MD] First quote for {Cid}: {Payload}", cid, json);
					}
					OnQuote?.Invoke(cid, json);
				}
			});
			hub.On<string, JsonElement>("GatewayTrade", (cid, json) =>
			{
				if (cid == _contractId)
				{
					if (!_firstTradeLogged)
					{
						_firstTradeLogged = true;
						if (!Concise()) _log.LogInformation("[MD] First trade for {Cid}: {Payload}", cid, json);
					}
					OnTrade?.Invoke(cid, json);
				}
			});
			hub.On<string, JsonElement>("GatewayDepth", (cid, json) => { if (cid == _contractId) OnDepth?.Invoke(cid, json); });

			// Common alt method names seen on some hubs: lowercase simple names and bars
			hub.On<string, object>("quote", (cid, payload) =>
			{
				if (cid == _contractId)
				{
					_lastQuoteSeen[cid] = DateTime.UtcNow;
					var json = AsJsonElement(payload);
					if (!_firstQuoteLogged)
					{
						_firstQuoteLogged = true;
						if (!Concise()) _log.LogInformation("[MD] First quote for {Cid}: {Payload}", cid, json);
					}
					OnQuote?.Invoke(cid, json);
				}
			});
			hub.On<string, object>("trade", (cid, payload) => { if (cid == _contractId) { var json = AsJsonElement(payload); if (!_firstTradeLogged) { _firstTradeLogged = true; if (!Concise()) _log.LogInformation("[MD] First trade for {Cid}: {Payload}", cid, json); } OnTrade?.Invoke(cid, json); } });
			hub.On<string, object>("depth", (cid, payload) => { if (cid == _contractId) { var json = AsJsonElement(payload); OnDepth?.Invoke(cid, json); } });
			hub.On<string, object>("bar", (cid, payload) => { if (cid == _contractId) { _lastBarSeen[cid] = DateTime.UtcNow; } });

			return hub;
		}

		private void AttachLifecycleHandlers(CancellationToken appCt)
		{
			if (_conn is null) return;

				_conn.Reconnecting += ex =>
				{
					_subscribed = false;
					if (!Concise()) _log.LogWarning(ex, "[MarketHub] Reconnecting…");
					return Task.CompletedTask;
				};

			_conn.Reconnected += async _ =>
			{
				if (!Concise()) _log.LogInformation("[MarketHub] Reconnected. Re-subscribing…");
				await SubscribeIfConnectedAsync(CancellationToken.None);
			};

			_conn.Closed += async ex =>
			{
				_subscribed = false;
				var now = DateTime.UtcNow;
				if (!Concise())
				{
					if (now - _lastClosedWarnUtc >= _closedWarnInterval)
					{
						_lastClosedWarnUtc = now;
						_log.LogWarning(ex, "[MarketHub] Closed. Rebuilding with fresh token…");
					}
					else
					{
						_log.LogDebug(ex, "[MarketHub] Closed. Rebuilding with fresh token… (suppressed)");
					}
				}
				var delay = TimeSpan.FromSeconds(1);
				for (int attempt = 1; attempt <= 5 && !appCt.IsCancellationRequested; attempt++)
				{
					try
					{
						await Task.Delay(delay, appCt);
						try { if (_conn is not null) await _conn.DisposeAsync(); } catch { }
						_conn = await BuildMarketHubAsync();
						AttachLifecycleHandlers(appCt);
						using var connectCts = CancellationTokenSource.CreateLinkedTokenSource(appCt);
						connectCts.CancelAfter(TimeSpan.FromSeconds(15));
						await _conn.StartAsync(connectCts.Token);
						await Task.Delay(200, appCt);
						await SubscribeIfConnectedAsync(CancellationToken.None);
 					if (!Concise())
 					{
 						var now2 = DateTime.UtcNow;
 						if (now2 - _lastRebuiltInfoUtc >= _rebuiltInfoInterval)
 						{
 							_lastRebuiltInfoUtc = now2;
 							_log.LogInformation("[MarketHub] Rebuilt and reconnected (attempt {Attempt}).", attempt);
 						}
 						else
 						{
 							_log.LogDebug("[MarketHub] Rebuilt and reconnected (attempt {Attempt}). (suppressed)", attempt);
  					}
  				}
  				return; // success
  				}
  				catch (Exception rex)
  				{
  					if (!Concise()) _log.LogWarning(rex, "[MarketHub] Rebuild attempt {Attempt} failed.", attempt);
  					delay = TimeSpan.FromMilliseconds(Math.Min(delay.TotalMilliseconds * 2, 5000));
  				}
  			}
  		};
		}

		public Microsoft.AspNetCore.SignalR.Client.HubConnection Connection => _conn!;

		public async ValueTask DisposeAsync()
		{
			if (_conn is not null)
			{
				try { await _conn.DisposeAsync(); }
				catch (Exception ex)
				{
					_log.LogDebug(ex, "[MarketHub] DisposeAsync swallowed exception.");
				}
			}
			_subLock.Dispose();
		}
	}
}
// If we go >10s without any tick while we believe we're subscribed, nudge a restart.
