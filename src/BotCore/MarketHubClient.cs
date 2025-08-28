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

			public record TradeTick(DateTime TimestampUtc, decimal Price, long Volume, string RawType);
			public event Action<string, TradeTick>? OnTrade;   // contractId, tick
			public event Action<string, decimal, decimal, decimal>? OnQuote; // contractId, last, bid, ask
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

				// Wait until active (fixes “connection is not active”)
				var t0 = DateTime.UtcNow;
				while (_conn!.State != HubConnectionState.Connected && DateTime.UtcNow - t0 < TimeSpan.FromSeconds(10))
				{
					await Task.Delay(100, ct);
				}
				if (_conn.State != HubConnectionState.Connected)
				{
					_log.LogWarning("[MarketHub] Subscribe skipped: connection not active (State={State})", _conn.State);
					return;
				}

				// Invoke only the documented ProjectX methods with a single string contractId
				bool anySucceeded = false;

				async Task Call(string method, string cid)
				{
					try
					{
						await _conn!.InvokeAsync(method, cid, ct);
						_log.LogInformation("[MarketHub] {Method}({Cid}) OK", method, cid);
						anySucceeded = true;
					}
					catch (Exception ex)
					{
						_log.LogWarning("[MarketHub] {Method}({Cid}) FAILED: {Error}", method, cid, ex.Message);
					}
				}

				await Call("SubscribeContractQuotes", _contractId);
				await Call("SubscribeContractTrades", _contractId);
				await Call("SubscribeContractMarketDepth", _contractId);

				_subscribed = anySucceeded;
				if (!anySucceeded)
				{
					_log.LogError("[MarketHub] Subscribe failed for {cid} using documented methods.", _contractId);
				}
				else
				{
					_log.LogInformation("[MarketHub] Subscribed (quotes/trades/depth) {ContractId}", _contractId);
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
			// Suppress noisy repeats of the same URL/JWT log
			var nowLog = DateTime.UtcNow;
			if (nowLog - _lastRebuiltInfoUtc >= _rebuiltInfoInterval)
			{
				_lastRebuiltInfoUtc = nowLog;
				_log.LogInformation("[MarketHub] Using URL={Url}", url);
			}
			else
			{
				_log.LogDebug("[MarketHub] Using URL={Url}", url);
			}
			// Do not include access_token in the URL; rely on AccessTokenProvider only
			var hub = new HubConnectionBuilder()
				.WithUrl(url, o =>
				{
					o.AccessTokenProvider = _getJwtAsync; // also provide via header for safety
					o.SkipNegotiation = true;
					o.Transports = HttpTransportType.WebSockets;
				})
				.ConfigureLogging(lb =>
				{
					lb.ClearProviders();
					lb.AddConsole();
					lb.SetMinimumLevel(LogLevel.Information);
					var concise = (Environment.GetEnvironmentVariable("APP_CONCISE_CONSOLE") ?? "true").Trim().ToLowerInvariant() is "1" or "true" or "yes";
					if (concise)
					{
						lb.AddFilter("Microsoft", LogLevel.Warning);
						lb.AddFilter("System", LogLevel.Warning);
						lb.AddFilter("Microsoft.AspNetCore.SignalR", LogLevel.Warning);
						lb.AddFilter("Microsoft.AspNetCore.Http.Connections", LogLevel.Warning);
					}
					// Suppress transport logs that can echo access_token
					lb.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Error);
					lb.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Error);
					lb.AddFilter("Microsoft.AspNetCore.Http.Connections.Client.Internal.WebSocketsTransport", LogLevel.Error);
				})
				.WithAutomaticReconnect(new ExpoRetry())
				.Build();

			hub.ServerTimeout = TimeSpan.FromSeconds(30);
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

			void EmitTrade(string cid, JsonElement t)
			{
				try
				{
					var ts  = t.GetProperty("timestamp").GetDateTime().ToUniversalTime();
					var px  = t.GetProperty("price").GetDecimal();
					var vol = t.TryGetProperty("volume", out var vv) ? (vv.TryGetInt64(out var v) ? v : 0L) : 0L;
					var typ = t.TryGetProperty("type", out var ty) ? ty.ToString() : string.Empty;
					OnTrade?.Invoke(cid, new TradeTick(ts, px, vol, typ));
				}
				catch { }
			}

			// Known TopstepX gateway-style methods
   hub.On<string, JsonElement>("GatewayQuote", (cid, q) =>
			{
				if (cid == _contractId)
				{
					_lastQuoteSeen[cid] = DateTime.UtcNow;
					if (!_firstQuoteLogged)
					{
						_firstQuoteLogged = true;
						if (!Concise()) _log.LogInformation("[MD] First quote for {Cid}: {Payload}", cid, q);
					}
					var last = q.TryGetProperty("lastPrice", out var lp) ? (lp.TryGetDecimal(out var d) ? d : 0m) : 0m;
					var bid  = q.TryGetProperty("bestBid", out var bb) ? (bb.TryGetDecimal(out var d2) ? d2 : 0m) : 0m;
					var ask  = q.TryGetProperty("bestAsk", out var ba) ? (ba.TryGetDecimal(out var d3) ? d3 : 0m) : 0m;
					OnQuote?.Invoke(cid, last, bid, ask);
				}
			});
   hub.On<string, JsonElement>("GatewayTrade", (cid, payload) =>
			{
				if (cid == _contractId)
				{
					if (!_firstTradeLogged)
					{
						_firstTradeLogged = true;
						if (!Concise()) _log.LogInformation("[MD] First trade for {Cid}: {Payload}", cid, payload);
					}
					if (payload.ValueKind == JsonValueKind.Array)
					{
						foreach (var t in payload.EnumerateArray()) EmitTrade(cid, t);
					}
					else
					{
						EmitTrade(cid, payload);
					}
				}
			});
  	hub.On<string, JsonElement>("GatewayDepth", (cid, json) => { if (cid == _contractId) { _lastBarSeen[cid] = DateTime.UtcNow; OnDepth?.Invoke(cid, json); } });

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
					var last = json.TryGetProperty("lastPrice", out var lp) ? (lp.TryGetDecimal(out var d) ? d : 0m) : 0m;
					var bid  = json.TryGetProperty("bestBid", out var bb) ? (bb.TryGetDecimal(out var d2) ? d2 : 0m) : 0m;
					var ask  = json.TryGetProperty("bestAsk", out var ba) ? (ba.TryGetDecimal(out var d3) ? d3 : 0m) : 0m;
					OnQuote?.Invoke(cid, last, bid, ask);
				}
			});
   hub.On<string, object>("trade", (cid, payload) => { if (cid == _contractId) { var json = AsJsonElement(payload); if (!_firstTradeLogged) { _firstTradeLogged = true; if (!Concise()) _log.LogInformation("[MD] First trade for {Cid}: {Payload}", cid, json); } EmitTrade(cid, json); } });
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
