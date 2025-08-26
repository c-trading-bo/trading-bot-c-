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
			private DateTime _lastClosedWarnUtc = DateTime.MinValue;
			private DateTime _lastRebuiltInfoUtc = DateTime.MinValue;
			private readonly TimeSpan _closedWarnInterval = TimeSpan.FromSeconds(GetEnvInt("MARKET_HUB_CLOSED_WARN_INTERVAL_SECONDS", 10));
			private readonly TimeSpan _rebuiltInfoInterval = TimeSpan.FromSeconds(GetEnvInt("MARKET_HUB_REBUILT_INFO_INTERVAL_SECONDS", 10));

			private static int GetEnvInt(string name, int def)
			{
				var s = Environment.GetEnvironmentVariable(name);
				return int.TryParse(s, out var v) && v > 0 ? v : def;
			}

			public event Action<string, JsonElement>? OnQuote;
			public event Action<string, JsonElement>? OnTrade;
			public event Action<string, JsonElement>? OnDepth;

			public TimeSpan LastQuoteSeenAge(string contractId)
				=> DateTime.UtcNow - (_lastQuoteSeen.TryGetValue(contractId, out var t) ? t : DateTime.MinValue);
			public TimeSpan LastBarSeenAge(string contractId)
				=> DateTime.UtcNow - (_lastBarSeen.TryGetValue(contractId, out var t) ? t : DateTime.MinValue);
			public void RecordBarSeen(string contractId)
				=> _lastBarSeen[contractId] = DateTime.UtcNow;

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

				try
				{
					await _conn.InvokeAsync("SubscribeContractQuotes", _contractId, ct);
					await _conn.InvokeAsync("SubscribeContractTrades", _contractId, ct);
					await _conn.InvokeAsync("SubscribeContractMarketDepth", _contractId, ct);

					_subscribed = true;
					_log.LogInformation("[MarketHub] Subscribed to {ContractId}", _contractId);
				}
				catch (InvalidOperationException ioe)
				{
					_log.LogDebug(ioe, "Subscribe raced with disconnect; will re-subscribe on Reconnected.");
					_subscribed = false;
				}
				catch (TaskCanceledException tce)
				{
					_log.LogWarning(tce, "Subscribe canceled (likely connection transition). Will retry on Reconnected.");
					_subscribed = false;
				}
				catch (Exception ex)
				{
					_log.LogWarning(ex, "Subscribe failed; will re-subscribe on Reconnected.");
					_subscribed = false;
				}
			}
			finally
			{
				_subLock.Release();
			}
		}


		private Task<HubConnection> BuildMarketHubAsync()
		{
			var rtcBase = (Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE") ?? "https://rtc.topstepx.com").TrimEnd('/');
			var url = $"{rtcBase}/hubs/market";
			var hub = new HubConnectionBuilder()
				.WithUrl(url, o =>
				{
					o.AccessTokenProvider = _getJwtAsync;
					o.Transports = HttpTransportType.WebSockets;
					o.SkipNegotiation = true;
				})
				.WithAutomaticReconnect(new ExpoRetry())
				.Build();

			hub.ServerTimeout = TimeSpan.FromSeconds(35);
			hub.KeepAliveInterval = TimeSpan.FromSeconds(10);
			hub.HandshakeTimeout = TimeSpan.FromSeconds(12);

			hub.On<string, JsonElement>("GatewayQuote", (cid, json) =>
			{
				if (cid == _contractId)
				{
					_lastQuoteSeen[cid] = DateTime.UtcNow;
					if (!_firstQuoteLogged)
					{
						_firstQuoteLogged = true;
						_log.LogInformation("[MD] First quote for {Cid}: {Payload}", cid, json);
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
						_log.LogInformation("[MD] First trade for {Cid}: {Payload}", cid, json);
					}
					OnTrade?.Invoke(cid, json);
				}
			});
			hub.On<string, JsonElement>("GatewayDepth", (cid, json) => { if (cid == _contractId) OnDepth?.Invoke(cid, json); });

			return Task.FromResult(hub);
		}

		private void AttachLifecycleHandlers(CancellationToken appCt)
		{
			if (_conn is null) return;

			_conn.Reconnecting += ex =>
			{
				_subscribed = false;
				_log.LogWarning(ex, "[MarketHub] Reconnecting…");
				return Task.CompletedTask;
			};

			_conn.Reconnected += async _ =>
			{
				_log.LogInformation("[MarketHub] Reconnected. Re-subscribing…");
				await SubscribeIfConnectedAsync(CancellationToken.None);
			};

			_conn.Closed += async ex =>
			{
				_subscribed = false;
				var now = DateTime.UtcNow;
				if (now - _lastClosedWarnUtc >= _closedWarnInterval)
				{
					_lastClosedWarnUtc = now;
					_log.LogWarning(ex, "[MarketHub] Closed. Rebuilding with fresh token…");
				}
				else
				{
					_log.LogDebug(ex, "[MarketHub] Closed. Rebuilding with fresh token… (suppressed)");
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
						return; // success
					}
					catch (Exception rex)
					{
						_log.LogWarning(rex, "[MarketHub] Rebuild attempt {Attempt} failed.", attempt);
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
