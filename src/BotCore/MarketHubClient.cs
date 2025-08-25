#nullable enable
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace BotCore
{
	public sealed class MarketHubClient : IAsyncDisposable
	{
		private readonly ILogger<MarketHubClient> _log;
		private readonly Func<string> _getJwt;

		private HubConnection? _conn;
		private string _contractId = string.Empty;

		private readonly SemaphoreSlim _subLock = new(1, 1);
		private volatile bool _disposed;
		private volatile bool _subscribed;

		public event Action<string, JsonElement>? OnQuote;
		public event Action<string, JsonElement>? OnTrade;
		public event Action<string, JsonElement>? OnDepth;

		public MarketHubClient(ILogger<MarketHubClient> log, Func<string> getJwt)
		{
			_log = log;
			_getJwt = getJwt;
		}

		public async Task StartAsync(string contractId, CancellationToken ct = default)
		{
			_contractId = contractId ?? throw new ArgumentNullException(nameof(contractId));
			if (_conn is not null) throw new InvalidOperationException("MarketHubClient already started.");

			string Jwt() => _getJwt();
			string UrlWithToken() => $"https://rtc.topstepx.com/hubs/market?access_token={Uri.EscapeDataString(Jwt())}";

			_conn = new HubConnectionBuilder()
				.WithUrl(UrlWithToken(), o =>
				{
					o.AccessTokenProvider = () => Task.FromResult<string?>(Jwt());
					o.Transports = HttpTransportType.WebSockets;
				})
				.WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10) })
				.Build();

			_conn.ServerTimeout = TimeSpan.FromSeconds(30);
			_conn.KeepAliveInterval = TimeSpan.FromSeconds(15);
			_conn.HandshakeTimeout = TimeSpan.FromSeconds(15);

			// Wire market event handlers once
			bool marketHandlersWired = false;
			void WireMarketHandlers(HubConnection hub)
			{
				if (marketHandlersWired) return;
				hub.On<string, JsonElement>("GatewayQuote", (cid, json) => { if (cid == _contractId) OnQuote?.Invoke(cid, json); });
				hub.On<string, JsonElement>("GatewayTrade", (cid, json) => { if (cid == _contractId) OnTrade?.Invoke(cid, json); });
				hub.On<string, JsonElement>("GatewayDepth", (cid, json) => { if (cid == _contractId) OnDepth?.Invoke(cid, json); });
				marketHandlersWired = true;
			}
			WireMarketHandlers(_conn);

			_conn.Reconnecting += err =>
			{
				_subscribed = false;
				_log.LogWarning(err, "[MarketHub] Reconnecting…");
				return Task.CompletedTask;
			};

			_conn.Reconnected += async _ =>
			{
				_log.LogInformation("[MarketHub] Reconnected. Re-subscribing…");
				// Set marketHub status as connected
				// You may need to inject StatusService here if not already
				await SubscribeIfConnectedAsync(CancellationToken.None);
			};

			_conn.Closed += async err =>
			{
				_subscribed = false;
				_log.LogWarning(err, "[MarketHub] Closed. Restarting with backoff…");
				// Set marketHub status as disconnected
				await RestartLoopAsync(UrlWithToken);
			};

			await _conn.StartAsync(ct);
			// Set marketHub status as connected
			// You may need to inject StatusService here if not already
			await Task.Delay(200, ct);
			await SubscribeIfConnectedAsync(ct);
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
					_log.LogDebug("Skip subscribe: state = {state}", _conn.State);
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

		private async Task RestartLoopAsync(Func<string> urlFactory)
		{
			if (_conn is null) return;

			var delay = TimeSpan.FromMilliseconds(500);
			for (int attempt = 1; attempt <= 8 && !_disposed; attempt++)
			{
				try
				{
					var newUrl = urlFactory();
					// Dispose old connection
					if (_conn != null)
					{
						try { await _conn.DisposeAsync(); } catch { }
					}
					// Rebuild connection with fresh token in query
					_conn = new HubConnectionBuilder()
						.WithUrl(newUrl, o =>
						{
							o.AccessTokenProvider = () => Task.FromResult<string?>(_getJwt());
							o.Transports = HttpTransportType.WebSockets;
							o.SkipNegotiation = true;
						})
						.WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10) })
						.Build();

					_conn.ServerTimeout = TimeSpan.FromSeconds(30);
					_conn.KeepAliveInterval = TimeSpan.FromSeconds(15);
					_conn.HandshakeTimeout = TimeSpan.FromSeconds(15);

					_conn.On<string, JsonElement>("GatewayQuote", (cid, json) => { if (cid == _contractId) OnQuote?.Invoke(cid, json); });
					_conn.On<string, JsonElement>("GatewayTrade", (cid, json) => { if (cid == _contractId) OnTrade?.Invoke(cid, json); });
					_conn.On<string, JsonElement>("GatewayDepth", (cid, json) => { if (cid == _contractId) OnDepth?.Invoke(cid, json); });

					_conn.Reconnecting += err =>
					{
						_subscribed = false;
						_log.LogWarning(err, "[MarketHub] Reconnecting…");
						return Task.CompletedTask;
					};

					_conn.Reconnected += async _ =>
					{
						_log.LogInformation("[MarketHub] Reconnected. Re-subscribing…");
						await SubscribeIfConnectedAsync(CancellationToken.None);
					};

					_conn.Closed += async err =>
					{
						_subscribed = false;
						_log.LogWarning(err, "[MarketHub] Closed. Restarting with backoff…");
						await RestartLoopAsync(urlFactory);
					};

					await _conn.StartAsync();
					if (_conn.State == HubConnectionState.Connected)
					{
						_log.LogInformation("[MarketHub] Restarted (attempt {Attempt}).", attempt);
						await Task.Delay(200);
						await SubscribeIfConnectedAsync(CancellationToken.None);
						return;
					}
				}
				catch (Exception ex)
				{
					_log.LogWarning(ex, "[MarketHub] Restart attempt {Attempt} failed; retry in {Delay}…", attempt, delay);
				}
				await Task.Delay(delay);
				delay = TimeSpan.FromMilliseconds(Math.Min(delay.TotalMilliseconds * 2, 5000));
			}
			_log.LogError("[MarketHub] Could not restart after repeated attempts.");
		}

		public Microsoft.AspNetCore.SignalR.Client.HubConnection Connection => _conn!;

		public async ValueTask DisposeAsync()
		{
			_disposed = true;
			if (_conn is not null)
			{
				try { await _conn.DisposeAsync(); } catch { }
			}
			_subLock.Dispose();
		}
	}
}
			// If we go >10s without any tick while we believe we're subscribed, nudge a restart.
