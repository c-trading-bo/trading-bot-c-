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
		private readonly Func<Task<string?>> _getJwtAsync;

		private HubConnection? _conn;
		private string _contractId = string.Empty;

		private readonly SemaphoreSlim _subLock = new(1, 1);
		private volatile bool _disposed;
		private volatile bool _subscribed;

		public event Action<string, JsonElement>? OnQuote;
		public event Action<string, JsonElement>? OnTrade;
		public event Action<string, JsonElement>? OnDepth;

		public MarketHubClient(ILogger<MarketHubClient> log, Func<Task<string?>> getJwtAsync)
		{
			_log = log;
			_getJwtAsync = getJwtAsync;
		}

		public async Task StartAsync(string contractId, CancellationToken ct = default)
		{
			_contractId = contractId ?? throw new ArgumentNullException(nameof(contractId));
			if (_conn is not null) throw new InvalidOperationException("MarketHubClient already started.");

			string Url() => "https://rtc.topstepx.com/hubs/market";

			_conn = new HubConnectionBuilder()
				.WithUrl(Url(), o =>
				{
					o.AccessTokenProvider = async () =>
					{
						var t = await _getJwtAsync();
						var ok = !string.IsNullOrWhiteSpace(t);
						_log.LogInformation("[MarketHub] AccessTokenProvider token present? {ok}", ok);
						return t;
					};
					o.Transports = HttpTransportType.WebSockets;
				})
				.WithAutomaticReconnect(new ExpoRetry())
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

			_conn.Closed += err =>
			{
				_subscribed = false;
				_log.LogWarning(err, "[MarketHub] Closed.");
				// Set marketHub status as disconnected; AutomaticReconnect will handle reconnects.
				return Task.CompletedTask;
			};

			await _conn.StartAsync(ct);
			// Set marketHub status as connected
			// You may need to inject StatusService here if not already
			await Task.Delay(200);
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
