// Agent: UserHubAgent
// Role: Manages SignalR user hub connection, event wiring, and logging for TopstepX.
// Integration: Used by orchestrator and other agents for user event streaming and order/trade logging.
using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using System.Net.WebSockets;
using Microsoft.AspNetCore.Http.Connections;

namespace BotCore
{
	/// <summary>
	/// Shared user hub logic for TopstepX events and logging.
	/// Also exposes typed JSON events for consumers (positions/trades/orders/accounts).
	/// </summary>
	public sealed class UserHubAgent : IAsyncDisposable
	{
		private readonly ILogger<UserHubAgent> _log;
		private readonly object? _statusService;
		private HubConnection _hub = default!;
		private bool _handlersWired;

		public event Action<JsonElement>? OnOrder;
		public event Action<JsonElement>? OnTrade;
		public event Action<JsonElement>? OnPosition;
		public event Action<JsonElement>? OnAccount;

		public HubConnection? GetConnection() => _hub;
		public HubConnection? Connection => _hub;

		private const string CloseStatusKey = "CloseStatus";

		public UserHubAgent(ILogger<UserHubAgent> log, object? statusService)
		{
			_log = log;
			_statusService = statusService;
		}

		public async Task ConnectAsync(string jwtToken, long accountId, CancellationToken ct)
		{
			if (_hub is not null && _hub.State == HubConnectionState.Connected)
			{
				_log.LogInformation("UserHub already connected.");
				return;
			}

			var url = "https://rtc.topstepx.com/hubs/user";
			_hub = new HubConnectionBuilder()
				.WithUrl(url, opt =>
				{
					opt.AccessTokenProvider = () => Task.FromResult<string?>(jwtToken); // token via AccessTokenProvider only
					opt.Transports = HttpTransportType.WebSockets;
					opt.SkipNegotiation = true;
				})
				.WithAutomaticReconnect(new ExpoRetry())
				.ConfigureLogging(lb => lb.AddConsole().SetMinimumLevel(LogLevel.Debug))
				.Build();

			WireEvents(_hub);

			_hub.ServerTimeout = TimeSpan.FromSeconds(60);
			_hub.KeepAliveInterval = TimeSpan.FromSeconds(15);
			_hub.HandshakeTimeout = TimeSpan.FromSeconds(15);

			// Wire up robust event logging
			_hub.Closed += ex =>
			{
				if (ex is HttpRequestException hre)
					_log.LogWarning(hre, "UserHub CLOSED. HTTP status: {Status}", hre.StatusCode);
				else if (ex is WebSocketException wse)
					_log.LogWarning(wse, "UserHub CLOSED. WebSocket error: {Err} / CloseStatus: {Close}", wse.WebSocketErrorCode, wse.Data != null && wse.Data.Contains(CloseStatusKey) ? wse.Data[CloseStatusKey] : null);
				else
					_log.LogWarning(ex, "UserHub CLOSED: {Message}", ex?.Message);
				// Let WithAutomaticReconnect handle recovery; avoid manual StartAsync noise
				return Task.CompletedTask;
			};
			// _hub.Reconnecting += ex =>
			// ...existing code...
			//     _log.LogWarning(ex, "UserHub RECONNECTING: {Message}", ex?.Message);
			//     return Task.CompletedTask;
			// };
			// _hub.Reconnected += id =>
			// {
			//     _log.LogInformation("UserHub RECONNECTED: ConnectionId={Id}", id);
			//     // Resubscribe after reconnect
			//     _ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeAccounts"), _log, CancellationToken.None);
			//     _ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeOrders",    accountId), _log, CancellationToken.None);
			//     _ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribePositions", accountId), _log, CancellationToken.None);
			//     _ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeTrades",    accountId), _log, CancellationToken.None);
			//     return Task.CompletedTask;
			// };

			await _hub.StartAsync(ct);
			_log.LogInformation("UserHub connected. State={State}", _hub.State);
			StatusSet("user.state", _hub.ConnectionId ?? string.Empty);
			await Task.Delay(250, ct); // ensure server is ready before subscribing
			await TrySubscribeAsync(() => _hub.InvokeAsync("SubscribeAccounts"), ct);
			await TrySubscribeAsync(() => _hub.InvokeAsync("SubscribeOrders", accountId), ct);
			await TrySubscribeAsync(() => _hub.InvokeAsync("SubscribePositions", accountId), ct);
			await TrySubscribeAsync(() => _hub.InvokeAsync("SubscribeTrades", accountId), ct);
		}

		private async Task TrySubscribeAsync(Func<Task> call, CancellationToken ct)
		{
			try
			{
				if (_hub is null) return;
				await HubSafe.InvokeWhenConnected(_hub, call, _log, ct);
			}
			catch (Exception ex)
			{
				_log.LogWarning(ex, "UserHub initial subscription invoke failed; will retry after reconnect.");
			}
		}

		private void WireEvents(HubConnection hub)
		{
			if (_handlersWired) return;
			hub.On<JsonElement>("GatewayUserAccount", data => { try { _log.LogInformation("Account evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnAccount?.Invoke(data); } catch { } });
			hub.On<JsonElement>("GatewayUserOrder", data => { try { _log.LogInformation("Order evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnOrder?.Invoke(data); } catch { } });
			hub.On<JsonElement>("GatewayUserPosition", data => { try { _log.LogInformation("Position evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnPosition?.Invoke(data); } catch { } });
			hub.On<JsonElement>("GatewayUserTrade", data => { try { _log.LogInformation("Trade evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnTrade?.Invoke(data); } catch { } });
			_handlersWired = true;

			hub.Closed += ex =>
			{
				StatusSet("user.state", string.Empty);
				if (ex is System.Net.Http.HttpRequestException hre)
					_log.LogWarning(hre, "UserHub CLOSED. HTTP status: {Status}", hre.StatusCode);
				else if (ex is System.Net.WebSockets.WebSocketException wse)
					_log.LogWarning(wse, "UserHub CLOSED. WebSocket error: {Err} / CloseStatus: {Close}", wse.WebSocketErrorCode, wse.Data != null && wse.Data.Contains(CloseStatusKey) ? wse.Data[CloseStatusKey] : null);
				else
					_log.LogWarning(ex, "UserHub CLOSED: {Message}", ex?.Message);
				return Task.CompletedTask;
			};
			hub.Reconnecting += ex =>
			{
				_log.LogWarning(ex, "UserHub RECONNECTING: {Message}", ex?.Message);
				return Task.CompletedTask;
			};
			hub.Reconnected += id =>
			{
				_log.LogInformation("UserHub RECONNECTED: ConnectionId={Id}", id);
				// re-subscribe here if needed
				return Task.CompletedTask;
			};
		}

		private void StatusSet(string key, object value)
		{
			try
			{
				var t = _statusService?.GetType();
				var m = t?.GetMethod("Set", new[] { typeof(string), typeof(object) });
				if (m != null)
				{
					m.Invoke(_statusService, new object?[] { key, value });
				}
			}
			catch { }
		}

		public async ValueTask DisposeAsync()
		{
			if (_hub != null)
			{
				await _hub.DisposeAsync();
			}
		}
	}
}
