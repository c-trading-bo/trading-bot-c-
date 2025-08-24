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
	/// </summary>
	public sealed class UserHubAgent : IAsyncDisposable
	{
	private readonly ILogger<UserHubAgent> _log;
	private readonly SupervisorAgent.StatusService _statusService;
	private HubConnection _hub = default!;
	private bool _handlersWired;
	private long _accountId;

		public HubConnection? GetConnection() => _hub;
		public HubConnection? Connection => _hub;

		private const string CloseStatusKey = "CloseStatus";

		public UserHubAgent(ILogger<UserHubAgent> log, SupervisorAgent.StatusService statusService)
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

			_accountId = accountId;
			var url = $"https://rtc.topstepx.com/hubs/user?access_token={Uri.EscapeDataString(jwtToken)}";
			_hub = new HubConnectionBuilder()
				.WithUrl(url, opt =>
				{
					opt.AccessTokenProvider = () => Task.FromResult<string?>(jwtToken); // match delegate type
					opt.Transports = HttpTransportType.WebSockets;
					// opt.SkipNegotiation = true; // enable later only if confirmed working
				})
				.ConfigureLogging(lb => lb.AddConsole().SetMinimumLevel(LogLevel.Debug))
				.Build();

			WireEvents(_hub);

			_hub.ServerTimeout     = TimeSpan.FromSeconds(60);
			_hub.KeepAliveInterval = TimeSpan.FromSeconds(15);
			_hub.HandshakeTimeout  = TimeSpan.FromSeconds(15);

			await _hub.StartAsync(ct);
			_log.LogInformation("UserHub connected. State={State}", _hub.State);
			_statusService.Set("user.state", _hub.ConnectionId ?? string.Empty);
			await Task.Delay(250, ct); // ensure server is ready before subscribing
			await HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeAccounts"), _log, ct);
			await HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeOrders",    accountId), _log, ct);
			await HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribePositions", accountId), _log, ct);
			await HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeTrades",    accountId), _log, ct);
		}

		private void WireEvents(HubConnection hub)
		{
			if (_handlersWired) return;
			hub.On<object>("GatewayUserAccount", data => _log.LogInformation("Account evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)));
			hub.On<object>("GatewayUserOrder",   data => _log.LogInformation("Order evt: {Json}",   System.Text.Json.JsonSerializer.Serialize(data)));
			hub.On<object>("GatewayUserPosition",data => _log.LogInformation("Position evt: {Json}",System.Text.Json.JsonSerializer.Serialize(data)));
			hub.On<object>("GatewayUserTrade",   data => _log.LogInformation("Trade evt: {Json}",   System.Text.Json.JsonSerializer.Serialize(data)));
			_handlersWired = true;

			hub.Closed += ex =>
			{
				_statusService.Set("user.state", string.Empty);
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
				_statusService.Set("user.state", string.Empty);
				_log.LogWarning(ex, "UserHub RECONNECTING: {Message}", ex?.Message);
				return Task.CompletedTask;
			};
			hub.Reconnected += id =>
			{
				_statusService.Set("user.state", id ?? string.Empty);
				_log.LogInformation("UserHub RECONNECTED: ConnectionId={Id}", id);
				_ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeAccounts"), _log, CancellationToken.None);
				_ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeOrders",    _accountId), _log, CancellationToken.None);
				_ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribePositions", _accountId), _log, CancellationToken.None);
				_ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeTrades",    _accountId), _log, CancellationToken.None);
				return Task.CompletedTask;
			};
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
