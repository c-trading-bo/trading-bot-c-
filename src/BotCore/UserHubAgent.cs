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

		public async Task ConnectAsync(string jwtToken, long accountId, CancellationToken appCt)
		{
			if (_hub is { State: HubConnectionState.Connected or HubConnectionState.Connecting })
			{
				_log.LogInformation("UserHub already connected or connecting.");
				return;
			}

			_accountId = accountId;
			var baseUrl = Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE") ?? "https://rtc.topstepx.com";
			var url = $"{baseUrl.TrimEnd('/')}/hubs/user";
			_hub = new HubConnectionBuilder()
				.WithUrl(url, opt =>
				{
					// Bearer token provider with refresh support: reads latest env or validates/renews when needed
					opt.AccessTokenProvider = async () =>
					{
						// Prefer latest value from env to allow external refresh
						var tok = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
						if (!string.IsNullOrWhiteSpace(tok)) return tok;

						// Fallback to initial token if provided
						if (!string.IsNullOrWhiteSpace(jwtToken)) return jwtToken;

						// Last resort: try to obtain via login key if present in env
						try
						{
							var user = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
							var key  = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
							if (!string.IsNullOrWhiteSpace(user) && !string.IsNullOrWhiteSpace(key))
							{
								using var http = new System.Net.Http.HttpClient { BaseAddress = new Uri(Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com") };
								var auth = new TopstepAuthAgent(http);
								var newTok = await auth.GetJwtAsync(user!, key!, CancellationToken.None);
								Environment.SetEnvironmentVariable("TOPSTEPX_JWT", newTok);
								_log.LogInformation("UserHub AccessTokenProvider: obtained fresh JWT via loginKey for {User}.", user);
								return newTok;
							}
						}
						catch (Exception ex)
						{
							_log.LogWarning(ex, "UserHub AccessTokenProvider: failed to obtain JWT via loginKey");
						}
						return null;
					};
					// Force WebSockets for production reliability
					opt.Transports = HttpTransportType.WebSockets;
				})
				.WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10) })
				.ConfigureLogging(lb =>
				{
					lb.AddConsole();
					lb.SetMinimumLevel(LogLevel.Debug);
				})
				.Build();

			WireEvents(_hub);

			_hub.ServerTimeout     = TimeSpan.FromSeconds(60);
			_hub.KeepAliveInterval = TimeSpan.FromSeconds(15);
			_hub.HandshakeTimeout  = TimeSpan.FromSeconds(15);

			// Use a short, connect-only CTS for StartAsync
			using var connectCts = CancellationTokenSource.CreateLinkedTokenSource(appCt);
			connectCts.CancelAfter(TimeSpan.FromSeconds(15));
			await _hub.StartAsync(connectCts.Token);

			_log.LogInformation("UserHub connected. State={State}", _hub.State);
			_statusService.Set("user.state", _hub.ConnectionId ?? string.Empty);

			// After connect, DO NOT reuse connectCts
			await Task.Delay(250, appCt); // ensure server is ready before subscribing
			await _hub.InvokeAsync("SubscribeAccounts");
			await _hub.InvokeAsync("SubscribeOrders",    accountId);
			await _hub.InvokeAsync("SubscribePositions", accountId);
			await _hub.InvokeAsync("SubscribeTrades",    accountId);
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
