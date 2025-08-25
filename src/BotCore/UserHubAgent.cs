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
			// Backwards-compatible overload: wrap into async provider that uses env or fallback jwtToken
			Func<Task<string?>> provider = async () =>
			{
				var tok = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
				if (!string.IsNullOrWhiteSpace(tok)) return tok;
				if (!string.IsNullOrWhiteSpace(jwtToken)) return jwtToken;
				// As a last resort, try loginKey if present
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
						return newTok;
					}
				}
				catch { }
				return tok; // may be null here; the async overload ensures logging and reconnect policy
			};
			await ConnectAsync(provider, accountId, appCt);
		}

		public async Task ConnectAsync(Func<Task<string?>> tokenProvider, long accountId, CancellationToken appCt)
		{
			if (_hub is { State: HubConnectionState.Connected or HubConnectionState.Connecting })
			{
				_log.LogInformation("UserHub already connected or connecting.");
				return;
			}

			_accountId = accountId;
			_hub = await BuildUserHubAsync(tokenProvider);

			using var connectCts = CancellationTokenSource.CreateLinkedTokenSource(appCt);
			connectCts.CancelAfter(TimeSpan.FromSeconds(15));
			await _hub.StartAsync(connectCts.Token);

			_log.LogInformation("UserHub connected. State={State}", _hub.State);
			_statusService.Set("user.state", _hub.ConnectionId ?? string.Empty);

			await Task.Delay(250, appCt);
			await HubSafe.InvokeIfConnected(_hub, () => _hub!.InvokeAsync("SubscribeAccounts"), _log, appCt);
			if (accountId > 0)
			{
				await HubSafe.InvokeIfConnected(_hub, () => _hub!.InvokeAsync("SubscribeOrders",    accountId), _log, appCt);
				await HubSafe.InvokeIfConnected(_hub, () => _hub!.InvokeAsync("SubscribePositions", accountId), _log, appCt);
				await HubSafe.InvokeIfConnected(_hub, () => _hub!.InvokeAsync("SubscribeTrades",    accountId), _log, appCt);
			}

			// Attach Closed rebuilder to ensure fresh token and resubscribe on server-initiated close
			AttachClosedRebuilder(tokenProvider, accountId, appCt);
		}

		private static string RtcBase() => (Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE") ?? "https://rtc.topstepx.com").TrimEnd('/');

		private async Task<HubConnection> BuildUserHubAsync(Func<Task<string?>> tokenProvider)
		{
			var tok = await tokenProvider() ?? string.Empty;
			var url = $"{RtcBase()}/hubs/user?access_token={Uri.EscapeDataString(tok)}";

			var hub = new HubConnectionBuilder()
				.WithUrl(url, o =>
				{
					o.Transports = HttpTransportType.WebSockets;
					o.SkipNegotiation = true;
				})
				.WithAutomaticReconnect(new ExpoRetry())
				.ConfigureLogging(lb =>
				{
					lb.AddConsole();
					lb.SetMinimumLevel(LogLevel.Debug);
				})
				.Build();

			hub.ServerTimeout = TimeSpan.FromSeconds(60);
			hub.KeepAliveInterval = TimeSpan.FromSeconds(15);
			hub.HandshakeTimeout = TimeSpan.FromSeconds(15);

			WireEvents(hub);
			return hub;
		}

		private void AttachClosedRebuilder(Func<Task<string?>> tokenProvider, long accountId, CancellationToken appCt)
		{
			_hub.Closed += async ex =>
			{
				_statusService.Set("user.state", string.Empty);
				_log.LogWarning(ex, "UserHub CLOSED: {Message}", ex?.Message);
				var delay = TimeSpan.FromSeconds(1);
				for (int attempt = 1; attempt <= 5 && !appCt.IsCancellationRequested; attempt++)
				{
					try
					{
						await Task.Delay(delay, appCt);
						try { await _hub.DisposeAsync(); } catch { }
						_hub = await BuildUserHubAsync(tokenProvider);
						using var connectCts = CancellationTokenSource.CreateLinkedTokenSource(appCt);
						connectCts.CancelAfter(TimeSpan.FromSeconds(15));
						await _hub.StartAsync(connectCts.Token);
						_statusService.Set("user.state", _hub.ConnectionId ?? string.Empty);
						await Task.Delay(250, appCt);
						await HubSafe.InvokeIfConnected(_hub, () => _hub!.InvokeAsync("SubscribeAccounts"), _log, appCt);
						if (accountId > 0)
						{
							await HubSafe.InvokeIfConnected(_hub, () => _hub!.InvokeAsync("SubscribeOrders",    accountId), _log, appCt);
							await HubSafe.InvokeIfConnected(_hub, () => _hub!.InvokeAsync("SubscribePositions", accountId), _log, appCt);
							await HubSafe.InvokeIfConnected(_hub, () => _hub!.InvokeAsync("SubscribeTrades",    accountId), _log, appCt);
						}
						// re-attach for future closes
						AttachClosedRebuilder(tokenProvider, accountId, appCt);
						return; // rebuilt successfully
					}
					catch (Exception retryEx)
					{
						_log.LogWarning(retryEx, "UserHub restart attempt {Attempt} failed.", attempt);
						delay = TimeSpan.FromMilliseconds(Math.Min(delay.TotalMilliseconds * 2, 5000));
					}
				}
			};
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
				if (_accountId > 0)
				{
					_ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeOrders",    _accountId), _log, CancellationToken.None);
					_ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribePositions", _accountId), _log, CancellationToken.None);
					_ = HubSafe.InvokeWhenConnected(_hub, () => _hub.InvokeAsync("SubscribeTrades",    _accountId), _log, CancellationToken.None);
				}
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
