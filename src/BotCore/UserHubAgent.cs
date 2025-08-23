// Agent: UserHubAgent
// Role: Manages SignalR user hub connection, event wiring, and logging for TopstepX.
// Integration: Used by orchestrator and other agents for user event streaming and order/trade logging.
using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.Http.Connections;

namespace BotCore
{
	/// <summary>
	/// Shared user hub logic for TopstepX events and logging.
	/// </summary>
	public sealed class UserHubAgent : IAsyncDisposable
	{
		private readonly ILogger<UserHubAgent> _log;
		private HubConnection _hub = default!;
		private readonly SemaphoreSlim _gate = new(1,1);
		private bool _wired;
		private Func<Exception, Task>? _onClosed;
		private Func<Exception, Task>? _onReconnecting;
		private Func<string, Task>? _onReconnected;

		public HubConnection? GetConnection() => _hub;
		public HubConnection? Connection => _hub;

		public UserHubAgent(ILogger<UserHubAgent> log) => _log = log;

		public async Task ConnectAsync(string jwtToken, long accountId, CancellationToken ct)
		{
			await _gate.WaitAsync(ct);
			try
			{
				if (_hub is { State: HubConnectionState.Connected or HubConnectionState.Connecting or HubConnectionState.Reconnecting })
				{
					_log.LogInformation("UserHub already connecting/connected. State={State}", _hub.State);
					return;
				}
				_hub = new HubConnectionBuilder()
					.WithUrl("https://rtc.topstepx.com/hubs/user", o =>
					{
						o.AccessTokenProvider = () => Task.FromResult<string?>(jwtToken);
						o.Transports = HttpTransportType.WebSockets;
					})
					.WithAutomaticReconnect()
					.Build();

				WireHandlersOnce();

				try
				{
					await _hub.StartAsync(ct);
					_log.LogInformation("UserHub connected. State={State}", _hub.State);
				}
				catch (Exception startEx)
				{
					_log.LogError(startEx, "UserHub StartAsync FAILED. State={State}", _hub.State);
					throw;
				}

				await HubSafe.InvokeAsync(_hub, () => _hub.InvokeAsync("JoinAccountRoom", accountId, ct), ct);
				await HubSafe.InvokeAsync(_hub, () => _hub.InvokeAsync("SubscribeUserEvents", accountId, ct), ct);
			}
			finally
			{
				_gate.Release();
			}
		}

		private void WireHandlersOnce()
		{
			if (_wired) return;
			_onClosed       = ex => { _log.LogWarning(ex, "UserHub CLOSED (reason above)"); return Task.CompletedTask; };
			_onReconnecting = ex => { _log.LogWarning(ex, "UserHub RECONNECTING"); return Task.CompletedTask; };
			_onReconnected  = id => { _log.LogInformation("UserHub RECONNECTED {Id}", id); return Task.CompletedTask; };

			_hub.Closed       += _onClosed;
			_hub.Reconnecting += _onReconnecting;
			_hub.Reconnected  += _onReconnected;

			_wired = true;
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
