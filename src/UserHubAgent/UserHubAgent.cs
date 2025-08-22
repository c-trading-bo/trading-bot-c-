using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;

namespace UserHubAgent
{
	/// <summary>
	/// Connects to TopstepX User Hub via SignalR, subscribes to order/trade events, and logs structured output.
	/// </summary>
	public sealed class UserHubAgent : IAsyncDisposable
	{
		private readonly ILogger<UserHubAgent> _log;
		private readonly PnLTracker _pnl;
		private HubConnection? _hub;
		private bool _wired;

		public UserHubAgent(ILogger<UserHubAgent> log, PnLTracker pnl)
		{
			_log = log;
			_pnl = pnl;
		}

		public async Task ConnectAsync(string jwt, long accountId, CancellationToken ct)
		{
			_hub = new HubConnectionBuilder()
				.WithUrl("https://rtc.topstepx.com/hubs/user", o =>
				{
					o.AccessTokenProvider = () => Task.FromResult(jwt);
				})
				.WithAutomaticReconnect()
				.Build();

			await _hub.StartAsync(ct);
			if (_wired) return; _wired = true;

			_hub.On<object>("GatewayUserOrder", data =>
			{
				try
				{
					var json = JsonSerializer.Serialize(data);
					_log.LogInformation($"ORDER => {json}");
				}
				catch (Exception ex)
				{
					_log.LogWarning(ex, "Failed to log GatewayUserOrder event");
				}
			});

			_hub.On<object>("GatewayUserTrade", data =>
			{
				try
				{
					var json = JsonSerializer.Serialize(data);
					_log.LogInformation($"TRADE => {json}");
				}
				catch (Exception ex)
				{
					_log.LogWarning(ex, "Failed to log GatewayUserTrade event");
				}
			});

			await _hub.InvokeAsync("SubscribeOrders", accountId, ct);
			await _hub.InvokeAsync("SubscribeTrades", accountId, ct);
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
