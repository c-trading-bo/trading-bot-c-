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
		private long _accountId;
		private string? _jwt;

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

			_accountId = accountId;

			// Guard: require a non-empty JWT; otherwise the hub will handshake then immediately close.
			if (string.IsNullOrWhiteSpace(jwtToken))
			{
				_log.LogError("UserHub ConnectAsync called with empty JWT token. Aborting connect.");
				throw new ArgumentException("jwtToken must be non-empty", nameof(jwtToken));
			}

			// Persist token for SignalR AccessTokenProvider and reconnects
			_jwt = jwtToken;

			// Allow overriding RTC base from env
			var rtcBase = (Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE")
				?? Environment.GetEnvironmentVariable("RTC_BASE")
				?? "https://rtc.topstepx.com").TrimEnd('/');
			var baseUrl = $"{rtcBase}/hubs/user";
			// Build URL without embedding token to avoid leaking credentials in logs
			var url = baseUrl;
			_log.LogInformation("[UserHub] Using URL={Url}", baseUrl);

			_hub = new HubConnectionBuilder()
				.WithUrl(url, opt =>
				{
     opt.AccessTokenProvider = () => Task.FromResult(_jwt ?? string.Empty);
					opt.Transports = HttpTransportType.WebSockets;
					// DO NOT force SkipNegotiation unless required; default is fine
				})
				.WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromMilliseconds(500), TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5) })
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
    				})
				.Build();

			WireEvents(_hub);

			_hub.ServerTimeout = TimeSpan.FromSeconds(60);
			_hub.KeepAliveInterval = TimeSpan.FromSeconds(15);
			_hub.HandshakeTimeout = TimeSpan.FromSeconds(15);

			// Re-subscribe automatically after reconnects (use lax token)
			_hub.Reconnected += async id =>
			{
				_log.LogInformation("UserHub RE reconnected. State={State} | ConnectionId={Id}", _hub.State, id);
				try { await SubscribeAllAsync(_hub, _accountId, CancellationToken.None, _log); }
				catch (Exception ex) { _log.LogWarning(ex, "UserHub SubscribeAllAsync failed after Reconnected"); }
			};

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
			await SubscribeAllAsync(_hub, _accountId, CancellationToken.None, _log);
		}

		private Task SubscribeAllAsync(HubConnection hub, long accountId, CancellationToken ct, ILogger log)
		{
			return HubSafe.InvokeWhenConnected(hub, async () =>
			{
				log?.LogInformation("[UserHub] subscribing …");
				await hub.InvokeAsync("SubscribeAccounts", cancellationToken: ct);
				await hub.InvokeAsync("SubscribeOrders", accountId, cancellationToken: ct);
				await hub.InvokeAsync("SubscribePositions", accountId, cancellationToken: ct);
				await hub.InvokeAsync("SubscribeTrades", accountId, cancellationToken: ct);
			}, log, ct, maxAttempts: 10, waitMs: 500);
		}

		private void WireEvents(HubConnection hub)
		{
			if (_handlersWired) return;
			var concise = (Environment.GetEnvironmentVariable("APP_CONCISE_CONSOLE") ?? "true").Trim().ToLowerInvariant() is "1" or "true" or "yes";

			string? TryGet(JsonElement d, string name)
			{
				if (d.ValueKind != JsonValueKind.Object) return null;
				foreach (var p in d.EnumerateObject())
				{
					if (string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase))
					{
						var v = p.Value;
						switch (v.ValueKind)
						{
							case JsonValueKind.String: return v.GetString();
							case JsonValueKind.Number:
								if (v.TryGetDecimal(out var dec)) return dec.ToString(System.Globalization.CultureInfo.InvariantCulture);
								break;
							case JsonValueKind.True: return "true";
							case JsonValueKind.False: return "false";
						}
					}
				}
				return null;
			}

			string OneLine(JsonElement d, string kind)
			{
				try
				{
					string sym = TryGet(d, "symbol") ?? TryGet(d, "contractName") ?? TryGet(d, "contractId") ?? "?";
					string side = TryGet(d, "side") ?? TryGet(d, "action") ?? string.Empty;
					string qty  = TryGet(d, "qty") ?? TryGet(d, "size") ?? string.Empty;
					string px   = TryGet(d, "price") ?? TryGet(d, "limitPrice") ?? TryGet(d, "fillPrice") ?? string.Empty;
					string st   = TryGet(d, "status") ?? string.Empty;
					var line = $"[{kind}] {sym} {side} {qty} @ {px} {st}".Replace("  ", " ").Trim();
					return line;
				}
				catch { return $"[{kind}]"; }
			}

			hub.On<JsonElement>("GatewayUserAccount", data => { try { if (concise) _log.LogInformation("[ACCOUNT] update"); else _log.LogInformation("Account evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnAccount?.Invoke(data); } catch { } });
			hub.On<JsonElement>("GatewayUserOrder", data => { try { if (concise) _log.LogInformation(OneLine(data, "ORDER")); else _log.LogInformation("Order evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnOrder?.Invoke(data); } catch { } });
			hub.On<JsonElement>("GatewayUserPosition", data => { try { if (concise) _log.LogInformation(OneLine(data, "POSITION")); else _log.LogInformation("Position evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnPosition?.Invoke(data); } catch { } });
			hub.On<JsonElement>("GatewayUserTrade", data => { try { if (concise) _log.LogInformation(OneLine(data, "TRADE")); else _log.LogInformation("Trade evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnTrade?.Invoke(data); } catch { } });
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
