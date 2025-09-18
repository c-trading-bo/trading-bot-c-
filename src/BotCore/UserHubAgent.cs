// Agent: UserHubAgent
// Role: Manages SignalR user hub connection, event wiring, and logging for TopstepX.
// Integration: Used by orchestrator and other agents for user event streaming and order/trade logging.
using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Net.WebSockets;
using Microsoft.AspNetCore.Http.Connections;
using BotCore.Models;
using System.Globalization;

namespace BotCore
{
    /// <summary>
    /// Shared user hub logic for TopstepX events and logging.
    /// Also exposes typed JSON events for consumers (positions/trades/orders/accounts).
    /// </summary>
    public sealed class UserHubAgent(ILogger<UserHubAgent> log, object? statusService) : IAsyncDisposable
    {
        private readonly ILogger<UserHubAgent> _log = log;
        private readonly object? _statusService = statusService;
        private HubConnection _hub = default!;
        private bool _handlersWired;
        private long _accountId;
        private string? _jwt;

        public event Action<JsonElement>? OnOrder;
        public event Action<JsonElement>? OnTrade;
        public event Action<JsonElement>? OnPosition;
        public event Action<JsonElement>? OnAccount;

        // 4️⃣ Wire UserHub Event Handlers - Add structured event handlers
        public event Action<OrderConfirmation>? OnOrderConfirmation;
        public event Action<FillConfirmation>? OnFillConfirmation;

        public HubConnection? GetConnection() => _hub;
        public HubConnection? Connection => _hub;

        private const string CloseStatusKey = "CloseStatus";

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
            // Build URL — include token only when explicitly enabled by RTC_URL_TOKEN=1
            var allowUrlToken = (Environment.GetEnvironmentVariable("RTC_URL_TOKEN") ?? "1").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            var url = (!allowUrlToken || string.IsNullOrWhiteSpace(_jwt)) ? baseUrl : $"{baseUrl}?access_token={Uri.EscapeDataString(_jwt)}";
            _log.LogInformation("[UserHub] Using URL={Url}", baseUrl);

            _hub = new HubConnectionBuilder()
                .WithUrl(url, opt =>
                {
                    // AccessTokenProvider in newer packages is Func<Task<string?>>; return nullable to match
                    opt.AccessTokenProvider = () => Task.FromResult<string?>(_jwt);
                    // Force WebSockets as in last known-good wiring
                    opt.Transports = HttpTransportType.WebSockets;
                })
                .WithAutomaticReconnect([TimeSpan.Zero, TimeSpan.FromMilliseconds(500), TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5)])
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
                        // Always suppress verbose client transport logs that can echo access_token in URLs
                        lb.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Error);
                        lb.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Error);
                        lb.AddFilter("Microsoft.AspNetCore.Http.Connections.Client.Internal.WebSocketsTransport", LogLevel.Error);
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
                try { await SubscribeAllAsync(_hub, _accountId, CancellationToken.None, _log).ConfigureAwait(false); }
                catch (Exception ex) { _log.LogWarning(ex, "UserHub SubscribeAllAsync failed after Reconnected"); }
            };

            // Closed logging is handled in WireEvents to avoid duplicate handlers
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

            await _hub.StartAsync(ct).ConfigureAwait(false);
            _log.LogInformation("UserHub connected. State={State}", _hub.State);
            StatusSet("user.state", _hub.ConnectionId ?? string.Empty);
            await Task.Delay(250, ct).ConfigureAwait(false); // ensure server is ready before subscribing
            await SubscribeAllAsync(_hub, _accountId, CancellationToken.None, _log).ConfigureAwait(false);
        }

        private static Task SubscribeAllAsync(HubConnection hub, long accountId, CancellationToken ct, ILogger log)
        {
            return HubSafe.InvokeWhenConnected(hub, async () =>
            {
                log?.LogInformation("[UserHub] subscribing …");
                await hub.InvokeAsync("SubscribeAccounts", cancellationToken: ct).ConfigureAwait(false);
                await hub.InvokeAsync("SubscribeOrders", accountId, cancellationToken: ct).ConfigureAwait(false);
                await hub.InvokeAsync("SubscribePositions", accountId, cancellationToken: ct).ConfigureAwait(false);
                await hub.InvokeAsync("SubscribeTrades", accountId, cancellationToken: ct).ConfigureAwait(false);
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
                    string qty = TryGet(d, "qty") ?? TryGet(d, "size") ?? string.Empty;
                    string px = TryGet(d, "price") ?? TryGet(d, "limitPrice") ?? TryGet(d, "fillPrice") ?? string.Empty;
                    string st = TryGet(d, "status") ?? string.Empty;
                    var line = $"[{kind}] {sym} {side} {qty} @ {px} {st}".Replace("  ", " ").Trim();
                    return line;
                }
                catch { return $"[{kind}]"; }
            }

            hub.On<JsonElement>("GatewayUserAccount", data => { try { if (concise) _log.LogInformation("[ACCOUNT] update"); else _log.LogInformation("Account evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnAccount?.Invoke(data); } catch { } });
            hub.On<JsonElement>("GatewayUserOrder", data => { 
                try { 
                    // 10️⃣ Standardize Logging Format - Order events
                    var orderId = TryGet(data, "orderId") ?? TryGet(data, "id") ?? "unknown";
                    var status = TryGet(data, "status") ?? "unknown";
                    var symbol = TryGet(data, "symbol") ?? TryGet(data, "contractName") ?? "unknown";
                    var accountIdStr = TryGet(data, "accountId") ?? "unknown";
                    
                    _log.LogInformation("ORDER account={AccountId} status={Status} orderId={OrderId} symbol={Symbol}", 
                        accountIdStr, status, orderId, symbol);
                        
                    if (!concise) _log.LogInformation("Order evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); 
                    OnOrder?.Invoke(data); 
                    
                    // 4️⃣ Wire UserHub Event Handlers - Parse and emit structured event
                    try
                    {
                        var orderConfirmation = ParseOrderConfirmation(data);
                        if (orderConfirmation != null)
                        {
                            OnOrderConfirmation?.Invoke(orderConfirmation);
                        }
                    }
                    catch (Exception parseEx)
                    {
                        _log.LogDebug(parseEx, "Failed to parse order confirmation from GatewayUserOrder");
                    }
                } catch { } 
            });
            hub.On<JsonElement>("GatewayUserPosition", data => { try { if (concise) _log.LogInformation(OneLine(data, "POSITION")); else _log.LogInformation("Position evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); OnPosition?.Invoke(data); } catch { } });
            hub.On<JsonElement>("GatewayUserTrade", data => { 
                try { 
                    // 10️⃣ Standardize Logging Format - Trade events
                    var orderId = TryGet(data, "orderId") ?? TryGet(data, "id") ?? "unknown";
                    var fillPrice = TryGet(data, "fillPrice") ?? TryGet(data, "price") ?? "0.00";
                    var quantity = TryGet(data, "quantity") ?? TryGet(data, "qty") ?? "0";
                    var accountIdStr = TryGet(data, "accountId") ?? "unknown";
                    var customTag = TryGet(data, "customTag") ?? TryGet(data, "tag") ?? "unknown";
                    var timestamp = TryGet(data, "timestamp") ?? TryGet(data, "time") ?? DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture);
                    
                    // Parse and format fillPrice to ensure 2 decimals
                    if (decimal.TryParse(fillPrice, out var fillPriceDecimal))
                    {
                        fillPrice = fillPriceDecimal.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);
                    }
                    
                    _log.LogInformation("TRADE account={AccountId} orderId={OrderId} fillPrice={FillPrice} qty={Quantity} time={Time} tag={CustomTag}", 
                        accountIdStr, orderId, fillPrice, quantity, timestamp, customTag);
                        
                    if (!concise) _log.LogInformation("Trade evt: {Json}", System.Text.Json.JsonSerializer.Serialize(data)); 
                    OnTrade?.Invoke(data); 
                    
                    // 4️⃣ Wire UserHub Event Handlers - Parse and emit structured event
                    try
                    {
                        var fillConfirmation = ParseFillConfirmation(data);
                        if (fillConfirmation != null)
                        {
                            OnFillConfirmation?.Invoke(fillConfirmation);
                        }
                    }
                    catch (Exception parseEx)
                    {
                        _log.LogDebug(parseEx, "Failed to parse fill confirmation from GatewayUserTrade");
                    }
                } catch { } 
            });
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
                var m = t?.GetMethod("Set", [typeof(string), typeof(object)]);
                m?.Invoke(_statusService, [key, value]);
            }
            catch { }
        }

        /// <summary>
        /// 4️⃣ Wire UserHub Event Handlers - Parse order confirmation from JSON
        /// </summary>
        private OrderConfirmation? ParseOrderConfirmation(JsonElement data)
        {
            try
            {
                if (data.ValueKind != JsonValueKind.Object) return null;

                string? orderId = null;
                string? customTag = null;
                string? status = null;
                string? reason = null;

                if (data.TryGetProperty("orderId", out var orderIdProp)) orderId = orderIdProp.GetString();
                if (data.TryGetProperty("id", out var idProp)) orderId ??= idProp.GetString();
                if (data.TryGetProperty("customTag", out var customTagProp)) customTag = customTagProp.GetString();
                if (data.TryGetProperty("status", out var statusProp)) status = statusProp.GetString();
                if (data.TryGetProperty("reason", out var reasonProp)) reason = reasonProp.GetString();

                if (string.IsNullOrEmpty(orderId)) return null;

                return new OrderConfirmation
                {
                    OrderId = orderId,
                    CustomTag = customTag ?? "",
                    Status = status ?? "",
                    Reason = reason ?? "",
                    Timestamp = DateTime.UtcNow
                };
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// 4️⃣ Wire UserHub Event Handlers - Parse fill confirmation from JSON
        /// </summary>
        private FillConfirmation? ParseFillConfirmation(JsonElement data)
        {
            try
            {
                if (data.ValueKind != JsonValueKind.Object) return null;

                string? orderId = null;
                string? customTag = null;
                decimal fillPrice = 0m;
                int quantity = 0;

                if (data.TryGetProperty("orderId", out var orderIdProp)) orderId = orderIdProp.GetString();
                if (data.TryGetProperty("id", out var idProp)) orderId ??= idProp.GetString();
                if (data.TryGetProperty("customTag", out var customTagProp)) customTag = customTagProp.GetString();
                if (data.TryGetProperty("fillPrice", out var fillPriceProp)) fillPriceProp.TryGetDecimal(out fillPrice);
                if (data.TryGetProperty("price", out var priceProp)) { if (fillPrice == 0m) priceProp.TryGetDecimal(out fillPrice); }
                if (data.TryGetProperty("quantity", out var quantityProp)) quantityProp.TryGetInt32(out quantity);
                if (data.TryGetProperty("qty", out var qtyProp)) { if (quantity == 0) qtyProp.TryGetInt32(out quantity); }

                if (string.IsNullOrEmpty(orderId) || fillPrice <= 0 || quantity <= 0) return null;

                return new FillConfirmation
                {
                    OrderId = orderId,
                    CustomTag = customTag ?? "",
                    FillPrice = fillPrice,
                    Quantity = quantity,
                    Timestamp = DateTime.UtcNow
                };
            }
            catch
            {
                return null;
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (_hub != null)
            {
                await _hub.DisposeAsync().ConfigureAwait(false);
            }
        }
    }
}
