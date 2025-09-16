using System.Text.Json;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
// using Trading.Safety;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// UserEventsService with real /hubs/user listener for GatewayUserTrade events
/// NO STUBS - Uses actual TopstepX user hub for trade confirmations
/// </summary>
public interface IUserEventsService
{
    Task<bool> ConnectAsync();
    Task SubscribeToTradesAsync(string accountId);
    event Action<TradeConfirmation> OnTradeConfirmed;
    event Action<OrderUpdate> OnOrderUpdate;
}

public record TradeConfirmation(string OrderId, string Symbol, decimal FillPrice, int Quantity, DateTime Time);
public record OrderUpdate(string OrderId, string Status, string Reason, DateTime Time);

public class UserEventsService : IUserEventsService, IAsyncDisposable, IDisposable
{
    private readonly ILogger<UserEventsService> _logger;
    private readonly AppOptions _config;
    private HubConnection? _hubConnection;
    private bool _isConnected = false;
    
    public event Action<TradeConfirmation>? OnTradeConfirmed;
    public event Action<OrderUpdate>? OnOrderUpdate;

    public UserEventsService(ILogger<UserEventsService> logger, IOptions<AppOptions> config)
    {
        _logger = logger;
        _config = config.Value;
    }

    public async Task<bool> ConnectAsync()
    {
        try
        {
            _logger.LogInformation("[USER] Connecting to TopstepX user hub");
            
            // Real SignalR connection to /hubs/user
            var hubUrl = $"{_config.ApiBase.TrimEnd('/')}/hubs/user";
            
            // Build the HubConnection
            _hubConnection = new HubConnectionBuilder()
                .WithUrl(hubUrl, options =>
                {
                    if (!string.IsNullOrEmpty(_config.AuthToken))
                    {
                        options.Headers.Add("Authorization", $"Bearer {_config.AuthToken}");
                    }
                })
                .WithAutomaticReconnect()
                .Build();

            // Subscribe to real hub events
            _hubConnection.On<object>("GatewayUserTrade", trade =>
            {
                try
                {
                    var tradeJson = JsonSerializer.Serialize(trade);
                    _logger.LogInformation("[USER] Trade confirmed: {Trade}", tradeJson);
                    
                    // Parse and emit trade confirmation
                    if (trade != null)
                    {
                        OnTradeConfirmed?.Invoke(new TradeConfirmation(
                            OrderId: ExtractProperty(trade, "orderId") ?? "UNKNOWN",
                            Symbol: ExtractProperty(trade, "symbol") ?? "UNKNOWN", 
                            FillPrice: ParseDecimal(ExtractProperty(trade, "fillPrice")),
                            Quantity: ParseInt(ExtractProperty(trade, "quantity")),
                            Time: DateTime.UtcNow
                        ));
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[USER] Error processing trade event");
                }
            });

            _hubConnection.On<object>("GatewayUserOrder", order =>
            {
                try
                {
                    var orderJson = JsonSerializer.Serialize(order);
                    _logger.LogInformation("[USER] Order update: {Order}", orderJson);
                    
                    // Parse and emit order update
                    if (order != null)
                    {
                        OnOrderUpdate?.Invoke(new OrderUpdate(
                            OrderId: ExtractProperty(order, "orderId") ?? "UNKNOWN",
                            Status: ExtractProperty(order, "status") ?? "UNKNOWN",
                            Reason: ExtractProperty(order, "reason") ?? "",
                            Time: DateTime.UtcNow
                        ));
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[USER] Error processing order event");
                }
            });

            // Start the connection
            await _hubConnection.StartAsync();
            _isConnected = true;
            
            _logger.LogInformation("[USER] Γ£à Connected to live TopstepX user events feed");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[USER] Failed to connect to user events");
            _isConnected = false;
            return false;
        }
    }

    public async Task SubscribeToTradesAsync(string accountId)
    {
        if (!_isConnected)
        {
            throw new InvalidOperationException("User events service not connected. Call ConnectAsync() first.");
        }
        
        try
        {
            _logger.LogInformation("[USER] Subscribing to trades for account");
            
            if (_hubConnection?.State != HubConnectionState.Connected)
            {
                _logger.LogWarning("[USER] Hub not connected, attempting to connect first");
                if (!await ConnectAsync())
                {
                    throw new InvalidOperationException("Failed to connect to hub");
                }
            }

            // Real subscription calls to TopstepX hub methods
            await _hubConnection!.InvokeAsync("SubscribeOrders", accountId);
            await _hubConnection!.InvokeAsync("SubscribeTrades", accountId);
            
            _logger.LogInformation("[USER] Γ£à Subscribed to live order updates and trade confirmations");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[USER] Failed to subscribe to trades");
            throw;
        }
    }

    // Helper methods for parsing dynamic objects from SignalR
    private static string? ExtractProperty(object obj, string propertyName)
    {
        try
        {
            if (obj is JsonElement element && element.TryGetProperty(propertyName, out var prop))
            {
                return prop.GetString();
            }
            
            // Try reflection as fallback
            var type = obj.GetType();
            var property = type.GetProperty(propertyName, System.Reflection.BindingFlags.IgnoreCase | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            return property?.GetValue(obj)?.ToString();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[USER-EVENTS] Failed to parse user event data: {ex.Message}");
            throw new InvalidOperationException("Failed to parse user event data - invalid JSON format", ex);
        }
    }

    private static decimal ParseDecimal(string? value) => decimal.TryParse(value, out var result) ? result : 0m;
    private static int ParseInt(string? value) => int.TryParse(value, out var result) ? result : 0;

    public async ValueTask DisposeAsync()
    {
        try
        {
            if (_hubConnection != null)
            {
                await _hubConnection.DisposeAsync();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[USER] Error disposing hub connection");
        }
    }

    public void Dispose()
    {
        DisposeAsync().AsTask().GetAwaiter().GetResult();
    }
}
