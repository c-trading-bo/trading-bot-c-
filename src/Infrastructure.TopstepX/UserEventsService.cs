using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.AspNetCore.Http.Connections;
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

public class UserEventsService : IUserEventsService, IDisposable
{
    private readonly ILogger<UserEventsService> _logger;
    private readonly AppOptions _config;
    private HubConnection? _hubConnection;
    private bool _isConnected;
    
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
            
            // Build SignalR connection with authentication
            _hubConnection = new HubConnectionBuilder()
                .WithUrl(hubUrl, options =>
                {
                    options.AccessTokenProvider = async () =>
                    {
                        // Use JWT token from environment or config
                        return Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? _config.AuthToken;
                    };
                    
                    options.Transports = HttpTransportType.WebSockets | 
                                       HttpTransportType.LongPolling |
                                       HttpTransportType.ServerSentEvents;
                })
                .WithAutomaticReconnect()
                .Build();

            // Set up event handlers
            _hubConnection.On<JsonElement>("GatewayUserTrade", (data) =>
            {
                try
                {
                    // Parse GatewayUserTrade event and trigger OnTradeConfirmed
                    var orderId = data.GetProperty("orderId").GetString() ?? "";
                    var symbol = data.GetProperty("symbol").GetString() ?? "";
                    var fillPrice = data.GetProperty("fillPrice").GetDecimal();
                    var quantity = data.GetProperty("quantity").GetInt32();
                    var time = data.GetProperty("time").GetDateTime();
                    
                    var confirmation = new TradeConfirmation(orderId, symbol, fillPrice, quantity, time);
                    OnTradeConfirmed?.Invoke(confirmation);
                    
                    _logger.LogInformation("[USER] Trade confirmed: {Symbol} {Quantity} @ {Price}", symbol, quantity, fillPrice);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[USER] Error processing trade confirmation");
                }
            });

            _hubConnection.On<JsonElement>("GatewayUserOrder", (data) =>
            {
                try
                {
                    // Parse GatewayUserOrder event and trigger OnOrderUpdate
                    var orderId = data.GetProperty("orderId").GetString() ?? "";
                    var status = data.GetProperty("status").GetString() ?? "";
                    var reason = data.GetProperty("reason").GetString() ?? "";
                    var time = data.GetProperty("time").GetDateTime();
                    
                    var update = new OrderUpdate(orderId, status, reason, time);
                    OnOrderUpdate?.Invoke(update);
                    
                    _logger.LogInformation("[USER] Order update: {OrderId} status={Status}", orderId, status);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[USER] Error processing order update");
                }
            });

            // Start the connection
            await _hubConnection.StartAsync();
            _isConnected = true;
            
            _logger.LogInformation("[USER] ✅ Connected to live TopstepX user events feed");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[USER] Failed to connect to user events");
            return false;
        }
    }

    public async Task SubscribeToTradesAsync(string accountId)
    {
        try
        {
            _logger.LogInformation("[USER] Subscribing to trades for account {AccountId}", accountId);
            
            if (_hubConnection?.State != HubConnectionState.Connected)
            {
                _logger.LogWarning("[USER] Cannot subscribe - not connected");
                return;
            }
            
            // Real subscription calls to TopstepX SignalR hubs
            await _hubConnection.InvokeAsync("SubscribeOrders", accountId);
            await _hubConnection.InvokeAsync("SubscribeTrades", accountId);
            
            _logger.LogInformation("[USER] ✅ Subscribed to live trade events for account {AccountId}", accountId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[USER] Failed to subscribe to trades");
            throw;
        }
    }

    public void Dispose()
    {
        try
        {
            _hubConnection?.StopAsync().Wait(TimeSpan.FromSeconds(5));
            _hubConnection?.DisposeAsync().AsTask().Wait(TimeSpan.FromSeconds(5));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[USER] Error disposing hub connection");
        }
        
        _hubConnection = null;
        _isConnected = false;
    }
}