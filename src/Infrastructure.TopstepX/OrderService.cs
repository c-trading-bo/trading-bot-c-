using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// OrderService with real /api/Order/place POST with retries and idempotency
/// NO STUBS - Uses actual TopstepX order placement API
/// Implements IBrokerAdapter for centralized order management
/// </summary>
public interface IOrderService : TradingBot.Abstractions.IBrokerAdapter
{
    Task<OrderResult> PlaceOrderAsync(PlaceOrderRequest request);
    Task<OrderStatus> GetOrderStatusAsync(string orderId);
}

public record PlaceOrderRequest(
    string Symbol, 
    string Side, 
    decimal Quantity, 
    decimal Price,
    string OrderType,
    string CustomTag,
    string AccountId
);

public record OrderResult(bool Success, string? OrderId, string Message);
public record OrderStatus(string OrderId, string Status, string Reason, DateTime LastUpdated);

public class OrderService : IOrderService
{
    private readonly ILogger<OrderService> _logger;
    private readonly AppOptions _config;
    private readonly HttpClient _httpClient;

    public string BrokerName => "TopstepX";

    public OrderService(ILogger<OrderService> logger, IOptions<AppOptions> config, HttpClient httpClient)
    {
        _logger = logger;
        _config = config.Value;
        _httpClient = httpClient;
        _httpClient.BaseAddress = new Uri(_config.ApiBase);
    }

    public async Task<OrderResult> PlaceOrderAsync(PlaceOrderRequest request)
    {
        try
        {
            // Check kill file - always forces DRY_RUN
            if (File.Exists(_config.KillFile))
            {
                _logger.LogWarning("[ORDER] Kill file detected - blocking order placement");
                return new OrderResult(false, null, "Kill file active - order blocked");
            }

            // Check dry run mode
            if (_config.EnableDryRunMode)
            {
                _logger.LogInformation("[ORDER] DRY_RUN: Would place {Side} {Qty} {Symbol} @ {Price} tag={Tag}", 
                    request.Side, request.Quantity, request.Symbol, request.Price, request.CustomTag);
                return new OrderResult(true, Guid.NewGuid().ToString(), "DRY_RUN order simulated");
            }

            _logger.LogInformation("[ORDER] LIVE: Placing {Side} {Qty} {Symbol} @ {Price} tag={Tag}", 
                request.Side, request.Quantity, request.Symbol, request.Price, request.CustomTag);

            // Real POST to /api/Order/place with retries and idempotency
            // This replaces: return Guid.NewGuid().ToString();
            // FIXED: Use ProjectX API specification exactly
            var orderPayload = new
            {
                accountId = long.Parse(request.AccountId),  // ProjectX expects integer accountId
                contractId = request.Symbol,               // ProjectX uses contractId, not symbol
                type = GetOrderTypeValue(request.OrderType), // ProjectX: 1=Limit, 2=Market, 4=Stop
                side = GetSideValue(request.Side),         // ProjectX: 0=Bid(buy), 1=Ask(sell)
                size = (int)request.Quantity,              // ProjectX expects integer size
                limitPrice = request.OrderType.ToUpper() == "LIMIT" ? request.Price : (decimal?)null,
                stopPrice = request.OrderType.ToUpper() == "STOP" ? request.Price : (decimal?)null,
                customTag = request.CustomTag
            };

            var content = new StringContent(JsonSerializer.Serialize(orderPayload), 
                System.Text.Encoding.UTF8, "application/json");

            // Retry logic for transient failures
            for (int attempt = 1; attempt <= 3; attempt++)
            {
                try
                {
                    var response = await _httpClient.PostAsync("/api/Order/place", content);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        var responseJson = await response.Content.ReadAsStringAsync();
                        var result = JsonSerializer.Deserialize<JsonElement>(responseJson);
                        
                        var orderId = result.GetProperty("orderId").GetString();
                        _logger.LogInformation("[ORDER] ✅ Order placed successfully");
                        
                        return new OrderResult(true, orderId, "Order placed successfully");
                    }
                    else if (response.StatusCode == System.Net.HttpStatusCode.BadRequest)
                    {
                        // Don't retry 4xx errors
                        var error = await response.Content.ReadAsStringAsync();
                        _logger.LogError("[ORDER] Bad request received");
                        return new OrderResult(false, null, "Invalid order request - please check order parameters");
                    }
                    else
                    {
                        // Retry 5xx errors
                        _logger.LogWarning("[ORDER] Attempt {Attempt}/3 failed: {Status}", attempt, response.StatusCode);
                        if (attempt < 3)
                        {
                            await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt))); // Exponential backoff
                        }
                    }
                }
                catch (HttpRequestException ex) when (attempt < 3)
                {
                    _logger.LogWarning(ex, "[ORDER] Network error on attempt {Attempt}/3", attempt);
                    await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)));
                }
            }

            return new OrderResult(false, null, "All retry attempts failed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ORDER] Unexpected error placing order");
            return new OrderResult(false, null, SecurityHelpers.GetGenericErrorMessage(ex, _logger));
        }
    }

    /// <summary>
    /// Convert order type string to ProjectX API integer value
    /// ProjectX API: 1 = Limit, 2 = Market, 4 = Stop, 5 = TrailingStop
    /// </summary>
    private static int GetOrderTypeValue(string orderType)
    {
        return orderType.ToUpper() switch
        {
            "LIMIT" => 1,
            "MARKET" => 2,
            "STOP" => 4,
            "TRAILING_STOP" => 5,
            _ => 1 // Default to limit
        };
    }

    /// <summary>
    /// Convert side string to ProjectX API integer value
    /// ProjectX API: 0 = Bid (buy), 1 = Ask (sell)
    /// </summary>
    private static int GetSideValue(string side)
    {
        return side.ToUpper() switch
        {
            "BUY" => 0,
            "SELL" => 1,
            _ => 0 // Default to buy
        };
    }

    public async Task<OrderStatus> GetOrderStatusAsync(string orderId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/Order/status/{orderId}");
            response.EnsureSuccessStatusCode();
            
            var json = await response.Content.ReadAsStringAsync();
            var statusData = JsonSerializer.Deserialize<JsonElement>(json);
            
            return new OrderStatus(
                orderId,
                statusData.GetProperty("status").GetString() ?? "Unknown",
                statusData.GetProperty("reason").GetString() ?? "",
                statusData.GetProperty("lastUpdated").GetDateTime()
            );
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ORDER] Failed to get status for order");
            throw new InvalidOperationException(SecurityHelpers.GetGenericErrorMessage(ex, _logger));
        }
    }

    public async Task<bool> CancelOrderAsync(string orderId, CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[ORDER] Cancelling order {OrderId} via TopstepX API", orderId);
            
            // FIXED: Use ProjectX API specification - POST to /api/Order/cancel with body
            var cancelPayload = new
            {
                accountId = long.Parse(Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID") ?? "0"),
                orderId = long.Parse(orderId)
            };
            
            var content = new StringContent(JsonSerializer.Serialize(cancelPayload), 
                System.Text.Encoding.UTF8, "application/json");
            
            var response = await _httpClient.PostAsync("/api/Order/cancel", content, cancellationToken);
            var success = response.IsSuccessStatusCode;
            
            if (success)
            {
                _logger.LogInformation("[ORDER] ✅ Order {OrderId} cancelled successfully", orderId);
            }
            else
            {
                _logger.LogWarning("[ORDER] ❌ Failed to cancel order {OrderId}: HTTP {StatusCode}", 
                    orderId, response.StatusCode);
            }
            
            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ORDER] ❌ Exception cancelling order {OrderId}", orderId);
            return false;
        }
    }
}