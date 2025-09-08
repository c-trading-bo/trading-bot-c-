using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using Trading.Safety;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// OrderService with real /api/Order/place POST with retries and idempotency
/// NO STUBS - Uses actual TopstepX order placement API
/// </summary>
public interface IOrderService
{
    Task<OrderResult> PlaceOrderAsync(PlaceOrderRequest request);
    Task<OrderStatus> GetOrderStatusAsync(string orderId);
    Task<bool> CancelOrderAsync(string orderId);
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
                _logger.LogInformation("[ORDER] DRY_RUN: Would place {Side} {Qty} {Symbol} @ {Price} tag={Tag} account={AccountId}", 
                    request.Side, request.Quantity, request.Symbol, request.Price, request.CustomTag, SecurityHelpers.MaskAccountId(request.AccountId));
                return new OrderResult(true, Guid.NewGuid().ToString(), "DRY_RUN order simulated");
            }

            _logger.LogInformation("[ORDER] LIVE: Placing {Side} {Qty} {Symbol} @ {Price} tag={Tag} account={AccountId}", 
                request.Side, request.Quantity, request.Symbol, request.Price, request.CustomTag, SecurityHelpers.MaskAccountId(request.AccountId));

            // Real POST to /api/Order/place with retries and idempotency
            // This replaces: return Guid.NewGuid().ToString();
            var orderPayload = new
            {
                symbol = request.Symbol,
                side = request.Side,
                quantity = request.Quantity,
                price = request.Price,
                orderType = request.OrderType,
                customTag = request.CustomTag,
                accountId = request.AccountId
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
                        _logger.LogInformation("[ORDER] ✅ Order placed successfully: {OrderId}", orderId);
                        
                        return new OrderResult(true, orderId, "Order placed successfully");
                    }
                    else if (response.StatusCode == System.Net.HttpStatusCode.BadRequest)
                    {
                        // Don't retry 4xx errors
                        var error = await response.Content.ReadAsStringAsync();
                        _logger.LogError("[ORDER] Bad request: {Error}", error);
                        return new OrderResult(false, null, $"Bad request: {error}");
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
            _logger.LogError(ex, "[ORDER] Failed to get status for order {OrderId}", SecurityHelpers.MaskAccountId(orderId));
            throw new InvalidOperationException(SecurityHelpers.GetGenericErrorMessage(ex, _logger));
        }
    }

    public async Task<bool> CancelOrderAsync(string orderId)
    {
        try
        {
            var response = await _httpClient.DeleteAsync($"/api/Order/cancel/{orderId}");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ORDER] Failed to cancel order {OrderId}", SecurityHelpers.MaskAccountId(orderId));
            return false;
        }
    }
}