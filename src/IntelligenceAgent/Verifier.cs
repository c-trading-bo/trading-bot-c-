using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
// Legacy removed: using TradingBot.Infrastructure.TopstepX;

namespace TradingBot.IntelligenceAgent;

/// <summary>
/// Trade verification service for TopstepX order and trade verification
/// Implements VerifyTodayAsync to query /api/Order/search and /api/Trade/search
/// </summary>
public interface IVerifier
{
    Task<VerificationResult> VerifyTodayAsync(CancellationToken cancellationToken = default);
}

public class Verifier : IVerifier
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<Verifier> _logger;

    public Verifier(HttpClient httpClient, ILogger<Verifier> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
    }

    public async Task<VerificationResult> VerifyTodayAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var utcToday = DateTime.UtcNow.Date;
            var utcNow = DateTime.UtcNow;

            _logger.LogInformation("Starting trade verification for UTC today: {UtcToday} to {UtcNow}", 
                utcToday, utcNow);

            // Query orders for today
            var orderStats = await QueryOrdersAsync(utcToday, utcNow, cancellationToken)
                .ConfigureAwait(false);

            // Query trades for today  
            var tradeStats = await QueryTradesAsync(utcToday, utcNow, cancellationToken)
                .ConfigureAwait(false);

            var result = new VerificationResult
            {
                Date = utcToday,
                OrdersByStatus = orderStats,
                TradesByStatus = tradeStats,
                Success = true,
                Timestamp = DateTime.UtcNow
            };

            // Emit human summary
            EmitHumanSummary(result);

            // Emit structured JSON line
            EmitStructuredLog(result);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Trade verification failed");

            // Structured error log without secret exposure
            var errorData = new
            {
                timestamp = DateTime.UtcNow,
                component = "verifier",
                operation = "verify_today",
                success = false,
                error_type = ex.GetType().Name,
                // No stack trace or inner details that might contain secrets
                reason = "verification_failed"
            };

            _logger.LogError("VERIFICATION_ERROR: {ErrorData}", 
                JsonSerializer.Serialize(errorData));

            return new VerificationResult
            {
                Date = DateTime.UtcNow.Date,
                Success = false,
                ErrorMessage = ex.Message,
                Timestamp = DateTime.UtcNow
            };
        }
    }

    private async Task<Dictionary<string, int>> QueryOrdersAsync(
        DateTime fromUtc, 
        DateTime toUtc, 
        CancellationToken cancellationToken)
    {
        var orderCounts = new Dictionary<string, int>
        {
            ["Placed"] = 0,
            ["PartiallyFilled"] = 0,
            ["Filled"] = 0,
            ["Cancelled"] = 0,
            ["Rejected"] = 0
        };

        try
        {
            var fromParam = fromUtc.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");
            var toParam = toUtc.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");
            
            var requestUri = $"/api/Order/search?from={fromParam}&to={toParam}";
            
            using var response = await _httpClient.GetAsync(requestUri, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("Order search returned {StatusCode}: {ReasonPhrase}", 
                    response.StatusCode, response.ReasonPhrase);
                return orderCounts;
            }

            var json = await response.Content.ReadAsStringAsync(cancellationToken)
                .ConfigureAwait(false);

            using var document = JsonDocument.Parse(json);
            
            if (document.RootElement.TryGetProperty("orders", out var ordersElement))
            {
                foreach (var order in ordersElement.EnumerateArray())
                {
                    if (order.TryGetProperty("status", out var statusElement))
                    {
                        var status = statusElement.GetString() ?? "Unknown";
                        if (orderCounts.ContainsKey(status))
                        {
                            orderCounts[status]++;
                        }
                    }
                }
            }

            _logger.LogDebug("Order query completed: {OrderCounts}", orderCounts);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to query orders");
        }

        return orderCounts;
    }

    private async Task<Dictionary<string, int>> QueryTradesAsync(
        DateTime fromUtc, 
        DateTime toUtc, 
        CancellationToken cancellationToken)
    {
        var tradeCounts = new Dictionary<string, int>
        {
            ["Executed"] = 0,
            ["Settled"] = 0
        };

        try
        {
            var fromParam = fromUtc.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");
            var toParam = toUtc.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");
            
            var requestUri = $"/api/Trade/search?from={fromParam}&to={toParam}";
            
            using var response = await _httpClient.GetAsync(requestUri, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("Trade search returned {StatusCode}: {ReasonPhrase}", 
                    response.StatusCode, response.ReasonPhrase);
                return tradeCounts;
            }

            var json = await response.Content.ReadAsStringAsync(cancellationToken)
                .ConfigureAwait(false);

            using var document = JsonDocument.Parse(json);
            
            if (document.RootElement.TryGetProperty("trades", out var tradesElement))
            {
                foreach (var trade in tradesElement.EnumerateArray())
                {
                    if (trade.TryGetProperty("status", out var statusElement))
                    {
                        var status = statusElement.GetString() ?? "Unknown";
                        if (tradeCounts.ContainsKey(status))
                        {
                            tradeCounts[status]++;
                        }
                        else
                        {
                            // Count all trades as executed by default
                            tradeCounts["Executed"]++;
                        }
                    }
                    else
                    {
                        // No status property - assume executed
                        tradeCounts["Executed"]++;
                    }
                }
            }

            _logger.LogDebug("Trade query completed: {TradeCounts}", tradeCounts);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to query trades");
        }

        return tradeCounts;
    }

    private void EmitHumanSummary(VerificationResult result)
    {
        var totalOrders = result.OrdersByStatus.Values.Sum();
        var totalTrades = result.TradesByStatus.Values.Sum();

        var summary = $"Trade Verification Summary for {result.Date:yyyy-MM-dd}:\n" +
                     $"Orders: {totalOrders} total " +
                     $"(Placed: {result.OrdersByStatus.GetValueOrDefault("Placed", 0)}, " +
                     $"Filled: {result.OrdersByStatus.GetValueOrDefault("Filled", 0)}, " +
                     $"Cancelled: {result.OrdersByStatus.GetValueOrDefault("Cancelled", 0)}, " +
                     $"Rejected: {result.OrdersByStatus.GetValueOrDefault("Rejected", 0)})\n" +
                     $"Trades: {totalTrades} total " +
                     $"(Executed: {result.TradesByStatus.GetValueOrDefault("Executed", 0)})";

        _logger.LogInformation("{Summary}", summary);
    }

    private void EmitStructuredLog(VerificationResult result)
    {
        var structuredData = new
        {
            timestamp = result.Timestamp,
            component = "verifier", 
            operation = "verify_today",
            date = result.Date.ToString("yyyy-MM-dd"),
            success = result.Success,
            orders_total = result.OrdersByStatus.Values.Sum(),
            orders_by_status = result.OrdersByStatus,
            trades_total = result.TradesByStatus.Values.Sum(),
            trades_by_status = result.TradesByStatus
        };

        _logger.LogInformation("VERIFICATION_RESULT: {StructuredData}", 
            JsonSerializer.Serialize(structuredData));
    }
}

public class VerificationResult
{
    public DateTime Date { get; set; }
    public Dictionary<string, int> OrdersByStatus { get; } = new();
    public Dictionary<string, int> TradesByStatus { get; } = new();
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public DateTime Timestamp { get; set; }
}