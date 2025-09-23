using System;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Helper class containing workflow execution utilities
/// Extracted from IntelligenceOrchestrator to reduce file size
/// </summary>
public static class WorkflowHelpers
{
    // Constants for market data defaults
    private const double DefaultMarketDataOpen = 5500.0;
    private const double DefaultMarketDataHigh = 5520.0;
    private const double DefaultMarketDataLow = 5480.0;
    private const double DefaultMarketDataClose = 5510.0;
    private const double DefaultVolume = 1000.0;
    private const double DefaultMarketDataBid = 5509.75;
    private const double DefaultMarketDataAsk = 5510.25;

    /// <summary>
    /// Extract market data from workflow context
    /// </summary>
    public static MarketData ExtractMarketDataFromWorkflow(WorkflowExecutionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        
        return new MarketData
        {
            Symbol = context.Parameters.GetValueOrDefault("symbol", "ES")?.ToString() ?? "ES",
            Open = Convert.ToDouble(context.Parameters.GetValueOrDefault("open", DefaultMarketDataOpen), CultureInfo.InvariantCulture),
            High = Convert.ToDouble(context.Parameters.GetValueOrDefault("high", DefaultMarketDataHigh), CultureInfo.InvariantCulture),
            Low = Convert.ToDouble(context.Parameters.GetValueOrDefault("low", DefaultMarketDataLow), CultureInfo.InvariantCulture),
            Close = Convert.ToDouble(context.Parameters.GetValueOrDefault("close", DefaultMarketDataClose), CultureInfo.InvariantCulture),
            Volume = Convert.ToDouble(context.Parameters.GetValueOrDefault("volume", DefaultVolume), CultureInfo.InvariantCulture),
            Bid = Convert.ToDouble(context.Parameters.GetValueOrDefault("bid", DefaultMarketDataBid), CultureInfo.InvariantCulture),
            Ask = Convert.ToDouble(context.Parameters.GetValueOrDefault("ask", DefaultMarketDataAsk), CultureInfo.InvariantCulture),
            Timestamp = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Create a successful workflow execution result
    /// </summary>
    public static WorkflowExecutionResult CreateSuccessResult()
    {
        return new WorkflowExecutionResult { Success = true };
    }
}