using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace BotCore.Execution;

/// <summary>
/// Analyzes market microstructure to predict execution costs and optimal order types.
/// Considers spread, volatility, volume, and order book depth for intelligent execution.
/// </summary>
public interface IMicrostructureAnalyzer
{
    /// <summary>
    /// Analyzes current market conditions for execution planning.
    /// </summary>
    Task<MicrostructureState> AnalyzeCurrentStateAsync(string symbol, CancellationToken ct = default);

    /// <summary>
    /// Predicts slippage for a market order of given size.
    /// </summary>
    Task<decimal> PredictMarketOrderSlippageAsync(
        string symbol,
        decimal quantity,
        bool isBuy,
        MicrostructureState? currentState = null,
        CancellationToken ct = default);

    /// <summary>
    /// Estimates fill probability for a limit order at specified price.
    /// </summary>
    Task<decimal> EstimateLimitOrderFillProbabilityAsync(
        string symbol,
        decimal limitPrice,
        decimal quantity,
        bool isBuy,
        TimeSpan timeHorizon,
        MicrostructureState? currentState = null,
        CancellationToken ct = default);

    /// <summary>
    /// Suggests optimal order type and parameters based on execution analysis.
    /// </summary>
    Task<ExecutionRecommendation> GetExecutionRecommendationAsync(
        TradeIntent intent,
        MicrostructureState? currentState = null,
        CancellationToken ct = default);
}

/// <summary>
/// Current microstructure state for a symbol
/// </summary>
public record MicrostructureState
{
    public string Symbol { get; init; } = "";
    public DateTime Timestamp { get; init; }
    public decimal BidPrice { get; init; }
    public decimal AskPrice { get; init; }
    public decimal BidAskSpread { get; init; }
    public decimal SpreadBps { get; init; }
    public decimal MidPrice { get; init; }
    public long BidSize { get; init; }
    public long AskSize { get; init; }
    public decimal OrderImbalance { get; init; } // (BidSize - AskSize) / (BidSize + AskSize)
    public decimal RecentVolume { get; init; }
    public decimal VolumeRate { get; init; } // Volume per minute
    public decimal Volatility { get; init; } // Recent realized volatility
    public decimal MicroVolatility { get; init; } // High-frequency price movements
    public decimal TickActivity { get; init; } // Recent tick count per minute
    public bool IsVolatile { get; init; } // High micro-volatility flag
    public bool IsLiquid { get; init; } // High volume/tight spread flag
    public MarketSession Session { get; init; }
    public IReadOnlyList<RecentTrade> RecentTrades { get; init; } = new List<RecentTrade>();
}

/// <summary>
/// Trade intent for execution analysis
/// </summary>
public record TradeIntent
{
    public string Symbol { get; init; } = "";
    public decimal Quantity { get; init; }
    public bool IsBuy { get; init; }
    public decimal? LimitPrice { get; init; }
    public decimal MaxSlippageBps { get; init; } = 5m; // Maximum acceptable slippage
    public TimeSpan MaxWaitTime { get; init; } = TimeSpan.FromMinutes(5);
    public ExecutionUrgency Urgency { get; init; } = ExecutionUrgency.Normal;
    public decimal ExpectedWinRate { get; init; } // From meta-labeler
    public decimal RMultiple { get; init; }
}

/// <summary>
/// Execution recommendation with optimal strategy
/// </summary>
public record ExecutionRecommendation
{
    public OrderType RecommendedOrderType { get; init; }
    public decimal? LimitPrice { get; init; }
    public decimal PredictedSlippageBps { get; init; }
    public decimal FillProbability { get; init; }
    public decimal ExpectedCostBps { get; init; }
    public decimal ExpectedValue { get; init; } // EV including costs
    public string Reasoning { get; init; } = "";
    public ExecutionRisk RiskAssessment { get; init; }
    public TimeSpan EstimatedFillTime { get; init; }
}

/// <summary>
/// Recent trade for microstructure analysis
/// </summary>
public record RecentTrade
{
    public DateTime Timestamp { get; init; }
    public decimal Price { get; init; }
    public long Size { get; init; }
    public bool WasBuyerInitiated { get; init; }
}

/// <summary>
/// Market session information
/// </summary>
public enum MarketSession
{
    PreMarket,
    OpeningAuction,
    RegularTrading,
    ClosingAuction,
    AfterHours
}

/// <summary>
/// Execution urgency level
/// </summary>
public enum ExecutionUrgency
{
    Low,      // Can wait for better prices
    Normal,   // Standard execution
    High,     // Need fast execution
    Urgent    // Execute immediately regardless of cost
}

/// <summary>
/// Order type options
/// </summary>
public enum OrderType
{
    Market,
    Limit,
    MarketableLimit, // Limit at best bid/offer
    MidpointLimit,   // Limit at midpoint
    HiddenLimit      // Hidden/iceberg order
}

/// <summary>
/// Execution risk assessment
/// </summary>
public enum ExecutionRisk
{
    Low,     // Liquid market, small size
    Medium,  // Normal conditions
    High,    // Volatile or illiquid
    Extreme  // Very risky execution
}
