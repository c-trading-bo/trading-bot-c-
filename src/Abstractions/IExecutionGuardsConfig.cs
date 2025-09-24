using System;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for execution guards and microstructure limits
    /// Replaces hardcoded spread/latency/volume/imbalance caps in execution logic
    /// </summary>
    public interface IExecutionGuardsConfig
    {
        /// <summary>
        /// Maximum allowed spread in ticks before rejecting orders
        /// </summary>
        double GetMaxSpreadTicks();

        /// <summary>
        /// Maximum allowed latency in milliseconds for order execution
        /// </summary>
        int GetMaxLatencyMs();

        /// <summary>
        /// Minimum volume threshold for order placement
        /// </summary>
        long GetMinVolumeThreshold();

        /// <summary>
        /// Maximum allowed bid-ask imbalance ratio
        /// </summary>
        double GetMaxImbalanceRatio();

        /// <summary>
        /// Maximum limit offset from current price in ticks
        /// </summary>
        double GetMaxLimitOffsetTicks();

        /// <summary>
        /// Circuit breaker threshold for order rejections per minute
        /// </summary>
        int GetCircuitBreakerThreshold();

        /// <summary>
        /// Trade analysis window in minutes
        /// </summary>
        int GetTradeAnalysisWindowMinutes();

        /// <summary>
        /// Volume analysis window in minutes  
        /// </summary>
        int GetVolumeAnalysisWindowMinutes();

        /// <summary>
        /// Micro volatility threshold for volatile market detection
        /// </summary>
        decimal GetMicroVolatilityThreshold();

        /// <summary>
        /// Maximum spread in basis points for liquid market
        /// </summary>
        decimal GetMaxSpreadBps();
    }
}