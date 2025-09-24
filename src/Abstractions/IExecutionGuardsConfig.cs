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
    }
}