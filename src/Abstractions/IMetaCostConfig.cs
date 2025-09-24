namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for meta-cost weights in RL/ML models
    /// Replaces hardcoded cost blending parameters
    /// </summary>
    public interface IMetaCostConfig
    {
        /// <summary>
        /// Weight for execution cost in meta-cost calculation
        /// </summary>
        double GetExecutionCostWeight();

        /// <summary>
        /// Weight for market impact cost
        /// </summary>
        double GetMarketImpactWeight();

        /// <summary>
        /// Weight for opportunity cost
        /// </summary>
        double GetOpportunityCostWeight();

        /// <summary>
        /// Weight for timing cost (latency penalties)
        /// </summary>
        double GetTimingCostWeight();

        /// <summary>
        /// Weight for volatility risk cost
        /// </summary>
        double GetVolatilityRiskWeight();

        /// <summary>
        /// Temperature parameter for cost blending (softmax)
        /// </summary>
        double GetCostBlendingTemperature();

        /// <summary>
        /// Whether to normalize cost weights to sum to 1.0
        /// </summary>
        bool NormalizeCostWeights();

        /// <summary>
        /// Adaptive weight adjustment rate (0.0-1.0)
        /// </summary>
        double GetAdaptiveWeightRate();
    }
}