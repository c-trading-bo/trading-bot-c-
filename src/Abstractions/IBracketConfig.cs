namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for bracket orders (take-profit/stop-loss)
    /// Replaces hardcoded TP/SL distances and bracket management parameters
    /// </summary>
    public interface IBracketConfig
    {
        /// <summary>
        /// Default take-profit distance in ATR multiples
        /// </summary>
        double GetDefaultTakeProfitAtrMultiple();

        /// <summary>
        /// Default stop-loss distance in ATR multiples
        /// </summary>
        double GetDefaultStopLossAtrMultiple();

        /// <summary>
        /// Minimum reward-to-risk ratio for bracket trades
        /// </summary>
        double GetMinRewardRiskRatio();

        /// <summary>
        /// Maximum reward-to-risk ratio cap
        /// </summary>
        double GetMaxRewardRiskRatio();

        /// <summary>
        /// Whether to enable trailing stops
        /// </summary>
        bool EnableTrailingStops();

        /// <summary>
        /// Trailing stop distance in ATR multiples
        /// </summary>
        double GetTrailingStopAtrMultiple();

        /// <summary>
        /// Default bracket mode ("OCO", "OSO", "Manual")
        /// </summary>
        string GetDefaultBracketMode();

        /// <summary>
        /// Whether bracket orders should be reduce-only
        /// </summary>
        bool BracketOrdersReduceOnly();

        /// <summary>
        /// Partial fill handling for bracket orders
        /// </summary>
        string GetPartialFillHandling();
        
        // Additional methods needed by consuming code
        double GetDefaultStopAtrMultiple();
        double GetDefaultTargetAtrMultiple();
        bool EnableTrailingStop { get; }
        bool ReduceOnlyMode { get; }
    }
}