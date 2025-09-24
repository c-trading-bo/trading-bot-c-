namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for execution policies and order types
    /// Replaces hardcoded order type defaults and execution policies
    /// </summary>
    public interface IExecutionPolicyConfig
    {
        /// <summary>
        /// Default order type for market entry
        /// </summary>
        string GetDefaultEntryOrderType();

        /// <summary>
        /// Default order type for position exits
        /// </summary>
        string GetDefaultExitOrderType();

        /// <summary>
        /// Whether to use aggressive fills during high volatility
        /// </summary>
        bool UseAggressiveFillsDuringVolatility();

        /// <summary>
        /// Time-in-force for limit orders in seconds
        /// </summary>
        int GetLimitOrderTimeoutSeconds();

        /// <summary>
        /// Whether to enable smart order routing
        /// </summary>
        bool EnableSmartOrderRouting();

        /// <summary>
        /// Maximum order size for iceberg orders
        /// </summary>
        int GetMaxIcebergSize();

        /// <summary>
        /// Minimum order size for execution
        /// </summary>
        int GetMinOrderSize();
    }
}