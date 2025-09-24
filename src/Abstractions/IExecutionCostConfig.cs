namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for execution costs and budgets
    /// Replaces hardcoded cost budgets and slippage allowances
    /// </summary>
    public interface IExecutionCostConfig
    {
        /// <summary>
        /// Maximum allowed slippage per trade in USD
        /// </summary>
        decimal GetMaxSlippageUsd();

        /// <summary>
        /// Daily execution cost budget in USD
        /// </summary>
        decimal GetDailyExecutionBudgetUsd();

        /// <summary>
        /// Commission per contract/share
        /// </summary>
        decimal GetCommissionPerContract();

        /// <summary>
        /// Market impact cost multiplier (0.0-1.0)
        /// </summary>
        double GetMarketImpactMultiplier();

        /// <summary>
        /// Expected slippage in ticks for different order types
        /// </summary>
        double GetExpectedSlippageTicks(string orderType);

        /// <summary>
        /// Cost threshold for order routing decisions
        /// </summary>
        decimal GetRoutingCostThresholdUsd();
    }
}