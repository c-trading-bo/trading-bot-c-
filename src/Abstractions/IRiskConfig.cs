namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for risk management parameters
    /// Replaces hardcoded risk limits and position sizing rules
    /// </summary>
    public interface IRiskConfig
    {
        /// <summary>
        /// Maximum daily loss limit in USD
        /// </summary>
        decimal GetMaxDailyLossUsd();

        /// <summary>
        /// Maximum weekly loss limit in USD
        /// </summary>
        decimal GetMaxWeeklyLossUsd();

        /// <summary>
        /// Risk per trade as percentage of account equity (0.0-1.0)
        /// </summary>
        double GetRiskPerTradePercent();

        /// <summary>
        /// Fixed risk per trade in USD (fallback when equity unknown)
        /// </summary>
        decimal GetFixedRiskPerTradeUsd();

        /// <summary>
        /// Maximum number of open positions
        /// </summary>
        int GetMaxOpenPositions();

        /// <summary>
        /// Maximum consecutive losses before cooldown
        /// </summary>
        int GetMaxConsecutiveLosses();

        /// <summary>
        /// CVaR confidence level (0.0-1.0, typically 0.95)
        /// </summary>
        double GetCvarConfidenceLevel();

        /// <summary>
        /// Target R-multiple for CVaR sizing
        /// </summary>
        double GetCvarTargetRMultiple();

        /// <summary>
        /// Regime-specific drawdown multipliers
        /// </summary>
        double GetRegimeDrawdownMultiplier(string regimeType);
    }
}