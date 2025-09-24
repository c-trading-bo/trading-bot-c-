namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for position sizing algorithms
    /// Replaces hardcoded sizing parameters in RL/ML models and traditional sizers
    /// </summary>
    public interface ISizerConfig
    {
        /// <summary>
        /// PPO learning rate for RL-based sizing
        /// </summary>
        double GetPpoLearningRate();

        /// <summary>
        /// CQL alpha parameter for conservative Q-learning
        /// </summary>
        double GetCqlAlpha();

        /// <summary>
        /// Meta-cost weight for blending different cost signals
        /// </summary>
        double GetMetaCostWeight(string costType);

        /// <summary>
        /// Position size multiplier baseline (typically 1.0)
        /// </summary>
        double GetPositionSizeMultiplierBaseline();

        /// <summary>
        /// Minimum position size multiplier (lower bound)
        /// </summary>
        double GetMinPositionSizeMultiplier();

        /// <summary>
        /// Maximum position size multiplier (upper bound)
        /// </summary>
        double GetMaxPositionSizeMultiplier();

        /// <summary>
        /// Exploration rate for bandit-based sizing (0.0-1.0)
        /// </summary>
        double GetExplorationRate();

        /// <summary>
        /// Weight floor for strategy selection (minimum allocation)
        /// </summary>
        double GetWeightFloor();

        /// <summary>
        /// Model refresh interval in minutes
        /// </summary>
        int GetModelRefreshIntervalMinutes();
    }
}