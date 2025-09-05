using System;
using System.Collections.Generic;
using System.Linq;

namespace OrchestratorAgent.Intelligence
{
    /// <summary>
    /// Classifies trading strategies into categories for strategy-specific intelligence routing
    /// </summary>
    public static class StrategyClassifier
    {
        public enum StrategyCategory
        {
            Breakout,      // Trend-following, momentum-based entries
            MeanReversion, // Counter-trend, reversal-based entries
            Momentum,      // Trend continuation, momentum persistence
            Scalping       // Quick in/out, microstructure-based
        }

        /// <summary>
        /// Maps strategy names to their primary categories based on analysis of their mechanics
        /// </summary>
        private static readonly Dictionary<string, StrategyCategory> StrategyMapping = new()
        {
            // BREAKOUT STRATEGIES - Clear trend-following/breakout mechanics
            { "S6", StrategyCategory.Breakout },   // Buy breakout with large target (3x ATR)
            { "S8", StrategyCategory.Breakout },   // Keltner channel breakout with EMA rising
            { "S2", StrategyCategory.Breakout },   // S2 appears to be breakout-oriented based on usage patterns
            
            // MEAN REVERSION STRATEGIES - Counter-trend mechanics
            { "S3", StrategyCategory.MeanReversion }, // Classic pullback/reversion strategy
            { "S11", StrategyCategory.MeanReversion }, // Sell reversion with configured ATR multiples
            { "S5", StrategyCategory.MeanReversion },  // Sell with moderate ATR (1.5x stop, 3x target)
            { "S7", StrategyCategory.MeanReversion },  // Sell with larger ATR (2x stop, 4x target)
            
            // MOMENTUM STRATEGIES - Trend continuation with increasing ATR requirements
            { "S1", StrategyCategory.Momentum },   // Basic momentum strategy
            { "S9", StrategyCategory.Momentum },   // High ATR momentum sell (0.8+ ATR, 2.5x stop, 5x target)
            { "S10", StrategyCategory.Momentum },  // High ATR momentum buy (0.9+ ATR, 3x stop, 6x target)
            { "S4", StrategyCategory.Momentum },   // Momentum variant
            
            // SCALPING STRATEGIES - High ATR requirements, large targets (quick moves)
            { "S12", StrategyCategory.Scalping },  // High ATR buy (1.0+ ATR, 3.5x stop, 7x target)
            { "S13", StrategyCategory.Scalping },  // High ATR sell (1.0+ ATR, 3.5x stop, 7x target)
            { "S14", StrategyCategory.Scalping }   // Highest ATR (1.1+ ATR, 4x stop, 8x target)
        };

        /// <summary>
        /// Gets the category for a given strategy name
        /// </summary>
        public static StrategyCategory GetCategory(string strategyName)
        {
            return StrategyMapping.TryGetValue(strategyName, out var category) 
                ? category 
                : StrategyCategory.Breakout; // Default fallback
        }

        /// <summary>
        /// Gets all strategy names in a given category
        /// </summary>
        public static IEnumerable<string> GetStrategiesInCategory(StrategyCategory category)
        {
            return StrategyMapping.Where(kvp => kvp.Value == category).Select(kvp => kvp.Key);
        }

        /// <summary>
        /// Gets environment variable prefix for a strategy category
        /// </summary>
        public static string GetEnvironmentPrefix(StrategyCategory category)
        {
            return category switch
            {
                StrategyCategory.Breakout => "BREAKOUT",
                StrategyCategory.MeanReversion => "REVERSION", 
                StrategyCategory.Momentum => "MOMENTUM",
                StrategyCategory.Scalping => "SCALP",
                _ => "GENERIC"
            };
        }

        /// <summary>
        /// Determines if a strategy should benefit from trending regime intelligence
        /// </summary>
        public static bool BenefitsFromTrendingRegime(string strategyName)
        {
            var category = GetCategory(strategyName);
            return category == StrategyCategory.Breakout || category == StrategyCategory.Momentum;
        }

        /// <summary>
        /// Determines if a strategy should benefit from ranging regime intelligence
        /// </summary>
        public static bool BenefitsFromRangingRegime(string strategyName)
        {
            var category = GetCategory(strategyName);
            return category == StrategyCategory.MeanReversion;
        }

        /// <summary>
        /// Determines if a strategy should benefit from microstructure intelligence
        /// </summary>
        public static bool BenefitsFromMicrostructure(string strategyName)
        {
            var category = GetCategory(strategyName);
            return category == StrategyCategory.Scalping || category == StrategyCategory.Momentum;
        }

        /// <summary>
        /// Gets strategy-specific intelligence requirements
        /// </summary>
        public static StrategyIntelligenceRequirements GetIntelligenceRequirements(string strategyName)
        {
            var category = GetCategory(strategyName);
            
            return new StrategyIntelligenceRequirements
            {
                Category = category,
                RequiresTrendingRegime = BenefitsFromTrendingRegime(strategyName),
                RequiresRangingRegime = BenefitsFromRangingRegime(strategyName),
                RequiresMicrostructure = BenefitsFromMicrostructure(strategyName),
                RequiresZoneBreakoutAnalysis = category == StrategyCategory.Breakout,
                RequiresZoneReversalAnalysis = category == StrategyCategory.MeanReversion,
                RequiresMomentumPersistence = category == StrategyCategory.Momentum,
                RequiresOrderFlowTiming = category == StrategyCategory.Scalping || category == StrategyCategory.Momentum,
                SentimentFilterType = category switch
                {
                    StrategyCategory.Breakout => SentimentFilterType.MomentumWeighted,
                    StrategyCategory.MeanReversion => SentimentFilterType.Contrarian,
                    StrategyCategory.Momentum => SentimentFilterType.TrendConfirming,
                    StrategyCategory.Scalping => SentimentFilterType.ShortTermBias,
                    _ => SentimentFilterType.None
                }
            };
        }
    }

    /// <summary>
    /// Intelligence requirements for a specific strategy
    /// </summary>
    public class StrategyIntelligenceRequirements
    {
        public StrategyClassifier.StrategyCategory Category { get; set; }
        public bool RequiresTrendingRegime { get; set; }
        public bool RequiresRangingRegime { get; set; }
        public bool RequiresMicrostructure { get; set; }
        public bool RequiresZoneBreakoutAnalysis { get; set; }
        public bool RequiresZoneReversalAnalysis { get; set; }
        public bool RequiresMomentumPersistence { get; set; }
        public bool RequiresOrderFlowTiming { get; set; }
        public SentimentFilterType SentimentFilterType { get; set; }
    }

    /// <summary>
    /// Types of sentiment filtering for different strategy categories
    /// </summary>
    public enum SentimentFilterType
    {
        None,              // No sentiment filtering
        MomentumWeighted,  // Breakout strategies - weight sentiment by momentum
        Contrarian,        // Mean reversion - fade extreme sentiment
        TrendConfirming,   // Momentum - confirm trend direction with sentiment
        ShortTermBias      // Scalping - use sentiment for directional bias
    }
}
