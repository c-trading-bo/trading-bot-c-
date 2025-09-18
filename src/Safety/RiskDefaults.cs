using System;

namespace TradingBot.Safety
{
    /// <summary>
    /// Centralized risk management defaults for the trading system
    /// Replaces scattered risk configuration throughout the codebase
    /// </summary>
    public static class RiskDefaults
    {
        /// <summary>
        /// Default maximum daily loss limit
        /// </summary>
        public static readonly decimal DefaultMaxDailyLoss = 1000m;

        /// <summary>
        /// Default maximum position size
        /// </summary>
        public static readonly decimal DefaultMaxPositionSize = 10000m;

        /// <summary>
        /// Default drawdown limit
        /// </summary>
        public static readonly decimal DefaultDrawdownLimit = 2000m;

        /// <summary>
        /// Default maximum risk per trade as percentage
        /// </summary>
        public static readonly decimal DefaultMaxRiskPerTradePercent = 0.01m; // 1%

        /// <summary>
        /// Default maximum number of trades per day
        /// </summary>
        public static readonly int DefaultMaxDailyTrades = 50;

        /// <summary>
        /// Default maximum number of trades per session
        /// </summary>
        public static readonly int DefaultMaxSessionTrades = 20;

        /// <summary>
        /// Default maximum portfolio exposure
        /// </summary>
        public static readonly decimal DefaultMaxPortfolioExposure = 50000m;

        /// <summary>
        /// Default maximum number of open positions
        /// </summary>
        public static readonly int DefaultMaxOpenPositions = 10;

        /// <summary>
        /// ES/MES tick size for price calculations
        /// </summary>
        public static readonly decimal EsTickSize = 0.25m;

        /// <summary>
        /// NQ/MNQ tick size for price calculations
        /// </summary>
        public static readonly decimal NqTickSize = 0.25m;

        /// <summary>
        /// Default stop loss percentage
        /// </summary>
        public static readonly decimal DefaultStopLossPercent = 0.005m; // 0.5%

        /// <summary>
        /// Default take profit percentage
        /// </summary>
        public static readonly decimal DefaultTakeProfitPercent = 0.01m; // 1%

        /// <summary>
        /// SDK adapter specific risk defaults
        /// </summary>
        public static class SdkAdapter
        {
            public static readonly decimal MaxRiskPercentPerOrder = 0.01m;
            public static readonly int MaxOrderRetries = 3;
            public static readonly TimeSpan OrderTimeout = TimeSpan.FromSeconds(30);
            public static readonly string[] SupportedInstruments = { "ES", "MES", "NQ", "MNQ", "RTY", "YM" };
        }

        /// <summary>
        /// ML/RL specific risk defaults
        /// </summary>
        public static class MachineLearning
        {
            public static readonly decimal MinConfidenceThreshold = 0.6m;
            public static readonly decimal MaxModelRiskPercent = 0.005m; // 0.5%
            public static readonly int MinHistoricalBarsRequired = 100;
            public static readonly TimeSpan ModelUpdateInterval = TimeSpan.FromHours(24);
        }

        /// <summary>
        /// UCB specific risk defaults
        /// </summary>
        public static class Ucb
        {
            public static readonly decimal ExplorationBonus = 0.3m;
            public static readonly decimal ConfidenceThreshold = 0.65m;
            public static readonly int MinDecisionsBeforeLive = 100;
            public static readonly decimal MaxUcbRiskPercent = 0.008m; // 0.8%
        }

        /// <summary>
        /// Get risk configuration for a specific environment
        /// </summary>
        public static RiskConfiguration GetConfigurationForEnvironment(string environment)
        {
            return environment?.ToUpperInvariant() switch
            {
                "PRODUCTION" => new RiskConfiguration
                {
                    MaxDailyLoss = DefaultMaxDailyLoss,
                    MaxPositionSize = DefaultMaxPositionSize,
                    DrawdownLimit = DefaultDrawdownLimit,
                    MaxRiskPerTradePercent = DefaultMaxRiskPerTradePercent,
                    MaxDailyTrades = DefaultMaxDailyTrades,
                    IsProduction = true
                },
                "STAGING" => new RiskConfiguration
                {
                    MaxDailyLoss = DefaultMaxDailyLoss * 0.5m,
                    MaxPositionSize = DefaultMaxPositionSize * 0.5m,
                    DrawdownLimit = DefaultDrawdownLimit * 0.5m,
                    MaxRiskPerTradePercent = DefaultMaxRiskPerTradePercent * 0.5m,
                    MaxDailyTrades = DefaultMaxDailyTrades / 2,
                    IsProduction = false
                },
                "DEVELOPMENT" or "DEV" => new RiskConfiguration
                {
                    MaxDailyLoss = 100m,
                    MaxPositionSize = 1000m,
                    DrawdownLimit = 200m,
                    MaxRiskPerTradePercent = 0.001m, // 0.1%
                    MaxDailyTrades = 10,
                    IsProduction = false
                },
                _ => new RiskConfiguration
                {
                    MaxDailyLoss = DefaultMaxDailyLoss,
                    MaxPositionSize = DefaultMaxPositionSize,
                    DrawdownLimit = DefaultDrawdownLimit,
                    MaxRiskPerTradePercent = DefaultMaxRiskPerTradePercent,
                    MaxDailyTrades = DefaultMaxDailyTrades,
                    IsProduction = false
                }
            };
        }
    }

    /// <summary>
    /// Risk configuration data structure
    /// </summary>
    public class RiskConfiguration
    {
        public decimal MaxDailyLoss { get; set; }
        public decimal MaxPositionSize { get; set; }
        public decimal DrawdownLimit { get; set; }
        public decimal MaxRiskPerTradePercent { get; set; }
        public int MaxDailyTrades { get; set; }
        public bool IsProduction { get; set; }
    }
}
