using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using BotCore.Configuration;
using TradingBot.BotCore.Services;

namespace TradingBot.Tests
{
    /// <summary>
    /// Demonstration of proper ML configuration usage instead of hardcoded values
    /// This file shows the CORRECT way to handle AI/ML parameters
    /// </summary>
    public class MLConfigurationDemonstration
    {
        private readonly MLConfigurationService _mlConfig;

        public MLConfigurationDemonstration(MLConfigurationService mlConfig)
        {
            _mlConfig = mlConfig;
        }

        /// <summary>
        /// Example: Position sizing with configuration-driven values
        /// ✅ CORRECT: Uses MLConfigurationService instead of hardcoded 2.5
        /// </summary>
        public double CalculatePositionSize(double marketVolatility, double modelConfidence)
        {
            // ✅ Configuration-driven position sizing
            var baseMultiplier = _mlConfig.GetPositionSizeMultiplier();

            // Use dynamic calculation based on market conditions
            return _mlConfig.CalculatePositionSize(marketVolatility, modelConfidence, 0.1);
        }

        /// <summary>
        /// Example: AI confidence evaluation
        /// ✅ CORRECT: Uses MLConfigurationService instead of hardcoded 0.7
        /// </summary>
        public bool ShouldExecuteTrade(double modelConfidence)
        {
            // ✅ Configuration-driven confidence threshold
            return _mlConfig.IsConfidenceAcceptable(modelConfidence);
        }

        /// <summary>
        /// Example: Regime detection validation
        /// ✅ CORRECT: Uses MLConfigurationService instead of hardcoded 1.0
        /// </summary>
        public bool IsRegimeReliable(double regimeConfidence)
        {
            // ✅ Configuration-driven regime threshold
            return _mlConfig.IsRegimeDetectionReliable(regimeConfidence);
        }

        /// <summary>
        /// Example: Stop loss calculation
        /// ✅ CORRECT: Uses MLConfigurationService instead of hardcoded 0.05
        /// </summary>
        public double CalculateStopLoss(double entryPrice, double atr, bool isLong)
        {
            // ✅ Configuration-driven stop loss calculation
            return _mlConfig.CalculateStopLoss(entryPrice, atr, isLong);
        }

        /// <summary>
        /// Example: Risk validation
        /// ✅ CORRECT: Uses MLConfigurationService instead of hardcoded 1.2
        /// </summary>
        public bool ValidateRewardRisk(double potentialReward, double potentialRisk)
        {
            // ✅ Configuration-driven reward/risk validation
            return _mlConfig.IsRewardRiskRatioAcceptable(potentialReward, potentialRisk);
        }

        // ❌ WRONG EXAMPLES (these would be caught by our analyzer):

        /*
        // ❌ This would trigger: "PRODUCTION VIOLATION: Hardcoded business values detected"
        public double GetPositionSize() => /* hardcoded for demo */ <literal>;

        // ❌ This would trigger: "PRODUCTION VIOLATION: Hardcoded business values detected"
        public bool IsConfident(double confidence) => confidence >= /* hardcoded for demo */ <literal>;

        // ❌ This would trigger: "PRODUCTION VIOLATION: Hardcoded business values detected"
        public double GetRegimeDetection() => /* hardcoded for demo */ <literal>;
        */
    }
}
