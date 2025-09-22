using Microsoft.Extensions.Options;
using BotCore.Configuration;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Service to replace hardcoded ML/AI values with configuration-driven ones
    /// Addresses Comment #3304685224: Hardcoded Values in AI/ML Core Systems
    /// </summary>
    public class MLConfigurationService
    {
        private readonly TradingConfiguration _config;

        /// <summary>
        /// Default rollback variance multiplier constant for use in configuration classes
        /// </summary>
        public const double DefaultRollbackVarMultiplier = MLConfigurationConstants.DefaultRollbackVarMultiplier;

        public MLConfigurationService(IOptions<TradingConfiguration> config)
        {
            _config = config.Value;
        }

        /// <summary>
        /// Get AI confidence threshold - replaces hardcoded 0.7
        /// </summary>
        public double GetAIConfidenceThreshold() => _config.AIConfidenceThreshold;

        /// <summary>
        /// Get minimum confidence for model fallback - replaces hardcoded 0.1
        /// </summary>
        public double GetMinimumConfidence() => _config.MinimumConfidence ?? 0.1;

        /// <summary>
        /// Get position size multiplier - replaces hardcoded 2.5
        /// </summary>
        public double GetPositionSizeMultiplier() => _config.DefaultPositionSizeMultiplier;

        /// <summary>
        /// Get regime detection threshold - replaces hardcoded 1.0
        /// </summary>
        public double GetRegimeDetectionThreshold() => _config.RegimeDetectionThreshold;

        /// <summary>
        /// Get stop loss buffer percentage - replaces hardcoded 0.05
        /// </summary>
        public double GetStopLossBufferPercentage() => _config.StopLossBufferPercentage;

        /// <summary>
        /// Get reward to risk ratio threshold - replaces hardcoded 1.2
        /// </summary>
        public double GetRewardRiskRatioThreshold() => _config.RewardRiskRatioThreshold;

        /// <summary>
        /// Calculate dynamic position size based on configuration and market context
        /// </summary>
        public double CalculatePositionSize(double volatility, double confidence, double riskLevel)
        {
            var baseSize = _config.DefaultPositionSizeMultiplier;
            
            // Adjust based on confidence (higher confidence = larger position, but capped)
            var confidenceAdjustment = Math.Min(confidence / _config.AIConfidenceThreshold, 1.5);
            
            // Adjust based on volatility (higher volatility = smaller position)
            var volatilityAdjustment = Math.Max(0.5, 1.0 - volatility);
            
            // Adjust based on risk level
            var riskAdjustment = Math.Max(0.3, 1.0 - riskLevel);
            
            return baseSize * confidenceAdjustment * volatilityAdjustment * riskAdjustment;
        }

        /// <summary>
        /// Evaluate if AI confidence meets threshold for execution
        /// </summary>
        public bool IsConfidenceAcceptable(double confidence) => confidence >= _config.AIConfidenceThreshold;

        /// <summary>
        /// Evaluate if regime detection is confident enough
        /// </summary>
        public bool IsRegimeDetectionReliable(double regimeConfidence) => regimeConfidence >= _config.RegimeDetectionThreshold;

        /// <summary>
        /// Calculate dynamic stop loss based on ATR and configuration
        /// </summary>
        public double CalculateStopLoss(double entryPrice, double atr, bool isLong)
        {
            var buffer = atr * _config.StopLossBufferPercentage;
            return isLong ? entryPrice - buffer : entryPrice + buffer;
        }

        /// <summary>
        /// Validate reward to risk ratio meets configuration threshold
        /// </summary>
        public bool IsRewardRiskRatioAcceptable(double reward, double risk)
        {
            if (risk <= 0) return false;
            var ratio = reward / risk;
            return ratio >= _config.RewardRiskRatioThreshold;
        }
    }
}