using System;

namespace TradingBot.Abstractions;

/// <summary>
/// Interface for ML/AI configuration service to avoid circular dependencies
/// Provides access to configuration-driven ML/AI values instead of hardcoded constants
/// </summary>
public interface IMLConfigurationService
{
    /// <summary>
    /// Get AI confidence threshold for trade execution
    /// </summary>
    double GetAIConfidenceThreshold();

    /// <summary>
    /// Get minimum confidence for model fallback
    /// </summary>
    double GetMinimumConfidence();

    /// <summary>
    /// Get position size multiplier for dynamic calculation
    /// </summary>
    double GetPositionSizeMultiplier();

    /// <summary>
    /// Get regime detection confidence threshold
    /// </summary>
    double GetRegimeDetectionThreshold();

    /// <summary>
    /// Get stop loss buffer percentage
    /// </summary>
    double GetStopLossBufferPercentage();

    /// <summary>
    /// Get reward to risk ratio threshold
    /// </summary>
    double GetRewardRiskRatioThreshold();

    /// <summary>
    /// Calculate dynamic position size based on configuration and market context
    /// </summary>
    double CalculatePositionSize(double volatility, double confidence, double riskLevel);

    /// <summary>
    /// Evaluate if AI confidence meets threshold for execution
    /// </summary>
    bool IsConfidenceAcceptable(double confidence);

    /// <summary>
    /// Evaluate if regime detection is confident enough
    /// </summary>
    bool IsRegimeDetectionReliable(double regimeConfidence);

    /// <summary>
    /// Calculate dynamic stop loss based on ATR and configuration
    /// </summary>
    double CalculateStopLoss(double entryPrice, double atr, bool isLong);

    /// <summary>
    /// Validate reward to risk ratio meets configuration threshold
    /// </summary>
    bool IsRewardRiskRatioAcceptable(double reward, double risk);
}