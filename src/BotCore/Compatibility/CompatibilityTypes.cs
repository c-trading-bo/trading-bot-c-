using System;
using System.Collections.Generic;

namespace BotCore.Compatibility;

// Note: EnhancedTradingDecision is defined in Services/EnhancedTradingBrainIntegration.cs
// Note: MarketContext is defined in Brain/UnifiedTradingBrain.cs

/// <summary>
/// Configuration source tracking for parameter bundles
/// </summary>
public record MarketContext
{
    public string Symbol { get; init; } = string.Empty;
    public decimal Price { get; init; }
    public decimal Volume { get; init; }
    public string MarketCondition { get; init; } = "Normal"; // Volatile, Trending, Ranging
    public decimal MaxPositionMultiplier { get; init; } = 1.0m;
    public decimal ConfidenceThreshold { get; init; } = 0.65m;
    public ConfigurationSource? ConfiguredParameters { get; init; }
    public DateTime TimestampUtc { get; init; } = DateTime.UtcNow;
}

// Note: ConfigurationSource is defined in StructuredConfigurationManager.cs

/// <summary>
/// Trading decision for compatibility kit
/// </summary>
public record TradingDecision
{
    public string Symbol { get; init; } = string.Empty;
    public TradingAction Action { get; init; } = TradingAction.Hold;
    public decimal Quantity { get; init; }
    public decimal Confidence { get; init; }
    public string Reasoning { get; init; } = string.Empty;
    public DateTime TimestampUtc { get; init; } = DateTime.UtcNow;
}

/// <summary>
/// Trading actions
/// </summary>
public enum TradingAction
{
    Hold,
    Buy,
    Sell
}

/// <summary>
/// Basic neural network for compatibility
/// </summary>
public class BasicNeuralNetwork
{
    public int InputDimension { get; }
    public int HiddenDimension { get; }
    
    public BasicNeuralNetwork(int inputDimension, int hiddenDimension)
    {
        InputDimension = inputDimension;
        HiddenDimension = hiddenDimension;
    }
    
    public double[] Predict(double[] input)
    {
        // Simple prediction for compatibility
        return new double[4]; // 4 strategies
    }
}