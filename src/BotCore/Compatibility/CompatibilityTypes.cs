using System;
using System.Collections.Generic;

namespace BotCore.Compatibility;

/// <summary>
/// Enhanced trading decision with parameter bundle tracking
/// </summary>
public record EnhancedTradingDecision
{
    public TradingDecision OriginalDecision { get; init; } = new();
    public ParameterBundle ParameterBundle { get; init; } = new();
    public ConfigurationSource ConfigurationSource { get; init; } = new();
    public DateTime TimestampUtc { get; init; } = DateTime.UtcNow;
    public string DecisionPath { get; init; } = string.Empty;
    public Dictionary<string, object> ActivelyUsedParameters { get; init; } = new();
}

/// <summary>
/// Market context with enhanced parameter support
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

/// <summary>
/// Configuration source from StructuredConfigurationManager
/// </summary>
public class ConfigurationSource
{
    public string Strategy { get; set; } = string.Empty;
    public Dictionary<string, object> BaseParameters { get; set; } = new();
    public Dictionary<string, object> RiskParameters { get; set; } = new();
    public Dictionary<string, Dictionary<string, object>> MarketConditionOverrides { get; set; } = new();
    public string LoadedFrom { get; set; } = string.Empty;
    public DateTime LastUpdated { get; set; }
}

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