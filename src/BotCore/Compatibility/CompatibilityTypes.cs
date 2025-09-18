using System;

namespace BotCore.Compatibility;

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