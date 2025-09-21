using System.ComponentModel.DataAnnotations;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Configuration options for DecisionPolicy system.
/// Manages thresholds for smart decision making with neutral band and hysteresis.
/// </summary>
public class DecisionPolicyOptions
{
    /// <summary>
    /// Confidence threshold above which we'll consider BUY decisions
    /// </summary>
    [Range(0.5, 1.0)]
    public decimal BullThreshold { get; set; } = 0.55m;

    /// <summary>
    /// Confidence threshold below which we'll consider SELL decisions
    /// </summary>
    [Range(0.0, 0.5)]
    public decimal BearThreshold { get; set; } = 0.45m;

    /// <summary>
    /// Hysteresis buffer to prevent oscillation between decisions
    /// </summary>
    [Range(0.005, 0.05)]
    public decimal HysteresisBuffer { get; set; } = 0.01m;

    /// <summary>
    /// Maximum number of decisions per minute to prevent over-trading
    /// </summary>
    [Range(1, 60)]
    public int MaxDecisionsPerMinute { get; set; } = 5;

    /// <summary>
    /// Minimum time between decisions in seconds
    /// </summary>
    [Range(5, 300)]
    public int MinTimeBetweenDecisionsSeconds { get; set; } = 30;

    /// <summary>
    /// Enable position-based decision modifications
    /// </summary>
    public bool EnablePositionBias { get; set; } = true;

    /// <summary>
    /// Maximum position size that affects decision thresholds
    /// </summary>
    [Range(1, 20)]
    public int MaxPositionForBias { get; set; } = 5;

    /// <summary>
    /// Validate configuration values
    /// </summary>
    public void Validate()
    {
        if (BullThreshold <= BearThreshold)
            throw new InvalidOperationException($"BullThreshold ({BullThreshold}) must be greater than BearThreshold ({BearThreshold})");

        if (BullThreshold - BearThreshold < HysteresisBuffer * 2)
            throw new InvalidOperationException($"Threshold spread ({BullThreshold - BearThreshold}) must be at least 2x HysteresisBuffer ({HysteresisBuffer * 2})");
    }
}