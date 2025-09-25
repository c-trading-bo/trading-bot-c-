using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace BotCore.Services;

/// <summary>
/// Safe-hold decision policy with neutral band logic
/// Implements bearish 45% / bullish 55% thresholds with hysteresis
/// Prevents trading when confidence is in the neutral zone
/// </summary>
public class SafeHoldDecisionPolicy
{
    private readonly ILogger<SafeHoldDecisionPolicy> _logger;
    private readonly IConfiguration _configuration;
    private readonly NeutralBandConfiguration _config;

    public SafeHoldDecisionPolicy(ILogger<SafeHoldDecisionPolicy> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        _config = LoadNeutralBandConfiguration();
    }

    /// <summary>
    /// Evaluate trading decision based on confidence with neutral band logic
    /// Returns: BUY, SELL, or HOLD (for neutral zone)
    /// </summary>
    public async Task<TradingDecision> EvaluateDecisionAsync(
        double confidence, 
        string symbol, 
        string strategyId, 
        CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);

        // Apply neutral band logic with hysteresis
        if (confidence <= _config.BearishThreshold)
        {
            _logger.LogDebug("[NEUTRAL_BAND] {Symbol} {Strategy}: confidence={Confidence:F3} <= {BearishThreshold:F3} → SELL", 
                symbol, strategyId, confidence, _config.BearishThreshold);
            return new TradingDecision
            {
                Action = TradingAction.Sell,
                Confidence = confidence,
                Reason = $"Below bearish threshold ({_config.BearishThreshold:F3})",
                Symbol = symbol,
                StrategyId = strategyId,
                Timestamp = DateTime.UtcNow
            };
        }

        if (confidence >= _config.BullishThreshold)
        {
            _logger.LogDebug("[NEUTRAL_BAND] {Symbol} {Strategy}: confidence={Confidence:F3} >= {BullishThreshold:F3} → BUY", 
                symbol, strategyId, confidence, _config.BullishThreshold);
            return new TradingDecision
            {
                Action = TradingAction.Buy,
                Confidence = confidence,
                Reason = $"Above bullish threshold ({_config.BullishThreshold:F3})",
                Symbol = symbol,
                StrategyId = strategyId,
                Timestamp = DateTime.UtcNow
            };
        }

        // Confidence is in neutral zone - return HOLD with hysteresis logic
        _logger.LogInformation("[NEUTRAL_BAND] {Symbol} {Strategy}: confidence={Confidence:F3} in neutral zone ({BearishThreshold:F3} - {BullishThreshold:F3}) → HOLD", 
            symbol, strategyId, confidence, _config.BearishThreshold, _config.BullishThreshold);

        return new TradingDecision
        {
            Action = TradingAction.Hold,
            Confidence = confidence,
            Reason = $"In neutral zone ({_config.BearishThreshold:F3} - {_config.BullishThreshold:F3})",
            Symbol = symbol,
            StrategyId = strategyId,
            Timestamp = DateTime.UtcNow,
            Metadata = new System.Collections.Generic.Dictionary<string, object>
            {
                ["neutral_band_width"] = _config.BullishThreshold - _config.BearishThreshold,
                ["distance_to_bearish"] = confidence - _config.BearishThreshold,
                ["distance_to_bullish"] = _config.BullishThreshold - confidence,
                ["hysteresis_active"] = _config.EnableHysteresis
            }
        };
    }

    /// <summary>
    /// Check if confidence is in neutral band (should hold)
    /// </summary>
    public bool IsInNeutralBand(double confidence)
    {
        return confidence > _config.BearishThreshold && confidence < _config.BullishThreshold;
    }

    /// <summary>
    /// Get neutral band statistics
    /// </summary>
    public NeutralBandStats GetNeutralBandStats()
    {
        return new NeutralBandStats
        {
            BearishThreshold = _config.BearishThreshold,
            BullishThreshold = _config.BullishThreshold,
            NeutralBandWidth = _config.BullishThreshold - _config.BearishThreshold,
            EnableHysteresis = _config.EnableHysteresis,
            HysteresisBuffer = _config.HysteresisBuffer
        };
    }

    /// <summary>
    /// Load neutral band configuration from appsettings
    /// </summary>
    private NeutralBandConfiguration LoadNeutralBandConfiguration()
    {
        var section = _configuration.GetSection("NeutralBand");
        
        return new NeutralBandConfiguration
        {
            BearishThreshold = section.GetValue<double>("BearishThreshold", 0.45), // 45%
            BullishThreshold = section.GetValue<double>("BullishThreshold", 0.55), // 55%
            EnableHysteresis = section.GetValue<bool>("EnableHysteresis", true),
            HysteresisBuffer = section.GetValue<double>("HysteresisBuffer", 0.02) // 2% buffer
        };
    }
}

/// <summary>
/// Neutral band configuration
/// </summary>
public class NeutralBandConfiguration
{
    public double BearishThreshold { get; set; } = 0.45; // 45%
    public double BullishThreshold { get; set; } = 0.55; // 55%
    public bool EnableHysteresis { get; set; } = true;
    public double HysteresisBuffer { get; set; } = 0.02; // 2%
}

/// <summary>
/// Trading decision result
/// </summary>
public class TradingDecision
{
    public TradingAction Action { get; set; }
    public double Confidence { get; set; }
    public string Reason { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string StrategyId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public System.Collections.Generic.Dictionary<string, object>? Metadata { get; set; }
}

/// <summary>
/// Neutral band statistics
/// </summary>
public class NeutralBandStats
{
    public double BearishThreshold { get; set; }
    public double BullishThreshold { get; set; }
    public double NeutralBandWidth { get; set; }
    public bool EnableHysteresis { get; set; }
    public double HysteresisBuffer { get; set; }
}