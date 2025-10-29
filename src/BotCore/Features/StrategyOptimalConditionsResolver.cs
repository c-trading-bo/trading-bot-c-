using Microsoft.Extensions.Logging;
using BotCore.Market;
using Zones;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Features;

/// <summary>
/// BACKTEST PARITY: Strategy optimal conditions tracker
/// Tracks when each strategy (S2, S3, S6, S11) performs best
/// Provides condition-based strategy selection recommendations
/// </summary>
public sealed class StrategyOptimalConditionsResolver : IFeatureResolver
{
    private readonly ILogger<StrategyOptimalConditionsResolver> _logger;
    private readonly IFeatureBus _featureBus;
    private readonly ConcurrentDictionary<string, ConditionTracker> _trackers = new();
    private readonly ConcurrentDictionary<string, double> _latestFeatures = new();

    private static readonly string[] FeatureKeys = new[]
    {
        "strategy.s2_optimal",     // Is S2 optimal for current conditions
        "strategy.s3_optimal",     // Is S3 optimal for current conditions
        "strategy.s6_optimal",     // Is S6 optimal for current conditions
        "strategy.s11_optimal",    // Is S11 optimal for current conditions
        "strategy.best_strategy",  // Best strategy ID for current conditions
        "strategy.condition_score" // Confidence score for strategy selection
    };

    public StrategyOptimalConditionsResolver(
        ILogger<StrategyOptimalConditionsResolver> logger,
        IFeatureBus featureBus)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _featureBus = featureBus ?? throw new ArgumentNullException(nameof(featureBus));
    }

    public Task OnBarAsync(string symbol, object barData, CancellationToken cancellationToken = default)
    {
        try
        {
            if (barData is not Bar bar)
            {
                _logger.LogWarning("[STRATEGY-CONDITIONS-RESOLVER] Invalid bar data type for {Symbol}", symbol);
                return Task.CompletedTask;
            }

            var tracker = _trackers.GetOrAdd(symbol, _ => new ConditionTracker());
            var conditions = DetermineMarketConditions(bar);
            
            var now = DateTime.UtcNow;

            // Determine which strategies are optimal for current conditions
            var s2Optimal = IsS2Optimal(conditions);
            var s3Optimal = IsS3Optimal(conditions);
            var s6Optimal = IsS6Optimal(conditions);
            var s11Optimal = IsS11Optimal(conditions);

            // Determine best strategy
            var bestStrategy = DetermineBestStrategy(conditions);
            var conditionScore = CalculateConditionScore(conditions);

            // Publish features
            _featureBus.Publish(symbol, now, "strategy.s2_optimal", s2Optimal ? 1.0 : 0.0);
            _featureBus.Publish(symbol, now, "strategy.s3_optimal", s3Optimal ? 1.0 : 0.0);
            _featureBus.Publish(symbol, now, "strategy.s6_optimal", s6Optimal ? 1.0 : 0.0);
            _featureBus.Publish(symbol, now, "strategy.s11_optimal", s11Optimal ? 1.0 : 0.0);
            _featureBus.Publish(symbol, now, "strategy.best_strategy", (double)bestStrategy);
            _featureBus.Publish(symbol, now, "strategy.condition_score", conditionScore);

            // Cache latest features
            var key = symbol;
            _latestFeatures[$"{key}::strategy.s2_optimal"] = s2Optimal ? 1.0 : 0.0;
            _latestFeatures[$"{key}::strategy.s3_optimal"] = s3Optimal ? 1.0 : 0.0;
            _latestFeatures[$"{key}::strategy.s6_optimal"] = s6Optimal ? 1.0 : 0.0;
            _latestFeatures[$"{key}::strategy.s11_optimal"] = s11Optimal ? 1.0 : 0.0;
            _latestFeatures[$"{key}::strategy.best_strategy"] = (double)bestStrategy;
            _latestFeatures[$"{key}::strategy.condition_score"] = conditionScore;

            _logger.LogTrace("[STRATEGY-CONDITIONS-RESOLVER] {Symbol}: Best={Best}, Score={Score:F2}",
                symbol, GetStrategyName(bestStrategy), conditionScore);

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[STRATEGY-CONDITIONS-RESOLVER] Error processing bar for {Symbol}", symbol);
            throw;
        }
    }

    public Task<double?> TryGetAsync(string symbol, string featureKey, CancellationToken cancellationToken = default)
    {
        var key = $"{symbol}::{featureKey}";
        if (_latestFeatures.TryGetValue(key, out var value))
        {
            return Task.FromResult<double?>(value);
        }
        return Task.FromResult<double?>(null);
    }

    public string[] GetAvailableFeatureKeys() => FeatureKeys;

    private MarketConditions DetermineMarketConditions(Bar bar)
    {
        var hour = bar.Start.Hour;
        var range = (double)(bar.High - bar.Low);
        var midPrice = (double)((bar.High + bar.Low) / 2.0m);
        
        return new MarketConditions
        {
            IsOpeningDrive = hour >= 9 && hour < 10,  // 9:00-10:00 AM
            IsAfternoonFade = hour >= 13 && hour < 16, // 1:00-4:00 PM
            IsOvernight = hour >= 18 || hour < 9,      // 6:00 PM - 9:00 AM
            IsHighVolume = bar.Volume > 10000,
            IsLowVolatility = range / Math.Max(0.01, midPrice) < 0.005,
            IsHighVolatility = range / Math.Max(0.01, midPrice) > 0.02
        };
    }

    private bool IsS2Optimal(MarketConditions conditions)
    {
        // S2 (VWAP Mean Reversion) works best in:
        // - High volume periods
        // - Ranging/low volatility
        // - Regular trading hours
        return conditions.IsHighVolume && 
               !conditions.IsOpeningDrive && 
               !conditions.IsOvernight;
    }

    private bool IsS3Optimal(MarketConditions conditions)
    {
        // S3 (Bollinger Compression) works best in:
        // - Low volatility periods
        // - Pre-market/overnight (compression builds)
        return conditions.IsLowVolatility || conditions.IsOvernight;
    }

    private bool IsS6Optimal(MarketConditions conditions)
    {
        // S6 (Momentum) works best in:
        // - Opening drive (high momentum)
        // - High volatility
        return conditions.IsOpeningDrive || conditions.IsHighVolatility;
    }

    private bool IsS11Optimal(MarketConditions conditions)
    {
        // S11 (ADR/IB Fade) works best in:
        // - Afternoon session (exhaustion after morning range)
        // - After high volatility periods
        return conditions.IsAfternoonFade;
    }

    private Strategy DetermineBestStrategy(MarketConditions conditions)
    {
        // Priority order based on conditions
        if (conditions.IsOpeningDrive)
            return Strategy.S6;
        
        if (conditions.IsAfternoonFade)
            return Strategy.S11;
        
        if (conditions.IsLowVolatility || conditions.IsOvernight)
            return Strategy.S3;
        
        return Strategy.S2; // Default to VWAP mean reversion
    }

    private double CalculateConditionScore(MarketConditions conditions)
    {
        // Calculate confidence score for strategy selection
        var score = 0.5; // Neutral baseline
        
        if (conditions.IsHighVolume) score += 0.1;
        if (conditions.IsOpeningDrive) score += 0.2;
        if (conditions.IsAfternoonFade) score += 0.15;
        if (conditions.IsLowVolatility) score += 0.1;
        
        return Math.Min(1.0, score);
    }

    private string GetStrategyName(Strategy strategy)
    {
        return strategy switch
        {
            Strategy.S2 => "S2-VWAP",
            Strategy.S3 => "S3-Compression",
            Strategy.S6 => "S6-Momentum",
            Strategy.S11 => "S11-Fade",
            _ => "Unknown"
        };
    }

    private enum Strategy
    {
        S2 = 2,
        S3 = 3,
        S6 = 6,
        S11 = 11
    }

    private class MarketConditions
    {
        public bool IsOpeningDrive { get; set; }
        public bool IsAfternoonFade { get; set; }
        public bool IsOvernight { get; set; }
        public bool IsHighVolume { get; set; }
        public bool IsLowVolatility { get; set; }
        public bool IsHighVolatility { get; set; }
    }

    private class ConditionTracker
    {
        // Future: Track historical performance per condition
        // For now, ready for future expansion to track condition-based performance
    }
}
