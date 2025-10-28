using Microsoft.Extensions.Logging;
using BotCore.Market;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Zones;

namespace BotCore.Features;

/// <summary>
/// BACKTEST PARITY: Feature resolver that feeds bars to PatternEngine and publishes pattern features
/// Ensures historical and live bars both trigger pattern analysis
/// </summary>
public sealed class PatternBarFeatureResolver : IFeatureResolver
{
    private readonly ILogger<PatternBarFeatureResolver> _logger;
    private readonly BotCore.Patterns.PatternEngine _patternEngine;
    private readonly IFeatureBus _featureBus;
    private readonly ConcurrentDictionary<string, List<BotCore.Models.Bar>> _barBuffers = new();
    private readonly ConcurrentDictionary<string, double> _latestFeatures = new();
    private const int MaxBufferSize = 100;

    private static readonly string[] FeatureKeys = new[]
    {
        "pattern.bull_score",
        "pattern.bear_score",
        "pattern.overall_bias",
        "pattern.confidence",
        "pattern.reversal_signal",
        "pattern.active_count"
    };

    public PatternBarFeatureResolver(
        ILogger<PatternBarFeatureResolver> logger,
        BotCore.Patterns.PatternEngine patternEngine,
        IFeatureBus featureBus)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _patternEngine = patternEngine ?? throw new ArgumentNullException(nameof(patternEngine));
        _featureBus = featureBus ?? throw new ArgumentNullException(nameof(featureBus));
    }

    public Task OnBarAsync(string symbol, object barData, CancellationToken cancellationToken = default)
    {
        try
        {
            if (barData is not Bar bar)
            {
                _logger.LogWarning("[PATTERN-BAR-RESOLVER] Invalid bar data type for {Symbol}", symbol);
                return Task.CompletedTask;
            }

            // Convert Market.Bar to Models.Bar for pattern engine
            var modelBar = new BotCore.Models.Bar(
                bar.Open,
                bar.High,
                bar.Low,
                bar.Close,
                bar.Volume,
                new DateTimeOffset(bar.Start).ToUnixTimeMilliseconds());

            // Maintain bar buffer for this symbol
            var buffer = _barBuffers.GetOrAdd(symbol, _ => new List<BotCore.Models.Bar>());
            
            lock (buffer)
            {
                buffer.Add(modelBar);
                
                // Keep buffer size manageable
                if (buffer.Count > MaxBufferSize)
                {
                    buffer.RemoveAt(0);
                }
            }

            // Get pattern scores if we have enough bars
            if (buffer.Count >= 3)
            {
                BotCore.Models.PatternScores scores;
                lock (buffer)
                {
                    scores = _patternEngine.GetScores(symbol, buffer);
                }

                // Publish pattern features to feature bus
                var now = DateTime.UtcNow;
                
                _featureBus.Publish(symbol, now, "pattern.bull_score", (decimal)scores.BullScore);
                _featureBus.Publish(symbol, now, "pattern.bear_score", (decimal)scores.BearScore);
                _featureBus.Publish(symbol, now, "pattern.overall_bias", (decimal)scores.OverallBias);
                _featureBus.Publish(symbol, now, "pattern.confidence", (decimal)scores.Confidence);
                _featureBus.Publish(symbol, now, "pattern.reversal_signal", (decimal)scores.ReversalSignal);
                
                if (scores.ActivePatterns != null)
                {
                    _featureBus.Publish(symbol, now, "pattern.active_count", scores.ActivePatterns.Count);
                }

                // Cache latest features
                var key = symbol;
                _latestFeatures[$"{key}::pattern.bull_score"] = scores.BullScore;
                _latestFeatures[$"{key}::pattern.bear_score"] = scores.BearScore;
                _latestFeatures[$"{key}::pattern.overall_bias"] = scores.OverallBias;
                _latestFeatures[$"{key}::pattern.confidence"] = scores.Confidence;
                _latestFeatures[$"{key}::pattern.reversal_signal"] = scores.ReversalSignal;
                _latestFeatures[$"{key}::pattern.active_count"] = scores.ActivePatterns?.Count ?? 0;

                _logger.LogTrace("[PATTERN-BAR-RESOLVER] {Symbol}: Bull={Bull:F3}, Bear={Bear:F3}, Active={Count}",
                    symbol, scores.BullScore, scores.BearScore, scores.ActivePatterns?.Count ?? 0);
            }

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PATTERN-BAR-RESOLVER] Error processing bar for {Symbol}", symbol);
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

    public string[] GetAvailableFeatureKeys()
    {
        return FeatureKeys;
    }
}
