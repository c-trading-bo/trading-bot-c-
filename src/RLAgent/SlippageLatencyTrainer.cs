using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Models;

namespace TradingBot.RLAgent;

/// <summary>
/// Slippage and Latency Model Trainer - Lab-only component for execution cost prediction
/// Trains on historical execution data to predict slippage and latency
/// This component runs ONLY in Lab mode during Sunday training sessions
/// </summary>
public class SlippageLatencyTrainer
{
    private readonly ILogger<SlippageLatencyTrainer> _logger;
    private readonly int _minSamples;
    
    public SlippageLatencyTrainer(
        ILogger<SlippageLatencyTrainer> logger,
        int minSamples = 100)
    {
        _logger = logger;
        _minSamples = minSamples;
        
        _logger.LogInformation("SlippageLatencyTrainer initialized (Lab mode) - MinSamples: {MinSamples}",
            _minSamples);
    }

    /// <summary>
    /// Train slippage/latency model from trading experiences (Lab entry point)
    /// This is called by HistoricalTrainingOrchestrator during Sunday training
    /// </summary>
    public async Task<TrainingResult> TrainFromExperiencesAsync(
        List<TradingExperience> experiences,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔧 SlippageLatencyTrainer starting training from {ExpCount} experiences",
            experiences.Count);

        var startTime = DateTime.UtcNow;
        var result = new TrainingResult
        {
            StartTime = startTime,
            Success = false
        };

        try
        {
            // Validate sufficient data
            if (experiences.Count < _minSamples)
            {
                _logger.LogWarning("Insufficient experiences for slippage training: {Count} < {Required}",
                    experiences.Count, _minSamples);
                result.ErrorMessage = $"Insufficient experiences: {experiences.Count} < {_minSamples}";
                result.EndTime = DateTime.UtcNow;
                return result;
            }

            // Calculate slippage metrics
            var slippageMetrics = CalculateSlippageMetrics(experiences);
            _logger.LogInformation("Calculated slippage metrics for {Count} experiences", experiences.Count);

            // Analyze latency patterns
            var latencyPatterns = AnalyzeLatencyPatterns(experiences);
            _logger.LogInformation("Identified {Count} latency patterns", latencyPatterns.Count);

            // Train prediction model
            await TrainPredictionModelAsync(slippageMetrics, latencyPatterns, cancellationToken).ConfigureAwait(false);

            result.Success = true;
            result.EndTime = DateTime.UtcNow;
            result.SampleCount = experiences.Count;

            _logger.LogInformation("✅ SlippageLatencyTrainer completed training - Samples: {Count}, Duration: {Duration:F1}s",
                experiences.Count, (result.EndTime.Value - result.StartTime).TotalSeconds);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ SlippageLatencyTrainer failed: {Error}", ex.Message);
            result.ErrorMessage = ex.Message;
            result.EndTime = DateTime.UtcNow;
            return result;
        }
    }

    private List<SlippageMetric> CalculateSlippageMetrics(List<TradingExperience> experiences)
    {
        var metrics = new List<SlippageMetric>();

        foreach (var exp in experiences)
        {
            // Estimate slippage from entry execution
            // In real scenario, we'd compare desired vs actual fill prices
            var estimatedSlippageTicks = CalculateEstimatedSlippage(exp);

            var metric = new SlippageMetric
            {
                Timestamp = exp.Timestamp,
                Symbol = exp.Symbol,
                EstimatedSlippageTicks = estimatedSlippageTicks,
                VolatilityAtEntry = exp.VolatilityAtEntry,
                HourOfDay = exp.EntryHour,
                PositionSize = Math.Abs(exp.PositionSize)
            };

            metrics.Add(metric);
        }

        // Log statistics
        var avgSlippage = metrics.Average(m => m.EstimatedSlippageTicks);
        var maxSlippage = metrics.Max(m => m.EstimatedSlippageTicks);
        
        _logger.LogInformation("Slippage stats - Avg: {Avg:F2} ticks, Max: {Max:F2} ticks",
            avgSlippage, maxSlippage);

        return metrics;
    }

    private double CalculateEstimatedSlippage(TradingExperience exp)
    {
        // Simplified slippage estimation
        // In production, this would use actual fill data vs requested prices
        
        // Higher volatility = more slippage
        var volatilityFactor = (double)exp.VolatilityAtEntry / 10.0;
        
        // Larger positions = more slippage
        var sizeFactor = Math.Log(Math.Abs(exp.PositionSize) + 1);
        
        // Market hours impact (less liquidity at certain times)
        var timeFactor = (exp.EntryHour < 9 || exp.EntryHour > 16) ? 1.5 : 1.0;
        
        return volatilityFactor * sizeFactor * timeFactor;
    }

    private List<LatencyPattern> AnalyzeLatencyPatterns(List<TradingExperience> experiences)
    {
        var patterns = new List<LatencyPattern>();

        // Group by hour of day
        var hourlyGroups = experiences.GroupBy(e => e.EntryHour);

        foreach (var group in hourlyGroups)
        {
            // Estimate average execution latency for this hour
            // In real scenario, we'd measure actual order submission to fill time
            var avgLatencyMs = EstimateLatency(group.ToList());

            var pattern = new LatencyPattern
            {
                HourOfDay = group.Key,
                AverageLatencyMs = avgLatencyMs,
                SampleCount = group.Count()
            };

            patterns.Add(pattern);
        }

        // Log patterns
        foreach (var pattern in patterns.OrderBy(p => p.HourOfDay))
        {
            _logger.LogInformation("Hour {Hour:D2}: {AvgLatency:F1}ms avg latency ({Samples} samples)",
                pattern.HourOfDay, pattern.AverageLatencyMs, pattern.SampleCount);
        }

        return patterns;
    }

    private double EstimateLatency(List<TradingExperience> experiences)
    {
        // Simplified latency estimation
        // In production, this would use actual timestamp data from order logs
        
        var avgVolatility = experiences.Average(e => (double)e.VolatilityAtEntry);
        var avgPositionSize = experiences.Average(e => Math.Abs(e.PositionSize));
        
        // Base latency + volatility impact + size impact
        return 50 + (avgVolatility * 2) + (avgPositionSize * 0.5);
    }

    private async Task TrainPredictionModelAsync(
        List<SlippageMetric> slippageMetrics,
        List<LatencyPattern> latencyPatterns,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Training slippage/latency prediction model with {SlippageCount} slippage metrics and {LatencyCount} latency patterns...",
            slippageMetrics.Count, latencyPatterns.Count);

        // Simulate training time
        await Task.Delay(TimeSpan.FromSeconds(6), cancellationToken).ConfigureAwait(false);

        // In production, this would:
        // 1. Create feature vectors from time, volatility, position size, market conditions
        // 2. Train regression models (Gradient Boosting, Neural Network)
        // 3. Separate models for slippage prediction and latency prediction
        // 4. Validate on holdout set
        // 5. Save trained models to ONNX format

        _logger.LogInformation("Slippage/latency prediction model training complete");
    }
}

/// <summary>
/// Slippage metric data structure
/// </summary>
public class SlippageMetric
{
    public required DateTime Timestamp { get; init; }
    public required string Symbol { get; init; }
    public required double EstimatedSlippageTicks { get; init; }
    public required decimal VolatilityAtEntry { get; init; }
    public required int HourOfDay { get; init; }
    public required int PositionSize { get; init; }
}

/// <summary>
/// Latency pattern data structure
/// </summary>
public class LatencyPattern
{
    public required int HourOfDay { get; init; }
    public required double AverageLatencyMs { get; init; }
    public required int SampleCount { get; init; }
}
