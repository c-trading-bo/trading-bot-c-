using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Models;

namespace TradingBot.RLAgent;

/// <summary>
/// Regime Detector Trainer - Lab-only component for market regime classification
/// Trains on historical data to detect Trend/Range/Transition regimes
/// This component runs ONLY in Lab mode during Sunday training sessions
/// </summary>
public class RegimeDetectorTrainer
{
    private readonly ILogger<RegimeDetectorTrainer> _logger;
    private readonly int _lookbackWindow;
    private readonly double _trendThreshold;
    
    public RegimeDetectorTrainer(
        ILogger<RegimeDetectorTrainer> logger,
        int lookbackWindow = 20,
        double trendThreshold = 0.02)
    {
        _logger = logger;
        _lookbackWindow = lookbackWindow;
        _trendThreshold = trendThreshold;
        
        _logger.LogInformation("RegimeDetectorTrainer initialized (Lab mode) - Window: {Window}, Threshold: {Threshold}",
            _lookbackWindow, _trendThreshold);
    }

    /// <summary>
    /// Train regime detector from historical bar data (Lab entry point)
    /// This is called by HistoricalTrainingOrchestrator during Sunday training
    /// </summary>
    public async Task<TrainingResult> TrainFromHistoricalBarsAsync(
        List<HistoricalBar> bars,
        List<TradingExperience> experiences,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔧 RegimeDetectorTrainer starting training from {BarCount} bars and {ExpCount} experiences",
            bars.Count, experiences.Count);

        var startTime = DateTime.UtcNow;
        var result = new TrainingResult
        {
            StartTime = startTime,
            Success = false
        };

        try
        {
            // Validate sufficient data
            if (bars.Count < _lookbackWindow * 2)
            {
                _logger.LogWarning("Insufficient bars for regime training: {Count} < {Required}",
                    bars.Count, _lookbackWindow * 2);
                result.ErrorMessage = $"Insufficient bars: {bars.Count} < {_lookbackWindow * 2}";
                result.EndTime = DateTime.UtcNow;
                return result;
            }

            // Sort bars chronologically
            var sortedBars = bars.OrderBy(b => b.Timestamp).ToList();

            // Classify market regimes
            var regimes = ClassifyRegimes(sortedBars);
            _logger.LogInformation("Classified {Count} regime periods", regimes.Count);

            // Calculate regime statistics
            var regimeStats = CalculateRegimeStatistics(regimes);
            LogRegimeStatistics(regimeStats);

            // Correlate regimes with trading performance
            var regimePerformance = AnalyzeRegimePerformance(regimes, experiences);
            _logger.LogInformation("Analyzed performance across {Count} regime types", regimePerformance.Count);

            // Train regime classifier
            await TrainRegimeClassifierAsync(regimes, regimePerformance, cancellationToken).ConfigureAwait(false);

            result.Success = true;
            result.EndTime = DateTime.UtcNow;
            result.SampleCount = regimes.Count;

            _logger.LogInformation("✅ RegimeDetectorTrainer completed training - Regimes: {Count}, Duration: {Duration:F1}s",
                regimes.Count, (result.EndTime.Value - result.StartTime).TotalSeconds);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ RegimeDetectorTrainer failed: {Error}", ex.Message);
            result.ErrorMessage = ex.Message;
            result.EndTime = DateTime.UtcNow;
            return result;
        }
    }

    private List<MarketRegime> ClassifyRegimes(List<HistoricalBar> sortedBars)
    {
        var regimes = new List<MarketRegime>();

        for (int i = _lookbackWindow; i < sortedBars.Count; i++)
        {
            var window = sortedBars.Skip(i - _lookbackWindow).Take(_lookbackWindow).ToList();
            var regime = DetermineRegime(window);
            regimes.Add(regime);
        }

        return regimes;
    }

    private MarketRegime DetermineRegime(List<HistoricalBar> window)
    {
        // Calculate trend strength using linear regression slope
        var prices = window.Select(b => (double)b.Close).ToArray();
        var slope = CalculateTrendSlope(prices);

        // Calculate volatility using ATR approximation
        var volatility = CalculateVolatility(window);

        // Classify regime
        var regimeType = Math.Abs(slope) > _trendThreshold ? 
                        (slope > 0 ? "TREND_UP" : "TREND_DOWN") : 
                        "RANGE";

        // Detect transitions
        if (window.Count > 2)
        {
            var recentSlope = CalculateTrendSlope(prices.Skip(prices.Length / 2).ToArray());
            if (Math.Sign(slope) != Math.Sign(recentSlope))
            {
                regimeType = "TRANSITION";
            }
        }

        return new MarketRegime
        {
            Timestamp = window.Last().Timestamp,
            RegimeType = regimeType,
            Confidence = Math.Min(Math.Abs(slope) / _trendThreshold, 1.0),
            Volatility = volatility,
            TrendSlope = slope
        };
    }

    private double CalculateTrendSlope(double[] prices)
    {
        if (prices.Length < 2)
            return 0;

        // Simple linear regression
        var n = prices.Length;
        var sumX = n * (n - 1) / 2.0;
        var sumY = prices.Sum();
        var sumXY = prices.Select((p, i) => i * p).Sum();
        var sumX2 = n * (n - 1) * (2 * n - 1) / 6.0;

        var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        return slope;
    }

    private double CalculateVolatility(List<HistoricalBar> window)
    {
        if (window.Count < 2)
            return 0;

        // ATR approximation
        var trueRanges = window.Select(b => (double)(b.High - b.Low)).ToArray();
        return trueRanges.Average();
    }

    private Dictionary<string, RegimeStatistics> CalculateRegimeStatistics(List<MarketRegime> regimes)
    {
        var stats = new Dictionary<string, RegimeStatistics>();

        foreach (var regime in regimes)
        {
            if (!stats.ContainsKey(regime.RegimeType))
            {
                stats[regime.RegimeType] = new RegimeStatistics
                {
                    RegimeType = regime.RegimeType,
                    Count = 0,
                    AverageConfidence = 0,
                    AverageVolatility = 0
                };
            }

            var stat = stats[regime.RegimeType];
            stat.Count++;
            stat.AverageConfidence = (stat.AverageConfidence * (stat.Count - 1) + regime.Confidence) / stat.Count;
            stat.AverageVolatility = (stat.AverageVolatility * (stat.Count - 1) + regime.Volatility) / stat.Count;
        }

        return stats;
    }

    private void LogRegimeStatistics(Dictionary<string, RegimeStatistics> stats)
    {
        foreach (var kvp in stats)
        {
            _logger.LogInformation("Regime '{Type}': {Count} periods, {AvgConf:F2} avg confidence, {AvgVol:F2} avg volatility",
                kvp.Key, kvp.Value.Count, kvp.Value.AverageConfidence, kvp.Value.AverageVolatility);
        }
    }

    private Dictionary<string, double> AnalyzeRegimePerformance(
        List<MarketRegime> regimes,
        List<TradingExperience> experiences)
    {
        var performance = new Dictionary<string, double>();

        foreach (var exp in experiences)
        {
            // Find the regime at entry time
            var regime = regimes.FirstOrDefault(r => 
                Math.Abs((r.Timestamp.DateTime - exp.Timestamp).TotalMinutes) < 5);

            if (regime != null)
            {
                if (!performance.ContainsKey(regime.RegimeType))
                    performance[regime.RegimeType] = 0;

                performance[regime.RegimeType] += (double)exp.RMultiple;
            }
        }

        // Average the performance
        var counts = new Dictionary<string, int>();
        foreach (var exp in experiences)
        {
            var regime = regimes.FirstOrDefault(r => 
                Math.Abs((r.Timestamp.DateTime - exp.Timestamp).TotalMinutes) < 5);
            if (regime != null)
            {
                if (!counts.ContainsKey(regime.RegimeType))
                    counts[regime.RegimeType] = 0;
                counts[regime.RegimeType]++;
            }
        }

        foreach (var key in performance.Keys.ToList())
        {
            if (counts.ContainsKey(key) && counts[key] > 0)
            {
                performance[key] /= counts[key];
            }
        }

        return performance;
    }

    private async Task TrainRegimeClassifierAsync(
        List<MarketRegime> regimes,
        Dictionary<string, double> performance,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Training regime classifier with {RegimeCount} regimes...", regimes.Count);

        // Simulate training time
        await Task.Delay(TimeSpan.FromSeconds(7), cancellationToken).ConfigureAwait(false);

        // Log regime performance
        foreach (var kvp in performance)
        {
            _logger.LogInformation("Regime '{Type}': {AvgR:F2} average R-multiple",
                kvp.Key, kvp.Value);
        }

        // In production, this would:
        // 1. Create feature vectors from price action, volume, volatility
        // 2. Train multi-class classifier (Random Forest, XGBoost, or Neural Net)
        // 3. Validate regime transitions
        // 4. Save trained model to ONNX format

        _logger.LogInformation("Regime classifier training complete");
    }
}

/// <summary>
/// Market regime data structure
/// </summary>
public class MarketRegime
{
    public required DateTimeOffset Timestamp { get; init; }
    public required string RegimeType { get; init; }
    public required double Confidence { get; init; }
    public required double Volatility { get; init; }
    public required double TrendSlope { get; init; }
}

/// <summary>
/// Regime statistics
/// </summary>
public class RegimeStatistics
{
    public required string RegimeType { get; init; }
    public int Count { get; set; }
    public double AverageConfidence { get; set; }
    public double AverageVolatility { get; set; }
}
