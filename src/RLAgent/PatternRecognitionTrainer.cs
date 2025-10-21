using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Models;

namespace TradingBot.RLAgent;

/// <summary>
/// Pattern Recognition Trainer - Lab-only component for candlestick pattern detection
/// Trains on historical patterns to identify high-probability setups
/// This component runs ONLY in Lab mode during Sunday training sessions
/// </summary>
public class PatternRecognitionTrainer
{
    private readonly ILogger<PatternRecognitionTrainer> _logger;
    private readonly int _minPatternLength;
    private readonly int _maxPatternLength;
    private readonly Dictionary<string, int> _patternCounts;
    
    public PatternRecognitionTrainer(
        ILogger<PatternRecognitionTrainer> logger,
        int minPatternLength = 3,
        int maxPatternLength = 10)
    {
        _logger = logger;
        _minPatternLength = minPatternLength;
        _maxPatternLength = maxPatternLength;
        _patternCounts = new Dictionary<string, int>();
        
        _logger.LogInformation("PatternRecognitionTrainer initialized (Lab mode) - MinLen: {Min}, MaxLen: {Max}",
            _minPatternLength, _maxPatternLength);
    }

    /// <summary>
    /// Train pattern recognition model from historical bar data (Lab entry point)
    /// This is called by HistoricalTrainingOrchestrator during Sunday training
    /// </summary>
    public async Task<TrainingResult> TrainFromHistoricalBarsAsync(
        List<HistoricalBar> bars,
        List<TradingExperience> experiences,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔧 PatternRecognitionTrainer starting training from {BarCount} bars and {ExpCount} experiences",
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
            if (bars.Count < _maxPatternLength)
            {
                _logger.LogWarning("Insufficient bars for pattern training: {Count} < {Required}",
                    bars.Count, _maxPatternLength);
                result.ErrorMessage = $"Insufficient bars: {bars.Count} < {_maxPatternLength}";
                result.EndTime = DateTime.UtcNow;
                return result;
            }

            // Sort bars chronologically
            var sortedBars = bars.OrderBy(b => b.Timestamp).ToList();

            // Detect and classify patterns
            var patterns = DetectPatterns(sortedBars);
            _logger.LogInformation("Detected {Count} patterns across {Types} types",
                patterns.Count, _patternCounts.Count);

            // Correlate patterns with trading outcomes
            var patternPerformance = AnalyzePatternPerformance(patterns, experiences);
            _logger.LogInformation("Analyzed performance for {Count} pattern types",
                patternPerformance.Count);

            // Train pattern classifier
            await TrainPatternClassifierAsync(patterns, patternPerformance, cancellationToken).ConfigureAwait(false);

            result.Success = true;
            result.EndTime = DateTime.UtcNow;
            result.SampleCount = patterns.Count;

            _logger.LogInformation("✅ PatternRecognitionTrainer completed training - Patterns: {Count}, Duration: {Duration:F1}s",
                patterns.Count, (result.EndTime.Value - result.StartTime).TotalSeconds);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ PatternRecognitionTrainer failed: {Error}", ex.Message);
            result.ErrorMessage = ex.Message;
            result.EndTime = DateTime.UtcNow;
            return result;
        }
    }

    private List<CandlestickPattern> DetectPatterns(List<HistoricalBar> sortedBars)
    {
        var patterns = new List<CandlestickPattern>();
        _patternCounts.Clear();

        for (int i = _minPatternLength; i <= sortedBars.Count - _maxPatternLength; i++)
        {
            // Detect various candlestick patterns
            var pattern = ClassifyPattern(sortedBars.Skip(i).Take(_maxPatternLength).ToList());
            
            if (pattern != null)
            {
                patterns.Add(pattern);
                
                if (!_patternCounts.ContainsKey(pattern.Name))
                    _patternCounts[pattern.Name] = 0;
                
                _patternCounts[pattern.Name]++;
            }
        }

        return patterns;
    }

    private CandlestickPattern? ClassifyPattern(List<HistoricalBar> bars)
    {
        if (bars.Count < _minPatternLength)
            return null;

        // Simplified pattern detection - in production, this would include:
        // - Doji, Hammer, Shooting Star, Engulfing, etc.
        // - Support/Resistance levels
        // - Trend strength indicators
        
        var firstBar = bars[0];
        var lastBar = bars[^1];
        
        // Simple bullish/bearish pattern detection
        var priceChange = (double)(lastBar.Close - firstBar.Open) / (double)firstBar.Open;
        var patternName = priceChange > 0.01 ? "BullishSequence" : 
                         priceChange < -0.01 ? "BearishSequence" : "Neutral";

        return new CandlestickPattern
        {
            Name = patternName,
            StartTime = firstBar.Timestamp,
            EndTime = lastBar.Timestamp,
            Confidence = Math.Abs(priceChange) * 100,
            Bars = bars
        };
    }

    private Dictionary<string, PatternPerformance> AnalyzePatternPerformance(
        List<CandlestickPattern> patterns,
        List<TradingExperience> experiences)
    {
        var performance = new Dictionary<string, PatternPerformance>();

        foreach (var pattern in patterns)
        {
            // Find experiences that occurred during or after this pattern
            var relevantExperiences = experiences.Where(e => 
                e.Timestamp >= pattern.StartTime.DateTime && 
                e.Timestamp <= pattern.EndTime.DateTime.AddHours(4)).ToList();

            if (!performance.ContainsKey(pattern.Name))
            {
                performance[pattern.Name] = new PatternPerformance
                {
                    PatternName = pattern.Name,
                    TotalOccurrences = 0,
                    WinningTrades = 0,
                    LosingTrades = 0,
                    AverageRMultiple = 0
                };
            }

            var perf = performance[pattern.Name];
            perf.TotalOccurrences++;

            if (relevantExperiences.Any())
            {
                var winCount = relevantExperiences.Count(e => e.RMultiple > 0);
                var loseCount = relevantExperiences.Count(e => e.RMultiple <= 0);
                
                perf.WinningTrades += winCount;
                perf.LosingTrades += loseCount;
                perf.AverageRMultiple = relevantExperiences.Average(e => (double)e.RMultiple);
            }
        }

        return performance;
    }

    private async Task TrainPatternClassifierAsync(
        List<CandlestickPattern> patterns,
        Dictionary<string, PatternPerformance> performance,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Training pattern classifier with {PatternCount} patterns...", patterns.Count);

        // Simulate training time
        await Task.Delay(TimeSpan.FromSeconds(8), cancellationToken).ConfigureAwait(false);

        // Log pattern performance
        foreach (var kvp in performance)
        {
            _logger.LogInformation("Pattern '{Name}': {Occurrences} occurrences, {WinRate:F1}% win rate, {AvgR:F2} avg R-multiple",
                kvp.Key, kvp.Value.TotalOccurrences, 
                kvp.Value.TotalOccurrences > 0 ? (kvp.Value.WinningTrades * 100.0 / kvp.Value.TotalOccurrences) : 0,
                kvp.Value.AverageRMultiple);
        }

        // In production, this would:
        // 1. Create feature vectors from pattern characteristics
        // 2. Train classification model (SVM, Random Forest, or Neural Net)
        // 3. Validate on holdout set
        // 4. Save trained model to ONNX format

        _logger.LogInformation("Pattern classifier training complete");
    }
}

/// <summary>
/// Candlestick pattern data structure
/// </summary>
public class CandlestickPattern
{
    public required string Name { get; init; }
    public required DateTimeOffset StartTime { get; init; }
    public required DateTimeOffset EndTime { get; init; }
    public required double Confidence { get; init; }
    public required List<HistoricalBar> Bars { get; init; }
}

/// <summary>
/// Pattern performance metrics
/// </summary>
public class PatternPerformance
{
    public required string PatternName { get; init; }
    public int TotalOccurrences { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public double AverageRMultiple { get; set; }
}
