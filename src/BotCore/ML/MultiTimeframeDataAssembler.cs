using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.ML;

/// <summary>
/// Multi-timeframe data assembler for coordinated training.
/// Creates synchronized training samples from multiple timeframes (5m + 1m).
/// 
/// Purpose: Assemble coordinated multi-timeframe samples where each sample contains:
/// - 5-minute context: Last 36 bars (3 hours of strategic context)
/// - 1-minute context: Last 60 bars (1 hour of tactical context)
/// - Label: Forward-looking outcome for supervised/RL learning
/// 
/// Design principles:
/// - Synchronized timestamps: All timeframes aligned to same decision point
/// - No lookahead bias: Only uses data available at decision time
/// - Deterministic: Same input always produces same output
/// - Production-ready: Full error handling and validation
/// </summary>
public class MultiTimeframeDataAssembler
{
    private readonly ILogger<MultiTimeframeDataAssembler> _logger;
    private readonly MultiTimeframeFeatureExtractor _featureExtractor;
    
    // Context window sizes
    private const int Context5mBars = 36;  // 3 hours of 5m bars
    private const int Context1mBars = 60;  // 1 hour of 1m bars
    
    // Forward-looking window for labeling
    private const int LabelLookaheadMinutes = 5;
    
    public MultiTimeframeDataAssembler(
        ILogger<MultiTimeframeDataAssembler> logger,
        MultiTimeframeFeatureExtractor featureExtractor)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _featureExtractor = featureExtractor ?? throw new ArgumentNullException(nameof(featureExtractor));
    }
    
    /// <summary>
    /// Assemble synchronized multi-timeframe samples from historical data.
    /// Each sample represents one decision point with full multi-timeframe context.
    /// </summary>
    /// <param name="symbol">Symbol to assemble (e.g., "ES", "NQ")</param>
    /// <param name="bars5m">5-minute bars, ordered chronologically</param>
    /// <param name="bars1m">1-minute bars, ordered chronologically</param>
    /// <returns>List of synchronized multi-timeframe samples</returns>
    public List<EnhancedMultiTimeframeSample> AssembleSamples(
        string symbol,
        List<BarData> bars5m,
        List<BarData> bars1m)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));
        }
        
        if (bars5m == null || bars1m == null)
        {
            throw new ArgumentNullException("Bars cannot be null");
        }
        
        var samples = new List<EnhancedMultiTimeframeSample>();
        
        try
        {
            // Ensure bars are sorted chronologically
            var sorted5m = bars5m.OrderBy(b => b.Timestamp).ToList();
            var sorted1m = bars1m.OrderBy(b => b.Timestamp).ToList();
            
            _logger.LogInformation(
                "[ASSEMBLER] Assembling multi-timeframe samples for {Symbol}: {Count5m} 5m bars, {Count1m} 1m bars",
                symbol, sorted5m.Count, sorted1m.Count);
            
            // Create index for fast 1m bar lookup
            var bars1mByTime = sorted1m.ToDictionary(b => b.Timestamp, b => b);
            
            // Iterate through 5m bars as decision points
            // Start after we have sufficient context
            for (int i = Context5mBars; i < sorted5m.Count - LabelLookaheadMinutes; i++)
            {
                var decisionTime = sorted5m[i].Timestamp;
                
                // Gather 5-minute context (last 36 bars)
                var context5m = sorted5m.Skip(i - Context5mBars).Take(Context5mBars).ToList();
                
                // Gather 1-minute context (last 60 bars ending at decision time)
                var context1m = GatherContext1m(decisionTime, bars1mByTime, sorted1m);
                
                if (context1m.Count < Context1mBars)
                {
                    // Skip if insufficient 1m data
                    continue;
                }
                
                // Extract features from both timeframes
                var features5m = _featureExtractor.Extract5mFeatures(context5m);
                var features1m = _featureExtractor.Extract1mFeatures(context1m);
                
                // Calculate label (forward-looking outcome)
                var label = CalculateLabel(sorted5m, i, LabelLookaheadMinutes);
                
                // Create synchronized sample using existing MultiTimeframeSample class
                var sample = new EnhancedMultiTimeframeSample
                {
                    Symbol = symbol,
                    Timestamp = decisionTime,
                    Context5m = context5m,
                    Context1m = context1m,
                    Features5m = features5m,
                    Features1m = features1m,
                    Label = label
                };
                
                samples.Add(sample);
            }
            
            _logger.LogInformation(
                "[ASSEMBLER] Created {SampleCount} synchronized samples for {Symbol}",
                samples.Count, symbol);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ASSEMBLER] Error assembling samples for {Symbol}", symbol);
            throw;
        }
        
        return samples;
    }
    
    /// <summary>
    /// Gather 1-minute context bars ending at decision time.
    /// </summary>
    private List<BarData> GatherContext1m(
        DateTimeOffset decisionTime,
        Dictionary<DateTimeOffset, BarData> bars1mByTime,
        List<BarData> allBars1m)
    {
        var context = new List<BarData>();
        
        // Find 1m bars in the 60-minute window ending at decision time
        for (int minutesBack = Context1mBars - 1; minutesBack >= 0; minutesBack--)
        {
            var targetTime = decisionTime.AddMinutes(-minutesBack);
            
            if (bars1mByTime.TryGetValue(targetTime, out var bar))
            {
                context.Add(bar);
            }
            else
            {
                // Try to find nearest bar within 1 minute tolerance
                var nearestBar = FindNearestBar(targetTime, allBars1m, TimeSpan.FromMinutes(1));
                if (nearestBar != null)
                {
                    context.Add(nearestBar);
                }
            }
        }
        
        return context;
    }
    
    /// <summary>
    /// Find nearest bar to target time within tolerance.
    /// </summary>
    private BarData? FindNearestBar(DateTimeOffset targetTime, List<BarData> bars, TimeSpan tolerance)
    {
        return bars
            .Where(b => Math.Abs((b.Timestamp - targetTime).TotalSeconds) <= tolerance.TotalSeconds)
            .OrderBy(b => Math.Abs((b.Timestamp - targetTime).TotalSeconds))
            .FirstOrDefault();
    }
    
    /// <summary>
    /// Calculate forward-looking label for supervised learning.
    /// Returns 1 if price went up, -1 if down, 0 if flat.
    /// </summary>
    private double CalculateLabel(List<BarData> bars5m, int currentIndex, int lookaheadMinutes)
    {
        var currentPrice = bars5m[currentIndex].Close;
        
        // Look ahead (lookaheadMinutes / 5) bars (since each bar is 5 minutes)
        var lookaheadBars = lookaheadMinutes / 5;
        var futureIndex = currentIndex + lookaheadBars;
        
        if (futureIndex >= bars5m.Count)
        {
            return 0.0; // Not enough future data
        }
        
        var futurePrice = bars5m[futureIndex].Close;
        var priceChange = futurePrice - currentPrice;
        var percentChange = (double)(priceChange / currentPrice);
        
        // Threshold for "significant" move (0.1% for ES/NQ)
        const double threshold = 0.001;
        
        if (percentChange > threshold)
        {
            return 1.0; // Up
        }
        else if (percentChange < -threshold)
        {
            return -1.0; // Down
        }
        else
        {
            return 0.0; // Flat
        }
    }
}

/// <summary>
/// Enhanced multi-timeframe sample that extends the base sample with additional context.
/// Contains full bar context and labels for coordinated multi-timeframe training.
/// </summary>
public class EnhancedMultiTimeframeSample : MultiTimeframeSample
{
    /// <summary>5-minute context bars (strategic timeframe)</summary>
    public List<BarData> Context5m { get; set; } = new();
    
    /// <summary>1-minute context bars (tactical timeframe)</summary>
    public List<BarData> Context1m { get; set; } = new();
    
    /// <summary>Label for supervised learning (1=up, -1=down, 0=flat)</summary>
    public double Label { get; set; }
}
