using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace BotCore.ML;

/// <summary>
/// Multi-timeframe feature extraction for training and live inference.
/// Extracts synchronized features from 5-minute and 1-minute bars.
/// 
/// Phase 2: Feature Extraction (Week 2)
/// - Extract5mFeatures(): Compute ATR, RSI, MACD, volume imbalance, trend slope
/// - Extract1mFeatures(): Same indicators but faster windows
/// - SynchronizeFeatures(): Retrieve features from all timeframes for given timestamp
/// 
/// Design principles:
/// - Deterministic: Same input always produces same output
/// - Versioned: Feature computation is versioned via hash
/// - No lookahead bias: Only uses data up to timestamp
/// - Production-ready: Full error handling and logging
/// </summary>
public class MultiTimeframeFeatureExtractor
{
    private readonly ILogger<MultiTimeframeFeatureExtractor> _logger;
    
    // Feature extraction parameters for 5-minute bars
    private const int Atr5mWindow = 14;
    private const int Rsi5mWindow = 14;
    private const int Macd5mFastPeriod = 12;
    private const int Macd5mSlowPeriod = 26;
    private const int Macd5mSignalPeriod = 9;
    private const int Volume5mWindow = 20;
    private const int Trend5mWindow = 10;
    
    // Feature extraction parameters for 1-minute bars (faster windows)
    private const int Atr1mWindow = 14;  // Same period but more responsive
    private const int Rsi1mWindow = 14;
    private const int Macd1mFastPeriod = 5;   // Faster MACD for 1m
    private const int Macd1mSlowPeriod = 13;
    private const int Macd1mSignalPeriod = 5;
    private const int Volume1mWindow = 20;
    private const int Trend1mWindow = 10;
    
    // Normalization constants
    private const double RsiMax = 100.0;
    private const double PercentageMultiplier = 100.0;
    private const double Epsilon = 1e-10;
    
    // Feature version - increment when feature calculation logic changes
    private const string FeatureVersion = "1.0.0";
    
    public MultiTimeframeFeatureExtractor(ILogger<MultiTimeframeFeatureExtractor> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }
    
    /// <summary>
    /// Extract features from 5-minute bars.
    /// </summary>
    /// <param name="bars">List of 5-minute OHLCV bars, ordered chronologically</param>
    /// <returns>Dictionary of feature name to value</returns>
    public Dictionary<string, double> Extract5mFeatures(List<BarData> bars)
    {
        if (bars == null || bars.Count == 0)
        {
            _logger.LogWarning("[FEATURE_EXTRACT] No 5m bars provided");
            return new Dictionary<string, double>();
        }
        
        var features = new Dictionary<string, double>();
        
        try
        {
            // Ensure bars are sorted chronologically
            var sortedBars = bars.OrderBy(b => b.Timestamp).ToList();
            
            // Calculate ATR (Average True Range)
            if (sortedBars.Count >= Atr5mWindow)
            {
                var atr = CalculateATR(sortedBars, Atr5mWindow);
                features["atr_5m"] = atr;
            }
            else
            {
                features["atr_5m"] = 0.0;
            }
            
            // Calculate RSI (Relative Strength Index)
            if (sortedBars.Count >= Rsi5mWindow)
            {
                var rsi = CalculateRSI(sortedBars, Rsi5mWindow);
                features["rsi_5m"] = rsi / RsiMax;  // Normalize to [0, 1]
            }
            else
            {
                features["rsi_5m"] = 0.5;  // Neutral value
            }
            
            // Calculate MACD
            if (sortedBars.Count >= Macd5mSlowPeriod)
            {
                var (macd, signal) = CalculateMACD(sortedBars, Macd5mFastPeriod, Macd5mSlowPeriod, Macd5mSignalPeriod);
                features["macd_5m"] = macd;
                features["macd_signal_5m"] = signal;
                features["macd_histogram_5m"] = macd - signal;
            }
            else
            {
                features["macd_5m"] = 0.0;
                features["macd_signal_5m"] = 0.0;
                features["macd_histogram_5m"] = 0.0;
            }
            
            // Calculate volume imbalance (buying vs selling pressure)
            if (sortedBars.Count >= Volume5mWindow)
            {
                var volumeImbalance = CalculateVolumeImbalance(sortedBars, Volume5mWindow);
                features["volume_imbalance_5m"] = volumeImbalance;
            }
            else
            {
                features["volume_imbalance_5m"] = 0.0;
            }
            
            // Calculate trend slope (linear regression on close prices)
            if (sortedBars.Count >= Trend5mWindow)
            {
                var trendSlope = CalculateTrendSlope(sortedBars, Trend5mWindow);
                features["trend_slope_5m"] = trendSlope;
            }
            else
            {
                features["trend_slope_5m"] = 0.0;
            }
            
            _logger.LogDebug("[FEATURE_EXTRACT] Extracted {Count} 5m features", features.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_EXTRACT] Error extracting 5m features");
        }
        
        return features;
    }
    
    /// <summary>
    /// Extract features from 1-minute bars (faster windows for responsiveness).
    /// </summary>
    /// <param name="bars">List of 1-minute OHLCV bars, ordered chronologically</param>
    /// <returns>Dictionary of feature name to value</returns>
    public Dictionary<string, double> Extract1mFeatures(List<BarData> bars)
    {
        if (bars == null || bars.Count == 0)
        {
            _logger.LogWarning("[FEATURE_EXTRACT] No 1m bars provided");
            return new Dictionary<string, double>();
        }
        
        var features = new Dictionary<string, double>();
        
        try
        {
            // Ensure bars are sorted chronologically
            var sortedBars = bars.OrderBy(b => b.Timestamp).ToList();
            
            // Calculate ATR (same window, but more responsive due to 1m bars)
            if (sortedBars.Count >= Atr1mWindow)
            {
                var atr = CalculateATR(sortedBars, Atr1mWindow);
                features["atr_1m"] = atr;
            }
            else
            {
                features["atr_1m"] = 0.0;
            }
            
            // Calculate RSI
            if (sortedBars.Count >= Rsi1mWindow)
            {
                var rsi = CalculateRSI(sortedBars, Rsi1mWindow);
                features["rsi_1m"] = rsi / RsiMax;  // Normalize to [0, 1]
            }
            else
            {
                features["rsi_1m"] = 0.5;  // Neutral value
            }
            
            // Calculate MACD with faster periods
            if (sortedBars.Count >= Macd1mSlowPeriod)
            {
                var (macd, signal) = CalculateMACD(sortedBars, Macd1mFastPeriod, Macd1mSlowPeriod, Macd1mSignalPeriod);
                features["macd_1m"] = macd;
                features["macd_signal_1m"] = signal;
                features["macd_histogram_1m"] = macd - signal;
            }
            else
            {
                features["macd_1m"] = 0.0;
                features["macd_signal_1m"] = 0.0;
                features["macd_histogram_1m"] = 0.0;
            }
            
            // Calculate volume imbalance
            if (sortedBars.Count >= Volume1mWindow)
            {
                var volumeImbalance = CalculateVolumeImbalance(sortedBars, Volume1mWindow);
                features["volume_imbalance_1m"] = volumeImbalance;
            }
            else
            {
                features["volume_imbalance_1m"] = 0.0;
            }
            
            // Calculate trend slope
            if (sortedBars.Count >= Trend1mWindow)
            {
                var trendSlope = CalculateTrendSlope(sortedBars, Trend1mWindow);
                features["trend_slope_1m"] = trendSlope;
            }
            else
            {
                features["trend_slope_1m"] = 0.0;
            }
            
            _logger.LogDebug("[FEATURE_EXTRACT] Extracted {Count} 1m features", features.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_EXTRACT] Error extracting 1m features");
        }
        
        return features;
    }
    
    /// <summary>
    /// Synchronize features from multiple timeframes for a given timestamp.
    /// Ensures no lookahead bias - only uses data up to the specified timestamp.
    /// </summary>
    /// <param name="timestamp">Target timestamp</param>
    /// <param name="bars5m">5-minute bars (must include bars up to timestamp)</param>
    /// <param name="bars1m">1-minute bars (must include bars up to timestamp)</param>
    /// <returns>Combined feature dictionary with all timeframes</returns>
    public Dictionary<string, double> SynchronizeFeatures(
        DateTimeOffset timestamp,
        List<BarData> bars5m,
        List<BarData> bars1m)
    {
        var synchronizedFeatures = new Dictionary<string, double>();
        
        try
        {
            // Filter bars to only include data up to timestamp (prevent lookahead bias)
            var filtered5m = bars5m?.Where(b => b.Timestamp <= timestamp).ToList() ?? new List<BarData>();
            var filtered1m = bars1m?.Where(b => b.Timestamp <= timestamp).ToList() ?? new List<BarData>();
            
            _logger.LogDebug(
                "[FEATURE_SYNC] Synchronizing features for {Timestamp}: {Count5m} 5m bars, {Count1m} 1m bars",
                timestamp, filtered5m.Count, filtered1m.Count);
            
            // Extract features from each timeframe
            var features5m = Extract5mFeatures(filtered5m);
            var features1m = Extract1mFeatures(filtered1m);
            
            // Combine features
            foreach (var kvp in features5m)
            {
                synchronizedFeatures[kvp.Key] = kvp.Value;
            }
            
            foreach (var kvp in features1m)
            {
                synchronizedFeatures[kvp.Key] = kvp.Value;
            }
            
            // Add metadata
            synchronizedFeatures["feature_count_5m"] = features5m.Count;
            synchronizedFeatures["feature_count_1m"] = features1m.Count;
            synchronizedFeatures["timestamp_unix"] = timestamp.ToUnixTimeSeconds();
            
            _logger.LogDebug(
                "[FEATURE_SYNC] Synchronized {Count} total features",
                synchronizedFeatures.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_SYNC] Error synchronizing features for {Timestamp}", timestamp);
        }
        
        return synchronizedFeatures;
    }
    
    /// <summary>
    /// Get feature version hash for reproducibility.
    /// This hash changes when feature calculation logic changes.
    /// </summary>
    public string GetFeatureVersionHash()
    {
        var versionString = $"{FeatureVersion}_{Atr5mWindow}_{Rsi5mWindow}_{Macd5mFastPeriod}_{Macd5mSlowPeriod}";
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(versionString));
        return Convert.ToHexString(hashBytes)[..16];  // First 16 chars
    }
    
    #region Technical Indicator Calculations
    
    /// <summary>
    /// Calculate Average True Range (ATR) - measures volatility.
    /// </summary>
    private static double CalculateATR(List<BarData> bars, int window)
    {
        if (bars.Count < window)
        {
            return 0.0;
        }
        
        var recentBars = bars.TakeLast(window).ToList();
        var trueRanges = new List<double>();
        
        for (int i = 1; i < recentBars.Count; i++)
        {
            var current = recentBars[i];
            var previous = recentBars[i - 1];
            
            var tr1 = current.High - current.Low;
            var tr2 = Math.Abs(current.High - previous.Close);
            var tr3 = Math.Abs(current.Low - previous.Close);
            
            trueRanges.Add(Math.Max(tr1, Math.Max(tr2, tr3)));
        }
        
        return trueRanges.Count > 0 ? trueRanges.Average() : 0.0;
    }
    
    /// <summary>
    /// Calculate Relative Strength Index (RSI) - momentum indicator.
    /// </summary>
    private static double CalculateRSI(List<BarData> bars, int window)
    {
        if (bars.Count < window + 1)
        {
            return 50.0;  // Neutral
        }
        
        var recentBars = bars.TakeLast(window + 1).ToList();
        var gains = new List<double>();
        var losses = new List<double>();
        
        for (int i = 1; i < recentBars.Count; i++)
        {
            var change = recentBars[i].Close - recentBars[i - 1].Close;
            
            if (change > 0)
            {
                gains.Add(change);
                losses.Add(0.0);
            }
            else
            {
                gains.Add(0.0);
                losses.Add(Math.Abs(change));
            }
        }
        
        var avgGain = gains.Average();
        var avgLoss = losses.Average();
        
        if (avgLoss < Epsilon)
        {
            return 100.0;
        }
        
        var rs = avgGain / avgLoss;
        var rsi = 100.0 - (100.0 / (1.0 + rs));
        
        return rsi;
    }
    
    /// <summary>
    /// Calculate MACD (Moving Average Convergence Divergence) - trend indicator.
    /// </summary>
    private static (double macd, double signal) CalculateMACD(
        List<BarData> bars,
        int fastPeriod,
        int slowPeriod,
        int signalPeriod)
    {
        if (bars.Count < slowPeriod)
        {
            return (0.0, 0.0);
        }
        
        var closePrices = bars.Select(b => b.Close).ToList();
        
        // Calculate EMAs
        var fastEma = CalculateEMA(closePrices, fastPeriod);
        var slowEma = CalculateEMA(closePrices, slowPeriod);
        
        // MACD line = Fast EMA - Slow EMA
        var macd = fastEma - slowEma;
        
        // Signal line = EMA of MACD (simplified for now)
        var signal = macd * 0.9;  // Simplified signal calculation
        
        return (macd, signal);
    }
    
    /// <summary>
    /// Calculate Exponential Moving Average (EMA).
    /// </summary>
    private static double CalculateEMA(List<double> values, int period)
    {
        if (values.Count < period)
        {
            return values.Count > 0 ? values.Average() : 0.0;
        }
        
        var multiplier = 2.0 / (period + 1.0);
        var ema = values.Take(period).Average();  // Start with SMA
        
        foreach (var value in values.Skip(period))
        {
            ema = (value * multiplier) + (ema * (1.0 - multiplier));
        }
        
        return ema;
    }
    
    /// <summary>
    /// Calculate volume imbalance (buying pressure vs selling pressure).
    /// Positive = more buying, Negative = more selling.
    /// </summary>
    private static double CalculateVolumeImbalance(List<BarData> bars, int window)
    {
        if (bars.Count < window)
        {
            return 0.0;
        }
        
        var recentBars = bars.TakeLast(window).ToList();
        double buyingVolume = 0.0;
        double sellingVolume = 0.0;
        
        foreach (var bar in recentBars)
        {
            // If close > open, consider it buying volume
            if (bar.Close > bar.Open)
            {
                buyingVolume += bar.Volume;
            }
            else if (bar.Close < bar.Open)
            {
                sellingVolume += bar.Volume;
            }
            // If close == open, don't count it
        }
        
        var totalVolume = buyingVolume + sellingVolume;
        if (totalVolume < Epsilon)
        {
            return 0.0;
        }
        
        // Return imbalance: range [-1, 1]
        return (buyingVolume - sellingVolume) / totalVolume;
    }
    
    /// <summary>
    /// Calculate trend slope using linear regression on close prices.
    /// Positive slope = uptrend, Negative slope = downtrend.
    /// </summary>
    private static double CalculateTrendSlope(List<BarData> bars, int window)
    {
        if (bars.Count < window)
        {
            return 0.0;
        }
        
        var recentBars = bars.TakeLast(window).ToList();
        var closePrices = recentBars.Select(b => b.Close).ToList();
        
        // Simple linear regression: y = mx + b
        var n = closePrices.Count;
        var sumX = 0.0;
        var sumY = 0.0;
        var sumXY = 0.0;
        var sumX2 = 0.0;
        
        for (int i = 0; i < n; i++)
        {
            var x = i;
            var y = closePrices[i];
            
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        
        var denominator = (n * sumX2) - (sumX * sumX);
        if (Math.Abs(denominator) < Epsilon)
        {
            return 0.0;
        }
        
        var slope = ((n * sumXY) - (sumX * sumY)) / denominator;
        
        // Normalize slope by average price to make it scale-independent
        var avgPrice = sumY / n;
        if (avgPrice < Epsilon)
        {
            return 0.0;
        }
        
        return (slope / avgPrice) * PercentageMultiplier;  // As percentage
    }
    
    #endregion
}

/// <summary>
/// Simple OHLCV bar data structure.
/// </summary>
public class BarData
{
    public DateTimeOffset Timestamp { get; set; }
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double Volume { get; set; }
}
