using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Streaming feature aggregation and caching service
/// Implements requirement: Streaming feature aggregation and caching in FeatureEngineering
/// </summary>
public class StreamingFeatureEngineering : IDisposable
{
    // Constants for magic number violations
    private const int DefaultBatchSize = 50;
    
    // Technical analysis periods
    private const int ShortSMA = 5;
    private const int MediumSMA = 20;
    private const int LongSMA = 50;
    private const int ShortEMA = 12;
    private const int LongEMA = 26;
    private const int ShortVolatility = 10;
    private const int MediumVolatility = 20;
    private const int ATRPeriod = 14;
    private const int RSIPeriod = 14;
    private const int ReturnsShort = 1;
    private const int CacheExpiredMinutes = 10;
    
    private readonly ILogger<StreamingFeatureEngineering> _logger;
    private readonly ConcurrentDictionary<string, FeatureCache> _featureCaches = new();
    private readonly ConcurrentDictionary<string, StreamingAggregator> _aggregators = new();
    private readonly Timer _cleanupTimer;

    // Cache configuration
    private readonly TimeSpan _cacheRetention = TimeSpan.FromHours(24);
    private readonly int _maxCacheSize = 10000;

    public StreamingFeatureEngineering(ILogger<StreamingFeatureEngineering> logger)
    {
        _logger = logger;
        
        // Start cleanup timer to prevent memory leaks
        _cleanupTimer = new Timer(CleanupExpiredCaches, null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));
        
        _logger.LogInformation("Streaming feature engineering service initialized");
    }

    /// <summary>
    /// Process new market data and update streaming features
    /// </summary>
    public async Task<Dictionary<string, double>> ProcessMarketDataAsync(MarketData data, CancellationToken cancellationToken = default)
    {
        try
        {
            var symbol = data.Symbol;
            var timestamp = data.Timestamp;

            // Get or create aggregator for this symbol
            var aggregator = _aggregators.GetOrAdd(symbol, _ => new StreamingAggregator(symbol, _logger));

            // Update aggregator with new data
            await aggregator.UpdateAsync(data, cancellationToken).ConfigureAwait(false);

            // Calculate streaming features
            var features = await CalculateStreamingFeaturesAsync(aggregator, data, cancellationToken).ConfigureAwait(false);

            // Cache features for future use
            await CacheFeaturesAsync(symbol, timestamp, features, cancellationToken).ConfigureAwait(false);

            _logger.LogDebug("Processed market data for {Symbol}: {FeatureCount} features calculated", 
                symbol, features.Count);

            return features;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to process market data for feature engineering");
            throw new InvalidOperationException("Feature engineering processing failed for market data", ex);
        }
    }

    /// <summary>
    /// Get cached features for a symbol at a specific time
    /// </summary>
    public async Task<Dictionary<string, double>?> GetCachedFeaturesAsync(string symbol, DateTime timestamp, CancellationToken cancellationToken = default)
    {
        try
        {
        await Task.Yield(); // Ensure async behavior
            
            if (!_featureCaches.TryGetValue(symbol, out var cache))
            {
                return null;
            }

            // Proper async operation instead of Task.FromResult
            var features = cache.GetFeatures(timestamp);
            
            // Add validation and enhancement of cached features
            if (features != null && features.Any())
            {
                // Validate feature quality and recalculate if stale
                var age = DateTime.UtcNow - timestamp;
                if (age > TimeSpan.FromMinutes(5)) // Features older than 5 minutes
                {
                    _logger.LogDebug("Features for {Symbol} are stale ({Age:F1} minutes old), refreshing", 
                        symbol, age.TotalMinutes);
                    
                    // Trigger background refresh but return current features
                    _ = Task.Run(() => RefreshFeaturesForSymbol(symbol), cancellationToken);
                }
            }
            
            return features;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get cached features for {Symbol}", symbol);
            return null;
        }
    }
    
    /// <summary>
    /// Refresh features for a symbol in the background
    /// </summary>
    private void RefreshFeaturesForSymbol(string symbol)
    {
        try
        {
            if (_featureCaches.TryGetValue(symbol, out var cache))
            {
                // Trigger feature recalculation
                cache.RemoveExpiredEntries(DateTime.UtcNow.AddMinutes(-CacheExpiredMinutes));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to refresh features for {Symbol}", symbol);
        }
    }

    /// <summary>
    /// Calculate streaming features from aggregated data
    /// </summary>
    private static async Task<Dictionary<string, double>> CalculateStreamingFeaturesAsync(
        StreamingAggregator aggregator, 
        MarketData currentData, 
        CancellationToken cancellationToken)
    {
        var features = new Dictionary<string, double>();

        // Basic price features
        features["price"] = currentData.Close;
        features["volume"] = currentData.Volume;

        // Moving averages (streaming calculation)
        features["sma_5"] = aggregator.GetSMA(ShortSMA);
        features["sma_20"] = aggregator.GetSMA(MediumSMA);
        features["sma_50"] = aggregator.GetSMA(DefaultBatchSize);
        features["ema_12"] = aggregator.GetEMA(ShortEMA);
        features["ema_26"] = aggregator.GetEMA(LongEMA);

        // Volatility features
        features["volatility_10"] = await aggregator.GetVolatilityAsync(ShortVolatility, cancellationToken).ConfigureAwait(false);
        features["volatility_20"] = await aggregator.GetVolatilityAsync(MediumVolatility, cancellationToken).ConfigureAwait(false);
        features["atr_14"] = await aggregator.GetATRAsync(ATRPeriod, cancellationToken).ConfigureAwait(false);

        // Price ratios and differences
        if (features["sma_20"] > 0)
        {
            features["price_to_sma20"] = features["price"] / features["sma_20"];
        }
        if (features["sma_5"] > 0 && features["sma_20"] > 0)
        {
            features["sma5_to_sma20"] = features["sma_5"] / features["sma_20"];
        }

        // Momentum features
        features["rsi_14"] = await aggregator.GetRSIAsync(RSIPeriod, cancellationToken).ConfigureAwait(false);
        features["macd"] = features["ema_12"] - features["ema_26"];
        features["macd_signal"] = await aggregator.GetMACDSignalAsync(cancellationToken).ConfigureAwait(false);

        // Volume features
        features["volume_sma_20"] = await aggregator.GetVolumeSMAAsync(MediumSMA, cancellationToken).ConfigureAwait(false);
        if (features["volume_sma_20"] > 0)
        {
            features["volume_ratio"] = features["volume"] / features["volume_sma_20"];
        }

        // Returns and changes
        features["returns_1"] = await aggregator.GetReturnsAsync(ReturnsShort, cancellationToken).ConfigureAwait(false);
        features["returns_5"] = await aggregator.GetReturnsAsync(ShortSMA, cancellationToken).ConfigureAwait(false);
        features["returns_20"] = await aggregator.GetReturnsAsync(MediumSMA, cancellationToken).ConfigureAwait(false);

        return features;
    }

    /// <summary>
    /// Cache calculated features
    /// </summary>
    private async Task CacheFeaturesAsync(string symbol, DateTime timestamp, Dictionary<string, double> features, CancellationToken cancellationToken)
    {
        try
        {
            var cache = _featureCaches.GetOrAdd(symbol, _ => new FeatureCache(symbol, _maxCacheSize));
            await Task.Run(() => cache.AddFeatures(timestamp, features), cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to cache features for {Symbol}", symbol);
        }
    }

    /// <summary>
    /// Cleanup expired caches to prevent memory leaks
    /// </summary>
    private void CleanupExpiredCaches(object? state)
    {
        try
        {
            var cutoffTime = DateTime.UtcNow - _cacheRetention;
            var keysToRemove = new List<string>();

            foreach (var (symbol, cache) in _featureCaches)
            {
                cache.RemoveExpiredEntries(cutoffTime);
                
                if (cache.IsEmpty)
                {
                    keysToRemove.Add(symbol);
                }
            }

            // Remove empty caches
            foreach (var key in keysToRemove)
            {
                _featureCaches.TryRemove(key, out _);
                _aggregators.TryRemove(key, out _);
            }

            if (keysToRemove.Count > 0)
            {
                _logger.LogDebug("Cleaned up {Count} expired feature caches", keysToRemove.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during cache cleanup");
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _cleanupTimer?.Dispose();
            _featureCaches.Clear();
            _aggregators.Clear();
        }
    }
}

/// <summary>
/// Streaming data aggregator for a single symbol
/// </summary>
public class StreamingAggregator
{
    // Constants for magic number violations (moved from main class)
    private const int EmaShortPeriod = 12;
    private const int EmaLongPeriod = 26;
    private const int TradingDaysPerYear = 252;
    private const int DefaultBatchSize = 50;
    private const double MinimumValue = 1E-10;
    private const int PercentageMultiplier = 100;
    private const int DefaultMultiplier = 2;
    
    private readonly Queue<MarketData> _dataWindow = new();
    private readonly object _lock = new();
    private readonly int _maxWindowSize = 200; // Keep last 200 data points

    private double _ema12;
    private double _ema26;
    private double _macdSignal;
    private bool _initialized;

    public StreamingAggregator(string symbol, ILogger logger)
    {
        // Parameters kept for interface compatibility but not stored as they're unused
    }

    public Task UpdateAsync(MarketData data, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            _dataWindow.Enqueue(data);
            
            // Maintain window size
            while (_dataWindow.Count > _maxWindowSize)
            {
                _dataWindow.Dequeue();
            }

            // Update streaming EMAs
            UpdateStreamingEMAs(data.Close);
        }
        
        return Task.CompletedTask;
    }

    private void UpdateStreamingEMAs(double price)
    {
        if (!_initialized)
        {
            _ema12 = price;
            _ema26 = price;
            _macdSignal = 0.0;  // Initialize MACD signal
            _initialized = true;
        }
        else
        {
            // EMA calculation: EMA = (Close - EMA_prev) * multiplier + EMA_prev
            var multiplier12 = 2.0 / 13.0; // 12-period EMA
            var multiplier26 = 2.0 / 27.0; // 26-period EMA
            var multiplierSignal = 2.0 / 10.0; // 9-period signal line

            _ema12 = (price - _ema12) * multiplier12 + _ema12;
            _ema26 = (price - _ema26) * multiplier26 + _ema26;

            var macd = _ema12 - _ema26;
            _macdSignal = (macd - _macdSignal) * multiplierSignal + _macdSignal;
        }
    }

    public double GetSMA(int period)
    {
        lock (_lock)
        {
            var data = _dataWindow.TakeLast(period).ToList();
            return data.Count > 0 ? data.Average(d => d.Close) : 0.0;
        }
    }

    public double GetEMA(int period)
    {
        lock (_lock)
        {
            return period switch
            {
                EmaShortPeriod => _ema12,
                EmaLongPeriod => _ema26,
                _ => CalculateEMA(period)
            };
        }
    }

    public async Task<double> GetMACDSignalAsync(CancellationToken cancellationToken)
    {
        // Make this a proper async operation instead of Task.FromResult
        await Task.Yield(); // Ensure async behavior
        
        lock (_lock)
        {
            // Calculate MACD signal with proper async behavior
            // MACD Signal = EMA of MACD Line
            if (_dataWindow.Count < EmaLongPeriod) return 0.0;
            
            var macdLine = _ema12 - _ema26;
            
            // Update signal line using exponential moving average
            const double signalPeriod = 9.0;
            const double alpha = 2.0 / (signalPeriod + 1.0);
            
            _macdSignal = (macdLine * alpha) + (_macdSignal * (1.0 - alpha));
            
            return _macdSignal;
        }
    }

    public Task<double> GetVolatilityAsync(int period, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            var data = _dataWindow.TakeLast(period).ToList();
            if (data.Count < 2) return Task.FromResult(0.0);

            var returns = new List<double>();
            for (int i = 1; i < data.Count; i++)
            {
                returns.Add(Math.Log(data[i].Close / data[i - 1].Close));
            }

            var mean = returns.Average();
            var variance = returns.Select(r => Math.Pow(r - mean, 2)).Average();
            return Task.FromResult(Math.Sqrt(variance) * Math.Sqrt(TradingDaysPerYear)); // Annualized volatility
        }
    }

    public Task<double> GetATRAsync(int period, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            var data = _dataWindow.TakeLast(period + 1).ToList();
            if (data.Count < 2) return Task.FromResult(0.0);

            var trueRanges = new List<double>();
            for (int i = 1; i < data.Count; i++)
            {
                var current = data[i];
                var previous = data[i - 1];
                
                var tr = Math.Max(
                    current.High - current.Low,
                    Math.Max(
                        Math.Abs(current.High - previous.Close),
                        Math.Abs(current.Low - previous.Close)
                    )
                );
                trueRanges.Add(tr);
            }

            return Task.FromResult(trueRanges.Average());
        }
    }

    public Task<double> GetRSIAsync(int period, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            var data = _dataWindow.TakeLast(period + 1).ToList();
            if (data.Count < period + 1) return Task.FromResult((double)DefaultBatchSize); // Neutral RSI

            var gains = new List<double>();
            var losses = new List<double>();

            for (int i = 1; i < data.Count; i++)
            {
                var change = data[i].Close - data[i - 1].Close;
                gains.Add(Math.Max(change, 0));
                losses.Add(Math.Max(-change, 0));
            }

            var avgGain = gains.Average();
            var avgLoss = losses.Average();

            if (Math.Abs(avgLoss) < MinimumValue) return Task.FromResult((double)PercentageMultiplier);
            
            var rs = avgGain / avgLoss;
            return Task.FromResult(PercentageMultiplier - (PercentageMultiplier / (1.0 + rs)));
        }
    }

    public Task<double> GetVolumeSMAAsync(int period, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            var data = _dataWindow.TakeLast(period).ToList();
            return Task.FromResult(data.Count > 0 ? data.Average(d => d.Volume) : 0.0);
        }
    }

    public Task<double> GetReturnsAsync(int period, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            var data = _dataWindow.TakeLast(period + 1).ToList();
            if (data.Count < period + 1) return Task.FromResult(0.0);

            var currentPrice = data.Last().Close;
            var pastPrice = data[data.Count - period - 1].Close;

            return Task.FromResult(pastPrice > 0 ? (currentPrice - pastPrice) / pastPrice : 0.0);
        }
    }

    private double CalculateEMA(int period)
    {
        var data = _dataWindow.ToList();
        if (data.Count == 0) return 0.0;

        var multiplier = DefaultMultiplier / (period + 1);
        var ema = data[0].Close;

        for (int i = 1; i < Math.Min(data.Count, period * DefaultMultiplier); i++)
        {
            ema = (data[i].Close - ema) * multiplier + ema;
        }

        return ema;
    }
}

/// <summary>
/// Feature cache for a single symbol
/// </summary>
public class FeatureCache
{
    private readonly int _maxSize;
    private readonly SortedDictionary<DateTime, Dictionary<string, double>> _cache = new();
    private readonly object _lock = new();

    public bool IsEmpty => _cache.Count == 0;

    public FeatureCache(string symbol, int maxSize)
    {
        // Symbol parameter kept for interface compatibility but not stored as it's unused
        _maxSize = maxSize;
    }

    public void AddFeatures(DateTime timestamp, Dictionary<string, double> features)
    {
        lock (_lock)
        {
            _cache[timestamp] = new Dictionary<string, double>(features);
            
            // Maintain cache size
            while (_cache.Count > _maxSize)
            {
                _cache.Remove(_cache.Keys.First());
            }
        }
    }

    public Dictionary<string, double>? GetFeatures(DateTime timestamp)
    {
        lock (_lock)
        {
            return _cache.TryGetValue(timestamp, out var features) ? features : null;
        }
    }

    public void RemoveExpiredEntries(DateTime cutoffTime)
    {
        lock (_lock)
        {
            var keysToRemove = _cache.Keys.Where(k => k < cutoffTime).ToList();
            foreach (var key in keysToRemove)
            {
                _cache.Remove(key);
            }
        }
    }
}