using Microsoft.Extensions.Logging;
using BotCore.ML;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Services;

/// <summary>
/// Live multi-timeframe feature computer for real-time trading.
/// Listens for bar completion events and computes synchronized features.
/// 
/// Phase 5: Live Inference Services (Week 5-6)
/// - Listen for bar completion events
/// - Compute 5m and 1m features when bars complete
/// - Cache features in memory
/// - Use EXACT same feature code as training
/// 
/// Design principles:
/// - Real-time: Features computed within 100ms of bar completion
/// - Deterministic: Same feature calculation as training
/// - Thread-safe: Concurrent access from multiple threads
/// - Production-ready: Comprehensive error handling and validation
/// </summary>
public class LiveMultiTimeframeFeatureComputer
{
    private readonly ILogger<LiveMultiTimeframeFeatureComputer> _logger;
    private readonly MultiTimeframeFeatureExtractor _featureExtractor;
    private readonly BarAggregationService _barAggregator;
    
    // Feature cache (symbol -> latest features)
    private readonly object _cacheLock = new();
    private readonly Dictionary<string, CachedFeatures> _featureCache = new();
    
    // Performance tracking
    private readonly object _perfLock = new();
    private double _avgComputeTimeMs = 0.0;
    private int _computeCount = 0;
    
    private const double PerformanceAlpha = 0.1; // Exponential moving average weight
    private const double WarningThresholdMs = 100.0; // Warn if computation takes > 100ms
    
    public LiveMultiTimeframeFeatureComputer(
        ILogger<LiveMultiTimeframeFeatureComputer> logger,
        MultiTimeframeFeatureExtractor featureExtractor,
        BarAggregationService barAggregator)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _featureExtractor = featureExtractor ?? throw new ArgumentNullException(nameof(featureExtractor));
        _barAggregator = barAggregator ?? throw new ArgumentNullException(nameof(barAggregator));
        
        // Subscribe to bar completion events
        _barAggregator.Bar1mCompleted += OnBar1mCompleted;
        _barAggregator.Bar5mCompleted += OnBar5mCompleted;
        
        _logger.LogInformation("[FEATURE_COMPUTER] Initialized and subscribed to bar completion events");
    }
    
    /// <summary>
    /// Get latest synchronized features for a symbol.
    /// Returns null if features not yet available.
    /// </summary>
    /// <param name="symbol">Symbol (e.g., "ES", "NQ")</param>
    /// <returns>Latest feature dictionary or null</returns>
    public Dictionary<string, double>? GetLatestFeatures(string symbol)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            return null;
        }
        
        lock (_cacheLock)
        {
            if (_featureCache.TryGetValue(symbol, out var cached))
            {
                return new Dictionary<string, double>(cached.Features);
            }
        }
        
        return null;
    }
    
    /// <summary>
    /// Get feature computation metadata (timestamp, latency, etc.).
    /// </summary>
    public FeatureMetadata? GetFeatureMetadata(string symbol)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            return null;
        }
        
        lock (_cacheLock)
        {
            if (_featureCache.TryGetValue(symbol, out var cached))
            {
                return cached.Metadata;
            }
        }
        
        return null;
    }
    
    /// <summary>
    /// Get average feature computation time in milliseconds.
    /// </summary>
    public double GetAverageComputeTimeMs()
    {
        lock (_perfLock)
        {
            return _avgComputeTimeMs;
        }
    }
    
    /// <summary>
    /// Handle 1-minute bar completion event.
    /// Updates 1m features only (5m features updated on 5m bar completion).
    /// </summary>
    private void OnBar1mCompleted(object? sender, BarCompletedEventArgs e)
    {
        try
        {
            var startTime = DateTimeOffset.UtcNow;
            
            // Get recent 1m bars for feature computation
            var bars1m = _barAggregator.GetCached1mBars(e.Symbol);
            
            if (bars1m.Count == 0)
            {
                _logger.LogWarning(
                    "[FEATURE_COMPUTER] No 1m bars cached for {Symbol}, cannot compute features",
                    e.Symbol);
                return;
            }
            
            // Convert to BarData format
            var barData1m = bars1m.Select(b => new BarData
            {
                Timestamp = b.Timestamp,
                Open = b.Open,
                High = b.High,
                Low = b.Low,
                Close = b.Close,
                Volume = b.Volume
            }).ToList();
            
            // Extract 1m features
            var features1m = _featureExtractor.Extract1mFeatures(barData1m);
            
            // Update cache (merge with existing 5m features if available)
            lock (_cacheLock)
            {
                if (_featureCache.TryGetValue(e.Symbol, out var existing))
                {
                    // Merge 1m features with existing 5m features
                    foreach (var kvp in features1m)
                    {
                        existing.Features[kvp.Key] = kvp.Value;
                    }
                    
                    existing.Metadata.Timestamp1m = e.Bar.Timestamp;
                    existing.Metadata.LastUpdate = DateTimeOffset.UtcNow;
                }
                else
                {
                    // First time - create cache entry with only 1m features
                    _featureCache[e.Symbol] = new CachedFeatures
                    {
                        Features = features1m,
                        Metadata = new FeatureMetadata
                        {
                            Symbol = e.Symbol,
                            Timestamp1m = e.Bar.Timestamp,
                            LastUpdate = DateTimeOffset.UtcNow
                        }
                    };
                }
            }
            
            var elapsedMs = (DateTimeOffset.UtcNow - startTime).TotalMilliseconds;
            UpdatePerformanceMetrics(elapsedMs);
            
            _logger.LogDebug(
                "[FEATURE_COMPUTER] Computed 1m features for {Symbol}: {FeatureCount} features in {ElapsedMs:F2}ms",
                e.Symbol, features1m.Count, elapsedMs);
            
            if (elapsedMs > WarningThresholdMs)
            {
                _logger.LogWarning(
                    "[FEATURE_COMPUTER] Slow feature computation for {Symbol}: {ElapsedMs:F2}ms (threshold: {Threshold}ms)",
                    e.Symbol, elapsedMs, WarningThresholdMs);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_COMPUTER] Error computing 1m features for {Symbol}", e.Symbol);
        }
    }
    
    /// <summary>
    /// Handle 5-minute bar completion event.
    /// Recomputes both 5m and 1m features for full synchronization.
    /// </summary>
    private void OnBar5mCompleted(object? sender, BarCompletedEventArgs e)
    {
        try
        {
            var startTime = DateTimeOffset.UtcNow;
            
            // Get recent bars for both timeframes
            var bars5m = _barAggregator.GetCached5mBars(e.Symbol);
            var bars1m = _barAggregator.GetCached1mBars(e.Symbol);
            
            if (bars5m.Count == 0 || bars1m.Count == 0)
            {
                _logger.LogWarning(
                    "[FEATURE_COMPUTER] Insufficient bars cached for {Symbol}: {Count5m} 5m, {Count1m} 1m",
                    e.Symbol, bars5m.Count, bars1m.Count);
                return;
            }
            
            // Convert to BarData format
            var barData5m = bars5m.Select(b => new BarData
            {
                Timestamp = b.Timestamp,
                Open = b.Open,
                High = b.High,
                Low = b.Low,
                Close = b.Close,
                Volume = b.Volume
            }).ToList();
            
            var barData1m = bars1m.Select(b => new BarData
            {
                Timestamp = b.Timestamp,
                Open = b.Open,
                High = b.High,
                Low = b.Low,
                Close = b.Close,
                Volume = b.Volume
            }).ToList();
            
            // Compute synchronized features (EXACT same code as training)
            var synchronizedFeatures = _featureExtractor.SynchronizeFeatures(
                e.Bar.Timestamp,
                barData5m,
                barData1m);
            
            // Update cache
            lock (_cacheLock)
            {
                _featureCache[e.Symbol] = new CachedFeatures
                {
                    Features = synchronizedFeatures,
                    Metadata = new FeatureMetadata
                    {
                        Symbol = e.Symbol,
                        Timestamp5m = e.Bar.Timestamp,
                        Timestamp1m = bars1m.Last().Timestamp,
                        LastUpdate = DateTimeOffset.UtcNow,
                        FeatureCount = synchronizedFeatures.Count
                    }
                };
            }
            
            var elapsedMs = (DateTimeOffset.UtcNow - startTime).TotalMilliseconds;
            UpdatePerformanceMetrics(elapsedMs);
            
            _logger.LogInformation(
                "[FEATURE_COMPUTER] Computed synchronized features for {Symbol}: {FeatureCount} features in {ElapsedMs:F2}ms",
                e.Symbol, synchronizedFeatures.Count, elapsedMs);
            
            if (elapsedMs > WarningThresholdMs)
            {
                _logger.LogWarning(
                    "[FEATURE_COMPUTER] Slow feature computation for {Symbol}: {ElapsedMs:F2}ms (threshold: {Threshold}ms)",
                    e.Symbol, elapsedMs, WarningThresholdMs);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_COMPUTER] Error computing 5m features for {Symbol}", e.Symbol);
        }
    }
    
    /// <summary>
    /// Update performance metrics with exponential moving average.
    /// </summary>
    private void UpdatePerformanceMetrics(double elapsedMs)
    {
        lock (_perfLock)
        {
            if (_computeCount == 0)
            {
                _avgComputeTimeMs = elapsedMs;
            }
            else
            {
                _avgComputeTimeMs = (PerformanceAlpha * elapsedMs) + ((1.0 - PerformanceAlpha) * _avgComputeTimeMs);
            }
            
            _computeCount++;
        }
    }
}

/// <summary>
/// Cached features with metadata.
/// </summary>
internal class CachedFeatures
{
    public Dictionary<string, double> Features { get; set; } = new();
    public FeatureMetadata Metadata { get; set; } = new();
}

/// <summary>
/// Feature computation metadata.
/// </summary>
public class FeatureMetadata
{
    public string Symbol { get; set; } = string.Empty;
    public DateTimeOffset? Timestamp5m { get; set; }
    public DateTimeOffset? Timestamp1m { get; set; }
    public DateTimeOffset LastUpdate { get; set; }
    public int FeatureCount { get; set; }
}
