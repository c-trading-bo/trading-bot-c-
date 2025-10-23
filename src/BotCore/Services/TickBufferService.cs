using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Services;

/// <summary>
/// Tick buffer service for ultra-fast tick-level feature computation.
/// Maintains a rolling window of recent ticks for microstructure analysis.
/// 
/// Phase 5: Live Inference Services (Week 5-6) - Advanced Feature
/// - Maintain rolling 10-second tick window
/// - Compute tick features on demand
/// - Ultra-fast access (&lt;1ms)
/// - Thread-safe tick storage
/// 
/// Design principles:
/// - Ultra-low latency: &lt;1ms access time
/// - Fixed memory footprint: 10-second rolling window
/// - Thread-safe: Concurrent tick ingestion
/// - Production-ready: Comprehensive error handling
/// </summary>
public class TickBufferService : IDisposable
{
    private readonly ILogger<TickBufferService> _logger;
    private readonly ConcurrentDictionary<string, SymbolTickBuffer> _buffers = new();
    private bool _disposed;
    
    // Buffer configuration
    internal const int BufferWindowSeconds = 10;
    internal const int MaxTicksPerSymbol = 10000; // Safety limit
    
    public TickBufferService(ILogger<TickBufferService> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }
    
    /// <summary>
    /// Add tick to buffer for a symbol.
    /// </summary>
    /// <param name="symbol">Symbol (e.g., "ES", "NQ")</param>
    /// <param name="tick">Tick data</param>
    public void AddTick(string symbol, TickData tick)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            _logger.LogWarning("[TICK_BUFFER] Received tick with null or empty symbol");
            return;
        }
        
        if (tick == null)
        {
            _logger.LogWarning("[TICK_BUFFER] Received null tick for {Symbol}", symbol);
            return;
        }
        
        try
        {
            var buffer = _buffers.GetOrAdd(symbol, s => new SymbolTickBuffer(s, _logger));
            buffer.AddTick(tick);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TICK_BUFFER] Error adding tick for {Symbol}", symbol);
        }
    }
    
    /// <summary>
    /// Get ticks from the last N seconds.
    /// </summary>
    /// <param name="symbol">Symbol</param>
    /// <param name="seconds">Number of seconds to look back (default: 10)</param>
    /// <returns>List of ticks in chronological order</returns>
    public List<TickData> GetRecentTicks(string symbol, int seconds = BufferWindowSeconds)
    {
        if (!_buffers.TryGetValue(symbol, out var buffer))
        {
            return new List<TickData>();
        }
        
        return buffer.GetRecentTicks(seconds);
    }
    
    /// <summary>
    /// Compute tick-level features for execution approval.
    /// </summary>
    /// <param name="symbol">Symbol</param>
    /// <returns>Tick features dictionary</returns>
    public Dictionary<string, double> ComputeTickFeatures(string symbol)
    {
        var features = new Dictionary<string, double>();
        
        if (!_buffers.TryGetValue(symbol, out var buffer))
        {
            _logger.LogWarning("[TICK_BUFFER] No tick data available for {Symbol}", symbol);
            return features;
        }
        
        try
        {
            var ticks = buffer.GetRecentTicks(BufferWindowSeconds);
            
            if (ticks.Count < 2)
            {
                _logger.LogDebug("[TICK_BUFFER] Insufficient ticks for {Symbol}: {Count}", symbol, ticks.Count);
                return features;
            }
            
            // Spread (bid-ask spread proxy using price changes)
            var spread = ComputeSpread(ticks);
            features["spread_bps"] = spread;
            
            // Order flow imbalance (buying vs selling pressure)
            var imbalance = ComputeOrderFlowImbalance(ticks);
            features["order_flow_imbalance"] = imbalance;
            
            // Tick intensity (ticks per second)
            var intensity = ComputeTickIntensity(ticks);
            features["tick_intensity"] = intensity;
            
            // Price momentum (short-term direction)
            var momentum = ComputePriceMomentum(ticks);
            features["price_momentum"] = momentum;
            
            // Volatility (tick-level price changes)
            var volatility = ComputeTickVolatility(ticks);
            features["tick_volatility"] = volatility;
            
            _logger.LogDebug(
                "[TICK_BUFFER] Computed {FeatureCount} tick features for {Symbol}",
                features.Count, symbol);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TICK_BUFFER] Error computing tick features for {Symbol}", symbol);
        }
        
        return features;
    }
    
    /// <summary>
    /// Get tick count for a symbol in the buffer.
    /// </summary>
    public int GetTickCount(string symbol)
    {
        if (!_buffers.TryGetValue(symbol, out var buffer))
        {
            return 0;
        }
        
        return buffer.GetTickCount();
    }
    
    #region Feature Calculations
    
    /// <summary>
    /// Compute spread in basis points (proxy using price volatility).
    /// </summary>
    private static double ComputeSpread(List<TickData> ticks)
    {
        if (ticks.Count < 2)
        {
            return 0.0;
        }
        
        // Use price range as spread proxy
        var prices = ticks.Select(t => t.Price).ToList();
        var minPrice = prices.Min();
        var maxPrice = prices.Max();
        var avgPrice = prices.Average();
        
        if (avgPrice < double.Epsilon)
        {
            return 0.0;
        }
        
        var spread = ((maxPrice - minPrice) / avgPrice) * 10000.0; // Basis points
        return spread;
    }
    
    /// <summary>
    /// Compute order flow imbalance (buying vs selling pressure).
    /// Positive = more buying, Negative = more selling.
    /// </summary>
    private static double ComputeOrderFlowImbalance(List<TickData> ticks)
    {
        if (ticks.Count < 2)
        {
            return 0.0;
        }
        
        double buyVolume = 0.0;
        double sellVolume = 0.0;
        
        for (int i = 1; i < ticks.Count; i++)
        {
            var priceChange = ticks[i].Price - ticks[i - 1].Price;
            
            if (priceChange > 0)
            {
                buyVolume += ticks[i].Size;
            }
            else if (priceChange < 0)
            {
                sellVolume += ticks[i].Size;
            }
        }
        
        var totalVolume = buyVolume + sellVolume;
        if (totalVolume < double.Epsilon)
        {
            return 0.0;
        }
        
        return (buyVolume - sellVolume) / totalVolume; // Range: [-1, 1]
    }
    
    /// <summary>
    /// Compute tick intensity (ticks per second).
    /// </summary>
    private static double ComputeTickIntensity(List<TickData> ticks)
    {
        if (ticks.Count < 2)
        {
            return 0.0;
        }
        
        var timeSpan = (ticks.Last().Timestamp - ticks.First().Timestamp).TotalSeconds;
        if (timeSpan < double.Epsilon)
        {
            return ticks.Count;
        }
        
        return ticks.Count / timeSpan;
    }
    
    /// <summary>
    /// Compute price momentum (weighted average of recent price changes).
    /// </summary>
    private static double ComputePriceMomentum(List<TickData> ticks)
    {
        if (ticks.Count < 2)
        {
            return 0.0;
        }
        
        double momentum = 0.0;
        double weightSum = 0.0;
        
        for (int i = 1; i < ticks.Count; i++)
        {
            var priceChange = ticks[i].Price - ticks[i - 1].Price;
            var weight = i; // More recent ticks have higher weight
            
            momentum += priceChange * weight;
            weightSum += weight;
        }
        
        if (weightSum < double.Epsilon)
        {
            return 0.0;
        }
        
        return momentum / weightSum;
    }
    
    /// <summary>
    /// Compute tick-level volatility (standard deviation of price changes).
    /// </summary>
    private static double ComputeTickVolatility(List<TickData> ticks)
    {
        if (ticks.Count < 2)
        {
            return 0.0;
        }
        
        var priceChanges = new List<double>();
        for (int i = 1; i < ticks.Count; i++)
        {
            priceChanges.Add(ticks[i].Price - ticks[i - 1].Price);
        }
        
        if (priceChanges.Count == 0)
        {
            return 0.0;
        }
        
        var mean = priceChanges.Average();
        var variance = priceChanges.Sum(x => Math.Pow(x - mean, 2)) / priceChanges.Count;
        
        return Math.Sqrt(variance);
    }
    
    #endregion
    
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        
        _disposed = true;
        _buffers.Clear();
    }
}

/// <summary>
/// Per-symbol tick buffer with automatic cleanup.
/// </summary>
internal class SymbolTickBuffer
{
    private readonly string _symbol;
    private readonly ILogger _logger;
    private readonly object _lock = new();
    private readonly List<TickData> _ticks = new();
    
    public SymbolTickBuffer(string symbol, ILogger logger)
    {
        _symbol = symbol;
        _logger = logger;
    }
    
    public void AddTick(TickData tick)
    {
        lock (_lock)
        {
            _ticks.Add(tick);
            
            // Cleanup old ticks (older than 10 seconds)
            var cutoff = DateTimeOffset.UtcNow.AddSeconds(-TickBufferService.BufferWindowSeconds);
            _ticks.RemoveAll(t => t.Timestamp < cutoff);
            
            // Safety limit to prevent memory growth
            if (_ticks.Count > TickBufferService.MaxTicksPerSymbol)
            {
                var excess = _ticks.Count - TickBufferService.MaxTicksPerSymbol;
                _ticks.RemoveRange(0, excess);
                
                _logger.LogWarning(
                    "[TICK_BUFFER] Buffer overflow for {Symbol}, removed {Count} oldest ticks",
                    _symbol, excess);
            }
        }
    }
    
    public List<TickData> GetRecentTicks(int seconds)
    {
        lock (_lock)
        {
            var cutoff = DateTimeOffset.UtcNow.AddSeconds(-seconds);
            return _ticks.Where(t => t.Timestamp >= cutoff).OrderBy(t => t.Timestamp).ToList();
        }
    }
    
    public int GetTickCount()
    {
        lock (_lock)
        {
            return _ticks.Count;
        }
    }
}
