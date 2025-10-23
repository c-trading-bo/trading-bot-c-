using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Services;

/// <summary>
/// Bar aggregation service for multi-timeframe trading.
/// Subscribes to tick/trade data and builds 1-minute and 5-minute bars in real-time.
/// 
/// Phase 5: Live Inference Services (Week 5-6)
/// - Subscribe to tick feed from TopstepX adapter
/// - Build 1m and 5m bars in real-time
/// - Publish events when bars complete
/// - Cache last 100 bars for feature computation
/// 
/// Design principles:
/// - Real-time: Bars updated as ticks arrive
/// - Event-driven: Publishes bar completion events
/// - Thread-safe: Concurrent access from multiple threads
/// - Production-ready: Comprehensive error handling
/// </summary>
public class BarAggregationService : IDisposable
{
    private readonly ILogger<BarAggregationService> _logger;
    private readonly ConcurrentDictionary<string, SymbolBarAggregator> _aggregators = new();
    private bool _disposed;
    
    // Cache configuration
    internal const int MaxCachedBars1m = 100;
    internal const int MaxCachedBars5m = 100;
    
    public event EventHandler<BarCompletedEventArgs>? Bar1mCompleted;
    public event EventHandler<BarCompletedEventArgs>? Bar5mCompleted;
    
    public BarAggregationService(ILogger<BarAggregationService> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }
    
    /// <summary>
    /// Process incoming tick data and update bars.
    /// Called for each tick/trade received from market data feed.
    /// </summary>
    /// <param name="symbol">Symbol (e.g., "ES", "NQ")</param>
    /// <param name="tick">Tick data</param>
    public void OnTick(string symbol, TickData tick)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            _logger.LogWarning("[BAR_AGG] Received tick with null or empty symbol");
            return;
        }
        
        if (tick == null)
        {
            _logger.LogWarning("[BAR_AGG] Received null tick for {Symbol}", symbol);
            return;
        }
        
        try
        {
            var aggregator = _aggregators.GetOrAdd(symbol, s => new SymbolBarAggregator(s, _logger));
            
            // Update 1m and 5m bars
            var (completed1m, completed5m) = aggregator.ProcessTick(tick);
            
            // Publish bar completion events
            if (completed1m != null)
            {
                Bar1mCompleted?.Invoke(this, new BarCompletedEventArgs 
                { 
                    Symbol = symbol, 
                    Bar = completed1m,
                    Timeframe = "1m"
                });
                
                _logger.LogDebug(
                    "[BAR_AGG] 1m bar completed for {Symbol} at {Timestamp}: C={Close}",
                    symbol, completed1m.Timestamp, completed1m.Close);
            }
            
            if (completed5m != null)
            {
                Bar5mCompleted?.Invoke(this, new BarCompletedEventArgs 
                { 
                    Symbol = symbol, 
                    Bar = completed5m,
                    Timeframe = "5m"
                });
                
                _logger.LogInformation(
                    "[BAR_AGG] 5m bar completed for {Symbol} at {Timestamp}: O={Open} H={High} L={Low} C={Close} V={Volume}",
                    symbol, completed5m.Timestamp, completed5m.Open, completed5m.High, 
                    completed5m.Low, completed5m.Close, completed5m.Volume);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BAR_AGG] Error processing tick for {Symbol}", symbol);
        }
    }
    
    /// <summary>
    /// Get cached 1-minute bars for a symbol (for feature computation).
    /// </summary>
    /// <param name="symbol">Symbol</param>
    /// <param name="count">Number of bars to retrieve (default: all cached)</param>
    /// <returns>List of 1m bars, ordered chronologically</returns>
    public List<AggregatedBar> GetCached1mBars(string symbol, int? count = null)
    {
        if (!_aggregators.TryGetValue(symbol, out var aggregator))
        {
            return new List<AggregatedBar>();
        }
        
        var bars = aggregator.GetCached1mBars();
        
        if (count.HasValue && count.Value > 0)
        {
            bars = bars.TakeLast(count.Value).ToList();
        }
        
        return bars;
    }
    
    /// <summary>
    /// Get cached 5-minute bars for a symbol (for feature computation).
    /// </summary>
    /// <param name="symbol">Symbol</param>
    /// <param name="count">Number of bars to retrieve (default: all cached)</param>
    /// <returns>List of 5m bars, ordered chronologically</returns>
    public List<AggregatedBar> GetCached5mBars(string symbol, int? count = null)
    {
        if (!_aggregators.TryGetValue(symbol, out var aggregator))
        {
            return new List<AggregatedBar>();
        }
        
        var bars = aggregator.GetCached5mBars();
        
        if (count.HasValue && count.Value > 0)
        {
            bars = bars.TakeLast(count.Value).ToList();
        }
        
        return bars;
    }
    
    /// <summary>
    /// Get current in-progress bar for a symbol and timeframe.
    /// </summary>
    public AggregatedBar? GetCurrentBar(string symbol, string timeframe)
    {
        if (!_aggregators.TryGetValue(symbol, out var aggregator))
        {
            return null;
        }
        
        return timeframe switch
        {
            "1m" => aggregator.GetCurrent1mBar(),
            "5m" => aggregator.GetCurrent5mBar(),
            _ => null
        };
    }
    
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        
        _disposed = true;
        _aggregators.Clear();
    }
}

/// <summary>
/// Per-symbol bar aggregator that manages 1m and 5m bar building.
/// </summary>
internal class SymbolBarAggregator
{
    private readonly string _symbol;
    private readonly ILogger _logger;
    private readonly object _lock = new();
    
    // Current bars being built
    private AggregatedBar? _current1mBar;
    private AggregatedBar? _current5mBar;
    
    // Cached completed bars
    private readonly List<AggregatedBar> _cached1mBars = new();
    private readonly List<AggregatedBar> _cached5mBars = new();
    
    public SymbolBarAggregator(string symbol, ILogger logger)
    {
        _symbol = symbol;
        _logger = logger;
    }
    
    /// <summary>
    /// Process tick and update bars. Returns completed bars if any.
    /// </summary>
    public (AggregatedBar? completed1m, AggregatedBar? completed5m) ProcessTick(TickData tick)
    {
        lock (_lock)
        {
            var timestamp = tick.Timestamp;
            AggregatedBar? completed1m = null;
            AggregatedBar? completed5m = null;
            
            // Determine bar boundaries
            var bar1mStart = GetBarStartTime(timestamp, 1);
            var bar5mStart = GetBarStartTime(timestamp, 5);
            
            // Process 1m bar
            if (_current1mBar == null || _current1mBar.Timestamp != bar1mStart)
            {
                // Bar boundary crossed - complete previous bar
                if (_current1mBar != null)
                {
                    completed1m = _current1mBar;
                    CacheBar(_cached1mBars, completed1m, BarAggregationService.MaxCachedBars1m);
                }
                
                // Start new 1m bar
                _current1mBar = new AggregatedBar
                {
                    Symbol = _symbol,
                    Timestamp = bar1mStart,
                    Open = tick.Price,
                    High = tick.Price,
                    Low = tick.Price,
                    Close = tick.Price,
                    Volume = tick.Size
                };
            }
            else
            {
                // Update current 1m bar
                _current1mBar.High = Math.Max(_current1mBar.High, tick.Price);
                _current1mBar.Low = Math.Min(_current1mBar.Low, tick.Price);
                _current1mBar.Close = tick.Price;
                _current1mBar.Volume += tick.Size;
            }
            
            // Process 5m bar
            if (_current5mBar == null || _current5mBar.Timestamp != bar5mStart)
            {
                // Bar boundary crossed - complete previous bar
                if (_current5mBar != null)
                {
                    completed5m = _current5mBar;
                    CacheBar(_cached5mBars, completed5m, BarAggregationService.MaxCachedBars5m);
                }
                
                // Start new 5m bar
                _current5mBar = new AggregatedBar
                {
                    Symbol = _symbol,
                    Timestamp = bar5mStart,
                    Open = tick.Price,
                    High = tick.Price,
                    Low = tick.Price,
                    Close = tick.Price,
                    Volume = tick.Size
                };
            }
            else
            {
                // Update current 5m bar
                _current5mBar.High = Math.Max(_current5mBar.High, tick.Price);
                _current5mBar.Low = Math.Min(_current5mBar.Low, tick.Price);
                _current5mBar.Close = tick.Price;
                _current5mBar.Volume += tick.Size;
            }
            
            return (completed1m, completed5m);
        }
    }
    
    public List<AggregatedBar> GetCached1mBars()
    {
        lock (_lock)
        {
            return new List<AggregatedBar>(_cached1mBars);
        }
    }
    
    public List<AggregatedBar> GetCached5mBars()
    {
        lock (_lock)
        {
            return new List<AggregatedBar>(_cached5mBars);
        }
    }
    
    public AggregatedBar? GetCurrent1mBar()
    {
        lock (_lock)
        {
            return _current1mBar;
        }
    }
    
    public AggregatedBar? GetCurrent5mBar()
    {
        lock (_lock)
        {
            return _current5mBar;
        }
    }
    
    /// <summary>
    /// Get bar start time for a given timestamp and interval.
    /// </summary>
    private static DateTimeOffset GetBarStartTime(DateTimeOffset timestamp, int intervalMinutes)
    {
        var totalMinutes = timestamp.Hour * 60 + timestamp.Minute;
        var barMinutes = (totalMinutes / intervalMinutes) * intervalMinutes;
        
        return new DateTimeOffset(
            timestamp.Year,
            timestamp.Month,
            timestamp.Day,
            barMinutes / 60,
            barMinutes % 60,
            0,
            timestamp.Offset);
    }
    
    /// <summary>
    /// Add bar to cache and maintain max size.
    /// </summary>
    private static void CacheBar(List<AggregatedBar> cache, AggregatedBar bar, int maxSize)
    {
        cache.Add(bar);
        
        // Remove oldest bars if cache exceeds max size
        while (cache.Count > maxSize)
        {
            cache.RemoveAt(0);
        }
    }
}

/// <summary>
/// Tick/trade data from market feed.
/// </summary>
public class TickData
{
    public DateTimeOffset Timestamp { get; set; }
    public double Price { get; set; }
    public double Size { get; set; }
}

/// <summary>
/// Aggregated OHLCV bar.
/// </summary>
public class AggregatedBar
{
    public string Symbol { get; set; } = string.Empty;
    public DateTimeOffset Timestamp { get; set; }
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double Volume { get; set; }
}

/// <summary>
/// Event args for bar completion.
/// </summary>
public class BarCompletedEventArgs : EventArgs
{
    public string Symbol { get; set; } = string.Empty;
    public AggregatedBar Bar { get; set; } = new();
    public string Timeframe { get; set; } = string.Empty;
}
