using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Text.Json;

namespace TradingBot.RLAgent;
public class FeatureConfig
{
    public Dictionary<RegimeType, RegimeProfile> RegimeProfiles { get; } = new();
    public RegimeProfile DefaultProfile { get; set; } = new();
    public int MaxBufferSize { get; set; } = 1000;
    public int TopKFeatures { get; set; } = 10;
    
    // Feature engineering configuration
    public int DefaultRsiPeriod { get; set; } = 14;
    public int DefaultMovingAveragePeriod { get; set; } = 20;
    public double DefaultMomentumThreshold { get; set; } = 0.5;
    public int MaxFeatureHistoryPeriods { get; set; } = 26;
    public int DefaultLookbackPeriods { get; set; } = 10;
    public int FeatureImportanceTopCount { get; set; } = 5;
    
    // Default sentinel values for missing features
    public double DefaultVolatilitySentinel { get; set; } = 0.15;
    public double DefaultRsiSentinel { get; set; } = 0.5;
    public double DefaultBollingerSentinel { get; set; } = 0.5;
    
    // Streaming configuration (merged from StreamingFeatureAggregator)
    public int StreamingStaleThresholdSeconds { get; set; } = 30;
    public int StreamingCleanupAfterMinutes { get; set; } = 30;
    public Collection<TimeSpan> StreamingTimeWindows { get; } = new() 
    {
        TimeSpan.FromSeconds(10),
        TimeSpan.FromMinutes(1),
        TimeSpan.FromMinutes(5),
        TimeSpan.FromMinutes(15)
    };
    public TimeSpan MicrostructureWindow { get; set; } = TimeSpan.FromMinutes(1);
}

/// <summary>
/// Regime-specific feature configuration profile
/// </summary>
public class RegimeProfile
{
    public int VolatilityLookback { get; set; } = 20;
    public int TrendLookback { get; set; } = 50;
    public int VolumeLookback { get; set; } = 20;
    public int RsiLookback { get; set; } = 14;
    public int BollingerLookback { get; set; } = 20;
    public int AtrLookback { get; set; } = 14;
    public int MicrostructureLookback { get; set; } = 100;
    public int OrderFlowLookback { get; set; } = 50;
    public double TradeDirectionDecay { get; set; } = 0.9;
}

/// <summary>
/// Feature vector output
/// </summary>
public class FeatureVector
{
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public RegimeType Regime { get; set; }
    public DateTime Timestamp { get; set; }
    public IReadOnlyList<double> Features { get; set; } = Array.Empty<double>();
    public IReadOnlyList<string> FeatureNames { get; set; } = Array.Empty<string>();
    public int FeatureCount { get; set; }
    public bool HasMissingValues { get; set; }
}

/// <summary>
/// Feature state for forward-filling and null handling
/// </summary>
public class FeatureState
{
    public DateTime LastUpdate { get; set; }
    public int FeatureCount { get; set; }
    public Dictionary<string, double> LastValidValues { get; } = new();
}

/// <summary>
/// Feature importance tracking
/// </summary>
public class FeatureImportanceTracker
{
    private const int MaxImportanceHistorySize = 100;
    
    private readonly Dictionary<string, List<double>> _importanceHistory = new();
    private readonly object _lock = new();

    public void UpdateImportance(string[] featureNames, double[] importanceScores)
    {
        ArgumentNullException.ThrowIfNull(featureNames);
        ArgumentNullException.ThrowIfNull(importanceScores);
        
        lock (_lock)
        {
            for (int i = 0; i < Math.Min(featureNames.Length, importanceScores.Length); i++)
            {
                var featureName = featureNames[i];
                var importance = importanceScores[i];

                if (!_importanceHistory.TryGetValue(featureName, out var history))
                {
                    history = new List<double>();
                    _importanceHistory[featureName] = history;
                }

                history.Add(importance);

                // Keep only recent history
                if (_importanceHistory[featureName].Count > MaxImportanceHistorySize)
                {
                    _importanceHistory[featureName].RemoveAt(0);
                }
            }
        }
    }

    public Dictionary<string, double> GetTopKFeatures(int k)
    {
        lock (_lock)
        {
            return _importanceHistory
                .Where(kv => kv.Value.Count > 0)
                .ToDictionary(kv => kv.Key, kv => kv.Value.Average())
                .OrderByDescending(kv => kv.Value)
                .Take(k)
                .ToDictionary(kv => kv.Key, kv => kv.Value);
        }
    }
}

/// <summary>
/// Daily feature importance report
/// </summary>
public class FeatureImportanceReport
{
    public DateTime GeneratedAt { get; set; }
    public Dictionary<string, Dictionary<string, double>> SymbolReports { get; } = new();
}

/// <summary>
/// Market tick data structure (merged from StreamingFeatureAggregator)
/// </summary>
public class MarketTick
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public double Price { get; set; }
    public double Volume { get; set; }
    public double Bid { get; set; }
    public double Ask { get; set; }
    public double Size { get; set; }
    public bool IsBuyAggressor { get; set; }
}

/// <summary>
/// Streaming features output (merged from StreamingFeatureAggregator)
/// </summary>
public class StreamingFeatures
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public Dictionary<string, double> Features { get; } = new();
    public Dictionary<string, double> MicrostructureFeatures { get; } = new();
    public Dictionary<string, Dictionary<string, double>> TimeWindowFeatures { get; } = new();
    public bool IsStale { get; set; }
}

/// <summary>
/// Symbol-specific streaming aggregator (merged from StreamingFeatureAggregator)
/// </summary>
public class StreamingSymbolAggregator : IDisposable
{
    private readonly string _symbol;
    private readonly FeatureConfig _config;
    private readonly Dictionary<TimeSpan, TimeWindowAggregator> _windowAggregators = new();
    private readonly MicrostructureCalculator _microstructureCalc;
    private readonly object _lock = new();
    private StreamingFeatures _currentFeatures = new();
    private bool _disposed;

    public DateTime LastUpdateTime { get; private set; } = DateTime.UtcNow;

    public StreamingSymbolAggregator(string symbol, FeatureConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        
        _symbol = symbol;
        _config = config;
        _microstructureCalc = new MicrostructureCalculator(config.MicrostructureWindow);

        // Initialize window aggregators
        foreach (var window in config.StreamingTimeWindows)
        {
            _windowAggregators[window] = new TimeWindowAggregator(window);
        }
    }

    public async Task<StreamingFeatures> ProcessTickAsync(MarketTick tick, CancellationToken cancellationToken)
    {
        // Process tick data asynchronously
        await Task.Run(() =>
        {
            // Perform intensive calculations off the main thread if needed
        }, cancellationToken).ConfigureAwait(false);

        lock (_lock)
        {
            LastUpdateTime = DateTime.UtcNow;

            // Update microstructure features
            _microstructureCalc.AddTick(tick);

            // Update all time window aggregators
            foreach (var aggregator in _windowAggregators.Values)
            {
                aggregator.AddTick(tick);
            }

            // Calculate new features
            _currentFeatures = CalculateFeatures();
            return _currentFeatures;
        }
    }

    public StreamingFeatures CurrentFeatures
    {
        get
        {
            lock (_lock)
            {
                return _currentFeatures;
            }
        }
    }

    private StreamingFeatures CalculateFeatures()
    {
        var features = new StreamingFeatures
        {
            Symbol = _symbol,
            Timestamp = DateTime.UtcNow
        };
        
        // Populate the microstructure features
        var microFeatures = _microstructureCalc.GetFeatures();
        foreach (var kvp in microFeatures)
        {
            features.MicrostructureFeatures[kvp.Key] = kvp.Value;
        }

        // Add time window features using configured windows
        foreach (var kvp in _windowAggregators.Where(w => _config.StreamingTimeWindows.Contains(w.Key)))
        {
            features.TimeWindowFeatures[kvp.Key.ToString()] = kvp.Value.GetFeatures();
        }

        return features;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            foreach (var aggregator in _windowAggregators.Values)
            {
                aggregator.Dispose();
            }
            _windowAggregators.Clear();
            _microstructureCalc.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Time window aggregator for streaming features (merged from StreamingFeatureAggregator)
/// </summary>
public class TimeWindowAggregator : IDisposable
{
    private readonly TimeSpan _window;
    private readonly List<MarketTick> _ticks = new();
    private readonly object _lock = new();
    private bool _disposed;

    public TimeWindowAggregator(TimeSpan window)
    {
        _window = window;
    }

    public void AddTick(MarketTick tick)
    {
        ArgumentNullException.ThrowIfNull(tick);
        lock (_lock)
        {
            _ticks.Add(tick);
            CleanOldTicks(tick.Timestamp);
        }
    }

    public Dictionary<string, double> GetFeatures()
    {
        lock (_lock)
        {
            if (_ticks.Count == 0)
            {
                return new Dictionary<string, double>();
            }

            var prices = _ticks.Select(t => t.Price).ToArray();
            var volumes = _ticks.Select(t => t.Volume).ToArray();

            return new Dictionary<string, double>
            {
                ["vwap"] = CalculateVWAP(),
                ["volatility"] = CalculateVolatility(prices),
                ["volume_sum"] = volumes.Sum(),
                ["volume_avg"] = volumes.Average(),
                ["tick_count"] = _ticks.Count,
                ["price_range"] = prices.Max() - prices.Min(),
                ["last_price"] = _ticks.Count > 0 ? _ticks[_ticks.Count - 1].Price : 0,
                ["first_price"] = _ticks.Count > 0 ? _ticks[0].Price : 0
            };
        }
    }

    private void CleanOldTicks(DateTime currentTime)
    {
        var cutoff = currentTime - _window;
        _ticks.RemoveAll(t => t.Timestamp < cutoff);
    }

    private double CalculateVWAP()
    {
        var totalValue = _ticks.Sum(t => t.Price * t.Volume);
        var totalVolume = _ticks.Sum(t => t.Volume);
        return totalVolume > 0 ? totalValue / totalVolume : 0;
    }

    private static double CalculateVolatility(double[] prices)
    {
        if (prices.Length < 2) return 0;
        
        var returns = new List<double>();
        for (int i = 1; i < prices.Length; i++)
        {
            if (prices[i - 1] > 0)
            {
                returns.Add((prices[i] - prices[i - 1]) / prices[i - 1]);
            }
        }
        
        if (returns.Count == 0) return 0;
        
        var mean = returns.Average();
        var variance = returns.Select(r => Math.Pow(r - mean, 2)).Average();
        return Math.Sqrt(variance);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            lock (_lock)
            {
                _ticks.Clear();
            }
            _disposed = true;
        }
    }
}

/// <summary>
/// Microstructure feature calculator (merged from StreamingFeatureAggregator)
/// </summary>
public class MicrostructureCalculator : IDisposable
{
    private readonly TimeSpan _window;
    private readonly List<MarketTick> _ticks = new();
    private readonly object _lock = new();
    private bool _disposed;

    public MicrostructureCalculator(TimeSpan window)
    {
        _window = window;
    }

    public void AddTick(MarketTick tick)
    {
        ArgumentNullException.ThrowIfNull(tick);
        
        lock (_lock)
        {
            _ticks.Add(tick);
            CleanOldTicks(tick.Timestamp);
        }
    }

    public Dictionary<string, double> GetFeatures()
    {
        lock (_lock)
        {
            if (_ticks.Count == 0)
            {
                return new Dictionary<string, double>();
            }

            return new Dictionary<string, double>
            {
                ["spread_avg"] = CalculateAverageSpread(),
                ["spread_current"] = _ticks.Count > 0 ? _ticks[_ticks.Count - 1].Ask - _ticks[_ticks.Count - 1].Bid : 0,
                ["order_flow_imbalance"] = CalculateOrderFlowImbalance(),
                ["price_impact"] = CalculatePriceImpact(),
                ["tick_direction"] = GetLastTickDirection(),
                ["volume_imbalance"] = CalculateVolumeImbalance(),
                ["effective_spread"] = CalculateEffectiveSpread()
            };
        }
    }

    private void CleanOldTicks(DateTime currentTime)
    {
        var cutoff = currentTime - _window;
        _ticks.RemoveAll(t => t.Timestamp < cutoff);
    }

    private double CalculateAverageSpread()
    {
        var spreads = _ticks.Select(t => t.Ask - t.Bid).Where(s => s > 0).ToArray();
        return spreads.Length > 0 ? spreads.Average() : 0;
    }

    private double CalculateOrderFlowImbalance()
    {
        var buyVolume = _ticks.Where(t => t.IsBuyAggressor).Sum(t => t.Volume);
        var sellVolume = _ticks.Where(t => !t.IsBuyAggressor).Sum(t => t.Volume);
        var totalVolume = buyVolume + sellVolume;
        
        return totalVolume > 0 ? (buyVolume - sellVolume) / totalVolume : 0;
    }

    private double CalculatePriceImpact()
    {
        if (_ticks.Count < 2) return 0;
        
        var priceChanges = new List<double>();
        for (int i = 1; i < _ticks.Count; i++)
        {
            var priceChange = _ticks[i].Price - _ticks[i - 1].Price;
            var volumeWeight = _ticks[i].Volume;
            if (volumeWeight > 0)
            {
                priceChanges.Add(Math.Abs(priceChange) / volumeWeight);
            }
        }
        
        return priceChanges.Count > 0 ? priceChanges.Average() : 0;
    }

    private double GetLastTickDirection()
    {
        if (_ticks.Count < 2) return 0;
        
        var last = _ticks[_ticks.Count - 1];
        var previous = _ticks[_ticks.Count - 2];
        
        if (last.Price > previous.Price) 
            return 1;
        else if (last.Price < previous.Price) 
            return -1;
        else 
            return 0;
    }

    private double CalculateVolumeImbalance()
    {
        var bidVolume = _ticks.Sum(t => t.Volume * (t.Price <= (t.Bid + t.Ask) / 2 ? 1 : 0));
        var askVolume = _ticks.Sum(t => t.Volume * (t.Price > (t.Bid + t.Ask) / 2 ? 1 : 0));
        var totalVolume = bidVolume + askVolume;
        
        return totalVolume > 0 ? (askVolume - bidVolume) / totalVolume : 0;
    }

    private double CalculateEffectiveSpread()
    {
        if (_ticks.Count == 0) return 0;
        
        var effectiveSpreads = _ticks.Select(t =>
        {
            var midPrice = (t.Bid + t.Ask) / 2;
            return midPrice > 0 ? 2 * Math.Abs(t.Price - midPrice) / midPrice : 0;
        }).Where(s => s > 0).ToArray();
        
        return effectiveSpreads.Length > 0 ? effectiveSpreads.Average() : 0;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            lock (_lock)
            {
                _ticks.Clear();
            }
            _disposed = true;
        }
    }
}

