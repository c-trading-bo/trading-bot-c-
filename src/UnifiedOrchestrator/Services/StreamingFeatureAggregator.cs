using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    public class StreamingOptions
    {
        public List<TimeSpan> TimeWindows { get; set; } = new() { TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(5) };
        public int MicrostructureWindow { get; set; } = 100;
        public int StaleThresholdSeconds { get; set; } = 30;
        public int MaxCacheSize { get; set; } = 10000;
    }

    public class MarketTick
    {
        public string Symbol { get; set; } = "";
        public DateTime Timestamp { get; set; }
        public double Price { get; set; }
        public int Volume { get; set; }
        public bool IsBuyInitiated { get; set; }
        public double Bid { get; set; }
        public double Ask { get; set; }
    }

    public class MicrostructureFeatures
    {
        public double OrderFlow { get; set; }
        public double VolumeWeightedSpread { get; set; }
        public double TickImbalance { get; set; }
        public double MidPriceChange { get; set; }
    }

    public class TimeWindowFeatures
    {
        public double VWAP { get; set; }
        public double Volatility { get; set; }
        public double VolumeProfile { get; set; }
        public double TrendStrength { get; set; }
        public int TickCount { get; set; }
    }

    public class StreamingFeatures
    {
        public string Symbol { get; set; } = "";
        public DateTime GeneratedAt { get; set; }
        public MicrostructureFeatures MicrostructureFeatures { get; set; } = new();
        public Dictionary<TimeSpan, TimeWindowFeatures> TimeWindowFeatures { get; set; } = new();
    }

    public class StreamingFeatureAggregator : IDisposable
    {
        private readonly ILogger<StreamingFeatureAggregator> _logger;
        private readonly StreamingOptions _options;
        private readonly ConcurrentDictionary<string, List<MarketTick>> _tickCache = new();
        private readonly ConcurrentDictionary<string, DateTime> _lastUpdateTime = new();
        private readonly Timer _cleanupTimer;

        public StreamingFeatureAggregator(ILogger<StreamingFeatureAggregator> logger, IOptions<StreamingOptions> options)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            
            // Setup cleanup timer to run every minute
            _cleanupTimer = new Timer(CleanupStaleData, null, TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(1));
        }

        public async Task<StreamingFeatures> ProcessTickAsync(MarketTick tick, CancellationToken cancellationToken)
        {
            try
            {
                // Add tick to cache
                var ticks = _tickCache.GetOrAdd(tick.Symbol, _ => new List<MarketTick>());
                
                lock (ticks)
                {
                    ticks.Add(tick);
                    
                    // Keep only recent ticks within the largest time window plus buffer
                    var maxWindow = _options.TimeWindows.Max();
                    var cutoffTime = tick.Timestamp - maxWindow - TimeSpan.FromMinutes(5);
                    ticks.RemoveAll(t => t.Timestamp < cutoffTime);
                    
                    // Limit cache size
                    if (ticks.Count > _options.MaxCacheSize)
                    {
                        ticks.RemoveRange(0, ticks.Count - _options.MaxCacheSize);
                    }
                }

                _lastUpdateTime[tick.Symbol] = DateTime.UtcNow;

                // Generate features
                var features = await GenerateFeaturesAsync(tick.Symbol, tick.Timestamp, cancellationToken);
                
                _logger.LogDebug("Generated features for {Symbol} at {Timestamp}", tick.Symbol, tick.Timestamp);
                
                return features;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to process tick for {Symbol}", tick.Symbol);
                throw;
            }
        }

        public StreamingFeatures? GetCachedFeatures(string symbol)
        {
            if (!_tickCache.TryGetValue(symbol, out var ticks) || ticks.Count == 0)
                return null;

            MarketTick? latestTick;
            lock (ticks)
            {
                latestTick = ticks.LastOrDefault();
            }

            if (latestTick == null)
                return null;

            return GenerateFeaturesAsync(symbol, latestTick.Timestamp, CancellationToken.None).Result;
        }

        public bool HasStaleFeatures()
        {
            var staleThreshold = DateTime.UtcNow - TimeSpan.FromSeconds(_options.StaleThresholdSeconds);
            return _lastUpdateTime.Values.Any(updateTime => updateTime < staleThreshold);
        }

        public List<string> GetStaleSymbols()
        {
            var staleThreshold = DateTime.UtcNow - TimeSpan.FromSeconds(_options.StaleThresholdSeconds);
            return _lastUpdateTime
                .Where(kvp => kvp.Value < staleThreshold)
                .Select(kvp => kvp.Key)
                .ToList();
        }

        private async Task<StreamingFeatures> GenerateFeaturesAsync(string symbol, DateTime timestamp, CancellationToken cancellationToken)
        {
            var features = new StreamingFeatures
            {
                Symbol = symbol,
                GeneratedAt = timestamp
            };

            if (!_tickCache.TryGetValue(symbol, out var ticks))
                return features;

            List<MarketTick> ticksCopy;
            lock (ticks)
            {
                ticksCopy = new List<MarketTick>(ticks);
            }

            if (ticksCopy.Count == 0)
                return features;

            // Generate microstructure features
            features.MicrostructureFeatures = GenerateMicrostructureFeatures(ticksCopy);

            // Generate time window features
            foreach (var window in _options.TimeWindows)
            {
                var windowFeatures = GenerateTimeWindowFeatures(ticksCopy, timestamp, window);
                features.TimeWindowFeatures[window] = windowFeatures;
            }

            await Task.CompletedTask; // For async consistency
            return features;
        }

        private MicrostructureFeatures GenerateMicrostructureFeatures(List<MarketTick> ticks)
        {
            var features = new MicrostructureFeatures();

            if (ticks.Count < 2)
                return features;

            var recentTicks = ticks.TakeLast(_options.MicrostructureWindow).ToList();

            // Order flow calculation
            var buyVolume = recentTicks.Where(t => t.IsBuyInitiated).Sum(t => t.Volume);
            var sellVolume = recentTicks.Where(t => !t.IsBuyInitiated).Sum(t => t.Volume);
            var totalVolume = buyVolume + sellVolume;
            
            if (totalVolume > 0)
            {
                features.OrderFlow = (double)(buyVolume - sellVolume) / totalVolume;
            }

            // Volume weighted spread
            var validSpreads = recentTicks.Where(t => t.Ask > t.Bid && t.Ask > 0 && t.Bid > 0);
            if (validSpreads.Any())
            {
                var weightedSpread = validSpreads.Sum(t => (t.Ask - t.Bid) * t.Volume);
                var totalVolumeForSpread = validSpreads.Sum(t => t.Volume);
                
                if (totalVolumeForSpread > 0)
                {
                    features.VolumeWeightedSpread = weightedSpread / totalVolumeForSpread;
                }
            }

            // Tick imbalance
            var upticks = recentTicks.Where((t, i) => i > 0 && t.Price > recentTicks[i - 1].Price).Count();
            var downticks = recentTicks.Where((t, i) => i > 0 && t.Price < recentTicks[i - 1].Price).Count();
            var totalDirectionalTicks = upticks + downticks;
            
            if (totalDirectionalTicks > 0)
            {
                features.TickImbalance = (double)(upticks - downticks) / totalDirectionalTicks;
            }

            // Mid price change
            if (recentTicks.Count > 1)
            {
                var firstTick = recentTicks.FirstOrDefault();
                var lastTick = recentTicks.LastOrDefault();
                
                if (firstTick != null && lastTick != null)
                {
                    var firstMid = (firstTick.Bid + firstTick.Ask) / 2.0;
                    var lastMid = (lastTick.Bid + lastTick.Ask) / 2.0;
                    
                    if (firstMid > 0)
                    {
                        features.MidPriceChange = (lastMid - firstMid) / firstMid;
                    }
                }
            }

            return features;
        }

        private TimeWindowFeatures GenerateTimeWindowFeatures(List<MarketTick> ticks, DateTime referenceTime, TimeSpan window)
        {
            var features = new TimeWindowFeatures();
            var cutoffTime = referenceTime - window;
            var windowTicks = ticks.Where(t => t.Timestamp >= cutoffTime && t.Timestamp <= referenceTime).ToList();

            if (windowTicks.Count == 0)
                return features;

            features.TickCount = windowTicks.Count;

            // VWAP calculation
            var totalVolume = windowTicks.Sum(t => t.Volume);
            if (totalVolume > 0)
            {
                features.VWAP = windowTicks.Sum(t => t.Price * t.Volume) / totalVolume;
            }

            // Volatility calculation (price returns)
            if (windowTicks.Count > 1)
            {
                var returns = new List<double>();
                for (int i = 1; i < windowTicks.Count; i++)
                {
                    if (windowTicks[i - 1].Price > 0)
                    {
                        var ret = Math.Log(windowTicks[i].Price / windowTicks[i - 1].Price);
                        returns.Add(ret);
                    }
                }

                if (returns.Count > 1)
                {
                    var meanReturn = returns.Average();
                    var variance = returns.Sum(r => Math.Pow(r - meanReturn, 2)) / (returns.Count - 1);
                    features.Volatility = Math.Sqrt(variance);
                }
            }

            // Volume profile (current volume vs average)
            if (windowTicks.Count > 0)
            {
                var avgVolume = windowTicks.Average(t => t.Volume);
                var lastTick = windowTicks.LastOrDefault();
                
                if (lastTick != null && avgVolume > 0)
                {
                    var currentVolume = lastTick.Volume;
                    features.VolumeProfile = currentVolume / avgVolume;
                }
            }

            // Trend strength (simple linear regression slope)
            if (windowTicks.Count > 2)
            {
                var n = windowTicks.Count;
                var sumX = Enumerable.Range(0, n).Sum();
                var sumY = windowTicks.Sum(t => t.Price);
                var sumXY = windowTicks.Select((t, i) => i * t.Price).Sum();
                var sumXX = Enumerable.Range(0, n).Sum(i => i * i);

                var slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
                features.TrendStrength = slope;
            }

            return features;
        }

        private void CleanupStaleData(object? state)
        {
            try
            {
                var staleThreshold = DateTime.UtcNow - TimeSpan.FromSeconds(_options.StaleThresholdSeconds * 2);
                var symbolsToRemove = new List<string>();

                foreach (var kvp in _lastUpdateTime)
                {
                    if (kvp.Value < staleThreshold)
                    {
                        symbolsToRemove.Add(kvp.Key);
                    }
                }

                foreach (var symbol in symbolsToRemove)
                {
                    _tickCache.TryRemove(symbol, out _);
                    _lastUpdateTime.TryRemove(symbol, out _);
                    _logger.LogDebug("Cleaned up stale data for symbol {Symbol}", symbol);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error during stale data cleanup");
            }
        }

        public void Dispose()
        {
            _cleanupTimer?.Dispose();
        }
    }
}