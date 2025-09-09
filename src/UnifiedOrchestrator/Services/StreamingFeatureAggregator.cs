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
    /// <summary>
    /// Streaming aggregator for microstructure and time-window features
    /// Precomputes and caches features for real-time ML inference
    /// </summary>
    public class StreamingFeatureAggregator : IDisposable
    {
        private readonly ILogger<StreamingFeatureAggregator> _logger;
        private readonly StreamingOptions _options;
        private readonly ConcurrentDictionary<string, SymbolAggregator> _symbolAggregators = new();
        private readonly Timer _cleanupTimer;
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private bool _disposed = false;

        public StreamingFeatureAggregator(
            ILogger<StreamingFeatureAggregator> logger,
            IOptions<StreamingOptions> options)
        {
            _logger = logger;
            _options = options.Value;

            // Start cleanup timer
            _cleanupTimer = new Timer(CleanupStaleData, null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));

            _logger.LogInformation("📊 Streaming Feature Aggregator initialized with {WindowCount} time windows", 
                _options.TimeWindows.Count);
        }

        /// <summary>
        /// Process new market tick and update features
        /// </summary>
        public async Task<StreamingFeatures> ProcessTickAsync(MarketTick tick, CancellationToken cancellationToken = default)
        {
            try
            {
                var aggregator = _symbolAggregators.GetOrAdd(tick.Symbol, _ => new SymbolAggregator(tick.Symbol, _options));
                var features = await aggregator.ProcessTickAsync(tick, cancellationToken);
                
                _logger.LogTrace("📈 Processed tick for {Symbol}: Price={Price}, Volume={Volume}", 
                    tick.Symbol, tick.Price, tick.Volume);
                
                return features;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to process tick for {Symbol}", tick.Symbol);
                throw;
            }
        }

        /// <summary>
        /// Get cached features for a symbol
        /// </summary>
        public StreamingFeatures? GetCachedFeatures(string symbol)
        {
            return _symbolAggregators.TryGetValue(symbol, out var aggregator) 
                ? aggregator.GetCurrentFeatures() 
                : null;
        }

        /// <summary>
        /// Get all cached features
        /// </summary>
        public Dictionary<string, StreamingFeatures> GetAllCachedFeatures()
        {
            return _symbolAggregators.ToDictionary(
                kvp => kvp.Key, 
                kvp => kvp.Value.GetCurrentFeatures()
            );
        }

        /// <summary>
        /// Check if features are stale for any symbol
        /// </summary>
        public bool HasStaleFeatures()
        {
            var cutoffTime = DateTime.UtcNow - TimeSpan.FromSeconds(_options.StaleThresholdSeconds);
            return _symbolAggregators.Values.Any(a => a.LastUpdateTime < cutoffTime);
        }

        /// <summary>
        /// Get stale symbols
        /// </summary>
        public List<string> GetStaleSymbols()
        {
            var cutoffTime = DateTime.UtcNow - TimeSpan.FromSeconds(_options.StaleThresholdSeconds);
            return _symbolAggregators
                .Where(kvp => kvp.Value.LastUpdateTime < cutoffTime)
                .Select(kvp => kvp.Key)
                .ToList();
        }

        private void CleanupStaleData(object? state)
        {
            try
            {
                var cutoffTime = DateTime.UtcNow - TimeSpan.FromMinutes(_options.CleanupAfterMinutes);
                var staleSymbols = _symbolAggregators
                    .Where(kvp => kvp.Value.LastUpdateTime < cutoffTime)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var symbol in staleSymbols)
                {
                    if (_symbolAggregators.TryRemove(symbol, out var aggregator))
                    {
                        aggregator.Dispose();
                        _logger.LogDebug("🗑️ Cleaned up stale aggregator for {Symbol}", symbol);
                    }
                }

                if (staleSymbols.Any())
                {
                    _logger.LogInformation("🧹 Cleaned up {Count} stale symbol aggregators", staleSymbols.Count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error during cleanup");
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _cancellationTokenSource.Cancel();
                _cleanupTimer?.Dispose();
                
                foreach (var aggregator in _symbolAggregators.Values)
                {
                    aggregator.Dispose();
                }
                _symbolAggregators.Clear();

                _cancellationTokenSource.Dispose();
                _disposed = true;
                
                _logger.LogInformation("🛑 Streaming Feature Aggregator disposed");
            }
        }
    }

    /// <summary>
    /// Symbol-specific feature aggregator
    /// </summary>
    public class SymbolAggregator : IDisposable
    {
        private readonly string _symbol;
        private readonly StreamingOptions _options;
        private readonly Dictionary<TimeSpan, TimeWindowAggregator> _windowAggregators = new();
        private readonly MicrostructureCalculator _microstructureCalc;
        private readonly object _lock = new();
        private StreamingFeatures _currentFeatures = new();
        private bool _disposed = false;

        public DateTime LastUpdateTime { get; private set; } = DateTime.UtcNow;

        public SymbolAggregator(string symbol, StreamingOptions options)
        {
            _symbol = symbol;
            _options = options;
            _microstructureCalc = new MicrostructureCalculator(options.MicrostructureWindow);

            // Initialize window aggregators
            foreach (var window in options.TimeWindows)
            {
                _windowAggregators[window] = new TimeWindowAggregator(window);
            }
        }

        public async Task<StreamingFeatures> ProcessTickAsync(MarketTick tick, CancellationToken cancellationToken)
        {
            await Task.Yield(); // Make it async

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
            }

            return _currentFeatures;
        }

        public StreamingFeatures GetCurrentFeatures()
        {
            lock (_lock)
            {
                return _currentFeatures;
            }
        }

        private StreamingFeatures CalculateFeatures()
        {
            var features = new StreamingFeatures
            {
                Symbol = _symbol,
                Timestamp = DateTime.UtcNow,
                MicrostructureFeatures = _microstructureCalc.GetFeatures(),
                TimeWindowFeatures = new Dictionary<TimeSpan, TimeWindowFeatures>()
            };

            foreach (var kvp in _windowAggregators)
            {
                features.TimeWindowFeatures[kvp.Key] = kvp.Value.GetFeatures();
            }

            return features;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _microstructureCalc.Dispose();
                foreach (var aggregator in _windowAggregators.Values)
                {
                    aggregator.Dispose();
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Time window aggregator for OHLCV and technical indicators
    /// </summary>
    public class TimeWindowAggregator : IDisposable
    {
        private readonly TimeSpan _window;
        private readonly Queue<MarketTick> _ticks = new();
        private readonly object _lock = new();
        private bool _disposed = false;

        public TimeWindowAggregator(TimeSpan window)
        {
            _window = window;
        }

        public void AddTick(MarketTick tick)
        {
            lock (_lock)
            {
                _ticks.Enqueue(tick);
                
                // Remove old ticks
                var cutoffTime = tick.Timestamp - _window;
                while (_ticks.Count > 0 && _ticks.Peek().Timestamp < cutoffTime)
                {
                    _ticks.Dequeue();
                }
            }
        }

        public TimeWindowFeatures GetFeatures()
        {
            lock (_lock)
            {
                if (_ticks.Count == 0)
                    return new TimeWindowFeatures();

                var ticksArray = _ticks.ToArray();
                var prices = ticksArray.Select(t => t.Price).ToArray();
                var volumes = ticksArray.Select(t => t.Volume).ToArray();

                return new TimeWindowFeatures
                {
                    Open = ticksArray.First().Price,
                    High = prices.Max(),
                    Low = prices.Min(),
                    Close = ticksArray.Last().Price,
                    Volume = volumes.Sum(),
                    TickCount = ticksArray.Length,
                    VWAP = CalculateVWAP(ticksArray),
                    Volatility = CalculateVolatility(prices),
                    RSI = CalculateRSI(prices),
                    MeanReversion = CalculateMeanReversion(prices),
                    Momentum = CalculateMomentum(prices),
                    VolumeProfile = CalculateVolumeProfile(ticksArray)
                };
            }
        }

        private double CalculateVWAP(MarketTick[] ticks)
        {
            if (!ticks.Any()) return 0;
            
            var totalValue = ticks.Sum(t => t.Price * t.Volume);
            var totalVolume = ticks.Sum(t => t.Volume);
            return totalVolume > 0 ? totalValue / totalVolume : ticks.Last().Price;
        }

        private double CalculateVolatility(double[] prices)
        {
            if (prices.Length < 2) return 0;
            
            var returns = new double[prices.Length - 1];
            for (int i = 1; i < prices.Length; i++)
            {
                returns[i - 1] = Math.Log(prices[i] / prices[i - 1]);
            }
            
            var mean = returns.Average();
            var variance = returns.Average(r => Math.Pow(r - mean, 2));
            return Math.Sqrt(variance) * Math.Sqrt(252 * 24 * 60); // Annualized
        }

        private double CalculateRSI(double[] prices, int period = 14)
        {
            if (prices.Length < period + 1) return 50; // Neutral RSI
            
            var gains = new List<double>();
            var losses = new List<double>();
            
            for (int i = 1; i < Math.Min(prices.Length, period + 1); i++)
            {
                var change = prices[i] - prices[i - 1];
                if (change > 0)
                {
                    gains.Add(change);
                    losses.Add(0);
                }
                else
                {
                    gains.Add(0);
                    losses.Add(-change);
                }
            }
            
            var avgGain = gains.Average();
            var avgLoss = losses.Average();
            
            if (avgLoss == 0) return 100;
            
            var rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }

        private double CalculateMeanReversion(double[] prices)
        {
            if (prices.Length < 2) return 0;
            
            var mean = prices.Average();
            var currentPrice = prices.Last();
            var stdDev = Math.Sqrt(prices.Average(p => Math.Pow(p - mean, 2)));
            
            return stdDev > 0 ? (currentPrice - mean) / stdDev : 0;
        }

        private double CalculateMomentum(double[] prices)
        {
            if (prices.Length < 2) return 0;
            return (prices.Last() - prices.First()) / prices.First();
        }

        private Dictionary<string, double> CalculateVolumeProfile(MarketTick[] ticks)
        {
            if (!ticks.Any()) return new Dictionary<string, double>();
            
            var buyVolume = ticks.Where(t => t.IsBuyInitiated).Sum(t => t.Volume);
            var sellVolume = ticks.Where(t => !t.IsBuyInitiated).Sum(t => t.Volume);
            var totalVolume = ticks.Sum(t => t.Volume);
            
            return new Dictionary<string, double>
            {
                ["BuyRatio"] = totalVolume > 0 ? buyVolume / totalVolume : 0.5,
                ["SellRatio"] = totalVolume > 0 ? sellVolume / totalVolume : 0.5,
                ["Imbalance"] = totalVolume > 0 ? (buyVolume - sellVolume) / totalVolume : 0
            };
        }

        public void Dispose()
        {
            if (!_disposed)
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
    /// Microstructure feature calculator
    /// </summary>
    public class MicrostructureCalculator : IDisposable
    {
        private readonly int _maxTicks;
        private readonly Queue<MarketTick> _recentTicks = new();
        private readonly object _lock = new();
        private bool _disposed = false;

        public MicrostructureCalculator(int maxTicks)
        {
            _maxTicks = maxTicks;
        }

        public void AddTick(MarketTick tick)
        {
            lock (_lock)
            {
                _recentTicks.Enqueue(tick);
                while (_recentTicks.Count > _maxTicks)
                {
                    _recentTicks.Dequeue();
                }
            }
        }

        public MicrostructureFeatures GetFeatures()
        {
            lock (_lock)
            {
                if (_recentTicks.Count < 2)
                    return new MicrostructureFeatures();

                var ticks = _recentTicks.ToArray();
                
                return new MicrostructureFeatures
                {
                    BidAskSpread = CalculateSpread(ticks),
                    OrderFlow = CalculateOrderFlow(ticks),
                    PriceImpact = CalculatePriceImpact(ticks),
                    TickRule = CalculateTickRule(ticks),
                    Aggressiveness = CalculateAggressiveness(ticks),
                    MarketDepth = CalculateMarketDepth(ticks)
                };
            }
        }

        private double CalculateSpread(MarketTick[] ticks)
        {
            // Simplified - would use actual bid/ask if available
            var prices = ticks.Select(t => t.Price).ToArray();
            var range = prices.Max() - prices.Min();
            return range / prices.Average() * 10000; // In basis points
        }

        private double CalculateOrderFlow(MarketTick[] ticks)
        {
            var buyVolume = ticks.Where(t => t.IsBuyInitiated).Sum(t => t.Volume);
            var sellVolume = ticks.Where(t => !t.IsBuyInitiated).Sum(t => t.Volume);
            var totalVolume = buyVolume + sellVolume;
            
            return totalVolume > 0 ? (buyVolume - sellVolume) / totalVolume : 0;
        }

        private double CalculatePriceImpact(MarketTick[] ticks)
        {
            if (ticks.Length < 3) return 0;
            
            var impacts = new List<double>();
            for (int i = 1; i < ticks.Length - 1; i++)
            {
                var volumeWeight = ticks[i].Volume / ticks.Average(t => t.Volume);
                var priceChange = Math.Abs(ticks[i + 1].Price - ticks[i].Price) / ticks[i].Price;
                impacts.Add(priceChange * volumeWeight);
            }
            
            return impacts.Any() ? impacts.Average() * 10000 : 0; // In basis points
        }

        private double CalculateTickRule(MarketTick[] ticks)
        {
            if (ticks.Length < 2) return 0;
            
            var upTicks = 0;
            var downTicks = 0;
            
            for (int i = 1; i < ticks.Length; i++)
            {
                if (ticks[i].Price > ticks[i - 1].Price)
                    upTicks++;
                else if (ticks[i].Price < ticks[i - 1].Price)
                    downTicks++;
            }
            
            var totalMoves = upTicks + downTicks;
            return totalMoves > 0 ? (double)(upTicks - downTicks) / totalMoves : 0;
        }

        private double CalculateAggressiveness(MarketTick[] ticks)
        {
            // Simplified aggressiveness measure
            var largeOrders = ticks.Where(t => t.Volume > ticks.Average(tick => tick.Volume) * 2).Count();
            return (double)largeOrders / ticks.Length;
        }

        private double CalculateMarketDepth(MarketTick[] ticks)
        {
            // Simplified market depth estimate
            var volumeVariability = CalculateVolumeVariability(ticks);
            return 1.0 / (1.0 + volumeVariability); // Inverse relationship
        }

        private double CalculateVolumeVariability(MarketTick[] ticks)
        {
            if (ticks.Length < 2) return 0;
            
            var volumes = ticks.Select(t => (double)t.Volume).ToArray();
            var mean = volumes.Average();
            var variance = volumes.Average(v => Math.Pow(v - mean, 2));
            return mean > 0 ? Math.Sqrt(variance) / mean : 0; // Coefficient of variation
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    _recentTicks.Clear();
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Configuration options for streaming aggregator
    /// </summary>
    public class StreamingOptions
    {
        public List<TimeSpan> TimeWindows { get; set; } = new()
        {
            TimeSpan.FromMinutes(1),
            TimeSpan.FromMinutes(5),
            TimeSpan.FromMinutes(15),
            TimeSpan.FromMinutes(30),
            TimeSpan.FromHours(1)
        };
        public int MicrostructureWindow { get; set; } = 1000;
        public int StaleThresholdSeconds { get; set; } = 30;
        public int CleanupAfterMinutes { get; set; } = 60;
    }

    /// <summary>
    /// Market tick data
    /// </summary>
    public class MarketTick
    {
        public string Symbol { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public double Price { get; set; }
        public int Volume { get; set; }
        public bool IsBuyInitiated { get; set; }
        public double Bid { get; set; }
        public double Ask { get; set; }
    }

    /// <summary>
    /// Streaming features for a symbol
    /// </summary>
    public class StreamingFeatures
    {
        public string Symbol { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public MicrostructureFeatures MicrostructureFeatures { get; set; } = new();
        public Dictionary<TimeSpan, TimeWindowFeatures> TimeWindowFeatures { get; set; } = new();
    }

    /// <summary>
    /// Microstructure features
    /// </summary>
    public class MicrostructureFeatures
    {
        public double BidAskSpread { get; set; }
        public double OrderFlow { get; set; }
        public double PriceImpact { get; set; }
        public double TickRule { get; set; }
        public double Aggressiveness { get; set; }
        public double MarketDepth { get; set; }
    }

    /// <summary>
    /// Time window features
    /// </summary>
    public class TimeWindowFeatures
    {
        public double Open { get; set; }
        public double High { get; set; }
        public double Low { get; set; }
        public double Close { get; set; }
        public long Volume { get; set; }
        public int TickCount { get; set; }
        public double VWAP { get; set; }
        public double Volatility { get; set; }
        public double RSI { get; set; }
        public double MeanReversion { get; set; }
        public double Momentum { get; set; }
        public Dictionary<string, double> VolumeProfile { get; set; } = new();
    }
}