using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Infra;

/// <summary>
/// COMPONENT 8: DATA FEED REDUNDANCY
/// Manages multiple data feeds with automatic failover, health monitoring, and data consistency validation
/// </summary>
public class RedundantDataFeedManager : IDisposable
{
    private readonly ILogger<RedundantDataFeedManager> _logger;
    private readonly List<IDataFeed> _dataFeeds = new();
    private readonly ConcurrentDictionary<string, DataFeedHealth> _feedHealth = new();
    private readonly ConcurrentDictionary<string, MarketData> _consolidatedData = new();
    private IDataFeed? _primaryFeed;
    private Timer? _healthCheckTimer;
    private Timer? _consistencyCheckTimer;
    private bool _disposed = false;

    public interface IDataFeed
    {
        string FeedName { get; }
        int Priority { get; }
        Task<bool> Connect();
        Task<MarketData?> GetMarketData(string symbol);
        Task<OrderBook?> GetOrderBook(string symbol);
        event EventHandler<MarketData>? OnDataReceived;
        event EventHandler<Exception>? OnError;
    }

    public class DataFeedHealth
    {
        public string FeedName { get; set; } = string.Empty;
        public bool IsHealthy { get; set; }
        public DateTime LastHealthCheck { get; set; }
        public DateTime LastDataReceived { get; set; }
        public double Latency { get; set; }
        public int ErrorCount { get; set; }
        public double DataQualityScore { get; set; }
    }

    public class MarketData
    {
        public string Symbol { get; set; } = string.Empty;
        public decimal Price { get; set; }
        public decimal Bid { get; set; }
        public decimal Ask { get; set; }
        public long Volume { get; set; }
        public DateTime Timestamp { get; set; }
        public string Source { get; set; } = string.Empty;
    }

    public class OrderBook
    {
        public string Symbol { get; set; } = string.Empty;
        public List<PriceLevel> Bids { get; set; } = new();
        public List<PriceLevel> Asks { get; set; } = new();
        public DateTime Timestamp { get; set; }
    }

    public class PriceLevel
    {
        public decimal Price { get; set; }
        public long Size { get; set; }
    }

    public class MarketDataConsistency
    {
        public string Symbol { get; set; } = string.Empty;
        public Dictionary<string, decimal> PriceByFeed { get; set; } = new();
        public decimal PriceDeviation { get; set; }
        public string OutlierFeed { get; set; } = string.Empty;
        public bool IsConsistent => PriceDeviation < 0.001m; // 0.1% tolerance
    }

    // Parameterless constructor for dependency injection
    public RedundantDataFeedManager() : this(null) { }

    public RedundantDataFeedManager(ILogger<RedundantDataFeedManager>? logger)
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<RedundantDataFeedManager>.Instance;
    }

    public async Task InitializeDataFeeds()
    {
        try
        {
            _logger.LogInformation("[DataFeed] Initializing redundant data feed system");

            // Add primary feed (TopstepX)
            _dataFeeds.Add(new TopstepXDataFeed { Priority = 1 });

            // Add backup feeds
            _dataFeeds.Add(new InteractiveBrokersDataFeed { Priority = 2 });
            _dataFeeds.Add(new TradingViewDataFeed { Priority = 3 });
            _dataFeeds.Add(new AlphaVantageDataFeed { Priority = 4 });

            // Sort by priority
            _dataFeeds.Sort((a, b) => a.Priority.CompareTo(b.Priority));

            // Connect to all feeds
            foreach (var feed in _dataFeeds)
            {
                try
                {
                    var connected = await feed.Connect();
                    _feedHealth[feed.FeedName] = new DataFeedHealth
                    {
                        FeedName = feed.FeedName,
                        IsHealthy = connected,
                        LastHealthCheck = DateTime.UtcNow,
                        DataQualityScore = 1.0
                    };

                    // Subscribe to events
                    feed.OnDataReceived += OnDataReceived;
                    feed.OnError += OnFeedError;

                    if (connected && _primaryFeed == null)
                    {
                        _primaryFeed = feed;
                        _logger.LogInformation("[DataFeed] Primary data feed set to: {FeedName}", feed.FeedName);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[DataFeed] Failed to connect to {FeedName}", feed.FeedName);
                }
            }

            // Start health monitoring
            _healthCheckTimer = new Timer(CheckFeedHealth, null, TimeSpan.Zero, TimeSpan.FromSeconds(10));
            _consistencyCheckTimer = new Timer(CheckDataConsistency, null, TimeSpan.Zero, TimeSpan.FromSeconds(5));

            _logger.LogInformation("[DataFeed] Data feed system initialized with {Count} feeds", _dataFeeds.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DataFeed] Failed to initialize data feed system");
            throw;
        }
    }

    public async Task<MarketData?> GetMarketData(string symbol)
    {
        MarketData? data = null;
        Exception? lastError = null;

        // Try primary feed first
        if (_primaryFeed != null && _feedHealth[_primaryFeed.FeedName].IsHealthy)
        {
            try
            {
                data = await _primaryFeed.GetMarketData(symbol);

                if (ValidateMarketData(data))
                {
                    _logger.LogDebug("[DataFeed] Got data for {Symbol} from primary feed {Feed}", 
                        symbol, _primaryFeed.FeedName);
                    return data;
                }
            }
            catch (Exception ex)
            {
                lastError = ex;
                await HandleFeedFailure(_primaryFeed, ex);
            }
        }

        // Failover to backup feeds
        foreach (var feed in _dataFeeds.Where(f => f != _primaryFeed))
        {
            if (_feedHealth[feed.FeedName].IsHealthy)
            {
                try
                {
                    _logger.LogWarning("[DataFeed] Failing over to {FeedName} for {Symbol}", 
                        feed.FeedName, symbol);

                    data = await feed.GetMarketData(symbol);

                    if (ValidateMarketData(data))
                    {
                        // Switch primary feed
                        _primaryFeed = feed;
                        _logger.LogWarning("[DataFeed] Data feed switched to {FeedName}", feed.FeedName);
                        return data;
                    }
                }
                catch (Exception ex)
                {
                    lastError = ex;
                    await HandleFeedFailure(feed, ex);
                }
            }
        }

        // All feeds failed
        var message = "All data feeds unavailable";
        _logger.LogCritical("[DataFeed] {Message}", message);
        throw new DataFeedException(message, lastError);
    }

    private bool ValidateMarketData(MarketData? data)
    {
        if (data == null) return false;
        if (data.Price <= 0) return false;
        if (data.Bid < 0 || data.Ask < 0) return false;
        if (data.Bid > data.Ask && data.Bid > 0 && data.Ask > 0) return false; // Crossed market
        if (DateTime.UtcNow - data.Timestamp > TimeSpan.FromMinutes(5)) return false; // Stale data
        
        return true;
    }

    private async Task HandleFeedFailure(IDataFeed feed, Exception ex)
    {
        if (_feedHealth.TryGetValue(feed.FeedName, out var health))
        {
            health.IsHealthy = false;
            health.ErrorCount++;
            
            _logger.LogError(ex, "[DataFeed] Feed {FeedName} failed (errors: {ErrorCount})", 
                feed.FeedName, health.ErrorCount);
        }

        await Task.CompletedTask;
    }

    private void CheckFeedHealth(object? state)
    {
        Parallel.ForEach(_dataFeeds, async feed =>
        {
            try
            {
                var startTime = DateTime.UtcNow;

                // Ping feed with test request
                var testData = await feed.GetMarketData("ES");

                var latency = (DateTime.UtcNow - startTime).TotalMilliseconds;

                if (_feedHealth.TryGetValue(feed.FeedName, out var health))
                {
                    health.IsHealthy = testData != null;
                    health.LastHealthCheck = DateTime.UtcNow;
                    health.Latency = latency;
                    health.DataQualityScore = CalculateDataQuality(testData);

                    // Alert on high latency
                    if (latency > 100)
                    {
                        _logger.LogWarning("[DataFeed] High latency on {FeedName}: {Latency}ms", 
                            feed.FeedName, latency);
                    }
                }
            }
            catch (Exception ex)
            {
                if (_feedHealth.TryGetValue(feed.FeedName, out var health))
                {
                    health.IsHealthy = false;
                    health.ErrorCount++;

                    if (health.ErrorCount > 5)
                    {
                        _logger.LogError(ex, "[DataFeed] Feed {FeedName} marked unhealthy", feed.FeedName);
                    }
                }
            }
        });

        // Ensure at least one feed is healthy
        if (!_feedHealth.Values.Any(h => h.IsHealthy))
        {
            _logger.LogCritical("[DataFeed] ALL DATA FEEDS DOWN - TRADING HALTED");
            HaltTrading();
        }
    }

    private void CheckDataConsistency(object? state)
    {
        var symbols = new[] { "ES", "NQ" };

        foreach (var symbol in symbols)
        {
            try
            {
                var consistency = new MarketDataConsistency
                {
                    Symbol = symbol,
                    PriceByFeed = new Dictionary<string, decimal>()
                };

                // Get price from each healthy feed
                var tasks = _dataFeeds.Where(f => _feedHealth[f.FeedName].IsHealthy)
                    .Select(async feed =>
                    {
                        try
                        {
                            var data = await feed.GetMarketData(symbol);
                            if (data != null)
                            {
                                consistency.PriceByFeed[feed.FeedName] = data.Price;
                            }
                        }
                        catch
                        {
                            // Skip failed feeds
                        }
                    });

                Task.WaitAll(tasks.ToArray(), TimeSpan.FromSeconds(5));

                if (consistency.PriceByFeed.Count >= 2)
                {
                    // Calculate deviation
                    var prices = consistency.PriceByFeed.Values.ToList();
                    var avgPrice = prices.Average();
                    var maxDeviation = prices.Max(p => Math.Abs(p - avgPrice) / avgPrice);

                    consistency.PriceDeviation = maxDeviation;

                    if (!consistency.IsConsistent)
                    {
                        // Find outlier
                        foreach (var kvp in consistency.PriceByFeed)
                        {
                            var deviation = Math.Abs(kvp.Value - avgPrice) / avgPrice;
                            if (Math.Abs(deviation - maxDeviation) < 0.0001m) // Close enough to max
                            {
                                consistency.OutlierFeed = kvp.Key;
                                break;
                            }
                        }

                        _logger.LogWarning("[DataFeed] Data inconsistency detected for {Symbol}: {OutlierFeed} deviates by {Deviation:P2}",
                            symbol, consistency.OutlierFeed, maxDeviation);

                        // Mark outlier feed as suspicious
                        if (_feedHealth.TryGetValue(consistency.OutlierFeed, out var health))
                        {
                            health.DataQualityScore *= 0.9; // Reduce quality score
                            if (health.DataQualityScore < 0.5)
                            {
                                health.IsHealthy = false;
                                _logger.LogWarning("[DataFeed] Marked {FeedName} as unhealthy due to poor data quality",
                                    consistency.OutlierFeed);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[DataFeed] Error checking data consistency for {Symbol}", symbol);
            }
        }
    }

    private double CalculateDataQuality(MarketData? data)
    {
        if (data == null) return 0.0;

        double score = 1.0;

        // Penalize stale data
        var ageMinutes = (DateTime.UtcNow - data.Timestamp).TotalMinutes;
        if (ageMinutes > 1) score *= 0.8;
        if (ageMinutes > 5) score *= 0.5;

        // Penalize invalid prices
        if (data.Price <= 0) score = 0.0;
        if (data.Bid > data.Ask && data.Bid > 0 && data.Ask > 0) score *= 0.3;

        return Math.Max(0.0, score);
    }

    private void OnDataReceived(object? sender, MarketData data)
    {
        if (sender is IDataFeed feed)
        {
            if (_feedHealth.TryGetValue(feed.FeedName, out var health))
            {
                health.LastDataReceived = DateTime.UtcNow;
            }

            // Update consolidated data
            _consolidatedData[data.Symbol] = data;
        }
    }

    private void OnFeedError(object? sender, Exception ex)
    {
        if (sender is IDataFeed feed)
        {
            _logger.LogError(ex, "[DataFeed] Error from {FeedName}", feed.FeedName);
            
            if (_feedHealth.TryGetValue(feed.FeedName, out var health))
            {
                health.ErrorCount++;
                if (health.ErrorCount > 3)
                {
                    health.IsHealthy = false;
                }
            }
        }
    }

    private void HaltTrading()
    {
        // Emergency trading halt - would integrate with trading system
        _logger.LogCritical("[DataFeed] Emergency trading halt triggered due to data feed failure");
    }

    public DataFeedStatus GetFeedStatus()
    {
        return new DataFeedStatus
        {
            PrimaryFeed = _primaryFeed?.FeedName ?? "None",
            HealthyFeeds = _feedHealth.Values.Count(h => h.IsHealthy),
            TotalFeeds = _dataFeeds.Count,
            FeedHealthDetails = _feedHealth.Values.ToList()
        };
    }

    public class DataFeedStatus
    {
        public string PrimaryFeed { get; set; } = string.Empty;
        public int HealthyFeeds { get; set; }
        public int TotalFeeds { get; set; }
        public List<DataFeedHealth> FeedHealthDetails { get; set; } = new();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            
            _healthCheckTimer?.Dispose();
            _consistencyCheckTimer?.Dispose();
            
            foreach (var feed in _dataFeeds)
            {
                try
                {
                    feed.OnDataReceived -= OnDataReceived;
                    feed.OnError -= OnFeedError;
                }
                catch { }
            }
            
            _logger.LogInformation("[DataFeed] Redundant Data Feed Manager disposed");
        }
    }
}

// Mock data feed implementations for testing
public class TopstepXDataFeed : RedundantDataFeedManager.IDataFeed
{
    public string FeedName => "TopstepX";
    public int Priority { get; set; } = 1;
    
    public event EventHandler<RedundantDataFeedManager.MarketData>? OnDataReceived;
    public event EventHandler<Exception>? OnError;

    public async Task<bool> Connect()
    {
        await Task.Delay(100);
        return true;
    }

    public async Task<RedundantDataFeedManager.MarketData?> GetMarketData(string symbol)
    {
        await Task.Delay(50);
        return new RedundantDataFeedManager.MarketData
        {
            Symbol = symbol,
            Price = 4500.25m,
            Bid = 4500.00m,
            Ask = 4500.50m,
            Volume = 1000,
            Timestamp = DateTime.UtcNow,
            Source = FeedName
        };
    }

    public async Task<RedundantDataFeedManager.OrderBook?> GetOrderBook(string symbol)
    {
        await Task.Delay(50);
        return new RedundantDataFeedManager.OrderBook
        {
            Symbol = symbol,
            Timestamp = DateTime.UtcNow
        };
    }
}

public class InteractiveBrokersDataFeed : RedundantDataFeedManager.IDataFeed
{
    public string FeedName => "InteractiveBrokers";
    public int Priority { get; set; } = 2;
    
    public event EventHandler<RedundantDataFeedManager.MarketData>? OnDataReceived;
    public event EventHandler<Exception>? OnError;

    public async Task<bool> Connect()
    {
        await Task.Delay(100);
        return true;
    }

    public async Task<RedundantDataFeedManager.MarketData?> GetMarketData(string symbol)
    {
        await Task.Delay(75);
        return new RedundantDataFeedManager.MarketData
        {
            Symbol = symbol,
            Price = 4500.30m, // Slightly different price
            Bid = 4500.05m,
            Ask = 4500.55m,
            Volume = 950,
            Timestamp = DateTime.UtcNow,
            Source = FeedName
        };
    }

    public async Task<RedundantDataFeedManager.OrderBook?> GetOrderBook(string symbol)
    {
        await Task.Delay(75);
        return new RedundantDataFeedManager.OrderBook
        {
            Symbol = symbol,
            Timestamp = DateTime.UtcNow
        };
    }
}

public class TradingViewDataFeed : RedundantDataFeedManager.IDataFeed
{
    public string FeedName => "TradingView";
    public int Priority { get; set; } = 3;
    
    public event EventHandler<RedundantDataFeedManager.MarketData>? OnDataReceived;
    public event EventHandler<Exception>? OnError;

    public async Task<bool> Connect()
    {
        await Task.Delay(100);
        return true;
    }

    public async Task<RedundantDataFeedManager.MarketData?> GetMarketData(string symbol)
    {
        await Task.Delay(100);
        return new RedundantDataFeedManager.MarketData
        {
            Symbol = symbol,
            Price = 4500.20m,
            Bid = 4499.95m,
            Ask = 4500.45m,
            Volume = 1100,
            Timestamp = DateTime.UtcNow,
            Source = FeedName
        };
    }

    public async Task<RedundantDataFeedManager.OrderBook?> GetOrderBook(string symbol)
    {
        await Task.Delay(100);
        return new RedundantDataFeedManager.OrderBook
        {
            Symbol = symbol,
            Timestamp = DateTime.UtcNow
        };
    }
}

public class AlphaVantageDataFeed : RedundantDataFeedManager.IDataFeed
{
    public string FeedName => "AlphaVantage";
    public int Priority { get; set; } = 4;
    
    public event EventHandler<RedundantDataFeedManager.MarketData>? OnDataReceived;
    public event EventHandler<Exception>? OnError;

    public async Task<bool> Connect()
    {
        await Task.Delay(100);
        return true;
    }

    public async Task<RedundantDataFeedManager.MarketData?> GetMarketData(string symbol)
    {
        await Task.Delay(200); // Slower feed
        return new RedundantDataFeedManager.MarketData
        {
            Symbol = symbol,
            Price = 4500.15m,
            Bid = 4499.90m,
            Ask = 4500.40m,
            Volume = 800,
            Timestamp = DateTime.UtcNow,
            Source = FeedName
        };
    }

    public async Task<RedundantDataFeedManager.OrderBook?> GetOrderBook(string symbol)
    {
        await Task.Delay(200);
        return new RedundantDataFeedManager.OrderBook
        {
            Symbol = symbol,
            Timestamp = DateTime.UtcNow
        };
    }
}

public class DataFeedException : Exception
{
    public DataFeedException(string message) : base(message) { }
    public DataFeedException(string message, Exception? innerException) : base(message, innerException) { }
}