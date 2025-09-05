using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace BotCore.Market;

/// <summary>
/// Data feed interface for redundant market data management
/// </summary>
public interface IDataFeed
{
    string FeedName { get; }
    int Priority { get; }
    Task<bool> ConnectAsync();
    Task<MarketData?> GetMarketDataAsync(string symbol);
    Task<OrderBook?> GetOrderBookAsync(string symbol);
    event EventHandler<MarketData>? OnDataReceived;
    event EventHandler<Exception>? OnError;
}

/// <summary>
/// Market data structure
/// </summary>
public class MarketData
{
    public string Symbol { get; set; } = string.Empty;
    public decimal Price { get; set; }
    public decimal Volume { get; set; }
    public decimal Bid { get; set; }
    public decimal Ask { get; set; }
    public DateTime Timestamp { get; set; }
    public string Source { get; set; } = string.Empty;
}

/// <summary>
/// Order book structure
/// </summary>
public class OrderBook
{
    public string Symbol { get; set; } = string.Empty;
    public List<OrderBookLevel> Bids { get; set; } = new();
    public List<OrderBookLevel> Asks { get; set; } = new();
    public DateTime Timestamp { get; set; }
}

public class OrderBookLevel
{
    public decimal Price { get; set; }
    public decimal Size { get; set; }
}

/// <summary>
/// Data feed health monitoring
/// </summary>
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

/// <summary>
/// Redundant data feed manager for high availability market data
/// </summary>
public class RedundantDataFeedManager : IDisposable
{
    private readonly ILogger<RedundantDataFeedManager> _logger;
    private readonly List<IDataFeed> _dataFeeds = new();
    private readonly ConcurrentDictionary<string, DataFeedHealth> _feedHealth = new();
    private readonly ConcurrentDictionary<string, MarketData> _consolidatedData = new();
    private IDataFeed? _primaryFeed;
    private readonly Timer _healthCheckTimer;
    private readonly Timer _consistencyCheckTimer;
    private bool _disposed = false;

    public event EventHandler<MarketData>? OnConsolidatedData;
    public event EventHandler<string>? OnFeedFailover;

    public RedundantDataFeedManager(ILogger<RedundantDataFeedManager> logger)
    {
        _logger = logger;
        _healthCheckTimer = new Timer(CheckFeedHealth, null, Timeout.Infinite, Timeout.Infinite);
        _consistencyCheckTimer = new Timer(CheckDataConsistency, null, Timeout.Infinite, Timeout.Infinite);
        
        _logger.LogInformation("[DataFeed] RedundantDataFeedManager initialized");
    }

    /// <summary>
    /// Initialize data feeds and start monitoring
    /// </summary>
    public async Task InitializeDataFeedsAsync()
    {
        _logger.LogInformation("[DataFeed] Initializing data feeds");
        
        // Add data feeds (these would be actual implementations)
        AddDataFeed(new TopstepXDataFeed { Priority = 1 });
        AddDataFeed(new BackupDataFeed { Priority = 2 });
        
        // Sort by priority
        _dataFeeds.Sort((a, b) => a.Priority.CompareTo(b.Priority));

        // Connect to all feeds
        foreach (var feed in _dataFeeds)
        {
            try
            {
                var connected = await feed.ConnectAsync();
                _feedHealth[feed.FeedName] = new DataFeedHealth
                {
                    FeedName = feed.FeedName,
                    IsHealthy = connected,
                    LastHealthCheck = DateTime.UtcNow
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
        _healthCheckTimer.Change(TimeSpan.Zero, TimeSpan.FromSeconds(10));
        _consistencyCheckTimer.Change(TimeSpan.Zero, TimeSpan.FromSeconds(5));
        
        await Task.CompletedTask;
    }

    public void AddDataFeed(IDataFeed dataFeed)
    {
        _dataFeeds.Add(dataFeed);
        _logger.LogDebug("[DataFeed] Added data feed: {FeedName} (Priority: {Priority})", 
            dataFeed.FeedName, dataFeed.Priority);
    }

    /// <summary>
    /// Get market data with automatic failover
    /// </summary>
    public async Task<MarketData?> GetMarketDataAsync(string symbol)
    {
        MarketData? data = null;
        Exception? lastError = null;

        // Try primary feed first
        if (_primaryFeed != null && _feedHealth.GetValueOrDefault(_primaryFeed.FeedName)?.IsHealthy == true)
        {
            try
            {
                data = await _primaryFeed.GetMarketDataAsync(symbol);
                if (ValidateMarketData(data))
                {
                    return data;
                }
            }
            catch (Exception ex)
            {
                lastError = ex;
                await HandleFeedFailureAsync(_primaryFeed, ex);
            }
        }

        // Failover to backup feeds
        foreach (var feed in _dataFeeds.Where(f => f != _primaryFeed))
        {
            if (_feedHealth.GetValueOrDefault(feed.FeedName)?.IsHealthy == true)
            {
                try
                {
                    _logger.LogWarning("[DataFeed] Failing over to {FeedName} for {Symbol}", feed.FeedName, symbol);

                    data = await feed.GetMarketDataAsync(symbol);
                    if (ValidateMarketData(data))
                    {
                        // Switch primary feed
                        _primaryFeed = feed;
                        OnFeedFailover?.Invoke(this, feed.FeedName);
                        _logger.LogWarning("[DataFeed] Data feed switched to {FeedName}", feed.FeedName);
                        return data;
                    }
                }
                catch (Exception ex)
                {
                    lastError = ex;
                    await HandleFeedFailureAsync(feed, ex);
                }
            }
        }

        // All feeds failed
        var errorMessage = $"All data feeds unavailable for {symbol}";
        _logger.LogCritical("[DataFeed] {ErrorMessage}", errorMessage);
        throw new InvalidOperationException(errorMessage, lastError);
    }

    private bool ValidateMarketData(MarketData? data)
    {
        if (data == null) return false;
        if (data.Price <= 0) return false;
        if (string.IsNullOrEmpty(data.Symbol)) return false;
        if (DateTime.UtcNow - data.Timestamp > TimeSpan.FromMinutes(5)) return false;
        
        return true;
    }

    private async Task HandleFeedFailureAsync(IDataFeed feed, Exception ex)
    {
        if (_feedHealth.TryGetValue(feed.FeedName, out var health))
        {
            health.IsHealthy = false;
            health.ErrorCount++;
            
            _logger.LogError(ex, "[DataFeed] Feed {FeedName} failed (Error count: {ErrorCount})", 
                feed.FeedName, health.ErrorCount);
        }
        
        await Task.CompletedTask;
    }

    private void OnDataReceived(object? sender, MarketData data)
    {
        try
        {
            if (sender is IDataFeed feed)
            {
                if (_feedHealth.TryGetValue(feed.FeedName, out var health))
                {
                    health.LastDataReceived = DateTime.UtcNow;
                }
                
                // Store consolidated data
                _consolidatedData[data.Symbol] = data;
                
                // Emit consolidated data event
                OnConsolidatedData?.Invoke(this, data);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DataFeed] Error processing received data");
        }
    }

    private void OnFeedError(object? sender, Exception ex)
    {
        if (sender is IDataFeed feed)
        {
            _ = Task.Run(async () => await HandleFeedFailureAsync(feed, ex));
        }
    }

    private void CheckFeedHealth(object? state)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                foreach (var feed in _dataFeeds)
                {
                    var startTime = DateTime.UtcNow;
                    
                    try
                    {
                        // Test feed with ping
                        var testData = await feed.GetMarketDataAsync("ES");
                        var latency = (DateTime.UtcNow - startTime).TotalMilliseconds;
                        
                        if (_feedHealth.TryGetValue(feed.FeedName, out var health))
                        {
                            health.IsHealthy = testData != null;
                            health.LastHealthCheck = DateTime.UtcNow;
                            health.Latency = latency;
                            health.DataQualityScore = CalculateDataQuality(testData);
                        }
                        
                        if (latency > 100)
                        {
                            _logger.LogWarning("[DataFeed] High latency on {FeedName}: {Latency}ms", feed.FeedName, latency);
                        }
                    }
                    catch (Exception ex)
                    {
                        if (_feedHealth.TryGetValue(feed.FeedName, out var health))
                        {
                            health.IsHealthy = false;
                            health.ErrorCount++;
                        }
                        
                        _logger.LogDebug(ex, "[DataFeed] Health check failed for {FeedName}", feed.FeedName);
                    }
                }
                
                // Ensure at least one feed is healthy
                if (!_feedHealth.Values.Any(h => h.IsHealthy))
                {
                    _logger.LogCritical("[DataFeed] ALL DATA FEEDS DOWN - TRADING HALTED");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[DataFeed] Error in health check");
            }
        });
    }

    private void CheckDataConsistency(object? state)
    {
        // Implementation for checking data consistency across feeds
        // This would compare prices from different feeds and detect outliers
    }

    private double CalculateDataQuality(MarketData? data)
    {
        if (data == null) return 0.0;
        
        var score = 1.0;
        
        // Reduce score for stale data
        var age = DateTime.UtcNow - data.Timestamp;
        if (age > TimeSpan.FromSeconds(30)) score -= 0.3;
        if (age > TimeSpan.FromMinutes(1)) score -= 0.5;
        
        // Reduce score for invalid prices
        if (data.Price <= 0) score = 0.0;
        if (data.Bid >= data.Ask && data.Bid > 0 && data.Ask > 0) score -= 0.2;
        
        return Math.Max(0.0, score);
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _logger.LogInformation("[DataFeed] Disposing RedundantDataFeedManager");
        
        _disposed = true;
        
        _healthCheckTimer?.Dispose();
        _consistencyCheckTimer?.Dispose();
        
        foreach (var feed in _dataFeeds)
        {
            if (feed is IDisposable disposableFeed)
            {
                disposableFeed.Dispose();
            }
        }
        
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// TopstepX data feed implementation
/// </summary>
public class TopstepXDataFeed : IDataFeed
{
    public string FeedName => "TopstepX";
    public int Priority { get; set; } = 1;
    
    public event EventHandler<MarketData>? OnDataReceived;
    public event EventHandler<Exception>? OnError;

    public async Task<bool> ConnectAsync()
    {
        await Task.Delay(100); // Simulate connection
        return true;
    }

    public async Task<MarketData?> GetMarketDataAsync(string symbol)
    {
        await Task.Delay(50); // Simulate network delay
        
        return new MarketData
        {
            Symbol = symbol,
            Price = 4500.00m + (decimal)(Random.Shared.NextDouble() * 10 - 5),
            Volume = 1000,
            Bid = 4499.75m,
            Ask = 4500.25m,
            Timestamp = DateTime.UtcNow,
            Source = FeedName
        };
    }

    public async Task<OrderBook?> GetOrderBookAsync(string symbol)
    {
        await Task.Delay(50);
        return new OrderBook { Symbol = symbol, Timestamp = DateTime.UtcNow };
    }
}

/// <summary>
/// Backup data feed implementation
/// </summary>
public class BackupDataFeed : IDataFeed
{
    public string FeedName => "Backup";
    public int Priority { get; set; } = 2;
    
    public event EventHandler<MarketData>? OnDataReceived;
    public event EventHandler<Exception>? OnError;

    public async Task<bool> ConnectAsync()
    {
        await Task.Delay(200); // Simulate slower connection
        return true;
    }

    public async Task<MarketData?> GetMarketDataAsync(string symbol)
    {
        await Task.Delay(100); // Simulate slower response
        
        return new MarketData
        {
            Symbol = symbol,
            Price = 4500.00m + (decimal)(Random.Shared.NextDouble() * 8 - 4),
            Volume = 800,
            Bid = 4499.50m,
            Ask = 4500.50m,
            Timestamp = DateTime.UtcNow,
            Source = FeedName
        };
    }

    public async Task<OrderBook?> GetOrderBookAsync(string symbol)
    {
        await Task.Delay(100);
        return new OrderBook { Symbol = symbol, Timestamp = DateTime.UtcNow };
    }
}