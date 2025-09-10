using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Text.Json;

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
        _ = Task.Run(async () =>
        {
            try
            {
                var symbols = new[] { "ES", "NQ", "YM", "RTY" }; // Common futures symbols
                
                foreach (var symbol in symbols)
                {
                    await CheckSymbolConsistencyAsync(symbol);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[DataConsistency] Error during consistency check");
            }
        });
    }

    private async Task CheckSymbolConsistencyAsync(string symbol)
    {
        try
        {
            var consistency = new DataConsistencyResult
            {
                Symbol = symbol,
                CheckTime = DateTime.UtcNow,
                FeedData = new Dictionary<string, MarketDataSnapshot>()
            };

            // Collect data from all healthy feeds
            var healthyFeeds = _dataFeeds.Where(f => 
                _feedHealth.TryGetValue(f.FeedName, out var h) && h.IsHealthy).ToList();

            if (healthyFeeds.Count < 2)
            {
                _logger.LogDebug("[DataConsistency] Insufficient feeds for consistency check: {Symbol}", symbol);
                return;
            }

            // Gather market data from each feed
            var tasks = healthyFeeds.Select(async feed =>
            {
                try
                {
                    var startTime = DateTime.UtcNow;
                    var data = await feed.GetMarketDataAsync(symbol);
                    var responseTime = DateTime.UtcNow - startTime;

                    if (data != null && ValidateMarketData(data))
                    {
                        consistency.FeedData[feed.FeedName] = new MarketDataSnapshot
                        {
                            FeedName = feed.FeedName,
                            Price = data.Price,
                            Bid = data.Bid,
                            Ask = data.Ask,
                            Volume = data.Volume,
                            Timestamp = data.Timestamp,
                            ResponseTime = responseTime,
                            DataAge = DateTime.UtcNow - data.Timestamp
                        };
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "[DataConsistency] Failed to get data from {FeedName} for {Symbol}", 
                        feed.FeedName, symbol);
                }
            });

            await Task.WhenAll(tasks);

            // Analyze consistency if we have enough data
            if (consistency.FeedData.Count >= 2)
            {
                AnalyzeDataConsistency(consistency);
                
                // Take action on inconsistencies
                if (!consistency.IsConsistent)
                {
                    await HandleDataInconsistencyAsync(consistency);
                }
                
                // Log periodic status
                if (DateTime.UtcNow.Second % 30 == 0) // Every 30 seconds
                {
                    LogConsistencyStatus(consistency);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DataConsistency] Error checking consistency for {Symbol}", symbol);
        }
    }

    private void AnalyzeDataConsistency(DataConsistencyResult consistency)
    {
        if (consistency.FeedData.Count < 2) return;

        var snapshots = consistency.FeedData.Values.ToList();
        
        // Analyze price consistency
        var prices = snapshots.Select(s => s.Price).ToList();
        var avgPrice = prices.Average();
        var maxDeviation = prices.Max(p => Math.Abs(p - avgPrice) / avgPrice);
        var priceStdDev = CalculateStandardDeviation(prices.Select(p => (double)p));

        // Analyze bid-ask consistency  
        var spreads = snapshots.Where(s => s.Ask > s.Bid).Select(s => s.Ask - s.Bid).ToList();
        var avgSpread = spreads.Any() ? spreads.Average() : 0m;
        var spreadDeviation = spreads.Any() ? spreads.Max(s => Math.Abs(s - avgSpread) / avgSpread) : 0m;

        // Analyze data freshness
        var dataAges = snapshots.Select(s => s.DataAge.TotalSeconds).ToList();
        var maxAge = dataAges.Max();
        var avgAge = dataAges.Average();

        // Set consistency metrics
        consistency.PriceDeviation = maxDeviation;
        consistency.PriceStandardDeviation = (decimal)priceStdDev;
        consistency.SpreadDeviation = spreadDeviation;
        consistency.MaxDataAge = TimeSpan.FromSeconds(maxAge);
        consistency.AverageDataAge = TimeSpan.FromSeconds(avgAge);

        // Determine overall consistency
        consistency.IsConsistent = 
            maxDeviation < 0.001m && // 0.1% price tolerance
            spreadDeviation < 0.05m && // 5% spread tolerance  
            maxAge < 30; // 30 second freshness tolerance

        // Identify outliers
        if (!consistency.IsConsistent)
        {
            // Find price outliers
            foreach (var snapshot in snapshots)
            {
                var deviation = Math.Abs(snapshot.Price - avgPrice) / avgPrice;
                if (deviation == maxDeviation && deviation > 0.001m)
                {
                    consistency.OutlierFeeds.Add(snapshot.FeedName);
                    consistency.Issues.Add($"Price outlier: {snapshot.FeedName} deviates by {deviation:P2}");
                }
            }

            // Find stale data
            foreach (var snapshot in snapshots)
            {
                if (snapshot.DataAge.TotalSeconds > 30)
                {
                    consistency.Issues.Add($"Stale data: {snapshot.FeedName} is {snapshot.DataAge.TotalSeconds:F1}s old");
                }
            }

            // Find slow feeds
            foreach (var snapshot in snapshots)
            {
                if (snapshot.ResponseTime.TotalMilliseconds > 500)
                {
                    consistency.Issues.Add($"Slow response: {snapshot.FeedName} took {snapshot.ResponseTime.TotalMilliseconds:F0}ms");
                }
            }
        }
    }

    private async Task HandleDataInconsistencyAsync(DataConsistencyResult consistency)
    {
        try
        {
            _logger.LogWarning("[DataConsistency] Inconsistency detected for {Symbol}: {Issues}", 
                consistency.Symbol, string.Join("; ", consistency.Issues));

            // Update feed health scores for outliers
            foreach (var outlierFeed in consistency.OutlierFeeds)
            {
                if (_feedHealth.TryGetValue(outlierFeed, out var health))
                {
                    health.DataQualityScore *= 0.95; // Reduce quality score
                    _logger.LogWarning("[DataConsistency] Reduced quality score for {FeedName} to {Score:F2}", 
                        outlierFeed, health.DataQualityScore);
                }
            }

            // Create consistency alert
            var alert = new
            {
                AlertType = "DATA_INCONSISTENCY",
                Symbol = consistency.Symbol,
                PriceDeviation = consistency.PriceDeviation,
                Issues = consistency.Issues,
                FeedCount = consistency.FeedData.Count,
                OutlierFeeds = consistency.OutlierFeeds,
                Timestamp = DateTime.UtcNow,
                Severity = consistency.PriceDeviation > 0.005m ? "HIGH" : "MEDIUM"
            };

            // Store alert (implement based on your alerting system)
            await StoreConsistencyAlertAsync(alert);

            // If deviation is severe, trigger failover
            if (consistency.PriceDeviation > 0.01m) // 1% deviation
            {
                _logger.LogError("[DataConsistency] SEVERE inconsistency detected for {Symbol}: {Deviation:P2}", 
                    consistency.Symbol, consistency.PriceDeviation);
                
                // Consider switching primary feed or halting trading
                await ConsiderFeedFailoverAsync(consistency);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DataConsistency] Error handling inconsistency for {Symbol}", consistency.Symbol);
        }
    }

    private void LogConsistencyStatus(DataConsistencyResult consistency)
    {
        var feedNames = string.Join(", ", consistency.FeedData.Keys);
        var avgPrice = consistency.FeedData.Values.Average(s => s.Price);
        
        if (consistency.IsConsistent)
        {
            _logger.LogDebug("[DataConsistency] ✅ {Symbol} consistent across {FeedCount} feeds: avg=${AvgPrice:F2}, deviation={Deviation:P3}", 
                consistency.Symbol, consistency.FeedData.Count, avgPrice, consistency.PriceDeviation);
        }
        else
        {
            _logger.LogInformation("[DataConsistency] ⚠️ {Symbol} inconsistent: deviation={Deviation:P2}, issues={IssueCount}", 
                consistency.Symbol, consistency.PriceDeviation, consistency.Issues.Count);
        }
    }

    private double CalculateStandardDeviation(IEnumerable<double> values)
    {
        var data = values.ToList();
        if (data.Count <= 1) return 0.0;
        
        var mean = data.Average();
        var variance = data.Sum(x => Math.Pow(x - mean, 2)) / (data.Count - 1);
        return Math.Sqrt(variance);
    }

    private async Task StoreConsistencyAlertAsync(object alert)
    {
        try
        {
            // Store in database or send to monitoring system
            _logger.LogWarning("[DataConsistency] ALERT: {Alert}", System.Text.Json.JsonSerializer.Serialize(alert));
            
            // Implement storage for consistency alerts
            await StoreConsistencyAlert(alert);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DataConsistency] Failed to store consistency alert");
        }
    }

    private async Task StoreConsistencyAlert(object alert)
    {
        try
        {
            // Store alert to file for monitoring system pickup
            var alertsDir = Path.Combine("logs", "consistency_alerts");
            Directory.CreateDirectory(alertsDir);
            
            var alertFile = Path.Combine(alertsDir, $"alert_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            var json = System.Text.Json.JsonSerializer.Serialize(alert, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(alertFile, json);
            
            _logger.LogDebug("[DataConsistency] Alert stored to {AlertFile}", alertFile);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DataConsistency] Failed to write consistency alert to file");
        }
    }

    private async Task ConsiderFeedFailoverAsync(DataConsistencyResult consistency)
    {
        try
        {
            // If primary feed is an outlier, switch to a consensus feed
            if (_primaryFeed != null && consistency.OutlierFeeds.Contains(_primaryFeed.FeedName))
            {
                // Find best consensus feed (non-outlier with highest quality score)
                var consensusFeed = _dataFeeds
                    .Where(f => !consistency.OutlierFeeds.Contains(f.FeedName))
                    .Where(f => _feedHealth.TryGetValue(f.FeedName, out var h) && h.IsHealthy)
                    .OrderByDescending(f => _feedHealth.GetValueOrDefault(f.FeedName)?.DataQualityScore ?? 0)
                    .FirstOrDefault();

                if (consensusFeed != null)
                {
                    _logger.LogWarning("[DataConsistency] Switching primary feed from {OldFeed} to {NewFeed} due to data inconsistency", 
                        _primaryFeed.FeedName, consensusFeed.FeedName);
                    
                    _primaryFeed = consensusFeed;
                    OnFeedFailover?.Invoke(this, consensusFeed.FeedName);
                }
            }
            
            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DataConsistency] Error during feed failover consideration");
        }
    }

    // Data structures for consistency checking
    public class DataConsistencyResult
    {
        public string Symbol { get; set; } = string.Empty;
        public DateTime CheckTime { get; set; }
        public Dictionary<string, MarketDataSnapshot> FeedData { get; set; } = new();
        public bool IsConsistent { get; set; }
        public decimal PriceDeviation { get; set; }
        public decimal PriceStandardDeviation { get; set; }
        public decimal SpreadDeviation { get; set; }
        public TimeSpan MaxDataAge { get; set; }
        public TimeSpan AverageDataAge { get; set; }
        public List<string> OutlierFeeds { get; set; } = new();
        public List<string> Issues { get; set; } = new();
    }

    public class MarketDataSnapshot
    {
        public string FeedName { get; set; } = string.Empty;
        public decimal Price { get; set; }
        public decimal Bid { get; set; }
        public decimal Ask { get; set; }
        public decimal Volume { get; set; }
        public DateTime Timestamp { get; set; }
        public TimeSpan ResponseTime { get; set; }
        public TimeSpan DataAge { get; set; }
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
    
#pragma warning disable CS0067 // Event is never used - reserved for future implementation
    public event EventHandler<MarketData>? OnDataReceived;
    public event EventHandler<Exception>? OnError;
#pragma warning restore CS0067

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
    
#pragma warning disable CS0067 // Event is never used - reserved for future implementation
    public event EventHandler<MarketData>? OnDataReceived;
    public event EventHandler<Exception>? OnError;
#pragma warning restore CS0067

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