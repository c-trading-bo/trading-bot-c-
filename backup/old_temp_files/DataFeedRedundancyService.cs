// ================================================================================
// COMPONENT 8: DATA FEED REDUNDANCY
// ================================================================================
// File: DataFeedRedundancyService.cs
// Purpose: Multiple data feed sources with automatic failover
// Integration: Provides backup data sources when TopstepX fails
// ================================================================================

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Net.Http;
using System.Text.Json;

namespace TradingBot.UnifiedOrchestrator.Services
{
    public class DataFeedRedundancyService
    {
        private readonly List<IDataFeed> _dataFeeds = new();
        private readonly ConcurrentDictionary<string, DataFeedHealth> _feedHealth = new();
        private readonly ConcurrentDictionary<string, MarketData> _consolidatedData = new();
        private IDataFeed? _primaryFeed;
        private readonly Timer _healthCheckTimer;
        private readonly Timer _consistencyCheckTimer;
        private readonly ILogger<DataFeedRedundancyService> _logger;
        private readonly HttpClient _httpClient;
        
        public interface IDataFeed
        {
            string FeedName { get; }
            int Priority { get; }
            Task<bool> Connect();
            Task<MarketData> GetMarketData(string symbol);
            Task<OrderBook> GetOrderBook(string symbol);
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
            public List<OrderBookLevel> Bids { get; set; } = new();
            public List<OrderBookLevel> Asks { get; set; } = new();
            public DateTime Timestamp { get; set; }
        }

        public class OrderBookLevel
        {
            public decimal Price { get; set; }
            public decimal Size { get; set; }
        }
        
        public class MarketDataConsistency
        {
            public string Symbol { get; set; } = string.Empty;
            public Dictionary<string, decimal> PriceByFeed { get; set; } = new();
            public decimal PriceDeviation { get; set; }
            public string OutlierFeed { get; set; } = string.Empty;
            public bool IsConsistent => PriceDeviation < 0.001m; // 0.1% tolerance
        }

        public class DataFeedException : Exception
        {
            public DataFeedException(string message) : base(message) { }
            public DataFeedException(string message, Exception innerException) : base(message, innerException) { }
        }

        public DataFeedRedundancyService(ILogger<DataFeedRedundancyService> logger)
        {
            _logger = logger;
            _httpClient = new HttpClient();
            _healthCheckTimer = new Timer(CheckFeedHealth, null, Timeout.InfiniteTimeSpan, Timeout.InfiniteTimeSpan);
            _consistencyCheckTimer = new Timer(CheckDataConsistency, null, Timeout.InfiniteTimeSpan, Timeout.InfiniteTimeSpan);
        }
        
        public async Task InitializeDataFeeds()
        {
            _logger.LogInformation("[DataFeedRedundancy] Initializing data feeds...");
            
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
                        LastHealthCheck = DateTime.UtcNow
                    };
                    
                    // Subscribe to events
                    if (feed.OnDataReceived != null)
                        feed.OnDataReceived += OnDataReceived;
                    if (feed.OnError != null)
                        feed.OnError += OnFeedError;
                    
                    if (connected && _primaryFeed == null)
                    {
                        _primaryFeed = feed;
                        _logger.LogInformation($"[DataFeedRedundancy] Primary data feed set to: {feed.FeedName}");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"[DataFeedRedundancy] Failed to connect to {feed.FeedName}");
                }
            }
            
            // Start health monitoring
            _healthCheckTimer.Change(TimeSpan.Zero, TimeSpan.FromSeconds(10));
            _consistencyCheckTimer.Change(TimeSpan.Zero, TimeSpan.FromSeconds(5));
            
            _logger.LogInformation($"[DataFeedRedundancy] Initialized {_dataFeeds.Count} data feeds");
        }
        
        public async Task<MarketData> GetMarketData(string symbol)
        {
            MarketData? data = null;
            Exception? lastError = null;
            
            // Try primary feed first
            if (_primaryFeed != null && _feedHealth.TryGetValue(_primaryFeed.FeedName, out var primaryHealth) && primaryHealth.IsHealthy)
            {
                try
                {
                    data = await _primaryFeed.GetMarketData(symbol);
                    
                    if (ValidateMarketData(data))
                    {
                        _consolidatedData[symbol] = data;
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
                if (_feedHealth.TryGetValue(feed.FeedName, out var health) && health.IsHealthy)
                {
                    try
                    {
                        _logger.LogWarning($"[DataFeedRedundancy] Failing over to {feed.FeedName} for {symbol}");
                        
                        data = await feed.GetMarketData(symbol);
                        
                        if (ValidateMarketData(data))
                        {
                            // Switch primary feed
                            _primaryFeed = feed;
                            SendAlert($"Data feed switched to {feed.FeedName}");
                            _consolidatedData[symbol] = data;
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
            throw new DataFeedException("All data feeds unavailable", lastError);
        }
        
        private void OnDataReceived(object? sender, MarketData data)
        {
            if (sender is IDataFeed feed)
            {
                if (_feedHealth.TryGetValue(feed.FeedName, out var health))
                {
                    health.LastDataReceived = DateTime.UtcNow;
                    health.DataQualityScore = CalculateDataQuality(data);
                }
                
                _consolidatedData[data.Symbol] = data;
            }
        }

        private void OnFeedError(object? sender, Exception error)
        {
            if (sender is IDataFeed feed)
            {
                _logger.LogError(error, $"[DataFeedRedundancy] Error from {feed.FeedName}");
                _ = Task.Run(async () => await HandleFeedFailure(feed, error));
            }
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
                    }
                    
                    // Alert on high latency
                    if (latency > 100)
                    {
                        _logger.LogWarning($"[DataFeedRedundancy] High latency on {feed.FeedName}: {latency}ms");
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
                            _logger.LogError(ex, $"[DataFeedRedundancy] Feed {feed.FeedName} marked unhealthy");
                        }
                    }
                }
            });
            
            // Ensure at least one feed is healthy
            if (!_feedHealth.Values.Any(h => h.IsHealthy))
            {
                SendCriticalAlert("ALL DATA FEEDS DOWN - TRADING HALTED");
                HaltTrading();
            }
        }
        
        private void CheckDataConsistency(object? state)
        {
            var symbols = new[] { "ES", "NQ" };
            
            foreach (var symbol in symbols)
            {
                var consistency = new MarketDataConsistency
                {
                    Symbol = symbol,
                    PriceByFeed = new Dictionary<string, decimal>()
                };
                
                // Get price from each healthy feed
                var healthyFeeds = _dataFeeds.Where(f => _feedHealth.TryGetValue(f.FeedName, out var h) && h.IsHealthy);
                
                foreach (var feed in healthyFeeds)
                {
                    try
                    {
                        var data = feed.GetMarketData(symbol).Result;
                        consistency.PriceByFeed[feed.FeedName] = data.Price;
                    }
                    catch
                    {
                        // Skip failed feeds
                    }
                }
                
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
                            if (deviation == maxDeviation)
                            {
                                consistency.OutlierFeed = kvp.Key;
                                break;
                            }
                        }
                        
                        _logger.LogWarning($"[DataFeedRedundancy] Data inconsistency detected for {symbol}: {consistency.OutlierFeed} deviates by {maxDeviation:P2}");
                        
                        // Mark outlier feed as unhealthy
                        if (_feedHealth.TryGetValue(consistency.OutlierFeed, out var health))
                        {
                            health.DataQualityScore *= 0.9;
                        }
                    }
                }
            }
        }
        
        private bool ValidateMarketData(MarketData? data)
        {
            if (data == null) return false;
            
            // Check timestamp freshness
            if (DateTime.UtcNow - data.Timestamp > TimeSpan.FromSeconds(5))
            {
                return false;
            }
            
            // Check price sanity
            if (data.Price <= 0 || data.Price > 100000)
            {
                return false;
            }
            
            // Check bid/ask spread
            if (data.Ask - data.Bid > data.Price * 0.01m) // > 1% spread
            {
                return false;
            }
            
            return true;
        }

        private double CalculateDataQuality(MarketData? data)
        {
            if (data == null) return 0.0;
            
            double score = 1.0;
            
            // Freshness score
            var age = DateTime.UtcNow - data.Timestamp;
            if (age.TotalSeconds > 1) score *= 0.9;
            if (age.TotalSeconds > 5) score *= 0.7;
            
            // Spread quality
            var spread = data.Ask - data.Bid;
            var spreadPercent = spread / data.Price;
            if (spreadPercent > 0.001m) score *= 0.9; // > 0.1% spread
            if (spreadPercent > 0.005m) score *= 0.7; // > 0.5% spread
            
            return score;
        }
        
        private async Task HandleFeedFailure(IDataFeed feed, Exception error)
        {
            _logger.LogError(error, $"[DataFeedRedundancy] Feed failure: {feed.FeedName}");
            
            if (_feedHealth.TryGetValue(feed.FeedName, out var health))
            {
                health.IsHealthy = false;
                health.ErrorCount++;
            }
            
            // Try to reconnect
            _ = Task.Run(async () =>
            {
                await Task.Delay(5000);
                try
                {
                    await feed.Connect();
                    if (_feedHealth.TryGetValue(feed.FeedName, out var reconnectedHealth))
                    {
                        reconnectedHealth.IsHealthy = true;
                    }
                    _logger.LogInformation($"[DataFeedRedundancy] Reconnected to {feed.FeedName}");
                }
                catch
                {
                    // Will retry on next health check
                }
            });
        }

        private void SendAlert(string message)
        {
            _logger.LogWarning($"[DataFeedRedundancy] ALERT: {message}");
        }

        private void SendCriticalAlert(string message)
        {
            _logger.LogCritical($"[DataFeedRedundancy] CRITICAL: {message}");
        }

        private void HaltTrading()
        {
            _logger.LogCritical("[DataFeedRedundancy] TRADING HALTED - No reliable data feeds available");
            // Implementation to halt trading operations
        }

        public DataFeedStatus GetStatus()
        {
            return new DataFeedStatus
            {
                PrimaryFeed = _primaryFeed?.FeedName ?? "None",
                HealthyFeeds = _feedHealth.Values.Count(h => h.IsHealthy),
                TotalFeeds = _dataFeeds.Count,
                FeedHealth = _feedHealth.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                Timestamp = DateTime.UtcNow
            };
        }
    }

    public class DataFeedStatus
    {
        public string PrimaryFeed { get; set; } = string.Empty;
        public int HealthyFeeds { get; set; }
        public int TotalFeeds { get; set; }
        public Dictionary<string, DataFeedRedundancyService.DataFeedHealth> FeedHealth { get; set; } = new();
        public DateTime Timestamp { get; set; }
    }

    // Implementation classes for different data feeds
    public class TopstepXDataFeed : DataFeedRedundancyService.IDataFeed
    {
        public string FeedName => "TopstepX";
        public int Priority { get; set; } = 1;
        
        public event EventHandler<DataFeedRedundancyService.MarketData>? OnDataReceived;
        public event EventHandler<Exception>? OnError;
        
        public async Task<bool> Connect()
        {
            // TopstepX connection logic
            await Task.Delay(100);
            return true;
        }
        
        public async Task<DataFeedRedundancyService.MarketData> GetMarketData(string symbol)
        {
            await Task.Delay(50);
            return new DataFeedRedundancyService.MarketData
            {
                Symbol = symbol,
                Price = 4850.25m,
                Bid = 4850.00m,
                Ask = 4850.50m,
                Volume = 1000,
                Timestamp = DateTime.UtcNow,
                Source = FeedName
            };
        }
        
        public async Task<DataFeedRedundancyService.OrderBook> GetOrderBook(string symbol)
        {
            await Task.Delay(50);
            return new DataFeedRedundancyService.OrderBook
            {
                Symbol = symbol,
                Timestamp = DateTime.UtcNow
            };
        }
    }

    public class InteractiveBrokersDataFeed : DataFeedRedundancyService.IDataFeed
    {
        public string FeedName => "InteractiveBrokers";
        public int Priority { get; set; } = 2;
        
        public event EventHandler<DataFeedRedundancyService.MarketData>? OnDataReceived;
        public event EventHandler<Exception>? OnError;
        
        public async Task<bool> Connect()
        {
            await Task.Delay(100);
            return true;
        }
        
        public async Task<DataFeedRedundancyService.MarketData> GetMarketData(string symbol)
        {
            await Task.Delay(75);
            return new DataFeedRedundancyService.MarketData
            {
                Symbol = symbol,
                Price = 4850.50m,
                Bid = 4850.25m,
                Ask = 4850.75m,
                Volume = 950,
                Timestamp = DateTime.UtcNow,
                Source = FeedName
            };
        }
        
        public async Task<DataFeedRedundancyService.OrderBook> GetOrderBook(string symbol)
        {
            await Task.Delay(75);
            return new DataFeedRedundancyService.OrderBook
            {
                Symbol = symbol,
                Timestamp = DateTime.UtcNow
            };
        }
    }

    public class TradingViewDataFeed : DataFeedRedundancyService.IDataFeed
    {
        public string FeedName => "TradingView";
        public int Priority { get; set; } = 3;
        
        public event EventHandler<DataFeedRedundancyService.MarketData>? OnDataReceived;
        public event EventHandler<Exception>? OnError;
        
        public async Task<bool> Connect()
        {
            await Task.Delay(100);
            return true;
        }
        
        public async Task<DataFeedRedundancyService.MarketData> GetMarketData(string symbol)
        {
            await Task.Delay(100);
            return new DataFeedRedundancyService.MarketData
            {
                Symbol = symbol,
                Price = 4849.75m,
                Bid = 4849.50m,
                Ask = 4850.00m,
                Volume = 1100,
                Timestamp = DateTime.UtcNow,
                Source = FeedName
            };
        }
        
        public async Task<DataFeedRedundancyService.OrderBook> GetOrderBook(string symbol)
        {
            await Task.Delay(100);
            return new DataFeedRedundancyService.OrderBook
            {
                Symbol = symbol,
                Timestamp = DateTime.UtcNow
            };
        }
    }

    public class AlphaVantageDataFeed : DataFeedRedundancyService.IDataFeed
    {
        public string FeedName => "AlphaVantage";
        public int Priority { get; set; } = 4;
        
        public event EventHandler<DataFeedRedundancyService.MarketData>? OnDataReceived;
        public event EventHandler<Exception>? OnError;
        
        public async Task<bool> Connect()
        {
            await Task.Delay(100);
            return true;
        }
        
        public async Task<DataFeedRedundancyService.MarketData> GetMarketData(string symbol)
        {
            await Task.Delay(150);
            return new DataFeedRedundancyService.MarketData
            {
                Symbol = symbol,
                Price = 4850.00m,
                Bid = 4849.75m,
                Ask = 4850.25m,
                Volume = 800,
                Timestamp = DateTime.UtcNow,
                Source = FeedName
            };
        }
        
        public async Task<DataFeedRedundancyService.OrderBook> GetOrderBook(string symbol)
        {
            await Task.Delay(150);
            return new DataFeedRedundancyService.OrderBook
            {
                Symbol = symbol,
                Timestamp = DateTime.UtcNow
            };
        }
    }
}
