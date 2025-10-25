using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.ML.Services;

/// <summary>
/// News Sentiment Analysis Service for alternative data integration
/// Addresses gap in HEDGE_FUND_GAP_ANALYSIS.md - Section 2 "Alternative Data Sources"
/// Provides basic news sentiment using free sources (GDELT, Reddit) and FinBERT
/// </summary>
public interface INewsSentimentService
{
    /// <summary>
    /// Get current sentiment score for a symbol
    /// </summary>
    Task<SentimentScore> GetSentimentAsync(
        string symbol,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get aggregated sentiment from multiple sources
    /// </summary>
    Task<AggregatedSentiment> GetAggregatedSentimentAsync(
        string symbol,
        TimeSpan lookbackPeriod,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get sentiment trend (bullish/bearish/neutral)
    /// </summary>
    Task<SentimentTrend> GetSentimentTrendAsync(
        string symbol,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if news sentiment data is available
    /// </summary>
    bool IsAvailable();
}

/// <summary>
/// Production implementation of news sentiment service
/// Integrates with Python-based FinBERT model for sentiment analysis
/// </summary>
public class NewsSentimentService : INewsSentimentService
{
    private readonly ILogger<NewsSentimentService> _logger;
    private readonly string _sentimentDataPath;
    private readonly bool _enabled;
    private readonly Dictionary<string, SentimentScore> _cache;
    private readonly SemaphoreSlim _cacheLock;
    private DateTime _lastUpdate;

    public NewsSentimentService(
        ILogger<NewsSentimentService> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _enabled = Environment.GetEnvironmentVariable("NEWS_SENTIMENT_ENABLED") != "0";
        
        _sentimentDataPath = Environment.GetEnvironmentVariable("NEWS_SENTIMENT_DATA_PATH") 
            ?? Path.Combine("./data", "news_sentiment");

        _cache = new Dictionary<string, SentimentScore>();
        _cacheLock = new SemaphoreSlim(1, 1);
        _lastUpdate = DateTime.MinValue;

        if (_enabled)
        {
            Directory.CreateDirectory(_sentimentDataPath);
            _logger.LogInformation(
                "News Sentiment Service initialized. Data path: {Path}",
                _sentimentDataPath);
        }
        else
        {
            _logger.LogInformation("News Sentiment Service disabled via configuration");
        }
    }

    public async Task<SentimentScore> GetSentimentAsync(
        string symbol,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            return SentimentScore.Neutral;
        }

        try
        {
            await RefreshCacheIfNeededAsync(cancellationToken).ConfigureAwait(false);

            await _cacheLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (_cache.TryGetValue(symbol, out var score))
                {
                    return score;
                }

                return SentimentScore.Neutral;
            }
            finally
            {
                _cacheLock.Release();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error getting sentiment for symbol: {Symbol}",
                symbol);
            return SentimentScore.Neutral;
        }
    }

    public async Task<AggregatedSentiment> GetAggregatedSentimentAsync(
        string symbol,
        TimeSpan lookbackPeriod,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            return new AggregatedSentiment
            {
                Symbol = symbol,
                OverallScore = 0.0,
                Confidence = 0.0,
                Sources = new List<string>()
            };
        }

        try
        {
            var sentimentFile = Path.Combine(
                _sentimentDataPath,
                $"{symbol}_aggregated.json");

            if (!File.Exists(sentimentFile))
            {
                return new AggregatedSentiment
                {
                    Symbol = symbol,
                    OverallScore = 0.0,
                    Confidence = 0.0,
                    Sources = new List<string>()
                };
            }

            var json = await File.ReadAllTextAsync(sentimentFile, cancellationToken)
                .ConfigureAwait(false);
            
            var sentiment = JsonSerializer.Deserialize<AggregatedSentiment>(json);
            return sentiment ?? new AggregatedSentiment
            {
                Symbol = symbol,
                OverallScore = 0.0,
                Confidence = 0.0,
                Sources = new List<string>()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error getting aggregated sentiment for symbol: {Symbol}",
                symbol);
            return new AggregatedSentiment
            {
                Symbol = symbol,
                OverallScore = 0.0,
                Confidence = 0.0,
                Sources = new List<string>()
            };
        }
    }

    public async Task<SentimentTrend> GetSentimentTrendAsync(
        string symbol,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            return SentimentTrend.Neutral;
        }

        try
        {
            var sentiment = await GetSentimentAsync(symbol, cancellationToken)
                .ConfigureAwait(false);

            if (sentiment.Score > 0.3)
            {
                return SentimentTrend.Bullish;
            }
            else if (sentiment.Score < -0.3)
            {
                return SentimentTrend.Bearish;
            }
            else
            {
                return SentimentTrend.Neutral;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error getting sentiment trend for symbol: {Symbol}",
                symbol);
            return SentimentTrend.Neutral;
        }
    }

    public bool IsAvailable()
    {
        return _enabled;
    }

    private async Task RefreshCacheIfNeededAsync(CancellationToken cancellationToken)
    {
        // Refresh cache every 5 minutes
        if (DateTime.UtcNow - _lastUpdate < TimeSpan.FromMinutes(5))
        {
            return;
        }

        await _cacheLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Double-check after acquiring lock
            if (DateTime.UtcNow - _lastUpdate < TimeSpan.FromMinutes(5))
            {
                return;
            }

            var latestFile = Path.Combine(_sentimentDataPath, "latest_news_sentiment.json");
            if (!File.Exists(latestFile))
            {
                _logger.LogWarning("Latest sentiment file not found: {Path}", latestFile);
                return;
            }

            var json = await File.ReadAllTextAsync(latestFile, cancellationToken)
                .ConfigureAwait(false);
            
            var sentiments = JsonSerializer.Deserialize<Dictionary<string, SentimentScore>>(json);
            if (sentiments != null)
            {
                _cache.Clear();
                foreach (var (symbol, score) in sentiments)
                {
                    _cache[symbol] = score;
                }
                
                _lastUpdate = DateTime.UtcNow;
                
                _logger.LogDebug(
                    "Refreshed sentiment cache with {Count} symbols",
                    _cache.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error refreshing sentiment cache");
        }
        finally
        {
            _cacheLock.Release();
        }
    }
}

/// <summary>
/// Sentiment score for a symbol
/// </summary>
public class SentimentScore
{
    public double Score { get; set; } // -1.0 (very bearish) to +1.0 (very bullish)
    public double Confidence { get; set; } // 0.0 to 1.0
    public DateTime Timestamp { get; set; }
    public string Source { get; set; } = string.Empty;

    public static SentimentScore Neutral => new()
    {
        Score = 0.0,
        Confidence = 0.0,
        Timestamp = DateTime.UtcNow,
        Source = "none"
    };
}

/// <summary>
/// Aggregated sentiment from multiple sources
/// </summary>
public class AggregatedSentiment
{
    public string Symbol { get; set; } = string.Empty;
    public double OverallScore { get; set; }
    public double Confidence { get; set; }
    public List<string> Sources { get; set; } = new();
    public Dictionary<string, double> SourceScores { get; set; } = new();
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Sentiment trend classification
/// </summary>
public enum SentimentTrend
{
    StronglyBearish = -2,
    Bearish = -1,
    Neutral = 0,
    Bullish = 1,
    StronglyBullish = 2
}
