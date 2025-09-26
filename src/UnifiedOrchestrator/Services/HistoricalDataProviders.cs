using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Backtest;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Features-based historical data provider (first tier in hierarchy)
/// Provides historical data from preprocessed feature files
/// </summary>
internal sealed class FeaturesHistoricalProvider : IHistoricalDataProvider
{
    private readonly ILogger<FeaturesHistoricalProvider> _logger;
    private readonly string _featuresRoot;
    
    public FeaturesHistoricalProvider(ILogger<FeaturesHistoricalProvider> logger, IConfiguration configuration)
    {
        _logger = logger;
        _featuresRoot = configuration.GetValue("Paths:FeaturesRoot", "datasets/features");
    }
    
    public async Task<IAsyncEnumerable<Quote>> GetHistoricalQuotesAsync(
        string symbol, 
        DateTime startTime, 
        DateTime endTime, 
        CancellationToken cancellationToken = default)
    {
        // Remove placeholder - return the actual implementation
        return GetQuotesFromFeaturesAsync(symbol, startTime, endTime, cancellationToken);
    }
    
    private async IAsyncEnumerable<Quote> GetQuotesFromFeaturesAsync(
        string symbol, 
        DateTime startTime, 
        DateTime endTime, 
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var featureFile = Path.Combine(_featuresRoot, $"{symbol.ToLowerInvariant()}_features.json");
        if (!File.Exists(featureFile))
        {
            _logger.LogWarning("Feature file not found: {FeatureFile}", featureFile);
            yield break;
        }
        
        try
        {
            await using var fileStream = new FileStream(featureFile, FileMode.Open, FileAccess.Read);
            var features = await JsonSerializer.DeserializeAsync<FeatureData[]>(fileStream, cancellationToken: cancellationToken).ConfigureAwait(false);
            
            if (features == null)
            {
                yield break;
            }
            
            foreach (var feature in features.Where(f => f.Time >= startTime && f.Time <= endTime))
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return ConvertFeatureToQuote(symbol, feature);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading features from {FeatureFile}", featureFile);
        }
    }
    
    public async Task<bool> IsDataAvailableAsync(
        string symbol, 
        DateTime startTime, 
        DateTime endTime, 
        CancellationToken cancellationToken = default)
    {
        var featureFile = Path.Combine(_featuresRoot, $"{symbol.ToLowerInvariant()}_features.json");
        return File.Exists(featureFile);
    }
    
    public async Task<(DateTime EarliestData, DateTime LatestData)> GetDataRangeAsync(
        string symbol, 
        CancellationToken cancellationToken = default)
    {
        var featureFile = Path.Combine(_featuresRoot, $"{symbol.ToLowerInvariant()}_features.json");
        if (!File.Exists(featureFile))
        {
            return (DateTime.MaxValue, DateTime.MinValue);
        }
        
        try
        {
            await using var fileStream = new FileStream(featureFile, FileMode.Open, FileAccess.Read);
            var features = await JsonSerializer.DeserializeAsync<FeatureData[]>(fileStream, cancellationToken: cancellationToken).ConfigureAwait(false);
            
            if (features?.Length > 0)
            {
                return (features.Min(f => f.Time), features.Max(f => f.Time));
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading feature file {FeatureFile}", featureFile);
        }
        
        return (DateTime.MaxValue, DateTime.MinValue);
    }
    
    private static Quote ConvertFeatureToQuote(string symbol, FeatureData feature)
    {
        return new Quote(
            Time: feature.Time,
            Symbol: symbol,
            Bid: feature.Price - 0.25m,
            Ask: feature.Price + 0.25m,
            Last: feature.Price,
            Volume: (int)(feature.Volume ?? 1000),
            Open: feature.Open ?? feature.Price,
            High: feature.High ?? feature.Price,
            Low: feature.Low ?? feature.Price,
            Close: feature.Price
        );
    }
}

/// <summary>
/// Local quotes historical data provider (second tier in hierarchy)
/// Provides historical data from local quote files
/// </summary>
internal sealed class LocalQuotesProvider : IHistoricalDataProvider
{
    private readonly ILogger<LocalQuotesProvider> _logger;
    private readonly string _quotesRoot;
    
    public LocalQuotesProvider(ILogger<LocalQuotesProvider> logger, IConfiguration configuration)
    {
        _logger = logger;
        _quotesRoot = configuration.GetValue("Paths:QuotesRoot", "datasets/quotes");
    }
    
    public async Task<IAsyncEnumerable<Quote>> GetHistoricalQuotesAsync(
        string symbol, 
        DateTime startTime, 
        DateTime endTime, 
        CancellationToken cancellationToken = default)
    {
        return GetQuotesFromLocalFilesAsync(symbol, startTime, endTime, cancellationToken);
    }
    
    private async IAsyncEnumerable<Quote> GetQuotesFromLocalFilesAsync(
        string symbol, 
        DateTime startTime, 
        DateTime endTime, 
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var quoteFile = Path.Combine(_quotesRoot, $"{symbol.ToLowerInvariant()}_quotes.json");
        if (!File.Exists(quoteFile))
        {
            _logger.LogWarning("Quote file not found: {QuoteFile}", quoteFile);
            yield break;
        }
        
        try
        {
            await using var fileStream = new FileStream(quoteFile, FileMode.Open, FileAccess.Read);
            var quotes = await JsonSerializer.DeserializeAsync<Quote[]>(fileStream, cancellationToken: cancellationToken).ConfigureAwait(false);
            
            if (quotes == null)
            {
                yield break;
            }
            
            foreach (var quote in quotes.Where(q => q.Time >= startTime && q.Time <= endTime))
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return quote;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading quotes from {QuoteFile}", quoteFile);
        }
    }
    
    public async Task<bool> IsDataAvailableAsync(
        string symbol, 
        DateTime startTime, 
        DateTime endTime, 
        CancellationToken cancellationToken = default)
    {
        var quoteFile = Path.Combine(_quotesRoot, $"{symbol.ToLowerInvariant()}_quotes.json");
        return File.Exists(quoteFile);
    }
    
    public async Task<(DateTime EarliestData, DateTime LatestData)> GetDataRangeAsync(
        string symbol, 
        CancellationToken cancellationToken = default)
    {
        var quoteFile = Path.Combine(_quotesRoot, $"{symbol.ToLowerInvariant()}_quotes.json");
        if (!File.Exists(quoteFile))
        {
            return (DateTime.MaxValue, DateTime.MinValue);
        }
        
        try
        {
            await using var fileStream = new FileStream(quoteFile, FileMode.Open, FileAccess.Read);
            var quotes = await JsonSerializer.DeserializeAsync<Quote[]>(fileStream, cancellationToken: cancellationToken).ConfigureAwait(false);
            
            if (quotes?.Length > 0)
            {
                return (quotes.Min(q => q.Time), quotes.Max(q => q.Time));
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading quote file {QuoteFile}", quoteFile);
        }
        
        return (DateTime.MaxValue, DateTime.MinValue);
    }
}

/// <summary>
/// Historical data resolver interface
/// </summary>
internal interface IHistoricalDataResolver : IHistoricalDataProvider
{
}

/// <summary>
/// Data resolver that prioritizes multiple historical data sources
/// Features → Quotes → TopstepX (fallback hierarchy)
/// </summary>
internal sealed class HistoricalDataResolver : IHistoricalDataResolver
{
    private readonly ILogger<HistoricalDataResolver> _logger;
    private readonly IEnumerable<IHistoricalDataProvider> _providers;
    
    public HistoricalDataResolver(ILogger<HistoricalDataResolver> logger, IEnumerable<IHistoricalDataProvider> providers)
    {
        _logger = logger;
        _providers = providers;
    }
    
    public async Task<IAsyncEnumerable<Quote>> GetHistoricalQuotesAsync(
        string symbol, 
        DateTime startTime, 
        DateTime endTime, 
        CancellationToken cancellationToken = default)
    {
        foreach (var provider in _providers)
        {
            try
            {
                if (await provider.IsDataAvailableAsync(symbol, startTime, endTime, cancellationToken).ConfigureAwait(false))
                {
                    _logger.LogInformation("Using provider {ProviderType} for {Symbol}", provider.GetType().Name, symbol);
                    return await provider.GetHistoricalQuotesAsync(symbol, startTime, endTime, cancellationToken).ConfigureAwait(false);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Provider {ProviderType} failed for {Symbol}", provider.GetType().Name, symbol);
            }
        }
        
        throw new InvalidOperationException($"No historical data provider available for {symbol} in range {startTime} to {endTime}");
    }
    
    public async Task<bool> IsDataAvailableAsync(
        string symbol, 
        DateTime startTime, 
        DateTime endTime, 
        CancellationToken cancellationToken = default)
    {
        foreach (var provider in _providers)
        {
            try
            {
                if (await provider.IsDataAvailableAsync(symbol, startTime, endTime, cancellationToken).ConfigureAwait(false))
                {
                    return true;
                }
            }
            catch
            {
                // Continue to next provider
            }
        }
        
        return false;
    }
    
    public async Task<(DateTime EarliestData, DateTime LatestData)> GetDataRangeAsync(
        string symbol, 
        CancellationToken cancellationToken = default)
    {
        DateTime earliestOverall = DateTime.MaxValue;
        DateTime latestOverall = DateTime.MinValue;
        
        foreach (var provider in _providers)
        {
            try
            {
                var (earliest, latest) = await provider.GetDataRangeAsync(symbol, cancellationToken).ConfigureAwait(false);
                if (earliest < earliestOverall) earliestOverall = earliest;
                if (latest > latestOverall) latestOverall = latest;
            }
            catch
            {
                // Continue to next provider
            }
        }
        
        return (earliestOverall, latestOverall);
    }
}

/// <summary>
/// Feature data structure for ML-ready historical data
/// </summary>
internal record FeatureData(
    DateTime Time,
    decimal Price,
    decimal? Volume = null,
    decimal? Open = null,
    decimal? High = null,
    decimal? Low = null
);