using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Data;

/// <summary>
/// Historical Data Provider for Lab training
/// Manages 90-day historical bar data with Parquet caching
/// Terminal does NOT use this - it uses real-time data only
/// </summary>
public sealed class HistoricalDataProvider
{
    private readonly ILogger<HistoricalDataProvider> _logger;
    private readonly string _cacheDirectory;
    private readonly SemaphoreSlim _refreshLock = new(1, 1);
    
    // Validation constants
    private const int ExpectedBarsPerDay = 390; // 6.5 trading hours * 60 minutes
    private const decimal MaxPriceGapPercent = 0.10m; // 10% max gap between bars
    private const int HistoricalDays = 90;

    public HistoricalDataProvider(ILogger<HistoricalDataProvider> logger, string? cacheDirectory = null)
    {
        _logger = logger;
        _cacheDirectory = cacheDirectory ?? Path.Combine(Directory.GetCurrentDirectory(), "data", "historical");
        
        // Ensure cache directories exist
        Directory.CreateDirectory(_cacheDirectory);
        Directory.CreateDirectory(Path.Combine(_cacheDirectory, "ES"));
        Directory.CreateDirectory(Path.Combine(_cacheDirectory, "NQ"));
        
        _logger.LogInformation("HistoricalDataProvider initialized - Cache: {CacheDirectory}", _cacheDirectory);
    }

    /// <summary>
    /// Download historical bars from TopstepX Historical API
    /// Only called by Lab during Saturday refresh
    /// </summary>
    public async Task<List<HistoricalBar>> DownloadHistoricalBarsAsync(
        string symbol, 
        DateTime from, 
        DateTime to, 
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Downloading historical bars for {Symbol} from {From:yyyy-MM-dd} to {To:yyyy-MM-dd}", 
            symbol, from, to);

        try
        {
            // TODO: Integration with TopstepX Historical API
            // For now, return empty list to avoid breaking build
            // This will be implemented when TopstepX Historical API client is available
            
            _logger.LogWarning("TopstepX Historical API integration pending - returning empty dataset");
            await Task.CompletedTask.ConfigureAwait(false);
            return new List<HistoricalBar>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to download historical bars for {Symbol}", symbol);
            throw;
        }
    }

    /// <summary>
    /// Retrieve cached bars from local Parquet storage
    /// Used by Lab training to load historical data
    /// </summary>
    public async Task<List<HistoricalBar>> GetCachedBarsAsync(
        string symbol, 
        DateTime from, 
        DateTime to, 
        CancellationToken cancellationToken = default)
    {
        _logger.LogDebug("Loading cached bars for {Symbol} from {From:yyyy-MM-dd} to {To:yyyy-MM-dd}", 
            symbol, from, to);

        var bars = new List<HistoricalBar>();
        var symbolPath = Path.Combine(_cacheDirectory, symbol);

        if (!Directory.Exists(symbolPath))
        {
            _logger.LogWarning("No cached data found for {Symbol} at {SymbolPath}", symbol, symbolPath);
            return bars;
        }

        // Get all Parquet files in date range
        var files = Directory.GetFiles(symbolPath, "*.parquet")
            .Select(f => new
            {
                Path = f,
                Date = ParseDateFromFilename(Path.GetFileNameWithoutExtension(f))
            })
            .Where(f => f.Date >= from.Date && f.Date <= to.Date)
            .OrderBy(f => f.Date)
            .ToList();

        foreach (var file in files)
        {
            try
            {
                var dailyBars = await LoadParquetFileAsync(file.Path, cancellationToken).ConfigureAwait(false);
                bars.AddRange(dailyBars.Where(b => b.Time >= from && b.Time <= to));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load Parquet file {File}", file.Path);
            }
        }

        _logger.LogInformation("Loaded {Count} cached bars for {Symbol}", bars.Count, symbol);
        return bars;
    }

    /// <summary>
    /// Refresh cache with latest data (runs on Saturday)
    /// Downloads latest bars and validates data integrity
    /// </summary>
    public async Task RefreshCacheAsync(CancellationToken cancellationToken = default)
    {
        await _refreshLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            _logger.LogInformation("Starting historical data cache refresh (Saturday schedule)");

            var symbols = new[] { "ES", "NQ" };
            var to = DateTime.UtcNow.Date;
            var from = to.AddDays(-HistoricalDays);

            foreach (var symbol in symbols)
            {
                _logger.LogInformation("Refreshing {Symbol} data from {From:yyyy-MM-dd} to {To:yyyy-MM-dd}", 
                    symbol, from, to);

                try
                {
                    // Download latest data
                    var bars = await DownloadHistoricalBarsAsync(symbol, from, to, cancellationToken).ConfigureAwait(false);

                    if (bars.Count > 0)
                    {
                        // Validate data quality
                        var validationResult = await ValidateDataQualityAsync(symbol, bars, cancellationToken).ConfigureAwait(false);
                        
                        if (!validationResult.IsValid)
                        {
                            _logger.LogWarning("Data quality validation failed for {Symbol}: {Reason}", 
                                symbol, string.Join(", ", validationResult.Errors));
                            continue;
                        }

                        // Cache validated data
                        await CacheBarsAsync(symbol, bars, cancellationToken).ConfigureAwait(false);
                        
                        _logger.LogInformation("✅ Successfully refreshed {Symbol} cache with {Count} bars", 
                            symbol, bars.Count);
                    }
                    else
                    {
                        _logger.LogWarning("No data downloaded for {Symbol}", symbol);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to refresh cache for {Symbol}", symbol);
                }
            }

            _logger.LogInformation("Historical data cache refresh completed");
        }
        finally
        {
            _refreshLock.Release();
        }
    }

    /// <summary>
    /// Validate data quality - check for gaps, outliers, correct bar count
    /// </summary>
    public async Task<ValidationResult> ValidateDataQualityAsync(
        string symbol,
        List<HistoricalBar> bars,
        CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var result = new ValidationResult { IsValid = true };

        if (bars.Count == 0)
        {
            result.IsValid = false;
            result.Errors.Add("No bars provided for validation");
            return result;
        }

        // Check for sufficient data
        var tradingDays = bars.Select(b => b.Time.Date).Distinct().Count();
        var expectedMinDays = HistoricalDays * 0.95; // Allow 5% missing days (holidays, etc.)
        
        if (tradingDays < expectedMinDays)
        {
            result.Warnings.Add($"Only {tradingDays} trading days found, expected at least {expectedMinDays:F0}");
        }

        // Check bars per day (should be around 390 for 1-minute bars)
        var barsPerDay = bars.Count / (decimal)tradingDays;
        if (barsPerDay < ExpectedBarsPerDay * 0.9m || barsPerDay > ExpectedBarsPerDay * 1.1m)
        {
            result.Warnings.Add($"Unexpected bars per day: {barsPerDay:F0} (expected ~{ExpectedBarsPerDay})");
        }

        // Check for large gaps in data
        var sortedBars = bars.OrderBy(b => b.Time).ToList();
        for (int i = 1; i < sortedBars.Count; i++)
        {
            var timeDiff = sortedBars[i].Time - sortedBars[i - 1].Time;
            
            // Flag gaps > 1 hour during trading hours
            if (timeDiff.TotalHours > 1 && !IsOvernightGap(sortedBars[i - 1].Time, sortedBars[i].Time))
            {
                result.Warnings.Add($"Large time gap at {sortedBars[i].Time:yyyy-MM-dd HH:mm}: {timeDiff.TotalMinutes:F0} minutes");
            }

            // Check for price outliers
            if (sortedBars[i - 1].Close > 0)
            {
                var priceChange = Math.Abs(sortedBars[i].Open - sortedBars[i - 1].Close) / sortedBars[i - 1].Close;
                if (priceChange > MaxPriceGapPercent)
                {
                    result.Warnings.Add($"Large price gap at {sortedBars[i].Time:yyyy-MM-dd HH:mm}: {priceChange:P2}");
                }
            }
        }

        // Check for invalid prices
        var invalidBars = bars.Where(b => b.Open <= 0 || b.High <= 0 || b.Low <= 0 || b.Close <= 0).ToList();
        if (invalidBars.Count > 0)
        {
            result.IsValid = false;
            result.Errors.Add($"Found {invalidBars.Count} bars with invalid prices (≤ 0)");
        }

        // Check OHLC consistency
        var inconsistentBars = bars.Where(b => 
            b.High < b.Low || 
            b.High < b.Open || 
            b.High < b.Close ||
            b.Low > b.Open ||
            b.Low > b.Close).ToList();
        
        if (inconsistentBars.Count > 0)
        {
            result.IsValid = false;
            result.Errors.Add($"Found {inconsistentBars.Count} bars with inconsistent OHLC values");
        }

        _logger.LogInformation("Data quality validation for {Symbol}: {Status} ({ErrorCount} errors, {WarningCount} warnings)", 
            symbol, result.IsValid ? "PASSED" : "FAILED", result.Errors.Count, result.Warnings.Count);

        return result;
    }

    #region Private Methods

    private async Task CacheBarsAsync(string symbol, List<HistoricalBar> bars, CancellationToken cancellationToken)
    {
        // Group bars by date and save to Parquet files
        var barsByDate = bars.GroupBy(b => b.Time.Date).OrderBy(g => g.Key);

        foreach (var dateGroup in barsByDate)
        {
            var filename = $"{dateGroup.Key:yyyy-MM-dd}.parquet";
            var filePath = Path.Combine(_cacheDirectory, symbol, filename);

            try
            {
                await SaveParquetFileAsync(filePath, dateGroup.ToList(), cancellationToken).ConfigureAwait(false);
                _logger.LogDebug("Cached {Count} bars to {File}", dateGroup.Count(), filename);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to cache bars to {File}", filePath);
            }
        }
    }

    private async Task<List<HistoricalBar>> LoadParquetFileAsync(string filePath, CancellationToken cancellationToken)
    {
        // TODO: Implement Parquet deserialization using Apache.Arrow or similar
        // For now, return empty list to avoid breaking build
        await Task.CompletedTask.ConfigureAwait(false);
        
        _logger.LogDebug("Loading Parquet file: {File} (implementation pending)", filePath);
        return new List<HistoricalBar>();
    }

    private async Task SaveParquetFileAsync(string filePath, List<HistoricalBar> bars, CancellationToken cancellationToken)
    {
        // TODO: Implement Parquet serialization using Apache.Arrow or similar
        // For now, just log to avoid breaking build
        await Task.CompletedTask.ConfigureAwait(false);
        
        _logger.LogDebug("Saving {Count} bars to Parquet file: {File} (implementation pending)", bars.Count, filePath);
    }

    private DateTime ParseDateFromFilename(string filename)
    {
        // Expected format: yyyy-MM-dd
        if (DateTime.TryParseExact(filename, "yyyy-MM-dd", null, 
            System.Globalization.DateTimeStyles.None, out var date))
        {
            return date;
        }
        
        _logger.LogWarning("Could not parse date from filename: {Filename}", filename);
        return DateTime.MinValue;
    }

    private bool IsOvernightGap(DateTime time1, DateTime time2)
    {
        // Check if gap is between trading sessions (after 4:00 PM to before 9:30 AM ET)
        // Simplified: just check if dates are different
        return time1.Date != time2.Date;
    }

    #endregion
}

/// <summary>
/// Historical bar data structure for Lab training
/// Distinct from real-time bar to avoid confusion
/// </summary>
public record HistoricalBar(
    DateTime Time,
    decimal Open,
    decimal High,
    decimal Low,
    decimal Close,
    long Volume
);

/// <summary>
/// Data quality validation result
/// </summary>
public class ValidationResult
{
    public bool IsValid { get; set; }
    public List<string> Errors { get; } = new();
    public List<string> Warnings { get; } = new();
}
