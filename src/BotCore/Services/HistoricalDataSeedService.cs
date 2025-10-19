using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using TradingBot.BotCore.Models;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Service for loading, validating, and auto-refreshing historical market data seed files.
    /// Provides fast bot startup by pre-loading historical bars from disk instead of API.
    /// Supports smart auto-refresh during futures maintenance window (5 PM ET daily, skip weekends).
    /// </summary>
    public class HistoricalDataSeedService : IHistoricalDataSeedService
    {
        private readonly ILogger<HistoricalDataSeedService> _logger;
        private readonly string _dataDirectory;
        private readonly TimeZoneInfo _easternTimeZone;
        
        // Maintenance window: 5:00 PM - 6:00 PM ET daily (futures market closed)
        private const int MaintenanceHourEt = 17; // 5 PM ET
        
        public HistoricalDataSeedService(ILogger<HistoricalDataSeedService> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _dataDirectory = Path.Combine(Directory.GetCurrentDirectory(), "data", "historical");
            
            try
            {
                _easternTimeZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
            }
            catch
            {
                _logger.LogWarning("Could not load Eastern timezone, using UTC offset");
                _easternTimeZone = TimeZoneInfo.Utc;
            }
        }

        /// <summary>
        /// Try to load and apply historical seed data with validation and reconciliation.
        /// </summary>
        public async Task<SeedApplyResult> TryApplySeedAsync(string[] symbols, CancellationToken cancellationToken = default)
        {
            if (symbols == null || symbols.Length == 0)
            {
                _logger.LogWarning("No symbols provided for seed loading");
                return SeedApplyResult.Failed("No symbols specified");
            }

            try
            {
                // Check if we need to refresh seed data first
                await RefreshSeedIfStaleAsync(cancellationToken).ConfigureAwait(false);

                // Load seed data from disk
                var seedData = await LoadSeedFromDiskAsync(symbols, cancellationToken).ConfigureAwait(false);
                
                if (seedData == null || seedData.Bars.Count == 0)
                {
                    _logger.LogWarning("No seed data available on disk");
                    return SeedApplyResult.Failed("No seed data files found");
                }

                // Validate seed integrity
                var validationResult = ValidateSeed(seedData);
                
                if (!validationResult.Passed)
                {
                    _logger.LogWarning("Seed validation failed: {Errors}", 
                        string.Join(", ", validationResult.Errors));
                    return SeedApplyResult.Failed($"Validation failed: {validationResult.Errors.First()}");
                }

                _logger.LogInformation(
                    "✅ Seed loaded and validated: {BarCount} bars from {OldestBar:yyyy-MM-dd} to {NewestBar:yyyy-MM-dd}",
                    seedData.Bars.Count,
                    validationResult.OldestBar,
                    validationResult.NewestBar);

                return SeedApplyResult.CreateSuccess(seedData.Bars, validationResult);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load historical seed data");
                return SeedApplyResult.Failed($"Exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Load historical seed data from disk JSON files.
        /// </summary>
        private async Task<SeedData?> LoadSeedFromDiskAsync(string[] symbols, CancellationToken cancellationToken)
        {
            var allBars = new List<HistoricalBar>();

            foreach (var symbol in symbols)
            {
                var filePath = Path.Combine(_dataDirectory, $"{symbol}_90days.json");
                
                if (!File.Exists(filePath))
                {
                    _logger.LogWarning("Seed file not found: {FilePath}", filePath);
                    continue;
                }

                try
                {
                    var fileInfo = new FileInfo(filePath);
                    _logger.LogDebug("Loading seed file: {FilePath} (Size: {Size} KB, Modified: {Modified})",
                        filePath, fileInfo.Length / 1024, fileInfo.LastWriteTime);

                    var json = await File.ReadAllTextAsync(filePath, cancellationToken).ConfigureAwait(false);
                    var seedFile = JsonSerializer.Deserialize<SeedFileFormat>(json);

                    if (seedFile?.Bars == null || seedFile.Bars.Count == 0)
                    {
                        _logger.LogWarning("Seed file {FilePath} is empty or invalid", filePath);
                        continue;
                    }

                    // Convert to internal bar format
                    foreach (var barDto in seedFile.Bars)
                    {
                        if (!TryParseBar(barDto, symbol, out var bar))
                        {
                            continue;
                        }

                        allBars.Add(bar);
                    }

                    _logger.LogInformation("Loaded {BarCount} bars for {Symbol} from {FilePath}",
                        seedFile.Bars.Count, symbol, Path.GetFileName(filePath));
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to load seed file: {FilePath}", filePath);
                }
            }

            return allBars.Count > 0 ? new SeedData { Bars = allBars } : null;
        }

        /// <summary>
        /// Validate seed data integrity (contiguous timestamps, sane volumes, no duplicates).
        /// </summary>
        private SeedValidationResult ValidateSeed(SeedData seedData)
        {
            var result = new SeedValidationResult
            {
                BarCount = seedData.Bars.Count,
                Errors = new List<string>()
            };

            if (seedData.Bars.Count == 0)
            {
                result.Errors.Add("No bars in seed data");
                result.Passed = false;
                return result;
            }

            // Sort by timestamp
            var sortedBars = seedData.Bars.OrderBy(b => b.Timestamp).ToList();
            
            result.OldestBar = sortedBars.First().Timestamp;
            result.NewestBar = sortedBars.Last().Timestamp;

            // Check for duplicates
            var timestamps = sortedBars.Select(b => b.Timestamp).ToList();
            var distinctTimestamps = timestamps.Distinct().Count();
            
            if (distinctTimestamps < timestamps.Count)
            {
                result.HasDuplicates = true;
                result.Errors.Add($"Found {timestamps.Count - distinctTimestamps} duplicate timestamps");
            }

            // Check volume sanity (no negative, no extreme values)
            var invalidVolumes = sortedBars.Where(b => b.Volume < 0 || b.Volume > 1_000_000).ToList();
            
            if (invalidVolumes.Any())
            {
                result.VolumeValid = false;
                result.Errors.Add($"Found {invalidVolumes.Count} bars with invalid volume");
            }
            else
            {
                result.VolumeValid = true;
            }

            // Check for large gaps (more than 30 minutes between bars during market hours)
            var gapCount = 0;
            for (int i = 1; i < sortedBars.Count; i++)
            {
                var timeDiff = (sortedBars[i].Timestamp - sortedBars[i - 1].Timestamp).TotalMinutes;
                
                // Allow larger gaps for weekends/holidays, but flag excessive gaps
                if (timeDiff > 30 && timeDiff < 48 * 60) // Between 30 min and 2 days
                {
                    gapCount++;
                }
            }

            if (gapCount > sortedBars.Count * 0.05m) // More than 5% gaps
            {
                result.HasGaps = true;
                result.Errors.Add($"Found {gapCount} significant time gaps");
            }

            result.Passed = !result.HasDuplicates && result.VolumeValid && !result.HasGaps;
            
            return result;
        }

        /// <summary>
        /// Auto-refresh seed data if stale. Runs during futures maintenance window (5 PM ET daily, skip weekends).
        /// </summary>
        private async Task<bool> RefreshSeedIfStaleAsync(CancellationToken cancellationToken)
        {
            try
            {
                var now = DateTimeOffset.UtcNow;
                var nowEt = TimeZoneInfo.ConvertTime(now, _easternTimeZone);

                // Skip weekends (Saturday = 6, Sunday = 0)
                if (nowEt.DayOfWeek == DayOfWeek.Saturday || nowEt.DayOfWeek == DayOfWeek.Sunday)
                {
                    _logger.LogDebug("Skipping seed refresh on weekend");
                    return false;
                }

                // Check if any seed file exists
                var seedFiles = new[] { "ES_90days.json", "NQ_90days.json" }
                    .Select(f => Path.Combine(_dataDirectory, f))
                    .Where(File.Exists)
                    .ToList();

                if (seedFiles.Count == 0)
                {
                    _logger.LogInformation("No seed files exist, skipping auto-refresh (will fetch on first manual run)");
                    return false;
                }

                // Check age of newest file
                var newestFile = seedFiles
                    .Select(f => new FileInfo(f))
                    .OrderByDescending(fi => fi.LastWriteTime)
                    .First();

                var age = DateTime.Now - newestFile.LastWriteTime;

                // Only refresh if:
                // 1. Data is older than 24 hours
                // 2. Current time is during maintenance window (5 PM ET)
                var needsRefresh = age.TotalHours > 24;
                var isMaintenanceWindow = nowEt.Hour == MaintenanceHourEt;

                if (!needsRefresh)
                {
                    _logger.LogDebug("Seed data is fresh ({Age:F1} hours old), skipping refresh", age.TotalHours);
                    return false;
                }

                if (!isMaintenanceWindow)
                {
                    _logger.LogDebug(
                        "Seed data is stale ({Age:F1} hours old) but not maintenance window (current: {Hour} ET, target: {Target} ET)",
                        age.TotalHours, nowEt.Hour, MaintenanceHourEt);
                    return false;
                }

                _logger.LogInformation(
                    "🔄 Seed data is {Age:F1} hours old, refreshing during maintenance window ({Hour} ET)...",
                    age.TotalHours, nowEt.Hour);

                // Run Python refresh script
                await RunPythonRefreshScriptAsync(cancellationToken).ConfigureAwait(false);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to check or refresh seed data, using existing data");
                return false;
            }
        }

        /// <summary>
        /// Execute Python script to refresh historical data in incremental mode.
        /// </summary>
        private async Task RunPythonRefreshScriptAsync(CancellationToken cancellationToken)
        {
            try
            {
                var scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "fetch-and-save-historical-data.py");
                
                if (!File.Exists(scriptPath))
                {
                    _logger.LogError("Python refresh script not found: {ScriptPath}", scriptPath);
                    return;
                }

                var processInfo = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{scriptPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Directory.GetCurrentDirectory()
                };

                // Set incremental mode
                processInfo.Environment["REFRESH_MODE"] = "incremental";

                using var process = new Process { StartInfo = processInfo };
                
                _logger.LogInformation("Starting Python refresh script: {Script}", scriptPath);
                
                process.Start();

                // Capture output
                var output = await process.StandardOutput.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
                var error = await process.StandardError.ReadToEndAsync(cancellationToken).ConfigureAwait(false);

                await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);

                if (process.ExitCode == 0)
                {
                    _logger.LogInformation("✅ Seed refresh completed successfully");
                    
                    if (!string.IsNullOrWhiteSpace(output))
                    {
                        _logger.LogDebug("Refresh output: {Output}", output);
                    }
                }
                else
                {
                    _logger.LogWarning("⚠️ Seed refresh failed with exit code {ExitCode}", process.ExitCode);
                    
                    if (!string.IsNullOrWhiteSpace(error))
                    {
                        _logger.LogWarning("Refresh error: {Error}", error);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to execute Python refresh script");
            }
        }

        /// <summary>
        /// Try to parse a bar DTO into internal bar format.
        /// </summary>
        private bool TryParseBar(BarDto barDto, string symbol, out HistoricalBar bar)
        {
            bar = null!;

            try
            {
                if (string.IsNullOrWhiteSpace(barDto.Timestamp))
                {
                    return false;
                }

                // Parse timestamp (format: "2025-08-31 17:00:00-05:00")
                if (!DateTimeOffset.TryParse(barDto.Timestamp, out var timestamp))
                {
                    _logger.LogWarning("Failed to parse timestamp: {Timestamp}", barDto.Timestamp);
                    return false;
                }

                bar = new HistoricalBar
                {
                    Symbol = symbol,
                    Timestamp = timestamp.DateTime,
                    Open = barDto.Open,
                    High = barDto.High,
                    Low = barDto.Low,
                    Close = barDto.Close,
                    Volume = barDto.Volume
                };

                return true;
            }
            catch
            {
                return false;
            }
        }
    }

    /// <summary>
    /// Format of seed JSON file on disk.
    /// </summary>
    internal class SeedFileFormat
    {
        [JsonPropertyName("symbol")]
        public string Symbol { get; set; } = string.Empty;
        
        [JsonPropertyName("bar_count")]
        public int BarCount { get; set; }
        
        [JsonPropertyName("bars")]
        public List<BarDto> Bars { get; set; } = new();
    }

    /// <summary>
    /// Bar DTO from JSON file.
    /// </summary>
    internal class BarDto
    {
        [JsonPropertyName("timestamp")]
        public string Timestamp { get; set; } = string.Empty;
        
        [JsonPropertyName("open")]
        public decimal Open { get; set; }
        
        [JsonPropertyName("high")]
        public decimal High { get; set; }
        
        [JsonPropertyName("low")]
        public decimal Low { get; set; }
        
        [JsonPropertyName("close")]
        public decimal Close { get; set; }
        
        [JsonPropertyName("volume")]
        public long Volume { get; set; }
    }
}
