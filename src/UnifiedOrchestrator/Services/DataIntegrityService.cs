using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Data integrity verification service for training data
/// Ensures data is complete, uncorrupted, and meets quality standards
/// </summary>
internal sealed class DataIntegrityService
{
    private readonly ILogger<DataIntegrityService> _logger;
    
    // Expected bars per trading day (ES/NQ futures trade nearly 24/5)
    private const int ExpectedBarsPerDay = 390; // Approximate for 5-min bars during main session
    private const double CompletenessThreshold = 95.0; // Require 95% data completeness

    public DataIntegrityService(ILogger<DataIntegrityService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Verify data completeness and integrity before training
    /// </summary>
    public async Task<DataVerificationResult> VerifyTrainingDataAsync(
        Dictionary<string, int> historicalData,
        int experienceCount,
        int expectedDays,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[DATA-INTEGRITY] Starting data verification...");

        var result = new DataVerificationResult
        {
            VerificationTime = DateTime.UtcNow,
            ExpectedTradingDays = expectedDays
        };

        // Check 1: Verify bar counts
        await VerifyBarCountsAsync(historicalData, expectedDays, result, cancellationToken).ConfigureAwait(false);

        // Check 2: Verify experiences availability
        VerifyExperiences(experienceCount, result);

        // Check 3: Compute data hash for change detection
        result.DataHash = ComputeDataHash(historicalData, experienceCount);

        // Check 4: Check for date gaps (if we had timestamp data)
        // This would require actual bar timestamps, which we don't have in the current format
        // For now, we'll estimate based on bar counts

        // Calculate overall completeness
        result.CalculateCompleteness();

        if (result.IsValid)
        {
            _logger.LogInformation("[DATA-INTEGRITY] ✅ Data verification PASSED - Completeness: {Percent:F1}%",
                result.CompletenessPercent);
        }
        else
        {
            _logger.LogError("[DATA-INTEGRITY] ❌ Data verification FAILED - Issues: {Count}",
                result.Issues.Count);
            foreach (var issue in result.Issues)
            {
                _logger.LogError("[DATA-INTEGRITY]   - {Issue}", issue);
            }
        }

        return result;
    }

    /// <summary>
    /// Verify bar counts for each symbol
    /// </summary>
    private async Task VerifyBarCountsAsync(
        Dictionary<string, int> historicalData,
        int expectedDays,
        DataVerificationResult result,
        CancellationToken cancellationToken)
    {
        var expectedTotalBars = expectedDays * ExpectedBarsPerDay;

        foreach (var kvp in historicalData)
        {
            var symbol = kvp.Key;
            var actualBars = kvp.Value;

            result.SymbolBarCounts[symbol] = actualBars;

            // Check if bar count is reasonable
            var completeness = (actualBars / (double)expectedTotalBars) * 100.0;

            if (actualBars == 0)
            {
                result.Issues.Add($"No bars loaded for {symbol}");
                result.IsValid = false;
            }
            else if (completeness < CompletenessThreshold)
            {
                result.Issues.Add($"{symbol}: Only {completeness:F1}% complete ({actualBars}/{expectedTotalBars} bars)");
                result.IsValid = false;
            }
            else
            {
                _logger.LogInformation("[DATA-INTEGRITY] ✓ {Symbol}: {Bars:N0} bars ({Completeness:F1}% complete)",
                    symbol, actualBars, completeness);
            }
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Verify experiences are available
    /// </summary>
    private void VerifyExperiences(int experienceCount, DataVerificationResult result)
    {
        result.ExperienceCount = experienceCount;

        if (experienceCount == 0)
        {
            result.Warnings.Add("No experiences available - this may be first training run");
            _logger.LogWarning("[DATA-INTEGRITY] ⚠️ No experiences available (first run?)");
        }
        else if (experienceCount < 100)
        {
            result.Warnings.Add($"Limited experiences: {experienceCount} (recommended: > 100)");
            _logger.LogWarning("[DATA-INTEGRITY] ⚠️ Limited experiences: {Count} (< 100)",
                experienceCount);
        }
        else
        {
            _logger.LogInformation("[DATA-INTEGRITY] ✓ Experiences: {Count:N0}", experienceCount);
        }
    }

    /// <summary>
    /// Compute hash of training data for change detection
    /// </summary>
    private string ComputeDataHash(Dictionary<string, int> historicalData, int experienceCount)
    {
        var dataString = string.Join("|",
            historicalData.OrderBy(kvp => kvp.Key).Select(kvp => $"{kvp.Key}:{kvp.Value}"))
            + $"|exp:{experienceCount}";

        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(dataString));
        var hash = Convert.ToHexString(hashBytes).ToLowerInvariant();

        _logger.LogInformation("[DATA-INTEGRITY] Data hash: {Hash}", hash[..16] + "...");
        return hash;
    }

    /// <summary>
    /// Phase 3: Validate historical data files exist and are complete
    /// Enhanced with comprehensive validation checks
    /// </summary>
    public async Task<HistoricalDataValidationResult> ValidateHistoricalDataFilesAsync(
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[DATA-INTEGRITY] Validating historical data files...");

        var result = new HistoricalDataValidationResult
        {
            ValidationTime = DateTime.UtcNow
        };

        var symbols = new[] { "ES", "NQ" };
        var dataDir = System.IO.Path.Combine(Directory.GetCurrentDirectory(), "data", "historical");

        // Check if directory exists
        if (!Directory.Exists(dataDir))
        {
            result.Issues.Add($"Historical data directory does not exist: {dataDir}");
            result.IsValid = false;
            _logger.LogError("[DATA-INTEGRITY] ❌ Historical data directory not found");
            return result;
        }

        foreach (var symbol in symbols)
        {
            var filePath = System.IO.Path.Combine(dataDir, $"{symbol}_90days.json");
            
            // File existence check
            if (!File.Exists(filePath))
            {
                result.Issues.Add($"Missing historical data file: {filePath}");
                result.IsValid = false;
                continue;
            }

            // File size check (should be >10MB, <500MB for reasonable data)
            var fileInfo = new FileInfo(filePath);
            var fileSizeMB = fileInfo.Length / (1024.0 * 1024.0);
            
            if (fileSizeMB < 10)
            {
                result.Warnings.Add($"{symbol}: File size too small ({fileSizeMB:F1} MB < 10 MB)");
            }
            else if (fileSizeMB > 500)
            {
                result.Warnings.Add($"{symbol}: File size unexpectedly large ({fileSizeMB:F1} MB > 500 MB)");
            }

            try
            {
                var fileContent = await File.ReadAllTextAsync(filePath, cancellationToken).ConfigureAwait(false);
                
                // Check if file is readable and parseable JSON
                System.Text.Json.JsonDocument data;
                try
                {
                    data = System.Text.Json.JsonDocument.Parse(fileContent);
                }
                catch (System.Text.Json.JsonException jsonEx)
                {
                    result.Issues.Add($"{symbol}: Invalid JSON format - {jsonEx.Message}");
                    result.IsValid = false;
                    continue;
                }
                
                var root = data.RootElement;
                
                // Check metadata
                if (!root.TryGetProperty("bar_count", out var barCountElement))
                {
                    result.Issues.Add($"{symbol}: Missing bar_count metadata");
                    result.IsValid = false;
                    continue;
                }

                var barCount = barCountElement.GetInt32();
                result.SymbolBarCounts[symbol] = barCount;

                // Check if bars array exists
                if (!root.TryGetProperty("bars", out var barsElement))
                {
                    result.Issues.Add($"{symbol}: Missing bars array");
                    result.IsValid = false;
                    continue;
                }

                // Validate bar count matches expected (90 days of 5-min bars)
                // ES: ~7020 bars per day (23 hours trading) * 90 days = ~631,800 bars
                // Allow variance for holidays and early closes
                const int ExpectedBarsPerDay = 7020;
                const int ExpectedDays = 90;
                var expectedBars = ExpectedBarsPerDay * ExpectedDays;
                var minBars = (int)(expectedBars * 0.7); // Allow 30% variance
                var maxBars = (int)(expectedBars * 1.3);
                
                if (barCount < minBars)
                {
                    result.Warnings.Add($"{symbol}: Low bar count {barCount:N0} (expected ~{expectedBars:N0}, min {minBars:N0})");
                }
                else if (barCount > maxBars)
                {
                    result.Warnings.Add($"{symbol}: High bar count {barCount:N0} (expected ~{expectedBars:N0}, max {maxBars:N0})");
                }

                // Date range validation
                if (root.TryGetProperty("start_date", out var startDateElement) &&
                    root.TryGetProperty("end_date", out var endDateElement))
                {
                    var startDateStr = startDateElement.GetString();
                    var endDateStr = endDateElement.GetString();
                    
                    if (DateTime.TryParse(startDateStr, out var startDate) &&
                        DateTime.TryParse(endDateStr, out var endDate))
                    {
                        var daysDiff = (endDate - startDate).TotalDays;
                        
                        // Check if approximately 90 days
                        if (daysDiff < 80)
                        {
                            result.Warnings.Add($"{symbol}: Date range too short ({daysDiff:F0} days < 80 days)");
                        }
                        else if (daysDiff > 100)
                        {
                            result.Warnings.Add($"{symbol}: Date range too long ({daysDiff:F0} days > 100 days)");
                        }
                        
                        // Check if end date is recent (within last week)
                        var age = DateTime.UtcNow - endDate.ToUniversalTime();
                        if (age.TotalDays > 7)
                        {
                            result.Warnings.Add($"{symbol}: Data end date is {age.TotalDays:F1} days old (>7 days)");
                        }
                        
                        result.DateRanges[symbol] = (startDate, endDate);
                    }
                }

                // Check freshness (data should be < 7 days old)
                if (root.TryGetProperty("fetched_at", out var fetchedAtElement))
                {
                    var fetchedAtStr = fetchedAtElement.GetString();
                    if (DateTime.TryParse(fetchedAtStr, out var fetchedAt))
                    {
                        var age = DateTime.UtcNow - fetchedAt.ToUniversalTime();
                        if (age.TotalDays > 7)
                        {
                            result.Warnings.Add($"{symbol}: Data fetched {age.TotalDays:F1} days ago (>7 days)");
                        }
                    }
                }

                // Validate bar data quality (sample first 100 bars)
                var barsArray = barsElement.EnumerateArray().Take(100).ToList();
                var invalidBars = 0;
                var nullTimestamps = 0;
                
                DateTime? prevTimestamp = null;
                var outOfOrderCount = 0;
                
                foreach (var bar in barsArray)
                {
                    // Check for null or missing timestamp
                    if (!bar.TryGetProperty("timestamp", out var tsElement) || tsElement.ValueKind == System.Text.Json.JsonValueKind.Null)
                    {
                        nullTimestamps++;
                        continue;
                    }
                    
                    var timestamp = tsElement.GetString();
                    if (string.IsNullOrEmpty(timestamp))
                    {
                        nullTimestamps++;
                        continue;
                    }
                    
                    // Check timestamp ordering
                    if (DateTime.TryParse(timestamp, out var currentTimestamp))
                    {
                        if (prevTimestamp.HasValue && currentTimestamp < prevTimestamp.Value)
                        {
                            outOfOrderCount++;
                        }
                        prevTimestamp = currentTimestamp;
                    }
                    
                    // Check OHLC values
                    if (bar.TryGetProperty("open", out var openEl) &&
                        bar.TryGetProperty("high", out var highEl) &&
                        bar.TryGetProperty("low", out var lowEl) &&
                        bar.TryGetProperty("close", out var closeEl))
                    {
                        var open = openEl.GetDecimal();
                        var high = highEl.GetDecimal();
                        var low = lowEl.GetDecimal();
                        var close = closeEl.GetDecimal();
                        
                        // Validate OHLC logic
                        if (open == 0 || high == 0 || low == 0 || close == 0)
                        {
                            invalidBars++;
                        }
                        else if (high < low || open > high || open < low || close > high || close < low)
                        {
                            invalidBars++;
                        }
                    }
                }
                
                if (nullTimestamps > 0)
                {
                    result.Warnings.Add($"{symbol}: Found {nullTimestamps} bars with null/missing timestamps (sampled 100 bars)");
                }
                
                if (outOfOrderCount > 0)
                {
                    result.Warnings.Add($"{symbol}: Found {outOfOrderCount} bars with out-of-order timestamps (sampled 100 bars)");
                }
                
                if (invalidBars > 0)
                {
                    result.Warnings.Add($"{symbol}: Found {invalidBars} bars with invalid OHLC values (sampled 100 bars)");
                }

                _logger.LogInformation("[DATA-INTEGRITY] ✓ {Symbol}: {Bars:N0} bars validated",
                    symbol, barCount);
            }
            catch (Exception ex)
            {
                result.Issues.Add($"{symbol}: Error reading file - {ex.Message}");
                result.IsValid = false;
            }
        }

        // Generate summary
        if (result.IsValid)
        {
            var summaryParts = new List<string>();
            foreach (var symbol in symbols)
            {
                if (result.SymbolBarCounts.TryGetValue(symbol, out var barCount))
                {
                    string dateRange = "date range unknown";
                    if (result.DateRanges.TryGetValue(symbol, out var range))
                    {
                        var days = (range.End - range.Start).TotalDays;
                        dateRange = $"{days:F0} days";
                    }
                    summaryParts.Add($"{symbol}: {barCount:N0} bars, {dateRange}");
                }
            }
            
            _logger.LogInformation("[DATA-INTEGRITY] ✅ Historical data validation PASSED - {Summary}",
                string.Join("; ", summaryParts));
        }
        else
        {
            _logger.LogError("[DATA-INTEGRITY] ❌ Historical data validation FAILED - {Count} issues",
                result.Issues.Count);
        }

        if (result.Warnings.Any())
        {
            _logger.LogWarning("[DATA-INTEGRITY] ⚠️ {Count} warnings found", result.Warnings.Count);
        }

        return result;
    }

    /// <summary>
    /// Phase 3: Detect gaps in bar timestamps (requires actual bar data)
    /// </summary>
    public List<(DateTime Start, DateTime End)> DetectTimeGaps(
        List<DateTime> timestamps,
        TimeSpan expectedInterval,
        TimeSpan maxGapTolerance)
    {
        var gaps = new List<(DateTime Start, DateTime End)>();

        if (timestamps.Count < 2)
            return gaps;

        for (int i = 1; i < timestamps.Count; i++)
        {
            var gap = timestamps[i] - timestamps[i - 1];
            
            // If gap is significantly larger than expected interval
            if (gap > expectedInterval + maxGapTolerance)
            {
                gaps.Add((timestamps[i - 1], timestamps[i]));
                _logger.LogWarning("[DATA-INTEGRITY] ⚠️ Gap detected: {Gap} between {Start} and {End}",
                    gap, timestamps[i - 1], timestamps[i]);
            }
        }

        return gaps;
    }

    /// <summary>
    /// Detect missing trading days (would require actual timestamp data)
    /// Currently estimates based on bar counts; can be enhanced with timestamp data
    /// </summary>
    public List<DateTime> DetectMissingTradingDays(DateTime startDate, DateTime endDate, int actualDays)
    {
        var missingDays = new List<DateTime>();

        // Calculate expected trading days (excluding weekends)
        var expectedDays = 0;
        var current = startDate;
        while (current <= endDate)
        {
            if (current.DayOfWeek != DayOfWeek.Saturday && current.DayOfWeek != DayOfWeek.Sunday)
            {
                expectedDays++;
            }
            current = current.AddDays(1);
        }

        // If actual days differ significantly from expected, something is wrong
        var discrepancy = Math.Abs(expectedDays - actualDays);
        if (discrepancy > 5) // Allow 5 days tolerance for holidays
        {
            _logger.LogWarning("[DATA-INTEGRITY] ⚠️ Day count discrepancy: expected ~{Expected}, got {Actual}",
                expectedDays, actualDays);
        }

        return missingDays;
    }

    /// <summary>
    /// Compare current data hash with previous run to detect changes
    /// </summary>
    public bool HasDataChanged(string currentHash, string? previousHash)
    {
        if (string.IsNullOrEmpty(previousHash))
        {
            _logger.LogInformation("[DATA-INTEGRITY] No previous hash - first run or cache miss");
            return true;
        }

        var changed = currentHash != previousHash;
        
        if (changed)
        {
            _logger.LogInformation("[DATA-INTEGRITY] Data has CHANGED since last run");
        }
        else
        {
            _logger.LogInformation("[DATA-INTEGRITY] Data unchanged since last run");
        }

        return changed;
    }
}

/// <summary>
/// Result of data verification
/// </summary>
public sealed class DataVerificationResult
{
    public DateTime VerificationTime { get; set; }
    public bool IsValid { get; set; } = true;
    public int ExpectedTradingDays { get; set; }
    public Dictionary<string, int> SymbolBarCounts { get; set; } = new();
    public int ExperienceCount { get; set; }
    public string DataHash { get; set; } = string.Empty;
    public List<string> Issues { get; set; } = new();
    public List<string> Warnings { get; set; } = new();
    public double CompletenessPercent { get; set; }

    public void CalculateCompleteness()
    {
        if (!SymbolBarCounts.Any())
        {
            CompletenessPercent = 0;
            return;
        }

        var expectedBarsPerSymbol = ExpectedTradingDays * 390;
        var totalExpected = SymbolBarCounts.Count * expectedBarsPerSymbol;
        var totalActual = SymbolBarCounts.Values.Sum();

        CompletenessPercent = totalExpected > 0
            ? (totalActual / (double)totalExpected) * 100.0
            : 0;
    }
}

/// <summary>
/// Phase 3: Result of historical data file validation
/// </summary>
public sealed class HistoricalDataValidationResult
{
    public DateTime ValidationTime { get; set; }
    public bool IsValid { get; set; } = true;
    public Dictionary<string, int> SymbolBarCounts { get; set; } = new();
    public Dictionary<string, (DateTime Start, DateTime End)> DateRanges { get; set; } = new();
    public List<string> Issues { get; set; } = new();
    public List<string> Warnings { get; set; } = new();
}
