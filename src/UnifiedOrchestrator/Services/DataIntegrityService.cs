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

        foreach (var symbol in symbols)
        {
            var filePath = System.IO.Path.Combine(dataDir, $"{symbol}_90days.json");
            
            if (!File.Exists(filePath))
            {
                result.Issues.Add($"Missing historical data file: {filePath}");
                result.IsValid = false;
                continue;
            }

            try
            {
                var fileContent = await File.ReadAllTextAsync(filePath, cancellationToken).ConfigureAwait(false);
                var data = System.Text.Json.JsonDocument.Parse(fileContent);
                
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

                // Validate minimum bar count (90 days * 390 bars/day ≈ 35,100 bars minimum)
                const int MinBarsFor90Days = 30000; // Allow some tolerance
                if (barCount < MinBarsFor90Days)
                {
                    result.Warnings.Add($"{symbol}: Low bar count {barCount} (expected > {MinBarsFor90Days})");
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
                            result.Warnings.Add($"{symbol}: Data is {age.TotalDays:F1} days old (>7 days)");
                        }
                    }
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

        if (result.IsValid)
        {
            _logger.LogInformation("[DATA-INTEGRITY] ✅ Historical data files validation PASSED");
        }
        else
        {
            _logger.LogError("[DATA-INTEGRITY] ❌ Historical data files validation FAILED");
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
    public List<string> Issues { get; set; } = new();
    public List<string> Warnings { get; set; } = new();
}
