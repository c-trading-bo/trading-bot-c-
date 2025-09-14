using System.Text.Json;
using BotCore.Models;
using Microsoft.Extensions.Logging;

namespace BotCore.Services;

/// <summary>
/// Service for analyzing execution quality and providing feedback to Intelligence pipeline.
/// Tracks slippage, fill quality, and zone test results.
/// </summary>
public class ExecutionAnalyzer
{
    private readonly ILogger<ExecutionAnalyzer> _logger;
    private readonly string _feedbackPath;
    private readonly JsonSerializerOptions _jsonOptions;

    public ExecutionAnalyzer(ILogger<ExecutionAnalyzer> logger, string? feedbackPath = null)
    {
        _logger = logger;
        _feedbackPath = feedbackPath ?? "data/zones/feedback";
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            WriteIndented = true
        };

        // Ensure feedback directory exists
        Directory.CreateDirectory(_feedbackPath);
    }

    /// <summary>
    /// Track fill quality metrics for execution analysis
    /// </summary>
    public async Task TrackFillQualityAsync(string symbol, decimal entryPrice, decimal fillPrice,
        int quantity, string strategy, DateTime timestamp)
    {
        try
        {
            var slippage = Math.Abs(fillPrice - entryPrice);
            var slippagePercent = entryPrice != 0 ? (slippage / entryPrice) * 100 : 0;

            var fillQuality = new
            {
                Symbol = symbol,
                Strategy = strategy,
                Timestamp = timestamp.ToString("O"),
                EntryPrice = entryPrice,
                FillPrice = fillPrice,
                Quantity = quantity,
                Slippage = Math.Round(slippage, 4),
                SlippagePercent = Math.Round(slippagePercent, 4),
                Quality = slippagePercent < 0.02m ? "excellent" :
                         slippagePercent < 0.05m ? "good" :
                         slippagePercent < 0.1m ? "fair" : "poor"
            };

            _logger.LogInformation("[EXEC_QUALITY] {Symbol} {Strategy} slippage={SlippagePercent:P2} quality={Quality}",
                symbol, strategy, slippagePercent / 100, fillQuality.Quality);

            // Save to execution quality log
            await SaveExecutionMetricsAsync(fillQuality);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error tracking fill quality for {Symbol}", symbol);
        }
    }

    /// <summary>
    /// Provide feedback on supply/demand zone tests
    /// </summary>
    public async Task ZoneFeedbackAsync(string symbol, decimal zoneLevel, string zoneType,
        bool successful, decimal entryPrice, decimal exitPrice, string reason)
    {
        try
        {
            var pnlPercent = entryPrice != 0 ? ((exitPrice - entryPrice) / entryPrice) * 100 : 0;

            var feedback = new
            {
                Symbol = symbol,
                ZoneLevel = Math.Round(zoneLevel, 2),
                ZoneType = zoneType, // "supply" or "demand"
                TestTime = DateTime.UtcNow.ToString("O"),
                Successful = successful,
                EntryPrice = Math.Round(entryPrice, 2),
                ExitPrice = Math.Round(exitPrice, 2),
                PnLPercent = Math.Round(pnlPercent, 2),
                Reason = reason
            };

            _logger.LogInformation("[ZONE_FEEDBACK] {Symbol} {ZoneType}@{ZoneLevel} success={Successful} pnl={PnLPercent:P2}",
                symbol, zoneType, zoneLevel, successful, pnlPercent / 100);

            // Update zone feedback data
            await UpdateZoneFeedbackAsync(symbol, zoneLevel, zoneType, successful);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error providing zone feedback for {Symbol}", symbol);
        }
    }

    /// <summary>
    /// Track pattern outcome (success/failure) for pattern recognition improvement
    /// </summary>
    public async Task PatternOutcomeAsync(string symbol, string patternType, bool successful,
        decimal confidence, string details)
    {
        try
        {
            var outcome = new
            {
                Symbol = symbol,
                PatternType = patternType,
                Timestamp = DateTime.UtcNow.ToString("O"),
                Successful = successful,
                Confidence = Math.Round(confidence, 3),
                Details = details,
                OutcomeQuality = successful && confidence > 0.7m ? "high_confidence_success" :
                                successful && confidence > 0.4m ? "medium_confidence_success" :
                                successful ? "low_confidence_success" :
                                confidence > 0.7m ? "high_confidence_failure" : "failed"
            };

            _logger.LogInformation("[PATTERN_OUTCOME] {Symbol} {PatternType} success={Successful} conf={Confidence:P1}",
                symbol, patternType, successful, confidence);

            await SavePatternOutcomeAsync(outcome);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error tracking pattern outcome for {Symbol}", symbol);
        }
    }

    /// <summary>
    /// Get execution quality metrics for the day
    /// </summary>
    public async Task<ExecutionMetrics?> GetDailyExecutionMetricsAsync()
    {
        try
        {
            var today = DateTime.Today.ToString("yyyy-MM-dd");
            var metricsFile = Path.Combine(_feedbackPath, $"execution_metrics_{today}.json");

            if (!File.Exists(metricsFile))
                return null;

            var json = await File.ReadAllTextAsync(metricsFile);
            return JsonSerializer.Deserialize<ExecutionMetrics>(json, _jsonOptions);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting daily execution metrics");
            return null;
        }
    }

    private async Task SaveExecutionMetricsAsync(object fillQuality)
    {
        try
        {
            var today = DateTime.Today.ToString("yyyy-MM-dd");
            var metricsFile = Path.Combine(_feedbackPath, $"execution_metrics_{today}.json");

            // Load existing metrics or create new
            var metrics = new List<object>();
            if (File.Exists(metricsFile))
            {
                var json = await File.ReadAllTextAsync(metricsFile);
                var existing = JsonSerializer.Deserialize<List<object>>(json);
                if (existing != null) metrics = existing;
            }

            metrics.Add(fillQuality);

            await File.WriteAllTextAsync(metricsFile,
                JsonSerializer.Serialize(metrics, _jsonOptions));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving execution metrics");
        }
    }

    private async Task UpdateZoneFeedbackAsync(string symbol, decimal zoneLevel, string zoneType, bool successful)
    {
        try
        {
            var feedbackFile = Path.Combine(_feedbackPath, $"{symbol}_feedback.json");
            var zoneKey = $"{zoneType}_{zoneLevel}";

            // Load existing feedback
            var feedback = new Dictionary<string, object>();
            if (File.Exists(feedbackFile))
            {
                var json = await File.ReadAllTextAsync(feedbackFile);
                var existing = JsonSerializer.Deserialize<Dictionary<string, object>>(json);
                if (existing != null) feedback = existing;
            }

            // Update zone feedback
            if (!feedback.ContainsKey(zoneKey))
            {
                feedback[zoneKey] = new { success_count = 0, test_count = 0, success_rate = 0.0 };
            }

            var zoneData = JsonSerializer.Deserialize<Dictionary<string, object>>(
                feedback[zoneKey].ToString() ?? "{}");

            var successCount = zoneData?.GetValueOrDefault("success_count", 0);
            var testCount = zoneData?.GetValueOrDefault("test_count", 0);

            var newSuccessCount = Convert.ToInt32(successCount) + (successful ? 1 : 0);
            var newTestCount = Convert.ToInt32(testCount) + 1;
            var newSuccessRate = newTestCount > 0 ? (double)newSuccessCount / newTestCount : 0;

            feedback[zoneKey] = new
            {
                success_count = newSuccessCount,
                test_count = newTestCount,
                success_rate = Math.Round(newSuccessRate, 3),
                last_test = DateTime.UtcNow.ToString("O")
            };

            await File.WriteAllTextAsync(feedbackFile,
                JsonSerializer.Serialize(feedback, _jsonOptions));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating zone feedback");
        }
    }

    private async Task SavePatternOutcomeAsync(object outcome)
    {
        try
        {
            var today = DateTime.Today.ToString("yyyy-MM-dd");
            var outcomeFile = Path.Combine(_feedbackPath, $"pattern_outcomes_{today}.json");

            var outcomes = new List<object>();
            if (File.Exists(outcomeFile))
            {
                var json = await File.ReadAllTextAsync(outcomeFile);
                var existing = JsonSerializer.Deserialize<List<object>>(json);
                if (existing != null) outcomes = existing;
            }

            outcomes.Add(outcome);

            await File.WriteAllTextAsync(outcomeFile,
                JsonSerializer.Serialize(outcomes, _jsonOptions));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving pattern outcome");
        }
    }
}

/// <summary>
/// Execution metrics model
/// </summary>
public class ExecutionMetrics
{
    public double AverageSlippagePercent { get; set; }
    public int TotalFills { get; set; }
    public int ExcellentFills { get; set; }
    public int GoodFills { get; set; }
    public int FairFills { get; set; }
    public int PoorFills { get; set; }
    public double QualityScore { get; set; }
}