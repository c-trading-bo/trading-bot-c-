using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Promotion;

/// <summary>
/// Phase 7.6: Promotion History Tracker
/// Maintains append-only audit log of all promotion attempts
/// Used for compliance, debugging, and reliability metrics
/// </summary>
internal sealed class PromotionHistoryTracker
{
    private readonly ILogger<PromotionHistoryTracker> _logger;
    private readonly string _historyFile;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    
    public PromotionHistoryTracker(ILogger<PromotionHistoryTracker> logger)
    {
        _logger = logger;
        var baseDir = Directory.GetCurrentDirectory();
        var logsDir = Path.Combine(baseDir, "logs");
        Directory.CreateDirectory(logsDir);
        _historyFile = Path.Combine(logsDir, "promotion_history.jsonl");
    }
    
    /// <summary>
    /// Log promotion decision (PROMOTE/DEFER/REJECT)
    /// </summary>
    public async Task LogPromotionAttemptAsync(
        string sessionId,
        string decision,
        string reason,
        Dictionary<string, object>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        var entry = new PromotionHistoryEntry
        {
            SessionId = sessionId,
            Timestamp = DateTime.UtcNow,
            EventType = "DECISION",
            Decision = decision,
            Reason = reason,
            Metadata = metadata ?? new Dictionary<string, object>()
        };
        
        await AppendEntryAsync(entry, cancellationToken).ConfigureAwait(false);
        
        _logger.LogInformation("[PROMOTION-HISTORY] Logged decision: {Decision} for session {SessionId}",
            decision, sessionId);
    }
    
    /// <summary>
    /// Log promotion outcome (SUCCESS/FAILED)
    /// </summary>
    public async Task LogPromotionOutcomeAsync(
        string sessionId,
        string outcome,
        int modelsPromoted,
        double durationSeconds,
        string? version = null,
        Dictionary<string, object>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        var entry = new PromotionHistoryEntry
        {
            SessionId = sessionId,
            Timestamp = DateTime.UtcNow,
            EventType = "OUTCOME",
            Outcome = outcome,
            ModelsPromoted = modelsPromoted,
            DurationSeconds = durationSeconds,
            Version = version,
            Metadata = metadata ?? new Dictionary<string, object>()
        };
        
        await AppendEntryAsync(entry, cancellationToken).ConfigureAwait(false);
        
        _logger.LogInformation("[PROMOTION-HISTORY] Logged outcome: {Outcome} for session {SessionId}",
            outcome, sessionId);
    }
    
    /// <summary>
    /// Log rollback event
    /// </summary>
    public async Task LogRollbackAsync(
        string sessionId,
        string reason,
        string? restoredVersion = null,
        Dictionary<string, object>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        var entry = new PromotionHistoryEntry
        {
            SessionId = sessionId,
            Timestamp = DateTime.UtcNow,
            EventType = "ROLLBACK",
            Reason = reason,
            Version = restoredVersion,
            Metadata = metadata ?? new Dictionary<string, object>()
        };
        
        await AppendEntryAsync(entry, cancellationToken).ConfigureAwait(false);
        
        _logger.LogWarning("[PROMOTION-HISTORY] Logged rollback for session {SessionId}: {Reason}",
            sessionId, reason);
    }
    
    /// <summary>
    /// Query promotion history within date range
    /// </summary>
    public async Task<List<PromotionHistoryEntry>> QueryHistoryAsync(
        DateTime? startDate = null,
        DateTime? endDate = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(_historyFile))
            {
                return new List<PromotionHistoryEntry>();
            }
            
            var entries = new List<PromotionHistoryEntry>();
            var lines = await File.ReadAllLinesAsync(_historyFile, cancellationToken).ConfigureAwait(false);
            
            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                
                try
                {
                    var entry = JsonSerializer.Deserialize<PromotionHistoryEntry>(line);
                    if (entry == null)
                        continue;
                    
                    // Filter by date range if specified
                    if (startDate.HasValue && entry.Timestamp < startDate.Value)
                        continue;
                    if (endDate.HasValue && entry.Timestamp > endDate.Value)
                        continue;
                    
                    entries.Add(entry);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[PROMOTION-HISTORY] Failed to parse history entry");
                }
            }
            
            return entries.OrderBy(e => e.Timestamp).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PROMOTION-HISTORY] Failed to query history");
            return new List<PromotionHistoryEntry>();
        }
    }
    
    /// <summary>
    /// Generate statistics from promotion history
    /// </summary>
    public async Task<PromotionStatistics> GenerateStatisticsAsync(
        DateTime? startDate = null,
        CancellationToken cancellationToken = default)
    {
        var stats = new PromotionStatistics();
        
        try
        {
            var history = await QueryHistoryAsync(startDate, null, cancellationToken).ConfigureAwait(false);
            
            var outcomes = history.Where(e => e.EventType == "OUTCOME").ToList();
            var rollbacks = history.Where(e => e.EventType == "ROLLBACK").ToList();
            var decisions = history.Where(e => e.EventType == "DECISION").ToList();
            
            stats.TotalPromotionAttempts = decisions.Count;
            stats.SuccessfulPromotions = outcomes.Count(e => e.Outcome == "SUCCESS");
            stats.FailedPromotions = outcomes.Count(e => e.Outcome == "FAILED");
            stats.RollbackCount = rollbacks.Count;
            
            if (stats.TotalPromotionAttempts > 0)
            {
                stats.SuccessRate = (double)stats.SuccessfulPromotions / stats.TotalPromotionAttempts;
            }
            
            // Calculate average duration
            var durationsWithValues = outcomes.Where(e => e.DurationSeconds > 0);
            if (durationsWithValues.Any())
            {
                stats.AverageDurationSeconds = durationsWithValues.Average(e => e.DurationSeconds);
            }
            
            // Count promotion decisions
            stats.PromoteDecisions = decisions.Count(d => d.Decision == "PROMOTE");
            stats.DeferDecisions = decisions.Count(d => d.Decision == "DEFER");
            stats.RejectDecisions = decisions.Count(d => d.Decision == "REJECT");
            
            // Find most common failure reasons
            var failureReasons = outcomes.Where(e => e.Outcome == "FAILED")
                .Select(e => e.Reason ?? "Unknown")
                .GroupBy(r => r)
                .OrderByDescending(g => g.Count())
                .Take(5)
                .ToDictionary(g => g.Key, g => g.Count());
            
            stats.TopFailureReasons = failureReasons;
            
            _logger.LogInformation("[PROMOTION-HISTORY] Generated statistics: {Success}/{Total} successful",
                stats.SuccessfulPromotions, stats.TotalPromotionAttempts);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PROMOTION-HISTORY] Failed to generate statistics");
        }
        
        return stats;
    }
    
    /// <summary>
    /// Append entry to history file (JSON Lines format)
    /// </summary>
    private async Task AppendEntryAsync(PromotionHistoryEntry entry, CancellationToken cancellationToken)
    {
        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var json = JsonSerializer.Serialize(entry);
            await File.AppendAllTextAsync(_historyFile, json + Environment.NewLine, cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }
}

/// <summary>
/// Promotion history entry (JSON Lines format)
/// </summary>
public sealed class PromotionHistoryEntry
{
    [JsonPropertyName("sessionId")]
    public string SessionId { get; set; } = string.Empty;
    
    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; }
    
    [JsonPropertyName("eventType")]
    public string EventType { get; set; } = string.Empty; // DECISION, OUTCOME, ROLLBACK
    
    [JsonPropertyName("decision")]
    public string? Decision { get; set; } // PROMOTE, DEFER, REJECT
    
    [JsonPropertyName("outcome")]
    public string? Outcome { get; set; } // SUCCESS, FAILED, ROLLED_BACK
    
    [JsonPropertyName("reason")]
    public string? Reason { get; set; }
    
    [JsonPropertyName("modelsPromoted")]
    public int ModelsPromoted { get; set; }
    
    [JsonPropertyName("durationSeconds")]
    public double DurationSeconds { get; set; }
    
    [JsonPropertyName("version")]
    public string? Version { get; set; }
    
    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Promotion statistics
/// </summary>
public sealed class PromotionStatistics
{
    public int TotalPromotionAttempts { get; set; }
    public int SuccessfulPromotions { get; set; }
    public int FailedPromotions { get; set; }
    public int RollbackCount { get; set; }
    public double SuccessRate { get; set; }
    public double AverageDurationSeconds { get; set; }
    
    public int PromoteDecisions { get; set; }
    public int DeferDecisions { get; set; }
    public int RejectDecisions { get; set; }
    
    public Dictionary<string, int> TopFailureReasons { get; set; } = new();
}
