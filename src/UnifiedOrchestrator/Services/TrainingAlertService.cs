using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Alert notification service for training events
/// Provides structured logging with optional webhook integration (Slack/Teams/Email)
/// Currently focuses on comprehensive logging; webhook delivery can be added later
/// </summary>
internal sealed class TrainingAlertService
{
    private readonly ILogger<TrainingAlertService> _logger;
    private readonly string _alertLogPath;
    private readonly bool _enableFileLogging;

    public TrainingAlertService(ILogger<TrainingAlertService> logger)
    {
        _logger = logger;
        _alertLogPath = Path.Combine(Directory.GetCurrentDirectory(), "state", "training_alerts.log");
        _enableFileLogging = true;
        
        // Ensure directory exists
        Directory.CreateDirectory(Path.GetDirectoryName(_alertLogPath)!);
    }

    /// <summary>
    /// Alert: Training run started
    /// </summary>
    public async Task AlertTrainingStartedAsync(
        string runId,
        string gitHash,
        Dictionary<string, object> parameters,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = "TRAINING_STARTED",
            RunId = runId,
            GitHash = gitHash,
            Timestamp = DateTime.UtcNow,
            Parameters = parameters
        };

        _logger.LogInformation("🚀 [ALERT] Training run STARTED - RunID: {RunId}, Git: {Git}",
            runId, gitHash ?? "N/A");

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Alert: Training run completed successfully
    /// </summary>
    public async Task AlertTrainingSuccessAsync(
        string runId,
        double durationMinutes,
        int modelsPromoted,
        int modelsDiscarded,
        Dictionary<string, object> metrics,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = "TRAINING_SUCCESS",
            RunId = runId,
            DurationMinutes = durationMinutes,
            ModelsPromoted = modelsPromoted,
            ModelsDiscarded = modelsDiscarded,
            Timestamp = DateTime.UtcNow,
            Metrics = metrics
        };

        _logger.LogInformation("✅ [ALERT] Training run SUCCESS - RunID: {RunId}, Duration: {Duration:F1}m, Promoted: {Promoted}, Discarded: {Discarded}",
            runId, durationMinutes, modelsPromoted, modelsDiscarded);

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Alert: Training run failed
    /// </summary>
    public async Task AlertTrainingFailureAsync(
        string runId,
        string errorMessage,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = "TRAINING_FAILURE",
            RunId = runId,
            ErrorMessage = errorMessage,
            Timestamp = DateTime.UtcNow
        };

        _logger.LogError("❌ [ALERT] Training run FAILED - RunID: {RunId}, Error: {Error}",
            runId, errorMessage);

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }
    
    /// <summary>
    /// Alert: Training run failed with details
    /// </summary>
    public async Task AlertTrainingFailureAsync(
        string runId,
        string errorMessage,
        List<string> failedComponents,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = "TRAINING_FAILURE",
            RunId = runId,
            ErrorMessage = errorMessage,
            FailedComponents = failedComponents,
            Timestamp = DateTime.UtcNow
        };

        _logger.LogError("❌ [ALERT] Training run FAILED - RunID: {RunId}, Error: {Error}, Failed: {Components}",
            runId, errorMessage, string.Join(", ", failedComponents));

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Alert: Missed scheduled training run
    /// </summary>
    public async Task AlertMissedScheduleAsync(
        DateTime expectedTime,
        string reason,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = "MISSED_SCHEDULE",
            ExpectedTime = expectedTime,
            Reason = reason,
            Timestamp = DateTime.UtcNow
        };

        _logger.LogWarning("⚠️ [ALERT] Missed scheduled training - Expected: {Expected}, Reason: {Reason}",
            expectedTime, reason);

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Alert: Pre-training health check failed
    /// </summary>
    public async Task AlertHealthCheckFailureAsync(
        string checkName,
        string errorMessage,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = "HEALTH_CHECK_FAILED",
            CheckName = checkName,
            ErrorMessage = errorMessage,
            Timestamp = DateTime.UtcNow
        };

        _logger.LogError("❌ [ALERT] Health check FAILED - Check: {Check}, Error: {Error}",
            checkName, errorMessage);

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Alert: Model promotion decision
    /// </summary>
    public async Task AlertPromotionDecisionAsync(
        string modelName,
        string version,
        bool promoted,
        string reason,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = promoted ? "MODEL_PROMOTED" : "MODEL_REJECTED",
            ModelName = modelName,
            Version = version,
            Promoted = promoted,
            Reason = reason,
            Timestamp = DateTime.UtcNow
        };

        if (promoted)
        {
            _logger.LogInformation("🎉 [ALERT] Model PROMOTED - {Model} v{Version}: {Reason}",
                modelName, version, reason);
        }
        else
        {
            _logger.LogInformation("🚫 [ALERT] Model REJECTED - {Model} v{Version}: {Reason}",
                modelName, version, reason);
        }

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Alert: Training timeout/watchdog triggered
    /// </summary>
    public async Task AlertTrainingTimeoutAsync(
        string runId,
        double maxHours,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = "TRAINING_TIMEOUT",
            RunId = runId,
            MaxHours = maxHours,
            Timestamp = DateTime.UtcNow
        };

        _logger.LogCritical("⏰ [ALERT] Training TIMEOUT - RunID: {RunId}, Exceeded {Hours}h maximum",
            runId, maxHours);

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Alert: Data integrity issue detected
    /// </summary>
    public async Task AlertDataIntegrityIssueAsync(
        string issueType,
        string description,
        CancellationToken cancellationToken = default)
    {
        var alert = new
        {
            EventType = "DATA_INTEGRITY_ISSUE",
            IssueType = issueType,
            Description = description,
            Timestamp = DateTime.UtcNow
        };

        _logger.LogWarning("⚠️ [ALERT] Data integrity issue - Type: {Type}, Details: {Details}",
            issueType, description);

        await LogAlertAsync(alert, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Log alert to file for audit trail (structured JSON format)
    /// </summary>
    private async Task LogAlertAsync(object alert, CancellationToken cancellationToken)
    {
        if (!_enableFileLogging)
        {
            return;
        }

        try
        {
            var json = JsonSerializer.Serialize(alert, new JsonSerializerOptions { WriteIndented = false });
            var logEntry = $"{json}\n";
            
            await File.AppendAllTextAsync(_alertLogPath, logEntry, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ALERT] Failed to write alert to file: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Get recent alerts for monitoring dashboard
    /// </summary>
    public async Task<List<Dictionary<string, object>>> GetRecentAlertsAsync(
        int count = 50,
        CancellationToken cancellationToken = default)
    {
        var alerts = new List<Dictionary<string, object>>();

        if (!File.Exists(_alertLogPath))
        {
            return alerts;
        }

        try
        {
            var lines = await File.ReadAllLinesAsync(_alertLogPath, cancellationToken).ConfigureAwait(false);
            
            // Get last N lines
            var recentLines = lines.Reverse().Take(count).Reverse();
            
            foreach (var line in recentLines)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                try
                {
                    var alert = JsonSerializer.Deserialize<Dictionary<string, object>>(line);
                    if (alert != null)
                    {
                        alerts.Add(alert);
                    }
                }
                catch
                {
                    // Skip malformed lines
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ALERT] Failed to read alert log: {Error}", ex.Message);
        }

        return alerts;
    }
}
