using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Represents one Sunday training session with complete state tracking
/// Enables session recovery, progress monitoring, and result reporting
/// </summary>
public sealed class TrainingSession
{
    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() }
    };

    /// <summary>
    /// Unique session identifier (e.g., "train-20250119-120004")
    /// </summary>
    public string SessionId { get; set; } = string.Empty;

    /// <summary>
    /// When session began (UTC)
    /// </summary>
    public DateTimeOffset StartTime { get; set; }

    /// <summary>
    /// When session finished (UTC), null if still running
    /// </summary>
    public DateTimeOffset? EndTime { get; set; }

    /// <summary>
    /// Current session status
    /// </summary>
    public TrainingSessionStatus Status { get; set; } = TrainingSessionStatus.NotStarted;

    /// <summary>
    /// Path to lock file preventing concurrent sessions
    /// </summary>
    public string LockFilePath { get; set; } = string.Empty;

    /// <summary>
    /// Total number of components to train (25: Heavy=11, Medium=7, Light=7)
    /// </summary>
    public int ComponentsTotal { get; set; }

    /// <summary>
    /// Count of successfully trained components
    /// </summary>
    public int ComponentsCompleted { get; set; }

    /// <summary>
    /// Count of failed components
    /// </summary>
    public int ComponentsFailed { get; set; }

    /// <summary>
    /// Names of components that failed
    /// </summary>
    public List<string> FailedComponentNames { get; set; } = new();

    /// <summary>
    /// Current training phase
    /// </summary>
    public TrainingPhase CurrentPhase { get; set; } = TrainingPhase.None;

    /// <summary>
    /// Name of component currently training (null if between components)
    /// </summary>
    public string? CurrentComponent { get; set; }

    /// <summary>
    /// How long session has been running
    /// </summary>
    [JsonIgnore]
    public TimeSpan TotalElapsedTime => EndTime.HasValue
        ? EndTime.Value - StartTime
        : DateTimeOffset.UtcNow - StartTime;

    /// <summary>
    /// Estimated time remaining (null if cannot estimate)
    /// </summary>
    public TimeSpan? EstimatedTimeRemaining { get; set; }

    /// <summary>
    /// Path to generated manifest file
    /// </summary>
    public string? ManifestFilePath { get; set; }

    /// <summary>
    /// Whether models were successfully promoted
    /// </summary>
    public bool? PromotionSuccess { get; set; }

    /// <summary>
    /// Additional session metadata
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Create lock file to prevent concurrent training sessions
    /// </summary>
    public void CreateLockFile()
    {
        try
        {
            var lockInfo = new
            {
                SessionId,
                StartTime = StartTime.ToString("O"),
                ProcessId = Environment.ProcessId,
                MachineName = Environment.MachineName
            };

            var json = JsonSerializer.Serialize(lockInfo, _jsonOptions);
            File.WriteAllText(LockFilePath, json);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to create lock file: {LockFilePath}", ex);
        }
    }

    /// <summary>
    /// Remove lock file when session completes
    /// </summary>
    public void RemoveLockFile()
    {
        try
        {
            if (File.Exists(LockFilePath))
            {
                File.Delete(LockFilePath);
            }
        }
        catch
        {
            // Best effort - don't fail session if lock file removal fails
        }
    }

    /// <summary>
    /// Save session checkpoint to disk for resumability
    /// </summary>
    public async Task SaveCheckpointAsync(string checkpointPath)
    {
        try
        {
            var directory = Path.GetDirectoryName(checkpointPath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var json = JsonSerializer.Serialize(this, _jsonOptions);
            await File.WriteAllTextAsync(checkpointPath, json).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to save checkpoint: {checkpointPath}", ex);
        }
    }

    /// <summary>
    /// Load session checkpoint from disk
    /// </summary>
    public static async Task<TrainingSession?> LoadCheckpointAsync(string checkpointPath)
    {
        try
        {
            if (!File.Exists(checkpointPath))
            {
                return null;
            }

            var json = await File.ReadAllTextAsync(checkpointPath).ConfigureAwait(false);
            return JsonSerializer.Deserialize<TrainingSession>(json, _jsonOptions);
        }
        catch
        {
            return null; // Checkpoint corrupted or incompatible
        }
    }

    /// <summary>
    /// Update progress counters and calculate ETA
    /// </summary>
    public void UpdateProgress(int completedCount, int failedCount, TimeSpan? estimatedRemaining = null)
    {
        ComponentsCompleted = completedCount;
        ComponentsFailed = failedCount;
        EstimatedTimeRemaining = estimatedRemaining;
    }

    /// <summary>
    /// Record component training success
    /// </summary>
    public void RecordComponentSuccess(string componentName)
    {
        ComponentsCompleted++;
        CurrentComponent = null;
    }

    /// <summary>
    /// Record component training failure with error details
    /// </summary>
    public void RecordComponentFailure(string componentName, string errorMessage)
    {
        ComponentsFailed++;
        FailedComponentNames.Add($"{componentName}: {errorMessage}");
        CurrentComponent = null;
    }

    /// <summary>
    /// Generate final session summary report
    /// </summary>
    public TrainingSessionSummary GenerateSummary()
    {
        return new TrainingSessionSummary
        {
            SessionId = SessionId,
            StartTime = StartTime,
            EndTime = EndTime ?? DateTimeOffset.UtcNow,
            Duration = TotalElapsedTime,
            Status = Status,
            ComponentsTotal = ComponentsTotal,
            ComponentsCompleted = ComponentsCompleted,
            ComponentsFailed = ComponentsFailed,
            SuccessRate = ComponentsTotal > 0
                ? (double)ComponentsCompleted / ComponentsTotal
                : 0.0,
            FailedComponents = new List<string>(FailedComponentNames),
            PromotionSuccess = PromotionSuccess ?? false,
            ManifestFilePath = ManifestFilePath
        };
    }
}

/// <summary>
/// Training session status enum
/// </summary>
public enum TrainingSessionStatus
{
    NotStarted,
    HealthChecks,
    Training,
    Validation,
    Promotion,
    Complete,
    Failed
}

/// <summary>
/// Training phase enum
/// </summary>
public enum TrainingPhase
{
    None,
    Heavy,
    Medium,
    Light
}

/// <summary>
/// Training session summary for reporting
/// </summary>
public sealed class TrainingSessionSummary
{
    public string SessionId { get; set; } = string.Empty;
    public DateTimeOffset StartTime { get; set; }
    public DateTimeOffset EndTime { get; set; }
    public TimeSpan Duration { get; set; }
    public TrainingSessionStatus Status { get; set; }
    public int ComponentsTotal { get; set; }
    public int ComponentsCompleted { get; set; }
    public int ComponentsFailed { get; set; }
    public double SuccessRate { get; set; }
    public List<string> FailedComponents { get; set; } = new();
    public bool PromotionSuccess { get; set; }
    public string? ManifestFilePath { get; set; }
}
