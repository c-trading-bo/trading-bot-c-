using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Training Checkpoint Service - Phase 13: Failure Handling & Recovery
/// Manages checkpoints for training sessions to enable resumption after failures
/// Saves state after each component to prevent loss of training progress
/// </summary>
internal sealed class TrainingCheckpointService
{
    private readonly ILogger<TrainingCheckpointService> _logger;
    private readonly string _checkpointDirectory;

    public TrainingCheckpointService(ILogger<TrainingCheckpointService> logger)
    {
        _logger = logger;
        _checkpointDirectory = Path.Combine(
            Directory.GetCurrentDirectory(),
            "artifacts",
            "checkpoints");
        
        Directory.CreateDirectory(_checkpointDirectory);
    }

    /// <summary>
    /// Save checkpoint with current training state
    /// Phase 13.1: Training Checkpoint System
    /// </summary>
    public async Task SaveCheckpointAsync(
        TrainingSessionState state,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var checkpointPath = Path.Combine(_checkpointDirectory, $"checkpoint-{state.SessionId}.json");
            
            _logger.LogInformation("[CHECKPOINT] Saving checkpoint for session {SessionId} - {Completed}/{Total} components completed",
                state.SessionId, state.ComponentsCompleted.Count, state.TotalComponents);

            var json = JsonSerializer.Serialize(state, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            await File.WriteAllTextAsync(checkpointPath, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogDebug("[CHECKPOINT] Checkpoint saved: {Path}", checkpointPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CHECKPOINT] Failed to save checkpoint: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Find most recent checkpoint file
    /// Phase 13.2: Checkpoint Resume Logic
    /// </summary>
    public string? FindMostRecentCheckpoint()
    {
        try
        {
            var checkpoints = Directory.GetFiles(_checkpointDirectory, "checkpoint-*.json")
                .Select(f => new FileInfo(f))
                .OrderByDescending(f => f.LastWriteTimeUtc)
                .ToList();

            if (checkpoints.Count == 0)
            {
                _logger.LogDebug("[CHECKPOINT] No checkpoints found");
                return null;
            }

            var mostRecent = checkpoints.First();
            _logger.LogInformation("[CHECKPOINT] Found checkpoint: {File} (modified {Time})",
                mostRecent.Name, mostRecent.LastWriteTimeUtc);

            return mostRecent.FullName;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CHECKPOINT] Failed to find checkpoints: {Error}", ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Load checkpoint from file
    /// Phase 13.2: Checkpoint Resume Logic
    /// </summary>
    public async Task<TrainingSessionState?> LoadCheckpointAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(checkpointPath))
            {
                _logger.LogWarning("[CHECKPOINT] Checkpoint file not found: {Path}", checkpointPath);
                return null;
            }

            var json = await File.ReadAllTextAsync(checkpointPath, cancellationToken).ConfigureAwait(false);
            var state = JsonSerializer.Deserialize<TrainingSessionState>(json);

            if (state == null)
            {
                _logger.LogError("[CHECKPOINT] Failed to deserialize checkpoint");
                return null;
            }

            _logger.LogInformation("[CHECKPOINT] Loaded checkpoint: Session {SessionId}, Phase {Phase}, Component {Current}/{Total}",
                state.SessionId, state.CurrentPhase, state.CurrentComponentIndex, state.TotalComponents);

            return state;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CHECKPOINT] Failed to load checkpoint: {Error}", ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Validate checkpoint integrity
    /// Phase 13.11: Session Recovery Validation
    /// </summary>
    public async Task<bool> ValidateCheckpointAsync(
        TrainingSessionState state,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[CHECKPOINT] Validating checkpoint for session {SessionId}...", state.SessionId);

            var issues = new List<string>();

            // Check 1: All completed components have model files on disk
            var modelRegistry = Path.Combine(Directory.GetCurrentDirectory(), "model_registry");
            foreach (var componentId in state.ComponentsCompleted)
            {
                var modelPath = Path.Combine(modelRegistry, $"{componentId}.onnx");
                if (!File.Exists(modelPath))
                {
                    issues.Add($"Missing model file for component {componentId}");
                }
            }

            // Check 2: System resources (just log, don't block)
            var gcInfo = GC.GetGCMemoryInfo();
            var availableMemoryGB = (gcInfo.TotalAvailableMemoryBytes - gcInfo.MemoryLoadBytes) / (1024.0 * 1024.0 * 1024.0);
            _logger.LogInformation("[CHECKPOINT] Available memory: {Memory:F1} GB", availableMemoryGB);

            // Check 3: Disk space (just log, don't block)
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
            var drive = new DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
            var freeSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
            _logger.LogInformation("[CHECKPOINT] Available disk space: {Space:F1} GB", freeSpaceGB);

            if (issues.Count > 0)
            {
                _logger.LogError("[CHECKPOINT] Validation failed: {Issues}", string.Join("; ", issues));
                return false;
            }

            _logger.LogInformation("[CHECKPOINT] Checkpoint validation passed");
            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CHECKPOINT] Validation failed with exception: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Archive old checkpoint (no longer needed)
    /// </summary>
    public async Task ArchiveCheckpointAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(checkpointPath))
                return;

            var archivePath = checkpointPath.Replace(".json", $"_archived_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            File.Move(checkpointPath, archivePath);
            
            _logger.LogInformation("[CHECKPOINT] Archived checkpoint: {Archive}", Path.GetFileName(archivePath));
            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[CHECKPOINT] Failed to archive checkpoint: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Delete checkpoint (training completed successfully)
    /// </summary>
    public void DeleteCheckpoint(string sessionId)
    {
        try
        {
            var checkpointPath = Path.Combine(_checkpointDirectory, $"checkpoint-{sessionId}.json");
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
                _logger.LogInformation("[CHECKPOINT] Deleted checkpoint for completed session {SessionId}", sessionId);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[CHECKPOINT] Failed to delete checkpoint: {Error}", ex.Message);
        }
    }
}

/// <summary>
/// Training session state for checkpointing
/// Phase 13.1: Training Checkpoint System
/// </summary>
internal class TrainingSessionState
{
    public string SessionId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime CheckpointTime { get; set; }
    
    // Training progress
    public string CurrentPhase { get; set; } = "Heavy"; // Heavy, Medium, Light
    public int CurrentComponentIndex { get; set; }
    public int TotalComponents { get; set; } = 273;
    
    // Component tracking
    public List<string> ComponentsCompleted { get; set; } = new();
    public List<ComponentFailure> ComponentsFailed { get; set; } = new();
    public List<string> ComponentsPending { get; set; } = new();
    
    // Training metrics
    public double TotalTrainingTimeMinutes { get; set; }
    public int ModelsGenerated { get; set; }
    
    // System resources at checkpoint
    public double DiskSpaceGB { get; set; }
    public double MemoryUsageGB { get; set; }
}

/// <summary>
/// Component failure record
/// Phase 13.3: Per-Component Failure Handling
/// </summary>
internal class ComponentFailure
{
    public string ComponentId { get; set; } = string.Empty;
    public string ErrorMessage { get; set; } = string.Empty;
    public string FailureType { get; set; } = "Unknown"; // Transient, Permanent, Resource
    public int RetryCount { get; set; }
    public DateTime FailedAt { get; set; }
}

/// <summary>
/// Component training result
/// Phase 13.3: Per-Component Failure Handling
/// </summary>
internal class ComponentTrainingResult
{
    public string ComponentId { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public string? FailureType { get; set; }
    public TimeSpan Duration { get; set; }
    public int RetryCount { get; set; }
}
