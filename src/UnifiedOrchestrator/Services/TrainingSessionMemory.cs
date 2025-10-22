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
/// Training Session Memory - Persists what models learned during each training session
/// 
/// Purpose: Prevent catastrophic forgetting by tracking learned patterns, weights, and parameters
/// - Saves model state after each training session
/// - Enables warm-start training from previous session
/// - Tracks learned patterns, features, and strategies
/// - Provides audit trail of learning progression
/// </summary>
public sealed class TrainingSessionMemory
{
    private readonly ILogger<TrainingSessionMemory> _logger;
    private readonly string _memoryDirectory;

    public TrainingSessionMemory(ILogger<TrainingSessionMemory> logger)
    {
        _logger = logger;
        _memoryDirectory = Path.Combine(Directory.GetCurrentDirectory(), "state", "training_memory");
        Directory.CreateDirectory(_memoryDirectory);
        
        _logger.LogInformation("[TRAINING-MEMORY] Initialized memory directory: {Directory}", _memoryDirectory);
    }

    /// <summary>
    /// Save what the model learned during this training session
    /// This creates a checkpoint that can be used for warm-start training
    /// </summary>
    public async Task SaveModelLearningAsync(
        string modelName,
        string sessionId,
        ModelLearningSnapshot snapshot,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var modelDir = Path.Combine(_memoryDirectory, modelName);
            Directory.CreateDirectory(modelDir);

            // Save the learning snapshot
            var snapshotPath = Path.Combine(modelDir, $"session_{sessionId}.json");
            var json = JsonSerializer.Serialize(snapshot, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(snapshotPath, json, cancellationToken).ConfigureAwait(false);

            // Update the latest pointer
            var latestPath = Path.Combine(modelDir, "latest.txt");
            await File.WriteAllTextAsync(latestPath, sessionId, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[TRAINING-MEMORY] ✓ Saved learning for {Model} - Session: {SessionId}", modelName, sessionId);
            _logger.LogInformation("[TRAINING-MEMORY]   - Learned {Count} new patterns", snapshot.LearnedPatterns.Count);
            _logger.LogInformation("[TRAINING-MEMORY]   - Updated {Count} parameters", snapshot.ParameterUpdates.Count);
            _logger.LogInformation("[TRAINING-MEMORY]   - Training loss: {Loss:F4}", snapshot.FinalTrainingLoss);
            _logger.LogInformation("[TRAINING-MEMORY]   - Validation score: {Score:F4}", snapshot.ValidationScore);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TRAINING-MEMORY] Failed to save learning for {Model}", modelName);
            throw;
        }
    }

    /// <summary>
    /// Load the most recent learning snapshot for warm-start training
    /// This allows models to continue learning from where they left off
    /// </summary>
    public async Task<ModelLearningSnapshot?> LoadLatestLearningAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var modelDir = Path.Combine(_memoryDirectory, modelName);
            var latestPath = Path.Combine(modelDir, "latest.txt");

            if (!File.Exists(latestPath))
            {
                _logger.LogInformation("[TRAINING-MEMORY] No previous learning found for {Model} - starting fresh", modelName);
                return null;
            }

            var sessionId = await File.ReadAllTextAsync(latestPath, cancellationToken).ConfigureAwait(false);
            var snapshotPath = Path.Combine(modelDir, $"session_{sessionId.Trim()}.json");

            if (!File.Exists(snapshotPath))
            {
                _logger.LogWarning("[TRAINING-MEMORY] Latest pointer exists but snapshot missing for {Model}", modelName);
                return null;
            }

            var json = await File.ReadAllTextAsync(snapshotPath, cancellationToken).ConfigureAwait(false);
            var snapshot = JsonSerializer.Deserialize<ModelLearningSnapshot>(json);

            if (snapshot == null)
            {
                _logger.LogWarning("[TRAINING-MEMORY] Failed to deserialize snapshot for {Model}", modelName);
                return null;
            }

            _logger.LogInformation("[TRAINING-MEMORY] ✓ Loaded previous learning for {Model}", modelName);
            _logger.LogInformation("[TRAINING-MEMORY]   - Session: {SessionId}", snapshot.SessionId);
            _logger.LogInformation("[TRAINING-MEMORY]   - Learned patterns: {Count}", snapshot.LearnedPatterns.Count);
            _logger.LogInformation("[TRAINING-MEMORY]   - Timestamp: {Time:yyyy-MM-dd HH:mm:ss}", snapshot.Timestamp);
            _logger.LogInformation("[TRAINING-MEMORY]   - Will use as warm-start for new training");

            return snapshot;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TRAINING-MEMORY] Failed to load learning for {Model}", modelName);
            return null;
        }
    }

    /// <summary>
    /// Get learning history for a specific model
    /// Shows progression of what the model learned over time
    /// </summary>
    public async Task<List<ModelLearningSnapshot>> GetLearningHistoryAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        var history = new List<ModelLearningSnapshot>();

        try
        {
            var modelDir = Path.Combine(_memoryDirectory, modelName);
            if (!Directory.Exists(modelDir))
            {
                return history;
            }

            var snapshotFiles = Directory.GetFiles(modelDir, "session_*.json");
            
            foreach (var file in snapshotFiles)
            {
                try
                {
                    var json = await File.ReadAllTextAsync(file, cancellationToken).ConfigureAwait(false);
                    var snapshot = JsonSerializer.Deserialize<ModelLearningSnapshot>(json);
                    
                    if (snapshot != null)
                    {
                        history.Add(snapshot);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TRAINING-MEMORY] Failed to load snapshot from {File}", file);
                }
            }

            history = history.OrderBy(s => s.Timestamp).ToList();
            
            _logger.LogInformation("[TRAINING-MEMORY] Retrieved {Count} historical sessions for {Model}", history.Count, modelName);
            
            return history;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TRAINING-MEMORY] Failed to get learning history for {Model}", modelName);
            return history;
        }
    }

    /// <summary>
    /// Verify that model retained knowledge from previous sessions
    /// This is the anti-forgetting check
    /// </summary>
    public async Task<(bool Retained, string Message)> VerifyKnowledgeRetentionAsync(
        string modelName,
        ModelLearningSnapshot currentSnapshot,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var history = await GetLearningHistoryAsync(modelName, cancellationToken).ConfigureAwait(false);
            
            if (history.Count == 0)
            {
                return (true, "First training session - no previous knowledge to retain");
            }

            var previousSnapshot = history.Last();

            // Check if critical patterns from previous session are still present
            var previousPatterns = previousSnapshot.LearnedPatterns.Select(p => p.PatternId).ToHashSet();
            var currentPatterns = currentSnapshot.LearnedPatterns.Select(p => p.PatternId).ToHashSet();
            
            var retainedPatterns = previousPatterns.Intersect(currentPatterns).Count();
            var retentionRate = previousPatterns.Count > 0 
                ? (decimal)retainedPatterns / previousPatterns.Count * 100 
                : 100;

            const decimal MinimumRetentionRate = 80.0m; // Must retain at least 80% of previous patterns

            if (retentionRate >= MinimumRetentionRate)
            {
                _logger.LogInformation("[TRAINING-MEMORY] ✅ KNOWLEDGE RETAINED for {Model}", modelName);
                _logger.LogInformation("[TRAINING-MEMORY]   - Previous patterns: {Previous}", previousPatterns.Count);
                _logger.LogInformation("[TRAINING-MEMORY]   - Still present: {Retained} ({Rate:F1}%)", retainedPatterns, retentionRate);
                _logger.LogInformation("[TRAINING-MEMORY]   - New patterns learned: {New}", currentPatterns.Except(previousPatterns).Count());
                
                return (true, $"Retained {retentionRate:F1}% of previous knowledge ({retainedPatterns}/{previousPatterns.Count} patterns)");
            }
            else
            {
                _logger.LogWarning("[TRAINING-MEMORY] ⚠️ CATASTROPHIC FORGETTING DETECTED for {Model}", modelName);
                _logger.LogWarning("[TRAINING-MEMORY]   - Previous patterns: {Previous}", previousPatterns.Count);
                _logger.LogWarning("[TRAINING-MEMORY]   - Still present: {Retained} ({Rate:F1}%)", retainedPatterns, retentionRate);
                _logger.LogWarning("[TRAINING-MEMORY]   - LOST patterns: {Lost}", previousPatterns.Except(currentPatterns).Count());
                
                return (false, $"Only retained {retentionRate:F1}% of previous knowledge - below {MinimumRetentionRate}% threshold");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TRAINING-MEMORY] Failed to verify knowledge retention for {Model}", modelName);
            return (false, $"Verification failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Log what the model learned in this session (proof of learning)
    /// </summary>
    public void LogLearningProof(string modelName, ModelLearningSnapshot snapshot)
    {
        _logger.LogInformation("[TRAINING-MEMORY] ═══════════════════════════════════════════════════════");
        _logger.LogInformation("[TRAINING-MEMORY] LEARNING PROOF - {Model}", modelName);
        _logger.LogInformation("[TRAINING-MEMORY] ═══════════════════════════════════════════════════════");
        _logger.LogInformation("[TRAINING-MEMORY] Session: {SessionId} at {Time:yyyy-MM-dd HH:mm:ss}", 
            snapshot.SessionId, snapshot.Timestamp);
        
        _logger.LogInformation("[TRAINING-MEMORY] Training Metrics:");
        _logger.LogInformation("[TRAINING-MEMORY]   - Training Loss: {Loss:F4} → {FinalLoss:F4} (Δ {Delta:F4})",
            snapshot.InitialTrainingLoss, snapshot.FinalTrainingLoss, 
            snapshot.InitialTrainingLoss - snapshot.FinalTrainingLoss);
        _logger.LogInformation("[TRAINING-MEMORY]   - Validation Score: {Score:F4}", snapshot.ValidationScore);
        _logger.LogInformation("[TRAINING-MEMORY]   - Epochs Trained: {Epochs}", snapshot.EpochsTrained);
        _logger.LogInformation("[TRAINING-MEMORY]   - Samples Processed: {Samples:N0}", snapshot.SamplesProcessed);

        _logger.LogInformation("[TRAINING-MEMORY] Learned Patterns: {Count}", snapshot.LearnedPatterns.Count);
        foreach (var pattern in snapshot.LearnedPatterns.Take(10))
        {
            _logger.LogInformation("[TRAINING-MEMORY]   - {Pattern}: Confidence {Confidence:F3}, Accuracy {Accuracy:F2}%",
                pattern.PatternName, pattern.Confidence, pattern.Accuracy * 100);
        }
        
        if (snapshot.LearnedPatterns.Count > 10)
        {
            _logger.LogInformation("[TRAINING-MEMORY]   - ... and {More} more patterns", 
                snapshot.LearnedPatterns.Count - 10);
        }

        _logger.LogInformation("[TRAINING-MEMORY] Parameter Updates: {Count}", snapshot.ParameterUpdates.Count);
        foreach (var param in snapshot.ParameterUpdates.Take(5))
        {
            _logger.LogInformation("[TRAINING-MEMORY]   - {Param}: {OldValue:F4} → {NewValue:F4}",
                param.Key, param.Value.OldValue, param.Value.NewValue);
        }
        
        if (snapshot.ParameterUpdates.Count > 5)
        {
            _logger.LogInformation("[TRAINING-MEMORY]   - ... and {More} more parameters",
                snapshot.ParameterUpdates.Count - 5);
        }

        _logger.LogInformation("[TRAINING-MEMORY] Model State Saved: {Path}", snapshot.ModelCheckpointPath);
        _logger.LogInformation("[TRAINING-MEMORY] ═══════════════════════════════════════════════════════");
    }
}

/// <summary>
/// Snapshot of what a model learned during a training session
/// </summary>
public sealed class ModelLearningSnapshot
{
    public string SessionId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string ModelName { get; set; } = string.Empty;
    public string ModelVersion { get; set; } = string.Empty;
    
    // Training metrics
    public decimal InitialTrainingLoss { get; set; }
    public decimal FinalTrainingLoss { get; set; }
    public decimal ValidationScore { get; set; }
    public int EpochsTrained { get; set; }
    public int SamplesProcessed { get; set; }
    
    // Learned patterns and strategies
    public List<LearnedPattern> LearnedPatterns { get; set; } = new();
    
    // Parameter updates (before → after)
    public Dictionary<string, ParameterUpdate> ParameterUpdates { get; set; } = new();
    
    // Checkpoint path for warm-start
    public string ModelCheckpointPath { get; set; } = string.Empty;
    
    // Feature importance learned
    public Dictionary<string, decimal> FeatureImportance { get; set; } = new();
}

/// <summary>
/// A pattern that the model learned to recognize
/// </summary>
public sealed class LearnedPattern
{
    public string PatternId { get; set; } = string.Empty;
    public string PatternName { get; set; } = string.Empty;
    public decimal Confidence { get; set; }
    public decimal Accuracy { get; set; }
    public int OccurrencesInData { get; set; }
}

/// <summary>
/// Update to a model parameter
/// </summary>
public sealed class ParameterUpdate
{
    public decimal OldValue { get; set; }
    public decimal NewValue { get; set; }
    public decimal ChangePercent => OldValue != 0 ? (NewValue - OldValue) / OldValue * 100 : 0;
}
