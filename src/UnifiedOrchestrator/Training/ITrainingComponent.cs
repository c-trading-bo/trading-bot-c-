using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Standard interface that all 25 training components implement
/// Provides unified way for orchestrator to invoke training on any component
/// </summary>
public interface ITrainingComponent
{
    /// <summary>
    /// Main training method taking configuration and returning result
    /// </summary>
    Task<TrainingResult> TrainAsync(
        TrainingConfiguration configuration,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns what data is needed (experiences, historical bars)
    /// </summary>
    Task<TrainingDataRequirements> GetRequiredDataAsync(
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if component can train (data available, dependencies met)
    /// </summary>
    Task<PrerequisiteCheckResult> ValidatePrerequisitesAsync(
        TrainingConfiguration configuration,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Persist trained model to disk
    /// </summary>
    Task SaveModelAsync(string modelPath, CancellationToken cancellationToken = default);

    /// <summary>
    /// Load existing model from disk for validation
    /// </summary>
    Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default);
}

/// <summary>
/// Training configuration passed to components
/// </summary>
public sealed class TrainingConfiguration
{
    /// <summary>
    /// Batch size for training
    /// </summary>
    public int BatchSize { get; set; } = 128;

    /// <summary>
    /// Number of training epochs
    /// </summary>
    public int Epochs { get; set; } = 10;

    /// <summary>
    /// Learning rate
    /// </summary>
    public double LearningRate { get; set; } = 0.0003;

    /// <summary>
    /// Experience data from database
    /// </summary>
    public List<object>? ExperienceData { get; set; }

    /// <summary>
    /// Historical bars (symbol -> bar list)
    /// </summary>
    public Dictionary<string, List<object>>? HistoricalBars { get; set; }

    /// <summary>
    /// Path to save training checkpoints
    /// </summary>
    public string? CheckpointPath { get; set; }

    /// <summary>
    /// Progress callback for updates during training
    /// </summary>
    public Action<TrainingProgress>? ProgressCallback { get; set; }

    /// <summary>
    /// Additional configuration parameters
    /// </summary>
    public Dictionary<string, object>? AdditionalParameters { get; set; }
}

/// <summary>
/// Training result returned by components
/// </summary>
public sealed class TrainingResult
{
    /// <summary>
    /// Whether training succeeded
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Final training loss metric
    /// </summary>
    public double FinalLoss { get; set; }

    /// <summary>
    /// How many epochs were completed
    /// </summary>
    public int EpochsCompleted { get; set; }

    /// <summary>
    /// Duration of training
    /// </summary>
    public TimeSpan TimeTaken { get; set; }

    /// <summary>
    /// Where model was saved
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// List of checkpoint file paths
    /// </summary>
    public List<string>? Checkpoints { get; set; }

    /// <summary>
    /// Error message if failed
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Additional metrics
    /// </summary>
    public Dictionary<string, double>? Metrics { get; set; }
}

/// <summary>
/// Progress updates during training
/// </summary>
public sealed class TrainingProgress
{
    /// <summary>
    /// Current epoch
    /// </summary>
    public int CurrentEpoch { get; set; }

    /// <summary>
    /// Total epochs
    /// </summary>
    public int TotalEpochs { get; set; }

    /// <summary>
    /// Current loss value
    /// </summary>
    public double CurrentLoss { get; set; }

    /// <summary>
    /// Estimated time remaining
    /// </summary>
    public TimeSpan? EstimatedTimeRemaining { get; set; }

    /// <summary>
    /// Progress percentage (0-100)
    /// </summary>
    public double ProgressPercentage => TotalEpochs > 0
        ? (double)CurrentEpoch / TotalEpochs * 100.0
        : 0.0;
}

/// <summary>
/// Data requirements for training component
/// </summary>
public sealed class TrainingDataRequirements
{
    /// <summary>
    /// Requires experience database
    /// </summary>
    public bool RequiresExperiences { get; set; }

    /// <summary>
    /// Requires historical bars
    /// </summary>
    public bool RequiresHistoricalData { get; set; }

    /// <summary>
    /// Minimum number of experiences needed
    /// </summary>
    public int MinimumExperiences { get; set; }

    /// <summary>
    /// Minimum number of historical bars needed
    /// </summary>
    public int MinimumHistoricalBars { get; set; }

    /// <summary>
    /// List of dependent component names that must train first
    /// </summary>
    public List<string>? DependentComponents { get; set; }
}

/// <summary>
/// Result of prerequisite validation
/// </summary>
public sealed class PrerequisiteCheckResult
{
    /// <summary>
    /// Whether all prerequisites are met
    /// </summary>
    public bool CanTrain { get; set; }

    /// <summary>
    /// List of unmet prerequisites
    /// </summary>
    public List<string>? UnmetPrerequisites { get; set; }

    /// <summary>
    /// Warning messages (non-blocking)
    /// </summary>
    public List<string>? Warnings { get; set; }
}
