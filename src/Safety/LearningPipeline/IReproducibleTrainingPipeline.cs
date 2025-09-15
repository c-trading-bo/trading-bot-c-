using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.LearningPipeline;

/// <summary>
/// Interface for reproducible training pipeline with complete metadata logging
/// </summary>
public interface IReproducibleTrainingPipeline
{
    /// <summary>
    /// Execute a training run with full metadata logging for reproducibility
    /// </summary>
    Task<TrainingRunResult> ExecuteTrainingRunAsync(TrainingRunConfiguration config, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Record training run metadata for audit and reproduction
    /// </summary>
    Task RecordTrainingMetadataAsync(TrainingRunMetadata metadata, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Retrieve training run metadata for reproduction
    /// </summary>
    Task<TrainingRunMetadata?> GetTrainingMetadataAsync(string runId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// List all training runs with optional filtering
    /// </summary>
    Task<List<TrainingRunMetadata>> ListTrainingRunsAsync(TrainingRunFilter? filter = null, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Reproduce a previous training run exactly
    /// </summary>
    Task<TrainingRunResult> ReproduceTrainingRunAsync(string originalRunId, CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for a training run
/// </summary>
public record TrainingRunConfiguration(
    string RunId,
    Dictionary<string, object> Hyperparameters,
    string DatasetHash,
    int? RandomSeed = null,
    string? ModelArchitecture = null,
    Dictionary<string, object>? EnvironmentVariables = null,
    List<string>? RequiredPackages = null
);

/// <summary>
/// Complete metadata for a training run
/// </summary>
public record TrainingRunMetadata(
    string RunId,
    DateTime StartTime,
    DateTime? EndTime,
    TrainingRunConfiguration Configuration,
    string DatasetHash,
    string? DataSliceDescription,
    Dictionary<string, object> Hyperparameters,
    int RandomSeed,
    string ModelOutputPath,
    string ModelHash,
    TrainingMetrics? Metrics = null,
    string? Status = null,
    Dictionary<string, string>? SystemInfo = null,
    List<string>? LogFiles = null
);

/// <summary>
/// Training metrics and results
/// </summary>
public record TrainingMetrics(
    double TrainingAccuracy,
    double ValidationAccuracy,
    double TestAccuracy,
    double TrainingLoss,
    double ValidationLoss,
    double TestLoss,
    int Epochs,
    TimeSpan TrainingDuration,
    Dictionary<string, double>? CustomMetrics = null
);

/// <summary>
/// Result of a training run
/// </summary>
public record TrainingRunResult(
    string RunId,
    bool Success,
    string ModelPath,
    string ModelHash,
    TrainingMetrics Metrics,
    List<string> LogMessages,
    Exception? Error = null
);

/// <summary>
/// Filter for querying training runs
/// </summary>
public record TrainingRunFilter(
    DateTime? StartDate = null,
    DateTime? EndDate = null,
    string? DatasetHash = null,
    string? Status = null,
    double? MinAccuracy = null,
    Dictionary<string, object>? HyperparameterFilter = null
);