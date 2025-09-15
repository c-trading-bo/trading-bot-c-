using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.ModelLifecycle;

/// <summary>
/// Model versioning interface for tracking and managing model versions with integrity validation
/// </summary>
public interface IModelVersionManager
{
    /// <summary>
    /// Load a specific model version by hash with integrity validation
    /// </summary>
    Task<ModelMetadata?> LoadModelAsync(string modelHash, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register a new model version with metadata and hash
    /// </summary>
    Task<string> RegisterModelAsync(ModelMetadata metadata, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get the currently active model metadata
    /// </summary>
    Task<ModelMetadata?> GetActiveModelAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Set the active model version (stage as "pending" before activation)
    /// </summary>
    Task SetActiveModelAsync(string modelHash, bool isPending = false, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// List all available model versions
    /// </summary>
    Task<List<ModelMetadata>> ListModelsAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validate model file integrity using stored hash
    /// </summary>
    Task<bool> ValidateModelIntegrityAsync(string modelHash, CancellationToken cancellationToken = default);
}

/// <summary>
/// Comprehensive model metadata with versioning and integrity information
/// </summary>
public record ModelMetadata(
    string Hash,
    string Version,
    DateTime CreatedAt,
    string FilePath,
    long FileSizeBytes,
    Dictionary<string, object> Hyperparameters,
    ModelPerformanceMetrics? Performance = null,
    string? Description = null,
    string? Status = null // "pending", "active", "rollback", "deprecated"
);

/// <summary>
/// Model performance metrics for evaluation and comparison
/// </summary>
public record ModelPerformanceMetrics(
    double Accuracy,
    double Precision,
    double Recall,
    double F1Score,
    double SharpeRatio,
    double MaxDrawdown,
    int TrainingSamples,
    DateTime EvaluatedAt
);