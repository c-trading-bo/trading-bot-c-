using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Backtest
{
    /// <summary>
    /// Model metadata card containing training and validation information
    /// Ensures temporal integrity - no future leakage in model selection
    /// </summary>
    public record ModelCard(
        string ModelId,
        string FamilyName,
        string Version,
        DateTime TrainedAt,
        DateTime TrainingDataStart,
        DateTime TrainingDataEnd,
        Dictionary<string, double> Metrics,
        string ModelPath,
        string ConfigPath,
        bool IsActive
    );

    /// <summary>
    /// File paths for model artifacts
    /// Points to ONNX model files, config files, and metadata
    /// </summary>
    public record ModelPaths(
        string OnnxModelPath,
        string ConfigPath,
        string MetadataPath
    );

    /// <summary>
    /// Interface for managing ML models with temporal integrity
    /// Prevents future leakage by loading only historically accurate models
    /// </summary>
    public interface IModelRegistry
    {
        /// <summary>
        /// Get the most recent model that was trained before the specified validation date
        /// Critical for walk-forward validation - prevents using future information
        /// </summary>
        /// <param name="familyName">Model family name (e.g., "ESStrategy", "MomentumModel")</param>
        /// <param name="asOfDate">Validation date - model must be trained before this date</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Historically appropriate model card</returns>
        Task<ModelCard?> GetModelAsOfDateAsync(
            string familyName, 
            DateTime asOfDate, 
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Get file paths for model artifacts
        /// Returns paths to ONNX model, configuration, and metadata files
        /// </summary>
        /// <param name="modelId">Unique model identifier</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Model file paths</returns>
        Task<ModelPaths?> GetModelPathsAsync(
            string modelId, 
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Register a new model in the registry
        /// Called after successful model training to make it available for backtesting
        /// </summary>
        /// <param name="modelCard">Model metadata to register</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>True if registration successful</returns>
        Task<bool> RegisterModelAsync(
            ModelCard modelCard, 
            CancellationToken cancellationToken = default);

        /// <summary>
        /// List all available models for a family
        /// Used for debugging and model selection validation
        /// </summary>
        /// <param name="familyName">Model family name</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>List of model cards in chronological order</returns>
        Task<List<ModelCard>> ListModelsAsync(
            string familyName, 
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Check if a model exists and is accessible
        /// Used to validate model availability before backtesting
        /// </summary>
        /// <param name="modelId">Model identifier to check</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>True if model exists and is accessible</returns>
        Task<bool> ModelExistsAsync(
            string modelId, 
            CancellationToken cancellationToken = default);
    }
}