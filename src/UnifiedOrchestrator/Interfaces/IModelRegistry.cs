using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for model registry - versioned, immutable artifact storage
/// </summary>
internal interface IModelRegistry
{
    /// <summary>
    /// Register a new model version
    /// </summary>
    Task<string> RegisterModelAsync(ModelVersion model, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get the current champion model for an algorithm
    /// </summary>
    Task<ModelVersion?> GetChampionAsync(string algorithm, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get a specific model version
    /// </summary>
    Task<ModelVersion?> GetModelAsync(string versionId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get all model versions for an algorithm
    /// </summary>
    Task<IReadOnlyList<ModelVersion>> GetModelsAsync(string algorithm, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Promote a challenger to champion
    /// </summary>
    Task<bool> PromoteToChampionAsync(string algorithm, string challengerVersionId, PromotionRecord promotionRecord, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Rollback to previous champion
    /// </summary>
    Task<bool> RollbackToPreviousAsync(string algorithm, string reason, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get promotion history for an algorithm
    /// </summary>
    Task<IReadOnlyList<PromotionRecord>> GetPromotionHistoryAsync(string algorithm, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validate model artifact integrity
    /// </summary>
    Task<bool> ValidateArtifactAsync(string versionId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Clean up old model versions (keep recent ones)
    /// </summary>
    Task CleanupOldModelsAsync(string algorithm, int keepCount = 10, CancellationToken cancellationToken = default);
}

/// <summary>
/// Interface for atomic model routing with champion/challenger support
/// </summary>
internal interface IModelRouter<T> where T : class
{
    /// <summary>
    /// Get the current champion model (read-only access)
    /// </summary>
    T? Current { get; }
    
    /// <summary>
    /// Get the current champion version information
    /// </summary>
    ModelVersion? CurrentVersion { get; }
    
    /// <summary>
    /// Atomically swap to a new champion model
    /// </summary>
    Task<bool> SwapAsync(T newModel, ModelVersion newVersion, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get model loading statistics
    /// </summary>
    Task<ModelRouterStats> GetStatsAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Statistics for model router
/// </summary>
internal class ModelRouterStats
{
    public string Algorithm { get; set; } = string.Empty;
    public string CurrentVersionId { get; set; } = string.Empty;
    public DateTime LoadedAt { get; set; }
    public int SwapCount { get; set; }
    public DateTime LastSwapAt { get; set; }
    public TimeSpan LastSwapDuration { get; set; }
    public bool IsHealthy { get; set; }
    public Dictionary<string, object> Metadata { get; } = new();
}

/// <summary>
/// Factory for creating model routers
/// </summary>
internal interface IModelRouterFactory
{
    /// <summary>
    /// Create a model router for an algorithm
    /// </summary>
    IModelRouter<T> CreateRouter<T>(string algorithm) where T : class;
    
    /// <summary>
    /// Get an existing router
    /// </summary>
    IModelRouter<T>? GetRouter<T>(string algorithm) where T : class;
    
    /// <summary>
    /// Get all active routers
    /// </summary>
    IReadOnlyDictionary<string, object> GetAllRouters();
}