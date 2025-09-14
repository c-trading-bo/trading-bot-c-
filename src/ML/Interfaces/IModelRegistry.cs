using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.ML.Models;

namespace TradingBot.ML.Interfaces;

/// <summary>
/// Model registry interface for managing ML models
/// </summary>
public interface IModelRegistry
{
    Task<bool> RegisterModelAsync(string modelName, string version, string modelPath, ModelMetrics metrics, CancellationToken cancellationToken = default);
    Task<string?> GetModelPathAsync(string modelName, string? version = null, CancellationToken cancellationToken = default);
    Task<ModelMetrics?> GetModelMetricsAsync(string modelName, string? version = null, CancellationToken cancellationToken = default);
    Task<List<string>> GetAvailableModelsAsync(CancellationToken cancellationToken = default);
    Task<List<string>> GetModelVersionsAsync(string modelName, CancellationToken cancellationToken = default);
    Task<bool> DeleteModelAsync(string modelName, string? version = null, CancellationToken cancellationToken = default);
}

/// <summary>
/// Feature store interface for managing training features
/// </summary>
public interface IFeatureStore
{
    Task<Dictionary<string, double[]>> GetFeaturesAsync(DateTime startDate, DateTime endDate, CancellationToken cancellationToken = default);
    Task<bool> StoreFeaturesAsync(string featureName, DateTime timestamp, double[] values, CancellationToken cancellationToken = default);
    Task<string[]> GetAvailableFeaturesAsync(CancellationToken cancellationToken = default);
    Task<Dictionary<string, double[]>> GetLatestFeaturesAsync(int count = 100, CancellationToken cancellationToken = default);
}