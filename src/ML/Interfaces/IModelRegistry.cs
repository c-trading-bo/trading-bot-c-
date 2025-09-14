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
    bool RegisterModel(string modelName, string version, string modelPath, ModelMetrics metrics);
    string? GetModelPath(string modelName, string? version = null);
    ModelMetrics? GetModelMetrics(string modelName, string? version = null);
    List<string> GetAvailableModels();
    List<string> GetModelVersions(string modelName);
    bool DeleteModel(string modelName, string? version = null);
}

/// <summary>
/// Feature store interface for managing training features
/// </summary>
public interface IFeatureStore
{
    Dictionary<string, double[]> GetFeatures(DateTime startDate, DateTime endDate);
    bool StoreFeatures(string featureName, DateTime timestamp, double[] values);
    string[] GetAvailableFeatures();
    Dictionary<string, double[]> GetLatestFeatures(int count = 100);
}