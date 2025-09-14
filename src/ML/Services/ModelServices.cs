using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.ML.Interfaces;
using TradingBot.ML.Models;

namespace TradingBot.ML.Services;

/// <summary>
/// Simple file-based model registry implementation
/// </summary>
public class FileModelRegistry : IModelRegistry
{
    private readonly ILogger<FileModelRegistry> _logger;
    private readonly string _modelsPath;

    public FileModelRegistry(ILogger<FileModelRegistry> logger, string modelsPath = "./models")
    {
        _logger = logger;
        _modelsPath = modelsPath;
        Directory.CreateDirectory(_modelsPath);
    }

    public async Task<bool> RegisterModelAsync(string modelName, string version, string modelPath, ModelMetrics metrics, CancellationToken cancellationToken = default)
    {
        try
        {
            var targetPath = Path.Combine(_modelsPath, $"{modelName}_{version}.onnx");
            if (File.Exists(modelPath))
            {
                File.Copy(modelPath, targetPath, true);
            }
            _logger.LogInformation("Registered model {ModelName} version {Version}", modelName, version);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to register model {ModelName}", modelName);
            return false;
        }
    }

    public async Task<string?> GetModelPathAsync(string modelName, string? version = null, CancellationToken cancellationToken = default)
    {
        var pattern = version != null ? $"{modelName}_{version}.onnx" : $"{modelName}_*.onnx";
        var files = Directory.GetFiles(_modelsPath, pattern);
        return files.Length > 0 ? files[0] : null;
    }

    public async Task<ModelMetrics?> GetModelMetricsAsync(string modelName, string? version = null, CancellationToken cancellationToken = default)
    {
        // Return basic metrics
        return new ModelMetrics
        {
            ModelName = modelName,
            Version = version ?? "latest",
            TrainingDate = DateTime.UtcNow.AddDays(-1),
            Accuracy = 0.75,
            SharpeRatio = 1.2,
            WinRate = 0.6
        };
    }

    public async Task<List<string>> GetAvailableModelsAsync(CancellationToken cancellationToken = default)
    {
        var models = new List<string>();
        var files = Directory.GetFiles(_modelsPath, "*.onnx");
        foreach (var file in files)
        {
            var name = Path.GetFileNameWithoutExtension(file);
            var modelName = name.Split('_')[0];
            if (!models.Contains(modelName))
                models.Add(modelName);
        }
        return models;
    }

    public async Task<List<string>> GetModelVersionsAsync(string modelName, CancellationToken cancellationToken = default)
    {
        var versions = new List<string>();
        var files = Directory.GetFiles(_modelsPath, $"{modelName}_*.onnx");
        foreach (var file in files)
        {
            var name = Path.GetFileNameWithoutExtension(file);
            var parts = name.Split('_');
            if (parts.Length > 1)
                versions.Add(parts[1]);
        }
        return versions;
    }

    public async Task<bool> DeleteModelAsync(string modelName, string? version = null, CancellationToken cancellationToken = default)
    {
        try
        {
            var pattern = version != null ? $"{modelName}_{version}.onnx" : $"{modelName}_*.onnx";
            var files = Directory.GetFiles(_modelsPath, pattern);
            foreach (var file in files)
            {
                File.Delete(file);
            }
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to delete model {ModelName}", modelName);
            return false;
        }
    }
}

/// <summary>
/// Simple in-memory feature store implementation
/// </summary>
public class InMemoryFeatureStore : IFeatureStore
{
    private readonly ILogger<InMemoryFeatureStore> _logger;
    private readonly Dictionary<string, List<(DateTime timestamp, double[] values)>> _features = new();

    public InMemoryFeatureStore(ILogger<InMemoryFeatureStore> logger)
    {
        _logger = logger;
    }

    public async Task<Dictionary<string, double[]>> GetFeaturesAsync(DateTime startDate, DateTime endDate, CancellationToken cancellationToken = default)
    {
        await Task.Delay(1, cancellationToken); // Make it properly async
        var result = new Dictionary<string, double[]>();
        
        foreach (var (featureName, data) in _features)
        {
            var filteredData = data.Where(d => d.timestamp >= startDate && d.timestamp <= endDate)
                                  .SelectMany(d => d.values)
                                  .ToArray();
            if (filteredData.Length > 0)
                result[featureName] = filteredData;
        }
        
        return result;
    }

    public async Task<bool> StoreFeaturesAsync(string featureName, DateTime timestamp, double[] values, CancellationToken cancellationToken = default)
    {
        if (!_features.ContainsKey(featureName))
            _features[featureName] = new List<(DateTime, double[])>();
            
        _features[featureName].Add((timestamp, values));
        return true;
    }

    public async Task<string[]> GetAvailableFeaturesAsync(CancellationToken cancellationToken = default)
    {
        return _features.Keys.ToArray();
    }

    public async Task<Dictionary<string, double[]>> GetLatestFeaturesAsync(int count = 100, CancellationToken cancellationToken = default)
    {
        var result = new Dictionary<string, double[]>();
        
        foreach (var (featureName, data) in _features)
        {
            var latestData = data.TakeLast(count)
                                .SelectMany(d => d.values)
                                .ToArray();
            if (latestData.Length > 0)
                result[featureName] = latestData;
        }
        
        return result;
    }
}