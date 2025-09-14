using System;
using System.Collections.Generic;
using MLModelMetrics = TradingBot.ML.Models.ModelMetrics;

namespace TradingBot.ML.HistoricalTrainer.Models;

/// <summary>
/// Configuration for historical training process
/// </summary>
public class HistoricalTrainingConfig
{
    public List<string> Symbols { get; set; } = new() { "ES", "NQ" };
    public DateTime StartDate { get; set; } = DateTime.UtcNow.AddYears(-2);
    public DateTime EndDate { get; set; } = DateTime.UtcNow.AddDays(-1);
    public int LookbackPeriod { get; set; } = 50;
    public int ForwardLookPeriod { get; set; } = 10;
    public int SampleStride { get; set; } = 5;
    public int MinimumBarsPerSymbol { get; set; } = 1000;
    public double MinimumReturnThreshold { get; set; } = 0.002; // 0.2% minimum return
    public int WalkForwardFolds { get; set; } = 10;
    public int MinimumTrainingSamples { get; set; } = 5000;
    public Dictionary<string, int> ModelParams { get; set; } = new()
    {
        ["NumberOfLeaves"] = 20,
        ["MinExampleCountPerLeaf"] = 10,
        ["NumberOfTrees"] = 100
    };
}

/// <summary>
/// Result of historical training process
/// </summary>
public class HistoricalTrainingResult
{
    public bool Success { get; set; }
    public string? ModelId { get; set; }
    public string? ModelPath { get; set; }
    public MLModelMetrics? TrainingMetrics { get; set; }
    public WalkForwardResults? WalkForwardResults { get; set; }
    public DatasetInfo? DatasetInfo { get; set; }
    public string? ErrorMessage { get; set; }
    public DateTime CompletedAt { get; set; }
}

/// <summary>
/// Dataset information for historical training
/// </summary>
public class DatasetInfo
{
    public int SampleCount { get; set; }
    public int FeatureCount { get; set; }
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public List<string> Symbols { get; set; } = new();
}

/// <summary>
/// Training dataset for historical training
/// </summary>
public class TrainingDataset
{
    public List<TrainingSample> Samples { get; set; } = new();
    public List<string> FeatureNames { get; set; } = new();
    public int FeatureCount { get; set; }
    public DateTime CreatedAt { get; set; }
}

/// <summary>
/// Individual training sample for historical training
/// </summary>
public class TrainingSample
{
    public Dictionary<string, double> Features { get; set; } = new();
    public double Label { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Results from walk-forward validation
/// </summary>
public class WalkForwardResults
{
    public List<TrainedModelResult> FoldResults { get; set; } = new();
    public double AverageAuc { get; set; }
    public int CompletedFolds { get; set; }
    public int TotalFolds { get; set; }
}

/// <summary>
/// Result of training a single model
/// </summary>
public class TrainedModelResult
{
    public byte[] ModelData { get; set; } = Array.Empty<byte>();
    public string ModelPath { get; set; } = string.Empty;
    public MLModelMetrics Metrics { get; set; } = new();
    public int TrainingSampleCount { get; set; }
    public int TestSampleCount { get; set; }
    public int FoldNumber { get; set; }
    public DateTime TrainedAt { get; set; }
}

/// <summary>
/// Deployed model information
/// </summary>
public class DeployedModel
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public string MetadataPath { get; set; } = string.Empty;
    public MLModelMetrics Metrics { get; set; } = new();
    public DateTime DeployedAt { get; set; }
}

/// <summary>
/// Model metadata for deployment
/// </summary>
public class ModelMetadata
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public List<string> FeatureNames { get; set; } = new();
    public Dictionary<string, object> Hyperparameters { get; set; } = new();
    public MLModelMetrics Metrics { get; set; } = new();
    public DateTime CreatedAt { get; set; }
    public DateTime DeployedAt { get; set; }
    public string Status { get; set; } = "Active";
    public HistoricalTrainingConfig TrainingConfig { get; set; } = new();
    public int TrainingSampleCount { get; set; }
    public int TestSampleCount { get; set; }
    public int FoldNumber { get; set; }
    public string ModelType { get; set; } = string.Empty;
}