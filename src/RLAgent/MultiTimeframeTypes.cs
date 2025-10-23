using System;
using System.Collections.Generic;

namespace TradingBot.RLAgent;

/// <summary>
/// Multi-timeframe batch containing synchronized 5m and 1m features.
/// </summary>
public class MultiTimeframeBatch
{
    public int BatchSize { get; set; }
    public List<string> Symbols { get; set; } = new();
    public List<DateTimeOffset> Timestamps { get; set; } = new();
    
    // Feature arrays [batch_size, num_features]
    public double[,] Features5m { get; set; } = new double[0, 0];
    public double[,] Features1m { get; set; } = new double[0, 0];
    
    // Attention masks for sequence models
    public int[,] Mask5m { get; set; } = new int[0, 0];
    public int[,] Mask1m { get; set; } = new int[0, 0];
    
    // Labels for supervised learning
    public double[] Labels { get; set; } = Array.Empty<double>();
}

/// <summary>
/// Complete multi-timeframe training dataset with train/val/test splits.
/// </summary>
public class MultiTimeframeTrainingData
{
    public List<MultiTimeframeBatch> TrainBatches { get; set; } = new();
    public List<MultiTimeframeBatch> ValidationBatches { get; set; } = new();
    public List<MultiTimeframeBatch> TestBatches { get; set; } = new();
    
    public DatasetStatistics Statistics { get; set; } = new();
    public string FeatureVersionHash { get; set; } = string.Empty;
    public DateTimeOffset PreparedAt { get; set; }
}

/// <summary>
/// Statistics about the prepared dataset.
/// </summary>
public class DatasetStatistics
{
    public int TotalSamples { get; set; }
    public int TrainSamples { get; set; }
    public int ValidationSamples { get; set; }
    public int TestSamples { get; set; }
    public Dictionary<double, int> LabelDistribution { get; set; } = new();
    public (DateTimeOffset start, DateTimeOffset end) TrainDateRange { get; set; }
    public (DateTimeOffset start, DateTimeOffset end) ValidationDateRange { get; set; }
    public (DateTimeOffset start, DateTimeOffset end) TestDateRange { get; set; }
    public int NumFeatures5m { get; set; }
    public int NumFeatures1m { get; set; }
    public int TotalFeatures { get; set; }
}
