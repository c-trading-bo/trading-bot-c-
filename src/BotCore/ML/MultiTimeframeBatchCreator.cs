using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.ML;

/// <summary>
/// Multi-timeframe batch creator for efficient GPU training.
/// Groups synchronized multi-timeframe samples into batches for parallel processing.
/// 
/// Purpose: Create batches that can be processed in parallel on GPU:
/// - Batch multiple samples together
/// - Pad sequences to uniform length
/// - Create attention masks for variable-length sequences
/// - Prepare data in format suitable for neural network training
/// 
/// Design principles:
/// - Efficient GPU utilization: Batch processing for parallelism
/// - Padded sequences: Handle variable-length contexts
/// - Masked attention: Ignore padding during model forward pass
/// - Deterministic: Same samples always produce same batches
/// </summary>
public class MultiTimeframeBatchCreator
{
    private readonly ILogger<MultiTimeframeBatchCreator> _logger;
    
    // Default batch size (can be overridden)
    private const int DefaultBatchSize = 32;
    
    // Maximum sequence lengths (for padding)
    private const int Max5mBars = 36;
    private const int Max1mBars = 60;
    
    public MultiTimeframeBatchCreator(ILogger<MultiTimeframeBatchCreator> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }
    
    /// <summary>
    /// Create batches from list of synchronized samples.
    /// </summary>
    /// <param name="samples">List of synchronized multi-timeframe samples</param>
    /// <param name="batchSize">Number of samples per batch</param>
    /// <param name="shuffle">Whether to shuffle samples before batching</param>
    /// <returns>List of batches ready for training</returns>
    public List<MultiTimeframeBatch> CreateBatches(
        List<EnhancedMultiTimeframeSample> samples,
        int batchSize = DefaultBatchSize,
        bool shuffle = false)
    {
        if (samples == null || samples.Count == 0)
        {
            throw new ArgumentException("Samples cannot be null or empty", nameof(samples));
        }
        
        if (batchSize <= 0)
        {
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));
        }
        
        var batches = new List<MultiTimeframeBatch>();
        
        try
        {
            // Optionally shuffle samples for training
            var processedSamples = shuffle ? ShuffleSamples(samples) : samples.ToList();
            
            _logger.LogInformation(
                "[BATCH_CREATOR] Creating batches from {SampleCount} samples with batch size {BatchSize}",
                processedSamples.Count, batchSize);
            
            // Create batches
            for (int i = 0; i < processedSamples.Count; i += batchSize)
            {
                var batchSamples = processedSamples.Skip(i).Take(batchSize).ToList();
                var batch = CreateBatch(batchSamples);
                batches.Add(batch);
            }
            
            _logger.LogInformation(
                "[BATCH_CREATOR] Created {BatchCount} batches (avg size: {AvgSize:F1})",
                batches.Count, processedSamples.Count / (double)batches.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BATCH_CREATOR] Error creating batches");
            throw;
        }
        
        return batches;
    }
    
    /// <summary>
    /// Create a single batch from list of samples.
    /// Handles padding and mask creation.
    /// </summary>
    private MultiTimeframeBatch CreateBatch(List<EnhancedMultiTimeframeSample> samples)
    {
        var batch = new MultiTimeframeBatch
        {
            BatchSize = samples.Count,
            Symbols = samples.Select(s => s.Symbol).ToList(),
            Timestamps = samples.Select(s => s.Timestamp).ToList()
        };
        
        // Extract feature arrays from each sample
        var features5mList = new List<Dictionary<string, double>>();
        var features1mList = new List<Dictionary<string, double>>();
        var labelsList = new List<double>();
        
        foreach (var sample in samples)
        {
            features5mList.Add(sample.Features5m);
            features1mList.Add(sample.Features1m);
            labelsList.Add(sample.Label);
        }
        
        // Convert feature dictionaries to arrays
        batch.Features5m = ConvertFeaturesToArray(features5mList);
        batch.Features1m = ConvertFeaturesToArray(features1mList);
        batch.Labels = labelsList.ToArray();
        
        // Create masks (all 1s for now, since we're using fixed-size feature vectors)
        batch.Mask5m = CreateMask(samples.Count, batch.Features5m.GetLength(1));
        batch.Mask1m = CreateMask(samples.Count, batch.Features1m.GetLength(1));
        
        return batch;
    }
    
    /// <summary>
    /// Convert list of feature dictionaries to 2D array.
    /// Shape: [batch_size, num_features]
    /// </summary>
    private double[,] ConvertFeaturesToArray(List<Dictionary<string, double>> featureList)
    {
        if (featureList.Count == 0)
        {
            return new double[0, 0];
        }
        
        // Get all unique feature names across all samples
        var allFeatureNames = featureList
            .SelectMany(f => f.Keys)
            .Distinct()
            .OrderBy(k => k)
            .ToList();
        
        var batchSize = featureList.Count;
        var numFeatures = allFeatureNames.Count;
        var featureArray = new double[batchSize, numFeatures];
        
        // Fill array with feature values
        for (int i = 0; i < batchSize; i++)
        {
            var features = featureList[i];
            for (int j = 0; j < numFeatures; j++)
            {
                var featureName = allFeatureNames[j];
                featureArray[i, j] = features.ContainsKey(featureName) ? features[featureName] : 0.0;
            }
        }
        
        return featureArray;
    }
    
    /// <summary>
    /// Create attention mask (all 1s for valid positions, 0s for padding).
    /// For now, all positions are valid since we use fixed-size feature vectors.
    /// </summary>
    private int[,] CreateMask(int batchSize, int sequenceLength)
    {
        var mask = new int[batchSize, sequenceLength];
        
        // All 1s - no padding needed for fixed-size feature vectors
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < sequenceLength; j++)
            {
                mask[i, j] = 1;
            }
        }
        
        return mask;
    }
    
    /// <summary>
    /// Shuffle samples using Fisher-Yates algorithm.
    /// </summary>
    private List<EnhancedMultiTimeframeSample> ShuffleSamples(List<EnhancedMultiTimeframeSample> samples)
    {
        var shuffled = samples.ToList();
        var random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = shuffled.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            var temp = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = temp;
        }
        
        return shuffled;
    }
}

/// <summary>
/// Represents a batch of multi-timeframe samples ready for training.
/// All arrays are aligned - index [i] across all arrays refers to same sample.
/// </summary>
public class MultiTimeframeBatch
{
    /// <summary>Number of samples in this batch</summary>
    public int BatchSize { get; set; }
    
    /// <summary>Symbol for each sample in batch</summary>
    public List<string> Symbols { get; set; } = new();
    
    /// <summary>Timestamp for each sample in batch</summary>
    public List<DateTimeOffset> Timestamps { get; set; } = new();
    
    /// <summary>
    /// 5-minute features for batch
    /// Shape: [batch_size, num_features_5m]
    /// </summary>
    public double[,] Features5m { get; set; } = new double[0, 0];
    
    /// <summary>
    /// 1-minute features for batch
    /// Shape: [batch_size, num_features_1m]
    /// </summary>
    public double[,] Features1m { get; set; } = new double[0, 0];
    
    /// <summary>
    /// Attention mask for 5-minute features
    /// Shape: [batch_size, num_features_5m]
    /// 1 = valid, 0 = padding
    /// </summary>
    public int[,] Mask5m { get; set; } = new int[0, 0];
    
    /// <summary>
    /// Attention mask for 1-minute features
    /// Shape: [batch_size, num_features_1m]
    /// 1 = valid, 0 = padding
    /// </summary>
    public int[,] Mask1m { get; set; } = new int[0, 0];
    
    /// <summary>
    /// Labels for supervised learning
    /// Shape: [batch_size]
    /// Values: 1.0 (up), -1.0 (down), 0.0 (flat)
    /// </summary>
    public double[] Labels { get; set; } = Array.Empty<double>();
}
