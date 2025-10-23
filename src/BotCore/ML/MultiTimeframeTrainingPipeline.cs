using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.ML;

/// <summary>
/// Demonstration pipeline showing how to use all multi-timeframe components together.
/// This is an example integration that prepares data for multi-branch model training.
/// 
/// FUTURE WORK: Full integration into production training requires:
/// 1. Multi-branch neural network architecture (separate branches for 5m and 1m features)
/// 2. Integration with CVaRPPOTrainer, LSTMTrainer, and other model trainers
/// 3. Model serialization/deserialization for multi-timeframe models
/// 4. Production deployment and inference pipeline updates
/// 
/// This class demonstrates the complete data preparation pipeline:
/// - Load historical data from both timeframes (MultiTimeframeDataLoader)
/// - Assemble synchronized samples (MultiTimeframeDataAssembler)
/// - Create batches for training (MultiTimeframeBatchCreator)
/// - Split into train/validation/test sets
/// 
/// Usage example:
/// <code>
/// var pipeline = serviceProvider.GetRequiredService&lt;MultiTimeframeTrainingPipeline&gt;();
/// var result = await pipeline.PrepareTrainingDataAsync("ES", trainRatio: 0.7);
/// 
/// // result.TrainBatches contains batches ready for model training
/// // result.ValBatches contains validation batches
/// // result.TestBatches contains test batches
/// </code>
/// </summary>
public class MultiTimeframeTrainingPipeline
{
    private readonly ILogger<MultiTimeframeTrainingPipeline> _logger;
    private readonly MultiTimeframeDataLoader _dataLoader;
    private readonly MultiTimeframeDataAssembler _dataAssembler;
    private readonly MultiTimeframeBatchCreator _batchCreator;
    private readonly MultiTimeframeFeatureExtractor _featureExtractor;
    
    public MultiTimeframeTrainingPipeline(
        ILogger<MultiTimeframeTrainingPipeline> logger,
        MultiTimeframeDataLoader dataLoader,
        MultiTimeframeDataAssembler dataAssembler,
        MultiTimeframeBatchCreator batchCreator,
        MultiTimeframeFeatureExtractor featureExtractor)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _dataLoader = dataLoader ?? throw new ArgumentNullException(nameof(dataLoader));
        _dataAssembler = dataAssembler ?? throw new ArgumentNullException(nameof(dataAssembler));
        _batchCreator = batchCreator ?? throw new ArgumentNullException(nameof(batchCreator));
        _featureExtractor = featureExtractor ?? throw new ArgumentNullException(nameof(featureExtractor));
    }
    
    /// <summary>
    /// Prepare complete training dataset with train/validation/test splits.
    /// This demonstrates the full pipeline from loading data to creating batches.
    /// </summary>
    /// <param name="symbol">Symbol to train on (e.g., "ES", "NQ")</param>
    /// <param name="trainRatio">Ratio of data for training (default 0.67)</param>
    /// <param name="valRatio">Ratio of data for validation (default 0.17)</param>
    /// <param name="batchSize">Batch size for training (default 32)</param>
    /// <param name="shuffle">Whether to shuffle training batches</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Prepared training data with batches</returns>
    public Task<MultiTimeframeTrainingData> PrepareTrainingDataAsync(
        string symbol,
        double trainRatio = 0.67,
        double valRatio = 0.17,
        int batchSize = 32,
        bool shuffle = true,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation(
                "[MTF_PIPELINE] Starting multi-timeframe training data preparation for {Symbol}",
                symbol);
            
            // Step 1: Load historical data from both timeframes
            _logger.LogInformation("[MTF_PIPELINE] Step 1: Loading historical data...");
            var (bars5m, bars1m) = _dataLoader.LoadHistoricalData(symbol);
            
            if (bars5m.Count == 0 || bars1m.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Insufficient historical data for {symbol}: 5m={bars5m.Count}, 1m={bars1m.Count}");
            }
            
            _logger.LogInformation(
                "[MTF_PIPELINE] Loaded {Count5m} 5m bars and {Count1m} 1m bars",
                bars5m.Count, bars1m.Count);
            
            // Step 2: Assemble synchronized samples
            _logger.LogInformation("[MTF_PIPELINE] Step 2: Assembling synchronized samples...");
            var samples = _dataAssembler.AssembleSamples(symbol, bars5m, bars1m);
            
            if (samples.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Failed to assemble any synchronized samples for {symbol}");
            }
            
            _logger.LogInformation(
                "[MTF_PIPELINE] Assembled {SampleCount} synchronized samples",
                samples.Count);
            
            // Step 3: Split into train/validation/test sets
            _logger.LogInformation("[MTF_PIPELINE] Step 3: Splitting data...");
            var (trainSamples, valSamples, testSamples) = SplitSamples(
                samples, trainRatio, valRatio);
            
            _logger.LogInformation(
                "[MTF_PIPELINE] Split complete: {Train} train, {Val} val, {Test} test",
                trainSamples.Count, valSamples.Count, testSamples.Count);
            
            // Step 4: Create batches for each set
            _logger.LogInformation("[MTF_PIPELINE] Step 4: Creating batches...");
            
            var trainBatches = _batchCreator.CreateBatches(trainSamples, batchSize, shuffle);
            var valBatches = _batchCreator.CreateBatches(valSamples, batchSize, shuffle: false);
            var testBatches = _batchCreator.CreateBatches(testSamples, batchSize, shuffle: false);
            
            _logger.LogInformation(
                "[MTF_PIPELINE] Created {TrainBatches} train batches, {ValBatches} val batches, {TestBatches} test batches",
                trainBatches.Count, valBatches.Count, testBatches.Count);
            
            // Step 5: Compute statistics
            var stats = ComputeDatasetStatistics(trainSamples, valSamples, testSamples);
            
            _logger.LogInformation(
                "[MTF_PIPELINE] Dataset statistics: {Stats}",
                System.Text.Json.JsonSerializer.Serialize(stats));
            
            // Step 6: Return complete training data
            return Task.FromResult(new MultiTimeframeTrainingData
            {
                Symbol = symbol,
                TrainBatches = trainBatches,
                ValidationBatches = valBatches,
                TestBatches = testBatches,
                TrainSamples = trainSamples,
                ValidationSamples = valSamples,
                TestSamples = testSamples,
                Statistics = stats,
                FeatureVersionHash = _featureExtractor.GetFeatureVersionHash(),
                PreparedAt = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MTF_PIPELINE] Failed to prepare training data for {Symbol}", symbol);
            throw;
        }
    }
    
    /// <summary>
    /// Split samples into train/validation/test sets chronologically.
    /// Uses chronological split to prevent data leakage.
    /// </summary>
    private (List<EnhancedMultiTimeframeSample> train, 
             List<EnhancedMultiTimeframeSample> val,
             List<EnhancedMultiTimeframeSample> test) SplitSamples(
        List<EnhancedMultiTimeframeSample> samples,
        double trainRatio,
        double valRatio)
    {
        if (trainRatio <= 0 || valRatio <= 0 || trainRatio + valRatio >= 1.0)
        {
            throw new ArgumentException("Invalid split ratios");
        }
        
        // Sort chronologically (oldest to newest)
        var sortedSamples = samples.OrderBy(s => s.Timestamp).ToList();
        
        // Calculate split indices
        var totalCount = sortedSamples.Count;
        var trainCount = (int)(totalCount * trainRatio);
        var valCount = (int)(totalCount * valRatio);
        
        // Split: oldest → train, middle → val, newest → test
        var trainSamples = sortedSamples.Take(trainCount).ToList();
        var valSamples = sortedSamples.Skip(trainCount).Take(valCount).ToList();
        var testSamples = sortedSamples.Skip(trainCount + valCount).ToList();
        
        return (trainSamples, valSamples, testSamples);
    }
    
    /// <summary>
    /// Compute statistics about the dataset for monitoring and validation.
    /// </summary>
    private DatasetStatistics ComputeDatasetStatistics(
        List<EnhancedMultiTimeframeSample> trainSamples,
        List<EnhancedMultiTimeframeSample> valSamples,
        List<EnhancedMultiTimeframeSample> testSamples)
    {
        var allSamples = trainSamples.Concat(valSamples).Concat(testSamples).ToList();
        
        // Label distribution (for classification tasks)
        var labelCounts = allSamples
            .GroupBy(s => s.Label)
            .ToDictionary(g => g.Key, g => g.Count());
        
        // Date ranges
        var trainStart = trainSamples.Any() ? trainSamples.Min(s => s.Timestamp) : DateTimeOffset.MinValue;
        var trainEnd = trainSamples.Any() ? trainSamples.Max(s => s.Timestamp) : DateTimeOffset.MinValue;
        var valStart = valSamples.Any() ? valSamples.Min(s => s.Timestamp) : DateTimeOffset.MinValue;
        var valEnd = valSamples.Any() ? valSamples.Max(s => s.Timestamp) : DateTimeOffset.MinValue;
        var testStart = testSamples.Any() ? testSamples.Min(s => s.Timestamp) : DateTimeOffset.MinValue;
        var testEnd = testSamples.Any() ? testSamples.Max(s => s.Timestamp) : DateTimeOffset.MinValue;
        
        // Feature counts
        var sample = allSamples.FirstOrDefault();
        var numFeatures5m = sample?.Features5m?.Count ?? 0;
        var numFeatures1m = sample?.Features1m?.Count ?? 0;
        
        return new DatasetStatistics
        {
            TotalSamples = allSamples.Count,
            TrainSamples = trainSamples.Count,
            ValidationSamples = valSamples.Count,
            TestSamples = testSamples.Count,
            LabelDistribution = labelCounts,
            TrainDateRange = (trainStart, trainEnd),
            ValidationDateRange = (valStart, valEnd),
            TestDateRange = (testStart, testEnd),
            NumFeatures5m = numFeatures5m,
            NumFeatures1m = numFeatures1m,
            TotalFeatures = numFeatures5m + numFeatures1m
        };
    }
}

/// <summary>
/// Complete training data prepared by the pipeline.
/// Contains batches and samples for train/validation/test sets.
/// </summary>
public class MultiTimeframeTrainingData
{
    /// <summary>Symbol this data is for</summary>
    public string Symbol { get; set; } = string.Empty;
    
    /// <summary>Training batches ready for model training</summary>
    public List<MultiTimeframeBatch> TrainBatches { get; set; } = new();
    
    /// <summary>Validation batches for hyperparameter tuning</summary>
    public List<MultiTimeframeBatch> ValidationBatches { get; set; } = new();
    
    /// <summary>Test batches for final evaluation</summary>
    public List<MultiTimeframeBatch> TestBatches { get; set; } = new();
    
    /// <summary>Raw training samples (for custom batching)</summary>
    public List<EnhancedMultiTimeframeSample> TrainSamples { get; set; } = new();
    
    /// <summary>Raw validation samples</summary>
    public List<EnhancedMultiTimeframeSample> ValidationSamples { get; set; } = new();
    
    /// <summary>Raw test samples</summary>
    public List<EnhancedMultiTimeframeSample> TestSamples { get; set; } = new();
    
    /// <summary>Dataset statistics</summary>
    public DatasetStatistics Statistics { get; set; } = new();
    
    /// <summary>Feature version hash for reproducibility</summary>
    public string FeatureVersionHash { get; set; } = string.Empty;
    
    /// <summary>When this data was prepared</summary>
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
