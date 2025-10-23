using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace BotCore.ML;

/// <summary>
/// Multi-timeframe data loader for training and validation.
/// Loads historical data from both 5-minute and 1-minute bars,
/// aligns timestamps, and creates synchronized training samples.
/// 
/// Phase 3: Data Loader Service (Week 2)
/// - LoadHistoricalData(): Load 5m + 1m files
/// - AlignTimestamps(): Find common timestamps across timeframes
/// - CreateSynchronizedSamples(): Build training samples with all timeframes
/// - SplitTrainValTest(): Use DynamicDataSplitStrategy to split data
/// 
/// Design principles:
/// - No data leakage: Test set is completely isolated
/// - Deterministic: Same input always produces same output
/// - Production-ready: Full error handling and validation
/// </summary>
public class MultiTimeframeDataLoader
{
    private readonly ILogger<MultiTimeframeDataLoader> _logger;
    private readonly MultiTimeframeFeatureExtractor _featureExtractor;
    private readonly string _dataDirectory;
    
    // JSON options for deserialization
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        AllowTrailingCommas = true
    };
    
    public MultiTimeframeDataLoader(
        ILogger<MultiTimeframeDataLoader> logger,
        MultiTimeframeFeatureExtractor featureExtractor,
        string dataDirectory = "data/historical")
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _featureExtractor = featureExtractor ?? throw new ArgumentNullException(nameof(featureExtractor));
        _dataDirectory = dataDirectory ?? throw new ArgumentNullException(nameof(dataDirectory));
    }
    
    /// <summary>
    /// Load historical data from JSON files for both timeframes.
    /// </summary>
    /// <param name="symbol">Symbol to load (e.g., "ES", "NQ")</param>
    /// <returns>Tuple of (5m bars, 1m bars)</returns>
    public (List<BarData> bars5m, List<BarData> bars1m) LoadHistoricalData(string symbol)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));
        }
        
        try
        {
            // Load 5-minute bars
            var file5m = Path.Combine(_dataDirectory, $"{symbol}_90days.json");
            var bars5m = LoadBarsFromFile(file5m, "5m");
            
            // Load 1-minute bars
            var file1m = Path.Combine(_dataDirectory, $"{symbol}_1m_90days.json");
            var bars1m = LoadBarsFromFile(file1m, "1m");
            
            _logger.LogInformation(
                "[DATA_LOADER] Loaded historical data for {Symbol}: {Count5m} 5m bars, {Count1m} 1m bars",
                symbol, bars5m.Count, bars1m.Count);
            
            return (bars5m, bars1m);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_LOADER] Error loading historical data for {Symbol}", symbol);
            throw;
        }
    }
    
    /// <summary>
    /// Align timestamps across timeframes to find common bar close times.
    /// For multi-timeframe synchronization, we want 5m bar close times that
    /// also have corresponding 1m bars.
    /// </summary>
    /// <param name="bars5m">5-minute bars</param>
    /// <param name="bars1m">1-minute bars</param>
    /// <returns>List of aligned timestamps (5m bar close times)</returns>
    public List<DateTimeOffset> AlignTimestamps(List<BarData> bars5m, List<BarData> bars1m)
    {
        if (bars5m == null || bars1m == null)
        {
            throw new ArgumentNullException("Bars cannot be null");
        }
        
        try
        {
            // Create index of 1m timestamps for fast lookup
            var timestamps1m = new HashSet<DateTimeOffset>(bars1m.Select(b => b.Timestamp));
            
            // Find 5m bar close times that have corresponding 1m data
            // For a 5m bar closing at time T, we need 1m bars at T-4, T-3, T-2, T-1, T
            var alignedTimestamps = new List<DateTimeOffset>();
            
            foreach (var bar5m in bars5m.OrderBy(b => b.Timestamp))
            {
                var closeTime = bar5m.Timestamp;
                
                // Check if we have all 5 corresponding 1m bars
                var has1mData = timestamps1m.Contains(closeTime) &&
                                timestamps1m.Contains(closeTime.AddMinutes(-1)) &&
                                timestamps1m.Contains(closeTime.AddMinutes(-2)) &&
                                timestamps1m.Contains(closeTime.AddMinutes(-3)) &&
                                timestamps1m.Contains(closeTime.AddMinutes(-4));
                
                if (has1mData)
                {
                    alignedTimestamps.Add(closeTime);
                }
            }
            
            _logger.LogInformation(
                "[DATA_LOADER] Aligned timestamps: {AlignedCount}/{Total5m} 5m bars have complete 1m data",
                alignedTimestamps.Count, bars5m.Count);
            
            return alignedTimestamps;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_LOADER] Error aligning timestamps");
            throw;
        }
    }
    
    /// <summary>
    /// Create synchronized training samples with features from all timeframes.
    /// Each sample represents one decision point with multi-timeframe context.
    /// </summary>
    /// <param name="symbol">Symbol name</param>
    /// <param name="bars5m">5-minute bars</param>
    /// <param name="bars1m">1-minute bars</param>
    /// <param name="alignedTimestamps">Pre-aligned timestamps</param>
    /// <returns>List of training samples with synchronized features</returns>
    public List<MultiTimeframeSample> CreateSynchronizedSamples(
        string symbol,
        List<BarData> bars5m,
        List<BarData> bars1m,
        List<DateTimeOffset> alignedTimestamps)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));
        }
        
        if (bars5m == null || bars1m == null || alignedTimestamps == null)
        {
            throw new ArgumentNullException("Input parameters cannot be null");
        }
        
        var samples = new List<MultiTimeframeSample>();
        
        try
        {
            _logger.LogInformation(
                "[DATA_LOADER] Creating synchronized samples for {Count} timestamps",
                alignedTimestamps.Count);
            
            foreach (var timestamp in alignedTimestamps)
            {
                // Extract synchronized features for this timestamp
                var features = _featureExtractor.SynchronizeFeatures(timestamp, bars5m, bars1m);
                
                if (features.Count == 0)
                {
                    _logger.LogWarning(
                        "[DATA_LOADER] No features extracted for timestamp {Timestamp}, skipping",
                        timestamp);
                    continue;
                }
                
                // Create sample
                var sample = new MultiTimeframeSample
                {
                    Timestamp = timestamp,
                    Symbol = symbol,
                    Features = features,
                    Features5m = features.Where(kvp => kvp.Key.EndsWith("_5m")).ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                    Features1m = features.Where(kvp => kvp.Key.EndsWith("_1m")).ToDictionary(kvp => kvp.Key, kvp => kvp.Value)
                };
                
                samples.Add(sample);
            }
            
            _logger.LogInformation(
                "[DATA_LOADER] Created {SampleCount} synchronized samples",
                samples.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_LOADER] Error creating synchronized samples");
            throw;
        }
        
        return samples;
    }
    
    /// <summary>
    /// Split synchronized samples into train/validation/test sets.
    /// Ensures no data leakage - test set is completely isolated chronologically.
    /// </summary>
    /// <param name="samples">All synchronized samples</param>
    /// <param name="trainRatio">Training set ratio (default 0.67)</param>
    /// <param name="valRatio">Validation set ratio (default 0.17)</param>
    /// <returns>Tuple of (train, validation, test) samples</returns>
    public (List<MultiTimeframeSample> train, List<MultiTimeframeSample> val, List<MultiTimeframeSample> test) 
        SplitTrainValTest(
            List<MultiTimeframeSample> samples,
            double trainRatio = 0.67,
            double valRatio = 0.17)
    {
        if (samples == null || samples.Count == 0)
        {
            throw new ArgumentException("Samples cannot be null or empty", nameof(samples));
        }
        
        if (trainRatio <= 0 || valRatio <= 0 || trainRatio + valRatio >= 1.0)
        {
            throw new ArgumentException("Invalid split ratios");
        }
        
        try
        {
            // Sort samples chronologically (oldest to newest)
            var sortedSamples = samples.OrderBy(s => s.Timestamp).ToList();
            
            // Calculate split indices
            var totalCount = sortedSamples.Count;
            var trainCount = (int)(totalCount * trainRatio);
            var valCount = (int)(totalCount * valRatio);
            var testCount = totalCount - trainCount - valCount;
            
            // Split chronologically: oldest → train, middle → val, newest → test
            var trainSamples = sortedSamples.Take(trainCount).ToList();
            var valSamples = sortedSamples.Skip(trainCount).Take(valCount).ToList();
            var testSamples = sortedSamples.Skip(trainCount + valCount).ToList();
            
            _logger.LogInformation(
                "[DATA_LOADER] Split data: {Train} train / {Val} val / {Test} test " +
                "(ratios: {TrainRatio:P0}/{ValRatio:P0}/{TestRatio:P0})",
                trainSamples.Count, valSamples.Count, testSamples.Count,
                trainRatio, valRatio, 1.0 - trainRatio - valRatio);
            
            // Log date ranges to verify chronological split
            if (trainSamples.Count > 0 && testSamples.Count > 0)
            {
                _logger.LogInformation(
                    "[DATA_LOADER] Date ranges: Train [{TrainStart} to {TrainEnd}], " +
                    "Val [{ValStart} to {ValEnd}], Test [{TestStart} to {TestEnd}]",
                    trainSamples.First().Timestamp, trainSamples.Last().Timestamp,
                    valSamples.Count > 0 ? valSamples.First().Timestamp : DateTimeOffset.MinValue,
                    valSamples.Count > 0 ? valSamples.Last().Timestamp : DateTimeOffset.MinValue,
                    testSamples.First().Timestamp, testSamples.Last().Timestamp);
            }
            
            return (trainSamples, valSamples, testSamples);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_LOADER] Error splitting data");
            throw;
        }
    }
    
    /// <summary>
    /// Get training batches from samples for mini-batch training.
    /// </summary>
    /// <param name="samples">Training samples</param>
    /// <param name="batchSize">Batch size</param>
    /// <returns>Batched samples</returns>
    public List<List<MultiTimeframeSample>> GetTrainingBatches(
        List<MultiTimeframeSample> samples,
        int batchSize = 32)
    {
        if (samples == null || samples.Count == 0)
        {
            throw new ArgumentException("Samples cannot be null or empty", nameof(samples));
        }
        
        if (batchSize <= 0)
        {
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));
        }
        
        var batches = new List<List<MultiTimeframeSample>>();
        
        for (int i = 0; i < samples.Count; i += batchSize)
        {
            var batch = samples.Skip(i).Take(batchSize).ToList();
            batches.Add(batch);
        }
        
        _logger.LogDebug(
            "[DATA_LOADER] Created {BatchCount} batches of size {BatchSize} from {SampleCount} samples",
            batches.Count, batchSize, samples.Count);
        
        return batches;
    }
    
    #region Private Helper Methods
    
    /// <summary>
    /// Load bars from JSON file.
    /// </summary>
    private List<BarData> LoadBarsFromFile(string filePath, string timeframe)
    {
        if (!File.Exists(filePath))
        {
            _logger.LogWarning("[DATA_LOADER] File not found: {FilePath}", filePath);
            return new List<BarData>();
        }
        
        try
        {
            var jsonContent = File.ReadAllText(filePath);
            var dataWrapper = JsonSerializer.Deserialize<HistoricalDataWrapper>(jsonContent, JsonOptions);
            
            if (dataWrapper?.Bars == null || dataWrapper.Bars.Count == 0)
            {
                _logger.LogWarning("[DATA_LOADER] No bars found in {FilePath}", filePath);
                return new List<BarData>();
            }
            
            // Convert to BarData objects
            var bars = dataWrapper.Bars.Select(b => new BarData
            {
                Timestamp = ParseTimestamp(b.Timestamp),
                Open = b.Open,
                High = b.High,
                Low = b.Low,
                Close = b.Close,
                Volume = b.Volume
            }).ToList();
            
            _logger.LogDebug(
                "[DATA_LOADER] Loaded {Count} {Timeframe} bars from {FilePath}",
                bars.Count, timeframe, filePath);
            
            return bars;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_LOADER] Error loading bars from {FilePath}", filePath);
            throw;
        }
    }
    
    /// <summary>
    /// Parse timestamp string to DateTimeOffset.
    /// Handles timezone information from data files.
    /// </summary>
    private static DateTimeOffset ParseTimestamp(string timestampStr)
    {
        if (string.IsNullOrWhiteSpace(timestampStr))
        {
            throw new ArgumentException("Timestamp cannot be null or empty");
        }
        
        // Try parsing with timezone
        if (DateTimeOffset.TryParse(timestampStr, out var result))
        {
            return result;
        }
        
        // Fallback: parse without timezone and assume UTC
        if (DateTime.TryParse(timestampStr, out var dt))
        {
            return new DateTimeOffset(dt, TimeSpan.Zero);
        }
        
        throw new FormatException($"Unable to parse timestamp: {timestampStr}");
    }
    
    #endregion
}

/// <summary>
/// Wrapper for historical data JSON format.
/// </summary>
internal class HistoricalDataWrapper
{
    public string Symbol { get; set; } = string.Empty;
    public string Timeframe { get; set; } = string.Empty;
    public List<BarDataJson> Bars { get; set; } = new();
}

/// <summary>
/// Bar data from JSON file.
/// </summary>
internal class BarDataJson
{
    public string Timestamp { get; set; } = string.Empty;
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double Volume { get; set; }
}

/// <summary>
/// Multi-timeframe training sample with synchronized features.
/// </summary>
public class MultiTimeframeSample
{
    public DateTimeOffset Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public Dictionary<string, double> Features { get; set; } = new();
    public Dictionary<string, double> Features5m { get; set; } = new();
    public Dictionary<string, double> Features1m { get; set; } = new();
}
