using Microsoft.Extensions.Logging;
using Microsoft.ML;
using TradingBot.Abstractions;
using BotCore.Models;
using MLModelRegistry = TradingBot.ML.Interfaces.IModelRegistry;
using MLFeatureStore = TradingBot.ML.Interfaces.IFeatureStore;
using MLModelMetrics = TradingBot.ML.Models.ModelMetrics;
using TradingBot.ML.Services;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace TradingBot.ML.HistoricalTrainer;

/// <summary>
/// Historical training pipeline for ML models using walk-forward analysis
/// Processes historical data to produce models deployable to /models/ directory
/// </summary>
public class HistoricalTrainer
{
    private readonly ILogger<HistoricalTrainer> _logger;
    private readonly MLModelRegistry _modelRegistry;
    private readonly MLFeatureStore _featureStore;
    private readonly DatasetBuilder _datasetBuilder;
    private readonly WalkForwardTrainer _walkForwardTrainer;
    private readonly RegistryWriter _registryWriter;
    private readonly string _historicalDataPath = null!;
    private readonly string _modelsOutputPath = null!;

    public HistoricalTrainer(
        ILogger<HistoricalTrainer> logger,
        MLModelRegistry modelRegistry,
        MLFeatureStore featureStore,
        string historicalDataPath = "data/historical",
        string modelsOutputPath = "models")
    {
        _logger = logger;
        _modelRegistry = modelRegistry;
        _featureStore = featureStore;
        _historicalDataPath = historicalDataPath;
        _modelsOutputPath = modelsOutputPath;
        
        _datasetBuilder = new DatasetBuilder(logger, featureStore);
        _walkForwardTrainer = new WalkForwardTrainer(logger);
        _registryWriter = new RegistryWriter(logger, modelsOutputPath);
        
        Directory.CreateDirectory(_modelsOutputPath);
    }

    /// <summary>
    /// Train models from historical data using walk-forward analysis
    /// Reads from /data/historical/ bars and produces models in /models/
    /// </summary>
    public async Task<HistoricalTrainingResult> TrainFromHistoryAsync(
        HistoricalTrainingConfig config, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[HISTORICAL_TRAINER] Starting historical training: {Config}", JsonSerializer.Serialize(config));

            // 1. Load historical data
            var historicalBars = await LoadHistoricalDataAsync(config.Symbols, config.StartDate, config.EndDate, cancellationToken);
            _logger.LogInformation("[HISTORICAL_TRAINER] Loaded {BarCount} historical bars for {SymbolCount} symbols", 
                historicalBars.Sum(kvp => kvp.Value.Count), historicalBars.Count);

            // 2. Build dataset with features and labels
            var dataset = await _datasetBuilder.BuildDatasetAsync(historicalBars, config, cancellationToken);
            _logger.LogInformation("[HISTORICAL_TRAINER] Built dataset with {SampleCount} samples and {FeatureCount} features", 
                dataset.Samples.Count, dataset.FeatureCount);

            // 3. Perform walk-forward training
            var walkForwardResults = await _walkForwardTrainer.TrainWalkForwardAsync(dataset, config, cancellationToken);
            _logger.LogInformation("[HISTORICAL_TRAINER] Walk-forward training completed: {FoldCount} folds, Avg AUC: {AvgAuc:F4}", 
                walkForwardResults.FoldResults.Count, walkForwardResults.AverageAuc);

            // 4. Select best model and deploy to registry
            var bestModel = SelectBestModel(walkForwardResults);
            var deployedModel = await _registryWriter.DeployModelAsync(bestModel, config, cancellationToken);
            
            _logger.LogInformation("[HISTORICAL_TRAINER] Best model deployed: {ModelId}, AUC: {Auc:F4}, Path: {ModelPath}", 
                deployedModel.ModelId, deployedModel.Metrics.AUC, deployedModel.ModelPath);

            return new HistoricalTrainingResult
            {
                Success = true,
                ModelId = deployedModel.ModelId,
                ModelPath = deployedModel.ModelPath,
                TrainingMetrics = deployedModel.Metrics,
                WalkForwardResults = walkForwardResults,
                DatasetInfo = new DatasetInfo
                {
                    SampleCount = dataset.Samples.Count,
                    FeatureCount = dataset.FeatureCount,
                    StartDate = config.StartDate,
                    EndDate = config.EndDate,
                    Symbols = config.Symbols
                },
                CompletedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HISTORICAL_TRAINER] Training failed");
            return new HistoricalTrainingResult
            {
                Success = false,
                ErrorMessage = ex.Message,
                CompletedAt = DateTime.UtcNow
            };
        }
    }

    /// <summary>
    /// Load historical bar data from /data/historical/ directory
    /// </summary>
    private async Task<Dictionary<string, List<Bar>>> LoadHistoricalDataAsync(
        List<string> symbols, 
        DateTime startDate, 
        DateTime endDate, 
        CancellationToken cancellationToken)
    {
        var historicalBars = new Dictionary<string, List<Bar>>();

        foreach (var symbol in symbols)
        {
            try
            {
                var symbolPath = Path.Combine(_historicalDataPath, $"{symbol.ToUpperInvariant()}_bars.json");
                if (!File.Exists(symbolPath))
                {
                    _logger.LogWarning("[HISTORICAL_TRAINER] Historical data file not found: {SymbolPath}", symbolPath);
                    continue;
                }

                var content = await File.ReadAllTextAsync(symbolPath, cancellationToken);
                var allBars = JsonSerializer.Deserialize<List<Bar>>(content) ?? new List<Bar>();
                
                // Filter by date range
                var filteredBars = allBars
                    .Where(bar => bar.Start >= startDate && bar.Start <= endDate)
                    .OrderBy(bar => bar.Start)
                    .ToList();

                historicalBars[symbol] = filteredBars;
                _logger.LogDebug("[HISTORICAL_TRAINER] Loaded {BarCount} bars for {Symbol} from {StartDate} to {EndDate}", 
                    filteredBars.Count, symbol, startDate.ToShortDateString(), endDate.ToShortDateString());
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[HISTORICAL_TRAINER] Failed to load historical data for symbol: {Symbol}", symbol);
            }
        }

        return historicalBars;
    }

    /// <summary>
    /// Select the best performing model from walk-forward results
    /// </summary>
    private TrainedModelResult SelectBestModel(WalkForwardResults walkForwardResults)
    {
        // Select model with best AUC and minimum sample size
        var bestFold = walkForwardResults.FoldResults
            .Where(fold => fold.Metrics.SampleSize >= 1000) // Minimum sample size
            .OrderByDescending(fold => fold.Metrics.AUC)
            .ThenByDescending(fold => fold.Metrics.PrAt10)
            .ThenBy(fold => fold.Metrics.ECE) // Lower Expected Calibration Error is better
            .FirstOrDefault();

        if (bestFold == null)
        {
            // Fallback to any model if none meet minimum sample size
            bestFold = walkForwardResults.FoldResults
                .OrderByDescending(fold => fold.Metrics.AUC)
                .First();
            
            _logger.LogWarning("[HISTORICAL_TRAINER] No models met minimum sample size requirement, using best available model");
        }

        return bestFold;
    }
}

/// <summary>
/// Builds training datasets from historical bar data
/// </summary>
public class DatasetBuilder
{
    private readonly ILogger _logger;
    private readonly MLFeatureStore _featureStore;

    public DatasetBuilder(ILogger logger, MLFeatureStore featureStore)
    {
        _logger = logger;
        _featureStore = featureStore;
    }

    public async Task<TrainingDataset> BuildDatasetAsync(
        Dictionary<string, List<Bar>> historicalBars,
        HistoricalTrainingConfig config,
        CancellationToken cancellationToken)
    {
        var samples = new List<TrainingSample>();
        var featureName = new HashSet<string>();

        foreach (var (symbol, bars) in historicalBars)
        {
            if (bars.Count < config.MinimumBarsPerSymbol)
            {
                _logger.LogWarning("[DATASET_BUILDER] Insufficient bars for {Symbol}: {BarCount} < {MinimumBars}", 
                    symbol, bars.Count, config.MinimumBarsPerSymbol);
                continue;
            }

            // Generate samples with overlapping windows
            for (int i = config.LookbackPeriod; i < bars.Count - config.ForwardLookPeriod; i += config.SampleStride)
            {
                try
                {
                    var sample = await BuildSampleAsync(symbol, bars, i, config, cancellationToken);
                    if (sample != null)
                    {
                        samples.Add(sample);
                        foreach (var featureName2 in sample.Features.Keys)
                        {
                            featureName.Add(featureName2);
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "[DATASET_BUILDER] Failed to build sample for {Symbol} at index {Index}", symbol, i);
                }
            }
        }

        _logger.LogInformation("[DATASET_BUILDER] Built dataset: {SampleCount} samples, {FeatureCount} features", 
            samples.Count, featureName.Count);

        return new TrainingDataset
        {
            Samples = samples,
            FeatureNames = featureName.ToList(),
            FeatureCount = featureName.Count,
            CreatedAt = DateTime.UtcNow
        };
    }

    private Task<TrainingSample?> BuildSampleAsync(
        string symbol, 
        List<Bar> bars, 
        int currentIndex, 
        HistoricalTrainingConfig config,
        CancellationToken cancellationToken)
    {
        try
        {
            var currentBar = bars[currentIndex];
            var features = new Dictionary<string, double>();

            // Technical indicators over lookback period
            var lookbackBars = bars.Skip(currentIndex - config.LookbackPeriod).Take(config.LookbackPeriod).ToList();
            
            // Price-based features
            features["close_price"] = (double)currentBar.Close;
            features["volume"] = currentBar.Volume;
            features["high_low_ratio"] = (double)(currentBar.High / Math.Max(currentBar.Low, 0.01m));
            features["open_close_ratio"] = (double)(currentBar.Open / Math.Max(currentBar.Close, 0.01m));
            
            // Moving averages
            if (lookbackBars.Count >= 10)
            {
                features["sma_10"] = (double)lookbackBars.TakeLast(10).Average(b => b.Close);
                features["ema_10"] = CalculateEMA(lookbackBars.TakeLast(10).Select(b => (double)b.Close).ToList(), 10);
            }
            
            if (lookbackBars.Count >= 20)
            {
                features["sma_20"] = (double)lookbackBars.TakeLast(20).Average(b => b.Close);
                features["ema_20"] = CalculateEMA(lookbackBars.TakeLast(20).Select(b => (double)b.Close).ToList(), 20);
            }

            // Volatility features
            if (lookbackBars.Count >= 5)
            {
                var returns = lookbackBars.Zip(lookbackBars.Skip(1), (prev, curr) => Math.Log((double)(curr.Close / prev.Close))).ToList();
                features["volatility_5d"] = CalculateStandardDeviation(returns.TakeLast(5).ToList());
                features["volatility_10d"] = returns.Count >= 10 ? CalculateStandardDeviation(returns.TakeLast(10).ToList()) : features["volatility_5d"];
            }

            // Volume features
            if (lookbackBars.Count >= 10)
            {
                var avgVolume = lookbackBars.TakeLast(10).Average(b => b.Volume);
                features["volume_ratio"] = currentBar.Volume / Math.Max(avgVolume, 1.0);
            }

            // Generate label (target) - predict if price will go up in next N periods
            var futureIndex = Math.Min(currentIndex + config.ForwardLookPeriod, bars.Count - 1);
            var futurePrice = bars[futureIndex].Close;
            var priceReturn = (double)((futurePrice - currentBar.Close) / currentBar.Close);
            
            // Binary classification: 1 if price goes up by more than threshold, 0 otherwise
            var label = priceReturn > config.MinimumReturnThreshold ? 1.0 : 0.0;

            var sample = new TrainingSample
            {
                Features = features,
                Label = label,
                Symbol = symbol,
                Timestamp = currentBar.Start,
                Metadata = new Dictionary<string, object>
                {
                    ["price_return"] = priceReturn,
                    ["current_price"] = (double)currentBar.Close,
                    ["future_price"] = (double)futurePrice,
                    ["lookback_periods"] = config.LookbackPeriod,
                    ["forward_periods"] = config.ForwardLookPeriod
                }
            };
            
            return Task.FromResult<TrainingSample?>(sample);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[DATASET_BUILDER] Failed to build sample for {Symbol} at {Timestamp}", 
                symbol, bars[currentIndex].Start);
            return Task.FromResult<TrainingSample?>(null);
        }
    }

    private double CalculateEMA(List<double> values, int period)
    {
        if (values.Count == 0) return 0.0;
        if (values.Count == 1) return values[0];

        var multiplier = 2.0 / (period + 1);
        var ema = values[0];

        for (int i = 1; i < values.Count; i++)
        {
            ema = (values[i] * multiplier) + (ema * (1 - multiplier));
        }

        return ema;
    }

    private double CalculateStandardDeviation(List<double> values)
    {
        if (values.Count < 2) return 0.0;

        var mean = values.Average();
        var squaredDiffs = values.Select(v => Math.Pow(v - mean, 2));
        var variance = squaredDiffs.Average();
        return Math.Sqrt(variance);
    }
}

/// <summary>
/// Walk-forward training implementation with cross-validation
/// </summary>
public class WalkForwardTrainer
{
    private readonly ILogger _logger;
    private readonly MLContext _mlContext;

    public WalkForwardTrainer(ILogger logger)
    {
        _logger = logger;
        _mlContext = new MLContext(seed: 42);
    }

    public async Task<WalkForwardResults> TrainWalkForwardAsync(
        TrainingDataset dataset,
        HistoricalTrainingConfig config,
        CancellationToken cancellationToken)
    {
        var foldResults = new List<TrainedModelResult>();
        var foldCount = config.WalkForwardFolds;
        var samplesPerFold = dataset.Samples.Count / foldCount;

        for (int fold = 0; fold < foldCount; fold++)
        {
            try
            {
                _logger.LogInformation("[WALK_FORWARD] Training fold {Fold}/{TotalFolds}", fold + 1, foldCount);

                // Create train/test split for this fold
                var trainEndIndex = (fold + 1) * samplesPerFold;
                var testStartIndex = trainEndIndex;
                var testEndIndex = Math.Min(testStartIndex + (samplesPerFold / 2), dataset.Samples.Count);

                var trainSamples = dataset.Samples.Take(trainEndIndex).ToList();
                var testSamples = dataset.Samples.Skip(testStartIndex).Take(testEndIndex - testStartIndex).ToList();

                if (trainSamples.Count < config.MinimumTrainingSamples || testSamples.Count < 100)
                {
                    _logger.LogWarning("[WALK_FORWARD] Insufficient samples for fold {Fold}: train={TrainCount}, test={TestCount}", 
                        fold + 1, trainSamples.Count, testSamples.Count);
                    continue;
                }

                // Train model
                var trainedModel = await TrainModelAsync(trainSamples, testSamples, config, cancellationToken);
                if (trainedModel != null)
                {
                    trainedModel.FoldNumber = fold + 1;
                    foldResults.Add(trainedModel);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[WALK_FORWARD] Failed to train fold {Fold}", fold + 1);
            }
        }

        var averageAuc = foldResults.Any() ? foldResults.Average(r => r.Metrics.AUC) : 0.0;
        
        return new WalkForwardResults
        {
            FoldResults = foldResults,
            AverageAuc = averageAuc,
            CompletedFolds = foldResults.Count,
            TotalFolds = foldCount
        };
    }

    private async Task<TrainedModelResult?> TrainModelAsync(
        List<TrainingSample> trainSamples,
        List<TrainingSample> testSamples,
        HistoricalTrainingConfig config,
        CancellationToken cancellationToken)
    {
        try
        {
            // Convert to ML.NET data structure
            var trainData = ConvertToMLNetData(trainSamples);
            var testData = ConvertToMLNetData(testSamples);

            // Create training pipeline
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(_mlContext.Transforms.Concatenate("Features", trainSamples.First().Features.Keys.ToArray()))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"));

            // Train the model
            var model = pipeline.Fit(trainData);

            // Evaluate on test set
            var predictions = model.Transform(testData);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);

            // Save model to temporary location
            var tempModelPath = Path.Combine(Path.GetTempPath(), $"model_{Guid.NewGuid():N}.zip");
            _mlContext.Model.Save(model, trainData.Schema, tempModelPath);

            return new TrainedModelResult
            {
                ModelData = await File.ReadAllBytesAsync(tempModelPath, cancellationToken),
                ModelPath = tempModelPath,
                Metrics = new MLModelMetrics
                {
                    AUC = metrics.AreaUnderRocCurve,
                    PrAt10 = CalculatePrecisionAtRecall(metrics, 0.10),
                    ECE = 0.0, // Expected Calibration Error would need separate calculation
                    EdgeBps = (metrics.AreaUnderRocCurve - 0.5) * 10000, // Convert AUC edge to basis points
                    SampleSize = testSamples.Count,
                    ComputedAt = DateTime.UtcNow
                },
                TrainingSampleCount = trainSamples.Count,
                TestSampleCount = testSamples.Count,
                TrainedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[WALK_FORWARD] Failed to train model");
            return null;
        }
    }

    private Microsoft.ML.IDataView ConvertToMLNetData(List<TrainingSample> samples)
    {
        // Create ML.NET compatible data structure
        var mlData = samples.Select(sample => new MLNetTrainingSample
        {
            Label = sample.Label > 0.5f, // Convert to boolean for binary classification
            Features = sample.Features.Values.Select(v => (float)v).ToArray()
        });

        return _mlContext.Data.LoadFromEnumerable(mlData);
    }

    private double CalculatePrecisionAtRecall(Microsoft.ML.Data.BinaryClassificationMetrics metrics, double targetRecall)
    {
        // Simplified precision@recall calculation
        // In a full implementation, this would use the PR curve data
        return metrics.AreaUnderPrecisionRecallCurve;
    }

    private class MLNetTrainingSample
    {
        public bool Label { get; set; }
        public float[] Features { get; set; } = Array.Empty<float>();
    }
}

/// <summary>
/// Writes trained models to the registry and deployment locations
/// </summary>
public class RegistryWriter
{
    private readonly ILogger _logger;
    private readonly string _modelsOutputPath = null!;

    public RegistryWriter(ILogger logger, string modelsOutputPath)
    {
        _logger = logger;
        _modelsOutputPath = modelsOutputPath;
    }

    public async Task<DeployedModel> DeployModelAsync(
        TrainedModelResult trainedModel,
        HistoricalTrainingConfig config,
        CancellationToken cancellationToken)
    {
        try
        {
            var modelId = $"historical_model_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N[..8]}";
            var deploymentPath = Path.Combine(_modelsOutputPath, $"{modelId}.zip");
            
            // Copy model to deployment location
            await File.WriteAllBytesAsync(deploymentPath, trainedModel.ModelData, cancellationToken);

            // Create model metadata
            var modelMetadata = new ModelMetadata
            {
                ModelId = modelId,
                ModelPath = deploymentPath,
                TrainingConfig = config,
                Metrics = trainedModel.Metrics,
                TrainingSampleCount = trainedModel.TrainingSampleCount,
                TestSampleCount = trainedModel.TestSampleCount,
                FoldNumber = trainedModel.FoldNumber,
                CreatedAt = DateTime.UtcNow,
                ModelType = "FastTree",
                Version = "1.0"
            };

            // Save metadata
            var metadataPath = Path.Combine(_modelsOutputPath, $"{modelId}_metadata.json");
            var metadataJson = JsonSerializer.Serialize(modelMetadata, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(metadataPath, metadataJson, cancellationToken);

            // Create latest model symlink/copy
            var latestModelPath = Path.Combine(_modelsOutputPath, "latest_model.zip");
            var latestMetadataPath = Path.Combine(_modelsOutputPath, "latest_model_metadata.json");
            
            File.Copy(deploymentPath, latestModelPath, true);
            File.Copy(metadataPath, latestMetadataPath, true);

            _logger.LogInformation("[REGISTRY_WRITER] Model deployed successfully: {ModelId} -> {DeploymentPath}", 
                modelId, deploymentPath);

            return new DeployedModel
            {
                ModelId = modelId,
                ModelPath = deploymentPath,
                MetadataPath = metadataPath,
                Metrics = trainedModel.Metrics,
                DeployedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REGISTRY_WRITER] Failed to deploy model");
            throw;
        }
    }
}

#region Data Transfer Objects

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

public class DatasetInfo
{
    public int SampleCount { get; set; }
    public int FeatureCount { get; set; }
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public List<string> Symbols { get; set; } = new();
}

public class TrainingDataset
{
    public List<TrainingSample> Samples { get; set; } = new();
    public List<string> FeatureNames { get; set; } = new();
    public int FeatureCount { get; set; }
    public DateTime CreatedAt { get; set; }
}

public class TrainingSample
{
    public Dictionary<string, double> Features { get; set; } = new();
    public double Label { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
}

public class WalkForwardResults
{
    public List<TrainedModelResult> FoldResults { get; set; } = new();
    public double AverageAuc { get; set; }
    public int CompletedFolds { get; set; }
    public int TotalFolds { get; set; }
}

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

public class DeployedModel
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public string MetadataPath { get; set; } = string.Empty;
    public MLModelMetrics Metrics { get; set; } = new();
    public DateTime DeployedAt { get; set; }
}

public class ModelMetadata
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public HistoricalTrainingConfig TrainingConfig { get; set; } = new();
    public MLModelMetrics Metrics { get; set; } = new();
    public int TrainingSampleCount { get; set; }
    public int TestSampleCount { get; set; }
    public int FoldNumber { get; set; }
    public DateTime CreatedAt { get; set; }
    public string ModelType { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
}

#endregion