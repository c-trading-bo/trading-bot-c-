using Microsoft.Extensions.Logging;
using Microsoft.ML;
using TradingBot.Abstractions;
using BotCore.Models;
using MLModelRegistry = TradingBot.ML.Interfaces.IModelRegistry;
using MLFeatureStore = TradingBot.ML.Interfaces.IFeatureStore;
using MLModelMetrics = TradingBot.ML.Models.ModelMetrics;
using TradingBot.ML.HistoricalTrainer.Models;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Diagnostics;

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
    private readonly HistoricalDatasetBuilder _datasetBuilder;
    private readonly HistoricalWalkForwardTrainer _walkForwardTrainer;
    private readonly HistoricalRegistryWriter _registryWriter;
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
        
        _datasetBuilder = new HistoricalDatasetBuilder(logger, featureStore);
        _walkForwardTrainer = new HistoricalWalkForwardTrainer(logger);
        _registryWriter = new HistoricalRegistryWriter(logger, modelsOutputPath);
        
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
            var historicalBars = await LoadHistoricalDataAsync(config.Symbols, config.StartDate, config.EndDate, cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("[HISTORICAL_TRAINER] Loaded {BarCount} historical bars for {SymbolCount} symbols", 
                historicalBars.Sum(kvp => kvp.Value.Count), historicalBars.Count);

            // 2. Build dataset with features and labels
            var dataset = await _datasetBuilder.BuildDatasetAsync(historicalBars, config, cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("[HISTORICAL_TRAINER] Built dataset with {SampleCount} samples and {FeatureCount} features", 
                dataset.Samples.Count, dataset.FeatureCount);

            // 3. Perform walk-forward training
            var walkForwardResults = await _walkForwardTrainer.TrainWalkForwardAsync(dataset, config, cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("[HISTORICAL_TRAINER] Walk-forward training completed: {FoldCount} folds, Avg AUC: {AvgAuc:F4}", 
                walkForwardResults.FoldResults.Count, walkForwardResults.AverageAuc);

            // 4. Select best model and deploy to registry
            var bestModel = SelectBestModel(walkForwardResults);
            var deployedModel = await _registryWriter.DeployModelAsync(bestModel, config, cancellationToken).ConfigureAwait(false);
            
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
    /// Load historical bar data from SDK adapter or fallback to /data/historical/ directory
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
                // PRIMARY: Try to get historical data via SDK adapter
                var sdkBars = await TryGetHistoricalDataViaSdkAsync(symbol, startDate, endDate, cancellationToken).ConfigureAwait(false);
                if (sdkBars.Any())
                {
                    // Convert BotCore.Models.Bar to the Bar type used by ML trainer
                    var mlBars = sdkBars.Select(bar => new Bar
                    {
                        Start = bar.Ts,
                        Open = (double)bar.Open,
                        High = (double)bar.High,
                        Low = (double)bar.Low,
                        Close = (double)bar.Close,
                        Volume = bar.Volume
                    }).ToList();
                    
                    historicalBars[symbol] = mlBars;
                    _logger.LogInformation("[HISTORICAL_TRAINER] Loaded {BarCount} bars for {Symbol} via SDK adapter", 
                        mlBars.Count, symbol);
                    continue;
                }

                // FALLBACK: Load from file system
                var symbolPath = Path.Combine(_historicalDataPath, $"{symbol.ToUpperInvariant()}_bars.json");
                if (!File.Exists(symbolPath))
                {
                    _logger.LogWarning("[HISTORICAL_TRAINER] Historical data file not found: {SymbolPath}", symbolPath);
                    continue;
                }

                var content = await File.ReadAllTextAsync(symbolPath, cancellationToken).ConfigureAwait(false);
                var allBars = JsonSerializer.Deserialize<List<Bar>>(content) ?? new List<Bar>();
                
                // Filter by date range
                var filteredBars = allBars
                    .Where(bar => bar.Start >= startDate && bar.Start <= endDate)
                    .OrderBy(bar => bar.Start)
                    .ToList();

                historicalBars[symbol] = filteredBars;
                _logger.LogDebug("[HISTORICAL_TRAINER] Loaded {BarCount} bars for {Symbol} from file system ({StartDate} to {EndDate})", 
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
    /// Try to get historical data via SDK bridge
    /// </summary>
    private async Task<List<BotCore.Models.Bar>> TryGetHistoricalDataViaSdkAsync(
        string symbol, 
        DateTime startDate, 
        DateTime endDate, 
        CancellationToken cancellationToken)
    {
        try
        {
            // Calculate number of bars needed (assuming 1-minute bars for now)
            var timespan = endDate - startDate;
            var estimatedBars = Math.Max(100, (int)(timespan.TotalMinutes / 5)); // Estimate for 5-min bars
            
            // Call Python SDK bridge to get historical bars
            var pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "python", "sdk_bridge.py");
            if (!File.Exists(pythonScript))
            {
                _logger.LogDebug("[HISTORICAL_TRAINER] SDK bridge script not found at {Path}", pythonScript);
                return new List<BotCore.Models.Bar>();
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"\"{pythonScript}\" get_historical_bars \"{symbol}\" \"5m\" {estimatedBars}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process == null)
            {
                _logger.LogWarning("[HISTORICAL_TRAINER] Failed to start SDK bridge process");
                return new List<BotCore.Models.Bar>();
            }

            var output = await process.StandardOutput.ReadToEndAsync().ConfigureAwait(false);
            var error = await process.StandardError.ReadToEndAsync().ConfigureAwait(false);
            await process.WaitForExitAsync().ConfigureAwait(false);

            if (process.ExitCode != 0)
            {
                _logger.LogDebug("[HISTORICAL_TRAINER] SDK bridge returned exit code {ExitCode}: {Error}", process.ExitCode, error);
                return new List<BotCore.Models.Bar>();
            }

            if (string.IsNullOrWhiteSpace(output))
            {
                _logger.LogDebug("[HISTORICAL_TRAINER] SDK bridge returned empty output");
                return new List<BotCore.Models.Bar>();
            }

            // Parse JSON response from SDK bridge
            var barData = JsonSerializer.Deserialize<List<Dictionary<string, object>>>(output);
            var bars = new List<BotCore.Models.Bar>();

            foreach (var bar in barData)
            {
                try
                {
                    var botBar = new BotCore.Models.Bar
                    {
                        ContractId = symbol,
                        Open = Convert.ToDecimal(bar["open"]),
                        High = Convert.ToDecimal(bar["high"]),
                        Low = Convert.ToDecimal(bar["low"]),
                        Close = Convert.ToDecimal(bar["close"]),
                        Volume = Convert.ToInt64(bar.GetValueOrDefault("volume", 0)),
                        Ts = DateTime.TryParse(bar["timestamp"].ToString(), out var ts) ? ts : DateTime.UtcNow
                    };
                    
                    // Filter by date range
                    if (botBar.Ts >= startDate && botBar.Ts <= endDate)
                    {
                        bars.Add(botBar);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning("[HISTORICAL_TRAINER] Failed to parse bar data: {Error}", ex.Message);
                }
            }

            return bars.OrderBy(b => b.Ts).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[HISTORICAL_TRAINER] SDK adapter historical data fetch failed for {Symbol}: {Error}", symbol, ex.Message);
            return new List<BotCore.Models.Bar>();
        }
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
public class HistoricalDatasetBuilder
{
    private readonly ILogger _logger;
    private readonly MLFeatureStore _featureStore;

    public HistoricalDatasetBuilder(ILogger logger, MLFeatureStore featureStore)
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
                    var sample = await BuildSampleAsync(symbol, bars, i, config, cancellationToken).ConfigureAwait(false);
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
public class HistoricalWalkForwardTrainer
{
    private readonly ILogger _logger;
    private readonly MLContext _mlContext;

    public HistoricalWalkForwardTrainer(ILogger logger)
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

        for (int fold; fold < foldCount; fold++)
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
                var trainedModel = await TrainModelAsync(trainSamples, testSamples, config, cancellationToken).ConfigureAwait(false);
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
            }.ConfigureAwait(false);
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
public class HistoricalRegistryWriter
{
    private readonly ILogger _logger;
    private readonly string _modelsOutputPath = null!;

    public HistoricalRegistryWriter(ILogger logger, string modelsOutputPath)
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
            await File.WriteAllBytesAsync(deploymentPath, trainedModel.ModelData, cancellationToken).ConfigureAwait(false);

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
            await File.WriteAllTextAsync(metadataPath, metadataJson, cancellationToken).ConfigureAwait(false);

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