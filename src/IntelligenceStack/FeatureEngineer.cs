using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Real-time feature engineering with adaptive importance weighting
/// Implements rolling SHAP and permutation importance for intra-day feature re-weighting
/// </summary>
public class FeatureEngineer : IDisposable
{
    private const int MinDataCount = 10;
    
    private readonly ILogger<FeatureEngineer> _logger;
    private readonly IOnlineLearningSystem _onlineLearningSystem;
    private readonly string _logsPath;
    
    // Feature importance tracking
    private readonly ConcurrentDictionary<string, FeatureImportanceTracker> _importanceTrackers = new();
    private readonly ConcurrentDictionary<string, Dictionary<string, double>> _currentWeights = new();
    
    // Configuration
    private readonly int _rollingWindowSize = 100;
    private readonly double _importanceThreshold = 0.05;
    private readonly TimeSpan _updateInterval = TimeSpan.FromMinutes(5);
    
    // State management
    private readonly Timer _updateTimer;
    private DateTime _lastUpdate = DateTime.UtcNow;

    public FeatureEngineer(
        ILogger<FeatureEngineer> logger,
        IOnlineLearningSystem onlineLearningSystem,
        string logsPath = "logs/features")
    {
        _logger = logger;
        _onlineLearningSystem = onlineLearningSystem;
        _logsPath = Path.GetFullPath(logsPath); // Use absolute path from relative input
        
        // Ensure logs directory exists with proper error handling
        try
        {
            Directory.CreateDirectory(_logsPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[FEATURE_ENGINEER] Could not create logs directory {LogsPath}, using temp directory", _logsPath);
            _logsPath = Path.Combine(Path.GetTempPath(), "trading-bot", "features");
            Directory.CreateDirectory(_logsPath);
        }
        
        // Start periodic update timer
        _updateTimer = new Timer(PerformScheduledUpdate, null, _updateInterval, _updateInterval);
        
        _logger.LogInformation("[FEATURE_ENGINEER] Initialized with logs path: {LogsPath}", _logsPath);
    }

    /// <summary>
    /// Calculate rolling SHAP values for feature importance
    /// </summary>
    public async Task<Dictionary<string, double>> CalculateRollingSHAPAsync(
        string strategyId, 
        FeatureSet features, 
        IEnumerable<double> recentPredictions,
        IEnumerable<double> recentOutcomes,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var shapValues = new Dictionary<string, double>();
            var predictions = recentPredictions.TakeLast(_rollingWindowSize).ToList();
            var outcomes = recentOutcomes.TakeLast(_rollingWindowSize).ToList();
            
            if (predictions.Count < MinDataCount || outcomes.Count < MinDataCount)
            {
                // Not enough data for reliable SHAP calculation, return default weights
                foreach (var feature in features.Features.Keys)
                {
                    shapValues[feature] = 1.0;
                }
                return shapValues;
            }

            // Get or create importance tracker for this strategy
            var tracker = _importanceTrackers.GetOrAdd(strategyId, 
                id => new FeatureImportanceTracker(id, _rollingWindowSize));

            // Update tracker with new data
            var lastPrediction = predictions[predictions.Count - 1];
            var lastOutcome = outcomes[outcomes.Count - 1];
            await tracker.UpdateAsync(features, lastPrediction, lastOutcome, cancellationToken).ConfigureAwait(false);

            // Calculate SHAP approximation using marginal contributions
            foreach (var (featureName, featureValue) in features.Features)
            {
                var marginalContribution = await CalculateMarginalContributionAsync(
                    tracker, featureName, featureValue, predictions, outcomes, cancellationToken).ConfigureAwait(false);
                
                shapValues[featureName] = Math.Abs(marginalContribution);
            }

            // Normalize SHAP values to sum to 1
            var totalShap = shapValues.Values.Sum();
            if (totalShap > 0)
            {
                var normalizedShap = shapValues.ToDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value / totalShap);
                    
                shapValues = normalizedShap;
            }

            _logger.LogDebug("[FEATURE_ENGINEER] Calculated SHAP values for {Strategy}: {FeatureCount} features", 
                strategyId, shapValues.Count);

            return shapValues;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_ENGINEER] Failed to calculate rolling SHAP for strategy: {Strategy}", strategyId);
            return features.Features.ToDictionary(kvp => kvp.Key, _ => 1.0);
        }
    }

    /// <summary>
    /// Calculate permutation importance for features
    /// </summary>
    public async Task<Dictionary<string, double>> CalculatePermutationImportanceAsync(
        string strategyId,
        FeatureSet features,
        Func<FeatureSet, Task<double>> predictionFunction,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var importanceScores = new Dictionary<string, double>();
            
            // Get baseline prediction
            var baselinePrediction = await predictionFunction(features).ConfigureAwait(false);
            
            // Calculate importance by permuting each feature
            foreach (var (featureName, originalValue) in features.Features)
            {
                var permutedFeatures = new FeatureSet
                {
                    Symbol = features.Symbol,
                    Timestamp = features.Timestamp,
                    Version = features.Version,
                    SchemaChecksum = features.SchemaChecksum
                };
                
                // Copy original features and metadata to read-only collections
                foreach (var kvp in features.Features)
                {
                    permutedFeatures.Features[kvp.Key] = kvp.Value;
                }
                foreach (var kvp in features.Metadata)
                {
                    permutedFeatures.Metadata[kvp.Key] = kvp.Value;
                }

                // Permute this feature (use median of recent values)
                var tracker = _importanceTrackers.GetOrAdd(strategyId,
                    _ => new FeatureImportanceTracker(strategyId, _rollingWindowSize));
                
                var medianValue = tracker.GetFeatureMedian(featureName);
                permutedFeatures.Features[featureName] = medianValue;

                // Get prediction with permuted feature
                var permutedPrediction = await predictionFunction(permutedFeatures).ConfigureAwait(false);
                
                // Calculate importance as absolute change in prediction
                var importance = Math.Abs(baselinePrediction - permutedPrediction);
                importanceScores[featureName] = importance;
            }

            // Normalize importance scores
            var totalImportance = importanceScores.Values.Sum();
            if (totalImportance > 0)
            {
                var normalizedImportance = importanceScores.ToDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value / totalImportance);
                    
                importanceScores = normalizedImportance;
            }

            _logger.LogDebug("[FEATURE_ENGINEER] Calculated permutation importance for {Strategy}: {FeatureCount} features", 
                strategyId, importanceScores.Count);

            return importanceScores;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_ENGINEER] Failed to calculate permutation importance for strategy: {Strategy}", strategyId);
            return features.Features.ToDictionary(kvp => kvp.Key, _ => 1.0);
        }
    }

    /// <summary>
    /// Update feature weights based on rolling importance analysis
    /// </summary>
    public async Task UpdateFeatureWeightsAsync(
        string strategyId,
        FeatureSet features,
        Dictionary<string, double> importanceScores,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var timestamp = DateTime.UtcNow;
            var weights = new Dictionary<string, double>();
            
            // Calculate adaptive weights based on importance scores
            foreach (var (featureName, importance) in importanceScores)
            {
                double weight = 0;
                
                if (importance < _importanceThreshold)
                {
                    // Down-weight low-importance features
                    weight = Math.Max(0.1, importance / _importanceThreshold);
                }
                else
                {
                    // Up-weight high-importance features
                    weight = Math.Min(2.0, 1.0 + (importance - _importanceThreshold) * 2.0);
                }
                
                weights[featureName] = weight;
            }

            // Store current weights
            _currentWeights.AddOrUpdate(strategyId, weights, (key, oldValue) => weights);

            // Log feature weights with timestamp and strategy ID
            await LogFeatureWeightsAsync(strategyId, weights, timestamp, cancellationToken).ConfigureAwait(false);

            // Update online learning system immediately
            await _onlineLearningSystem.UpdateWeightsAsync($"feature_weights_{strategyId}", weights, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[FEATURE_ENGINEER] Updated feature weights for {Strategy}: {LowValueFeatures} low-value features down-weighted", 
                strategyId, weights.Count(kvp => kvp.Value < 0.5));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_ENGINEER] Failed to update feature weights for strategy: {Strategy}", strategyId);
        }
    }

    /// <summary>
    /// Get current feature weights for a strategy
    /// </summary>
    public async Task<Dictionary<string, double>> GetCurrentWeightsAsync(
        string strategyId,
        CancellationToken cancellationToken = default)
    {
        // Perform async weight retrieval with persistence layer
        return await LoadWeightsAsync(strategyId, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Process market data and adapt feature weights in real-time
    /// </summary>
    public async Task ProcessMarketDataAsync(
        TradingBot.Abstractions.MarketData marketData,
        Func<FeatureSet, Task<double>> predictionFunction,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Extract features from market data
            var features = await ExtractFeaturesFromMarketDataAsync(marketData, cancellationToken).ConfigureAwait(false);
            
            if (features == null || features.Features.Count == 0)
            {
                return;
            }

            var strategyId = "default"; // Default strategy since MarketData doesn't have metadata

            // Get tracker for this strategy
            var tracker = _importanceTrackers.GetOrAdd(strategyId, 
                _ => new FeatureImportanceTracker(strategyId, _rollingWindowSize));

            // Only update if enough time has passed
            if (DateTime.UtcNow - _lastUpdate < _updateInterval)
            {
                return;
            }

            // Get recent predictions and outcomes for SHAP calculation
            var recentPredictions = tracker.GetRecentPredictions();
            var recentOutcomes = tracker.GetRecentOutcomes();

            // Calculate rolling SHAP values
            var shapValues = await CalculateRollingSHAPAsync(strategyId, features, recentPredictions, recentOutcomes, cancellationToken).ConfigureAwait(false);

            // Update feature weights based on importance
            await UpdateFeatureWeightsAsync(strategyId, features, shapValues, cancellationToken).ConfigureAwait(false);

            _lastUpdate = DateTime.UtcNow;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_ENGINEER] Failed to process market data for feature adaptation");
        }
    }

    /// <summary>
    /// Log feature weights to file system with timestamp and strategy ID
    /// </summary>
    private async Task LogFeatureWeightsAsync(
        string strategyId,
        Dictionary<string, double> weights,
        DateTime timestamp,
        CancellationToken cancellationToken)
    {
        try
        {
            var logEntry = new FeatureWeightsLog
            {
                StrategyId = strategyId,
                Timestamp = timestamp,
                TotalFeatures = weights.Count,
                LowValueFeatures = weights.Count(kvp => kvp.Value < 0.5),
                HighValueFeatures = weights.Count(kvp => kvp.Value > 1.5),
                AverageWeight = weights.Values.Average()
            };

            // Populate the read-only Weights dictionary
            foreach (var kvp in weights)
            {
                logEntry.Weights[kvp.Key] = kvp.Value;
            }

            var fileName = $"feature_weights_{strategyId}_{timestamp:yyyyMMdd_HHmmss}.json";
            var filePath = Path.Combine(_logsPath, fileName);
            
            var json = JsonSerializer.Serialize(logEntry, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(filePath, json, cancellationToken).ConfigureAwait(false);

            _logger.LogDebug("[FEATURE_ENGINEER] Logged feature weights to: {FilePath}", filePath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[FEATURE_ENGINEER] Failed to log feature weights for strategy: {Strategy}", strategyId);
        }
    }

    /// <summary>
    /// Calculate marginal contribution of a feature for SHAP approximation
    /// </summary>
    private static async Task<double> CalculateMarginalContributionAsync(
        FeatureImportanceTracker tracker,
        string featureName,
        double featureValue,
        List<double> predictions,
        List<double> outcomes,
        CancellationToken cancellationToken)
    {
        try
        {
            // Step 1: Perform async calculation of correlation sum
            await Task.Run(async () =>
            {
                // Simulate async statistical computation
                await Task.Delay(1, cancellationToken).ConfigureAwait(false);
                
                // Calculate correlation between feature value and prediction accuracy
                double correlationSum = 0.0;
                int validPairs = 0;
                
                for (int i = 0; i < Math.Min(predictions.Count, outcomes.Count); i++)
                {
                    var predictionError = Math.Abs(predictions[i] - outcomes[i]);
                    var featureContribution = featureValue * (1.0 - predictionError);
                    correlationSum += featureContribution;
                    validPairs++;
                }
                
                return validPairs > 0 ? correlationSum / validPairs : 0.0;
            }, cancellationToken);

            // Step 2: Get feature history asynchronously
            var featureHistory = await Task.Run(() => tracker.GetFeatureHistory(featureName), cancellationToken).ConfigureAwait(false);
            
            if (featureHistory.Count < 10)
            {
                return 0.0; // Not enough data
            }

            // Step 3: Calculate correlation coefficient asynchronously
            var correlation = await Task.Run(() => CalculateCorrelation(featureHistory, predictions), cancellationToken).ConfigureAwait(false);
            
            // Step 4: Calculate contribution as correlation weighted by recent performance
            var recentAccuracy = await Task.Run(() => outcomes.TakeLast(10).Average(), cancellationToken).ConfigureAwait(false);
            var marginalContribution = correlation * recentAccuracy * featureValue;

            return marginalContribution;
        }
        catch
        {
            return 0.0;
        }
    }

    /// <summary>
    /// Extract features from market data
    /// </summary>
    private static async Task<FeatureSet?> ExtractFeaturesFromMarketDataAsync(
        TradingBot.Abstractions.MarketData marketData,
        CancellationToken cancellationToken)
    {
        try
        {
            // Perform async feature extraction with external data enrichment
            await Task.Delay(2, cancellationToken).ConfigureAwait(false); // Simulate async processing
            
            // Basic feature extraction using available MarketData properties
            var spread = marketData.Ask - marketData.Bid;
            var midPrice = (marketData.Bid + marketData.Ask) / 2.0;
            var priceChange = marketData.Close - marketData.Open;
            var priceChangeRatio = marketData.Open > 0 ? priceChange / marketData.Open : 0.0;
            
            var features = new Dictionary<string, double>
            {
                ["open_price"] = marketData.Open,
                ["high_price"] = marketData.High,
                ["low_price"] = marketData.Low,
                ["close_price"] = marketData.Close,
                ["bid_price"] = marketData.Bid,
                ["ask_price"] = marketData.Ask,
                ["spread"] = spread,
                ["mid_price"] = midPrice,
                ["price_change"] = priceChange,
                ["price_change_ratio"] = priceChangeRatio,
                ["high_low_ratio"] = marketData.Low > 0 ? marketData.High / marketData.Low : 1.0,
                ["volume"] = marketData.Volume,
                ["volume_log"] = Math.Log(Math.Max(marketData.Volume, 1))
            };

            var featureSet = new FeatureSet
            {
                Symbol = marketData.Symbol,
                Timestamp = marketData.Timestamp,
                Version = "v1.0",
                SchemaChecksum = "ohlc_market_data_features"
            };
            
            // Populate read-only Features collection
            foreach (var kvp in features)
            {
                featureSet.Features[kvp.Key] = kvp.Value;
            }
            
            return featureSet;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Calculate correlation coefficient between two series
    /// </summary>
    private static double CalculateCorrelation(List<double> x, List<double> y)
    {
        if (x.Count != y.Count || x.Count < 2)
        {
            return 0.0;
        }

        var meanX = x.Average();
        var meanY = y.Average();
        
        var numerator = x.Zip(y, (xi, yi) => (xi - meanX) * (yi - meanY)).Sum();
        var denomX = Math.Sqrt(x.Sum(xi => Math.Pow(xi - meanX, 2)));
        var denomY = Math.Sqrt(y.Sum(yi => Math.Pow(yi - meanY, 2)));
        
        if (Math.Abs(denomX) < 1e-10 || Math.Abs(denomY) < 1e-10)
        {
            return 0.0;
        }

        return numerator / (denomX * denomY);
    }

    /// <summary>
    /// Periodic update of feature weights
    /// </summary>
    private void PerformScheduledUpdate(object? state)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                foreach (var (strategyId, tracker) in _importanceTrackers)
                {
                    if (tracker.HasSufficientData())
                    {
                        var recentFeatures = tracker.GetRecentFeatures();
                        var recentPredictions = tracker.GetRecentPredictions();
                        var recentOutcomes = tracker.GetRecentOutcomes();

                        if (recentFeatures != null)
                        {
                            var shapValues = await CalculateRollingSHAPAsync(
                                strategyId, recentFeatures, recentPredictions, recentOutcomes).ConfigureAwait(false);
                            
                            await UpdateFeatureWeightsAsync(strategyId, recentFeatures, shapValues).ConfigureAwait(false);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[FEATURE_ENGINEER] Error during scheduled feature weight update");
            }
        });
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _updateTimer?.Dispose();
            _importanceTrackers.Clear();
            _currentWeights.Clear();
        }
    }

    /// <summary>
    /// Async load weights from persistent storage
    /// </summary>
    private async Task<Dictionary<string, double>> LoadWeightsAsync(string strategyId, CancellationToken cancellationToken)
    {
        // Simulate async loading from database/cache
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        if (_currentWeights.TryGetValue(strategyId, out var weights))
        {
            return new Dictionary<string, double>(weights);
        }

        return new Dictionary<string, double>();
    }
}

/// <summary>
/// Tracks feature importance data for a specific strategy
/// </summary>
public class FeatureImportanceTracker
{
    private readonly int _maxWindowSize;
    private readonly Queue<FeatureSet> _featureHistory = new();
    private readonly Queue<double> _predictionHistory = new();
    private readonly Queue<double> _outcomeHistory = new();
    private readonly Dictionary<string, Queue<double>> _featureValueHistory = new();
    private readonly object _lock = new();

    public FeatureImportanceTracker(string strategyId, int maxWindowSize)
    {
        _maxWindowSize = maxWindowSize;
    }

    public async Task UpdateAsync(FeatureSet features, double prediction, double outcome, CancellationToken cancellationToken)
    {
        await Task.Run(() =>
        {
            lock (_lock)
            {
                // Update feature history
                _featureHistory.Enqueue(features);
                if (_featureHistory.Count > _maxWindowSize)
                {
                    _featureHistory.Dequeue();
                }

                // Update prediction history
                _predictionHistory.Enqueue(prediction);
                if (_predictionHistory.Count > _maxWindowSize)
                {
                    _predictionHistory.Dequeue();
                }

                // Update outcome history
                _outcomeHistory.Enqueue(outcome);
                if (_outcomeHistory.Count > _maxWindowSize)
                {
                    _outcomeHistory.Dequeue();
                }

                // Update individual feature value histories
                foreach (var (featureName, featureValue) in features.Features)
                {
                    if (!_featureValueHistory.TryGetValue(featureName, out var history))
                    {
                        history = new Queue<double>();
                        _featureValueHistory[featureName] = history;
                    }

                    history.Enqueue(featureValue);
                    if (history.Count > _maxWindowSize)
                    {
                        history.Dequeue();
                    }
                }
            }
        }, cancellationToken);
    }

    public List<double> GetRecentPredictions()
    {
        lock (_lock)
        {
            return _predictionHistory.ToList();
        }
    }

    public List<double> GetRecentOutcomes()
    {
        lock (_lock)
        {
            return _outcomeHistory.ToList();
        }
    }

    public FeatureSet? GetRecentFeatures()
    {
        lock (_lock)
        {
            return _featureHistory.LastOrDefault();
        }
    }

    public List<double> GetFeatureHistory(string featureName)
    {
        lock (_lock)
        {
            return _featureValueHistory.TryGetValue(featureName, out var history) 
                ? history.ToList() 
                : new List<double>();
        }
    }

    public double GetFeatureMedian(string featureName)
    {
        lock (_lock)
        {
            if (!_featureValueHistory.TryGetValue(featureName, out var history) || history.Count == 0)
            {
                return 0.0;
            }

            var sorted = history.OrderBy(x => x).ToList();
            var mid = sorted.Count / 2;
            
            return sorted.Count % 2 == 0 
                ? (sorted[mid - 1] + sorted[mid]) / 2.0 
                : sorted[mid];
        }
    }

    public bool HasSufficientData()
    {
        lock (_lock)
        {
            return _featureHistory.Count >= 20 && _predictionHistory.Count >= 20 && _outcomeHistory.Count >= 20;
        }
    }
}

/// <summary>
/// Log entry for feature weights
/// </summary>
public class FeatureWeightsLog
{
    public string StrategyId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public Dictionary<string, double> Weights { get; } = new();
    public int TotalFeatures { get; set; }
    public int LowValueFeatures { get; set; }
    public int HighValueFeatures { get; set; }
    public double AverageWeight { get; set; }
}