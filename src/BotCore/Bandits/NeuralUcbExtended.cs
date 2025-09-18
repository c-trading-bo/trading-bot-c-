using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.ML;

namespace BotCore.Bandits;

/// <summary>
/// Extended Neural UCB Bandit for strategy-parameter bundle selection
/// 
/// Instead of selecting just strategies (S2, S3, S6, S11), this selects
/// strategy-parameter combinations (S2-1.3x-0.65, S3-1.0x-0.70, etc.)
/// 
/// This enables learning optimal parameters for each strategy rather than
/// using hardcoded values like MaxPositionMultiplier = 2.5 and confidenceThreshold = 0.7
/// </summary>
public class NeuralUcbExtended : IDisposable
{
    private readonly ILogger<NeuralUcbExtended> _logger;
    private readonly NeuralUcbBandit _underlyingBandit;
    private readonly List<ParameterBundle> _availableBundles;
    private readonly Dictionary<string, BundlePerformance> _bundlePerformance = new();
    private readonly object _performanceLock = new();
    
    // Configuration
    private readonly NeuralUcbExtendedConfig _config;
    
    // Performance tracking
    private readonly Dictionary<string, int> _bundleSelectionCount = new();
    private readonly Dictionary<string, decimal> _bundleRewardSum = new();
    private DateTime _lastPerformanceUpdate = DateTime.UtcNow;
    
    public NeuralUcbExtended(
        ILogger<NeuralUcbExtended> logger,
        INeuralNetwork networkTemplate,
        NeuralUcbExtendedConfig? config = null)
    {
        _logger = logger;
        _config = config ?? new NeuralUcbExtendedConfig();
        
        // Create underlying bandit with extended configuration
        var banditConfig = new NeuralUcbConfig
        {
            ExplorationWeight = _config.ExplorationWeight,
            InputDimension = _config.InputDimension,
            MinSamplesForTraining = _config.MinSamplesForTraining,
            MinSamplesForUncertainty = _config.MinSamplesForUncertainty,
            MaxTrainingDataSize = _config.MaxTrainingDataSize,
            RetrainingInterval = _config.RetrainingInterval,
            UncertaintyEstimationSamples = _config.UncertaintyEstimationSamples
        };
        
        _underlyingBandit = new NeuralUcbBandit(networkTemplate, banditConfig);
        
        // Initialize available bundles
        _availableBundles = ParameterBundleFactory.CreateAllBundles();
        
        _logger.LogInformation("[NEURAL-UCB-EXTENDED] Initialized with {BundleCount} parameter bundles", 
            _availableBundles.Count);
        
        // Log bundle examples
        var examples = _availableBundles.Take(5).Select(b => b.BundleId);
        _logger.LogInformation("[NEURAL-UCB-EXTENDED] Example bundles: {Examples}", 
            string.Join(", ", examples));
    }
    
    /// <summary>
    /// Select optimal strategy-parameter bundle using Neural UCB
    /// This replaces the need for hardcoded MaxPositionMultiplier and confidenceThreshold
    /// </summary>
    public async Task<BundleSelection> SelectBundleAsync(
        MarketContext marketContext,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogDebug("[NEURAL-UCB-EXTENDED] Selecting bundle for market context");
            
            // Get recommended bundles based on market conditions
            var recommendedBundles = GetRecommendedBundles(marketContext);
            
            // Extract bundle IDs for underlying bandit
            var bundleIds = recommendedBundles.Select(b => b.BundleId).ToList();
            
            // Create context vector from market context
            var contextVector = CreateContextVector(marketContext);
            
            // Use underlying Neural UCB to select best bundle
            var banditSelection = await _underlyingBandit.SelectArmAsync(
                bundleIds, contextVector, cancellationToken).ConfigureAwait(false);
            
            // Parse selected bundle
            var selectedBundle = ParameterBundleFactory.ParseBundle(banditSelection.SelectedArm) 
                                ?? ParameterBundleFactory.GetDefaultBundle();
            
            // Track selection
            TrackBundleSelection(selectedBundle);
            
            // Create bundle selection result
            var result = new BundleSelection
            {
                Bundle = selectedBundle,
                UcbValue = banditSelection.UcbValue,
                Prediction = banditSelection.Prediction,
                Uncertainty = 1m - banditSelection.Confidence,
                SelectionReason = $"Neural UCB Extended: {banditSelection.SelectionReason}",
                ContextFeatures = banditSelection.ContextFeatures,
                Timestamp = DateTime.UtcNow
            };
            
            _logger.LogInformation("[NEURAL-UCB-EXTENDED] Selected bundle: {BundleId} " +
                                 "strategy={Strategy} mult={Mult:F1}x thr={Thr:F2} ucb={UcbValue:F3}",
                selectedBundle.BundleId, selectedBundle.Strategy, 
                selectedBundle.Mult, selectedBundle.Thr, result.UcbValue);
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[NEURAL-UCB-EXTENDED] Error selecting bundle, using default");
            
            // Emergency fallback to default conservative bundle
            var defaultBundle = ParameterBundleFactory.GetDefaultBundle();
            return new BundleSelection
            {
                Bundle = defaultBundle,
                UcbValue = 0.5m,
                Prediction = 0.5m,
                Uncertainty = 1.0m,
                SelectionReason = "Emergency fallback due to selection error",
                Timestamp = DateTime.UtcNow
            };
        }
    }
    
    /// <summary>
    /// Update bundle performance with trading outcome
    /// This enables the system to learn which parameter combinations work best
    /// </summary>
    public async Task UpdateBundlePerformanceAsync(
        string bundleId,
        MarketContext marketContext,
        decimal reward,
        Dictionary<string, object>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogDebug("[NEURAL-UCB-EXTENDED] Updating bundle performance: {BundleId} reward={Reward:F3}",
                bundleId, reward);
            
            // Update underlying bandit
            var contextVector = CreateContextVector(marketContext);
            await _underlyingBandit.UpdateArmAsync(bundleId, contextVector, reward, cancellationToken)
                .ConfigureAwait(false);
            
            // Update local performance tracking
            UpdateLocalPerformance(bundleId, reward, metadata);
            
            _logger.LogDebug("[NEURAL-UCB-EXTENDED] Bundle performance updated successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[NEURAL-UCB-EXTENDED] Error updating bundle performance");
        }
    }
    
    /// <summary>
    /// Get bundle performance statistics for analysis
    /// </summary>
    public async Task<Dictionary<string, BundlePerformance>> GetBundlePerformanceAsync(
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Get statistics from underlying bandit
            var armStats = await _underlyingBandit.GetArmStatisticsAsync(cancellationToken)
                .ConfigureAwait(false);
            
            var bundleStats = new Dictionary<string, BundlePerformance>();
            
            lock (_performanceLock)
            {
                foreach (var bundle in _availableBundles)
                {
                    var bundleId = bundle.BundleId;
                    var performance = _bundlePerformance.GetValueOrDefault(bundleId, new BundlePerformance
                    {
                        BundleId = bundleId,
                        Bundle = bundle
                    });
                    
                    // Enhance with bandit statistics if available
                    if (armStats.TryGetValue(bundleId, out var armStat))
                    {
                        performance = performance with
                        {
                            SelectionCount = armStat.UpdateCount,
                            AverageReward = armStat.AverageReward,
                            ModelComplexity = armStat.ModelNorm,
                            UncertaintyLevel = armStat.ConfidenceWidth,
                            LastUpdated = armStat.LastUpdated
                        };
                    }
                    
                    bundleStats[bundleId] = performance;
                }
            }
            
            return bundleStats;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[NEURAL-UCB-EXTENDED] Error getting bundle performance");
            return new Dictionary<string, BundlePerformance>();
        }
    }
    
    /// <summary>
    /// Analyze feature importance for bundle selection
    /// </summary>
    public async Task<FeatureImportanceReport> AnalyzeBundleFeatureImportanceAsync(
        CancellationToken cancellationToken = default)
    {
        try
        {
            return await _underlyingBandit.AnalyzeFeatureImportanceAsync(cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[NEURAL-UCB-EXTENDED] Error analyzing feature importance");
            return new FeatureImportanceReport
            {
                FeatureWeights = new Dictionary<string, decimal>(),
                TotalArms = _availableBundles.Count,
                ActiveArms = 0,
                GeneratedAt = DateTime.UtcNow
            };
        }
    }
    
    /// <summary>
    /// Get recommended bundles based on market conditions
    /// </summary>
    private List<ParameterBundle> GetRecommendedBundles(MarketContext marketContext)
    {
        // Determine market condition from context
        var condition = DetermineMarketCondition(marketContext);
        
        // Get recommended bundles for this condition
        var recommended = ParameterBundleFactory.GetRecommendedBundles(condition);
        
        // Apply additional filtering based on configuration
        if (_config.EnableConservativeFiltering)
        {
            recommended = recommended
                .Where(b => b.Mult <= _config.MaxPositionMultiplier)
                .Where(b => b.Thr >= _config.MinConfidenceThreshold)
                .ToList();
        }
        
        // Ensure we always have at least some bundles available
        if (recommended.Count == 0)
        {
            _logger.LogWarning("[NEURAL-UCB-EXTENDED] No recommended bundles, using default set");
            recommended = ParameterBundleFactory.CreateBundlesForStrategy("S2"); // Conservative fallback
        }
        
        _logger.LogDebug("[NEURAL-UCB-EXTENDED] Using {Count} recommended bundles for condition {Condition}",
            recommended.Count, condition);
        
        return recommended;
    }
    
    /// <summary>
    /// Determine market condition from context
    /// </summary>
    private static MarketCondition DetermineMarketCondition(MarketContext marketContext)
    {
        // Simple heuristic based on volatility and volume indicators
        var volatility = marketContext.Features.GetValueOrDefault("volatility", 0.0m);
        var volume = marketContext.Features.GetValueOrDefault("volume", 0.0m);
        var trend = marketContext.Features.GetValueOrDefault("trend", 0.0m);
        
        return (volatility, volume, trend) switch
        {
            var (v, _, _) when v > 0.7m => MarketCondition.Volatile,
            var (_, vol, _) when vol > 0.8m => MarketCondition.HighVolume,
            var (_, vol, _) when vol < 0.3m => MarketCondition.LowVolume,
            var (_, _, t) when Math.Abs(t) > 0.6m => MarketCondition.Trending,
            var (_, _, t) when Math.Abs(t) < 0.3m => MarketCondition.Ranging,
            _ => MarketCondition.Unknown
        };
    }
    
    /// <summary>
    /// Create context vector from market context
    /// </summary>
    private static ContextVector CreateContextVector(MarketContext marketContext)
    {
        var features = new Dictionary<string, decimal>();
        
        // Map MarketContext properties to features
        features["price"] = (decimal)marketContext.Price;
        features["volume"] = (decimal)marketContext.Volume;
        features["bid"] = (decimal)marketContext.Bid;
        features["ask"] = (decimal)marketContext.Ask;
        features["signalStrength"] = (decimal)marketContext.SignalStrength;
        features["confidenceLevel"] = (decimal)marketContext.ConfidenceLevel;
        features["modelConfidence"] = (decimal)marketContext.ModelConfidence;
        features["newsIntensity"] = (decimal)marketContext.NewsIntensity;
        
        // Map technical indicators
        foreach (var kvp in marketContext.TechnicalIndicators)
        {
            features[$"tech_{kvp.Key}"] = (decimal)kvp.Value;
        }
        
        // Add derived features
        features["timestamp"] = (decimal)DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        features["hour"] = DateTime.UtcNow.Hour;
        features["dayOfWeek"] = (decimal)DateTime.UtcNow.DayOfWeek;
        features["isFomcDay"] = marketContext.IsFomcDay ? 1m : 0m;
        features["isCpiDay"] = marketContext.IsCpiDay ? 1m : 0m;
        
        // Calculate volatility indicator from bid-ask spread
        if (marketContext.Ask > 0 && marketContext.Bid > 0)
        {
            features["volatility"] = (decimal)((marketContext.Ask - marketContext.Bid) / marketContext.Ask);
        }
        else
        {
            features["volatility"] = 0.5m; // Default moderate volatility
        }
        
        // Calculate trend indicator (simplified)
        features["trend"] = (decimal)marketContext.SignalStrength;
        
        return new ContextVector { Features = features };
    }
    
    /// <summary>
    /// Track bundle selection for local performance monitoring
    /// </summary>
    private void TrackBundleSelection(ParameterBundle bundle)
    {
        lock (_performanceLock)
        {
            var bundleId = bundle.BundleId;
            
            _bundleSelectionCount[bundleId] = _bundleSelectionCount.GetValueOrDefault(bundleId, 0) + 1;
            
            if (!_bundlePerformance.ContainsKey(bundleId))
            {
                _bundlePerformance[bundleId] = new BundlePerformance
                {
                    BundleId = bundleId,
                    Bundle = bundle,
                    SelectionCount = 0,
                    AverageReward = 0m,
                    LastUpdated = DateTime.UtcNow
                };
            }
            
            _bundlePerformance[bundleId] = _bundlePerformance[bundleId] with 
            { 
                SelectionCount = _bundleSelectionCount[bundleId],
                LastUpdated = DateTime.UtcNow 
            };
        }
    }
    
    /// <summary>
    /// Update local performance tracking
    /// </summary>
    private void UpdateLocalPerformance(string bundleId, decimal reward, Dictionary<string, object>? metadata)
    {
        lock (_performanceLock)
        {
            if (_bundlePerformance.TryGetValue(bundleId, out var current))
            {
                var newCount = current.SelectionCount;
                var newAverage = newCount > 0 ? 
                    (current.AverageReward * (newCount - 1) + reward) / newCount : reward;
                
                _bundlePerformance[bundleId] = current with
                {
                    AverageReward = newAverage,
                    TotalReward = current.TotalReward + reward,
                    LastReward = reward,
                    LastUpdated = DateTime.UtcNow,
                    Metadata = metadata ?? new Dictionary<string, object>()
                };
            }
        }
        
        _lastPerformanceUpdate = DateTime.UtcNow;
    }
    
    public void Dispose()
    {
        _underlyingBandit?.Dispose();
    }
}

/// <summary>
/// Configuration for Neural UCB Extended
/// </summary>
public record NeuralUcbExtendedConfig
{
    public decimal ExplorationWeight { get; init; } = 0.1m;
    public int InputDimension { get; init; } = 15;
    public int MinSamplesForTraining { get; init; } = 20;
    public int MinSamplesForUncertainty { get; init; } = 10;
    public int MaxTrainingDataSize { get; init; } = 1000;
    public TimeSpan RetrainingInterval { get; init; } = TimeSpan.FromMinutes(30);
    public int UncertaintyEstimationSamples { get; init; } = 5;
    
    // Extended configuration for bundle selection
    public bool EnableConservativeFiltering { get; init; } = true;
    public decimal MaxPositionMultiplier { get; init; } = 1.6m;
    public decimal MinConfidenceThreshold { get; init; } = 0.60m;
    public int MaxRecommendedBundles { get; init; } = 18; // Half of total bundles
}

/// <summary>
/// Bundle performance tracking
/// </summary>
public record BundlePerformance
{
    public string BundleId { get; init; } = string.Empty;
    public ParameterBundle Bundle { get; init; } = ParameterBundleFactory.GetDefaultBundle();
    public int SelectionCount { get; init; }
    public decimal AverageReward { get; init; }
    public decimal TotalReward { get; init; }
    public decimal LastReward { get; init; }
    public decimal ModelComplexity { get; init; }
    public decimal UncertaintyLevel { get; init; }
    public DateTime LastUpdated { get; init; }
    public Dictionary<string, object> Metadata { get; init; } = new();
}