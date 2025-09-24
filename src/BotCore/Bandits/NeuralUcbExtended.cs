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
/// using hardcoded values like MaxPositionMultiplier = 2.0 and confidenceThreshold = 0.7
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
    /// Get recommended bundles based on market conditions with bracket optimization
    /// </summary>
    private List<ParameterBundle> GetRecommendedBundles(MarketContext marketContext)
    {
        // Determine market condition from context
        var condition = DetermineMarketCondition(marketContext);
        
        // Get recommended bundles for this condition
        var recommended = ParameterBundleFactory.GetRecommendedBundles(condition);
        
        // Apply bracket-specific filtering if enabled
        if (_config.BracketSelection.EnableAdaptiveBrackets)
        {
            recommended = ApplyBracketFiltering(recommended, marketContext);
        }
        
        // Apply additional filtering based on configuration
        if (_config.EnableConservativeFiltering)
        {
            recommended = recommended
                .Where(b => b.Mult <= _config.MaxPositionMultiplier)
                .Where(b => b.Thr >= _config.MinConfidenceThreshold)
                .Where(b => b.BracketMode.RiskRewardRatio >= _config.BracketSelection.MinRiskRewardRatio)
                .Where(b => b.BracketMode.RiskRewardRatio <= _config.BracketSelection.MaxRiskRewardRatio)
                .ToList();
        }
        
        // Limit to configured maximum
        if (recommended.Count > _config.MaxRecommendedBundles)
        {
            recommended = recommended.Take(_config.MaxRecommendedBundles).ToList();
        }
        
        // Ensure we always have at least some bundles available
        if (recommended.Count == 0)
        {
            _logger.LogWarning("[NEURAL-UCB-EXTENDED] No recommended bundles, using default set");
            recommended = ParameterBundleFactory.CreateBundlesForStrategy("S2")
                .Where(b => b.BracketMode.ModeType == "Conservative")
                .Take(5)
                .ToList(); // Conservative fallback
        }
        
        _logger.LogDebug("[NEURAL-UCB-EXTENDED] Using {Count} recommended bundles for condition {Condition}",
            recommended.Count, condition);
        
        return recommended;
    }
    
    /// <summary>
    /// Apply bracket-specific filtering based on current market conditions
    /// </summary>
    private List<ParameterBundle> ApplyBracketFiltering(List<ParameterBundle> bundles, MarketContext marketContext)
    {
        var filteredBundles = bundles.AsEnumerable();
        
        // Volatility-based filtering
        if (_config.BracketSelection.EnableVolatilityAdjustments)
        {
            var volatility = CalculateBracketVolatility(marketContext);
            
            if (volatility > _config.BracketSelection.HighVolatilityThreshold)
            {
                // High volatility: prefer conservative and scalping brackets
                filteredBundles = filteredBundles.Where(b => 
                    b.BracketMode.ModeType is "Conservative" or "Scalping");
            }
            else if (volatility < _config.BracketSelection.LowVolatilityThreshold)
            {
                // Low volatility: prefer aggressive and swing brackets  
                filteredBundles = filteredBundles.Where(b => 
                    b.BracketMode.ModeType is "Aggressive" or "Swing");
            }
        }
        
        // Time-based filtering
        if (_config.BracketSelection.EnableTimeBasedBrackets)
        {
            var currentHour = DateTime.UtcNow.Hour;
            var (scalpStart, scalpEnd) = _config.BracketSelection.ScalpingHours;
            
            if (currentHour >= scalpStart && currentHour <= scalpEnd)
            {
                // Active trading hours: favor scalping and balanced brackets
                filteredBundles = filteredBundles.Where(b => 
                    b.BracketMode.ModeType is "Scalping" or "Balanced" or "Conservative");
            }
            else
            {
                // Off hours: favor swing and aggressive brackets for longer holds
                filteredBundles = filteredBundles.Where(b => 
                    b.BracketMode.ModeType is "Swing" or "Aggressive");
            }
        }
        
        var result = filteredBundles.ToList();
        
        // If filtering removed too many options, relax constraints
        if (result.Count < 3)
        {
            _logger.LogDebug("[NEURAL-UCB-EXTENDED] Bracket filtering too restrictive, relaxing constraints");
            return bundles.Where(b => b.BracketMode.ModeType is "Conservative" or "Balanced").ToList();
        }
        
        return result;
    }
    
    /// <summary>
    /// Determine market condition from context with bracket optimization focus
    /// </summary>
    private static MarketCondition DetermineMarketCondition(MarketContext marketContext)
    {
        // Enhanced market condition detection using bracket-specific features
        var volatility = CalculateBracketVolatility(marketContext);
        var trend = CalculateBracketTrend(marketContext);
        var momentum = CalculateBracketMomentum(marketContext);
        var riskEnvironment = CalculateBracketRiskEnvironment(marketContext);
        var timeScore = CalculateBracketTimeScore(DateTime.UtcNow);
        
        // Use traditional volume if available, otherwise estimate from context
        var volume = marketContext.Features.TryGetValue("volume", out var vol) ? vol : (decimal)marketContext.Volume / 1000000m;
        
        // Enhanced condition logic with bracket considerations
        return (volatility, volume, trend, momentum, riskEnvironment, timeScore) switch
        {
            // High volatility scenarios - favor conservative/scalping brackets
            var (v, _, _, _, risk, _) when v > 0.7m || risk > 0.8m => MarketCondition.Volatile,
            
            // High volume scenarios - can support aggressive brackets
            var (_, vol, _, _, _, _) when vol > 0.8m => MarketCondition.HighVolume,
            
            // Low volume scenarios - favor tight brackets
            var (_, vol, _, _, _, _) when vol < 0.3m => MarketCondition.LowVolume,
            
            // Strong trending scenarios - favor swing/aggressive brackets
            var (_, _, t, m, _, _) when t > 0.6m && m > 0.6m => MarketCondition.Trending,
            
            // Ranging/choppy scenarios - favor scalping/balanced brackets
            var (_, _, t, _, _, _) when t < 0.3m => MarketCondition.Ranging,
            
            // Default condition
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
            features["volatility"] = 0m;
        }
        
        // Add bracket-specific features for enhanced bracket mode selection
        features["bracketVolatility"] = CalculateBracketVolatility(marketContext);
        features["bracketTrend"] = CalculateBracketTrend(marketContext);
        features["bracketMomentum"] = CalculateBracketMomentum(marketContext);
        features["bracketTimeScore"] = CalculateBracketTimeScore(DateTime.UtcNow);
        features["bracketRiskEnvironment"] = CalculateBracketRiskEnvironment(marketContext);
        
        return new ContextVector(features);
    }
    
    /// <summary>
    /// Calculate volatility measure specifically for bracket selection
    /// Combines bid-ask spread, recent price movement, and volume volatility
    /// </summary>
    private static decimal CalculateBracketVolatility(MarketContext marketContext)
    {
        var spreadVolatility = 0m;
        if (marketContext.Ask > 0 && marketContext.Bid > 0)
        {
            spreadVolatility = (decimal)((marketContext.Ask - marketContext.Bid) / marketContext.Ask);
        }
        
        // Add ATR-based volatility if available
        var atrVolatility = marketContext.TechnicalIndicators.TryGetValue("atr", out var atr) 
            ? (decimal)Math.Min(atr / marketContext.Price, 0.1) // Cap at 10%
            : 0m;
            
        // Volume volatility indicator
        var volumeVolatility = marketContext.Volume > 0 
            ? (decimal)Math.Min(marketContext.Volume / 1000000, 1.0) // Normalize volume
            : 0m;
        
        // Weighted combination favoring spread and ATR
        return (spreadVolatility * 0.4m) + (atrVolatility * 0.5m) + (volumeVolatility * 0.1m);
    }
    
    /// <summary>
    /// Calculate trend strength for bracket selection
    /// Strong trends favor wider brackets (swing/aggressive), weak trends favor tight brackets
    /// </summary>
    private static decimal CalculateBracketTrend(MarketContext marketContext)
    {
        // Use moving average indicators if available
        var ma20 = marketContext.TechnicalIndicators.TryGetValue("ma20", out var ma20Val) ? (decimal)ma20Val : (decimal)marketContext.Price;
        var ma50 = marketContext.TechnicalIndicators.TryGetValue("ma50", out var ma50Val) ? (decimal)ma50Val : (decimal)marketContext.Price;
        
        var currentPrice = (decimal)marketContext.Price;
        
        // Trend score based on price position relative to moving averages
        var trendScore = 0m;
        
        if (currentPrice > ma20 && ma20 > ma50)
            trendScore = 0.8m; // Strong uptrend
        else if (currentPrice < ma20 && ma20 < ma50)
            trendScore = -0.8m; // Strong downtrend
        else if (Math.Abs(currentPrice - ma20) / currentPrice < 0.01m)
            trendScore = 0m; // Sideways/ranging
        else
            trendScore = (currentPrice - ma20) / currentPrice * 0.5m; // Weak trend
        
        // Normalize to 0-1 range for bracket selection
        return Math.Abs(trendScore);
    }
    
    /// <summary>
    /// Calculate momentum for bracket timing decisions
    /// High momentum favors aggressive brackets, low momentum favors conservative brackets
    /// </summary>
    private static decimal CalculateBracketMomentum(MarketContext marketContext)
    {
        var signalStrength = (decimal)marketContext.SignalStrength;
        var confidence = (decimal)marketContext.ConfidenceLevel;
        
        // RSI-based momentum if available
        var rsiMomentum = marketContext.TechnicalIndicators.TryGetValue("rsi", out var rsi) 
            ? Math.Abs((decimal)rsi - 50m) / 50m // Distance from neutral (50)
            : 0.5m;
        
        // Combine signal strength, confidence, and RSI momentum
        return (signalStrength * 0.4m) + (confidence * 0.3m) + (rsiMomentum * 0.3m);
    }
    
    /// <summary>
    /// Calculate time-based score for bracket selection
    /// Active trading hours favor scalping, off-hours favor swing brackets
    /// </summary>
    private static decimal CalculateBracketTimeScore(DateTime currentTime)
    {
        var hour = currentTime.Hour;
        var dayOfWeek = currentTime.DayOfWeek;
        
        // Market hours scoring (EST): 9:30 AM - 4:00 PM = active, others = inactive
        var timeScore = hour switch
        {
            >= 9 and <= 16 => 1.0m,  // Active trading hours - favor scalping/aggressive
            >= 7 and < 9 => 0.7m,    // Pre-market - moderate activity
            > 16 and <= 20 => 0.6m,  // After hours - moderate activity
            _ => 0.3m                 // Overnight - favor conservative/swing
        };
        
        // Weekend adjustment
        if (dayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday)
        {
            timeScore *= 0.2m; // Much lower activity on weekends
        }
        
        return timeScore;
    }
    
    /// <summary>
    /// Calculate overall risk environment for bracket selection
    /// High risk environments favor conservative brackets
    /// </summary>
    private static decimal CalculateBracketRiskEnvironment(MarketContext marketContext)
    {
        var newsIntensity = (decimal)marketContext.NewsIntensity;
        var isFomcDay = marketContext.IsFomcDay ? 1m : 0m;
        var isCpiDay = marketContext.IsCpiDay ? 1m : 0m;
        
        // VIX-like indicator if available
        var vixLevel = marketContext.TechnicalIndicators.TryGetValue("vix", out var vix) 
            ? Math.Min((decimal)vix / 40m, 1m) // Normalize VIX (40+ = high risk)
            : 0.5m;
        
        // High risk events increase risk environment score
        var eventRisk = (isFomcDay * 0.8m) + (isCpiDay * 0.6m) + (newsIntensity * 0.4m);
        
        // Combine VIX and event risk (0 = low risk, 1 = high risk)
        return Math.Min((vixLevel * 0.6m) + (eventRisk * 0.4m), 1m);
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
    public int InputDimension { get; init; } = 18; // Increased from 15 to accommodate bracket features
    public int MinSamplesForTraining { get; init; } = 20;
    public int MinSamplesForUncertainty { get; init; } = 10;
    public int MaxTrainingDataSize { get; init; } = 1000;
    public TimeSpan RetrainingInterval { get; init; } = TimeSpan.FromMinutes(30);
    public int UncertaintyEstimationSamples { get; init; } = 5;
    
    // Extended configuration for bundle selection
    public bool EnableConservativeFiltering { get; init; } = true;
    public decimal MaxPositionMultiplier { get; init; } = 1.6m;
    public decimal MinConfidenceThreshold { get; init; } = 0.60m;
    public int MaxRecommendedBundles { get; init; } = 36; // Increased to accommodate bracket combinations
    
    // Bracket-specific configuration
    public BracketSelectionConfig BracketSelection { get; init; } = new();
}

/// <summary>
/// Configuration for bracket mode selection within Neural UCB
/// </summary>
public record BracketSelectionConfig
{
    /// <summary>
    /// Enable adaptive bracket selection based on market conditions
    /// </summary>
    public bool EnableAdaptiveBrackets { get; init; } = true;
    
    /// <summary>
    /// Weight for bracket performance in overall bundle evaluation
    /// Range: 0.0 (ignore brackets) to 1.0 (brackets only)
    /// </summary>
    public decimal BracketPerformanceWeight { get; init; } = 0.3m;
    
    /// <summary>
    /// Minimum risk-to-reward ratio for bracket modes
    /// </summary>
    public decimal MinRiskRewardRatio { get; init; } = 1.2m;
    
    /// <summary>
    /// Maximum risk-to-reward ratio for bracket modes
    /// </summary>
    public decimal MaxRiskRewardRatio { get; init; } = 3.0m;
    
    /// <summary>
    /// Enable volatility-based bracket adjustments
    /// </summary>
    public bool EnableVolatilityAdjustments { get; init; } = true;
    
    /// <summary>
    /// High volatility threshold for bracket mode selection
    /// </summary>
    public decimal HighVolatilityThreshold { get; init; } = 0.7m;
    
    /// <summary>
    /// Low volatility threshold for bracket mode selection
    /// </summary>
    public decimal LowVolatilityThreshold { get; init; } = 0.3m;
    
    /// <summary>
    /// Enable time-based bracket preferences (e.g., scalping during active hours)
    /// </summary>
    public bool EnableTimeBasedBrackets { get; init; } = true;
    
    /// <summary>
    /// Market hours when scalping brackets are preferred (24-hour format)
    /// </summary>
    public (int Start, int End) ScalpingHours { get; init; } = (9, 16); // 9 AM to 4 PM EST
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