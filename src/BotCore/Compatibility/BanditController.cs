using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.Bandits;

namespace BotCore.Compatibility;

/// <summary>
/// Bandit controller that adds a second layer of parameter bundle selection
/// to your existing Neural UCB strategy selection system
/// 
/// Component Mapping Approach: Your existing Neural UCB continues choosing among 
/// fourteen strategies, while this controller selects parameter bundles for each strategy.
/// This creates strategy-parameter combinations rather than replacing your proven strategy logic.
/// </summary>
public class BanditController : IDisposable
{
    private readonly ILogger<BanditController> _logger;
    private readonly BanditControllerConfig _config;
    private readonly NeuralUcbExtended _neuralUcbExtended;
    
    // Thompson sampling for bundle selection
    private readonly Dictionary<string, BundleStatistics> _bundleStats = new();
    private readonly object _statsLock = new();
    
    public BanditController(ILogger<BanditController> logger, BanditControllerConfig config)
    {
        _logger = logger;
        _config = config;
        
        // Initialize Neural UCB Extended for bundle selection
        _neuralUcbExtended = new NeuralUcbExtended(
            logger.CreateLogger<NeuralUcbExtended>(),
            new BasicNeuralNetwork(config.InputDimension, config.HiddenDimension),
            new NeuralUcbExtendedConfig
            {
                ExplorationWeight = config.ExplorationWeight,
                InputDimension = config.InputDimension
            });
        
        InitializeBundleStatistics();
        
        _logger.LogInformation("BanditController initialized with {BundleCount} parameter bundles", 
            _bundleStats.Count);
    }
    
    private void InitializeBundleStatistics()
    {
        // Initialize Thompson sampling statistics for all 36 bundle combinations
        var strategies = new[] { "S2", "S3", "S6", "S11" };
        var multipliers = new[] { 1.0m, 1.3m, 1.6m };
        var thresholds = new[] { 0.60m, 0.65m, 0.70m };
        
        foreach (var strategy in strategies)
        {
            foreach (var mult in multipliers)
            {
                foreach (var thr in thresholds)
                {
                    var bundle = new ParameterBundle
                    {
                        Strategy = strategy,
                        Mult = mult,
                        Thr = thr
                    };
                    
                    _bundleStats[bundle.BundleId] = new BundleStatistics
                    {
                        BundleId = bundle.BundleId,
                        Alpha = 1.0, // Beta distribution parameters
                        Beta = 1.0,
                        TotalSelections = 0,
                        TotalReward = 0.0
                    };
                }
            }
        }
    }
    
    /// <summary>
    /// Select parameter bundle based on market context using Thompson sampling
    /// </summary>
    public async Task<ParameterBundle> SelectParameterBundleAsync(
        MarketContext marketContext,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Use Neural UCB Extended for sophisticated bundle selection
            var selectedBundle = await _neuralUcbExtended.SelectBundleAsync(marketContext);
            
            // Update Thompson sampling statistics
            lock (_statsLock)
            {
                if (_bundleStats.ContainsKey(selectedBundle.BundleId))
                {
                    _bundleStats[selectedBundle.BundleId].TotalSelections++;
                }
            }
            
            _logger.LogDebug("Selected parameter bundle: {BundleId} for market context: Volatility={Volatility}, Trend={Trend}", 
                selectedBundle.BundleId, marketContext.Volatility, marketContext.IsTrending);
            
            return selectedBundle;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error selecting parameter bundle, using safe default");
            return ParameterBundle.CreateSafeDefault();
        }
    }
    
    /// <summary>
    /// Update bundle performance with reward feedback
    /// </summary>
    public async Task UpdateWithRewardAsync(
        ParameterBundle bundle,
        decimal reward,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Update Neural UCB Extended
            await _neuralUcbExtended.UpdateWithRewardAsync(bundle, (double)reward);
            
            // Update Thompson sampling statistics
            lock (_statsLock)
            {
                if (_bundleStats.ContainsKey(bundle.BundleId))
                {
                    var stats = _bundleStats[bundle.BundleId];
                    stats.TotalReward += (double)reward;
                    
                    // Update Beta distribution parameters
                    if (reward > 0)
                    {
                        stats.Alpha += (double)reward;
                    }
                    else
                    {
                        stats.Beta += Math.Abs((double)reward);
                    }
                }
            }
            
            _logger.LogDebug("Updated bundle {BundleId} with reward {Reward}", 
                bundle.BundleId, reward);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating bundle reward for {BundleId}", bundle.BundleId);
        }
    }
    
    /// <summary>
    /// Get performance statistics for all bundles
    /// </summary>
    public Dictionary<string, BundleStatistics> GetBundleStatistics()
    {
        lock (_statsLock)
        {
            return new Dictionary<string, BundleStatistics>(_bundleStats);
        }
    }
    
    public void Dispose()
    {
        _neuralUcbExtended?.Dispose();
        _logger.LogInformation("BanditController disposed");
    }
}

/// <summary>
/// Thompson sampling statistics for bundle performance tracking
/// </summary>
public class BundleStatistics
{
    public string BundleId { get; set; } = string.Empty;
    public double Alpha { get; set; } = 1.0;
    public double Beta { get; set; } = 1.0;
    public int TotalSelections { get; set; }
    public double TotalReward { get; set; }
    public double AverageReward => TotalSelections > 0 ? TotalReward / TotalSelections : 0.0;
    public double SuccessRate => Alpha / (Alpha + Beta);
}

/// <summary>
/// Configuration for bandit controller
/// </summary>
public class BanditControllerConfig
{
    public double ExplorationWeight { get; set; } = 0.1;
    public int InputDimension { get; set; } = 20;
    public int HiddenDimension { get; set; } = 64;
    public int MinSamplesForLearning { get; set; } = 10;
}