using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;

namespace BotCore.Compatibility;

/// <summary>
/// Reward system connector for feedback and continuous learning
/// 
/// Reward System Connection: Connects trading outcomes back to the learning system
/// for continuous improvement of parameter bundle selection.
/// </summary>
public class RewardSystemConnector : IDisposable
{
    private readonly ILogger<RewardSystemConnector> _logger;
    private readonly IServiceProvider _serviceProvider;
    
    // Integration with existing feedback systems
    private readonly ITradingFeedbackService? _feedbackService;
    private readonly IPerformanceTracker? _performanceTracker;
    
    // Reward calculation configuration
    private readonly RewardSystemConfig _config;
    
    public RewardSystemConnector(ILogger<RewardSystemConnector> logger, IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        
        // Get existing services (optional dependencies)
        _feedbackService = serviceProvider.GetService<ITradingFeedbackService>();
        _performanceTracker = serviceProvider.GetService<IPerformanceTracker>();
        
        // Initialize configuration
        _config = new RewardSystemConfig();
        
        _logger.LogInformation("RewardSystemConnector initialized - Connected to existing feedback systems");
    }
    
    /// <summary>
    /// Register decision for future feedback
    /// </summary>
    public async Task RegisterDecisionForFeedbackAsync(
        EnhancedTradingDecision decision,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Register with existing feedback service if available
            if (_feedbackService != null)
            {
                await _feedbackService.RegisterDecisionAsync(
                    decision.OriginalDecision,
                    decision.ParameterBundle.BundleId,
                    cancellationToken);
                
                _logger.LogDebug("Registered decision {BundleId} with existing feedback service", 
                    decision.ParameterBundle.BundleId);
            }
            
            // Register with performance tracker if available
            if (_performanceTracker != null)
            {
                await _performanceTracker.TrackDecisionAsync(decision.OriginalDecision, cancellationToken);
                
                _logger.LogDebug("Registered decision {Symbol} with performance tracker", 
                    decision.OriginalDecision.Symbol);
            }
            
            _logger.LogDebug("Decision registered for feedback: {BundleId}", decision.ParameterBundle.BundleId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error registering decision for feedback");
        }
    }
    
    /// <summary>
    /// Process trading outcome and calculate reward
    /// </summary>
    public async Task<decimal> ProcessOutcomeAndCalculateRewardAsync(
        EnhancedTradingDecision originalDecision,
        TradingOutcome outcome,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Calculate base reward from outcome
            var baseReward = CalculateBaseReward(outcome);
            
            // Apply bundle-specific adjustments
            var bundleAdjustedReward = ApplyBundleSpecificAdjustments(
                baseReward, originalDecision.ParameterBundle, outcome);
            
            // Apply time-based decay
            var timeAdjustedReward = ApplyTimeDecay(bundleAdjustedReward, originalDecision.TimestampUtc);
            
            // Apply risk-adjusted scoring
            var riskAdjustedReward = ApplyRiskAdjustment(
                timeAdjustedReward, originalDecision.ParameterBundle, outcome);
            
            // Integrate with existing performance tracking
            if (_performanceTracker != null)
            {
                await _performanceTracker.UpdatePerformanceAsync(
                    originalDecision.OriginalDecision.Symbol,
                    outcome,
                    riskAdjustedReward,
                    cancellationToken);
            }
            
            _logger.LogDebug("Calculated reward {Reward} for bundle {BundleId}: PnL={PnL}", 
                riskAdjustedReward, originalDecision.ParameterBundle.BundleId, outcome.ProfitLoss);
            
            return riskAdjustedReward;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing outcome and calculating reward");
            return 0; // Neutral reward on error
        }
    }
    
    private decimal CalculateBaseReward(TradingOutcome outcome)
    {
        // Base reward calculation based on profit/loss
        var profitRatio = outcome.ProfitLoss / Math.Max(outcome.RiskAmount, 1m);
        
        return outcome.ProfitLoss switch
        {
            > 0 => Math.Min(profitRatio * _config.ProfitMultiplier, _config.MaxReward),
            < 0 => Math.Max(profitRatio * _config.LossMultiplier, _config.MinReward),
            _ => 0m
        };
    }
    
    private decimal ApplyBundleSpecificAdjustments(
        decimal baseReward,
        ParameterBundle bundle,
        TradingOutcome outcome)
    {
        decimal adjustment = 1.0m;
        
        // Strategy-specific adjustments
        adjustment *= bundle.Strategy switch
        {
            "S2" => _config.StrategyMultipliers.GetValueOrDefault("S2", 1.0m),
            "S3" => _config.StrategyMultipliers.GetValueOrDefault("S3", 1.0m),
            "S6" => _config.StrategyMultipliers.GetValueOrDefault("S6", 1.0m),
            "S11" => _config.StrategyMultipliers.GetValueOrDefault("S11", 1.0m),
            _ => 1.0m
        };
        
        // Multiplier-based adjustments (higher risk should be rewarded more for success)
        if (outcome.ProfitLoss > 0)
        {
            adjustment *= bundle.Mult; // Reward higher multipliers more for success
        }
        else
        {
            adjustment /= bundle.Mult; // Penalize higher multipliers more for failure
        }
        
        // Confidence threshold adjustments
        var confidenceAdjustment = bundle.Thr switch
        {
            >= 0.70m => 1.1m, // Reward high confidence requirements
            <= 0.60m => 0.9m, // Slightly penalize low confidence requirements
            _ => 1.0m
        };
        adjustment *= confidenceAdjustment;
        
        return baseReward * adjustment;
    }
    
    private decimal ApplyTimeDecay(decimal reward, DateTime decisionTime)
    {
        var timeElapsed = DateTime.UtcNow - decisionTime;
        var decayHours = (decimal)timeElapsed.TotalHours;
        
        // Apply exponential decay for delayed rewards
        var decayFactor = (decimal)Math.Exp(-decayHours / _config.RewardDecayHours);
        
        return reward * Math.Max(decayFactor, _config.MinDecayFactor);
    }
    
    private decimal ApplyRiskAdjustment(
        decimal reward,
        ParameterBundle bundle,
        TradingOutcome outcome)
    {
        // Risk-adjusted scoring based on Sharpe ratio concept
        var riskAdjustedReturn = outcome.ProfitLoss / Math.Max(outcome.VolatilityMeasure, 0.01m);
        
        // Apply risk scaling
        var riskScale = Math.Min(Math.Abs(riskAdjustedReturn), _config.MaxRiskScale);
        
        return reward * (1.0m + riskScale * _config.RiskAdjustmentFactor);
    }
    
    /// <summary>
    /// Get reward statistics for analysis
    /// </summary>
    public async Task<RewardStatistics> GetRewardStatisticsAsync(
        string bundleId,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // This would typically query a database or cache
            // For now, return basic statistics
            
            return new RewardStatistics
            {
                BundleId = bundleId,
                TotalRewards = 0,
                AverageReward = 0,
                RewardCount = 0,
                LastUpdated = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting reward statistics for {BundleId}", bundleId);
            throw;
        }
    }
    
    public void Dispose()
    {
        _logger.LogInformation("RewardSystemConnector disposed");
    }
}

/// <summary>
/// Trading outcome for reward calculation
/// </summary>
public class TradingOutcome
{
    public decimal ProfitLoss { get; set; }
    public decimal RiskAmount { get; set; }
    public decimal VolatilityMeasure { get; set; }
    public TimeSpan Duration { get; set; }
    public bool WasSuccessful => ProfitLoss > 0;
    public DateTime OutcomeTime { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Performance tracking for parameter combinations
/// </summary>
public class ParameterCombinationPerformance
{
    public string BundleId { get; }
    public int DecisionCount { get; private set; }
    public int SuccessfulDecisions { get; private set; }
    public decimal TotalReward { get; private set; }
    public decimal AverageReward => DecisionCount > 0 ? TotalReward / DecisionCount : 0;
    public double SuccessRate => DecisionCount > 0 ? (double)SuccessfulDecisions / DecisionCount : 0;
    public DateTime LastUpdated { get; private set; } = DateTime.UtcNow;
    
    public ParameterCombinationPerformance(string bundleId)
    {
        BundleId = bundleId;
    }
    
    public void RecordDecision(EnhancedTradingDecision decision)
    {
        DecisionCount++;
        LastUpdated = DateTime.UtcNow;
    }
    
    public void RecordOutcome(TradingOutcome outcome)
    {
        if (outcome.WasSuccessful)
        {
            SuccessfulDecisions++;
        }
        
        TotalReward += outcome.ProfitLoss;
        LastUpdated = DateTime.UtcNow;
    }
}

/// <summary>
/// Reward statistics for analysis
/// </summary>
public class RewardStatistics
{
    public string BundleId { get; set; } = string.Empty;
    public decimal TotalRewards { get; set; }
    public decimal AverageReward { get; set; }
    public int RewardCount { get; set; }
    public DateTime LastUpdated { get; set; }
}

/// <summary>
/// Configuration for reward system
/// </summary>
public class RewardSystemConfig
{
    public decimal ProfitMultiplier { get; set; } = 1.0m;
    public decimal LossMultiplier { get; set; } = 1.5m; // Penalize losses more
    public decimal MaxReward { get; set; } = 10.0m;
    public decimal MinReward { get; set; } = -10.0m;
    public decimal RewardDecayHours { get; set; } = 24.0m; // 24-hour half-life
    public decimal MinDecayFactor { get; set; } = 0.1m;
    public decimal MaxRiskScale { get; set; } = 2.0m;
    public decimal RiskAdjustmentFactor { get; set; } = 0.1m;
    
    public Dictionary<string, decimal> StrategyMultipliers { get; set; } = new()
    {
        ["S2"] = 1.0m,
        ["S3"] = 1.0m,
        ["S6"] = 1.0m,
        ["S11"] = 1.0m
    };
}

// Placeholder interfaces for existing services
public interface ITradingFeedbackService
{
    Task RegisterDecisionAsync(TradingDecision decision, string bundleId, CancellationToken cancellationToken = default);
}

public interface IPerformanceTracker
{
    Task TrackDecisionAsync(TradingDecision decision, CancellationToken cancellationToken = default);
    Task UpdatePerformanceAsync(string symbol, TradingOutcome outcome, decimal reward, CancellationToken cancellationToken = default);
}