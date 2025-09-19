using System;
using System.Collections.Generic;
using System.Linq;
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
    /// Process trading outcome and calculate reward with advanced performance metrics
    /// Includes Sharpe ratio, maximum drawdown, win rate calculations for reward feedback
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
            
            // Calculate advanced performance metrics
            var performanceMetrics = await CalculateAdvancedPerformanceMetricsAsync(
                originalDecision.ParameterBundle.BundleId, outcome, cancellationToken);
            
            // Apply Sharpe ratio adjustment
            var sharpeAdjustedReward = ApplySharpeRatioAdjustment(baseReward, performanceMetrics);
            
            // Apply maximum drawdown penalty
            var drawdownAdjustedReward = ApplyDrawdownPenalty(sharpeAdjustedReward, performanceMetrics);
            
            // Apply win rate bonus/penalty
            var winRateAdjustedReward = ApplyWinRateAdjustment(drawdownAdjustedReward, performanceMetrics);
            
            // Apply CVaR (Conditional Value at Risk) adjustment
            var cvarAdjustedReward = ApplyCVaRAdjustment(winRateAdjustedReward, performanceMetrics, outcome);
            
            // Apply bundle-specific adjustments
            var bundleAdjustedReward = ApplyBundleSpecificAdjustments(
                cvarAdjustedReward, originalDecision.ParameterBundle, outcome);
            
            // Apply time-based decay
            var timeAdjustedReward = ApplyTimeDecay(bundleAdjustedReward, originalDecision.TimestampUtc);
            
            // Final risk adjustment
            var finalReward = ApplyRiskAdjustment(
                timeAdjustedReward, originalDecision.ParameterBundle, outcome);
            
            // Store performance metrics for future calculations
            await StorePerformanceMetricsAsync(originalDecision.ParameterBundle.BundleId, 
                performanceMetrics, outcome, cancellationToken);
            
            // Integrate with existing performance tracking
            if (_performanceTracker != null)
            {
                await _performanceTracker.UpdatePerformanceAsync(
                    originalDecision.OriginalDecision.Symbol,
                    outcome,
                    finalReward,
                    cancellationToken);
            }
            
            _logger.LogInformation("Calculated reward {Reward} for bundle {BundleId}: PnL={PnL}, Sharpe={Sharpe}, MaxDD={MaxDD}, WinRate={WinRate}, CVaR={CVaR}", 
                finalReward, originalDecision.ParameterBundle.BundleId, outcome.ProfitLoss,
                performanceMetrics.SharpeRatio, performanceMetrics.MaxDrawdown, 
                performanceMetrics.WinRate, performanceMetrics.CVaR);
            
            return finalReward;
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
    /// Calculate advanced performance metrics including Sharpe ratio, max drawdown, win rate, CVaR
    /// </summary>
    private async Task<AdvancedPerformanceMetrics> CalculateAdvancedPerformanceMetricsAsync(
        string bundleId, 
        TradingOutcome outcome,
        CancellationToken cancellationToken)
    {
        try
        {
            // Get historical performance data for this bundle
            var historicalOutcomes = await GetHistoricalOutcomesAsync(bundleId, cancellationToken);
            historicalOutcomes.Add(outcome); // Include current outcome
            
            var returns = historicalOutcomes.Select(o => o.ProfitLoss).ToList();
            var count = returns.Count;
            
            if (count < 2)
            {
                // Not enough data for meaningful calculations
                return new AdvancedPerformanceMetrics();
            }
            
            // Calculate metrics
            var avgReturn = returns.Average();
            var variance = returns.Sum(r => (r - avgReturn) * (r - avgReturn)) / (count - 1);
            var stdDev = (decimal)Math.Sqrt((double)variance);
            
            var sharpeRatio = stdDev > 0 ? avgReturn / stdDev : 0;
            var maxDrawdown = CalculateMaxDrawdown(returns);
            var winRate = (decimal)returns.Count(r => r > 0) / count;
            var cvar = CalculateCVaR(returns, 0.05m); // 5% CVaR
            
            return new AdvancedPerformanceMetrics
            {
                SharpeRatio = sharpeRatio,
                MaxDrawdown = Math.Abs(maxDrawdown),
                WinRate = winRate,
                CVaR = Math.Abs(cvar),
                AverageReturn = avgReturn,
                Volatility = stdDev,
                TradeCount = count
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating advanced performance metrics for {BundleId}", bundleId);
            return new AdvancedPerformanceMetrics();
        }
    }
    
    private decimal CalculateMaxDrawdown(List<decimal> returns)
    {
        var cumulativeReturns = new List<decimal>();
        decimal cumulative = 0;
        
        foreach (var ret in returns)
        {
            cumulative += ret;
            cumulativeReturns.Add(cumulative);
        }
        
        decimal maxDrawdown = 0;
        decimal peak = cumulativeReturns[0];
        
        foreach (var value in cumulativeReturns)
        {
            if (value > peak)
            {
                peak = value;
            }
            
            var drawdown = peak - value;
            if (drawdown > maxDrawdown)
            {
                maxDrawdown = drawdown;
            }
        }
        
        return maxDrawdown;
    }
    
    private decimal CalculateCVaR(List<decimal> returns, decimal confidenceLevel)
    {
        var sortedReturns = returns.OrderBy(r => r).ToList();
        var cutoffIndex = (int)(returns.Count * confidenceLevel);
        
        if (cutoffIndex == 0) return 0;
        
        var tailReturns = sortedReturns.Take(cutoffIndex);
        return tailReturns.Average();
    }
    
    private decimal ApplySharpeRatioAdjustment(decimal baseReward, AdvancedPerformanceMetrics metrics)
    {
        if (metrics.SharpeRatio == 0) return baseReward;
        
        // Reward high Sharpe ratios, penalize low ones
        var sharpeAdjustment = 1.0m + Math.Min(metrics.SharpeRatio / 2.0m, 0.5m);
        return baseReward * Math.Max(sharpeAdjustment, 0.5m);
    }
    
    private decimal ApplyDrawdownPenalty(decimal reward, AdvancedPerformanceMetrics metrics)
    {
        if (metrics.MaxDrawdown == 0) return reward;
        
        // Penalize high drawdowns
        var drawdownPenalty = Math.Min(metrics.MaxDrawdown / 100m, 0.5m); // Max 50% penalty
        return reward * (1.0m - drawdownPenalty);
    }
    
    private decimal ApplyWinRateAdjustment(decimal reward, AdvancedPerformanceMetrics metrics)
    {
        // Adjust based on win rate (0.5 = neutral, higher = bonus, lower = penalty)
        var winRateAdjustment = 0.5m + (metrics.WinRate - 0.5m);
        return reward * Math.Max(winRateAdjustment, 0.2m); // Minimum 20% of original reward
    }
    
    private decimal ApplyCVaRAdjustment(decimal reward, AdvancedPerformanceMetrics metrics, TradingOutcome currentOutcome)
    {
        if (metrics.CVaR == 0) return reward;
        
        // CVaR adjustment: penalize if current outcome is worse than CVaR
        if (currentOutcome.ProfitLoss < -metrics.CVaR)
        {
            var cvarPenalty = Math.Min(0.3m, Math.Abs(currentOutcome.ProfitLoss) / (metrics.CVaR * 2));
            return reward * (1.0m - cvarPenalty);
        }
        
        return reward;
    }
    
    private async Task<List<TradingOutcome>> GetHistoricalOutcomesAsync(string bundleId, CancellationToken cancellationToken)
    {
        // This would typically query a database or cache for historical outcomes
        // For now, return empty list (will be enhanced when persistence is added)
        return new List<TradingOutcome>();
    }
    
    private async Task StorePerformanceMetricsAsync(
        string bundleId, 
        AdvancedPerformanceMetrics metrics, 
        TradingOutcome outcome,
        CancellationToken cancellationToken)
    {
        try
        {
            // Store metrics for future calculations
            // This would typically persist to a database or cache
            _logger.LogDebug("Stored performance metrics for bundle {BundleId}: Sharpe={Sharpe}, MaxDD={MaxDD}", 
                bundleId, metrics.SharpeRatio, metrics.MaxDrawdown);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error storing performance metrics for {BundleId}", bundleId);
        }
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

/// <summary>
/// Advanced performance metrics for sophisticated reward calculation
/// </summary>
public class AdvancedPerformanceMetrics
{
    public decimal SharpeRatio { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal WinRate { get; set; }
    public decimal CVaR { get; set; } // Conditional Value at Risk
    public decimal AverageReturn { get; set; }
    public decimal Volatility { get; set; }
    public int TradeCount { get; set; }
    public DateTime CalculatedAt { get; set; } = DateTime.UtcNow;
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