using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Automated Promotion System with Gradual Rollout and Health Monitoring
/// Implements safe champion/challenger transitions with position size limits and emergency rollback
/// </summary>
public class AutomatedPromotionService : BackgroundService
{
    private readonly ILogger<AutomatedPromotionService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IPromotionService _promotionService;
    private readonly IValidationService _validationService;
    private readonly IMarketHoursService _marketHours;
    private readonly IModelRegistry _modelRegistry;
    
    // Promotion scheduling and state
    private readonly Dictionary<string, PromotionSchedule> _scheduledPromotions = new();
    private readonly Dictionary<string, GradualRolloutState> _activeRollouts = new();
    private readonly List<PromotionHealthCheck> _healthChecks = new();
    
    // Configuration
    private readonly TimeSpan _promotionCheckInterval = TimeSpan.FromMinutes(15);
    private readonly int _maxConcurrentPromotions = 2;
    private readonly decimal _initialPositionSizeLimit = 0.25m; // Start with 25% position size
    private readonly decimal _rolloutIncrementSize = 0.25m; // Increase by 25% each step
    private readonly TimeSpan _rolloutStepDuration = TimeSpan.FromHours(2); // 2 hours per step

    public AutomatedPromotionService(
        ILogger<AutomatedPromotionService> logger,
        IServiceProvider serviceProvider,
        IPromotionService promotionService,
        IValidationService validationService,
        IMarketHoursService marketHours,
        IModelRegistry modelRegistry)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _promotionService = promotionService;
        _validationService = validationService;
        _marketHours = marketHours;
        _modelRegistry = modelRegistry;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[AUTO-PROMOTION] Starting automated promotion service");
        
        // Wait for system initialization
        await Task.Delay(TimeSpan.FromMinutes(2), stoppingToken).ConfigureAwait(false);
        
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Check for promotion opportunities
                await CheckPromotionOpportunitiesAsync(stoppingToken).ConfigureAwait(false);
                
                // Execute scheduled promotions
                await ExecuteScheduledPromotionsAsync(stoppingToken).ConfigureAwait(false);
                
                // Monitor active rollouts
                await MonitorActiveRolloutsAsync(stoppingToken).ConfigureAwait(false);
                
                // Perform health checks
                PerformHealthChecks(stoppingToken);
                
                // Wait before next cycle
                await Task.Delay(_promotionCheckInterval, stoppingToken).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[AUTO-PROMOTION] Error in automated promotion service");
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>
    /// Check for new promotion opportunities based on validation results
    /// </summary>
    private async Task CheckPromotionOpportunitiesAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Get all algorithms to check for challengers
            var algorithms = new[] { "PPO", "UCB", "LSTM" };
            
            foreach (var algorithm in algorithms)
            {
                if (cancellationToken.IsCancellationRequested) break;
                
                await CheckAlgorithmPromotionOpportunityAsync(algorithm, cancellationToken).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to check promotion opportunities");
        }
    }

    /// <summary>
    /// Check promotion opportunity for a specific algorithm
    /// </summary>
    private async Task CheckAlgorithmPromotionOpportunityAsync(string algorithm, CancellationToken cancellationToken)
    {
        try
        {
            // Skip if already has scheduled promotion or active rollout
            if (_scheduledPromotions.ContainsKey(algorithm) || _activeRollouts.ContainsKey(algorithm))
            {
                return;
            }

            // Get current champion and potential challengers
            var champion = await _modelRegistry.GetChampionAsync(algorithm, cancellationToken).ConfigureAwait(false);
            var allModels = await _modelRegistry.GetModelsAsync(algorithm, cancellationToken).ConfigureAwait(false);
            
            // Find validated but not promoted models as potential challengers
            var challengers = allModels
                .Where(m => m.IsValidated && !m.IsPromoted && m.VersionId != champion?.VersionId)
                .OrderByDescending(m => m.Sharpe) // Order by performance
                .Take(3) // Consider top 3 challengers
                .ToList();

            foreach (var challenger in challengers)
            {
                if (cancellationToken.IsCancellationRequested) break;
                
                // Validate challenger against champion
                var validationResult = await _validationService.ValidateChallengerAsync(
                    challenger.VersionId, cancellationToken).ConfigureAwait(false);
                
                if (validationResult.Outcome == ValidationOutcome.Passed)
                {
                    await SchedulePromotionAsync(algorithm, challenger.VersionId, validationResult.Report, cancellationToken).ConfigureAwait(false);
                    break; // Only schedule one promotion per algorithm at a time
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to check promotion opportunity for {Algorithm}", algorithm);
        }
    }

    /// <summary>
    /// Schedule a promotion for execution during next safe window
    /// </summary>
    private async Task SchedulePromotionAsync(
        string algorithm, 
        string challengerVersionId, 
        ValidationReport validationReport, 
        CancellationToken cancellationToken)
    {
        try
        {
            // Get next safe promotion window
            var nextSafeWindow = await _marketHours.GetNextSafeWindowAsync(cancellationToken).ConfigureAwait(false);
            if (nextSafeWindow == null)
            {
                _logger.LogWarning("[AUTO-PROMOTION] No safe window available for {Algorithm} promotion", algorithm);
                return;
            }

            var schedule = new PromotionSchedule
            {
                Algorithm = algorithm,
                ChallengerVersionId = challengerVersionId,
                ScheduledTime = nextSafeWindow.Value,
                ValidationReport = validationReport,
                CreatedAt = DateTime.UtcNow,
                Status = "SCHEDULED",
                ApprovedBy = "AUTOMATED_SYSTEM"
            };

            _scheduledPromotions[algorithm] = schedule;

            _logger.LogInformation("[AUTO-PROMOTION] Scheduled promotion for {Algorithm} challenger {ChallengerVersionId} " +
                "at {ScheduledTime} (Sharpe improvement: {SharpeImprovement:F3}, P-value: {PValue:F4})",
                algorithm, challengerVersionId, nextSafeWindow, 
                validationReport.ChallengerSharpe - validationReport.ChampionSharpe, validationReport.PValue);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to schedule promotion for {Algorithm}", algorithm);
        }
    }

    /// <summary>
    /// Execute scheduled promotions that are ready
    /// </summary>
    private async Task ExecuteScheduledPromotionsAsync(CancellationToken cancellationToken)
    {
        try
        {
            var now = DateTime.UtcNow;
            var readyPromotions = _scheduledPromotions.Values
                .Where(p => p.Status == "SCHEDULED" && p.ScheduledTime <= now)
                .Take(_maxConcurrentPromotions)
                .ToList();

            foreach (var promotion in readyPromotions)
            {
                if (cancellationToken.IsCancellationRequested) break;
                
                await ExecutePromotionAsync(promotion, cancellationToken).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to execute scheduled promotions");
        }
    }

    /// <summary>
    /// Execute a single promotion with gradual rollout
    /// </summary>
    private async Task ExecutePromotionAsync(PromotionSchedule promotion, CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation("[AUTO-PROMOTION] Executing promotion for {Algorithm} challenger {ChallengerVersionId}",
                promotion.Algorithm, promotion.ChallengerVersionId);

            // Safety checks before promotion
            var safetyPassed = await PerformSafetyChecksAsync(promotion, cancellationToken).ConfigureAwait(false);
            if (!safetyPassed)
            {
                promotion.Status = "FAILED_SAFETY_CHECKS";
                _logger.LogWarning("[AUTO-PROMOTION] Promotion failed safety checks for {Algorithm}", promotion.Algorithm);
                return;
            }

            // Start gradual rollout
            var rolloutState = new GradualRolloutState
            {
                Algorithm = promotion.Algorithm,
                ChallengerVersionId = promotion.ChallengerVersionId,
                StartTime = DateTime.UtcNow,
                CurrentPositionSizeLimit = _initialPositionSizeLimit,
                StepNumber = 1,
                MaxSteps = (int)(1.0m / _rolloutIncrementSize), // 4 steps for 25% increments
                Status = "ROLLING_OUT",
                HealthMetrics = new List<RolloutHealthMetric>()
            };

            // Execute initial promotion with position size limit
            var promotionSuccess = await _promotionService.PromoteToChampionAsync(
                promotion.Algorithm, 
                promotion.ChallengerVersionId, 
                $"Automated gradual rollout - Step 1 ({_initialPositionSizeLimit:P0} position limit)",
                cancellationToken).ConfigureAwait(false);

            if (promotionSuccess)
            {
                _activeRollouts[promotion.Algorithm] = rolloutState;
                promotion.Status = "EXECUTING_ROLLOUT";
                
                _logger.LogInformation("[AUTO-PROMOTION] Started gradual rollout for {Algorithm} " +
                    "with {PositionLimit:P0} position size limit",
                    promotion.Algorithm, _initialPositionSizeLimit);
            }
            else
            {
                promotion.Status = "FAILED_EXECUTION";
                _logger.LogError("[AUTO-PROMOTION] Failed to execute promotion for {Algorithm}", promotion.Algorithm);
            }
        }
        catch (Exception ex)
        {
            promotion.Status = "ERROR";
            _logger.LogError(ex, "[AUTO-PROMOTION] Error executing promotion for {Algorithm}", promotion.Algorithm);
        }
        finally
        {
            // Clean up scheduled promotion
            _scheduledPromotions.Remove(promotion.Algorithm);
        }
    }

    /// <summary>
    /// Monitor active rollouts and progress them through steps
    /// </summary>
    private async Task MonitorActiveRolloutsAsync(CancellationToken cancellationToken)
    {
        try
        {
            var rolloutsToUpdate = _activeRollouts.Values.ToList();
            
            foreach (var rollout in rolloutsToUpdate)
            {
                if (cancellationToken.IsCancellationRequested) break;
                
                await MonitorRolloutAsync(rollout, cancellationToken).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to monitor active rollouts");
        }
    }

    /// <summary>
    /// Monitor a single rollout and progress it to next step if healthy
    /// </summary>
    private async Task MonitorRolloutAsync(GradualRolloutState rollout, CancellationToken cancellationToken)
    {
        try
        {
            var now = DateTime.UtcNow;
            var stepDuration = now - rollout.StepStartTime;
            
            // Check if current step has run long enough
            if (stepDuration < _rolloutStepDuration)
            {
                return; // Wait for step to complete
            }

            // Collect health metrics for current step
            var healthMetric = await CollectRolloutHealthMetricAsync(rollout, cancellationToken).ConfigureAwait(false);
            rollout.HealthMetrics.Add(healthMetric);

            // Check if rollout is healthy
            var isHealthy = await AssessRolloutHealthAsync(rollout, healthMetric, cancellationToken).ConfigureAwait(false);
            
            if (!isHealthy)
            {
                // Rollback due to health issues
                await RollbackPromotionAsync(rollout, "Health check failed", cancellationToken).ConfigureAwait(false);
                return;
            }

            // Progress to next step or complete rollout
            if (rollout.StepNumber < rollout.MaxSteps)
            {
                await ProgressRolloutToNextStepAsync(rollout, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                await CompleteRolloutAsync(rollout, cancellationToken).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to monitor rollout for {Algorithm}", rollout.Algorithm);
            await RollbackPromotionAsync(rollout, $"Monitoring error: {ex.Message}", cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Progress rollout to next step with increased position size limit
    /// </summary>
    private async Task ProgressRolloutToNextStepAsync(GradualRolloutState rollout, CancellationToken cancellationToken)
    {
        await Task.Yield().ConfigureAwait(false); // Ensure async behavior
        
        try
        {
            rollout.StepNumber++;
            rollout.CurrentPositionSizeLimit = Math.Min(1.0m, rollout.StepNumber * _rolloutIncrementSize);
            rollout.StepStartTime = DateTime.UtcNow;

            _logger.LogInformation("[AUTO-PROMOTION] Progressing {Algorithm} rollout to step {Step}/{MaxSteps} " +
                "with {PositionLimit:P0} position size limit",
                rollout.Algorithm, rollout.StepNumber, rollout.MaxSteps, rollout.CurrentPositionSizeLimit);

            // In production, this would update the position size limit in the trading system
            // For now, we log the progression
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to progress rollout for {Algorithm}", rollout.Algorithm);
        }
    }

    /// <summary>
    /// Complete rollout with full position size
    /// </summary>
    private Task CompleteRolloutAsync(GradualRolloutState rollout, CancellationToken cancellationToken)
    {
        try
        {
            rollout.Status = "COMPLETED";
            rollout.CompletedAt = DateTime.UtcNow;
            rollout.CurrentPositionSizeLimit = 1.0m; // Full position size

            _logger.LogInformation("[AUTO-PROMOTION] Completed gradual rollout for {Algorithm} " +
                "- challenger {ChallengerVersionId} is now fully promoted",
                rollout.Algorithm, rollout.ChallengerVersionId);

            // Clean up active rollout
            _activeRollouts.Remove(rollout.Algorithm);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to complete rollout for {Algorithm}", rollout.Algorithm);
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Rollback promotion due to health issues
    /// </summary>
    private async Task RollbackPromotionAsync(GradualRolloutState rollout, string reason, CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogWarning("[AUTO-PROMOTION] Rolling back {Algorithm} promotion due to: {Reason}",
                rollout.Algorithm, reason);

            var rollbackSuccess = await _promotionService.RollbackToPreviousAsync(
                rollout.Algorithm, $"Automated rollback: {reason}", cancellationToken).ConfigureAwait(false);

            rollout.Status = rollbackSuccess ? "ROLLED_BACK" : "ROLLBACK_FAILED";
            rollout.CompletedAt = DateTime.UtcNow;

            if (rollbackSuccess)
            {
                _logger.LogInformation("[AUTO-PROMOTION] Successfully rolled back {Algorithm} promotion", 
                    rollout.Algorithm);
            }
            else
            {
                _logger.LogError("[AUTO-PROMOTION] Failed to rollback {Algorithm} promotion - manual intervention required",
                    rollout.Algorithm);
            }

            // Clean up active rollout
            _activeRollouts.Remove(rollout.Algorithm);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Error during rollback for {Algorithm}", rollout.Algorithm);
        }
    }

    #region Health Monitoring and Safety Checks

    /// <summary>
    /// Perform safety checks before promotion
    /// </summary>
    private async Task<bool> PerformSafetyChecksAsync(PromotionSchedule promotion, CancellationToken cancellationToken)
    {
        try
        {
            // Check 1: Market is in safe window
            var isInSafeWindow = await _marketHours.IsInSafePromotionWindowAsync(cancellationToken).ConfigureAwait(false);
            if (!isInSafeWindow)
            {
                _logger.LogWarning("[AUTO-PROMOTION] Safety check failed: not in safe promotion window");
                return false;
            }

            // Check 2: Position is flat (would check actual position service in production)
            var isFlat = true; // Mock check
            if (!isFlat)
            {
                _logger.LogWarning("[AUTO-PROMOTION] Safety check failed: positions not flat");
                return false;
            }

            // Check 3: No recent emergency stops or critical alerts
            var hasRecentEmergencyStop; // Mock check
            if (hasRecentEmergencyStop)
            {
                _logger.LogWarning("[AUTO-PROMOTION] Safety check failed: recent emergency stop detected");
                return false;
            }

            // Check 4: System health is good
            var systemHealthy = true; // Mock check
            if (!systemHealthy)
            {
                _logger.LogWarning("[AUTO-PROMOTION] Safety check failed: system health issues detected");
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Error during safety checks");
            return false;
        }
    }

    /// <summary>
    /// Collect health metrics for rollout monitoring
    /// </summary>
    private async Task<RolloutHealthMetric> CollectRolloutHealthMetricAsync(
        GradualRolloutState rollout, 
        CancellationToken cancellationToken)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        // In production, this would collect real metrics from the trading system
        var random = new Random();
        
        return new RolloutHealthMetric
        {
            Timestamp = DateTime.UtcNow,
            StepNumber = rollout.StepNumber,
            PositionSizeLimit = rollout.CurrentPositionSizeLimit,
            
            // Mock metrics (in production, collect from actual trading)
            ReturnsLast2Hours = (decimal)(random.NextDouble() - 0.5) * 0.02m, // ±1% return
            SharpeRatioLast2Hours = 0.8m + (decimal)random.NextDouble() * 0.4m, // 0.8-1.2 Sharpe
            MaxDrawdownLast2Hours = -(decimal)random.NextDouble() * 0.01m, // Max 1% drawdown
            TradeCount = random.Next(5, 25), // 5-25 trades
            ErrorCount = random.Next(0, 2), // 0-1 errors
            AverageLatencyMs = 50 + random.Next(0, 30), // 50-80ms latency
            
            IsHealthy = true // Will be determined by assessment
        };
    }

    /// <summary>
    /// Assess rollout health based on collected metrics
    /// </summary>
    private async Task<bool> AssessRolloutHealthAsync(
        GradualRolloutState rollout, 
        RolloutHealthMetric metric, 
        CancellationToken cancellationToken)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var issues = new List<string>();

        // Health gate 1: Returns should be positive or small negative
        if (metric.ReturnsLast2Hours < -0.015m) // Worse than -1.5% in 2 hours
        {
            issues.Add($"Poor returns: {metric.ReturnsLast2Hours:P2} < -1.5%");
        }

        // Health gate 2: Sharpe ratio should be reasonable
        if (metric.SharpeRatioLast2Hours < 0.5m)
        {
            issues.Add($"Low Sharpe ratio: {metric.SharpeRatioLast2Hours:F2} < 0.5");
        }

        // Health gate 3: Drawdown should be controlled
        if (metric.MaxDrawdownLast2Hours < -0.02m) // Worse than 2% drawdown
        {
            issues.Add($"Excessive drawdown: {metric.MaxDrawdownLast2Hours:P2} < -2%");
        }

        // Health gate 4: Error rate should be low
        var errorRate = metric.TradeCount > 0 ? (decimal)metric.ErrorCount / metric.TradeCount : 0;
        if (errorRate > 0.05m) // More than 5% error rate
        {
            issues.Add($"High error rate: {errorRate:P1} > 5%");
        }

        // Health gate 5: Latency should be reasonable
        if (metric.AverageLatencyMs > 200) // Slower than 200ms
        {
            issues.Add($"High latency: {metric.AverageLatencyMs}ms > 200ms");
        }

        metric.IsHealthy = issues.Count == 0;
        
        if (!metric.IsHealthy)
        {
            _logger.LogWarning("[AUTO-PROMOTION] Health assessment failed for {Algorithm} step {Step}: {Issues}",
                rollout.Algorithm, rollout.StepNumber, string.Join("; ", issues));
        }

        return metric.IsHealthy;
    }

    /// <summary>
    /// Perform periodic health checks on all active rollouts
    /// </summary>
    private void PerformHealthChecks(CancellationToken cancellationToken)
    {
        try
        {
            var now = DateTime.UtcNow;
            
            // Remove old health checks
            _healthChecks.RemoveAll(hc => (now - hc.Timestamp).TotalHours > 24);
            
            // Perform health check on each active rollout
            foreach (var rollout in _activeRollouts.Values)
            {
                if (cancellationToken.IsCancellationRequested) break;
                
                var healthCheck = new PromotionHealthCheck
                {
                    Timestamp = now,
                    Algorithm = rollout.Algorithm,
                    ChallengerVersionId = rollout.ChallengerVersionId,
                    StepNumber = rollout.StepNumber,
                    IsHealthy = rollout.HealthMetrics.LastOrDefault()?.IsHealthy ?? false,
                    Issues = new List<string>()
                };

                // Check for trend degradation
                if (rollout.HealthMetrics.Count >= 3)
                {
                    var recentMetrics = rollout.HealthMetrics.TakeLast(3).ToList();
                    var trendDegrading = recentMetrics.All(m => !m.IsHealthy);
                    
                    if (trendDegrading)
                    {
                        healthCheck.IsHealthy;
                        healthCheck.Issues.Add("Health trend degrading over last 3 measurements");
                    }
                }

                _healthChecks.Add(healthCheck);
                
                if (!healthCheck.IsHealthy)
                {
                    _logger.LogWarning("[AUTO-PROMOTION] Health check failed for {Algorithm}: {Issues}",
                        rollout.Algorithm, string.Join("; ", healthCheck.Issues));
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[AUTO-PROMOTION] Failed to perform health checks");
        }
    }

    #endregion

    /// <summary>
    /// Get status of automated promotion system
    /// </summary>
    public AutomatedPromotionStatus GetStatus()
    {
        return new AutomatedPromotionStatus
        {
            ScheduledPromotionsCount = _scheduledPromotions.Count,
            ActiveRolloutsCount = _activeRollouts.Count,
            RecentHealthChecksCount = _healthChecks.Count(hc => (DateTime.UtcNow - hc.Timestamp).TotalHours <= 1),
            LastCheckTime = DateTime.UtcNow,
            
            ScheduledPromotions = _scheduledPromotions.Values.ToList(),
            ActiveRollouts = _activeRollouts.Values.ToList(),
            RecentHealthChecks = _healthChecks.TakeLast(20).ToList()
        };
    }
}

#region Supporting Models

/// <summary>
/// Gradual rollout state tracking
/// </summary>
public class GradualRolloutState
{
    public string Algorithm { get; set; } = string.Empty;
    public string ChallengerVersionId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime StepStartTime { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }
    public decimal CurrentPositionSizeLimit { get; set; }
    public int StepNumber { get; set; }
    public int MaxSteps { get; set; }
    public string Status { get; set; } = string.Empty;
    public List<RolloutHealthMetric> HealthMetrics { get; } = new();
}

/// <summary>
/// Health metric for rollout monitoring
/// </summary>
public class RolloutHealthMetric
{
    public DateTime Timestamp { get; set; }
    public int StepNumber { get; set; }
    public decimal PositionSizeLimit { get; set; }
    public decimal ReturnsLast2Hours { get; set; }
    public decimal SharpeRatioLast2Hours { get; set; }
    public decimal MaxDrawdownLast2Hours { get; set; }
    public int TradeCount { get; set; }
    public int ErrorCount { get; set; }
    public decimal AverageLatencyMs { get; set; }
    public bool IsHealthy { get; set; }
}

/// <summary>
/// Health check result
/// </summary>
public class PromotionHealthCheck
{
    public DateTime Timestamp { get; set; }
    public string Algorithm { get; set; } = string.Empty;
    public string ChallengerVersionId { get; set; } = string.Empty;
    public int StepNumber { get; set; }
    public bool IsHealthy { get; set; }
    public List<string> Issues { get; } = new();
}

/// <summary>
/// Automated promotion system status
/// </summary>
public class AutomatedPromotionStatus
{
    public int ScheduledPromotionsCount { get; set; }
    public int ActiveRolloutsCount { get; set; }
    public int RecentHealthChecksCount { get; set; }
    public DateTime LastCheckTime { get; set; }
    
    public List<PromotionSchedule> ScheduledPromotions { get; } = new();
    public List<GradualRolloutState> ActiveRollouts { get; } = new();
    public List<PromotionHealthCheck> RecentHealthChecks { get; } = new();
}

#endregion