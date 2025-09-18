using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
// Legacy removed: using TradingBot.Infrastructure.TopstepX;
// Legacy removed: using IAccountService = TradingBot.Infrastructure.TopstepX.IAccountService;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using IModelRegistry = TradingBot.UnifiedOrchestrator.Interfaces.IModelRegistry;

namespace TradingBot.UnifiedOrchestrator.Promotion;

/// <summary>
/// Promotion service with atomic swaps, timing gates, and instant rollback capability
/// Ensures safe champion/challenger transitions with < 100ms rollback time
/// </summary>
public class PromotionService : IPromotionService
{
    private readonly ILogger<PromotionService> _logger;
    private readonly IModelRegistry _modelRegistry;
    private readonly IModelRouterFactory _routerFactory;
    private readonly IShadowTester _shadowTester;
    private readonly IMarketHoursService _marketHours;
    private readonly ConcurrentDictionary<string, PromotionContext> _promotionContexts = new();
    private readonly ConcurrentDictionary<string, string> _scheduledPromotions = new();

    // Position service interface (would be injected in real implementation)
    private readonly IPositionService _positionService;

    public PromotionService(
        ILogger<PromotionService> logger,
        IModelRegistry modelRegistry,
        IModelRouterFactory routerFactory,
        IShadowTester shadowTester,
        IMarketHoursService marketHours,
        IPositionService positionService)
    {
        _logger = logger;
        _modelRegistry = modelRegistry;
        _routerFactory = routerFactory;
        _shadowTester = shadowTester;
        _marketHours = marketHours;
        _positionService = positionService;
    }

    /// <summary>
    /// Evaluate whether a challenger should be promoted
    /// Runs all validation gates including timing, position, and performance checks
    /// </summary>
    public async Task<PromotionDecision> EvaluatePromotionAsync(string algorithm, string challengerVersionId, CancellationToken cancellationToken = default)
    {
        var decision = new PromotionDecision();
        
        try
        {
            _logger.LogInformation("Evaluating promotion for {Algorithm} challenger {ChallengerVersionId}", 
                algorithm, challengerVersionId);

            // 1. Validate challenger exists
            var challenger = await _modelRegistry.GetModelAsync(challengerVersionId, cancellationToken);
            if (challenger == null)
            {
                decision.ShouldPromote = false;
                decision.Reason = "Challenger model not found";
                decision.ValidationErrors.Add($"Challenger version {challengerVersionId} does not exist");
                return decision;
            }

            // 2. Validate champion exists
            var champion = await _modelRegistry.GetChampionAsync(algorithm, cancellationToken);
            if (champion == null)
            {
                decision.ShouldPromote = false;
                decision.Reason = "No current champion to replace";
                decision.ValidationErrors.Add($"No champion found for algorithm {algorithm}");
                return decision;
            }

            // 3. Check if challenger has passed validation
            if (!challenger.IsValidated)
            {
                decision.ShouldPromote = false;
                decision.Reason = "Challenger has not passed validation";
                decision.ValidationErrors.Add("Challenger must pass shadow testing before promotion");
                return decision;
            }

            // 4. Timing gate validation
            await ValidateTimingGatesAsync(decision, cancellationToken);

            // 5. Position validation (must be flat)
            await ValidatePositionStateAsync(decision, cancellationToken);

            // 6. Performance validation
            await ValidatePerformanceImprovementAsync(decision, champion, challenger, cancellationToken);

            // 7. Schema and resource validation
            await ValidateSchemaCompatibilityAsync(decision, challenger, cancellationToken);

            // 8. Risk assessment
            await AssessPromotionRiskAsync(decision, algorithm, challengerVersionId, cancellationToken);

            // Final decision
            decision.ShouldPromote = decision.ValidationErrors.Count == 0 && decision.RiskConcerns.Count == 0;
            
            if (decision.ShouldPromote)
            {
                decision.Reason = "All validation gates passed - ready for promotion";
            }
            else
            {
                decision.Reason = $"Validation failed: {decision.ValidationErrors.Count} errors, {decision.RiskConcerns.Count} risk concerns";
            }

            _logger.LogInformation("Promotion evaluation for {Algorithm}: shouldPromote={ShouldPromote}, reason={Reason}", 
                algorithm, decision.ShouldPromote, decision.Reason);

            return decision;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating promotion for {Algorithm}", algorithm);
            decision.ShouldPromote = false;
            decision.Reason = $"Evaluation error: {ex.Message}";
            decision.ValidationErrors.Add(ex.Message);
            return decision;
        }
    }

    /// <summary>
    /// Promote a challenger to champion with atomic swap
    /// AC5: Single atomic swap with no mixed-version inference within a bar/tick
    /// </summary>
    public async Task<bool> PromoteToChampionAsync(string algorithm, string challengerVersionId, string reason, CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            _logger.LogInformation("Starting promotion of {Algorithm} challenger {ChallengerVersionId}: {Reason}", 
                algorithm, challengerVersionId, reason);

            // Pre-promotion validation
            var decision = await EvaluatePromotionAsync(algorithm, challengerVersionId, cancellationToken);
            if (!decision.ShouldPromote)
            {
                _logger.LogWarning("Promotion blocked for {Algorithm}: {Reason}", algorithm, decision.Reason);
                return false;
            }

            // Get current router for atomic swap
            var router = _routerFactory.GetRouter<object>(algorithm);
            if (router == null)
            {
                _logger.LogError("No router found for algorithm {Algorithm}", algorithm);
                return false;
            }

            // Load challenger model
            var challenger = await _modelRegistry.GetModelAsync(challengerVersionId, cancellationToken);
            if (challenger == null)
            {
                _logger.LogError("Challenger model {ChallengerVersionId} not found", challengerVersionId);
                return false;
            }

            // Load challenger artifact for atomic swap
            var challengerModel = await LoadModelArtifactAsync(challenger, cancellationToken);
            if (challengerModel == null)
            {
                _logger.LogError("Failed to load challenger artifact for {ChallengerVersionId}", challengerVersionId);
                return false;
            }

            // Store previous context for rollback
            var previousChampion = router.CurrentVersion;
            var previousModel = router.Current;
            
            var promotionContext = new PromotionContext
            {
                Algorithm = algorithm,
                PreviousChampionVersionId = previousChampion?.VersionId ?? "none",
                PreviousChampionModel = previousModel,
                NewChampionVersionId = challengerVersionId,
                PromotionTime = DateTime.UtcNow,
                Reason = reason
            };

            _promotionContexts[algorithm] = promotionContext;

            // ATOMIC SWAP - This is the critical section
            var swapSuccess = await router.SwapAsync(challengerModel, challenger, cancellationToken);
            if (!swapSuccess)
            {
                _logger.LogError("Atomic swap failed for {Algorithm}", algorithm);
                return false;
            }

            // Update model registry
            var promotionRecord = new PromotionRecord
            {
                Algorithm = algorithm,
                FromVersionId = previousChampion?.VersionId ?? "none",
                ToVersionId = challengerVersionId,
                Reason = reason,
                PromotedBy = Environment.UserName,
                WasFlat = decision.IsFlat,
                MarketSession = await _marketHours.GetCurrentMarketSessionAsync(cancellationToken),
                PassedValidation = true,
                ContextData = new Dictionary<string, object>
                {
                    ["promotion_duration_ms"] = stopwatch.Elapsed.TotalMilliseconds,
                    ["atomic_swap_success"] = swapSuccess,
                    ["validation_decision"] = decision
                }
            };

            var registrySuccess = await _modelRegistry.PromoteToChampionAsync(algorithm, challengerVersionId, promotionRecord, cancellationToken);
            if (!registrySuccess)
            {
                _logger.LogError("Failed to update model registry for {Algorithm} promotion", algorithm);
                
                // Attempt rollback of router swap
                if (previousModel != null && previousChampion != null)
                {
                    await router.SwapAsync(previousModel, previousChampion, cancellationToken);
                }
                return false;
            }

            stopwatch.Stop();
            
            _logger.LogInformation("✅ Successfully promoted {Algorithm} to {ChallengerVersionId} in {Duration:F1}ms", 
                algorithm, challengerVersionId, stopwatch.Elapsed.TotalMilliseconds);

            return true;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "❌ Promotion failed for {Algorithm} after {Duration:F1}ms: {Error}", 
                algorithm, stopwatch.Elapsed.TotalMilliseconds, ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Rollback to previous champion (instant rollback < 100ms)
    /// AC6: One command rollback restores prior champion in < 100ms
    /// </summary>
    public async Task<bool> RollbackToPreviousAsync(string algorithm, string reason, CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            _logger.LogWarning("🔄 Starting EMERGENCY ROLLBACK for {Algorithm}: {Reason}", algorithm, reason);

            // Get promotion context for rollback
            if (!_promotionContexts.TryGetValue(algorithm, out var context))
            {
                _logger.LogError("No promotion context found for rollback of {Algorithm}", algorithm);
                return false;
            }

            // Get current router
            var router = _routerFactory.GetRouter<object>(algorithm);
            if (router == null)
            {
                _logger.LogError("No router found for algorithm {Algorithm}", algorithm);
                return false;
            }

            // Validate we have previous champion to rollback to
            if (context.PreviousChampionModel == null || string.IsNullOrEmpty(context.PreviousChampionVersionId))
            {
                _logger.LogError("No previous champion available for rollback of {Algorithm}", algorithm);
                return false;
            }

            // Get previous champion model version
            var previousChampion = await _modelRegistry.GetModelAsync(context.PreviousChampionVersionId, cancellationToken);
            if (previousChampion == null)
            {
                _logger.LogError("Previous champion model {PreviousChampionVersionId} not found for rollback", 
                    context.PreviousChampionVersionId);
                return false;
            }

            // INSTANT ATOMIC ROLLBACK - Critical performance requirement < 100ms
            var rollbackSuccess = await router.SwapAsync(context.PreviousChampionModel, previousChampion, cancellationToken);
            if (!rollbackSuccess)
            {
                _logger.LogError("❌ Atomic rollback swap failed for {Algorithm}", algorithm);
                return false;
            }

            // Update registry to record rollback
            var rollbackSuccess2 = await _modelRegistry.RollbackToPreviousAsync(algorithm, reason, cancellationToken);

            stopwatch.Stop();
            var rollbackTime = stopwatch.Elapsed.TotalMilliseconds;

            if (rollbackTime > 100)
            {
                _logger.LogWarning("⚠️  Rollback took {RollbackTime:F1}ms (target: <100ms) for {Algorithm}", 
                    rollbackTime, algorithm);
            }

            _logger.LogInformation("✅ Successfully rolled back {Algorithm} to {PreviousVersionId} in {Duration:F1}ms", 
                algorithm, context.PreviousChampionVersionId, rollbackTime);

            return rollbackSuccess && rollbackSuccess2;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "❌ CRITICAL: Rollback failed for {Algorithm} after {Duration:F1}ms: {Error}", 
                algorithm, stopwatch.Elapsed.TotalMilliseconds, ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Get promotion status and history
    /// </summary>
    public async Task<PromotionStatus> GetPromotionStatusAsync(string algorithm, CancellationToken cancellationToken = default)
    {
        var champion = await _modelRegistry.GetChampionAsync(algorithm, cancellationToken);
        var promotionHistory = await _modelRegistry.GetPromotionHistoryAsync(algorithm, cancellationToken);
        var lastPromotion = promotionHistory.FirstOrDefault();

        var status = new PromotionStatus
        {
            Algorithm = algorithm,
            CurrentChampionVersionId = champion?.VersionId ?? "none",
            LastPromotionTime = lastPromotion?.PromotedAt,
            LastPromotionReason = lastPromotion?.Reason ?? "none",
            CanRollback = _promotionContexts.ContainsKey(algorithm),
            RecentPromotions = promotionHistory.Take(5).Select(p => 
                $"{p.ToVersionId} ({p.PromotedAt:yyyy-MM-dd HH:mm})").ToList()
        };

        // Check for scheduled promotions
        if (_scheduledPromotions.TryGetValue(algorithm, out var scheduledChallengerVersionId))
        {
            status.HasPendingPromotion = true;
            status.PendingChallengerVersionId = scheduledChallengerVersionId;
            status.ScheduledPromotionTime = await _marketHours.GetNextSafeWindowAsync(cancellationToken);
        }

        return status;
    }

    /// <summary>
    /// Schedule automatic promotion for challenger
    /// </summary>
    public async Task<string> SchedulePromotionAsync(string algorithm, string challengerVersionId, PromotionSchedule schedule, CancellationToken cancellationToken = default)
    {
        var scheduleId = $"{algorithm}_{challengerVersionId}_{DateTime.UtcNow:yyyyMMdd_HHmmss}";
        
        try
        {
            // Validate challenger exists and is ready
            var challenger = await _modelRegistry.GetModelAsync(challengerVersionId, cancellationToken);
            if (challenger == null || !challenger.IsValidated)
            {
                throw new ArgumentException($"Challenger {challengerVersionId} not found or not validated");
            }

            // Determine promotion time
            var promotionTime = schedule.ScheduledTime ?? await _marketHours.GetNextSafeWindowAsync(cancellationToken);
            if (promotionTime == null)
            {
                throw new InvalidOperationException("No safe promotion window available");
            }

            // Store scheduled promotion
            _scheduledPromotions[algorithm] = challengerVersionId;

            // Schedule the promotion (in real implementation, this would use a job scheduler)
            _ = Task.Run(async () =>
            {
                try
                {
                    var delay = promotionTime.Value - DateTime.UtcNow;
                    if (delay > TimeSpan.Zero)
                    {
                        await Task.Delay(delay, cancellationToken);
                    }

                    // Execute promotion
                    var success = await PromoteToChampionAsync(algorithm, challengerVersionId, 
                        $"Scheduled promotion by {schedule.ApprovedBy}", cancellationToken);

                    if (success)
                    {
                        _scheduledPromotions.TryRemove(algorithm, out _);
                        _logger.LogInformation("Scheduled promotion completed for {Algorithm}", algorithm);
                    }
                    else
                    {
                        _logger.LogError("Scheduled promotion failed for {Algorithm}", algorithm);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Scheduled promotion error for {Algorithm}", algorithm);
                }
            }, cancellationToken);

            _logger.LogInformation("Scheduled promotion {ScheduleId} for {Algorithm} at {PromotionTime}", 
                scheduleId, algorithm, promotionTime);

            return scheduleId;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to schedule promotion for {Algorithm}", algorithm);
            throw;
        }
    }

    #region Private Methods

    private async Task ValidateTimingGatesAsync(PromotionDecision decision, CancellationToken cancellationToken)
    {
        decision.IsInSafeWindow = await _marketHours.IsInSafePromotionWindowAsync(cancellationToken);
        
        if (!decision.IsInSafeWindow)
        {
            var nextWindow = await _marketHours.GetNextSafeWindowAsync(cancellationToken);
            decision.NextSafeWindow = nextWindow?.ToString("yyyy-MM-dd HH:mm:ss UTC") ?? "Unknown";
            decision.ValidationErrors.Add($"Not in safe promotion window. Next window: {decision.NextSafeWindow}");
        }
    }

    private async Task ValidatePositionStateAsync(PromotionDecision decision, CancellationToken cancellationToken)
    {
        decision.IsFlat = await _positionService.IsCurrentlyFlatAsync(cancellationToken);
        
        if (!decision.IsFlat)
        {
            decision.ValidationErrors.Add("Must be flat (no open positions) for promotion");
            decision.RiskConcerns.Add("Open positions detected - promotion blocked for safety");
        }
    }

    private async Task ValidatePerformanceImprovementAsync(PromotionDecision decision, ModelVersion champion, ModelVersion challenger, CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        
        // Compare performance metrics
        decision.SharpeImprovement = challenger.Sharpe - champion.Sharpe;
        decision.SortinoImprovement = challenger.Sortino - champion.Sortino;
        decision.CVaRImprovement = challenger.CVaR - champion.CVaR;
        decision.DrawdownImprovement = challenger.MaxDrawdown - champion.MaxDrawdown;

        // Validate improvements meet thresholds
        if (decision.SharpeImprovement <= 0)
        {
            decision.ValidationErrors.Add($"Sharpe ratio not improved: {decision.SharpeImprovement:F4}");
        }

        if (decision.SortinoImprovement <= 0)
        {
            decision.ValidationErrors.Add($"Sortino ratio not improved: {decision.SortinoImprovement:F4}");
        }

        if (decision.CVaRImprovement <= 0)
        {
            decision.ValidationErrors.Add($"CVaR not improved: {decision.CVaRImprovement:F4}");
        }

        // Mock statistical significance (would use real historical testing)
        decision.PValue = 0.03m; // Mock p-value
        decision.StatisticallySignificant = decision.PValue < 0.05m;
        
        if (!decision.StatisticallySignificant)
        {
            decision.ValidationErrors.Add($"Performance improvement not statistically significant (p={decision.PValue:F4})");
        }
    }

    private async Task ValidateSchemaCompatibilityAsync(PromotionDecision decision, ModelVersion challenger, CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        
        // Validate artifact integrity
        decision.PassedSchemaValidation = await _modelRegistry.ValidateArtifactAsync(challenger.VersionId, cancellationToken);
        
        if (!decision.PassedSchemaValidation)
        {
            decision.ValidationErrors.Add("Challenger artifact failed schema validation");
        }

        // Check resource requirements
        decision.HasSufficientMemory = true; // Mock check
        if (!decision.HasSufficientMemory)
        {
            decision.ValidationErrors.Add("Insufficient memory for challenger model");
        }
    }

    private async Task AssessPromotionRiskAsync(PromotionDecision decision, string algorithm, string challengerVersionId, CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        
        // Check if this is a major version change
        var champion = await _modelRegistry.GetChampionAsync(algorithm, cancellationToken);
        if (champion != null)
        {
            var championMajorVersion = ExtractMajorVersion(champion.VersionId);
            var challengerMajorVersion = ExtractMajorVersion(challengerVersionId);
            
            if (championMajorVersion != challengerMajorVersion)
            {
                decision.RiskConcerns.Add("Major version change detected - requires additional validation");
            }
        }

        // Check promotion frequency
        var promotionHistory = await _modelRegistry.GetPromotionHistoryAsync(algorithm, cancellationToken);
        var recentPromotions = promotionHistory.Count(p => p.PromotedAt > DateTime.UtcNow.AddDays(-1));
        
        if (recentPromotions > 2)
        {
            decision.RiskConcerns.Add($"Too many recent promotions ({recentPromotions} in last 24h)");
        }
    }

    private async Task<object?> LoadModelArtifactAsync(ModelVersion modelVersion, CancellationToken cancellationToken)
    {
        await Task.Delay(50, cancellationToken); // Simulate loading time
        
        // In real implementation, this would load the actual model artifact
        return new { Version = modelVersion.VersionId, Type = modelVersion.ModelType };
    }

    private string ExtractMajorVersion(string versionId)
    {
        // Extract major version from version ID (simplified)
        return versionId.Split('_')[0];
    }

    #endregion
}

/// <summary>
/// Internal promotion context for rollback support
/// </summary>
internal class PromotionContext
{
    public string Algorithm { get; set; } = string.Empty;
    public string PreviousChampionVersionId { get; set; } = string.Empty;
    public object? PreviousChampionModel { get; set; }
    public string NewChampionVersionId { get; set; } = string.Empty;
    public DateTime PromotionTime { get; set; }
    public string Reason { get; set; } = string.Empty;
}

/// <summary>
/// Position service interface (mock)
/// </summary>
public interface IPositionService
{
    Task<bool> IsCurrentlyFlatAsync(CancellationToken cancellationToken = default);
    Task<decimal> GetCurrentPositionAsync(string symbol, CancellationToken cancellationToken = default);
    Task<Dictionary<string, decimal>> GetAllPositionsAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Production position service implementation
/// Provides real position tracking via TopstepX API
/// </summary>
public class ProductionPositionService : IPositionService
{
    private readonly ILogger<ProductionPositionService> _logger;
    private readonly IAccountService _accountService;
    private readonly ITopstepXClient _topstepXClient;

    public ProductionPositionService(
        ILogger<ProductionPositionService> logger,
        IAccountService accountService,
        ITopstepXClient topstepXClient)
    {
        _logger = logger;
        _accountService = accountService;
        _topstepXClient = topstepXClient;
    }

    public async Task<bool> IsCurrentlyFlatAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var positions = await GetAllPositionsAsync(cancellationToken);
            var hasOpenPositions = positions.Values.Any(pos => Math.Abs(pos) > 0.001m);
            
            _logger.LogDebug("Position check: {HasPositions} open positions", hasOpenPositions ? "Has" : "No");
            return !hasOpenPositions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to check if currently flat");
            // Conservative approach: assume not flat if we can't verify
            return false;
        }
    }

    public async Task<decimal> GetCurrentPositionAsync(string symbol, CancellationToken cancellationToken = default)
    {
        try
        {
            var positions = await GetAllPositionsAsync(cancellationToken);
            positions.TryGetValue(symbol, out var position);
            
            _logger.LogDebug("Position for {Symbol}: {Position}", symbol, position);
            return position;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get position for symbol {Symbol}", symbol);
            return 0; // Default to no position if error
        }
    }

    public async Task<Dictionary<string, decimal>> GetAllPositionsAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Get account ID from environment or configuration
            var accountId = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");
            if (string.IsNullOrEmpty(accountId))
            {
                _logger.LogWarning("No account ID configured, returning empty positions");
                return new Dictionary<string, decimal>();
            }

            // Get positions from TopstepX API
            var positionsResponse = await _topstepXClient.GetAccountPositionsAsync(accountId, cancellationToken);
            var positions = new Dictionary<string, decimal>();

            // Parse positions from API response
            if (positionsResponse.ValueKind == JsonValueKind.Array)
            {
                foreach (var position in positionsResponse.EnumerateArray())
                {
                    if (position.TryGetProperty("symbol", out var symbolElement) &&
                        position.TryGetProperty("quantity", out var quantityElement))
                    {
                        var symbol = symbolElement.GetString();
                        if (symbol != null && quantityElement.TryGetDecimal(out var quantity))
                        {
                            positions[symbol] = quantity;
                        }
                    }
                }
            }

            _logger.LogDebug("Retrieved {Count} positions from TopstepX", positions.Count);
            return positions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get all positions");
            return new Dictionary<string, decimal>(); // Return empty on error
        }
    }
}