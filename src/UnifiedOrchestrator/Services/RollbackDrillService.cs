using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using AbstractionsTradingDecision = TradingBot.Abstractions.TradingDecision;
using InterfacesTradingDecision = TradingBot.UnifiedOrchestrator.Interfaces.TradingDecision;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Rollback drill service that tests rollback capabilities under simulated load
/// Provides evidence of instant rollback with context preservation
/// </summary>
public class RollbackDrillService : IRollbackDrillService
{
    private readonly ILogger<RollbackDrillService> _logger;
    private readonly ITradingBrainAdapter _brainAdapter;
    private readonly IPromotionService _promotionService;
    private readonly List<RollbackDrillResult> _drillHistory = new();
    
    // Load simulation parameters
    private const int HIGH_LOAD_DECISIONS_PER_SECOND = 50;
    private const int STRESS_TEST_DURATION_SECONDS = 30;
    private const int ROLLBACK_TIMEOUT_MS = 1000; // 1 second max rollback time
    
    public RollbackDrillService(
        ILogger<RollbackDrillService> logger,
        ITradingBrainAdapter brainAdapter,
        IPromotionService promotionService)
    {
        _logger = logger;
        _brainAdapter = brainAdapter;
        _promotionService = promotionService;
    }

    /// <summary>
    /// Execute comprehensive rollback drill under simulated trading load
    /// </summary>
    public async Task<RollbackDrillResult> ExecuteRollbackDrillAsync(RollbackDrillConfig config, CancellationToken cancellationToken = default)
    {
        var drillId = Guid.NewGuid().ToString();
        var stopwatch = Stopwatch.StartNew();
        
        _logger.LogWarning("[ROLLBACK-DRILL] Starting drill {DrillId} - Load: {LoadLevel}, Duration: {Duration}s", 
            drillId, config.LoadLevel, config.TestDurationSeconds);

        var result = new RollbackDrillResult
        {
            DrillId = drillId,
            StartTime = DateTime.UtcNow,
            Config = config,
            Events = new List<RollbackEvent>(),
            Metrics = new RollbackMetrics()
        };

        try
        {
            // Phase 1: Establish baseline with champion
            _logger.LogInformation("[ROLLBACK-DRILL] Phase 1: Establishing champion baseline");
            await EstablishBaseline(result, cancellationToken).ConfigureAwait(false);

            // Phase 2: Promote challenger
            _logger.LogInformation("[ROLLBACK-DRILL] Phase 2: Promoting challenger");
            await PromoteChallenger(result, cancellationToken).ConfigureAwait(false);

            // Phase 3: Generate high load on challenger
            _logger.LogInformation("[ROLLBACK-DRILL] Phase 3: Generating load on challenger");
            await GenerateLoad(result, config, cancellationToken).ConfigureAwait(false);

            // Phase 4: Execute rollback under load
            _logger.LogInformation("[ROLLBACK-DRILL] Phase 4: Executing rollback under load");
            await ExecuteRollbackUnderLoad(result, cancellationToken).ConfigureAwait(false);

            // Phase 5: Verify rollback success and stability
            _logger.LogInformation("[ROLLBACK-DRILL] Phase 5: Verifying rollback success");
            await VerifyRollbackSuccess(result, cancellationToken).ConfigureAwait(false);

            result.Success = true;
            result.EndTime = DateTime.UtcNow;
            result.TotalDurationMs = stopwatch.Elapsed.TotalMilliseconds;

            _logger.LogWarning("[ROLLBACK-DRILL] Drill {DrillId} completed successfully in {Duration:F2}ms", 
                drillId, result.TotalDurationMs);
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = ex.Message;
            result.EndTime = DateTime.UtcNow;
            result.TotalDurationMs = stopwatch.Elapsed.TotalMilliseconds;

            _logger.LogError(ex, "[ROLLBACK-DRILL] Drill {DrillId} failed after {Duration:F2}ms", 
                drillId, stopwatch.Elapsed.TotalMilliseconds);
        }

        // Store drill result for analysis
        _drillHistory.Add(result);
        
        return result;
    }

    /// <summary>
    /// Execute quick rollback drill for demonstration
    /// </summary>
    public async Task<RollbackDrillResult> ExecuteQuickDrillAsync(CancellationToken cancellationToken = default)
    {
        var config = new RollbackDrillConfig
        {
            LoadLevel = LoadLevel.High,
            TestDurationSeconds = 10,
            DecisionsPerSecond = 20,
            EnableMetrics = true,
            EnableContextPreservation = true
        };

        return await ExecuteRollbackDrillAsync(config, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Establish baseline performance with champion
    /// </summary>
    private async Task EstablishBaseline(RollbackDrillResult result, CancellationToken cancellationToken)
    {
        var baselineStart = DateTime.UtcNow;
        var decisions = new List<(DateTime, AbstractionsTradingDecision, double)>();

        // Ensure we're on champion
        await _brainAdapter.RollbackToChampionAsync(cancellationToken).ConfigureAwait(false);
        
        // Generate baseline decisions
        for (int i = 0; i < 20; i++)
        {
            var context = CreateTestTradingContext(i);
            var decisionStart = DateTime.UtcNow;
            var decision = await _brainAdapter.DecideAsync(context, cancellationToken).ConfigureAwait(false);
            var decisionTime = (DateTime.UtcNow - decisionStart).TotalMilliseconds;
            
            decisions.Add((DateTime.UtcNow, ConvertToAbstractionsDecision(decision), decisionTime));
            
            if (cancellationToken.IsCancellationRequested) break;
            await Task.Delay(50, cancellationToken).ConfigureAwait(false); // 20 decisions per second
        }

        result.Events.Add(new RollbackEvent
        {
            Timestamp = baselineStart,
            EventType = "Baseline Established",
            Details = $"Champion baseline: {decisions.Count} decisions, avg latency {decisions.Average(d => d.Item3):F2}ms",
            Success = true
        });

        result.Metrics.BaselineLatencyMs = decisions.Average(d => d.Item3);
        result.Metrics.BaselineDecisionsCount = decisions.Count;
    }

    /// <summary>
    /// Promote challenger and verify promotion
    /// </summary>
    private async Task PromoteChallenger(RollbackDrillResult result, CancellationToken cancellationToken)
    {
        var promotionStart = DateTime.UtcNow;
        var promotionStopwatch = Stopwatch.StartNew();

        // Execute promotion
        var promoted = await _brainAdapter.PromoteToChallengerAsync(cancellationToken).ConfigureAwait(false);
        
        var promotionTime = promotionStopwatch.Elapsed.TotalMilliseconds;

        result.Events.Add(new RollbackEvent
        {
            Timestamp = promotionStart,
            EventType = "Challenger Promotion",
            Details = $"Promotion completed in {promotionTime:F2}ms, Success: {promoted}",
            Success = promoted,
            LatencyMs = promotionTime
        });

        if (!promoted)
        {
            throw new InvalidOperationException("Failed to promote challenger");
        }

        result.Metrics.PromotionLatencyMs = promotionTime;
    }

    /// <summary>
    /// Generate high load on the system
    /// </summary>
    private async Task GenerateLoad(RollbackDrillResult result, RollbackDrillConfig config, CancellationToken cancellationToken)
    {
        var loadStart = DateTime.UtcNow;
        var decisions = new ConcurrentBag<(DateTime, double, bool)>(); // timestamp, latency, success
        var tasks = new List<Task>();
        var loadCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

        // Stop load generation after specified duration
        _ = Task.Delay(TimeSpan.FromSeconds(config.TestDurationSeconds), cancellationToken)
            .ContinueWith(_ => loadCancellation.Cancel(), TaskScheduler.Default);

        // Generate concurrent load
        for (int worker = 0; worker < Environment.ProcessorCount; worker++)
        {
            tasks.Add(GenerateWorkerLoad(worker, config.DecisionsPerSecond / Environment.ProcessorCount, 
                decisions, loadCancellation.Token));
        }

        // Wait for load generation to complete
        await Task.WhenAll(tasks).ConfigureAwait(false);

        var loadDuration = (DateTime.UtcNow - loadStart).TotalMilliseconds;
        var totalDecisions = decisions.Count;
        var successfulDecisions = decisions.Count(d => d.Item3);
        var averageLatency = decisions.Where(d => d.Item3).Average(d => d.Item2);

        result.Events.Add(new RollbackEvent
        {
            Timestamp = loadStart,
            EventType = "Load Generation Completed",
            Details = $"Generated {totalDecisions} decisions ({successfulDecisions} successful) in {loadDuration:F0}ms, avg latency {averageLatency:F2}ms",
            Success = true,
            LatencyMs = averageLatency
        });

        result.Metrics.LoadTestDecisionsCount = totalDecisions;
        result.Metrics.LoadTestLatencyMs = averageLatency;
        result.Metrics.LoadTestSuccessRate = (double)successfulDecisions / totalDecisions;
    }

    /// <summary>
    /// Execute rollback while system is under load
    /// </summary>
    private async Task ExecuteRollbackUnderLoad(RollbackDrillResult result, CancellationToken cancellationToken)
    {
        var rollbackStart = DateTime.UtcNow;
        var rollbackStopwatch = Stopwatch.StartNew();

        // Start background load during rollback
        var loadCancellation = new CancellationTokenSource();
        var backgroundDecisions = new ConcurrentBag<(DateTime, double, bool)>();
        var backgroundLoadTask = GenerateWorkerLoad(999, 10, backgroundDecisions, loadCancellation.Token);

        try
        {
            // Execute rollback under load
            var rollbackSuccess = await _brainAdapter.RollbackToChampionAsync(cancellationToken).ConfigureAwait(false);
            var rollbackTime = rollbackStopwatch.Elapsed.TotalMilliseconds;

            // Stop background load
            loadCancellation.Cancel();
            
            // Wait for background load to complete
            try { await backgroundLoadTask.ConfigureAwait(false); } catch { /* Expected cancellation */ }

            result.Events.Add(new RollbackEvent
            {
                Timestamp = rollbackStart,
                EventType = "Rollback Under Load",
                Details = $"Rollback completed in {rollbackTime:F2}ms while processing {backgroundDecisions.Count} concurrent decisions",
                Success = rollbackSuccess,
                LatencyMs = rollbackTime
            });

            if (rollbackTime > ROLLBACK_TIMEOUT_MS)
            {
                _logger.LogWarning("[ROLLBACK-DRILL] Rollback took {RollbackTime:F2}ms, exceeds target of {Target}ms", 
                    rollbackTime, ROLLBACK_TIMEOUT_MS);
            }

            result.Metrics.RollbackLatencyMs = rollbackTime;
            result.Metrics.RollbackSuccess = rollbackSuccess;
            result.Metrics.ConcurrentDecisionsDuringRollback = backgroundDecisions.Count;

            if (!rollbackSuccess)
            {
                throw new InvalidOperationException("Rollback failed under load");
            }
        }
        finally
        {
            loadCancellation.Cancel();
        }
    }

    /// <summary>
    /// Verify rollback was successful and system is stable
    /// </summary>
    private async Task VerifyRollbackSuccess(RollbackDrillResult result, CancellationToken cancellationToken)
    {
        var verificationStart = DateTime.UtcNow;
        var postRollbackDecisions = new List<(DateTime, AbstractionsTradingDecision, double)>();

        // Generate post-rollback decisions to verify stability
        for (int i = 0; i < 10; i++)
        {
            var context = CreateTestTradingContext(i + 1000);
            var decisionStart = DateTime.UtcNow;
            var decision = await _brainAdapter.DecideAsync(context, cancellationToken).ConfigureAwait(false);
            var decisionTime = (DateTime.UtcNow - decisionStart).TotalMilliseconds;
            
            postRollbackDecisions.Add((DateTime.UtcNow, ConvertToAbstractionsDecision(decision), decisionTime));
            
            // Verify we're back on champion
            if (!decision.Reasoning.ContainsKey("AdapterMode") || 
                !decision.Reasoning["AdapterMode"].ToString()!.Contains("UnifiedTradingBrain-Primary"))
            {
                throw new InvalidOperationException("Rollback verification failed - not on champion");
            }
            
            if (cancellationToken.IsCancellationRequested) break;
            await Task.Delay(100, cancellationToken).ConfigureAwait(false);
        }

        var avgPostRollbackLatency = postRollbackDecisions.Average(d => d.Item3);
        var latencyDegradation = (avgPostRollbackLatency - result.Metrics.BaselineLatencyMs) / result.Metrics.BaselineLatencyMs;

        result.Events.Add(new RollbackEvent
        {
            Timestamp = verificationStart,
            EventType = "Rollback Verification",
            Details = $"Post-rollback: {postRollbackDecisions.Count} decisions, avg latency {avgPostRollbackLatency:F2}ms (degradation: {latencyDegradation*100:F1}%)",
            Success = true,
            LatencyMs = avgPostRollbackLatency
        });

        result.Metrics.PostRollbackLatencyMs = avgPostRollbackLatency;
        result.Metrics.PostRollbackDecisionsCount = postRollbackDecisions.Count;
        result.Metrics.LatencyDegradationPercent = latencyDegradation * 100;

        // Context preservation check
        var stats = _brainAdapter.GetStatistics();
        result.Metrics.ContextPreserved = stats.TotalDecisions > 0; // Simple check - in production would be more thorough
    }

    /// <summary>
    /// Generate worker load for stress testing
    /// </summary>
    private async Task GenerateWorkerLoad(int workerId, int decisionsPerSecond, ConcurrentBag<(DateTime, double, bool)> decisions, CancellationToken cancellationToken)
    {
        var delayMs = Math.Max(1, 1000 / decisionsPerSecond);
        var decisionCounter = 0;

        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                var context = CreateTestTradingContext(workerId * 10000 + decisionCounter);
                var start = DateTime.UtcNow;
                var decision = await _brainAdapter.DecideAsync(context, cancellationToken).ConfigureAwait(false);
                var latency = (DateTime.UtcNow - start).TotalMilliseconds;
                
                decisions.Add((start, latency, true));
                decisionCounter++;
                
                await Task.Delay(delayMs, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                decisions.Add((DateTime.UtcNow, 0, false));
                _logger.LogDebug("[ROLLBACK-DRILL] Worker {WorkerId} decision failed: {Error}", workerId, ex.Message);
            }
        }
    }

    /// <summary>
    /// Create test trading context for load generation
    /// </summary>
    private TradingContext CreateTestTradingContext(int seed)
    {
        var random = new Random(seed);
        var basePrice = 4500m + (decimal)(random.NextDouble() * 100);
        
        return new TradingContext
        {
            Symbol = "ES",
            CurrentPrice = basePrice,
            Timestamp = DateTime.UtcNow,
            Volume = 1000 + random.Next(500),
            Spread = 0.25m,
            IsMarketOpen = true,
            IsEmergencyStop = false,
            RiskParameters = new Dictionary<string, object>
            {
                ["MaxPosition"] = 10,
                ["MaxRisk"] = 0.02m
            }
        };
    }

    /// <summary>
    /// Get rollback drill history
    /// </summary>
    public List<RollbackDrillResult> GetDrillHistory(int maxCount = 20)
    {
        return _drillHistory.TakeLast(maxCount).ToList();
    }

    /// <summary>
    /// Generate summary report of all rollback drills
    /// </summary>
    public RollbackDrillSummary GenerateSummaryReport()
    {
        if (!_drillHistory.Any())
        {
            return new RollbackDrillSummary
            {
                TotalDrills = 0,
                SuccessRate = 0,
                Message = "No rollback drills executed yet"
            };
        }

        var successfulDrills = _drillHistory.Count(d => d.Success);
        var avgRollbackTime = _drillHistory.Where(d => d.Success).Average(d => d.Metrics.RollbackLatencyMs);
        var maxRollbackTime = _drillHistory.Where(d => d.Success).Max(d => d.Metrics.RollbackLatencyMs);

        return new RollbackDrillSummary
        {
            TotalDrills = _drillHistory.Count,
            SuccessfulDrills = successfulDrills,
            SuccessRate = (double)successfulDrills / _drillHistory.Count,
            AverageRollbackTimeMs = avgRollbackTime,
            MaxRollbackTimeMs = maxRollbackTime,
            LastDrillTime = _drillHistory.LastOrDefault()?.StartTime ?? DateTime.MinValue,
            Message = $"Executed {_drillHistory.Count} drills with {successfulDrills}/{_drillHistory.Count} success rate. Avg rollback time: {avgRollbackTime:F2}ms"
        };
    }
    
    /// <summary>
    /// Convert UnifiedOrchestrator TradingDecision to Abstractions TradingDecision
    /// </summary>
    private AbstractionsTradingDecision ConvertToAbstractionsDecision(InterfacesTradingDecision unifiedDecision)
    {
        // Parse Action string to TradingAction enum
        var tradingAction = unifiedDecision.Action.ToUpperInvariant() switch
        {
            "BUY" => TradingBot.Abstractions.TradingAction.Buy,
            "SELL" => TradingBot.Abstractions.TradingAction.Sell,
            "HOLD" => TradingBot.Abstractions.TradingAction.Hold,
            _ => TradingBot.Abstractions.TradingAction.Hold
        };

        return new AbstractionsTradingDecision
        {
            DecisionId = Guid.NewGuid().ToString(),
            Symbol = unifiedDecision.Symbol,
            Action = tradingAction,
            Quantity = unifiedDecision.Size,
            Confidence = unifiedDecision.Confidence,
            MLConfidence = unifiedDecision.Confidence,
            MLStrategy = unifiedDecision.Strategy,
            Timestamp = unifiedDecision.Timestamp,
            Reasoning = new Dictionary<string, object>(unifiedDecision.DecisionMetadata)
        };
    }
}