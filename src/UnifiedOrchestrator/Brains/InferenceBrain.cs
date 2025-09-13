using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Brains;

/// <summary>
/// Read-only inference brain wrapping AtomicModelRouter instances
/// No training or parameter mutation allowed - only decision making
/// </summary>
public class InferenceBrain : IInferenceBrain
{
    private readonly ILogger<InferenceBrain> _logger;
    private readonly IModelRouterFactory _routerFactory;
    
    // Algorithm-specific routers
    private readonly IModelRouter<object> _ppoRouter;
    private readonly IModelRouter<object> _ucbRouter;
    private readonly IModelRouter<object> _lstmRouter;
    
    // Statistics tracking
    private int _totalDecisions = 0;
    private int _decisionsToday = 0;
    private int _errorCount = 0;
    private readonly List<double> _processingTimes = new();
    private readonly DateTime _startTime = DateTime.UtcNow;
    private DateTime _lastDecisionTime = DateTime.MinValue;
    private DateTime _lastResetDate = DateTime.UtcNow.Date;

    public InferenceBrain(
        ILogger<InferenceBrain> logger, 
        IModelRouterFactory routerFactory)
    {
        _logger = logger;
        _routerFactory = routerFactory;
        
        // Create read-only routers for each algorithm
        _ppoRouter = _routerFactory.CreateRouter<object>("PPO");
        _ucbRouter = _routerFactory.CreateRouter<object>("UCB");
        _lstmRouter = _routerFactory.CreateRouter<object>("LSTM");
        
        _logger.LogInformation("InferenceBrain initialized with read-only model routers");
    }

    /// <summary>
    /// Make a trading decision using current champion models (read-only)
    /// This is the main entry point for live trading inference
    /// </summary>
    public async Task<TradingDecision> DecideAsync(TradingContext context, CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var riskWarnings = new List<string>();
        
        try
        {
            // Reset daily counters if new day
            if (DateTime.UtcNow.Date > _lastResetDate)
            {
                _decisionsToday = 0;
                _lastResetDate = DateTime.UtcNow.Date;
            }

            // Validate input context
            if (context == null)
            {
                throw new ArgumentNullException(nameof(context));
            }

            // Check emergency stop
            if (context.IsEmergencyStop)
            {
                return CreateEmergencyStopDecision(context, stopwatch.Elapsed);
            }

            // Ensure all models are ready
            if (!await IsReadyAsync(cancellationToken))
            {
                riskWarnings.Add("One or more champion models not ready");
                return CreateFallbackDecision(context, stopwatch.Elapsed, riskWarnings);
            }

            // Get current champion models (read-only access)
            var ppoModel = _ppoRouter.Current;
            var ucbModel = _ucbRouter.Current;
            var lstmModel = _lstmRouter.Current;
            
            var ppoVersion = _ppoRouter.CurrentVersion;
            var ucbVersion = _ucbRouter.CurrentVersion;
            var lstmVersion = _lstmRouter.CurrentVersion;

            // Perform risk checks
            var passedRiskChecks = PerformRiskChecks(context, riskWarnings);
            if (!passedRiskChecks && riskWarnings.Any(w => w.Contains("CRITICAL")))
            {
                return CreateRiskStopDecision(context, stopwatch.Elapsed, riskWarnings);
            }

            // Make inference decisions from each algorithm
            var decisions = new Dictionary<string, AlgorithmDecision>();
            
            // PPO Decision (if available)
            if (ppoModel != null && ppoVersion != null)
            {
                decisions["PPO"] = await MakePPODecision(ppoModel, context, cancellationToken);
            }
            
            // UCB Decision (if available)
            if (ucbModel != null && ucbVersion != null)
            {
                decisions["UCB"] = await MakeUCBDecision(ucbModel, context, cancellationToken);
            }
            
            // LSTM Decision (if available)
            if (lstmModel != null && lstmVersion != null)
            {
                decisions["LSTM"] = await MakeLSTMDecision(lstmModel, context, cancellationToken);
            }

            // Ensemble decision making
            var finalDecision = await EnsembleDecisions(decisions, context, cancellationToken);
            
            // Build complete trading decision with full attribution
            var tradingDecision = new TradingDecision
            {
                Symbol = context.Symbol,
                Timestamp = context.Timestamp,
                Action = finalDecision.Action,
                Size = finalDecision.Size,
                Confidence = finalDecision.Confidence,
                Strategy = finalDecision.Strategy,
                
                // Model version attribution (required for AC3)
                PPOVersionId = ppoVersion?.VersionId ?? "none",
                UCBVersionId = ucbVersion?.VersionId ?? "none", 
                LSTMVersionId = lstmVersion?.VersionId ?? "none",
                AlgorithmVersions = new Dictionary<string, string>
                {
                    ["PPO"] = ppoVersion?.VersionId ?? "none",
                    ["UCB"] = ucbVersion?.VersionId ?? "none",
                    ["LSTM"] = lstmVersion?.VersionId ?? "none"
                },
                AlgorithmHashes = new Dictionary<string, string>
                {
                    ["PPO"] = ppoVersion?.ArtifactHash ?? "none",
                    ["UCB"] = ucbVersion?.ArtifactHash ?? "none", 
                    ["LSTM"] = lstmVersion?.ArtifactHash ?? "none"
                },
                
                // Performance metadata
                ProcessingTimeMs = (decimal)stopwatch.Elapsed.TotalMilliseconds,
                AlgorithmConfidences = decisions.ToDictionary(d => d.Key, d => d.Value.Confidence),
                
                // Risk assessment
                PassedRiskChecks = passedRiskChecks,
                RiskWarnings = riskWarnings,
                
                DecisionMetadata = new Dictionary<string, object>
                {
                    ["EnsembleMethod"] = finalDecision.EnsembleMethod,
                    ["ParticipatingAlgorithms"] = decisions.Keys.ToList(),
                    ["InferenceTimeMs"] = stopwatch.Elapsed.TotalMilliseconds,
                    ["ModelLoadTimes"] = new Dictionary<string, DateTime>
                    {
                        ["PPO"] = ppoVersion?.CreatedAt ?? DateTime.MinValue,
                        ["UCB"] = ucbVersion?.CreatedAt ?? DateTime.MinValue,
                        ["LSTM"] = lstmVersion?.CreatedAt ?? DateTime.MinValue
                    }
                }
            };

            // Update statistics
            _totalDecisions++;
            _decisionsToday++;
            _lastDecisionTime = DateTime.UtcNow;
            _processingTimes.Add(stopwatch.Elapsed.TotalMilliseconds);
            
            // Keep only recent processing times for stats
            if (_processingTimes.Count > 1000)
            {
                _processingTimes.RemoveRange(0, 500);
            }

            _logger.LogInformation("[INFERENCE] {Symbol}: {Action} {Size} | Strategy={Strategy} | Confidence={Confidence:P1} | " +
                "PPO={PPOVersion} UCB={UCBVersion} LSTM={LSTMVersion} | {ProcessingTime:F1}ms",
                context.Symbol, finalDecision.Action, finalDecision.Size, finalDecision.Strategy, finalDecision.Confidence,
                ppoVersion?.VersionId[..8] ?? "none", ucbVersion?.VersionId[..8] ?? "none", lstmVersion?.VersionId[..8] ?? "none",
                stopwatch.Elapsed.TotalMilliseconds);

            return tradingDecision;
        }
        catch (Exception ex)
        {
            _errorCount++;
            _logger.LogError(ex, "[INFERENCE] Error making decision for {Symbol}", context.Symbol);
            return CreateErrorDecision(context, stopwatch.Elapsed, ex.Message);
        }
    }

    /// <summary>
    /// Get current champion model versions for all algorithms
    /// </summary>
    public async Task<Dictionary<string, ModelVersion?>> GetChampionVersionsAsync(CancellationToken cancellationToken = default)
    {
        return await Task.FromResult(new Dictionary<string, ModelVersion?>
        {
            ["PPO"] = _ppoRouter.CurrentVersion,
            ["UCB"] = _ucbRouter.CurrentVersion,
            ["LSTM"] = _lstmRouter.CurrentVersion
        });
    }

    /// <summary>
    /// Check if all champion models are loaded and ready
    /// </summary>
    public async Task<bool> IsReadyAsync(CancellationToken cancellationToken = default)
    {
        var ppoStats = await _ppoRouter.GetStatsAsync(cancellationToken);
        var ucbStats = await _ucbRouter.GetStatsAsync(cancellationToken);
        var lstmStats = await _lstmRouter.GetStatsAsync(cancellationToken);
        
        // At least one algorithm must be ready
        return ppoStats.IsHealthy || ucbStats.IsHealthy || lstmStats.IsHealthy;
    }

    /// <summary>
    /// Get inference statistics
    /// </summary>
    public async Task<InferenceStats> GetStatsAsync(CancellationToken cancellationToken = default)
    {
        var ppoStats = await _ppoRouter.GetStatsAsync(cancellationToken);
        var ucbStats = await _ucbRouter.GetStatsAsync(cancellationToken);
        var lstmStats = await _lstmRouter.GetStatsAsync(cancellationToken);
        
        return new InferenceStats
        {
            TotalDecisions = _totalDecisions,
            DecisionsToday = _decisionsToday,
            AverageProcessingTimeMs = _processingTimes.Count > 0 ? (decimal)_processingTimes.Average() : 0,
            MaxProcessingTimeMs = _processingTimes.Count > 0 ? (decimal)_processingTimes.Max() : 0,
            LastDecisionTime = _lastDecisionTime,
            ErrorCount = _errorCount,
            StartTime = _startTime,
            
            ModelHealth = new Dictionary<string, bool>
            {
                ["PPO"] = ppoStats.IsHealthy,
                ["UCB"] = ucbStats.IsHealthy,
                ["LSTM"] = lstmStats.IsHealthy
            },
            
            ModelLoadTimes = new Dictionary<string, DateTime>
            {
                ["PPO"] = ppoStats.LoadedAt,
                ["UCB"] = ucbStats.LoadedAt,
                ["LSTM"] = lstmStats.LoadedAt
            },
            
            ModelSwapCounts = new Dictionary<string, int>
            {
                ["PPO"] = ppoStats.SwapCount,
                ["UCB"] = ucbStats.SwapCount,
                ["LSTM"] = lstmStats.SwapCount
            }
        };
    }

    #region Private Methods

    private bool PerformRiskChecks(TradingContext context, List<string> warnings)
    {
        var passed = true;
        
        // Daily loss limit check
        if (context.DailyPnL <= context.DailyLossLimit)
        {
            warnings.Add($"CRITICAL: Daily loss limit exceeded: {context.DailyPnL:C} <= {context.DailyLossLimit:C}");
            passed = false;
        }
        
        // Max drawdown check
        if (context.UnrealizedPnL <= -Math.Abs(context.MaxDrawdown))
        {
            warnings.Add($"CRITICAL: Max drawdown exceeded: {context.UnrealizedPnL:C} <= {-Math.Abs(context.MaxDrawdown):C}");
            passed = false;
        }
        
        // Position size sanity check
        if (Math.Abs(context.CurrentPosition) > 10) // Arbitrary large position check
        {
            warnings.Add($"WARNING: Large position detected: {context.CurrentPosition}");
        }
        
        return passed;
    }

    private async Task<AlgorithmDecision> MakePPODecision(object ppoModel, TradingContext context, CancellationToken cancellationToken)
    {
        // Placeholder for actual PPO inference
        // In real implementation, this would call the PPO model
        await Task.Delay(1, cancellationToken); // Simulate inference time
        
        return new AlgorithmDecision
        {
            Action = "HOLD",
            Size = 0,
            Confidence = 0.5m,
            Strategy = "PPO"
        };
    }

    private async Task<AlgorithmDecision> MakeUCBDecision(object ucbModel, TradingContext context, CancellationToken cancellationToken)
    {
        // Placeholder for actual UCB inference
        await Task.Delay(1, cancellationToken);
        
        return new AlgorithmDecision
        {
            Action = "HOLD", 
            Size = 0,
            Confidence = 0.5m,
            Strategy = "UCB"
        };
    }

    private async Task<AlgorithmDecision> MakeLSTMDecision(object lstmModel, TradingContext context, CancellationToken cancellationToken)
    {
        // Placeholder for actual LSTM inference
        await Task.Delay(1, cancellationToken);
        
        return new AlgorithmDecision
        {
            Action = "HOLD",
            Size = 0,
            Confidence = 0.5m,
            Strategy = "LSTM"
        };
    }

    private async Task<EnsembleDecision> EnsembleDecisions(Dictionary<string, AlgorithmDecision> decisions, TradingContext context, CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        
        if (decisions.Count == 0)
        {
            return new EnsembleDecision
            {
                Action = "HOLD",
                Size = 0,
                Confidence = 0,
                Strategy = "FALLBACK",
                EnsembleMethod = "NO_MODELS"
            };
        }
        
        // Simple ensemble: use highest confidence decision
        var bestDecision = decisions.Values.OrderByDescending(d => d.Confidence).First();
        
        return new EnsembleDecision
        {
            Action = bestDecision.Action,
            Size = bestDecision.Size,
            Confidence = bestDecision.Confidence,
            Strategy = bestDecision.Strategy,
            EnsembleMethod = "HIGHEST_CONFIDENCE"
        };
    }

    private TradingDecision CreateEmergencyStopDecision(TradingContext context, TimeSpan processingTime)
    {
        return new TradingDecision
        {
            Symbol = context.Symbol,
            Timestamp = context.Timestamp,
            Action = "HOLD",
            Size = 0,
            Confidence = 0,
            Strategy = "EMERGENCY_STOP",
            ProcessingTimeMs = (decimal)processingTime.TotalMilliseconds,
            PassedRiskChecks = false,
            RiskWarnings = new List<string> { "EMERGENCY_STOP_ACTIVE" }
        };
    }

    private TradingDecision CreateRiskStopDecision(TradingContext context, TimeSpan processingTime, List<string> riskWarnings)
    {
        return new TradingDecision
        {
            Symbol = context.Symbol,
            Timestamp = context.Timestamp,
            Action = "HOLD",
            Size = 0,
            Confidence = 0,
            Strategy = "RISK_STOP",
            ProcessingTimeMs = (decimal)processingTime.TotalMilliseconds,
            PassedRiskChecks = false,
            RiskWarnings = riskWarnings
        };
    }

    private TradingDecision CreateFallbackDecision(TradingContext context, TimeSpan processingTime, List<string> warnings)
    {
        return new TradingDecision
        {
            Symbol = context.Symbol,
            Timestamp = context.Timestamp,
            Action = "HOLD",
            Size = 0,
            Confidence = 0,
            Strategy = "FALLBACK",
            ProcessingTimeMs = (decimal)processingTime.TotalMilliseconds,
            PassedRiskChecks = true,
            RiskWarnings = warnings
        };
    }

    private TradingDecision CreateErrorDecision(TradingContext context, TimeSpan processingTime, string errorMessage)
    {
        return new TradingDecision
        {
            Symbol = context.Symbol,
            Timestamp = context.Timestamp,
            Action = "HOLD",
            Size = 0,
            Confidence = 0,
            Strategy = "ERROR",
            ProcessingTimeMs = (decimal)processingTime.TotalMilliseconds,
            PassedRiskChecks = false,
            RiskWarnings = new List<string> { $"ERROR: {errorMessage}" }
        };
    }

    #endregion
}

/// <summary>
/// Internal algorithm decision structure
/// </summary>
internal class AlgorithmDecision
{
    public string Action { get; set; } = "HOLD";
    public decimal Size { get; set; }
    public decimal Confidence { get; set; }
    public string Strategy { get; set; } = string.Empty;
}

/// <summary>
/// Internal ensemble decision structure
/// </summary>
internal class EnsembleDecision
{
    public string Action { get; set; } = "HOLD";
    public decimal Size { get; set; }
    public decimal Confidence { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public string EnsembleMethod { get; set; } = string.Empty;
}