using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using BotCore.Brain;
using BotCore.Models;
using BotCore.Market;
using BotCore.Risk;
using BotCore.Strategy;

namespace TradingBot.UnifiedOrchestrator.Brains;

/// <summary>
/// Adapter that bridges UnifiedTradingBrain (champion) with InferenceBrain (challenger)
/// Maintains UnifiedTradingBrain as primary decision maker while shadow testing InferenceBrain
/// Provides gradual transition path based on proven statistical performance
/// </summary>
public class TradingBrainAdapter : ITradingBrainAdapter
{
    private readonly ILogger<TradingBrainAdapter> _logger;
    private readonly UnifiedTradingBrain _unifiedBrain; // Champion - current production brain
    private readonly IInferenceBrain _inferenceBrain; // Challenger - new architecture
    private readonly IShadowTester _shadowTester;
    private readonly IPromotionService _promotionService;

    // Decision routing and comparison tracking
    private readonly List<DecisionComparison> _recentComparisons = new();
    private readonly object _comparisonLock = new object();
    private DateTime _lastPromotionCheck = DateTime.MinValue;
    private bool _useInferenceBrainPrimary = false; // Start with UnifiedTradingBrain as primary
    
    // Performance tracking
    private int _totalDecisions = 0;
    private int _agreementCount = 0;
    private int _disagreementCount = 0;
    private double _agreementRate => _totalDecisions > 0 ? (double)_agreementCount / _totalDecisions : 0.0;
    
    public TradingBrainAdapter(
        ILogger<TradingBrainAdapter> logger,
        UnifiedTradingBrain unifiedBrain,
        IInferenceBrain inferenceBrain,
        IShadowTester shadowTester,
        IPromotionService promotionService)
    {
        _logger = logger;
        _unifiedBrain = unifiedBrain;
        _inferenceBrain = inferenceBrain;
        _shadowTester = shadowTester;
        _promotionService = promotionService;
        
        _logger.LogInformation("TradingBrainAdapter initialized - UnifiedTradingBrain as champion, InferenceBrain as challenger");
    }

    /// <summary>
    /// Make trading decision using champion brain with challenger shadow testing
    /// </summary>
    public async Task<TradingDecision> DecideAsync(TradingContext context, CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        TradingDecision championDecision;
        TradingDecision challengerDecision;
        
        try
        {
            // Convert TradingContext to inputs for UnifiedTradingBrain
            var brainDecision = await GetUnifiedBrainDecisionAsync(context, cancellationToken);
            championDecision = ConvertBrainDecisionToTradingDecision(brainDecision, context, "UnifiedTradingBrain");

            // Get challenger decision in parallel (shadow testing)
            challengerDecision = await _inferenceBrain.DecideAsync(context, cancellationToken);
            
            // Track decision comparison for statistical analysis
            await TrackDecisionComparisonAsync(championDecision, challengerDecision, context);
            
            // Use primary brain's decision (UnifiedTradingBrain by default, InferenceBrain if promoted)
            var primaryDecision = _useInferenceBrainPrimary ? challengerDecision : championDecision;
            var secondaryDecision = _useInferenceBrainPrimary ? championDecision : challengerDecision;
            
            // Add adapter metadata
            primaryDecision.Metadata["AdapterMode"] = _useInferenceBrainPrimary ? "InferenceBrain-Primary" : "UnifiedTradingBrain-Primary";
            primaryDecision.Metadata["ShadowBrainUsed"] = _useInferenceBrainPrimary ? "UnifiedTradingBrain" : "InferenceBrain";
            primaryDecision.Metadata["AgreementRate"] = _agreementRate.ToString("F4");
            primaryDecision.Metadata["ProcessingTimeMs"] = stopwatch.Elapsed.TotalMilliseconds.ToString("F2");
            
            // Check for promotion opportunity (every hour)
            if (DateTime.UtcNow - _lastPromotionCheck > TimeSpan.FromHours(1))
            {
                _ = Task.Run(async () => await CheckPromotionOpportunityAsync(), cancellationToken);
                _lastPromotionCheck = DateTime.UtcNow;
            }
            
            _logger.LogInformation(
                "[ADAPTER] Decision made - Primary: {PrimaryAlgorithm}, Action: {Action}, Agreement: {Agreement}%, Processing: {ProcessingTime}ms",
                _useInferenceBrainPrimary ? "InferenceBrain" : "UnifiedTradingBrain",
                primaryDecision.Action,
                (_agreementRate * 100).ToString("F1"),
                stopwatch.Elapsed.TotalMilliseconds.ToString("F1"));
                
            return primaryDecision;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ADAPTER] Error in decision making, falling back to UnifiedTradingBrain");
            
            // Fallback to UnifiedTradingBrain on any error
            var fallbackDecision = await GetUnifiedBrainDecisionAsync(context, cancellationToken);
            return ConvertBrainDecisionToTradingDecision(fallbackDecision, context, "UnifiedTradingBrain-Fallback");
        }
    }

    /// <summary>
    /// Get decision from UnifiedTradingBrain with proper context conversion
    /// </summary>
    private async Task<BrainDecision> GetUnifiedBrainDecisionAsync(TradingContext context, CancellationToken cancellationToken)
    {
        // Convert TradingContext to UnifiedTradingBrain inputs
        var symbol = context.Symbol ?? "ES";
        
        // Create mock objects for UnifiedTradingBrain (these would come from real data in production)
        var env = new Env 
        { 
            Symbol = symbol,
            CurrentPrice = context.CurrentPrice,
            Timestamp = context.Timestamp
        };
        
        var levels = new Levels
        {
            Support = context.CurrentPrice * 0.998m,
            Resistance = context.CurrentPrice * 1.002m
        };
        
        var bars = new List<Bar>
        {
            new Bar
            {
                Symbol = symbol,
                Timestamp = context.Timestamp,
                Open = context.CurrentPrice,
                High = context.CurrentPrice,
                Low = context.CurrentPrice,
                Close = context.CurrentPrice,
                Volume = 1000
            }
        };
        
        var riskEngine = new RiskEngine();
        
        return await _unifiedBrain.MakeIntelligentDecisionAsync(symbol, env, levels, bars, riskEngine, cancellationToken);
    }

    /// <summary>
    /// Convert BrainDecision to TradingDecision
    /// </summary>
    private TradingDecision ConvertBrainDecisionToTradingDecision(BrainDecision brainDecision, TradingContext context, string algorithmName)
    {
        var tradingDecision = new TradingDecision
        {
            Action = ConvertToTradingAction(brainDecision.SignalAction),
            Confidence = (double)brainDecision.Confidence,
            RiskLevel = brainDecision.RiskLevel.ToString(),
            Reasoning = brainDecision.Reasoning,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>
            {
                ["Algorithm"] = algorithmName,
                ["VersionId"] = brainDecision.VersionId ?? "unknown",
                ["ArtifactHash"] = "legacy-unified-brain",
                ["ProcessingTimeMs"] = brainDecision.ProcessingTimeMs.ToString("F2"),
                ["StrategySelected"] = brainDecision.SelectedStrategy ?? "unknown",
                ["Confidence"] = brainDecision.Confidence.ToString("F4"),
                ["RiskLevel"] = brainDecision.RiskLevel.ToString()
            }
        };

        return tradingDecision;
    }

    /// <summary>
    /// Convert BrainDecision.SignalAction to TradingAction
    /// </summary>
    private TradingAction ConvertToTradingAction(SignalAction signalAction)
    {
        return signalAction switch
        {
            SignalAction.Buy => TradingAction.Buy,
            SignalAction.Sell => TradingAction.Sell,
            SignalAction.Hold => TradingAction.Hold,
            SignalAction.StopLoss => TradingAction.Sell,
            SignalAction.TakeProfit => TradingAction.Sell,
            _ => TradingAction.Hold
        };
    }

    /// <summary>
    /// Track comparison between champion and challenger decisions for statistical analysis
    /// </summary>
    private async Task TrackDecisionComparisonAsync(TradingDecision championDecision, TradingDecision challengerDecision, TradingContext context)
    {
        var comparison = new DecisionComparison
        {
            Timestamp = DateTime.UtcNow,
            Context = context,
            ChampionDecision = championDecision,
            ChallengerDecision = challengerDecision,
            Agreement = AreDecisionsEquivalent(championDecision, challengerDecision),
            ConfidenceDelta = Math.Abs(championDecision.Confidence - challengerDecision.Confidence)
        };

        lock (_comparisonLock)
        {
            _recentComparisons.Add(comparison);
            _totalDecisions++;
            
            if (comparison.Agreement)
                _agreementCount++;
            else
                _disagreementCount++;
            
            // Keep only last 1000 comparisons
            if (_recentComparisons.Count > 1000)
                _recentComparisons.RemoveAt(0);
        }

        // Submit to shadow tester for performance analysis
        await _shadowTester.RecordDecisionAsync(
            challengerDecision.Metadata.GetValueOrDefault("Algorithm", "InferenceBrain").ToString()!,
            context,
            challengerDecision);
    }

    /// <summary>
    /// Check if two decisions are equivalent (same action with similar confidence)
    /// </summary>
    private bool AreDecisionsEquivalent(TradingDecision decision1, TradingDecision decision2)
    {
        const double confidenceThreshold = 0.1; // 10% confidence tolerance
        
        return decision1.Action == decision2.Action && 
               Math.Abs(decision1.Confidence - decision2.Confidence) <= confidenceThreshold;
    }

    /// <summary>
    /// Check if challenger has proven superiority and should be promoted
    /// </summary>
    private async Task CheckPromotionOpportunityAsync()
    {
        try
        {
            if (_recentComparisons.Count < 100) // Need minimum sample size
                return;

            // Get recent shadow test results
            var shadowResults = await _shadowTester.GetRecentResultsAsync("InferenceBrain", TimeSpan.FromHours(24));
            
            if (shadowResults.Count < 50) // Need sufficient shadow test data
                return;

            // Check if promotion criteria are met
            var promotionEvaluation = await _promotionService.EvaluatePromotionAsync(
                "UnifiedTradingBrain", // Current champion
                "InferenceBrain",      // Challenger
                CancellationToken.None);

            if (promotionEvaluation.ShouldPromote)
            {
                _logger.LogWarning(
                    "[ADAPTER] InferenceBrain challenger shows superior performance - " +
                    "Statistical significance: p={PValue:F6}, Sharpe improvement: {SharpeImprovement:F4}",
                    promotionEvaluation.StatisticalSignificance.PValue,
                    promotionEvaluation.PerformanceMetrics.SharpeRatio);

                // This is just evaluation - actual promotion would require manual approval
                _logger.LogInformation("[ADAPTER] Promotion opportunity detected but requires manual approval");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ADAPTER] Error during promotion opportunity check");
        }
    }

    /// <summary>
    /// Manually promote InferenceBrain to primary (for testing/demonstration)
    /// </summary>
    public async Task<bool> PromoteToChallengerAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogWarning("[ADAPTER] Manual promotion to InferenceBrain requested");
            
            // Perform promotion with atomic swap
            _useInferenceBrainPrimary = true;
            
            _logger.LogWarning("[ADAPTER] Promoted InferenceBrain to primary decision maker");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ADAPTER] Failed to promote InferenceBrain");
            return false;
        }
    }

    /// <summary>
    /// Rollback to UnifiedTradingBrain (champion)
    /// </summary>
    public async Task<bool> RollbackToChampionAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogWarning("[ADAPTER] Rollback to UnifiedTradingBrain requested");
            
            // Perform rollback with atomic swap
            _useInferenceBrainPrimary = false;
            
            _logger.LogWarning("[ADAPTER] Rolled back to UnifiedTradingBrain as primary decision maker");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ADAPTER] Failed to rollback to UnifiedTradingBrain");
            return false;
        }
    }

    /// <summary>
    /// Get current adapter statistics
    /// </summary>
    public AdapterStatistics GetStatistics()
    {
        lock (_comparisonLock)
        {
            return new AdapterStatistics
            {
                TotalDecisions = _totalDecisions,
                AgreementCount = _agreementCount,
                DisagreementCount = _disagreementCount,
                AgreementRate = _agreementRate,
                CurrentPrimary = _useInferenceBrainPrimary ? "InferenceBrain" : "UnifiedTradingBrain",
                LastDecisionTime = _recentComparisons.Count > 0 ? _recentComparisons[^1].Timestamp : DateTime.MinValue
            };
        }
    }
}

/// <summary>
/// Statistics for the adapter performance
/// </summary>
public class AdapterStatistics
{
    public int TotalDecisions { get; set; }
    public int AgreementCount { get; set; }
    public int DisagreementCount { get; set; }
    public double AgreementRate { get; set; }
    public string CurrentPrimary { get; set; } = string.Empty;
    public DateTime LastDecisionTime { get; set; }
}

/// <summary>
/// Comparison between champion and challenger decisions
/// </summary>
public class DecisionComparison
{
    public DateTime Timestamp { get; set; }
    public TradingContext Context { get; set; } = null!;
    public TradingDecision ChampionDecision { get; set; } = null!;
    public TradingDecision ChallengerDecision { get; set; } = null!;
    public bool Agreement { get; set; }
    public double ConfidenceDelta { get; set; }
}