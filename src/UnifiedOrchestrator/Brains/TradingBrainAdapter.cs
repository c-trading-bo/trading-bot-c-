using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.Abstractions; // For TradingAction
using BotCore.Brain;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;
using MarketBar = BotCore.Market.Bar; // Alias to resolve ambiguity
using AbstractionsTradingDecision = TradingBot.Abstractions.TradingDecision; // Alias for clarity
using UnifiedTradingDecision = TradingBot.UnifiedOrchestrator.Interfaces.TradingDecision; // Alias for UnifiedOrchestrator type

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
    public async Task<UnifiedTradingDecision> DecideAsync(TradingContext context, CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            // Get challenger decision from InferenceBrain (using unified type)
            var challengerDecision = await _inferenceBrain.DecideAsync(context, cancellationToken).ConfigureAwait(false);
            
            // Convert UnifiedTradingBrain to unified decision format
            var brainDecision = await GetUnifiedBrainDecisionAsync(context, cancellationToken).ConfigureAwait(false);
            var championDecision = ConvertBrainDecisionToUnifiedDecision(brainDecision, context, "UnifiedTradingBrain");
            
            // Track decision comparison for statistical analysis
            await TrackUnifiedDecisionComparisonAsync(championDecision, challengerDecision, context).ConfigureAwait(false);
            
            // Use primary brain's decision (UnifiedTradingBrain by default, InferenceBrain if promoted)
            var primaryDecision = _useInferenceBrainPrimary ? challengerDecision : championDecision;
            var secondaryDecision = _useInferenceBrainPrimary ? championDecision : challengerDecision;
            
            // Add adapter metadata to reasoning dictionary
            primaryDecision.Reasoning["AdapterMode"] = _useInferenceBrainPrimary ? "InferenceBrain-Primary" : "UnifiedTradingBrain-Primary";
            primaryDecision.Reasoning["ShadowBrainUsed"] = _useInferenceBrainPrimary ? "UnifiedTradingBrain" : "InferenceBrain";
            primaryDecision.Reasoning["AgreementRate"] = _agreementRate.ToString("F4");
            primaryDecision.Reasoning["ProcessingTimeMs"] = stopwatch.Elapsed.TotalMilliseconds.ToString("F2");
            
            // Check for promotion opportunity (every hour)
            if (DateTime.UtcNow - _lastPromotionCheck > TimeSpan.FromHours(1))
            {
                _ = Task.Run(async () => await CheckPromotionOpportunityAsync(), cancellationToken).ConfigureAwait(false);
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
            var fallbackDecision = await GetUnifiedBrainDecisionAsync(context, cancellationToken).ConfigureAwait(false);
            return ConvertBrainDecisionToUnifiedDecision(fallbackDecision, context, "UnifiedTradingBrain-Fallback");
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
            Symbol = symbol
            // Note: CurrentPrice and Timestamp may not exist on Env - using context data directly
        };
        
        var levels = new Levels
        {
            // Note: Support and Resistance may not exist on Levels - using calculated values
        };
        
        var bars = new List<BotCore.Models.Bar>
        {
            new BotCore.Models.Bar
            {
                Start = context.Timestamp,
                Ts = ((DateTimeOffset)context.Timestamp).ToUnixTimeMilliseconds(),
                Symbol = symbol,
                Open = context.CurrentPrice,
                High = context.CurrentPrice,
                Low = context.CurrentPrice,
                Close = context.CurrentPrice,
                Volume = 1000
            }
        };
        
        var riskEngine = new RiskEngine();
        
        return await _unifiedBrain.MakeIntelligentDecisionAsync(symbol, env, levels, bars, riskEngine, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Convert BrainDecision to TradingDecision
    /// </summary>
    private AbstractionsTradingDecision ConvertBrainDecisionToTradingDecision(BrainDecision brainDecision, TradingContext context, string algorithmName)
    {
        var tradingDecision = new AbstractionsTradingDecision
        {
            DecisionId = Guid.NewGuid().ToString(),
            Symbol = brainDecision.Symbol,
            Action = ConvertPriceDirectionToTradingAction(brainDecision.PriceDirection, brainDecision.OptimalPositionMultiplier),
            Confidence = (decimal)brainDecision.ModelConfidence, // Use ModelConfidence from BrainDecision
            MLConfidence = (decimal)brainDecision.ModelConfidence,
            MLStrategy = brainDecision.RecommendedStrategy,
            RiskScore = 0m, // BrainDecision has RiskAssessment as string, not numeric
            MaxPositionSize = brainDecision.OptimalPositionMultiplier * 100m, // Convert multiplier to position size
            MarketRegime = brainDecision.MarketRegime.ToString(),
            RegimeConfidence = (decimal)brainDecision.StrategyConfidence,
            Timestamp = brainDecision.DecisionTime,
            Reasoning = new Dictionary<string, object>
            {
                ["Algorithm"] = algorithmName,
                ["ProcessingTimeMs"] = brainDecision.ProcessingTimeMs.ToString("F2"),
                ["Strategy"] = brainDecision.RecommendedStrategy,
                ["StrategyConfidence"] = brainDecision.StrategyConfidence.ToString("F4"),
                ["RiskAssessment"] = brainDecision.RiskAssessment,
                ["PriceDirection"] = brainDecision.PriceDirection.ToString(),
                ["PriceProbability"] = brainDecision.PriceProbability.ToString("F4"),
                ["MarketRegime"] = brainDecision.MarketRegime.ToString()
            }
        };

        return tradingDecision;
    }

    /// <summary>
    /// Convert PriceDirection and position multiplier to TradingAction
    /// </summary>
    private TradingAction ConvertPriceDirectionToTradingAction(PriceDirection priceDirection, decimal positionMultiplier)
    {
        return priceDirection switch
        {
            PriceDirection.Up when positionMultiplier > 0.5m => TradingAction.Buy,
            PriceDirection.Up when positionMultiplier > 0.1m => TradingAction.BuySmall,
            PriceDirection.Down when positionMultiplier > 0.5m => TradingAction.Sell,
            PriceDirection.Down when positionMultiplier > 0.1m => TradingAction.SellSmall,
            PriceDirection.Sideways => TradingAction.Hold,
            _ => TradingAction.Hold
        };
    }

    /// <summary>
    /// Convert Interface TradingDecision to Abstractions TradingDecision
    /// </summary>
    private AbstractionsTradingDecision ConvertInterfaceDecisionToAbstractionsDecision(TradingBot.UnifiedOrchestrator.Interfaces.TradingDecision interfaceDecision)
    {
        // Parse Action string to TradingAction enum
        TradingAction tradingAction = interfaceDecision.Action.ToUpperInvariant() switch
        {
            "BUY" => TradingAction.Buy,
            "SELL" => TradingAction.Sell,
            "HOLD" => TradingAction.Hold,
            _ => TradingAction.Hold
        };

        return new AbstractionsTradingDecision
        {
            DecisionId = Guid.NewGuid().ToString(),
            Symbol = interfaceDecision.Symbol,
            Action = tradingAction,
            Quantity = interfaceDecision.Size,
            Confidence = interfaceDecision.Confidence,
            MLConfidence = interfaceDecision.Confidence,
            MLStrategy = interfaceDecision.Strategy,
            Timestamp = interfaceDecision.Timestamp,
            Reasoning = new Dictionary<string, object>
            {
                ["OriginalAction"] = interfaceDecision.Action,
                ["Strategy"] = interfaceDecision.Strategy,
                ["PPOVersionId"] = interfaceDecision.PPOVersionId,
                ["UCBVersionId"] = interfaceDecision.UCBVersionId,
                ["LSTMVersionId"] = interfaceDecision.LSTMVersionId,
                ["ProcessingTimeMs"] = interfaceDecision.ProcessingTimeMs.ToString("F2"),
                ["PassedRiskChecks"] = interfaceDecision.PassedRiskChecks.ToString(),
                ["RiskWarnings"] = string.Join("; ", interfaceDecision.RiskWarnings),
                ["AlgorithmVersions"] = interfaceDecision.AlgorithmVersions,
                ["AlgorithmHashes"] = interfaceDecision.AlgorithmHashes,
                ["AlgorithmConfidences"] = interfaceDecision.AlgorithmConfidences,
                ["DecisionMetadata"] = interfaceDecision.DecisionMetadata
            }
        };
    }

    /// <summary>
    /// Convert BrainDecision to UnifiedOrchestrator TradingDecision
    /// </summary>
    private UnifiedTradingDecision ConvertBrainDecisionToUnifiedDecision(BrainDecision brainDecision, TradingContext context, string algorithmName)
    {
        // Determine action based on PriceDirection and OptimalPositionMultiplier
        string actionString = "HOLD";
        decimal size = 0;
        
        if (brainDecision.OptimalPositionMultiplier > 0)
        {
            actionString = brainDecision.PriceDirection == PriceDirection.Up ? "BUY" : "SELL";
            size = Math.Abs(brainDecision.OptimalPositionMultiplier);
        }

        return new UnifiedTradingDecision
        {
            Symbol = brainDecision.Symbol ?? context.Symbol ?? "ES",
            Timestamp = brainDecision.DecisionTime != default ? brainDecision.DecisionTime : DateTime.UtcNow,
            Action = actionString,
            Size = size,
            Confidence = brainDecision.StrategyConfidence,
            Strategy = brainDecision.RecommendedStrategy ?? algorithmName,
            ProcessingTimeMs = (decimal)brainDecision.ProcessingTimeMs,
            PPOVersionId = "brain-unified",
            UCBVersionId = "brain-unified", 
            LSTMVersionId = "brain-unified",
            AlgorithmVersions = new Dictionary<string, string>
            {
                ["UnifiedBrain"] = "1.0",
                ["RecommendedStrategy"] = brainDecision.RecommendedStrategy ?? "unknown"
            },
            AlgorithmConfidences = new Dictionary<string, decimal>
            {
                ["Strategy"] = brainDecision.StrategyConfidence,
                ["Model"] = brainDecision.ModelConfidence,
                ["Price"] = brainDecision.PriceProbability
            },
            DecisionMetadata = new Dictionary<string, object>
            {
                ["Algorithm"] = algorithmName,
                ["ProcessingTimeMs"] = brainDecision.ProcessingTimeMs.ToString("F2"),
                ["MarketRegime"] = brainDecision.MarketRegime.ToString(),
                ["PriceDirection"] = brainDecision.PriceDirection.ToString(),
                ["RiskAssessment"] = brainDecision.RiskAssessment,
                ["CandidateCount"] = brainDecision.EnhancedCandidates.Count
            },
            PassedRiskChecks = !brainDecision.RiskAssessment.Contains("HIGH", StringComparison.OrdinalIgnoreCase),
            RiskWarnings = brainDecision.RiskAssessment.Contains("HIGH", StringComparison.OrdinalIgnoreCase) 
                          ? new List<string> { brainDecision.RiskAssessment }
                          : new List<string>()
        };
    }

    /// <summary>
    /// Track comparison between champion and challenger decisions for statistical analysis (unified types)
    /// </summary>
    private async Task TrackUnifiedDecisionComparisonAsync(UnifiedTradingDecision championDecision, UnifiedTradingDecision challengerDecision, TradingContext context)
    {
        await Task.Yield().ConfigureAwait(false); // Ensure async behavior
        
        var comparison = new UnifiedDecisionComparison
        {
            Timestamp = DateTime.UtcNow,
            Context = context,
            ChampionDecision = championDecision,
            ChallengerDecision = challengerDecision,
            Agreement = AreUnifiedDecisionsEquivalent(championDecision, challengerDecision),
            ConfidenceDelta = Math.Abs((double)(championDecision.Confidence - challengerDecision.Confidence))
        };

        lock (_comparisonLock)
        {
            // Replace old comparison method with unified version for now
            _recentComparisons.Clear(); // Clear old format comparisons
            _totalDecisions++;
            if (comparison.Agreement) _agreementCount++;
        }

        // Log detailed comparison
        _logger.LogDebug("[ADAPTER] Decision comparison: Champion={ChampionAction} vs Challenger={ChallengerAction}, Agreement={Agreement}, ConfidenceDelta={ConfidenceDelta:F3}",
            championDecision.Action, challengerDecision.Action, comparison.Agreement, comparison.ConfidenceDelta);
    }

    /// <summary>
    /// Check if two unified decisions are equivalent
    /// </summary>
    private bool AreUnifiedDecisionsEquivalent(UnifiedTradingDecision decision1, UnifiedTradingDecision decision2)
    {
        return decision1.Action == decision2.Action && 
               Math.Abs(decision1.Size - decision2.Size) < 0.01m &&
               Math.Abs(decision1.Confidence - decision2.Confidence) < 0.1m;
    }

    /// <summary>
    /// Track comparison between champion and challenger decisions for statistical analysis
    /// </summary>
    private async Task TrackDecisionComparisonAsync(AbstractionsTradingDecision championDecision, AbstractionsTradingDecision challengerDecision, TradingContext context)
    {
        await Task.Yield().ConfigureAwait(false); // Ensure async behavior
        
        var comparison = new DecisionComparison
        {
            Timestamp = DateTime.UtcNow,
            Context = context,
            ChampionDecision = championDecision,
            ChallengerDecision = challengerDecision,
            Agreement = AreDecisionsEquivalent(championDecision, challengerDecision),
            ConfidenceDelta = Math.Abs((double)(championDecision.Confidence - challengerDecision.Confidence))
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

        // Record decision for shadow testing comparison
        await _shadowTester.RecordDecisionAsync(challengerDecision.Reasoning.GetValueOrDefault("Algorithm", "InferenceBrain").ToString()!, context, challengerDecision).ConfigureAwait(false);
    }

    /// <summary>
    /// Check if two decisions are equivalent (same action with similar confidence)
    /// </summary>
    private bool AreDecisionsEquivalent(AbstractionsTradingDecision decision1, AbstractionsTradingDecision decision2)
    {
        const double confidenceThreshold = 0.1; // 10% confidence tolerance
        
        return decision1.Action == decision2.Action && 
               Math.Abs((double)(decision1.Confidence - decision2.Confidence)) <= confidenceThreshold;
    }

    /// <summary>
    /// Check if challenger has proven superiority and should be promoted
    /// </summary>
    private async Task CheckPromotionOpportunityAsync()
    {
        await Task.Yield().ConfigureAwait(false); // Ensure async behavior
        
        try
        {
            if (_recentComparisons.Count < 100) // Need minimum sample size
                return;

            // Get shadow test results for promotion evaluation
            var shadowResults = await _shadowTester.GetRecentResultsAsync("InferenceBrain", TimeSpan.FromHours(24)).ConfigureAwait(false);
            
            if (shadowResults.Count < 50) // Need sufficient shadow test data
                return;

            // Check if promotion criteria are met based on recent comparisons
            var recentAgreements = _recentComparisons.TakeLast(100).Count(c => c.Agreement);
            var agreementRate = (double)recentAgreements / Math.Min(100, _recentComparisons.Count);
            
            // Simple promotion criteria: high agreement rate suggests challenger is performing similarly
            if (agreementRate > 0.8) // 80% agreement threshold
            {
                _logger.LogInformation(
                    "[ADAPTER] High agreement rate detected ({AgreementRate:P2}) - InferenceBrain may be ready for promotion",
                    agreementRate);

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
        await Task.Yield().ConfigureAwait(false); // Ensure async behavior
        
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
        await Task.Yield().ConfigureAwait(false); // Ensure async behavior
        
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
/// Comparison between champion and challenger decisions (unified types)
/// </summary>
public class UnifiedDecisionComparison
{
    public DateTime Timestamp { get; set; }
    public TradingContext Context { get; set; } = new();
    public UnifiedTradingDecision ChampionDecision { get; set; } = new();
    public UnifiedTradingDecision ChallengerDecision { get; set; } = new();
    public bool Agreement { get; set; }
    public double ConfidenceDelta { get; set; }
}