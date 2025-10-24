using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Brain;
using BotCore.Brain.Models;
using BotCore.Services;
using BotCore.Risk;
using BotCore.Models;
using BotCore.Fusion;
using BotCore.Strategy;
using TradingBot.Abstractions;
using TradingBot.IntelligenceStack;
using TradingBot.RLAgent;
using System.Text.Json;
using System.Linq;
using static BotCore.Brain.UnifiedTradingBrain;

// Explicit type alias to resolve Bar ambiguity  
using ModelBar = BotCore.Models.Bar;

namespace BotCore.Services;

/// <summary>
/// 🎯 UNIFIED DECISION ROUTER - SAFE FALLBACK ON SYSTEM FAILURES 🎯
/// 
/// Creates cascading decision system that tries each brain in order:
/// 1. EnhancedBrainIntegration: Multi-model ensemble with cloud learning
/// 2. UnifiedTradingBrain: Neural UCB strategy selection + CVaR-PPO sizing
/// 3. IntelligenceOrchestrator: Basic ML/RL fallback
/// 
/// SAFETY: Returns HOLD in emergency when all systems fail to preserve safety
/// RESULT: Reliable trading actions with safe standdown on system failures
/// </summary>
public class UnifiedDecisionRouter
{
    // Floating point comparison tolerance
    private const double Epsilon = 1e-10;
    
    // Trading analysis constants
    private const int MAX_DECISION_HISTORY = 1000;          // Maximum decisions to retain in memory
    
    // Trading schedule constants (UTC hours)
    private const int OPENING_DRIVE_START_HOUR = 9;         // Opening drive strategy start time
    private const int OPENING_DRIVE_END_HOUR = 10;          // Opening drive strategy end time
    private const int LUNCH_MEAN_REVERSION_START = 11;      // Lunch mean reversion start time
    private const int LUNCH_MEAN_REVERSION_END = 13;        // Lunch mean reversion end time
    private const int AFTERNOON_TRADING_START = 14;         // Afternoon trading start time
    private const int AFTERNOON_TRADING_END = 16;           // Afternoon trading end time
    
    // Price level calculation constants (ticks)
    private const decimal SUPPORT_RESISTANCE_TICK_OFFSET_1 = 10;  // First level support/resistance offset
    private const decimal SUPPORT_RESISTANCE_TICK_OFFSET_2 = 20;  // Second level support/resistance offset
    private const decimal SUPPORT_RESISTANCE_TICK_OFFSET_3 = 30;  // Third level support/resistance offset
    private const int WEEKLY_PIVOT_OFFSET_TICKS = 5;              // Weekly pivot offset in ticks
    private const int MONTHLY_PIVOT_OFFSET_TICKS = 5;             // Monthly pivot offset in ticks
    
    // Environment defaults
    private const decimal DEFAULT_ATR_VALUE = 5.0m;               // Default ATR value when not available
    private const decimal DEFAULT_VOLUME_Z_VALUE = 0.5m;          // Default volume z-score
    
    // Bar synthesis constants
    private const int SYNTHETIC_BAR_COUNT = 10;                   // Number of synthetic bars to generate
    private const decimal SYNTHETIC_BAR_VARIATION = 2m;           // Max price variation for synthetic bars
    private const decimal SYNTHETIC_BAR_RANGE = 1m;               // High-low range for synthetic bars
    private const decimal MIN_VOLUME_MULTIPLIER = 0.8m;           // Minimum volume multiplier
    private const decimal VOLUME_MULTIPLIER_RANGE = 0.4m;         // Volume multiplier range
    
    // Risk engine defaults
    private const decimal DEFAULT_RISK_PER_TRADE = 100m;          // Default risk per trade
    private const decimal DEFAULT_MAX_DAILY_DRAWDOWN = 1000m;     // Default max daily drawdown
    private const int DEFAULT_MAX_OPEN_POSITIONS = 1;             // Default max open positions
    
    // Emergency state constants
    private const decimal EMERGENCY_CONFIDENCE = 0.0m;            // No confidence in emergency state
    private const decimal EMERGENCY_QUANTITY = 0m;                // No position sizing in emergency
    
    // Decision ID generation constants
    private const int DECISION_ID_RANDOM_MIN = 1000;              // Minimum random number for decision ID
    private const int DECISION_ID_RANDOM_MAX = 9999;              // Maximum random number for decision ID

    private readonly ILogger<UnifiedDecisionRouter> _logger;
    
    // AI Brain components in priority order - Strategy Knowledge Graph & Decision Fusion is highest priority
    private readonly DecisionFusionCoordinator? _decisionFusion;
    private readonly EnhancedTradingBrainIntegration _enhancedBrain;
    private readonly UnifiedTradingBrain _unifiedBrain;
    private readonly TradingBot.Abstractions.IIntelligenceOrchestrator? _intelligenceOrchestrator;
    
    // Strategy routing configuration
    // Removed _random field - using System.Security.Cryptography.RandomNumberGenerator for secure randomness
    
    // Decision tracking for learning
    private readonly List<DecisionOutcome> _decisionHistory = new();
    private readonly object _historyLock = new();
    
    public UnifiedDecisionRouter(
        ILogger<UnifiedDecisionRouter> logger,
        IServiceProvider serviceProvider,
        UnifiedTradingBrain unifiedBrain,
        EnhancedTradingBrainIntegration enhancedBrain)
    {
        _logger = logger;
        _unifiedBrain = unifiedBrain;
        _enhancedBrain = enhancedBrain;
        
        // Try to get IntelligenceOrchestrator (optional, removed as dead code but keep for backward compatibility)
        try
        {
            _intelligenceOrchestrator = serviceProvider.GetService<TradingBot.Abstractions.IIntelligenceOrchestrator>();
            if (_intelligenceOrchestrator != null)
            {
                _logger.LogInformation("🤖 [INTELLIGENCE-ORCHESTRATOR] Legacy IntelligenceOrchestrator available");
            }
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogWarning(ex, "IntelligenceOrchestrator service resolution failed, continuing without it");
            _intelligenceOrchestrator = null;
        }
        
        // Try to get Decision Fusion Coordinator (optional, may not be registered in all environments)
        try
        {
            _decisionFusion = serviceProvider.GetService<DecisionFusionCoordinator>();
            if (_decisionFusion != null)
            {
                _logger.LogInformation("🧠 [DECISION-FUSION] Strategy Knowledge Graph & Decision Fusion system available");
            }
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogWarning(ex, "Decision Fusion Coordinator service resolution failed, continuing without it");
            _decisionFusion = null;
        }
        
        _logger.LogInformation("🎯 [DECISION-ROUTER] Unified Decision Router initialized");
        _logger.LogInformation("📊 [DECISION-ROUTER] Services wired: Fusion={FusionAvailable}, Enhanced=True, Unified=True, Intelligence={IntelligenceAvailable}", 
            _decisionFusion != null, _intelligenceOrchestrator != null);
    }
    
    /// <summary>
    /// Main decision routing method - Returns safe HOLD when all systems fail
    /// </summary>
    public async Task<UnifiedTradingDecision> RouteDecisionAsync(
        string symbol,
        TradingBot.Abstractions.MarketContext marketContext,
        CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;
        var decisionId = GenerateDecisionId();
        
        try
        {
            _logger.LogDebug("🎯 [DECISION-ROUTER] Routing decision for {Symbol}", symbol);
            
            // Step 0: Try Strategy Knowledge Graph & Decision Fusion (Highest Priority)
            if (_decisionFusion != null)
            {
                var decision = await TryDecisionFusionAsync(symbol, marketContext, cancellationToken).ConfigureAwait(false);
                if (decision != null && decision.Action != TradingAction.Hold)
                {
                    decision.DecisionId = decisionId;
                    decision.DecisionSource = "StrategyFusion";
                    decision.ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
                    
                    await TrackDecisionAsync(decision, "strategy_fusion").ConfigureAwait(false);
                    _logger.LogInformation("🧠 [STRATEGY-FUSION] Decision: {Action} {Symbol} strategy={Strategy} confidence={Confidence:P1}",
                        decision.Action, symbol, decision.Strategy ?? "Unknown", decision.Confidence);
                    return decision;
                }
            }
            
            // Step 1: Try Enhanced Brain Integration (Secondary)
            var decision2 = await TryEnhancedBrainAsync(symbol, marketContext, cancellationToken).ConfigureAwait(false);
            if (decision2 != null && decision2.Action != TradingAction.Hold)
            {
                decision2.DecisionId = decisionId;
                decision2.DecisionSource = "EnhancedBrain";
                decision2.ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
                
                await TrackDecisionAsync(decision2, "enhanced_brain").ConfigureAwait(false);
                _logger.LogInformation("🧠 [ENHANCED-BRAIN] Decision: {Action} {Symbol} confidence={Confidence:P1}",
                    decision2.Action, symbol, decision2.Confidence);
                return decision2;
            }
            
            // Step 2: Try Unified Trading Brain (Tertiary)
            var decision3 = await TryUnifiedBrainAsync(symbol, marketContext, cancellationToken).ConfigureAwait(false);
            if (decision3 != null && decision3.Action != TradingAction.Hold)
            {
                decision3.DecisionId = decisionId;
                decision3.DecisionSource = "UnifiedBrain";
                decision3.ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
                
                await TrackDecisionAsync(decision3, "unified_brain").ConfigureAwait(false);
                _logger.LogInformation("🎯 [UNIFIED-BRAIN] Decision: {Action} {Symbol} confidence={Confidence:P1}",
                    decision3.Action, symbol, decision3.Confidence);
                return decision3;
            }
            
            // Step 3: Try Intelligence Orchestrator (Fallback)
            var decision4 = await TryIntelligenceOrchestratorAsync(marketContext, cancellationToken).ConfigureAwait(false);
            if (decision4 != null && decision4.Action != TradingAction.Hold)
            {
                decision4.DecisionId = decisionId;
                decision4.DecisionSource = "IntelligenceOrchestrator";
                decision4.ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
                
                await TrackDecisionAsync(decision4, "intelligence_orchestrator").ConfigureAwait(false);
                _logger.LogInformation("🤖 [INTELLIGENCE-ORCHESTRATOR] Decision: {Action} {Symbol} confidence={Confidence:P1}",
                    decision4.Action, symbol, decision4.Confidence);
                return decision4;
            }
            
            // Step 4: ULTIMATE FALLBACK - Stand down when all brains return HOLD
            // This ensures upstream failures are not masked by forcing arbitrary decisions
            var standdownDecision = CreateStanddownDecision(symbol);
            standdownDecision.DecisionId = decisionId;
            standdownDecision.DecisionSource = "SystemStanddown";
            standdownDecision.ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
            
            await TrackDecisionAsync(standdownDecision, "system_standdown").ConfigureAwait(false);
            _logger.LogWarning("⚠️ [SYSTEM-STANDDOWN] All brains returned HOLD, standing down for: {Symbol}",
                symbol);
            return standdownDecision;
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "❌ [DECISION-ROUTER] Invalid arguments for decision routing: {Symbol}", symbol);
            
            // Emergency fallback
            var emergencyDecision = CreateEmergencyDecision(symbol);
            emergencyDecision.DecisionId = decisionId;
            emergencyDecision.DecisionSource = "Emergency";
            emergencyDecision.ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
            
            return emergencyDecision;
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [DECISION-ROUTER] Operation error during decision routing: {Symbol}", symbol);
            
            // Emergency fallback
            var emergencyDecision = CreateEmergencyDecision(symbol);
            emergencyDecision.DecisionId = decisionId;
            emergencyDecision.DecisionSource = "Emergency";
            emergencyDecision.ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
            
            return emergencyDecision;
        }
    }
    
    /// <summary>
    /// Try Enhanced Brain Integration - Multi-model ensemble with cloud learning
    /// </summary>
    /// <summary>
    /// Try Strategy Knowledge Graph & Decision Fusion - Highest priority ML-enhanced decision making
    /// </summary>
    private async Task<UnifiedTradingDecision?> TryDecisionFusionAsync(
        string symbol,
        TradingBot.Abstractions.MarketContext marketContext,
        CancellationToken cancellationToken)
    {
        try
        {
            // Get fusion recommendation asynchronously
            var fusionRecommendation = await _decisionFusion!.DecideAsync(symbol, cancellationToken).ConfigureAwait(false);
            
            if (fusionRecommendation != null)
            {
                // Convert Strategy Recommendation to UnifiedTradingDecision
                var tradingAction = fusionRecommendation.Intent == BotCore.Strategy.StrategyIntent.Buy 
                    ? TradingAction.Buy 
                    : TradingAction.Sell;
                
                var decision = new UnifiedTradingDecision
                {
                    Symbol = symbol,
                    Action = tradingAction,
                    Confidence = (decimal)fusionRecommendation.Confidence,
                    Strategy = fusionRecommendation.StrategyName,
                    Timestamp = DateTime.UtcNow,
                    Reasoning = new Dictionary<string, object>
                    {
                        ["evidence_count"] = fusionRecommendation.Evidence.Count,
                        ["evidence_factors"] = fusionRecommendation.Evidence.Select(e => e.Name).ToList(),
                        ["telemetry_tags"] = fusionRecommendation.TelemetryTags.ToList(),
                        ["strategy_intent"] = fusionRecommendation.Intent.ToString()
                    }
                };

                _logger.LogDebug("🧠 [STRATEGY-FUSION] Fusion recommendation: {Strategy} {Intent} confidence={Confidence:F2}",
                    fusionRecommendation.StrategyName, fusionRecommendation.Intent, fusionRecommendation.Confidence);
                
                return decision;
            }
            
            _logger.LogTrace("🧠 [STRATEGY-FUSION] No fusion recommendation for {Symbol}", symbol);
            return null;
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogWarning(ex, "⚠️ [STRATEGY-FUSION] Fusion service operation error for {Symbol}", symbol);
            return null;
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "⚠️ [STRATEGY-FUSION] Invalid arguments for fusion decision: {Symbol}", symbol);
            return null;
        }
    }

    /// <summary>
    /// Try Enhanced Brain Integration - Multi-model ensemble
    /// </summary>
    private async Task<UnifiedTradingDecision?> TryEnhancedBrainAsync(
        string symbol,
        TradingBot.Abstractions.MarketContext marketContext,
        CancellationToken cancellationToken)
    {
        try
        {
            // Convert MarketContext to EnhancedBrain format
            var enhancedContext = ConvertToEnhancedContext(marketContext);
            var availableStrategies = GetAvailableStrategies();
            
            var enhancedDecision = await _enhancedBrain.MakeEnhancedDecisionAsync(
                symbol, enhancedContext, availableStrategies, cancellationToken).ConfigureAwait(false);
            
            if (enhancedDecision?.EnhancementApplied == true)
            {
                return ConvertFromEnhancedDecision(enhancedDecision);
            }
            
            return null;
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogWarning(ex, "⚠️ [ENHANCED-BRAIN] Brain operation error");
            return null;
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "⚠️ [ENHANCED-BRAIN] Invalid arguments for decision");
            return null;
        }
    }
    
    /// <summary>
    /// Try Unified Trading Brain - Neural UCB + CVaR-PPO + LSTM
    /// </summary>
    private async Task<UnifiedTradingDecision?> TryUnifiedBrainAsync(
        string symbol,
        TradingBot.Abstractions.MarketContext marketContext,
        CancellationToken cancellationToken)
    {
        try
        {
            // Convert MarketContext to UnifiedBrain format
            var env = ConvertToEnv(marketContext);
            var levels = ConvertToLevels(marketContext);
            var bars = ConvertToBars(marketContext);
            using var risk = CreateRiskEngine();
            
            var brainDecision = await _unifiedBrain.MakeIntelligentDecisionAsync(
                symbol, env, levels, bars, risk, null, cancellationToken).ConfigureAwait(false);
            
            return ConvertFromBrainDecision(brainDecision);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogWarning(ex, "⚠️ [UNIFIED-BRAIN] Brain operation error");
            return null;
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "⚠️ [UNIFIED-BRAIN] Invalid arguments for decision");
            return null;
        }
    }
    
    /// <summary>
    /// Try Intelligence Orchestrator - Basic ML/RL fallback
    /// </summary>
    private async Task<UnifiedTradingDecision?> TryIntelligenceOrchestratorAsync(
                TradingBot.Abstractions.MarketContext marketContext,
        CancellationToken cancellationToken)
    {
        // IntelligenceOrchestrator has been removed as dead code - return null to skip this fallback
        if (_intelligenceOrchestrator == null)
        {
            return null;
        }
        
        try
        {
            var abstractionContext = EnhanceMarketContext(marketContext);
            var decision = await _intelligenceOrchestrator.MakeDecisionAsync(abstractionContext, cancellationToken).ConfigureAwait(false);
            
            return ConvertFromAbstractionDecision(decision);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogWarning(ex, "⚠️ [INTELLIGENCE-ORCHESTRATOR] Orchestrator operation error");
            return null;
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "⚠️ [INTELLIGENCE-ORCHESTRATOR] Invalid arguments for decision");
            return null;
        }
    }
    
    /// <summary>
    /// Create standdown decision when all brains return HOLD - allows system to stand down gracefully
    /// </summary>
    private static UnifiedTradingDecision CreateStanddownDecision(string symbol)
    {
        return new UnifiedTradingDecision
        {
            Symbol = symbol,
            Action = TradingAction.Hold,
            Confidence = 0m, // No confidence for standdown
            Quantity = 0m, // No quantity for standdown
            Strategy = "SYSTEM_STANDDOWN",
            Reasoning = new Dictionary<string, object>
            {
                ["source"] = "System standdown - all brains returned HOLD",
                ["safety"] = "Standing down prevents masking upstream failures",
                ["action"] = "No trading action taken"
            },
            Timestamp = DateTime.UtcNow
        };
    }
    
    /// <summary>
    /// Create emergency decision - Last resort fallback that safely stands down
    /// </summary>
    private static UnifiedTradingDecision CreateEmergencyDecision(string symbol)
    {
        // Emergency decisions should stand down safely rather than mask upstream faults
        // Return minimal action that preserves safety
        return new UnifiedTradingDecision
        {
            Symbol = symbol,
            Action = TradingAction.Hold, // Safe default - don't trade when all systems fail
            Confidence = EMERGENCY_CONFIDENCE, // No confidence in emergency state
            Quantity = EMERGENCY_QUANTITY, // No position sizing in emergency
            Strategy = "EMERGENCY_STANDDOWN",
            Reasoning = new Dictionary<string, object>
            {
                ["source"] = "Emergency standdown - all decision systems failed",
                ["rationale"] = "Preserving safety by avoiding trades during system failures",
                ["safety"] = "No trading action - system standdown mode"
            },
            Timestamp = DateTime.UtcNow
        };
    }
    
    /// <summary>
    /// Analyze market conditions for forced decision making
    /// </summary>
    /// <summary>
    /// Track decision for learning and performance analysis
    /// </summary>
    private Task TrackDecisionAsync(UnifiedTradingDecision decision, string source)
    {
        try
        {
            var outcome = new DecisionOutcome
            {
                DecisionId = decision.DecisionId,
                Source = source,
                Symbol = decision.Symbol,
                Action = decision.Action,
                Confidence = decision.Confidence,
                Timestamp = decision.Timestamp,
                Strategy = decision.Strategy
            };
            
            lock (_historyLock)
            {
                _decisionHistory.Add(outcome);
                
                // Keep only last decisions in memory for performance
                if (_decisionHistory.Count > MAX_DECISION_HISTORY)
                {
                    _decisionHistory.RemoveAt(0);
                }
            }
            
            // Log decision tracking
            _logger.LogDebug("📊 [DECISION-TRACKING] Tracked {Source} decision: {Action} {Symbol}", 
                source, decision.Action, decision.Symbol);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [DECISION-TRACKING] Tracking service operation error");
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "❌ [DECISION-TRACKING] Invalid decision for tracking");
        }

        return Task.CompletedTask;
    }
    
    /// <summary>
    /// Submit trading outcome for learning improvement
    /// </summary>
    public async Task SubmitTradingOutcomeAsync(
        string decisionId,
        decimal realizedPnL,
        bool wasCorrect,
        TimeSpan holdTime,
        CancellationToken cancellationToken = default)
    {
        try
        {
            DecisionOutcome? outcome;
            
            lock (_historyLock)
            {
                outcome = _decisionHistory.Find(d => d.DecisionId == decisionId);
            }
            
            if (outcome == null)
            {
                _logger.LogWarning("⚠️ [DECISION-FEEDBACK] Decision {DecisionId} not found in history", decisionId);
                return;
            }
            
            // Update outcome with results
            outcome.RealizedPnL = realizedPnL;
            outcome.WasCorrect = wasCorrect;
            outcome.HoldTime = holdTime;
            outcome.OutcomeReceived = true;
            
            // Submit feedback to the appropriate brain
            await SubmitFeedbackToBrainAsync(outcome, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("📈 [DECISION-FEEDBACK] Outcome submitted: {DecisionId} PnL={PnL:C2} Correct={Correct}",
                decisionId, realizedPnL, wasCorrect);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [DECISION-FEEDBACK] Feedback service operation error");
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "❌ [DECISION-FEEDBACK] Invalid outcome data");
        }
    }
    
    /// <summary>
    /// Submit feedback to the brain that made the decision
    /// </summary>
    private async Task SubmitFeedbackToBrainAsync(DecisionOutcome outcome, CancellationToken cancellationToken)
    {
        try
        {
            switch (outcome.Source)
            {
                case "enhanced_brain":
                    _enhancedBrain.SubmitTradingOutcome(
                        outcome.Symbol,
                        outcome.Strategy,
                        outcome.Action.ToString(),
                        outcome.RealizedPnL,
                        new Dictionary<string, object>
                        {
                            ["decision_id"] = outcome.DecisionId,
                            ["hold_time"] = outcome.HoldTime.TotalMinutes,
                            ["was_correct"] = outcome.WasCorrect
                        });
                    break;
                    
                case "unified_brain":
                    await _unifiedBrain.LearnFromResultAsync(
                        outcome.Symbol,
                        outcome.Strategy,
                        outcome.RealizedPnL,
                        outcome.WasCorrect,
                        outcome.HoldTime,
                        cancellationToken).ConfigureAwait(false);
                    break;
                    
                case "intelligence_orchestrator":
                    // Intelligence orchestrator learning would be implemented here
                    break;
                    
                default:
                    _logger.LogDebug("🔇 [DECISION-FEEDBACK] No feedback handler for source: {Source}", outcome.Source);
                    break;
            }
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [DECISION-FEEDBACK] Brain feedback operation error for {Source}", outcome.Source);
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "❌ [DECISION-FEEDBACK] Invalid feedback data for {Source}", outcome.Source);
        }
    }
    
    /// <summary>
    /// Get performance statistics for each brain
    /// </summary>
    public DecisionRouterStats GetPerformanceStats()
    {
        lock (_historyLock)
        {
            var stats = new DecisionRouterStats();
            
            foreach (var group in _decisionHistory.Where(d => d.OutcomeReceived).GroupBy(d => d.Source))
            {
                var decisions = group.ToList();
                var sourceStats = new SourceStats
                {
                    Source = group.Key,
                    TotalDecisions = decisions.Count,
                    WinningDecisions = decisions.Count(d => d.WasCorrect),
                    WinRate = decisions.Count > 0 ? (decimal)decisions.Count(d => d.WasCorrect) / decisions.Count : 0,
                    TotalPnL = decisions.Sum(d => d.RealizedPnL),
                    AverageHoldTime = decisions.Count > 0 ? 
                        TimeSpan.FromTicks((long)decisions.Average(d => d.HoldTime.Ticks)) : TimeSpan.Zero
                };
                
                stats.SourceStatsInternal.Add(sourceStats);
            }
            
            return stats;
        }
    }
    
    #region Conversion Methods
    
    private static Dictionary<string, object> ConvertToEnhancedContext(TradingBot.Abstractions.MarketContext context)
    {
        return new Dictionary<string, object>
        {
            ["symbol"] = context.Symbol,
            ["price"] = context.Price,
            ["volume"] = context.Volume,
            ["timestamp"] = context.Timestamp,
            ["technical_indicators"] = context.TechnicalIndicators
        };
    }
    
    private static List<string> GetAvailableStrategies()
    {
        var hour = DateTime.UtcNow.Hour;
        return hour switch
        {
            >= OPENING_DRIVE_START_HOUR and <= OPENING_DRIVE_END_HOUR => new List<string> { "S6" }, // Opening drive
            >= LUNCH_MEAN_REVERSION_START and <= LUNCH_MEAN_REVERSION_END => new List<string> { "S2" }, // Lunch mean reversion
            >= AFTERNOON_TRADING_START and <= AFTERNOON_TRADING_END => new List<string> { "S11", "S3" }, // Afternoon
            _ => new List<string> { "S2", "S3", "S6", "S11" } // All strategies
        };
    }
    
    private static Env ConvertToEnv(TradingBot.Abstractions.MarketContext context)
    {
        return new Env
        {
            Symbol = context.Symbol,
            atr = (decimal)(context.TechnicalIndicators.GetValueOrDefault("atr", (double)DEFAULT_ATR_VALUE)),
            volz = (decimal)(context.TechnicalIndicators.GetValueOrDefault("volume_z", (double)DEFAULT_VOLUME_Z_VALUE))
        };
    }
    
    private static Levels ConvertToLevels(TradingBot.Abstractions.MarketContext context)
    {
        var price = (decimal)context.Price;
        return new Levels
        {
            Support1 = price - SUPPORT_RESISTANCE_TICK_OFFSET_1,
            Support2 = price - SUPPORT_RESISTANCE_TICK_OFFSET_2,
            Support3 = price - SUPPORT_RESISTANCE_TICK_OFFSET_3,
            Resistance1 = price + SUPPORT_RESISTANCE_TICK_OFFSET_1,
            Resistance2 = price + SUPPORT_RESISTANCE_TICK_OFFSET_2,
            Resistance3 = price + SUPPORT_RESISTANCE_TICK_OFFSET_3,
            VWAP = price,
            DailyPivot = price,
            WeeklyPivot = price + WEEKLY_PIVOT_OFFSET_TICKS, // Small weekly pivot offset
            MonthlyPivot = price - MONTHLY_PIVOT_OFFSET_TICKS  // Small monthly pivot offset
        };
    }
    
    private List<ModelBar> ConvertToBars(TradingBot.Abstractions.MarketContext context)
    {
        var bars = new List<ModelBar>();
        var price = (decimal)context.Price;
        var volume = (decimal)context.Volume;
        
        // Create synthetic bars for the brain
        for (int i = 0; i < SYNTHETIC_BAR_COUNT; i++)
        {
            var variation = (decimal)(System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 10000) / 10000.0 - 0.5) * SYNTHETIC_BAR_VARIATION;
            bars.Add(new ModelBar
            {
                Symbol = context.Symbol,
                Start = context.Timestamp.AddMinutes(-i),
                Ts = ((DateTimeOffset)context.Timestamp.AddMinutes(-i)).ToUnixTimeMilliseconds(),
                Open = price + variation,
                High = price + variation + SYNTHETIC_BAR_RANGE,
                Low = price + variation - SYNTHETIC_BAR_RANGE,
                Close = price + variation,
                Volume = (int)(volume * (MIN_VOLUME_MULTIPLIER + (decimal)(System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 10000) / 10000.0) * VOLUME_MULTIPLIER_RANGE))
            });
        }
        
        return bars;
    }
    
    private static RiskEngine CreateRiskEngine()
    {
        var riskEngine = new RiskEngine();
        riskEngine.cfg.RiskPerTrade = DEFAULT_RISK_PER_TRADE;
        riskEngine.cfg.MaxDailyDrawdown = DEFAULT_MAX_DAILY_DRAWDOWN;
        riskEngine.cfg.MaxOpenPositions = DEFAULT_MAX_OPEN_POSITIONS;
        return riskEngine;
    }
    
    private static TradingBot.Abstractions.MarketContext EnhanceMarketContext(TradingBot.Abstractions.MarketContext context)
    {
        // Enhance the context with additional properties if not already set
        if (Math.Abs(context.Bid) < Epsilon && Math.Abs(context.Ask) < Epsilon)
        {
            var newContext = new TradingBot.Abstractions.MarketContext
            {
                Symbol = context.Symbol,
                Price = context.Price,
                Volume = context.Volume,
                Bid = context.Price - 0.25,
                Ask = context.Price + 0.25,
                Timestamp = context.Timestamp,
                // Copy all other properties
                CurrentRegime = context.CurrentRegime,
                Regime = context.Regime,
                ModelConfidence = context.ModelConfidence,
                PrimaryBias = context.PrimaryBias,
                IsFomcDay = context.IsFomcDay,
                IsCpiDay = context.IsCpiDay,
                NewsIntensity = context.NewsIntensity,
                SignalStrength = context.SignalStrength,
                ConfidenceLevel = context.ConfidenceLevel
            };
            
            // Copy TechnicalIndicators dictionary contents
            foreach (var indicator in context.TechnicalIndicators)
            {
                newContext.TechnicalIndicators[indicator.Key] = indicator.Value;
            }
            
            return newContext;
        }
        return context;
    }
    
    private static UnifiedTradingDecision ConvertFromEnhancedDecision(EnhancedTradingDecision enhanced)
    {
        // Map enhanced decision timing signal to action
        var action = enhanced.MarketTimingSignal switch
        {
            "STRONG_BUY" or "BUY" => TradingAction.Buy,
            "STRONG_SELL" or "SELL" => TradingAction.Sell,
            _ => enhanced.OriginalDecision.Strategy.Contains("BUY", StringComparison.Ordinal) ? TradingAction.Buy : TradingAction.Sell
        };
        
        return new UnifiedTradingDecision
        {
            Symbol = enhanced.OriginalDecision.Symbol,
            Action = action,
            Confidence = enhanced.EnhancedConfidence,
            Quantity = enhanced.EnhancedPositionSize,
            Strategy = enhanced.EnhancedStrategy,
            Reasoning = new Dictionary<string, object>
            {
                ["enhancement_applied"] = enhanced.EnhancementApplied,
                ["enhancement_reason"] = enhanced.EnhancementReason,
                ["market_timing_signal"] = enhanced.MarketTimingSignal,
                ["original_strategy"] = enhanced.OriginalDecision.Strategy
            },
            Timestamp = enhanced.Timestamp
        };
    }
    
    private static UnifiedTradingDecision ConvertFromBrainDecision(BrainDecision brain)
    {
        // Convert brain decision to unified format
        var action = brain.PriceDirection switch
        {
            PriceDirection.Up => TradingAction.Buy,
            PriceDirection.Down => TradingAction.Sell,
            _ => brain.RecommendedStrategy.Contains("11", StringComparison.Ordinal) ? TradingAction.Sell : TradingAction.Buy // S11 is fade strategy
        };
        
        return new UnifiedTradingDecision
        {
            Symbol = brain.Symbol,
            Action = action,
            Confidence = brain.ModelConfidence,
            Quantity = brain.OptimalPositionMultiplier,
            Strategy = brain.RecommendedStrategy,
            Reasoning = new Dictionary<string, object>
            {
                ["recommended_strategy"] = brain.RecommendedStrategy,
                ["strategy_confidence"] = brain.StrategyConfidence,
                ["price_direction"] = brain.PriceDirection.ToString(),
                ["price_probability"] = brain.PriceProbability,
                ["market_regime"] = brain.MarketRegime.ToString(),
                ["risk_assessment"] = brain.RiskAssessment,
                ["processing_time_ms"] = brain.ProcessingTimeMs
            },
            Timestamp = brain.DecisionTime
        };
    }
    
    private static UnifiedTradingDecision ConvertFromAbstractionDecision(TradingBot.Abstractions.TradingDecision decision)
    {
        return new UnifiedTradingDecision
        {
            Symbol = decision.Signal?.Symbol ?? "UNKNOWN",
            Action = decision.Action,
            Confidence = decision.Confidence,
            Quantity = decision.MaxPositionSize > 0 ? decision.MaxPositionSize : 1m,
            Strategy = decision.MLStrategy ?? "INTELLIGENCE",
            Reasoning = decision.Reasoning ?? new Dictionary<string, object>(),
            Timestamp = decision.Timestamp
        };
    }
    
    private static string GenerateDecisionId()
    {
        return $"UD{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}_{System.Security.Cryptography.RandomNumberGenerator.GetInt32(DECISION_ID_RANDOM_MIN, DECISION_ID_RANDOM_MAX)}";
    }
    
    #endregion
}

#region Supporting Models

public class UnifiedTradingDecision
{
    public string DecisionId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public TradingAction Action { get; set; }
    public decimal Confidence { get; set; }
    public decimal Quantity { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public string DecisionSource { get; set; } = string.Empty;
    public Dictionary<string, object> Reasoning { get; init; } = new();
    public DateTime Timestamp { get; set; }
    public double ProcessingTimeMs { get; set; }
}



public class MarketAnalysis
{
    private readonly List<string> _signals = new();
    
    public bool IsUptrend { get; set; }
    public decimal Strength { get; set; }
    public IReadOnlyList<string> Signals => _signals;
    internal List<string> SignalsInternal => _signals;
}

public class DecisionOutcome
{
    public string DecisionId { get; set; } = string.Empty;
    public string Source { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public TradingAction Action { get; set; }
    public decimal Confidence { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    
    // Outcome data
    public bool OutcomeReceived { get; set; }
    public decimal RealizedPnL { get; set; }
    public bool WasCorrect { get; set; }
    public TimeSpan HoldTime { get; set; }
}

public class DecisionRouterStats
{
    private readonly List<SourceStats> _sourceStats = new();
    
    public IReadOnlyList<SourceStats> SourceStats => _sourceStats;
    internal List<SourceStats> SourceStatsInternal => _sourceStats;
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}

public class SourceStats
{
    public string Source { get; set; } = string.Empty;
    public int TotalDecisions { get; set; }
    public int WinningDecisions { get; set; }
    public decimal WinRate { get; set; }
    public decimal TotalPnL { get; set; }
    public TimeSpan AverageHoldTime { get; set; }
}

public class StrategyConfig
{
    public string Name { get; set; } = string.Empty;
    public IReadOnlyList<int> OptimalHours { get; set; } = Array.Empty<int>();
}

#endregion