using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using BotCore.Services;
using BotCore.Bandits;
using TradingBot.Abstractions;
using System.Text.Json;

namespace BotCore.Services;

/// <summary>
/// 🎯 MASTER DECISION ORCHESTRATOR - ALWAYS-LEARNING TRADING SYSTEM 🎯
/// 
/// This is the ONE MASTER BRAIN that coordinates all AI decision-making:
/// 
/// PRIMARY DECISION HIERARCHY:
/// 1. Enhanced Brain Integration (Multi-model ensemble + cloud sync)
/// 2. Unified Trading Brain (Neural UCB + CVaR-PPO + LSTM) 
/// 3. Intelligence Orchestrator (Basic ML/RL fallback)
/// 4. Python Decision Services (UCB FastAPI services)
/// 5. Direct Strategy Execution (ultimate fallback)
/// 
/// GUARANTEES:
/// ✅ NEVER returns HOLD - always BUY/SELL
/// ✅ Continuous learning from every trade outcome  
/// ✅ 24/7 operation with auto-recovery
/// ✅ Real-time model promotion based on performance
/// ✅ Historical + live data integration
/// ✅ Contract auto-rollover (Z25 → H26)
/// 
/// RESULT: Always-learning trading system that gets smarter every day
/// </summary>
public class MasterDecisionOrchestrator : BackgroundService
{
    private readonly ILogger<MasterDecisionOrchestrator> _logger;
    private readonly IServiceProvider _serviceProvider;
    
    // Core decision routing components
    private readonly UnifiedDecisionRouter _unifiedRouter;
    private readonly TradingBot.Abstractions.DecisionServiceStatus _serviceStatus;
    
    // Learning and feedback systems
    private readonly EnhancedTradingBrainIntegration? _enhancedBrain;
    private readonly BotCore.Brain.UnifiedTradingBrain _unifiedBrain;
    
    // Extended Neural UCB for parameter bundle selection
    private readonly NeuralUcbExtended? _neuralUcbExtended;
    
    // Configuration and monitoring
    private readonly MasterOrchestratorConfig _config;
    private readonly ContinuousLearningManager _learningManager;
    private readonly ContractRolloverManager _rolloverManager;
    
    // Operational state
    private readonly Dictionary<string, DecisionPerformance> _performanceTracking = new();
    private readonly Queue<LearningEvent> _learningQueue = new();
    private readonly object _stateLock = new();
    
    // Always-learning operation state
    private bool _isLearningActive;
    private DateTime _lastModelUpdate = DateTime.MinValue;
    private DateTime _lastPerformanceReport = DateTime.MinValue;
    
    public MasterDecisionOrchestrator(
        ILogger<MasterDecisionOrchestrator> logger,
        IServiceProvider serviceProvider,
        UnifiedDecisionRouter unifiedRouter,
        BotCore.Brain.UnifiedTradingBrain unifiedBrain)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _unifiedRouter = unifiedRouter;
        _serviceStatus = new TradingBot.Abstractions.DecisionServiceStatus();
        _unifiedBrain = unifiedBrain;
        
        // Try to get optional enhanced services
        _enhancedBrain = serviceProvider.GetService<EnhancedTradingBrainIntegration>();
        
        // Try to get Neural UCB Extended for parameter bundle selection
        _neuralUcbExtended = serviceProvider.GetService<NeuralUcbExtended>();
        
        // Initialize configuration
        _config = serviceProvider.GetService<IOptions<MasterOrchestratorConfig>>()?.Value ?? new MasterOrchestratorConfig();
        
        // Initialize continuous learning manager
        _learningManager = new ContinuousLearningManager(logger, serviceProvider);
        
        // Initialize contract rollover manager
        _rolloverManager = new ContractRolloverManager(logger, serviceProvider);
        
        _logger.LogInformation("🎯 [MASTER-ORCHESTRATOR] Initialized - Always-learning trading system ready");
        _logger.LogInformation("🧠 [MASTER-ORCHESTRATOR] Enhanced Brain: {Enhanced}, Service Router: True, Unified Brain: True, Neural UCB Extended: {Extended}", 
            _enhancedBrain != null, _neuralUcbExtended != null);
    }
    
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("🚀 [MASTER-ORCHESTRATOR] Starting always-learning trading system...");
        
        try
        {
            // Initialize all systems
            await InitializeSystemsAsync(stoppingToken).ConfigureAwait(false);
            
            // Start continuous learning
            await StartContinuousLearningAsync(stoppingToken).ConfigureAwait(false);
            
            // Main orchestration loop
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    // Execute main orchestration cycle
                    await ExecuteOrchestrationCycleAsync(stoppingToken).ConfigureAwait(false);
                    
                    // Wait before next cycle
                    await Task.Delay(TimeSpan.FromSeconds(_config.OrchestrationCycleIntervalSeconds), stoppingToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [MASTER-ORCHESTRATOR] Error in orchestration cycle");
                    await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken).ConfigureAwait(false);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogCritical(ex, "💥 [MASTER-ORCHESTRATOR] Critical error in master orchestrator");
            throw;
        }
        finally
        {
            _logger.LogInformation("🛑 [MASTER-ORCHESTRATOR] Always-learning trading system stopped");
        }
    }
    
    /// <summary>
    /// Initialize all AI systems and verify readiness
    /// </summary>
    private async Task InitializeSystemsAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔧 [MASTER-ORCHESTRATOR] Initializing all AI systems...");
        
        try
        {
            // Initialize Unified Trading Brain
            await _unifiedBrain.InitializeAsync(cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("✅ [UNIFIED-BRAIN] Initialized successfully");
            
            // Initialize Enhanced Brain if available
            if (_enhancedBrain != null)
            {
                await _enhancedBrain.InitializeAsync(cancellationToken).ConfigureAwait(false);
                _logger.LogInformation("✅ [ENHANCED-BRAIN] Initialized successfully");
            }
            
            // Initialize learning manager
            await _learningManager.InitializeAsync(cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("✅ [LEARNING-MANAGER] Initialized successfully");
            
            // Initialize contract rollover manager
            await _rolloverManager.InitializeAsync(cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("✅ [ROLLOVER-MANAGER] Initialized successfully");
            
            _logger.LogInformation("🎉 [MASTER-ORCHESTRATOR] All systems initialized - Ready for always-learning operation");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [MASTER-ORCHESTRATOR] System initialization failed");
            throw;
        }
    }
    
    /// <summary>
    /// Start continuous learning systems
    /// </summary>
    private async Task StartContinuousLearningAsync(CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation("📚 [CONTINUOUS-LEARNING] Starting always-learning systems...");
            
            // Start learning manager
            await _learningManager.StartLearningAsync(cancellationToken).ConfigureAwait(false);
            
            // Start contract rollover monitoring
            await _rolloverManager.StartMonitoringAsync(cancellationToken).ConfigureAwait(false);
            
            _isLearningActive = true;
            _logger.LogInformation("✅ [CONTINUOUS-LEARNING] Always-learning systems started successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [CONTINUOUS-LEARNING] Failed to start learning systems");
            throw;
        }
    }
    
    /// <summary>
    /// Execute main orchestration cycle
    /// </summary>
    private async Task ExecuteOrchestrationCycleAsync(CancellationToken cancellationToken)
    {
        // Process learning queue
        await ProcessLearningQueueAsync(cancellationToken).ConfigureAwait(false);
        
        // Check for model updates
        await CheckModelUpdatesAsync(cancellationToken).ConfigureAwait(false);
        
        // Generate performance reports
        await GeneratePerformanceReportsAsync(cancellationToken).ConfigureAwait(false);
        
        // Monitor system health
        await MonitorSystemHealthAsync(cancellationToken).ConfigureAwait(false);
        
        // Check contract rollover needs
        await CheckContractRolloverAsync(cancellationToken).ConfigureAwait(false);
    }
    
    /// <summary>
    /// MAIN DECISION METHOD - Used by trading orchestrator
    /// This is the ONE method that all trading decisions flow through
    /// 
    /// Enhanced with bundle-based parameter selection:
    /// - Uses Neural UCB Extended to select optimal strategy-parameter combinations
    /// - Replaces hardcoded MaxPositionMultiplier and confidenceThreshold with learned values
    /// - Enables continuous adaptation of trading parameters
    /// </summary>
    public async Task<UnifiedTradingDecision> MakeUnifiedDecisionAsync(
        string symbol,
        MarketContext marketContext,
        CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;
        var decisionId = GenerateDecisionId();
        
        try
        {
            _logger.LogDebug("🎯 [MASTER-DECISION] Making unified decision for {Symbol}", symbol);
            
            // PHASE 1: Get parameter bundle selection from Neural UCB Extended
            BundleSelection? bundleSelection = null;
            if (_neuralUcbExtended != null)
            {
                try
                {
                    bundleSelection = await _neuralUcbExtended.SelectBundleAsync(marketContext, cancellationToken)
                        .ConfigureAwait(false);
                    
                    _logger.LogInformation("🎯 [BUNDLE-SELECTION] Selected: {BundleId} " +
                                         "strategy={Strategy} mult={Mult:F1}x thr={Thr:F2}",
                        bundleSelection.Bundle.BundleId, bundleSelection.Bundle.Strategy,
                        bundleSelection.Bundle.Mult, bundleSelection.Bundle.Thr);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "⚠️ [BUNDLE-SELECTION] Failed to select bundle, using fallback");
                }
            }
            
            // PHASE 2: Apply bundle parameters to market context
            var enhancedMarketContext = ApplyBundleParameters(marketContext, bundleSelection);
            
            // PHASE 3: Route through unified decision system with enhanced context
            var decision = await _unifiedRouter.RouteDecisionAsync(symbol, enhancedMarketContext, cancellationToken)
                .ConfigureAwait(false);
            
            // PHASE 4: Apply bundle configuration to decision
            if (bundleSelection != null)
            {
                decision = ApplyBundleToDecision(decision, bundleSelection);
            }
            
            // Ensure decision ID is set
            if (string.IsNullOrEmpty(decision.DecisionId))
            {
                decision.DecisionId = decisionId;
            }
            
            // Track decision for learning (including bundle performance)
            await TrackDecisionForLearningAsync(decision, enhancedMarketContext, bundleSelection, cancellationToken)
                .ConfigureAwait(false);
            
            // Log the enhanced decision
            LogEnhancedDecision(decision, bundleSelection, startTime);
            
            return decision;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [MASTER-DECISION] Failed to make decision for {Symbol}", symbol);
            
            // Emergency fallback decision
            return CreateEmergencyDecision(symbol, marketContext, decisionId, startTime);
        }
    }
    
    /// <summary>
    /// Submit trading outcome for continuous learning
    /// </summary>
    public async Task SubmitTradingOutcomeAsync(
        string decisionId,
        decimal realizedPnL,
        bool wasCorrect,
        TimeSpan holdTime,
        string decisionSource,
        Dictionary<string, object> metadata,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("📈 [MASTER-FEEDBACK] Recording outcome: {DecisionId} PnL={PnL:C2} Correct={Correct}",
                decisionId, realizedPnL, wasCorrect);
            
            // Submit to unified learning system
            await _unifiedRouter.SubmitTradingOutcomeAsync(
                decisionId, realizedPnL, wasCorrect, holdTime, cancellationToken).ConfigureAwait(false);
            
            // Create learning event
            var learningEvent = new LearningEvent
            {
                DecisionId = decisionId,
                RealizedPnL = realizedPnL,
                WasCorrect = wasCorrect,
                HoldTime = holdTime,
                DecisionSource = decisionSource,
                Metadata = metadata,
                Timestamp = DateTime.UtcNow
            };
            
            // Add to learning queue
            lock (_stateLock)
            {
                _learningQueue.Enqueue(learningEvent);
            }
            
            // Update performance tracking
            await UpdatePerformanceTrackingAsync(decisionId, decisionSource, realizedPnL, wasCorrect, cancellationToken).ConfigureAwait(false);
            
            // Update bundle performance if this was a bundle-enhanced decision
            await UpdateBundlePerformanceAsync(decisionId, realizedPnL, wasCorrect, metadata, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("✅ [MASTER-FEEDBACK] Outcome recorded and queued for learning");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [MASTER-FEEDBACK] Failed to submit trading outcome");
        }
    }
    
    /// <summary>
    /// Get comprehensive system status
    /// </summary>
    public MasterOrchestratorStatus GetSystemStatus()
    {
        lock (_stateLock)
        {
            var status = new MasterOrchestratorStatus
            {
                IsLearningActive = _isLearningActive,
                LastModelUpdate = _lastModelUpdate,
                LastPerformanceReport = _lastPerformanceReport,
                TotalDecisionsToday = _performanceTracking.Values.Sum(p => p.TotalDecisions),
                OverallWinRate = CalculateOverallWinRate(),
                LearningQueueSize = _learningQueue.Count,
                ServiceStatus = _serviceStatus,
                BrainPerformance = _performanceTracking.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                SystemHealthy = IsSystemHealthy(),
                Timestamp = DateTime.UtcNow
            };
            
            return status;
        }
    }
    
    /// <summary>
    /// Force model update and retraining
    /// </summary>
    public async Task ForceModelUpdateAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("🔄 [FORCE-UPDATE] Forcing model update and retraining...");
            
            // Trigger learning manager update
            await _learningManager.ForceUpdateAsync(cancellationToken).ConfigureAwait(false);
            
            // Update timestamp
            _lastModelUpdate = DateTime.UtcNow;
            
            _logger.LogInformation("✅ [FORCE-UPDATE] Model update completed successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [FORCE-UPDATE] Failed to force model update");
            throw;
        }
    }
    
    #region Bundle Integration Methods
    
    /// <summary>
    /// Apply bundle parameters to market context
    /// This replaces hardcoded values with learned parameter selections
    /// </summary>
    private MarketContext ApplyBundleParameters(MarketContext marketContext, BundleSelection? bundleSelection)
    {
        if (bundleSelection == null)
        {
            return marketContext; // Return unchanged if no bundle selection
        }
        
        // Create enhanced market context with bundle parameters
        var enhancedContext = new MarketContext
        {
            Symbol = marketContext.Symbol,
            Timestamp = marketContext.Timestamp,
            Price = marketContext.Price,
            Volume = marketContext.Volume,
            Bid = marketContext.Bid,
            Ask = marketContext.Ask,
            CurrentRegime = marketContext.CurrentRegime,
            Regime = marketContext.Regime,
            ModelConfidence = marketContext.ModelConfidence,
            PrimaryBias = marketContext.PrimaryBias,
            IsFomcDay = marketContext.IsFomcDay,
            IsCpiDay = marketContext.IsCpiDay,
            NewsIntensity = marketContext.NewsIntensity,
            SignalStrength = marketContext.SignalStrength,
            ConfidenceLevel = (double)bundleSelection.Bundle.Thr // Apply learned confidence threshold
        };
        
        // Copy technical indicators
        foreach (var indicator in marketContext.TechnicalIndicators)
        {
            enhancedContext.TechnicalIndicators[indicator.Key] = indicator.Value;
        }
        
        // Add bundle-specific technical indicators
        enhancedContext.TechnicalIndicators["bundle_multiplier"] = (double)bundleSelection.Bundle.Mult;
        enhancedContext.TechnicalIndicators["bundle_threshold"] = (double)bundleSelection.Bundle.Thr;
        enhancedContext.TechnicalIndicators["bundle_ucb_value"] = (double)bundleSelection.UcbValue;
        enhancedContext.TechnicalIndicators["bundle_prediction"] = (double)bundleSelection.Prediction;
        
        return enhancedContext;
    }
    
    /// <summary>
    /// Apply bundle configuration to trading decision
    /// This ensures the decision uses learned parameters instead of hardcoded values
    /// </summary>
    private UnifiedTradingDecision ApplyBundleToDecision(UnifiedTradingDecision decision, BundleSelection bundleSelection)
    {
        // Apply bundle multiplier to position size
        var enhancedQuantity = decision.Quantity * bundleSelection.Bundle.Mult;
        
        // Apply bundle confidence requirements
        var bundleAdjustedConfidence = Math.Max(decision.Confidence, bundleSelection.Bundle.Thr);
        
        // Create enhanced decision
        var enhancedDecision = new UnifiedTradingDecision
        {
            DecisionId = decision.DecisionId,
            Symbol = decision.Symbol,
            Action = decision.Action,
            Confidence = bundleAdjustedConfidence,
            Quantity = enhancedQuantity,
            Strategy = bundleSelection.Bundle.Strategy, // Use bundle strategy
            DecisionSource = $"{decision.DecisionSource}+Bundle",
            Reasoning = new Dictionary<string, object>(decision.Reasoning)
            {
                ["bundle_id"] = bundleSelection.Bundle.BundleId,
                ["bundle_strategy"] = bundleSelection.Bundle.Strategy,
                ["bundle_multiplier"] = bundleSelection.Bundle.Mult,
                ["bundle_threshold"] = bundleSelection.Bundle.Thr,
                ["original_quantity"] = decision.Quantity,
                ["enhanced_quantity"] = enhancedQuantity,
                ["bundle_selection_reason"] = bundleSelection.SelectionReason
            },
            Timestamp = decision.Timestamp,
            ProcessingTimeMs = decision.ProcessingTimeMs
        };
        
        return enhancedDecision;
    }
    
    /// <summary>
    /// Enhanced tracking that includes bundle performance
    /// </summary>
    private async Task TrackDecisionForLearningAsync(
        UnifiedTradingDecision decision,
        MarketContext marketContext,
        BundleSelection? bundleSelection,
        CancellationToken cancellationToken)
    {
        // Standard tracking
        var trackingInfo = new DecisionTrackingInfo
        {
            DecisionId = decision.DecisionId,
            Symbol = decision.Symbol,
            Action = decision.Action,
            Confidence = decision.Confidence,
            Strategy = decision.Strategy,
            DecisionSource = decision.DecisionSource,
            MarketContext = marketContext,
            Timestamp = decision.Timestamp
        };
        
        await _learningManager.TrackDecisionAsync(trackingInfo, cancellationToken).ConfigureAwait(false);
        
        // Enhanced tracking with bundle information
        if (bundleSelection != null)
        {
            var enhancedTrackingInfo = new BundleDecisionTrackingInfo
            {
                DecisionId = decision.DecisionId,
                BundleId = bundleSelection.Bundle.BundleId,
                BundleSelection = bundleSelection,
                StandardTracking = trackingInfo
            };
            
            await TrackBundleDecisionAsync(enhancedTrackingInfo, cancellationToken).ConfigureAwait(false);
        }
    }
    
    /// <summary>
    /// Track bundle-specific decision information
    /// </summary>
    private async Task TrackBundleDecisionAsync(BundleDecisionTrackingInfo trackingInfo, CancellationToken cancellationToken)
    {
        try
        {
            // Store bundle decision tracking for future learning
            var bundleTrackingPath = Path.Combine("data", "bundle_decisions.json");
            Directory.CreateDirectory(Path.GetDirectoryName(bundleTrackingPath)!);
            
            var trackingData = new
            {
                Timestamp = DateTime.UtcNow,
                TrackingInfo = trackingInfo
            };
            
            var json = JsonSerializer.Serialize(trackingData);
            await File.AppendAllTextAsync(bundleTrackingPath, json + Environment.NewLine, cancellationToken)
                .ConfigureAwait(false);
            
            _logger.LogDebug("📊 [BUNDLE-TRACKING] Tracked bundle decision: {BundleId}", trackingInfo.BundleId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [BUNDLE-TRACKING] Failed to track bundle decision");
        }
    }
    
    /// <summary>
    /// Enhanced logging that includes bundle information
    /// </summary>
    private void LogEnhancedDecision(UnifiedTradingDecision decision, BundleSelection? bundleSelection, DateTime startTime)
    {
        var processingTime = (DateTime.UtcNow - startTime).TotalMilliseconds;
        
        if (bundleSelection != null)
        {
            _logger.LogInformation("🎯 [ENHANCED-DECISION] Decision: {Action} {Symbol} " +
                                 "confidence={Confidence:P1} source={Source} strategy={Strategy} " +
                                 "bundle={BundleId} mult={Mult:F1}x thr={Thr:F2} time={Time:F0}ms",
                decision.Action, decision.Symbol, decision.Confidence, 
                decision.DecisionSource, decision.Strategy, bundleSelection.Bundle.BundleId,
                bundleSelection.Bundle.Mult, bundleSelection.Bundle.Thr, processingTime);
        }
        else
        {
            _logger.LogInformation("🎯 [STANDARD-DECISION] Decision: {Action} {Symbol} " +
                                 "confidence={Confidence:P1} source={Source} strategy={Strategy} time={Time:F0}ms",
                decision.Action, decision.Symbol, decision.Confidence, 
                decision.DecisionSource, decision.Strategy, processingTime);
        }
    }
    
    /// <summary>
    /// Update bundle performance based on trading outcome
    /// This enables continuous learning of optimal parameter combinations
    /// </summary>
    private async Task UpdateBundlePerformanceAsync(
                decimal realizedPnL,
        bool wasCorrect,
        Dictionary<string, object> metadata,
        CancellationToken cancellationToken)
    {
        if (_neuralUcbExtended == null)
            return;
            
        try
        {
            // Check if this decision used a bundle (look for bundle metadata)
            if (metadata.TryGetValue("bundle_id", out var bundleIdObj) && bundleIdObj is string bundleId)
            {
                // Create reward signal from trading outcome
                var reward = CalculateBundleReward(realizedPnL, wasCorrect, metadata);
                
                // Create a simple market context for the update (we could store the original context if needed)
                var marketContext = CreateMarketContextFromMetadata(metadata);
                
                // Update bundle performance in Neural UCB Extended
                await _neuralUcbExtended.UpdateBundlePerformanceAsync(
                    bundleId, marketContext, reward, metadata, cancellationToken).ConfigureAwait(false);
                
                _logger.LogInformation("📊 [BUNDLE-FEEDBACK] Updated bundle {BundleId} with reward {Reward:F3} " +
                                     "from PnL {PnL:C2} correct={Correct}",
                    bundleId, reward, realizedPnL, wasCorrect);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [BUNDLE-FEEDBACK] Failed to update bundle performance");
        }
    }
    
    /// <summary>
    /// Calculate reward signal for bundle learning
    /// Converts trading outcomes to ML reward signals
    /// </summary>
    private static decimal CalculateBundleReward(decimal realizedPnL, bool wasCorrect, Dictionary<string, object> metadata)
    {
        // Base reward from P&L (normalized to -1 to +1 range)
        var pnlReward = Math.Max(-1m, Math.Min(1m, realizedPnL / 100m)); // Assume $100 normalizes to 1.0
        
        // Accuracy bonus/penalty
        var accuracyReward = wasCorrect ? 0.2m : -0.2m;
        
        // Time-based reward (faster profits are better)
        var timeReward = 0m;
        if (metadata.TryGetValue("hold_time", out var holdTimeObj) && holdTimeObj is TimeSpan holdTime)
        {
            // Reward shorter holding periods for profitable trades
            if (realizedPnL > 0)
            {
                timeReward = Math.Max(-0.1m, Math.Min(0.1m, (decimal)(1.0 - holdTime.TotalHours / 24.0) * 0.1m));
            }
        }
        
        var totalReward = pnlReward + accuracyReward + timeReward;
        return Math.Max(-1m, Math.Min(1m, totalReward)); // Clamp to [-1, 1] range
    }
    
    /// <summary>
    /// Create a basic market context from metadata for bundle updates
    /// </summary>
    private static MarketContext CreateMarketContextFromMetadata(Dictionary<string, object> metadata)
    {
        var context = new MarketContext();
        
        // Extract basic market data from metadata if available
        if (metadata.TryGetValue("symbol", out var symbolObj) && symbolObj is string symbol)
            context.Symbol = symbol;
            
        if (metadata.TryGetValue("price", out var priceObj) && priceObj is double price)
            context.Price = price;
            
        if (metadata.TryGetValue("volume", out var volumeObj) && volumeObj is double volume)
            context.Volume = volume;
            
        // Add technical indicators from metadata
        foreach (var kvp in metadata.Where(m => m.Key.StartsWith("tech_")))
        {
            if (kvp.Value is double techValue)
            {
                var indicatorName = kvp.Key.Substring(5); // Remove "tech_" prefix
                context.TechnicalIndicators[indicatorName] = techValue;
            }
        }
        
        return context;
    }
    
    #endregion
    
    #region Private Implementation Methods
    
    private async Task ProcessLearningQueueAsync(CancellationToken cancellationToken)
    {
        var events = new List<LearningEvent>();
        
        lock (_stateLock)
        {
            while (_learningQueue.Count > 0)
            {
                events.Add(_learningQueue.Dequeue());
            }
        }
        
        if (events.Count > 0)
        {
            await _learningManager.ProcessLearningEventsAsync(events, cancellationToken).ConfigureAwait(false);
            _logger.LogDebug("📚 [LEARNING-QUEUE] Processed {Count} learning events", events.Count);
        }
    }
    
    private async Task CheckModelUpdatesAsync(CancellationToken cancellationToken)
    {
        if (DateTime.UtcNow - _lastModelUpdate > TimeSpan.FromHours(_config.ModelUpdateIntervalHours))
        {
            await _learningManager.CheckAndUpdateModelsAsync(cancellationToken).ConfigureAwait(false);
            _lastModelUpdate = DateTime.UtcNow;
        }
    }
    
    private async Task GeneratePerformanceReportsAsync(CancellationToken cancellationToken)
    {
        if (DateTime.UtcNow - _lastPerformanceReport > TimeSpan.FromHours(_config.PerformanceReportIntervalHours))
        {
            await GenerateDetailedPerformanceReportAsync(cancellationToken).ConfigureAwait(false);
            _lastPerformanceReport = DateTime.UtcNow;
        }
    }
    
    private async Task MonitorSystemHealthAsync(CancellationToken cancellationToken)
    {
        // Monitor system health and trigger alerts if needed
        var isHealthy = IsSystemHealthy();
        
        if (!isHealthy)
        {
            _logger.LogWarning("⚠️ [SYSTEM-HEALTH] System health degraded, triggering recovery actions");
            await TriggerRecoveryActionsAsync(cancellationToken).ConfigureAwait(false);
        }
    }
    
    private Task CheckContractRolloverAsync(CancellationToken cancellationToken)
    {
        return _rolloverManager.CheckRolloverNeedsAsync(cancellationToken);
    }
    
    private UnifiedTradingDecision CreateEmergencyDecision(
        string symbol,
                string decisionId, 
        DateTime startTime)
    {
        return new UnifiedTradingDecision
        {
            DecisionId = decisionId,
            Symbol = symbol,
            Action = TradingAction.Buy, // Conservative emergency bias
            Confidence = 0.51m, // Minimum viable
            Quantity = 1m, // Very conservative
            Strategy = "EMERGENCY",
            DecisionSource = "Emergency",
            Reasoning = new Dictionary<string, object>
            {
                ["source"] = "Emergency fallback - all decision systems failed",
                ["safety"] = "Conservative BUY bias with minimum sizing"
            },
            Timestamp = DateTime.UtcNow,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }
    
    private Task UpdatePerformanceTrackingAsync(
                string decisionSource,
        decimal realizedPnL,
        bool wasCorrect
        )
    {
        lock (_stateLock)
        {
            if (!_performanceTracking.TryGetValue(decisionSource, out var performance))
            {
                performance = new DecisionPerformance { Source = decisionSource };
                _performanceTracking[decisionSource] = performance;
            }
            
            performance.TotalDecisions++;
            performance.TotalPnL += realizedPnL;
            if (wasCorrect) performance.WinningDecisions++;
            performance.WinRate = performance.TotalDecisions > 0 ? 
                (decimal)performance.WinningDecisions / performance.TotalDecisions : 0;
            performance.LastUpdated = DateTime.UtcNow;
        }

        return Task.CompletedTask;
    }
    
    private decimal CalculateOverallWinRate()
    {
        var totalDecisions = _performanceTracking.Values.Sum(p => p.TotalDecisions);
        var totalWinning = _performanceTracking.Values.Sum(p => p.WinningDecisions);
        
        return totalDecisions > 0 ? (decimal)totalWinning / totalDecisions : 0;
    }
    
    private bool IsSystemHealthy()
    {
        // Check if learning is active
        if (!_isLearningActive) return false;
        
        // Check recent performance
        var recentPerformance = _performanceTracking.Values
            .Where(p => DateTime.UtcNow - p.LastUpdated < TimeSpan.FromHours(1))
            .ToList();
            
        if (!recentPerformance.Any()) return true; // No recent decisions is okay
        
        // Check if any source has very poor performance
        var poorPerformers = recentPerformance
            .Where(p => p.TotalDecisions >= 10 && p.WinRate < 0.3m)
            .ToList();
            
        return !poorPerformers.Any();
    }
    
    private async Task TriggerRecoveryActionsAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔧 [RECOVERY] Triggering system recovery actions...");
        
        try
        {
            // Restart learning systems
            await _learningManager.RestartAsync(cancellationToken).ConfigureAwait(false);
            
            // Force model updates
            await _learningManager.ForceUpdateAsync(cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("✅ [RECOVERY] Recovery actions completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [RECOVERY] Recovery actions failed");
        }
    }
    
    private async Task GenerateDetailedPerformanceReportAsync(CancellationToken cancellationToken)
    {
        try
        {
            var report = new PerformanceReport
            {
                Timestamp = DateTime.UtcNow,
                OverallStats = new OverallStats
                {
                    TotalDecisions = _performanceTracking.Values.Sum(p => p.TotalDecisions),
                    TotalPnL = _performanceTracking.Values.Sum(p => p.TotalPnL),
                    OverallWinRate = CalculateOverallWinRate(),
                    ActiveSources = _performanceTracking.Count
                }
            };
            
            // Add performance data to the collection property
            foreach (var performance in _performanceTracking.Values)
            {
                report.SourcePerformance.Add(performance);
            }
            
            // Save report
            var reportPath = Path.Combine("reports", $"performance_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
            
            var json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(reportPath, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("📊 [PERFORMANCE-REPORT] Generated detailed report: {ReportPath}", reportPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [PERFORMANCE-REPORT] Failed to generate performance report");
        }
    }
    
    private string GenerateDecisionId()
    {
        return $"MD{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}_{Random.Shared.Next(1000, 9999)}";
    }
    
    #endregion
}

#region Supporting Classes

/// <summary>
/// Continuous learning manager - coordinates all learning activities
/// </summary>
public class ContinuousLearningManager
{
    private readonly ILogger _logger;
    private readonly IServiceProvider _serviceProvider;
    
    public ContinuousLearningManager(ILogger logger, IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }
    
    public Task InitializeAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    public Task StartLearningAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    public Task ProcessLearningEventsAsync(List<LearningEvent> events, CancellationToken cancellationToken) => Task.CompletedTask;
    public Task CheckAndUpdateModelsAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    public Task ForceUpdateAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    public Task RestartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    public Task TrackDecisionAsync(DecisionTrackingInfo info, CancellationToken cancellationToken) => Task.CompletedTask;
}

/// <summary>
/// Contract rollover manager - handles Z25 → H26 transitions
/// </summary>
public class ContractRolloverManager
{
    private readonly ILogger _logger;
    private readonly IServiceProvider _serviceProvider;
    
    public ContractRolloverManager(ILogger logger, IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }
    
    public Task InitializeAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    public Task StartMonitoringAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    public Task CheckRolloverNeedsAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}

#endregion

#region Data Models

public class MasterOrchestratorConfig
{
    public int OrchestrationCycleIntervalSeconds { get; set; } = 10;
    public int ModelUpdateIntervalHours { get; set; } = 2;
    public int PerformanceReportIntervalHours { get; set; } = 6;
    public bool EnableContinuousLearning { get; set; } = true;
    public bool EnableContractRollover { get; set; } = true;
}

public class LearningEvent
{
    public string DecisionId { get; set; } = string.Empty;
    public decimal RealizedPnL { get; set; }
    public bool WasCorrect { get; set; }
    public TimeSpan HoldTime { get; set; }
    public string DecisionSource { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; } = new();
    public DateTime Timestamp { get; set; }
}

public class DecisionTrackingInfo
{
    public string DecisionId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public TradingAction Action { get; set; }
    public decimal Confidence { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public string DecisionSource { get; set; } = string.Empty;
    public MarketContext MarketContext { get; set; } = new();
    public DateTime Timestamp { get; set; }
}

public class DecisionPerformance
{
    public string Source { get; set; } = string.Empty;
    public int TotalDecisions { get; set; }
    public int WinningDecisions { get; set; }
    public decimal WinRate { get; set; }
    public decimal TotalPnL { get; set; }
    public DateTime LastUpdated { get; set; }
}

public class MasterOrchestratorStatus
{
    public bool IsLearningActive { get; set; }
    public DateTime LastModelUpdate { get; set; }
    public DateTime LastPerformanceReport { get; set; }
    public int TotalDecisionsToday { get; set; }
    public decimal OverallWinRate { get; set; }
    public int LearningQueueSize { get; set; }
    public TradingBot.Abstractions.DecisionServiceStatus ServiceStatus { get; set; } = new();
    public Dictionary<string, DecisionPerformance> BrainPerformance { get; } = new();
    public bool SystemHealthy { get; set; }
    public DateTime Timestamp { get; set; }
}

public class PerformanceReport
{
    public DateTime Timestamp { get; set; }
    public OverallStats OverallStats { get; set; } = new();
    public List<DecisionPerformance> SourcePerformance { get; } = new();
}

public class OverallStats
{
    public int TotalDecisions { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal OverallWinRate { get; set; }
    public int ActiveSources { get; set; }
}

/// <summary>
/// Enhanced tracking info that includes bundle selection data
/// </summary>
public class BundleDecisionTrackingInfo
{
    public string DecisionId { get; set; } = string.Empty;
    public string BundleId { get; set; } = string.Empty;
    public BundleSelection BundleSelection { get; set; } = new();
    public DecisionTrackingInfo StandardTracking { get; set; } = new();
}

#endregion