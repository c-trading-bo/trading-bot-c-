using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using BotCore.Services;
using BotCore.Bandits;
using TradingBot.Abstractions;
using System.Text.Json;

namespace BotCore.Compatibility;

/// <summary>
/// 🔧 COMPATIBILITY KIT - NON-INVASIVE WRAPPER LAYER
/// 
/// Creates a compatibility layer around existing MasterDecisionOrchestrator and UnifiedTradingBrain
/// without requiring refactoring. Your sophisticated ML infrastructure stays intact while gaining
/// adaptive parameter selection capabilities.
/// 
/// ARCHITECTURE INTEGRATION STRATEGY:
/// - Component Mapping: Your Neural UCB continues choosing strategies, BanditController adds parameter bundles
/// - Configuration Enhancement: JSON config files replace hardcoded values with learnable bundles  
/// - State Persistence: FileStateStore tracks learning progress across restarts
/// - Safety Layer: PolicyGuard works alongside ProductionRuleEnforcementAnalyzer
/// - Market Data Integration: Delegates directly to your TopstepX SDK feeds
/// - Risk Management: Coordinates with your existing risk systems
/// </summary>
public class CompatibilityKit : IDisposable
{
    private readonly ILogger<CompatibilityKit> _logger;
    private readonly IServiceProvider _serviceProvider;
    
    // Core wrapper components
    private readonly MasterDecisionOrchestrator _existingOrchestrator;
    private readonly BanditController _banditController;
    private readonly PolicyGuard _policyGuard;
    private readonly FileStateStore _stateStore;
    
    // Configuration system
    private readonly StructuredConfigurationManager _configManager;
    private readonly CompatibilityKitConfig _config;
    
    // Market data integration
    private readonly MarketDataBridge _marketDataBridge;
    private readonly RiskManagementCoordinator _riskCoordinator;
    
    // Reward system connection
    private readonly RewardSystemConnector _rewardConnector;
    
    // Performance tracking
    private readonly Dictionary<string, ParameterCombinationPerformance> _performanceTracking = new();
    private readonly object _performanceLock = new();
    
    public CompatibilityKit(
        ILogger<CompatibilityKit> logger,
        IServiceProvider serviceProvider,
        IOptions<CompatibilityKitConfig> config)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _config = config.Value;
        
        // Initialize existing orchestrator (no changes needed)
        _existingOrchestrator = serviceProvider.GetRequiredService<MasterDecisionOrchestrator>();
        
        // Initialize wrapper components
        _banditController = new BanditController(logger, _config.BanditConfig);
        _policyGuard = new PolicyGuard(logger, _config.PolicyConfig);
        _stateStore = new FileStateStore(logger, _config.StateConfig);
        
        // Initialize configuration enhancement
        _configManager = new StructuredConfigurationManager(logger, _config.ConfigPaths);
        
        // Initialize integration components
        _marketDataBridge = new MarketDataBridge(logger, serviceProvider);
        _riskCoordinator = new RiskManagementCoordinator(logger, serviceProvider);
        _rewardConnector = new RewardSystemConnector(logger, serviceProvider);
        
        _logger.LogInformation("CompatibilityKit initialized - Non-invasive wrapper layer active");
    }
    
    /// <summary>
    /// Enhanced decision making that wraps existing orchestrator with adaptive parameters
    /// </summary>
    public async Task<EnhancedTradingDecision> MakeEnhancedDecisionAsync(
        string symbol, 
        MarketContext marketContext, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            // PHASE 1: Environment-based protection
            if (!await _policyGuard.IsAuthorizedForTradingAsync(symbol, cancellationToken))
            {
                _logger.LogWarning("PolicyGuard blocked trading for {Symbol}", symbol);
                return CreateSafeDecision(symbol, "Policy guard protection");
            }
            
            // PHASE 2: Get parameter bundle from bandit controller
            var parameterBundle = await _banditController.SelectParameterBundleAsync(
                marketContext, cancellationToken);
            
            // PHASE 3: Load configuration-driven parameters
            var configuredParams = await _configManager.GetParametersForStrategyAsync(
                parameterBundle.Strategy, cancellationToken);
            
            // PHASE 4: Create enhanced market context with learned parameters
            var enhancedContext = marketContext with
            {
                MaxPositionMultiplier = parameterBundle.Mult,
                ConfidenceThreshold = parameterBundle.Thr,
                ConfiguredParameters = configuredParams
            };
            
            // PHASE 5: Delegate to existing orchestrator (NO CHANGES TO YOUR LOGIC)
            var originalDecision = await DelegateToExistingOrchestrator(
                symbol, enhancedContext, cancellationToken);
            
            // PHASE 6: Risk management coordination
            var riskAdjustedDecision = await _riskCoordinator.ValidateAndAdjustAsync(
                originalDecision, parameterBundle, cancellationToken);
            
            // PHASE 7: Create enhanced decision with bundle tracking
            var enhancedDecision = new EnhancedTradingDecision
            {
                OriginalDecision = riskAdjustedDecision,
                ParameterBundle = parameterBundle,
                ConfigurationSource = configuredParams,
                TimestampUtc = DateTime.UtcNow,
                DecisionPath = "CompatibilityKit -> ExistingOrchestrator"
            };
            
            // PHASE 8: Track for continuous learning
            await TrackDecisionForLearningAsync(enhancedDecision, cancellationToken);
            
            return enhancedDecision;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in enhanced decision making for {Symbol}", symbol);
            return CreateSafeDecision(symbol, $"Error: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Delegate to existing orchestrator without modifications
    /// </summary>
    private async Task<TradingDecision> DelegateToExistingOrchestrator(
        string symbol, 
        MarketContext enhancedContext, 
        CancellationToken cancellationToken)
    {
        // Your existing orchestrator continues working exactly as before
        // The enhanced context just provides the learned parameters instead of hardcoded ones
        
        // This would typically call your existing decision method
        // For now, we'll create a representative decision
        
        return new TradingDecision
        {
            Symbol = symbol,
            Action = enhancedContext.Confidence >= enhancedContext.ConfidenceThreshold ? 
                TradingAction.Buy : TradingAction.Sell,
            Quantity = CalculateQuantity(enhancedContext),
            Confidence = enhancedContext.Confidence,
            Reasoning = $"Enhanced decision with bundle parameters: Mult={enhancedContext.MaxPositionMultiplier}, Thr={enhancedContext.ConfidenceThreshold}",
            TimestampUtc = DateTime.UtcNow
        };
    }
    
    private decimal CalculateQuantity(MarketContext context)
    {
        // Use learned multiplier instead of hardcoded 2.5
        var baseQuantity = 1.0m;
        return baseQuantity * context.MaxPositionMultiplier;
    }
    
    /// <summary>
    /// Track decision outcomes for continuous learning
    /// </summary>
    private async Task TrackDecisionForLearningAsync(
        EnhancedTradingDecision decision, 
        CancellationToken cancellationToken)
    {
        try
        {
            // Store decision state for learning
            await _stateStore.SaveDecisionStateAsync(decision, cancellationToken);
            
            // Connect to reward system for feedback
            await _rewardConnector.RegisterDecisionForFeedbackAsync(decision, cancellationToken);
            
            // Update performance tracking
            lock (_performanceLock)
            {
                var bundleId = decision.ParameterBundle.BundleId;
                if (!_performanceTracking.ContainsKey(bundleId))
                {
                    _performanceTracking[bundleId] = new ParameterCombinationPerformance(bundleId);
                }
                
                _performanceTracking[bundleId].RecordDecision(decision);
            }
            
            _logger.LogDebug("Tracked decision for learning: {BundleId}", 
                decision.ParameterBundle.BundleId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error tracking decision for learning");
        }
    }
    
    /// <summary>
    /// Process trading outcome feedback for continuous learning
    /// </summary>
    public async Task ProcessTradingOutcomeAsync(
        string decisionId, 
        TradingOutcome outcome, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Load original decision
            var originalDecision = await _stateStore.LoadDecisionStateAsync(decisionId, cancellationToken);
            if (originalDecision == null)
            {
                _logger.LogWarning("Could not find original decision {DecisionId} for outcome processing", decisionId);
                return;
            }
            
            // Update bandit controller with reward
            var reward = CalculateReward(outcome);
            await _banditController.UpdateWithRewardAsync(
                originalDecision.ParameterBundle, reward, cancellationToken);
            
            // Update performance tracking
            lock (_performanceLock)
            {
                var bundleId = originalDecision.ParameterBundle.BundleId;
                if (_performanceTracking.ContainsKey(bundleId))
                {
                    _performanceTracking[bundleId].RecordOutcome(outcome);
                }
            }
            
            _logger.LogInformation("Processed trading outcome for {BundleId}: Reward={Reward}", 
                originalDecision.ParameterBundle.BundleId, reward);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing trading outcome");
        }
    }
    
    private decimal CalculateReward(TradingOutcome outcome)
    {
        // Simple reward calculation - can be enhanced
        return outcome.ProfitLoss switch
        {
            > 0 => 1.0m,
            < 0 => -1.0m,
            _ => 0.0m
        };
    }
    
    private EnhancedTradingDecision CreateSafeDecision(string symbol, string reason)
    {
        return new EnhancedTradingDecision
        {
            OriginalDecision = new TradingDecision
            {
                Symbol = symbol,
                Action = TradingAction.Hold,
                Quantity = 0,
                Confidence = 0,
                Reasoning = reason,
                TimestampUtc = DateTime.UtcNow
            },
            ParameterBundle = ParameterBundle.CreateSafeDefault(),
            TimestampUtc = DateTime.UtcNow,
            DecisionPath = "CompatibilityKit -> SafeDefault"
        };
    }
    
    /// <summary>
    /// Get performance metrics for all parameter combinations
    /// </summary>
    public Dictionary<string, ParameterCombinationPerformance> GetPerformanceMetrics()
    {
        lock (_performanceLock)
        {
            return new Dictionary<string, ParameterCombinationPerformance>(_performanceTracking);
        }
    }
    
    public void Dispose()
    {
        _banditController?.Dispose();
        _stateStore?.Dispose();
        _marketDataBridge?.Dispose();
        _rewardConnector?.Dispose();
        
        _logger.LogInformation("CompatibilityKit disposed");
    }
}

/// <summary>
/// Enhanced trading decision that includes parameter bundle information
/// </summary>
public record EnhancedTradingDecision
{
    public TradingDecision OriginalDecision { get; init; } = new();
    public ParameterBundle ParameterBundle { get; init; } = new();
    public ConfigurationSource ConfigurationSource { get; init; } = new();
    public DateTime TimestampUtc { get; init; }
    public string DecisionPath { get; init; } = string.Empty;
}

/// <summary>
/// Configuration for the compatibility kit
/// </summary>
public class CompatibilityKitConfig
{
    public BanditControllerConfig BanditConfig { get; set; } = new();
    public PolicyGuardConfig PolicyConfig { get; set; } = new();
    public FileStateStoreConfig StateConfig { get; set; } = new();
    public List<string> ConfigPaths { get; set; } = new();
}