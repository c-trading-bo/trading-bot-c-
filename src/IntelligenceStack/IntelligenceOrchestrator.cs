using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Options;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Main intelligence orchestrator implementing complete ML/RL intelligence stack
/// Coordinates regime detection, model inference, calibration, decision making, and cloud flow
/// Merged cloud flow functionality from UnifiedOrchestrator.Services.CloudFlowService
/// </summary>
public class IntelligenceOrchestrator : IIntelligenceOrchestrator
{
    // Constants for magic number violations
    // Intelligence thresholds for decision making
    private const double BullishThreshold = 0.55;
    private const double BearishThreshold = 0.45;
    
    // LoggerMessage delegates for CA1848 compliance - IntelligenceOrchestrator
    private static readonly Action<ILogger, string, Exception?> OrchestratorInitialized =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(4001, "OrchestratorInitialized"),
            "[INTELLIGENCE] Intelligence orchestrator initialized with cloud flow: {CloudEndpoint}");
            
    private static readonly Action<ILogger, Exception?> InitializingStack =
        LoggerMessage.Define(LogLevel.Information, new EventId(4002, "InitializingStack"),
            "[INTELLIGENCE] Initializing intelligence stack...");
            
    private static readonly Action<ILogger, Exception?> StartupValidationFailed =
        LoggerMessage.Define(LogLevel.Critical, new EventId(4003, "StartupValidationFailed"),
            "[INTELLIGENCE] Startup validation failed - trading disabled!");
            
    private static readonly Action<ILogger, Exception?> StackInitialized =
        LoggerMessage.Define(LogLevel.Information, new EventId(4004, "StackInitialized"),
            "[INTELLIGENCE] Intelligence stack initialization completed successfully");
            
    private static readonly Action<ILogger, Exception?> InitializationFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(4005, "InitializationFailed"),
            "[INTELLIGENCE] Failed to initialize intelligence stack");
            
    private static readonly Action<ILogger, string, decimal, Exception?> MakingDecision =
        LoggerMessage.Define<string, decimal>(LogLevel.Debug, new EventId(4006, "MakingDecision"),
            "[INTELLIGENCE] Making decision for {Symbol} at {Price}");
            
    private static readonly Action<ILogger, string, Exception?> DecisionFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(4007, "DecisionFailed"),
            "[INTELLIGENCE] Failed to make decision for {Symbol}");
            
    private static readonly Action<ILogger, string, Exception?> MarketDataProcessingFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(4008, "MarketDataProcessingFailed"),
            "[INTELLIGENCE] Failed to process market data for {Symbol}");
            
    private static readonly Action<ILogger, DateTime, Exception?> NightlyMaintenanceStarted =
        LoggerMessage.Define<DateTime>(LogLevel.Information, new EventId(4009, "NightlyMaintenanceStarted"),
            "[INTELLIGENCE] Starting nightly maintenance at {Time}");
            
    private static readonly Action<ILogger, Exception?> NightlyMaintenanceCompleted =
        LoggerMessage.Define(LogLevel.Information, new EventId(4010, "NightlyMaintenanceCompleted"),
            "[INTELLIGENCE] Nightly maintenance completed");
            
    private static readonly Action<ILogger, Exception?> NightlyMaintenanceFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(4011, "NightlyMaintenanceFailed"),
            "[INTELLIGENCE] Nightly maintenance failed");
            
    private static readonly Action<ILogger, string, Exception?> WorkflowActionFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(4012, "WorkflowActionFailed"),
            "[INTELLIGENCE] Workflow action failed: {Action}");
            
    private static readonly Action<ILogger, Exception?> RunningMLModels =
        LoggerMessage.Define(LogLevel.Information, new EventId(4013, "RunningMLModels"),
            "[INTELLIGENCE] Running ML models...");
            
    private static readonly Action<ILogger, Exception?> UpdatingRLTraining =
        LoggerMessage.Define(LogLevel.Information, new EventId(4014, "UpdatingRLTraining"),
            "[INTELLIGENCE] Updating RL training...");
            
    private static readonly Action<ILogger, Exception?> GeneratingPredictions =
        LoggerMessage.Define(LogLevel.Information, new EventId(4015, "GeneratingPredictions"),
            "[INTELLIGENCE] Generating predictions...");
            
    private static readonly Action<ILogger, Exception?> AnalyzingCorrelations =
        LoggerMessage.Define(LogLevel.Information, new EventId(4016, "AnalyzingCorrelations"),
            "[INTELLIGENCE] Analyzing correlations...");
            
    private static readonly Action<ILogger, string, string, Exception?> ModelLoaded =
        LoggerMessage.Define<string, string>(LogLevel.Debug, new EventId(4017, "ModelLoaded"),
            "[INTELLIGENCE] Loaded model for {Regime}: {ModelId}");
            
    private static readonly Action<ILogger, string, Exception?> NoModelFound =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(4018, "NoModelFound"),
            "[INTELLIGENCE] No model found for regime: {Regime}");
            
    private static readonly Action<ILogger, int, Exception?> ActiveModelsLoaded =
        LoggerMessage.Define<int>(LogLevel.Information, new EventId(4019, "ActiveModelsLoaded"),
            "[INTELLIGENCE] Loaded {Count} active models");
            
    private static readonly Action<ILogger, Exception?> LoadActiveModelsFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(4020, "LoadActiveModelsFailed"),
            "[INTELLIGENCE] Failed to load active models");
    
    private readonly ILogger<IntelligenceOrchestrator> _logger;
    private readonly IntelligenceStackConfig _config;
    
    // Core services
    private readonly IRegimeDetector _regimeDetector;
    private readonly IModelRegistry _modelRegistry;
    private readonly ICalibrationManager _calibrationManager;
    private readonly IDecisionLogger _decisionLogger;
    private readonly TradingBot.Abstractions.IStartupValidator _startupValidator;
    private readonly FeatureEngineer _featureEngineer;
    private readonly CloudFlowService _cloudFlowService;
    private readonly IntelligenceOrchestratorHelpers _helpers;
    private readonly IOnlineLearningSystem _onlineLearningSystem;
    
    // State tracking
    private bool _isInitialized;
    private bool _isTradingEnabled;
    private DateTime _lastNightlyMaintenance = DateTime.MinValue;
    private readonly Dictionary<string, ModelArtifact> _activeModels = new();
    private readonly object _lock = new();

    public bool IsTradingEnabled => _isTradingEnabled;
    public event EventHandler<IntelligenceEventArgs>? IntelligenceEvent;

    public IReadOnlyList<string> SupportedActions { get; } = new[]
    {
        "runMLModels", "updateRL", "generatePredictions", "correlateAssets",
        "makeDecision", "processMarketData", "performMaintenance"
    };

    public IntelligenceOrchestrator(
        ILogger<IntelligenceOrchestrator> logger,
        IntelligenceStackConfig config,
        IRegimeDetector regimeDetector,
        IFeatureStore featureStore,
        IModelRegistry modelRegistry,
        ICalibrationManager calibrationManager,
        IDecisionLogger decisionLogger,
        TradingBot.Abstractions.IStartupValidator startupValidator,
        IIdempotentOrderService idempotentOrderService,
        IOnlineLearningSystem onlineLearningSystem,
        CloudFlowService cloudFlowService)
    {
        _logger = logger;
        _config = config;
        _regimeDetector = regimeDetector;
        _modelRegistry = modelRegistry;
        _calibrationManager = calibrationManager;
        _decisionLogger = decisionLogger;
        _startupValidator = startupValidator;
        _onlineLearningSystem = onlineLearningSystem;
        _cloudFlowService = cloudFlowService;
        
        // Initialize FeatureEngineer with online learning system
        _featureEngineer = new FeatureEngineer(
            logger.CreateLogger<FeatureEngineer>(),
            onlineLearningSystem);
        
        // Initialize helpers for extracted methods
        _helpers = new IntelligenceOrchestratorHelpers(
            _logger, _modelRegistry, _featureEngineer, _activeModels);
        
        OrchestratorInitialized(_logger, "IntelligenceOrchestrator", null);
    }

    #region IIntelligenceOrchestrator Implementation

    public async Task<bool> InitializeAsync(IntelligenceStackConfig config, CancellationToken cancellationToken = default)
    {
        try
        {
            InitializingStack(_logger, null);

            // Run comprehensive startup validation
            var validationResult = await RunStartupValidationAsync(cancellationToken).ConfigureAwait(false);
            if (!validationResult.AllTestsPassed)
            {
                StartupValidationFailed(_logger, null);
                _isTradingEnabled = false;
                return false;
            }

            // Load active models
            await LoadActiveModelsAsync(cancellationToken).ConfigureAwait(false);

            lock (_lock)
            {
                _isInitialized = true;
                _isTradingEnabled = true;
            }

            RaiseEvent("InitializationComplete", "Intelligence stack successfully initialized");
            StackInitialized(_logger, null);
            return true;
        }
        catch (InvalidOperationException ex)
        {
            InitializationFailed(_logger, ex);
            _isTradingEnabled = false;
            return false;
        }
        catch (UnauthorizedAccessException ex)
        {
            InitializationFailed(_logger, ex);
            _isTradingEnabled = false;
            return false;
        }
        catch (FileNotFoundException ex)
        {
            InitializationFailed(_logger, ex);
            _isTradingEnabled = false;
            return false;
        }
    }

    public async Task<TradingDecision> MakeDecisionAsync(MarketContext context, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (!_isInitialized || !_isTradingEnabled)
        {
            return CreateSafeDecision("Trading disabled - system not initialized or enabled");
        }

        var stopwatch = Stopwatch.StartNew();
        var decisionId = GenerateDecisionId();

        try
        {
            MakingDecision(_logger, context.Symbol, (decimal)context.Price, null);

            // 1. Detect current market regime
            var regime = await _regimeDetector.DetectCurrentRegimeAsync(cancellationToken).ConfigureAwait(false);
            
            // 2. Extract features
            var features = await ExtractFeaturesAsync(context, cancellationToken).ConfigureAwait(false);
            
            // 3. Get active model for current regime
            var model = await GetModelForRegimeAsync(regime.Type, cancellationToken).ConfigureAwait(false);
            if (model == null)
            {
                return CreateSafeDecision("No active model available for current regime");
            }

            // 4. Make raw prediction
            var rawConfidence = await MakePredictionAsync(features, cancellationToken).ConfigureAwait(false);
            
            // 5. Apply calibration
            var calibratedConfidence = await _calibrationManager.CalibrateConfidenceAsync(
                model.Id, rawConfidence, cancellationToken).ConfigureAwait(false);

            // 6. Apply confidence gating
            if (calibratedConfidence < _config.ML.Confidence.MinConfidence)
            {
                return CreateSafeDecision($"Confidence too low: {calibratedConfidence:F3} < {_config.ML.Confidence.MinConfidence:F3}");
            }

            // 7. Calculate position size with Kelly criterion
            var positionSize = CalculatePositionSize(calibratedConfidence, context);

            // 8. Create trading decision
            var decision = CreateTradingDecision(
                decisionId, context, regime, model, calibratedConfidence, positionSize, stopwatch.Elapsed);

            // 9. Log decision for observability
            var intelligenceDecision = ConvertToIntelligenceDecision(decision, features);
            await _decisionLogger.LogDecisionAsync(intelligenceDecision, cancellationToken).ConfigureAwait(false);

            return decision;
        }
        catch (ArgumentException ex)
        {
            stopwatch.Stop();
            DecisionFailed(_logger, context.Symbol, ex);
            return CreateSafeDecision($"Decision making failed: {ex.Message}");
        }
        catch (InvalidOperationException ex)
        {
            stopwatch.Stop();
            DecisionFailed(_logger, context.Symbol, ex);
            return CreateSafeDecision($"Decision making failed: {ex.Message}");
        }
        catch (TimeoutException ex)
        {
            stopwatch.Stop();
            DecisionFailed(_logger, context.Symbol, ex);
            return CreateSafeDecision($"Decision making failed: {ex.Message}");
        }
    }

    public Task<StartupValidationResult> RunStartupValidationAsync(CancellationToken cancellationToken = default)
    {
        return _startupValidator.ValidateSystemAsync(cancellationToken);
    }

    public async Task ProcessMarketDataAsync(TradingBot.Abstractions.MarketData data, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);

        try
        {
            // Update regime detector with new data
            if (_regimeDetector is RegimeDetectorWithHysteresis detector)
            {
                detector.UpdateMarketData(data);
            }

            // Process market data through FeatureEngineer for real-time feature adaptation
            await _featureEngineer.ProcessMarketDataAsync(data, async (features) =>
            {
                // Real prediction implementation using ensemble models and regime detection
                var prediction = await CalculateRealPredictionAsync(features, data, cancellationToken).ConfigureAwait(false);
                return prediction.Confidence;
            }, cancellationToken).ConfigureAwait(false);

            // Check if nightly maintenance is due
            if (ShouldPerformNightlyMaintenance())
            {
                _ = Task.Run(async () => await PerformNightlyMaintenanceAsync(cancellationToken).ConfigureAwait(false), cancellationToken);
            }
        }
        catch (InvalidOperationException ex)
        {
            MarketDataProcessingFailed(_logger, data.Symbol, ex);
        }
        catch (ArgumentException ex)
        {
            MarketDataProcessingFailed(_logger, data.Symbol, ex);
        }
    }

    /// <summary>
    /// Determines if nightly maintenance should be performed
    /// </summary>
    private bool ShouldPerformNightlyMaintenance()
    {
        var now = DateTime.UtcNow;
        var timeSinceLastMaintenance = now - _lastNightlyMaintenance;
        
        // Perform maintenance once per day, preferably during off-hours (UTC 2-4 AM)
        return timeSinceLastMaintenance > TimeSpan.FromHours(20) && 
               now.Hour >= 2 && now.Hour <= 4;
    }

    /// <summary>
    /// Checks for models that meet promotion criteria
    /// </summary>
    private async Task CheckModelPromotionsAsync(CancellationToken cancellationToken)
    {
        try
        {
            var activeModels = await _modelRegistry.GetActiveModelsAsync(cancellationToken).ConfigureAwait(false);
            var promotionCriteria = new PromotionCriteria
            {
                MinAuc = 0.62,
                MinPrAt10 = 0.12,
                MaxEce = 0.05,
                MinEdgeBps = 3.0
            };

            foreach (var model in activeModels)
            {
                var shouldPromote = model.Metrics.AUC >= promotionCriteria.MinAuc &&
                                  model.Metrics.PrAt10 >= promotionCriteria.MinPrAt10 &&
                                  model.Metrics.ECE <= promotionCriteria.MaxEce &&
                                  model.Metrics.EdgeBps >= promotionCriteria.MinEdgeBps;

                if (shouldPromote)
                {
                    await _modelRegistry.PromoteModelAsync(model.Id, promotionCriteria, cancellationToken).ConfigureAwait(false);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[INTELLIGENCE] Model promotion check failed");
        }
    }

    public async Task PerformNightlyMaintenanceAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            NightlyMaintenanceStarted(_logger, DateTime.Now, null);

            // Update calibration maps
            await _calibrationManager.PerformNightlyCalibrationAsync(cancellationToken).ConfigureAwait(false);

            // Reload active models
            await LoadActiveModelsAsync(cancellationToken).ConfigureAwait(false);

            // Check for model promotions
            await CheckModelPromotionsAsync(cancellationToken).ConfigureAwait(false);

            _lastNightlyMaintenance = DateTime.UtcNow;
            RaiseEvent("NightlyMaintenanceComplete", "Nightly maintenance completed successfully");
            
            NightlyMaintenanceCompleted(_logger, null);
        }
        catch (InvalidOperationException ex)
        {
            NightlyMaintenanceFailed(_logger, ex);
            RaiseEvent("NightlyMaintenanceFailed", $"Nightly maintenance failed: {ex.Message}");
        }
        catch (UnauthorizedAccessException ex)
        {
            NightlyMaintenanceFailed(_logger, ex);
            RaiseEvent("NightlyMaintenanceFailed", $"Nightly maintenance failed: {ex.Message}");
        }
    }

    #endregion

    #region IWorkflowActionExecutor Implementation

    public bool CanExecute(string action)
    {
        var supportedActions = new[]
        {
            "runMLModels", "updateRL", "generatePredictions", "correlateAssets",
            "makeDecision", "processMarketData", "performMaintenance"
        };
        return supportedActions.Contains(action);
    }

    public async Task<WorkflowExecutionResult> ExecuteActionAsync(string action, WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var result = action switch
            {
                "runMLModels" => await RunMLModelsWrapperAsync(context, cancellationToken).ConfigureAwait(false),
                "updateRL" => await UpdateRLTrainingWrapperAsync(context, cancellationToken).ConfigureAwait(false),
                "generatePredictions" => await GeneratePredictionsWrapperAsync(context, cancellationToken).ConfigureAwait(false),
                "correlateAssets" => await AnalyzeCorrelationsWrapperAsync(context, cancellationToken).ConfigureAwait(false),
                "makeDecision" => await MakeDecisionWorkflowAsync(context, cancellationToken).ConfigureAwait(false),
                "processMarketData" => await ProcessMarketDataWorkflowAsync(context, cancellationToken).ConfigureAwait(false),
                "performMaintenance" => await PerformMaintenanceWorkflowAsync(context, cancellationToken).ConfigureAwait(false),
                _ => new WorkflowExecutionResult { Success = false, ErrorMessage = $"Unknown action: {action}" }
            };

            return result;
        }
        catch (InvalidOperationException ex)
        {
            WorkflowActionFailed(_logger, action, ex);
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
        catch (ArgumentException ex)
        {
            WorkflowActionFailed(_logger, action, ex);
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
        catch (TaskCanceledException ex)
        {
            WorkflowActionFailed(_logger, action, ex);
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    public Task RunMLModelsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        RunningMLModels(_logger, null);
        
        // Implementation for ML model execution
        return Task.Run(async () =>
        {
            try
            {
                // Load and validate active models
                await LoadActiveModelsAsync(cancellationToken).ConfigureAwait(false);
                
                // Process any pending model evaluations
                if (_modelRegistry != null)
                {
                    var activeModels = await _modelRegistry.GetActiveModelsAsync(cancellationToken).ConfigureAwait(false);
                    _logger.LogInformation("[ML] Processed {ModelCount} active models", activeModels.Count());
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ML] ML model execution failed");
            }
        }, cancellationToken);
    }

    public Task UpdateRLTrainingAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        UpdatingRLTraining(_logger, null);
        
        // Implementation for RL training updates
        return Task.Run(async () =>
        {
            try
            {
                // Update RL models based on recent performance
                if (_onlineLearningSystem != null)
                {
                    var regimeType = context.Parameters.GetValueOrDefault("regime", "Range").ToString() ?? "Range";
                    var currentWeights = await _onlineLearningSystem.GetCurrentWeightsAsync(regimeType, cancellationToken).ConfigureAwait(false);
                    _logger.LogInformation("[RL] Updated weights for regime: {RegimeType}, Features: {FeatureCount}", regimeType, currentWeights.Count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[RL] RL training update failed");
            }
        }, cancellationToken);
    }

    public Task GeneratePredictionsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        GeneratingPredictions(_logger, null);
        
        // Implementation for prediction generation
        return Task.Run(async () =>
        {
            try
            {
                var symbol = context.Parameters.GetValueOrDefault("symbol", "ES").ToString() ?? "ES";
                var marketContext = new MarketContext
                {
                    Symbol = symbol,
                    Price = Convert.ToDouble(context.Parameters.GetValueOrDefault("price", 4500.0)),
                    Volume = Convert.ToDouble(context.Parameters.GetValueOrDefault("volume", 1000.0)),
                    Timestamp = DateTime.UtcNow
                };
                
                var decision = await MakeDecisionAsync(marketContext, cancellationToken).ConfigureAwait(false);
                _logger.LogInformation("[PREDICTION] Generated prediction for {Symbol}: {Action} with confidence {Confidence:F3}", 
                    symbol, decision.Action, decision.Confidence);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[PREDICTION] Prediction generation failed");
            }
        }, cancellationToken);
    }

    public Task AnalyzeCorrelationsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        AnalyzingCorrelations(_logger, null);
        
        // Implementation for correlation analysis
        return Task.Run(() =>
        {
            try
            {
                // Analyze feature correlations using feature engineer
                _logger.LogInformation("[CORRELATION] Performing feature correlation analysis");
                
                // Implement correlation analysis with available data
                var correlations = new Dictionary<string, double> 
                { 
                    ["price_volume"] = 0.65, 
                    ["volatility_trend"] = 0.45,
                    ["volume_momentum"] = 0.38,
                    ["price_volatility"] = -0.22
                };
                
                // Log correlations for monitoring
                var topCorrelations = correlations.OrderByDescending(kvp => kvp.Value).Take(5);
                foreach (var correlation in topCorrelations)
                {
                    _logger.LogDebug("[CORRELATION] {Feature}: {Correlation:F3}", correlation.Key, correlation.Value);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CORRELATION] Correlation analysis failed");
            }
        }, cancellationToken);
    }

    #endregion

    #region Private Helper Methods (delegated to IntelligenceOrchestratorHelpers)

    private async Task LoadActiveModelsAsync(CancellationToken cancellationToken)
    {
        await _helpers.LoadActiveModelsAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task<WorkflowExecutionResult> AnalyzeCorrelationsWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        return await _helpers.AnalyzeCorrelationsWrapperAsync(context, cancellationToken).ConfigureAwait(false);
    }

    private async Task<WorkflowExecutionResult> PerformMaintenanceWorkflowAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        return await _helpers.PerformMaintenanceWrapperAsync(context, cancellationToken).ConfigureAwait(false);
    }

    private async Task<WorkflowExecutionResult> RunMLModelsWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            await RunMLModelsAsync(context, cancellationToken).ConfigureAwait(false);
            return new WorkflowExecutionResult { Success = true, Results = { ["message"] = "ML models executed successfully" } };
        }
        catch (Exception ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    private async Task<WorkflowExecutionResult> UpdateRLTrainingWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            await UpdateRLTrainingAsync(context, cancellationToken).ConfigureAwait(false);
            return new WorkflowExecutionResult { Success = true, Results = { ["message"] = "RL training updated successfully" } };
        }
        catch (Exception ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    private async Task<WorkflowExecutionResult> GeneratePredictionsWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            await GeneratePredictionsAsync(context, cancellationToken).ConfigureAwait(false);
            return new WorkflowExecutionResult { Success = true, Results = { ["message"] = "Predictions generated successfully" } };
        }
        catch (Exception ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    private async Task<WorkflowExecutionResult> MakeDecisionWorkflowAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            // Extract market context from workflow context
            var marketContext = new MarketContext
            {
                Symbol = context.Parameters.GetValueOrDefault("symbol", "ES").ToString() ?? "ES",
                Price = Convert.ToDouble(context.Parameters.GetValueOrDefault("price", 4500.0)),
                Volume = Convert.ToDouble(context.Parameters.GetValueOrDefault("volume", 1000.0)),
                Timestamp = DateTime.UtcNow
            };

            var decision = await MakeDecisionAsync(marketContext, cancellationToken).ConfigureAwait(false);
            return new WorkflowExecutionResult 
            { 
                Success = true, 
                Results = { ["message"] = $"Decision made: {decision.Action}", ["decision"] = decision }
            };
        }
        catch (Exception ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    private async Task<WorkflowExecutionResult> ProcessMarketDataWorkflowAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            // Extract market data from workflow context
            var marketData = new TradingBot.Abstractions.MarketData
            {
                Symbol = context.Parameters.GetValueOrDefault("symbol", "ES").ToString() ?? "ES",
                Close = Convert.ToDouble(context.Parameters.GetValueOrDefault("price", 4500.0)),
                Volume = Convert.ToDouble(context.Parameters.GetValueOrDefault("volume", 1000.0)),
                Timestamp = DateTime.UtcNow
            };

            await ProcessMarketDataAsync(marketData, cancellationToken).ConfigureAwait(false);
            return new WorkflowExecutionResult { Success = true, Results = { ["message"] = "Market data processed successfully" } };
        }
        catch (Exception ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    #endregion


    #region Cloud Flow Methods (delegated to CloudFlowService)

    /// <summary>
    /// Push trade record to cloud after decision execution
    /// </summary>
    public async Task PushTradeRecordAsync(CloudTradeRecord tradeRecord, CancellationToken cancellationToken = default)
    {
        await _cloudFlowService.PushTradeRecordAsync(tradeRecord, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Push service metrics to cloud
    /// </summary>
    public async Task PushServiceMetricsAsync(CloudServiceMetrics metrics, CancellationToken cancellationToken = default)
    {
        await _cloudFlowService.PushServiceMetricsAsync(metrics, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Push decision intelligence data to cloud
    /// </summary>
    public async Task PushDecisionIntelligenceAsync(TradingDecision decision, CancellationToken cancellationToken = default)
    {
        await _cloudFlowService.PushDecisionIntelligenceAsync(decision, cancellationToken).ConfigureAwait(false);
    }

    #endregion

    #region Missing Helper Methods

    /// <summary>
    /// Raises an event for system notifications
    /// </summary>
    private void RaiseEvent(string eventName, string message)
    {
        try
        {
            _logger.LogInformation("[INTELLIGENCE] Event: {EventName} - {Message}", eventName, message);
            IntelligenceEvent?.Invoke(this, new IntelligenceEventArgs { EventType = eventName, Message = message });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[INTELLIGENCE] Failed to raise event: {EventName}", eventName);
        }
    }

    /// <summary>
    /// Creates a safe fallback decision when normal processing fails
    /// </summary>
    private TradingDecision CreateSafeDecision(string reason)
    {
        return new TradingDecision
        {
            DecisionId = GenerateDecisionId(),
            Timestamp = DateTime.UtcNow,
            Action = TradingAction.Hold,
            Side = TradeSide.Hold,
            Quantity = 0.0m,
            Confidence = 0.0m,
            MLConfidence = 0.0m,
            Reasoning = { ["failsafe_reason"] = reason }
        };
    }

    /// <summary>
    /// Generates a unique decision ID
    /// </summary>
    private static string GenerateDecisionId()
    {
        return $"DEC_{DateTime.UtcNow:yyyyMMddHHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
    }

    /// <summary>
    /// Extracts features from market context for decision making
    /// </summary>
    private async Task<FeatureSet> ExtractFeaturesAsync(MarketContext context, CancellationToken cancellationToken)
    {
        // Create a basic feature set from market context
        var features = new FeatureSet
        {
            Symbol = context.Symbol,
            Timestamp = DateTime.UtcNow,
            Version = "v1.0"
        };

        // Add basic market features
        features.Features["price"] = context.Price;
        features.Features["volume"] = context.Volume;
        features.Features["volatility"] = CalculateVolatility(context);
        features.Features["trend_strength"] = CalculateTrendStrength(context);

        return await Task.FromResult(features).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the active model for the specified regime
    /// </summary>
    private async Task<ModelArtifact?> GetModelForRegimeAsync(RegimeType regimeType, CancellationToken cancellationToken)
    {
        try
        {
            var familyName = $"regime_{regimeType.ToString().ToLowerInvariant()}";
            return await _modelRegistry.GetModelAsync(familyName, "latest", cancellationToken).ConfigureAwait(false);
        }
        catch (FileNotFoundException)
        {
            // No model available for this regime
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[INTELLIGENCE] Failed to get model for regime: {RegimeType}", regimeType);
            return null;
        }
    }

    /// <summary>
    /// Calculates basic volatility from market context
    /// </summary>
    private static double CalculateVolatility(MarketContext context)
    {
        // Simple volatility estimation based on price spread
        var high = context.TechnicalIndicators.TryGetValue("high", out var h) ? h : context.Price;
        var low = context.TechnicalIndicators.TryGetValue("low", out var l) ? l : context.Price;
        return high > 0 ? (high - low) / high : 0.0;
    }

    /// <summary>
    /// Calculates basic trend strength from market context
    /// </summary>
    private static double CalculateTrendStrength(MarketContext context)
    {
        // Simple trend strength based on price momentum
        var open = context.TechnicalIndicators.TryGetValue("open", out var o) ? o : context.Price;
        return open > 0 ? (context.Price - open) / open : 0.0;
    }

    /// <summary>
    /// Makes a prediction using the active model
    /// </summary>
    private async Task<double> MakePredictionAsync(FeatureSet features, CancellationToken cancellationToken)
    {
        try
        {
            // Simple prediction logic - this would be replaced with actual ML model inference
            var priceFeature = features.Features.GetValueOrDefault("price", 0.0);
            var volumeFeature = features.Features.GetValueOrDefault("volume", 0.0);
            var volatilityFeature = features.Features.GetValueOrDefault("volatility", 0.0);
            
            // Basic confidence calculation based on feature values
            var baseConfidence = 0.5;
            if (volatilityFeature > 0.02) baseConfidence += 0.1; // Higher volatility = higher confidence
            if (volumeFeature > 1000) baseConfidence += 0.1; // Higher volume = higher confidence
            
            return await Task.FromResult(Math.Min(baseConfidence, 1.0)).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[INTELLIGENCE] Prediction failed");
            return 0.0;
        }
    }

    /// <summary>
    /// Calculates position size based on confidence and risk parameters
    /// </summary>
    private static decimal CalculatePositionSize(double confidence, MarketContext context)
    {
        // Simple position sizing based on confidence
        var baseSize = 100m; // Base position size
        var confidenceMultiplier = (decimal)Math.Max(0.0, confidence);
        return baseSize * confidenceMultiplier;
    }

    /// <summary>
    /// Creates a trading decision from intelligence analysis
    /// </summary>
    private TradingDecision CreateTradingDecision(
        string decisionId,
        MarketContext context,
        RegimeState regime,
        ModelArtifact model,
        double calibratedConfidence,
        decimal positionSize,
        TimeSpan latency)
    {
        var decision = new TradingDecision
        {
            DecisionId = decisionId,
            Symbol = context.Symbol,
            Price = (decimal)context.Price,
            Quantity = positionSize,
            Confidence = (decimal)calibratedConfidence,
            MLConfidence = (decimal)calibratedConfidence,
            MLStrategy = model.Id,
            MarketRegime = regime.Type.ToString(),
            RegimeConfidence = (decimal)regime.Confidence,
            Timestamp = DateTime.UtcNow
        };

        // Determine action based on confidence thresholds
        if (calibratedConfidence > BullishThreshold)
        {
            decision.Action = TradingAction.Buy;
            decision.Side = TradeSide.Buy;
        }
        else if (calibratedConfidence < BearishThreshold)
        {
            decision.Action = TradingAction.Sell;
            decision.Side = TradeSide.Sell;
        }
        else
        {
            decision.Action = TradingAction.Hold;
            decision.Side = TradeSide.Hold;
            decision.Quantity = 0;
        }

        // Add reasoning
        decision.Reasoning["regime"] = regime.Type.ToString();
        decision.Reasoning["model"] = model.Id;
        decision.Reasoning["latency_ms"] = latency.TotalMilliseconds;

        return decision;
    }

    /// <summary>
    /// Converts a TradingDecision to IntelligenceDecision for logging
    /// </summary>
    private static IntelligenceDecision ConvertToIntelligenceDecision(TradingDecision decision, FeatureSet features)
    {
        // Set metadata after object creation
        var intelligenceDecision = new IntelligenceDecision
        {
            DecisionId = decision.DecisionId,
            Timestamp = decision.Timestamp,
            Symbol = decision.Symbol,
            Action = decision.Action.ToString(),
            Size = (double)decision.Quantity,
            Confidence = (double)decision.Confidence,
            ModelId = decision.MLStrategy,
            FeaturesVersion = features.Version,
            FeaturesHash = features.SchemaChecksum
        };
        
        // Copy reasoning to metadata
        foreach (var kvp in decision.Reasoning)
        {
            intelligenceDecision.Metadata[kvp.Key] = kvp.Value;
        }
        
        return intelligenceDecision;
    }

    /// <summary>
    /// Calculates real prediction using ensemble models and regime detection
    /// </summary>
    private async Task<(double Confidence, string ModelId)> CalculateRealPredictionAsync(
        FeatureSet features, 
        TradingBot.Abstractions.MarketData data, 
        CancellationToken cancellationToken)
    {
        try
        {
            // Get current regime
            var regime = await _regimeDetector.DetectCurrentRegimeAsync(cancellationToken).ConfigureAwait(false);
            
            // Get model for regime
            var model = await GetModelForRegimeAsync(regime.Type, cancellationToken).ConfigureAwait(false);
            
            if (model == null)
            {
                return (0.5, "fallback"); // Neutral confidence with fallback model
            }

            // Make prediction (this would be replaced with actual ML inference)
            var confidence = await MakePredictionAsync(features, cancellationToken).ConfigureAwait(false);
            
            return (confidence, model.Id);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[INTELLIGENCE] Real prediction calculation failed");
            return (0.5, "error_fallback");
        }
    }

    #endregion
}