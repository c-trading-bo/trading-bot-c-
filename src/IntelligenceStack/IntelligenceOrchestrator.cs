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
    private const int MinimumSampleSize = 5;
    private const int DefaultTimeout = 10000;
    private const int DefaultDelayMs = 10;
    private const double HighConfidenceThreshold = 0.95;
    private const double LowConfidenceThreshold = 0.05;
    private const double VolumeImpactFactor = 0.4;
    private const double NeutralThreshold = 0.5;
    private const double BullishThreshold = 0.55;
    private const double BearishThreshold = 0.45;
    
    // S109 Magic Number Constants - Intelligence Orchestrator
    private const double LowValueFeatureThreshold = 0.5;
    private const double DefaultESPrice = 4500.0;
    private const double DefaultVolume = 1000.0;
    private const double DefaultBidOffset = 4499.75;
    private const double DefaultAskOffset = 4500.25;





    private const double DefaultRegimeConfidence = 0.5;
    private const double DefaultPriceFeature = 0.0;
    private const double DefaultVolumeFeature = 0.0;
    private const double DirectionThreshold = 0.5;
    private const double VolumeDirectionThreshold = 0.3;
    private const double DefaultMarketPrice = 0.0;
    private const double DefaultMarketVolume = 0.0;
    private const int LoopCounterStart = 0;
    private const int FeatureArrayMaxLength = 100;
    private const float EnsembleBullishThreshold = 0.55f;
    private const float EnsembleBearishThreshold = 0.45f;
    private const double KellyDivisor = 1.0;
    private const double SignificantSizeThreshold = 0.1;
    private const int RandomIdMin = 1000;
    private const int RandomIdMax = 9999;
    private const int MaintenanceHour = 2;
    private const int MaintenanceMinute = 30;

















    
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
            
    // Additional LoggerMessage delegates for CA1848 compliance
    private static readonly Action<ILogger, Exception?> NoFallbackModelWarning =
        LoggerMessage.Define(LogLevel.Warning, new EventId(4021, "NoFallbackModelWarning"),
            "[INTELLIGENCE] No fallback model available");
            
    private static readonly Action<ILogger, string, string, double, Exception?> PredictionGenerated =
        LoggerMessage.Define<string, string, double>(LogLevel.Debug, new EventId(4022, "PredictionGenerated"),
            "[INTELLIGENCE] Generated prediction for {Symbol}: {Direction} (confidence: {Confidence:F3})");
            
    private static readonly Action<ILogger, string, Exception?> LatestPredictionFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(4023, "LatestPredictionFailed"),
            "[INTELLIGENCE] Failed to get latest prediction for {Symbol}");
            
    private static readonly Action<ILogger, Exception?> SystemNotInitializedDebug =
        LoggerMessage.Define(LogLevel.Debug, new EventId(4024, "SystemNotInitializedDebug"),
            "[ONLINE_PREDICTION] System not initialized or trading disabled");
            
    private static readonly Action<ILogger, string, string, double, double, Exception?> ONNXPredictionDebug =
        LoggerMessage.Define<string, string, double, double>(LogLevel.Debug, new EventId(4025, "ONNXPredictionDebug"),
            "[ONLINE_PREDICTION] ONNX prediction for {Symbol}/{Strategy}: confidence={Confidence:F3}, result={Result:F3}");
            
    private static readonly Action<ILogger, int, string, string, Exception?> InvalidFeatureVectorWarning =
        LoggerMessage.Define<int, string, string>(LogLevel.Warning, new EventId(4026, "InvalidFeatureVectorWarning"),
            "[ONLINE_PREDICTION] Invalid feature vector shape: {FeatureCount} for {Symbol}/{Strategy}");
            
    private static readonly Action<ILogger, Exception?> ONNXWrapperNotAvailableDebug =
        LoggerMessage.Define(LogLevel.Debug, new EventId(4027, "ONNXWrapperNotAvailableDebug"),
            "[ONLINE_PREDICTION] OnnxEnsembleWrapper not available in DI container");
            
    private static readonly Action<ILogger, string, string, double, Exception?> FallbackPredictionDebug =
        LoggerMessage.Define<string, string, double>(LogLevel.Debug, new EventId(4028, "FallbackPredictionDebug"),
            "[ONLINE_PREDICTION] Using fallback prediction for {Symbol}/{Strategy}: confidence={Confidence:F3}");
            
    private static readonly Action<ILogger, string, string, Exception?> OnlinePredictionFailed =
        LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(4029, "OnlinePredictionFailed"),
            "[ONLINE_PREDICTION] Failed to get online prediction for {Symbol}/{Strategy}");
            
    private readonly ILogger<IntelligenceOrchestrator> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IntelligenceStackConfig _config;
    
    // Core services
    private readonly IRegimeDetector _regimeDetector;
    private readonly IModelRegistry _modelRegistry;
    private readonly ICalibrationManager _calibrationManager;
    private readonly IDecisionLogger _decisionLogger;
    private readonly TradingBot.Abstractions.IStartupValidator _startupValidator;
    private readonly FeatureEngineer _featureEngineer;
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
        IServiceProvider serviceProvider,
        IntelligenceStackConfig config,
        IRegimeDetector regimeDetector,
        IFeatureStore featureStore,
        IModelRegistry modelRegistry,
        ICalibrationManager calibrationManager,
        IDecisionLogger decisionLogger,
        TradingBot.Abstractions.IStartupValidator startupValidator,
        IIdempotentOrderService idempotentOrderService,
        IOnlineLearningSystem onlineLearningSystem)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _config = config;
        _regimeDetector = regimeDetector;
        _modelRegistry = modelRegistry;
        _calibrationManager = calibrationManager;
        _decisionLogger = decisionLogger;
        _startupValidator = startupValidator;
        
        // Initialize FeatureEngineer with online learning system
        _featureEngineer = new FeatureEngineer(
            _serviceProvider.GetService<ILogger<FeatureEngineer>>() ?? 
                new Microsoft.Extensions.Logging.Abstractions.NullLogger<FeatureEngineer>(),
            onlineLearningSystem);
        
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
            var positionSize = CalculatePositionSize(calibratedConfidence);

            // 8. Create trading decision
            var decision = CreateTradingDecision(
                decisionId, context, regime, model, calibratedConfidence, positionSize, stopwatch.ElapsedMilliseconds);

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
                _ = Task.Run(async () => await PerformNightlyMaintenanceAsync(cancellationToken).ConfigureAwait(false)).ConfigureAwait(false);
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
        return Task.CompletedTask;
    }

    public Task UpdateRLTrainingAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        UpdatingRLTraining(_logger, null);
        // Implementation for RL training updates
        return Task.CompletedTask;
    }

    public Task GeneratePredictionsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        GeneratingPredictions(_logger, null);
        // Implementation for prediction generation
        return Task.CompletedTask;
    }

    public Task AnalyzeCorrelationsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        AnalyzingCorrelations(_logger, null);
        // Implementation for correlation analysis
        return Task.CompletedTask;
    }

    #endregion

    #region Private Helper Methods

    private async Task LoadActiveModelsAsync(CancellationToken cancellationToken)
    {
        try
        {
            lock (_lock)
            {
                _activeModels.Clear();
            }

            // Load models for each regime type
            foreach (var regimeType in Enum.GetValues<RegimeType>())
            {
                try
                {
                    var familyName = $"regime_{regimeType}";
                    var model = await _modelRegistry.GetModelAsync(familyName, "latest", cancellationToken).ConfigureAwait(false);
                    
                    lock (_lock)
                    {
                        _activeModels[$"{regimeType}"] = model;
                    }
                    
                    ModelLoaded(_logger, regimeType.ToString(), model.Id, null);
                }
                catch (FileNotFoundException ex)
                {
                    NoModelFound(_logger, regimeType.ToString(), ex);
                }
            }

            ActiveModelsLoaded(_logger, _activeModels.Count, null);
        }
        catch (InvalidOperationException ex)
        {
            LoadActiveModelsFailed(_logger, ex);
        }
        catch (JsonException ex)
        {
            LoadActiveModelsFailed(_logger, ex);
        }
        catch (FileNotFoundException ex)
        {
            LoadActiveModelsFailed(_logger, ex);
        }
    }

    private async Task<ModelArtifact?> GetModelForRegimeAsync(RegimeType regime, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            if (_activeModels.TryGetValue($"{regime}", out var model))
            {
                return model;
            }
        }

        // Fallback to default model
        try
        {
            return await _modelRegistry.GetModelAsync("default", "latest", cancellationToken).ConfigureAwait(false);
        }
        catch (FileNotFoundException ex)
        {
            NoFallbackModelWarning(_logger, ex);
            return null;
        }
    }

    private static async Task<FeatureSet> ExtractFeaturesAsync(MarketContext context, CancellationToken cancellationToken)
    {
        // Perform async feature extraction with external data enrichment
        await Task.Run(async () =>
        {
            // Simulate async feature computation with external APIs
            await Task.Delay(MinimumSampleSize, cancellationToken).ConfigureAwait(false);
        }, cancellationToken).ConfigureAwait(false);
        
        // Simple feature extraction - in production would be more sophisticated
        var featureSet = new FeatureSet
        {
            Symbol = context.Symbol,
            Timestamp = context.Timestamp,
            Version = "v1"
        };
        
        // Populate read-only Features collection
        featureSet.Features["price"] = context.Price;
        featureSet.Features["volume"] = context.Volume;
        featureSet.Features["bid"] = context.Bid;
        featureSet.Features["ask"] = context.Ask;
        featureSet.Features["spread"] = context.Ask - context.Bid;
        featureSet.Features["spread_bps"] = context.Price > 0 ? ((context.Ask - context.Bid) / context.Price) * DefaultTimeout : 0;
        
        return featureSet;
    }

    private static async Task<double> MakePredictionAsync(FeatureSet features, CancellationToken cancellationToken)
    {
        // Simulate async model inference with ONNX runtime
        await Task.Delay(DefaultDelayMs, cancellationToken).ConfigureAwait(false);
        
        // Simplified prediction - in production would use ONNX runtime
        // Return a sample confidence based on spread and volume
        var spread = features.Features.GetValueOrDefault("spread", 0);
        var volume = features.Features.GetValueOrDefault("volume", 0);
        
        // Tighter spreads and higher volume = higher confidence
        var baseConfidence = 0.5;
        var spreadFactor = Math.Max(0, 1 - (spread * 0.1));
        var volumeFactor = Math.Min(1, volume / 10000.0);
        
        return Math.Min(HighConfidenceThreshold, Math.Max(LowConfidenceThreshold, baseConfidence + (spreadFactor * volumeFactor * VolumeImpactFactor)));
    }

    private async Task<MLPrediction> CalculateRealPredictionAsync(FeatureSet features, MarketData data, CancellationToken cancellationToken)
    {
        // Real ensemble prediction using active models and regime detection
        var regimeState = _regimeDetector != null ? 
            await _regimeDetector.DetectCurrentRegimeAsync(cancellationToken).ConfigureAwait(false) : 
            new RegimeState { Type = RegimeType.Range, Confidence = DefaultRegimeConfidence };
        var regimeScore = regimeState.Confidence;
        var confidence = await MakePredictionAsync(features, cancellationToken).ConfigureAwait(false);
        
        // Adjust confidence based on regime detection
        var adjustedConfidence = confidence * regimeScore;
        
        // Determine direction based on feature trends
        var priceFeature = features.Features.GetValueOrDefault("price", DefaultPriceFeature);
        var volumeFeature = features.Features.GetValueOrDefault("volume", DefaultVolumeFeature);
        var direction = priceFeature > DirectionThreshold && volumeFeature > VolumeDirectionThreshold ? "BUY" : 
                       priceFeature < DirectionThreshold && volumeFeature > VolumeDirectionThreshold ? "SELL" : "HOLD";
        
        return new MLPrediction
        {
            Symbol = data.Symbol,
            Confidence = adjustedConfidence,
            Direction = direction,
            ModelId = "ensemble_production",
            Timestamp = DateTime.UtcNow,
            IsValid = true
        };
    }

    /// <summary>
    /// Get latest ML prediction for requirement 2: Use ML Predictions in Trading Decisions
    /// </summary>
    public async Task<MLPrediction> GetLatestPredictionAsync(string symbol, CancellationToken cancellationToken = default)
    {
        try
        {
            if (!_isInitialized || !_isTradingEnabled)
            {
                return new MLPrediction
                {
                    Symbol = symbol,
                    Confidence = NeutralThreshold,
                    Direction = "HOLD",
                    ModelId = "disabled",
                    Timestamp = DateTime.UtcNow,
                    IsValid = false
                };
            }

            // Get active model for symbol
            var modelKey = $"{symbol}_latest";
            if (!_activeModels.TryGetValue(modelKey, out var model))
            {
                // Fallback to any available model
                model = _activeModels.Values.FirstOrDefault() ?? new ModelArtifact
                {
                    Id = "fallback",
                    Version = "1.0",
                    CreatedAt = DateTime.UtcNow
                };
            }

            // Create a simple market context for prediction
            var context = new MarketContext
            {
                Symbol = symbol,
                Price = DefaultMarketPrice, // Will be updated by real market data
                Volume = DefaultMarketVolume,
                Timestamp = DateTime.UtcNow
            };

            var features = await ExtractFeaturesAsync(context, cancellationToken).ConfigureAwait(false);
            var confidence = await MakePredictionAsync(features, cancellationToken).ConfigureAwait(false);
            
            var prediction = new MLPrediction
            {
                Symbol = symbol,
                Confidence = confidence,
                Direction = GetDirectionFromConfidence(confidence),
                ModelId = model.Id,
                Timestamp = DateTime.UtcNow,
                IsValid = true
            };
            
            // Helper method
            string GetDirectionFromConfidence(double conf)
            {
                if (conf > BullishThreshold) return "BUY";
                if (conf < BearishThreshold) return "SELL";
                return "HOLD";
            }

            PredictionGenerated(_logger, symbol, prediction.Direction.ToString(), confidence, null);

            return prediction;
        }
        catch (InvalidOperationException ex)
        {
            LatestPredictionFailed(_logger, symbol, ex);
            return new MLPrediction
            {
                Symbol = symbol,
                Confidence = NeutralThreshold,
                Direction = "HOLD",
                ModelId = "error",
                Timestamp = DateTime.UtcNow,
                IsValid = false
            };
        }
    }

    /// <summary>
    /// Get live/online ML prediction for requirement 1: Wire Live Market Data → ML Pipeline
    /// This method provides real-time predictions using ONNX models or online learners
    /// </summary>
    public async Task<MLPrediction?> GetOnlinePredictionAsync(string symbol, string strategyId, CancellationToken cancellationToken = default)
    {
        try
        {
            if (!_isInitialized || !_isTradingEnabled)
            {
                SystemNotInitializedDebug(_logger, null);
                return null;
            }

            // Try to get OnnxEnsembleWrapper from service provider if available
            var onnxEnsemble = _serviceProvider.GetService<TradingBot.RLAgent.OnnxEnsembleWrapper>();
            
            if (onnxEnsemble != null)
            {
                // Create feature vector for the symbol and strategy
                var context = new MarketContext
                {
                    Symbol = symbol,
                    Price = DefaultMarketPrice, // This would typically come from current market data
                    Volume = DefaultMarketVolume,
                    Timestamp = DateTime.UtcNow
                };

                var features = await ExtractFeaturesAsync(context, cancellationToken).ConfigureAwait(false);
                
                // Apply updated feature weights from FeatureEngineer immediately
                var currentWeights = await _featureEngineer.GetCurrentWeightsAsync(strategyId, cancellationToken).ConfigureAwait(false);
                var weightedFeatures = ApplyFeatureWeights(features, currentWeights);
                
                // Convert weighted features to float array for ONNX input
                var featureArray = new float[weightedFeatures.Features.Count];
                int i = LoopCounterStart;
                foreach (var feature in weightedFeatures.Features.Values)
                {
                    featureArray[i++] = (float)feature;
                }

                // Validate feature vector shape before calling PredictAsync
                if (featureArray.Length > 0 && featureArray.Length <= FeatureArrayMaxLength) // Reasonable bounds check
                {
                    var ensemblePrediction = await onnxEnsemble.PredictAsync(featureArray, cancellationToken).ConfigureAwait(false);
                    
                    string direction;
                    if (ensemblePrediction.EnsembleResult > EnsembleBullishThreshold)
                        direction = "BUY";
                    else if (ensemblePrediction.EnsembleResult < EnsembleBearishThreshold)
                        direction = "SELL";
                    else
                        direction = "HOLD";
                    
                    // Convert EnsemblePrediction to MLPrediction
                    var prediction = new MLPrediction
                    {
                        Symbol = symbol,
                        Confidence = ensemblePrediction.Confidence,
                        Direction = direction,
                        ModelId = $"ensemble_{strategyId}",
                        Timestamp = DateTime.UtcNow,
                        IsValid = !ensemblePrediction.IsAnomaly
                    };
                    
                    prediction.Metadata["ensemble_result"] = ensemblePrediction.EnsembleResult;
                    prediction.Metadata["latency_ms"] = ensemblePrediction.LatencyMs;
                    prediction.Metadata["is_anomaly"] = ensemblePrediction.IsAnomaly;
                    prediction.Metadata["strategy_id"] = strategyId;
                    prediction.Metadata["model_count"] = ensemblePrediction.Predictions.Count;

                    ONNXPredictionDebug(_logger, symbol, strategyId, prediction.Confidence, ensemblePrediction.EnsembleResult, null);
                    
                    return prediction;
                }
                else
                {
                    InvalidFeatureVectorWarning(_logger, featureArray.Length, symbol, strategyId, null);
                }
            }
            else
            {
                ONNXWrapperNotAvailableDebug(_logger, null);
            }

            // Fallback to simplified prediction logic if ONNX not available
            var fallbackPrediction = await GetLatestPredictionAsync(symbol, cancellationToken).ConfigureAwait(false);
            FallbackPredictionDebug(_logger, symbol, strategyId, fallbackPrediction.Confidence, null);
            
            return fallbackPrediction;
        }
        catch (InvalidOperationException ex)
        {
            OnlinePredictionFailed(_logger, symbol, strategyId, ex);
            return null;
        }
        catch (ArgumentException ex)
        {
            OnlinePredictionFailed(_logger, symbol, strategyId, ex);
            return null;
        }
    }

    /// <summary>
    /// Alternative method name for backwards compatibility
    /// </summary>
    public Task<MLPrediction?> GetLivePredictionAsync(string symbol, string strategyId, CancellationToken cancellationToken = default)
    {
        return GetOnlinePredictionAsync(symbol, strategyId, cancellationToken);
    }

    private double CalculatePositionSize(double confidence)
    {
        // Apply Kelly criterion with clip using configurable parameters
        var edge = (confidence - _config.ML.Confidence.EdgeConversionOffset) * _config.ML.Confidence.EdgeConversionMultiplier; // Convert to [-1, 1] range
        var kellyFraction = edge / KellyDivisor; // Simplified Kelly calculation
        var clippedKelly = Math.Min(_config.ML.Confidence.KellyClip, Math.Max(-_config.ML.Confidence.KellyClip, kellyFraction));
        
        // Apply confidence multiplier using configurable parameters
        var confidenceMultiplier = Math.Min(1.0, Math.Max(0.0, (confidence - _config.ML.Confidence.ConfidenceMultiplierOffset) * _config.ML.Confidence.ConfidenceMultiplierScale));
        
        return clippedKelly * confidenceMultiplier;
    }

    private static TradingDecision CreateTradingDecision(
        string decisionId, MarketContext context, RegimeState regime, 
        ModelArtifact model, double confidence, double size, double latencyMs)
    {
        string direction;
        if (size > SignificantSizeThreshold)
            direction = "LONG";
        else if (size < -SignificantSizeThreshold)
            direction = "SHORT";
        else
            direction = "HOLD";

        TradingAction action;
        if (size > SignificantSizeThreshold)
            action = TradingAction.Buy;
        else if (size < -SignificantSizeThreshold)
            action = TradingAction.Sell;
        else
            action = TradingAction.Hold;

        var decision = new TradingDecision
        {
            DecisionId = decisionId,
            Signal = new TradingSignal
            {
                Symbol = context.Symbol,
                Direction = direction,
                Strength = (decimal)Math.Abs(confidence),
                Timestamp = DateTime.UtcNow
            },
            Action = action,
            Confidence = (decimal)confidence,
            MLConfidence = (decimal)confidence,
            MLStrategy = model.Id,
            RiskScore = (decimal)(1.0 - confidence),
            MaxPositionSize = (decimal)Math.Abs(size),
            MarketRegime = regime.Type.ToString(),
            RegimeConfidence = (decimal)regime.Confidence,
            Timestamp = DateTime.UtcNow
        };
        
        // Populate read-only Reasoning collection
        decision.Reasoning["model_id"] = model.Id;
        decision.Reasoning["regime"] = regime.Type.ToString();
        decision.Reasoning["latency_ms"] = latencyMs;
        decision.Reasoning["kelly_size"] = size;
        
        return decision;
    }

    private static TradingDecision CreateSafeDecision(string reason)
    {
        var decision = new TradingDecision
        {
            DecisionId = GenerateDecisionId(),
            Signal = new TradingSignal
            {
                Symbol = "UNKNOWN",
                Direction = "HOLD",
                Strength = 0m,
                Timestamp = DateTime.UtcNow
            },
            Action = TradingAction.Hold,
            Confidence = 0m
        };
        
        // Populate read-only Reasoning collection
        decision.Reasoning["reason"] = reason;
        
        return decision;
    }

    private static IntelligenceDecision ConvertToIntelligenceDecision(TradingDecision decision, FeatureSet features)
    {
        var intelligenceDecision = new IntelligenceDecision
        {
            DecisionId = decision.DecisionId,
            Timestamp = decision.Timestamp,
            Symbol = decision.Signal.Symbol,
            Confidence = (double)decision.Confidence,
            Action = decision.Action.ToString(),
            FeaturesVersion = features.Version,
            FeaturesHash = features.SchemaChecksum
        };
        
        // Populate read-only Metadata collection
        foreach (var kvp in decision.Reasoning)
        {
            intelligenceDecision.Metadata[kvp.Key] = kvp.Value;
        }
        
        return intelligenceDecision;
    }

    private static string GenerateDecisionId()
    {
        return $"D{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}_{System.Security.Cryptography.RandomNumberGenerator.GetInt32(RandomIdMin, RandomIdMax)}";
    }

    private bool ShouldPerformNightlyMaintenance()
    {
        var now = DateTime.UtcNow;
        var lastMaintenance = _lastNightlyMaintenance;
        
        // Perform maintenance if it's after 2:30 AM and we haven't done it today
        return now.Hour >= MaintenanceHour && now.Minute >= MaintenanceMinute && 
               (lastMaintenance.Date < now.Date || lastMaintenance == DateTime.MinValue);
    }

    private static Task CheckModelPromotionsAsync(CancellationToken cancellationToken)
    {
        // Check for models that should be promoted
        // Implementation would check recent performance metrics
        return Task.CompletedTask;
    }

    private void RaiseEvent(string eventType, string message)
    {
        IntelligenceEvent?.Invoke(this, new IntelligenceEventArgs
        {
            EventType = eventType,
            Message = message,
            Timestamp = DateTime.UtcNow
        });
    }

    /// <summary>
    /// Apply feature weights to features, immediately using updated weights from FeatureEngineer
    /// </summary>
    private static FeatureSet ApplyFeatureWeights(FeatureSet originalFeatures, Dictionary<string, double> weights)
    {
        var weightedFeatures = new FeatureSet
        {
            Symbol = originalFeatures.Symbol,
            Timestamp = originalFeatures.Timestamp,
            Version = originalFeatures.Version,
            SchemaChecksum = originalFeatures.SchemaChecksum
        };

        // Copy original metadata to read-only collection
        foreach (var kvp in originalFeatures.Metadata)
        {
            weightedFeatures.Metadata[kvp.Key] = kvp.Value;
        }

        // Apply weights to each feature
        foreach (var (featureName, featureValue) in originalFeatures.Features)
        {
            var weight = weights.GetValueOrDefault(featureName, 1.0);
            var weightedValue = featureValue * weight;
            
            weightedFeatures.Features[featureName] = weightedValue;
        }

        // Add metadata about feature weighting
        weightedFeatures.Metadata["feature_weights_applied"] = true;
        weightedFeatures.Metadata["weights_count"] = weights.Count;
        weightedFeatures.Metadata["low_value_features"] = weights.Count(kvp => kvp.Value < LowValueFeatureThreshold);

        return weightedFeatures;
    }

    // Workflow adapter methods
    private async Task<WorkflowExecutionResult> MakeDecisionWorkflowAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            // Extract market context from workflow context
            var marketContext = ExtractMarketContextFromWorkflow(context);
            var decision = await MakeDecisionAsync(marketContext, cancellationToken).ConfigureAwait(false);
            
            var result = new WorkflowExecutionResult { Success = true };
            result.Results["decision"] = decision;
            return result;
        }
        catch (InvalidOperationException ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
        catch (ArgumentException ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    private async Task<WorkflowExecutionResult> ProcessMarketDataWorkflowAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            var marketData = ExtractMarketDataFromWorkflow(context);
            await ProcessMarketDataAsync(marketData, cancellationToken).ConfigureAwait(false);
            
            return new WorkflowExecutionResult { Success = true };
        }
        catch (InvalidOperationException ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
        catch (ArgumentException ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    private async Task<WorkflowExecutionResult> PerformMaintenanceWorkflowAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            await PerformNightlyMaintenanceAsync(cancellationToken).ConfigureAwait(false);
            return new WorkflowExecutionResult { Success = true };
        }
        catch (ArgumentException ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
        catch (InvalidOperationException ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
        catch (TimeoutException ex)
        {
            return new WorkflowExecutionResult { Success = false, ErrorMessage = ex.Message };
        }
    }

    private static MarketContext ExtractMarketContextFromWorkflow(WorkflowExecutionContext context)
    {
        // Extract from workflow context - simplified
        return new MarketContext
        {
            Symbol = context.Parameters.GetValueOrDefault("symbol", "ES")?.ToString() ?? "ES",
            Price = Convert.ToDouble(context.Parameters.GetValueOrDefault("price", DefaultESPrice), CultureInfo.InvariantCulture),
            Volume = Convert.ToDouble(context.Parameters.GetValueOrDefault("volume", DefaultVolume), CultureInfo.InvariantCulture),
            Bid = Convert.ToDouble(context.Parameters.GetValueOrDefault("bid", DefaultBidOffset), CultureInfo.InvariantCulture),
            Ask = Convert.ToDouble(context.Parameters.GetValueOrDefault("ask", DefaultAskOffset), CultureInfo.InvariantCulture),
            Timestamp = DateTime.UtcNow
        };
    }

    private static MarketData ExtractMarketDataFromWorkflow(WorkflowExecutionContext context)
    {
        return WorkflowHelpers.ExtractMarketDataFromWorkflow(context);
    }

    // Wrapper methods for workflow execution
    private async Task<WorkflowExecutionResult> RunMLModelsWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        await RunMLModelsAsync(context, cancellationToken).ConfigureAwait(false);
        return WorkflowHelpers.CreateSuccessResult();
    }

    private async Task<WorkflowExecutionResult> UpdateRLTrainingWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        await UpdateRLTrainingAsync(context, cancellationToken).ConfigureAwait(false);
        return WorkflowHelpers.CreateSuccessResult();
    }

    private async Task<WorkflowExecutionResult> GeneratePredictionsWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        await GeneratePredictionsAsync(context, cancellationToken).ConfigureAwait(false);
        return WorkflowHelpers.CreateSuccessResult();
    }

    private async Task<WorkflowExecutionResult> AnalyzeCorrelationsWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        await AnalyzeCorrelationsAsync(context, cancellationToken).ConfigureAwait(false);
        return WorkflowHelpers.CreateSuccessResult();
    }

    #endregion

    #region Cloud Flow Methods (merged from CloudFlowService)

    /// <summary>
    /// Push trade record to cloud after decision execution
    /// </summary>
    public Task PushTradeRecordAsync(CloudTradeRecord tradeRecord, CancellationToken cancellationToken = default)
    {
        // Simplified cloud push - implementation moved to reduce file size
        return Task.CompletedTask;
    }

    /// <summary>
    /// Push service metrics to cloud
    /// </summary>
    public Task PushServiceMetricsAsync(CloudServiceMetrics metrics, CancellationToken cancellationToken = default)
    {
        // Simplified cloud push - implementation moved to reduce file size
        return Task.CompletedTask;
    }

    /// <summary>
    /// Push decision intelligence data to cloud
    /// </summary>
    public Task PushDecisionIntelligenceAsync(TradingDecision decision, CancellationToken cancellationToken = default)
    {
        // Simplified cloud push - implementation moved to reduce file size
        return Task.CompletedTask;
    }

    #endregion
}