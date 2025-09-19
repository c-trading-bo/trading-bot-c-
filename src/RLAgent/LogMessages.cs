using Microsoft.Extensions.Logging;

namespace TradingBot.RLAgent;

/// <summary>
/// High-performance logging message delegates for RLAgent
/// Addresses CA1848 violations by using LoggerMessage.Define for performance
/// </summary>
internal static class LogMessages
{
    // Algorithm Factory Messages
    private static readonly Action<ILogger, string, Exception?> _sacAlgorithmCreated =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(1001, nameof(SacAlgorithmCreated)),
            "[ALGO_FACTORY] Created SAC algorithm with config: {Config}");

    private static readonly Action<ILogger, string, Exception?> _metaLearningAlgorithmCreated =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(1002, nameof(MetaLearningAlgorithmCreated)),
            "[ALGO_FACTORY] Created Meta-Learning algorithm with config: {Config}");

    private static readonly Action<ILogger, string, Exception?> _cvarPpoAlgorithmCreated =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(1003, nameof(CVaRPPOAlgorithmCreated)),
            "[ALGO_FACTORY] Created CVaR-PPO algorithm with config: {Config}");

    // Meta Learner Messages
    private static readonly Action<ILogger, int, int, Exception?> _metaLearnerInitialized =
        LoggerMessage.Define<int, int>(LogLevel.Information, new EventId(2001, nameof(MetaLearnerInitialized)),
            "[META] Initialized meta-learner with state_dim={StateDim}, action_dim={ActionDim}");

    private static readonly Action<ILogger, string, int, Exception?> _taskAdaptationStarted =
        LoggerMessage.Define<string, int>(LogLevel.Debug, new EventId(2002, nameof(TaskAdaptationStarted)),
            "[META] Adapting to task: {TaskId} with {SupportSize} training instances");

    private static readonly Action<ILogger, double, int, int, Exception?> _metaTrainingCompleted =
        LoggerMessage.Define<double, int, int>(LogLevel.Information, new EventId(2003, nameof(MetaTrainingCompleted)),
            "[META] Meta-training completed: loss={Loss:F4}, tasks={TaskCount}, updates={Updates}");

    // CVaR-PPO Messages
    private static readonly Action<ILogger, int, int, double, Exception?> _cvarPpoInitialized =
        LoggerMessage.Define<int, int, double>(LogLevel.Information, new EventId(3001, nameof(CVaRPPOInitialized)),
            "[CVAR_PPO] Initialized with {StateSize} state size, {ActionSize} action size, CVaR alpha: {Alpha}");

    // SAC Messages
    private static readonly Action<ILogger, int, int, int, Exception?> _sacInitialized =
        LoggerMessage.Define<int, int, int>(LogLevel.Information, new EventId(4001, nameof(SACInitialized)),
            "[SAC] Initialized with state_dim={StateDim}, action_dim={ActionDim}, hidden_dim={HiddenDim}");

    // Feature Engineering Messages
    private static readonly Action<ILogger, Exception?> _featureEngineeringDisposed =
        LoggerMessage.Define(LogLevel.Information, new EventId(5001, nameof(FeatureEngineeringDisposed)),
            "[FEATURE_ENG] Disposed successfully");

    // Model Hot Reload Messages
    private static readonly Action<ILogger, string, Exception?> _modelHotReloadInitialized =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(6001, nameof(ModelHotReloadInitialized)),
            "[HOT_RELOAD] Model hot-reload manager initialized, watching: {WatchDirectory}");

    private static readonly Action<ILogger, string, Exception?> _modelFileChangeDetected =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(6002, nameof(ModelFileChangeDetected)),
            "[HOT_RELOAD] Model file change detected: {ModelPath}");

    private static readonly Action<ILogger, string, string, Exception?> _hotReloadStarted =
        LoggerMessage.Define<string, string>(LogLevel.Information, new EventId(6003, nameof(HotReloadStarted)),
            "[HOT_RELOAD] Starting hot-reload for: {ModelPath} as {CandidateName}");

    private static readonly Action<ILogger, string, Exception?> _oldModelUnloaded =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(6004, nameof(OldModelUnloaded)),
            "[HOT_RELOAD] Unloaded old model: {OldModelName}");

    private static readonly Action<ILogger, string, Exception?> _hotReloadCompleted =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(6005, nameof(HotReloadCompleted)),
            "[HOT_RELOAD] Hot-reload completed successfully: {CandidateName} is now live");

    private static readonly Action<ILogger, string, Exception?> _smokeTestsPassed =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(6006, nameof(SmokeTestsPassed)),
            "[HOT_RELOAD] Smoke tests passed for: {ModelName}");

    private static readonly Action<ILogger, Exception?> _modelHotReloadDisposed =
        LoggerMessage.Define(LogLevel.Information, new EventId(6007, nameof(ModelHotReloadDisposed)),
            "[HOT_RELOAD] Model hot-reload manager disposed");

    // CVaR-PPO Training Messages
    private static readonly Action<ILogger, int, int, Exception?> _cvarPpoTrainingStarted =
        LoggerMessage.Define<int, int>(LogLevel.Information, new EventId(3002, nameof(CVaRPPOTrainingStarted)),
            "[CVAR_PPO] Starting training episode {Episode} with {ExperienceCount} experiences");

    private static readonly Action<ILogger, int, double, double, double, double, double, Exception?> _cvarPpoTrainingCompleted =
        LoggerMessage.Define<int, double, double, double, double, double>(LogLevel.Information, new EventId(3003, nameof(CVaRPPOTrainingCompleted)),
            "[CVAR_PPO] Training completed - Episode: {Episode}, Total Loss: {Loss:F4}, Policy Loss: {PolicyLoss:F4}, Value Loss: {ValueLoss:F4}, CVaR Loss: {CVaRLoss:F4}, Avg Reward: {Reward:F4}");

    private static readonly Action<ILogger, string, string, Exception?> _cvarPpoModelSaved =
        LoggerMessage.Define<string, string>(LogLevel.Information, new EventId(3004, nameof(CVaRPPOModelSaved)),
            "[CVAR_PPO] Model saved: {ModelPath} (version: {Version})");

    private static readonly Action<ILogger, string, int, double, Exception?> _cvarPpoModelMetadataLoaded =
        LoggerMessage.Define<string, int, double>(LogLevel.Information, new EventId(3005, nameof(CVaRPPOModelMetadataLoaded)),
            "[CVAR_PPO] Loaded model metadata - Version: {Version}, Episode: {Episode}, Avg Reward: {Reward:F3}");

    private static readonly Action<ILogger, string, Exception?> _cvarPpoModelLoaded =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(3006, nameof(CVaRPPOModelLoaded)),
            "[CVAR_PPO] Model loaded successfully: {ModelPath}");

    private static readonly Action<ILogger, Exception?> _cvarPpoNetworksInitialized =
        LoggerMessage.Define(LogLevel.Information, new EventId(3007, nameof(CVaRPPONetworksInitialized)),
            "[CVAR_PPO] Neural networks initialized");

    private static readonly Action<ILogger, Exception?> _cvarPpoCheckpointSaved =
        LoggerMessage.Define(LogLevel.Information, new EventId(3008, nameof(CVaRPPOCheckpointSaved)),
            "[CVAR_PPO] Checkpoint saved due to performance improvement");

    private static readonly Action<ILogger, Exception?> _cvarPpoDisposed =
        LoggerMessage.Define(LogLevel.Information, new EventId(3009, nameof(CVaRPPODisposed)),
            "[CVAR_PPO] Disposed successfully");

    // ONNX Ensemble Messages
    private static readonly Action<ILogger, int, int, Exception?> _onnxEnsembleInitialized =
        LoggerMessage.Define<int, int>(LogLevel.Information, new EventId(8001, nameof(OnnxEnsembleInitialized)),
            "[RL-ENSEMBLE] ONNX Ensemble Wrapper initialized with {MaxBatchSize} batch size, {MaxConcurrentBatches} concurrent batches");

    private static readonly Action<ILogger, string, Exception?> _modelFileNotFound =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(8002, nameof(ModelFileNotFound)),
            "[RL-ENSEMBLE] Model file not found: {ModelPath}");

    private static readonly Action<ILogger, string, double, Exception?> _modelLoaded =
        LoggerMessage.Define<string, double>(LogLevel.Information, new EventId(8003, nameof(ModelLoaded)),
            "[RL-ENSEMBLE] Model loaded: {ModelName} with confidence {Confidence:F2}");

    private static readonly Action<ILogger, string, Exception?> _modelLoadFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(8004, nameof(ModelLoadFailed)),
            "[RL-ENSEMBLE] Failed to load model: {ModelName}");

    private static readonly Action<ILogger, string, Exception?> _modelUnloaded =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(8005, nameof(ModelUnloaded)),
            "[RL-ENSEMBLE] Model unloaded: {ModelName}");

    private static readonly Action<ILogger, string, Exception?> _modelUnloadFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(8006, nameof(ModelUnloadFailed)),
            "[RL-ENSEMBLE] Failed to unload model: {ModelName}");

    private static readonly Action<ILogger, Exception?> _anomalousInputBlocked =
        LoggerMessage.Define(LogLevel.Warning, new EventId(8007, nameof(AnomalousInputBlocked)),
            "[RL-ENSEMBLE] Anomalous input detected and blocked");

    private static readonly Action<ILogger, Exception?> _batchProcessingError =
        LoggerMessage.Define(LogLevel.Error, new EventId(8008, nameof(BatchProcessingError)),
            "[RL-ENSEMBLE] Error in batch processing task");

    private static readonly Action<ILogger, int, double, Exception?> _batchProcessed =
        LoggerMessage.Define<int, double>(LogLevel.Debug, new EventId(8009, nameof(BatchProcessed)),
            "[RL-ENSEMBLE] Processed batch of {BatchSize} in {LatencyMs:F2}ms");

    private static readonly Action<ILogger, Exception?> _inferenceBatchError =
        LoggerMessage.Define(LogLevel.Error, new EventId(8010, nameof(InferenceBatchError)),
            "[RL-ENSEMBLE] Error processing inference batch");

    private static readonly Action<ILogger, string, Exception?> _modelInferenceError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(8011, nameof(ModelInferenceError)),
            "[RL-ENSEMBLE] Error in model inference: {ModelName}");

    private static readonly Action<ILogger, Exception?> _gpuAccelerationEnabled =
        LoggerMessage.Define(LogLevel.Information, new EventId(8012, nameof(GpuAccelerationEnabled)),
            "[RL-ENSEMBLE] GPU acceleration enabled for ONNX inference");

    private static readonly Action<ILogger, Exception?> _gpuAccelerationFailed =
        LoggerMessage.Define(LogLevel.Warning, new EventId(8013, nameof(GpuAccelerationFailed)),
            "[RL-ENSEMBLE] Failed to enable GPU acceleration, falling back to CPU");

    private static readonly Action<ILogger, string, string, string, Exception?> _modelValidationPassed =
        LoggerMessage.Define<string, string, string>(LogLevel.Debug, new EventId(8014, nameof(ModelValidationPassed)),
            "[RL-ENSEMBLE] Model validation passed: {ModelName} - Input: {InputShape}, Output: {OutputShape}");

    private static readonly Action<ILogger, string, Exception?> _modelValidationFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(8015, nameof(ModelValidationFailed)),
            "[RL-ENSEMBLE] Model validation failed: {ModelName}");

    private static readonly Action<ILogger, Exception?> _batchProcessingTimeout =
        LoggerMessage.Define(LogLevel.Warning, new EventId(8016, nameof(BatchProcessingTimeout)),
            "[RL-ENSEMBLE] Timeout waiting for batch processing task to complete");

    private static readonly Action<ILogger, Exception?> _onnxEnsembleDisposed =
        LoggerMessage.Define(LogLevel.Information, new EventId(8017, nameof(OnnxEnsembleDisposed)),
            "[RL-ENSEMBLE] ONNX Ensemble Wrapper disposed");

    // Feature Engineering Messages
    private static readonly Action<ILogger, int, Exception?> _featureEngineeringInitialized =
        LoggerMessage.Define<int>(LogLevel.Information, new EventId(9001, nameof(FeatureEngineeringInitialized)),
            "[FEATURE_ENG] Initialized with {ProfileCount} regime profiles and streaming aggregation");

    private static readonly Action<ILogger, int, string, string, string, Exception?> _featuresGenerated =
        LoggerMessage.Define<int, string, string, string>(LogLevel.Debug, new EventId(9002, nameof(FeaturesGenerated)),
            "[FEATURE_ENG] Generated {FeatureCount} features for {Symbol} {Strategy} {Regime}");

    private static readonly Action<ILogger, string, string, string, Exception?> _featureGenerationError =
        LoggerMessage.Define<string, string, string>(LogLevel.Error, new EventId(9003, nameof(FeatureGenerationError)),
            "[FEATURE_ENG] Error generating features for {Symbol} {Strategy} {Regime}");

    private static readonly Action<ILogger, int, string, Exception?> _featureImportanceUpdated =
        LoggerMessage.Define<int, string>(LogLevel.Debug, new EventId(9004, nameof(FeatureImportanceUpdated)),
            "[FEATURE_ENG] Updated importance for {FeatureCount} features: {FeatureKey}");

    private static readonly Action<ILogger, string, string, string, Exception?> _featureImportanceError =
        LoggerMessage.Define<string, string, string>(LogLevel.Warning, new EventId(9005, nameof(FeatureImportanceError)),
            "[FEATURE_ENG] Error updating feature importance for {Symbol} {Strategy} {Regime}");

    private static readonly Action<ILogger, string, string, Exception?> _streamingTickError =
        LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(9006, nameof(StreamingTickError)),
            "[FEATURE_ENG] Failed to process streaming tick for {Symbol}: {ErrorMessage}");

    private static readonly Action<ILogger, string, Exception?> _staleAggregatorCleaned =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(9007, nameof(StaleAggregatorCleaned)),
            "[FEATURE_ENG] Cleaned up stale streaming aggregator for {Symbol}");

    private static readonly Action<ILogger, int, Exception?> _staleAggregatorsCleanup =
        LoggerMessage.Define<int>(LogLevel.Information, new EventId(9008, nameof(StaleAggregatorsCleanup)),
            "[FEATURE_ENG] Cleaned up {Count} stale streaming symbol aggregators");

    private static readonly Action<ILogger, Exception?> _streamingCleanupError =
        LoggerMessage.Define(LogLevel.Error, new EventId(9009, nameof(StreamingCleanupError)),
            "[FEATURE_ENG] Error during streaming cleanup");

    private static readonly Action<ILogger, string, Exception?> _featureForwardFilled =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(9010, nameof(FeatureForwardFilled)),
            "[FEATURE_ENG] Forward-filled NaN for feature: {FeatureName}");

    // Position Sizing Messages
    private static readonly Action<ILogger, double, Exception?> _positionSizingInitialized =
        LoggerMessage.Define<double>(LogLevel.Information, new EventId(10001, nameof(PositionSizingInitialized)),
            "[POSITION_SIZING] Initialized with max allocation: {MaxAllocation}, regime-based clipping enabled");

    private static readonly Action<ILogger, string, string, string, string, Exception?> _positionSizingCalculated =
        LoggerMessage.Define<string, string, string, string>(LogLevel.Debug, new EventId(10002, nameof(PositionSizingCalculated)),
            "[POSITION_SIZING] {Symbol} {Strategy} {Regime}: {Values}");

    private static readonly Action<ILogger, string, string, Exception?> _positionSizingArgumentError =
        LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(10003, nameof(PositionSizingArgumentError)),
            "[POSITION_SIZING] Invalid arguments for position sizing {Symbol} {Strategy}");

    private static readonly Action<ILogger, string, string, Exception?> _positionSizingOperationError =
        LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(10004, nameof(PositionSizingOperationError)),
            "[POSITION_SIZING] Invalid operation for position sizing {Symbol} {Strategy}");

    private static readonly Action<ILogger, string, string, Exception?> _positionSizingDivisionError =
        LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(10005, nameof(PositionSizingDivisionError)),
            "[POSITION_SIZING] Division by zero in position sizing {Symbol} {Strategy}");

    private static readonly Action<ILogger, string, Exception?> _riskRejected =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(10006, nameof(RiskRejected)),
            "[POSITION_SIZING] Risk ≤ 0 detected, rejecting position: {Symbol}");

    private static readonly Action<ILogger, string, int, int, int, Exception?> _stepChangeLimited =
        LoggerMessage.Define<string, int, int, int>(LogLevel.Information, new EventId(10007, nameof(StepChangeLimited)),
            "[POSITION_SIZING] Step change limited: {Symbol} {CurrentContracts} → {NewContracts} limited to {LimitedContracts}");

    private static readonly Action<ILogger, double, double, double, Exception?> _riskCalculated =
        LoggerMessage.Define<double, double, double>(LogLevel.Debug, new EventId(10008, nameof(RiskCalculated)),
            "[POSITION_SIZING] Risk calculation: Price={Price:F2}, Stop={Stop:F2}, Risk={Risk:F2}");

    // Additional Feature Engineering Messages
    private static readonly Action<ILogger, string, Exception?> _featureAppliedSentinel =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(9011, nameof(FeatureAppliedSentinel)),
            "[FEATURE_ENG] Applied sentinel value for feature: {FeatureName}");

    private static readonly Action<ILogger, Exception?> _featureEngineeringDisposed2 =
        LoggerMessage.Define(LogLevel.Information, new EventId(9012, nameof(FeatureEngineeringDisposed2)),
            "[FEATURE_ENG] Disposed successfully");

    // Public methods for high-performance logging
    public static void SacAlgorithmCreated(ILogger logger, string config) => _sacAlgorithmCreated(logger, config, null);
    public static void MetaLearningAlgorithmCreated(ILogger logger, string config) => _metaLearningAlgorithmCreated(logger, config, null);
    public static void CVaRPPOAlgorithmCreated(ILogger logger, string config) => _cvarPpoAlgorithmCreated(logger, config, null);
    public static void MetaLearnerInitialized(ILogger logger, int stateDim, int actionDim) => _metaLearnerInitialized(logger, stateDim, actionDim, null);
    public static void TaskAdaptationStarted(ILogger logger, string taskId, int supportSize) => _taskAdaptationStarted(logger, taskId, supportSize, null);
    public static void MetaTrainingCompleted(ILogger logger, double loss, int taskCount, int updates) => _metaTrainingCompleted(logger, loss, taskCount, updates, null);
    public static void CVaRPPOInitialized(ILogger logger, int stateSize, int actionSize, double cvarAlpha) => _cvarPpoInitialized(logger, stateSize, actionSize, cvarAlpha, null);
    public static void SACInitialized(ILogger logger, int stateDim, int actionDim, int hiddenDim) => _sacInitialized(logger, stateDim, actionDim, hiddenDim, null);
    public static void FeatureEngineeringDisposed(ILogger logger) => _featureEngineeringDisposed(logger, null);
    public static void ModelHotReloadInitialized(ILogger logger, string watchDirectory) => _modelHotReloadInitialized(logger, watchDirectory, null);
    public static void ModelFileChangeDetected(ILogger logger, string modelPath) => _modelFileChangeDetected(logger, modelPath, null);
    public static void HotReloadStarted(ILogger logger, string modelPath, string candidateName) => _hotReloadStarted(logger, modelPath, candidateName, null);
    public static void OldModelUnloaded(ILogger logger, string oldModelName) => _oldModelUnloaded(logger, oldModelName, null);
    public static void HotReloadCompleted(ILogger logger, string candidateName) => _hotReloadCompleted(logger, candidateName, null);
    public static void SmokeTestsPassed(ILogger logger, string modelName) => _smokeTestsPassed(logger, modelName, null);
    public static void ModelHotReloadDisposed(ILogger logger) => _modelHotReloadDisposed(logger, null);
    public static void CVaRPPOTrainingStarted(ILogger logger, int episode, int experienceCount) => _cvarPpoTrainingStarted(logger, episode, experienceCount, null);
    public static void CVaRPPOTrainingCompleted(ILogger logger, int episode, double loss, double policyLoss, double valueLoss, double cvarLoss, double reward) => _cvarPpoTrainingCompleted(logger, episode, loss, policyLoss, valueLoss, cvarLoss, reward, null);
    public static void CVaRPPOModelSaved(ILogger logger, string modelPath, string version) => _cvarPpoModelSaved(logger, modelPath, version, null);
    public static void CVaRPPOModelMetadataLoaded(ILogger logger, string version, int episode, double reward) => _cvarPpoModelMetadataLoaded(logger, version, episode, reward, null);
    public static void CVaRPPOModelLoaded(ILogger logger, string modelPath) => _cvarPpoModelLoaded(logger, modelPath, null);
    public static void CVaRPPONetworksInitialized(ILogger logger) => _cvarPpoNetworksInitialized(logger, null);
    public static void CVaRPPOCheckpointSaved(ILogger logger) => _cvarPpoCheckpointSaved(logger, null);
    public static void CVaRPPODisposed(ILogger logger) => _cvarPpoDisposed(logger, null);
    // Position Sizing Messages
    private static readonly Action<ILogger, string, double, Exception?> _positionSizeCalculated =
        LoggerMessage.Define<string, double>(LogLevel.Information, new EventId(7001, nameof(PositionSizeCalculated)),
            "[POSITION_SIZING] Calculated position size for {Symbol}: {Size}");

    public static void PositionSizeCalculated(ILogger logger, string symbol, double size) => _positionSizeCalculated(logger, symbol, size, null);
    public static void OnnxEnsembleInitialized(ILogger logger, int maxBatchSize, int maxConcurrentBatches) => _onnxEnsembleInitialized(logger, maxBatchSize, maxConcurrentBatches, null);
    public static void ModelFileNotFound(ILogger logger, string modelPath) => _modelFileNotFound(logger, modelPath, null);
    public static void ModelLoaded(ILogger logger, string modelName, double confidence) => _modelLoaded(logger, modelName, confidence, null);
    public static void ModelLoadFailed(ILogger logger, string modelName, Exception ex) => _modelLoadFailed(logger, modelName, ex);
    public static void ModelUnloaded(ILogger logger, string modelName) => _modelUnloaded(logger, modelName, null);
    public static void ModelUnloadFailed(ILogger logger, string modelName, Exception ex) => _modelUnloadFailed(logger, modelName, ex);
    public static void AnomalousInputBlocked(ILogger logger) => _anomalousInputBlocked(logger, null);
    public static void BatchProcessingError(ILogger logger, Exception ex) => _batchProcessingError(logger, ex);
    public static void BatchProcessed(ILogger logger, int batchSize, double latencyMs) => _batchProcessed(logger, batchSize, latencyMs, null);
    public static void InferenceBatchError(ILogger logger, Exception ex) => _inferenceBatchError(logger, ex);
    public static void ModelInferenceError(ILogger logger, string modelName, Exception ex) => _modelInferenceError(logger, modelName, ex);
    public static void GpuAccelerationEnabled(ILogger logger) => _gpuAccelerationEnabled(logger, null);
    public static void GpuAccelerationFailed(ILogger logger, Exception ex) => _gpuAccelerationFailed(logger, ex);
    public static void ModelValidationPassed(ILogger logger, string modelName, string inputShape, string outputShape) => _modelValidationPassed(logger, modelName, inputShape, outputShape, null);
    public static void ModelValidationFailed(ILogger logger, string modelName, Exception ex) => _modelValidationFailed(logger, modelName, ex);
    public static void BatchProcessingTimeout(ILogger logger, Exception ex) => _batchProcessingTimeout(logger, ex);
    public static void OnnxEnsembleDisposed(ILogger logger) => _onnxEnsembleDisposed(logger, null);
    public static void FeatureEngineeringInitialized(ILogger logger, int profileCount) => _featureEngineeringInitialized(logger, profileCount, null);
    public static void FeaturesGenerated(ILogger logger, int featureCount, string symbol, string strategy, string regime) => _featuresGenerated(logger, featureCount, symbol, strategy, regime, null);
    public static void FeatureGenerationError(ILogger logger, string symbol, string strategy, string regime, Exception ex) => _featureGenerationError(logger, symbol, strategy, regime, ex);
    public static void FeatureImportanceUpdated(ILogger logger, int featureCount, string featureKey) => _featureImportanceUpdated(logger, featureCount, featureKey, null);
    public static void FeatureImportanceError(ILogger logger, string symbol, string strategy, string regime, Exception ex) => _featureImportanceError(logger, symbol, strategy, regime, ex);
    public static void StreamingTickError(ILogger logger, string symbol, string errorMessage, Exception ex) => _streamingTickError(logger, symbol, errorMessage, ex);
    public static void StaleAggregatorCleaned(ILogger logger, string symbol) => _staleAggregatorCleaned(logger, symbol, null);
    public static void StaleAggregatorsCleanup(ILogger logger, int count) => _staleAggregatorsCleanup(logger, count, null);
    public static void StreamingCleanupError(ILogger logger, Exception ex) => _streamingCleanupError(logger, ex);
    public static void FeatureForwardFilled(ILogger logger, string featureName) => _featureForwardFilled(logger, featureName, null);
    public static void PositionSizingInitialized(ILogger logger, double maxAllocation) => _positionSizingInitialized(logger, maxAllocation, null);
    public static void PositionSizingCalculated(ILogger logger, string symbol, string strategy, string regime, double kelly, double sac, double clip, int final) => 
        _positionSizingCalculated(logger, symbol, strategy, regime, $"Kelly={kelly:F3}, SAC={sac:F3}, Clip={clip:F3}, Final={final} contracts", null);
    public static void PositionSizingArgumentError(ILogger logger, string symbol, string strategy, Exception ex) => _positionSizingArgumentError(logger, symbol, strategy, ex);
    public static void PositionSizingOperationError(ILogger logger, string symbol, string strategy, Exception ex) => _positionSizingOperationError(logger, symbol, strategy, ex);
    public static void PositionSizingDivisionError(ILogger logger, string symbol, string strategy, Exception ex) => _positionSizingDivisionError(logger, symbol, strategy, ex);
    public static void RiskRejected(ILogger logger, string symbol) => _riskRejected(logger, symbol, null);
    public static void StepChangeLimited(ILogger logger, string symbol, int currentContracts, int newContracts, int limitedContracts) => _stepChangeLimited(logger, symbol, currentContracts, newContracts, limitedContracts, null);
    public static void RiskCalculated(ILogger logger, double price, double stop, double risk) => _riskCalculated(logger, price, stop, risk, null);
    public static void FeatureAppliedSentinel(ILogger logger, string featureName) => _featureAppliedSentinel(logger, featureName, null);
    public static void FeatureEngineeringDisposed2(ILogger logger) => _featureEngineeringDisposed2(logger, null);

    // Additional CVaR-PPO specific messages
    private static readonly Action<ILogger, Exception?> _cvarPpoTrainingArgumentError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3010, nameof(CVaRPPOTrainingArgumentError)),
            "[CVAR_PPO] Invalid arguments during training");

    private static readonly Action<ILogger, Exception?> _cvarPpoActionSelectionArgumentError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3011, nameof(CVaRPPOActionSelectionArgumentError)),
            "[CVAR_PPO] Invalid arguments for action selection");

    private static readonly Action<ILogger, Exception?> _cvarPpoActionSelectionOperationError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3012, nameof(CVaRPPOActionSelectionOperationError)),
            "[CVAR_PPO] Invalid operation during action selection");

    private static readonly Action<ILogger, Exception?> _cvarPpoActionSelectionMemoryError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3013, nameof(CVaRPPOActionSelectionMemoryError)),
            "[CVAR_PPO] Out of memory during action selection");

    private static readonly Action<ILogger, double, double, double, double, Exception?> _cvarPpoActionSelected =
        LoggerMessage.Define<double, double, double, double>(LogLevel.Debug, new EventId(3014, nameof(CVaRPPOActionSelected)),
            "[CVAR_PPO] Action selected: {Action} (prob: {Prob:F3}, value: {Value:F3}, cvar: {CVaR:F3})");

    private static readonly Action<ILogger, Exception?> _cvarPpoUpdateParametersError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3015, nameof(CVaRPPOUpdateParametersError)),
            "[CVAR_PPO] Error updating network parameters");

    private static readonly Action<ILogger, Exception?> _cvarPpoUpdateParametersArgumentError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3016, nameof(CVaRPPOUpdateParametersArgumentError)),
            "[CVAR_PPO] Invalid arguments for parameter update");

    private static readonly Action<ILogger, Exception?> _cvarPpoUpdateParametersOperationError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3017, nameof(CVaRPPOUpdateParametersOperationError)),
            "[CVAR_PPO] Invalid operation during parameter update");

    private static readonly Action<ILogger, double, Exception?> _cvarPpoGradientClipped =
        LoggerMessage.Define<double>(LogLevel.Debug, new EventId(3018, nameof(CVaRPPOGradientClipped)),
            "[CVAR_PPO] Gradient clipped: norm={Norm:F3}");

    private static readonly Action<ILogger, double, double, bool, Exception?> _cvarPpoExperienceAdded =
        LoggerMessage.Define<double, double, bool>(LogLevel.Debug, new EventId(3019, nameof(CVaRPPOExperienceAdded)),
            "[CVAR_PPO] Added experience - Action: {Action}, Reward: {Reward:F3}, Done: {Done}");

    private static readonly Action<ILogger, Exception?> _cvarPpoModelSaveAccessDenied =
        LoggerMessage.Define(LogLevel.Error, new EventId(3020, nameof(CVaRPPOModelSaveAccessDenied)),
            "[CVAR_PPO] Access denied when saving model");

    private static readonly Action<ILogger, Exception?> _cvarPpoModelSaveDirectoryNotFound =
        LoggerMessage.Define(LogLevel.Error, new EventId(3021, nameof(CVaRPPOModelSaveDirectoryNotFound)),
            "[CVAR_PPO] Directory not found when saving model");

    private static readonly Action<ILogger, Exception?> _cvarPpoModelSaveIOError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3022, nameof(CVaRPPOModelSaveIOError)),
            "[CVAR_PPO] IO error when saving model");

    private static readonly Action<ILogger, string, Exception?> _cvarPpoModelPathNotExists =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3023, nameof(CVaRPPOModelPathNotExists)),
            "[CVAR_PPO] Model path does not exist: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _cvarPpoMissingNetworkFiles =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3024, nameof(CVaRPPOMissingNetworkFiles)),
            "[CVAR_PPO] Missing network files in: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _cvarPpoModelFileNotFound =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3025, nameof(CVaRPPOModelFileNotFound)),
            "[CVAR_PPO] Model file not found: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _cvarPpoModelLoadAccessDenied =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3026, nameof(CVaRPPOModelLoadAccessDenied)),
            "[CVAR_PPO] Access denied loading model: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _cvarPpoInvalidModelFile =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3027, nameof(CVaRPPOInvalidModelFile)),
            "[CVAR_PPO] Invalid model file: {ModelPath}");

    // SAC specific messages  
    private static readonly Action<ILogger, Exception?> _sacActionSelectionArgumentError =
        LoggerMessage.Define(LogLevel.Error, new EventId(4002, nameof(SACActionSelectionArgumentError)),
            "[SAC] Invalid arguments for action selection");

    private static readonly Action<ILogger, Exception?> _sacActionSelectionOperationError =
        LoggerMessage.Define(LogLevel.Error, new EventId(4003, nameof(SACActionSelectionOperationError)),
            "[SAC] Invalid operation during action selection");

    private static readonly Action<ILogger, Exception?> _sacActionSelectionMemoryError =
        LoggerMessage.Define(LogLevel.Error, new EventId(4004, nameof(SACActionSelectionMemoryError)),
            "[SAC] Out of memory during action selection");

    private static readonly Action<ILogger, Exception?> _sacTrainingDebug =
        LoggerMessage.Define(LogLevel.Debug, new EventId(4005, nameof(SACTrainingDebug)),
            "[SAC] Starting training iteration");

    private static readonly Action<ILogger, Exception?> _sacTrainingArgumentError =
        LoggerMessage.Define(LogLevel.Error, new EventId(4006, nameof(SACTrainingArgumentError)),
            "[SAC] Invalid arguments for training");

    private static readonly Action<ILogger, Exception?> _sacTrainingOperationError =
        LoggerMessage.Define(LogLevel.Error, new EventId(4007, nameof(SACTrainingOperationError)),
            "[SAC] Invalid operation during training");

    private static readonly Action<ILogger, Exception?> _sacTrainingMemoryError =
        LoggerMessage.Define(LogLevel.Error, new EventId(4008, nameof(SACTrainingMemoryError)),
            "[SAC] Out of memory during training");

    private static readonly Action<ILogger, Exception?> _sacTrainingCancelled =
        LoggerMessage.Define(LogLevel.Information, new EventId(4009, nameof(SACTrainingCancelled)),
            "[SAC] Training was cancelled");

    private static readonly Action<ILogger, double, double, double, double, double, Exception?> _sacTrainingCompleted =
        LoggerMessage.Define<double, double, double, double, double>(LogLevel.Debug, new EventId(4010, nameof(SACTrainingCompleted)),
            "[SAC] Training completed: Actor={ActorLoss:F4}, Critic1={CriticLoss1:F4}, Critic2={CriticLoss2:F4}, Value={ValueLoss:F4}, Entropy={Entropy:F4}");

    private static readonly Action<ILogger, Exception?> _sacTrainingOperationFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(4011, nameof(SACTrainingOperationFailed)),
            "[SAC] Training failed due to invalid operation");

    private static readonly Action<ILogger, Exception?> _sacTrainingMemoryFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(4012, nameof(SACTrainingMemoryFailed)),
            "[SAC] Training failed due to memory exhaustion");

    // MetaLearner specific messages
    private static readonly Action<ILogger, int, int, double, Exception?> _metaAdaptationStep =
        LoggerMessage.Define<int, int, double>(LogLevel.Debug, new EventId(2004, nameof(MetaAdaptationStep)),
            "[META] Adaptation step {Step}/{TotalSteps}, loss: {Loss:F4}");

    private static readonly Action<ILogger, string, double, Exception?> _metaExperienceStored =
        LoggerMessage.Define<string, double>(LogLevel.Debug, new EventId(2005, nameof(MetaExperienceStored)),
            "[META] Stored experience for task: {TaskId}, reward: {Reward:F3}");

    private static readonly Action<ILogger, string, Exception?> _metaTaskAdaptationArgumentError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(2006, nameof(MetaTaskAdaptationArgumentError)),
            "[META] Invalid arguments for task adaptation: {TaskId}");

    private static readonly Action<ILogger, string, Exception?> _metaTaskAdaptationOperationError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(2007, nameof(MetaTaskAdaptationOperationError)),
            "[META] Invalid operation during task adaptation: {TaskId}");

    private static readonly Action<ILogger, string, Exception?> _metaTaskAdaptationMemoryError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(2008, nameof(MetaTaskAdaptationMemoryError)),
            "[META] Out of memory during task adaptation: {TaskId}");

    private static readonly Action<ILogger, double, int, int, Exception?> _metaTrainingInfo =
        LoggerMessage.Define<double, int, int>(LogLevel.Information, new EventId(2009, nameof(MetaTrainingInfo)),
            "[META] Meta-training progress: loss={Loss:F4}, updates={Updates}, episodes={Episodes}");

    private static readonly Action<ILogger, string, Exception?> _metaTrainingArgumentError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(2010, nameof(MetaTrainingArgumentError)),
            "[META] Invalid arguments for meta-training: {Message}");

    private static readonly Action<ILogger, string, Exception?> _metaTrainingOperationError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(2011, nameof(MetaTrainingOperationError)),
            "[META] Invalid operation during meta-training: {Message}");

    private static readonly Action<ILogger, string, Exception?> _metaTrainingMemoryError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(2012, nameof(MetaTrainingMemoryError)),
            "[META] Out of memory during meta-training: {Message}");

    // ONNX specific messages
    private static readonly Action<ILogger, Exception?> _onnxGpuAccelerationEnabled =
        LoggerMessage.Define(LogLevel.Information, new EventId(8018, nameof(OnnxGpuAccelerationEnabled)),
            "[RL-ENSEMBLE] GPU acceleration enabled for ONNX inference");

    private static readonly Action<ILogger, string, string, string, Exception?> _onnxModelValidationDebug =
        LoggerMessage.Define<string, string, string>(LogLevel.Debug, new EventId(8019, nameof(OnnxModelValidationDebug)),
            "[RL-ENSEMBLE] Model validation passed: {ModelName} - Input: {InputShape}, Output: {OutputShape}");

    private static readonly Action<ILogger, Exception?> _onnxDisposedDebug =
        LoggerMessage.Define(LogLevel.Information, new EventId(8020, nameof(OnnxDisposedDebug)),
            "[RL-ENSEMBLE] ONNX Ensemble Wrapper disposed");

    // Feature Engineering Debug Messages  
    private static readonly Action<ILogger, string, Exception?> _featureEngineeringDebug =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(9013, nameof(FeatureEngineeringDebug)),
            "[FEATURE_ENG] {Message}");

    private static readonly Action<ILogger, string, double, double, Exception?> _featureStreamingTickProcessed =
        LoggerMessage.Define<string, double, double>(LogLevel.Trace, new EventId(9014, nameof(FeatureStreamingTickProcessed)),
            "[FEATURE_ENG] Processed streaming tick for {Symbol}: Price={Price}, Volume={Volume}");

    // Model Hot Reload Messages
    private static readonly Action<ILogger, string, Exception?> _modelHotReloadError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6008, nameof(ModelHotReloadError)),
            "[HOT_RELOAD] {Message}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadWarning =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(6009, nameof(ModelHotReloadWarning)),
            "[HOT_RELOAD] {Message}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadInfo =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(6010, nameof(ModelHotReloadInfo)),
            "[HOT_RELOAD] {Message}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadDebug =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(6011, nameof(ModelHotReloadDebug)),
            "[HOT_RELOAD] {Message}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadWarningParam =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(6012, nameof(ModelHotReloadWarningParam)),
            "[HOT_RELOAD] Hot-reload already in progress, skipping: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadFailedLoad =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6013, nameof(ModelHotReloadFailedLoad)),
            "[HOT_RELOAD] Failed to load candidate model: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadSmokeTestFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6014, nameof(ModelHotReloadSmokeTestFailed)),
            "[HOT_RELOAD] Smoke tests failed for candidate model: {CandidateName}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadCompleted =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(6015, nameof(ModelHotReloadCompleted)),
            "[HOT_RELOAD] Hot-reload completed successfully: {CandidateName} is now live");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadFileNotFound =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6016, nameof(ModelHotReloadFileNotFound)),
            "[HOT_RELOAD] Model file not found during hot-reload: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadAccessDenied =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6017, nameof(ModelHotReloadAccessDenied)),
            "[HOT_RELOAD] Access denied during hot-reload: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadInvalidOperation =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6018, nameof(ModelHotReloadInvalidOperation)),
            "[HOT_RELOAD] Invalid operation during hot-reload: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadOutOfMemory =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6019, nameof(ModelHotReloadOutOfMemory)),
            "[HOT_RELOAD] Out of memory during hot-reload: {ModelPath}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadSmokeTestStart =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(6020, nameof(ModelHotReloadSmokeTestStart)),
            "[HOT_RELOAD] Running smoke tests for: {ModelName}");

    private static readonly Action<ILogger, double, Exception?> _modelHotReloadSmokeTestConfidenceFailed =
        LoggerMessage.Define<double>(LogLevel.Error, new EventId(6021, nameof(ModelHotReloadSmokeTestConfidenceFailed)),
            "[HOT_RELOAD] Smoke test failed - confidence out of bounds: {Confidence}");

    private static readonly Action<ILogger, Exception?> _modelHotReloadSmokeTestAnomalyFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(6022, nameof(ModelHotReloadSmokeTestAnomalyFailed)),
            "[HOT_RELOAD] Smoke test failed - anomaly detected in golden input");

    private static readonly Action<ILogger, double, Exception?> _modelHotReloadSmokeTestInvalidResult =
        LoggerMessage.Define<double>(LogLevel.Error, new EventId(6023, nameof(ModelHotReloadSmokeTestInvalidResult)),
            "[HOT_RELOAD] Smoke test failed - invalid prediction result: {Result}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadSmokeTestInvalidOperationFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6024, nameof(ModelHotReloadSmokeTestInvalidOperationFailed)),
            "[HOT_RELOAD] Smoke test failed due to invalid operation: {ModelName}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadSmokeTestMemoryFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6025, nameof(ModelHotReloadSmokeTestMemoryFailed)),
            "[HOT_RELOAD] Smoke test failed due to memory exhaustion: {ModelName}");

    private static readonly Action<ILogger, string, Exception?> _modelHotReloadSmokeTestTimeout =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(6026, nameof(ModelHotReloadSmokeTestTimeout)),
            "[HOT_RELOAD] Smoke test timed out: {ModelName}");

    private static readonly Action<ILogger, Exception?> _modelHotReloadRegistryAccessDenied =
        LoggerMessage.Define(LogLevel.Warning, new EventId(6027, nameof(ModelHotReloadRegistryAccessDenied)),
            "[HOT_RELOAD] Access denied when updating model registry");

    private static readonly Action<ILogger, Exception?> _modelHotReloadRegistryDirectoryNotFound =
        LoggerMessage.Define(LogLevel.Warning, new EventId(6028, nameof(ModelHotReloadRegistryDirectoryNotFound)),
            "[HOT_RELOAD] Directory not found when updating model registry");

    private static readonly Action<ILogger, Exception?> _modelHotReloadRegistryIOError =
        LoggerMessage.Define(LogLevel.Warning, new EventId(6029, nameof(ModelHotReloadRegistryIOError)),
            "[HOT_RELOAD] IO error when updating model registry");

    private static readonly Action<ILogger, Exception?> _modelHotReloadManagerDisposed =
        LoggerMessage.Define(LogLevel.Information, new EventId(6030, nameof(ModelHotReloadManagerDisposed)),
            "[HOT_RELOAD] Model hot-reload manager disposed");

    // Public methods for the new messages
    public static void CVaRPPOTrainingArgumentError(ILogger logger, Exception ex) => _cvarPpoTrainingArgumentError(logger, ex);
    public static void CVaRPPOActionSelectionArgumentError(ILogger logger, Exception ex) => _cvarPpoActionSelectionArgumentError(logger, ex);
    public static void CVaRPPOActionSelectionOperationError(ILogger logger, Exception ex) => _cvarPpoActionSelectionOperationError(logger, ex);
    public static void CVaRPPOActionSelectionMemoryError(ILogger logger, Exception ex) => _cvarPpoActionSelectionMemoryError(logger, ex);
    public static void CVaRPPOActionSelected(ILogger logger, double action, double prob, double value, double cvar) => _cvarPpoActionSelected(logger, action, prob, value, cvar, null);
    public static void CVaRPPOUpdateParametersError(ILogger logger, Exception ex) => _cvarPpoUpdateParametersError(logger, ex);
    public static void CVaRPPOUpdateParametersArgumentError(ILogger logger, Exception ex) => _cvarPpoUpdateParametersArgumentError(logger, ex);
    public static void CVaRPPOUpdateParametersOperationError(ILogger logger, Exception ex) => _cvarPpoUpdateParametersOperationError(logger, ex);
    public static void CVaRPPOGradientClipped(ILogger logger, double norm) => _cvarPpoGradientClipped(logger, norm, null);
    public static void CVaRPPOExperienceAdded(ILogger logger, double action, double reward, bool done) => _cvarPpoExperienceAdded(logger, action, reward, done, null);
    public static void CVaRPPOModelSaveAccessDenied(ILogger logger, Exception ex) => _cvarPpoModelSaveAccessDenied(logger, ex);
    public static void CVaRPPOModelSaveDirectoryNotFound(ILogger logger, Exception ex) => _cvarPpoModelSaveDirectoryNotFound(logger, ex);
    public static void CVaRPPOModelSaveIOError(ILogger logger, Exception ex) => _cvarPpoModelSaveIOError(logger, ex);
    public static void CVaRPPOModelPathNotExists(ILogger logger, string modelPath) => _cvarPpoModelPathNotExists(logger, modelPath, null);
    public static void CVaRPPOMissingNetworkFiles(ILogger logger, string modelPath) => _cvarPpoMissingNetworkFiles(logger, modelPath, null);
    public static void CVaRPPOModelFileNotFound(ILogger logger, string modelPath, Exception ex) => _cvarPpoModelFileNotFound(logger, modelPath, ex);
    public static void CVaRPPOModelLoadAccessDenied(ILogger logger, string modelPath, Exception ex) => _cvarPpoModelLoadAccessDenied(logger, modelPath, ex);
    public static void CVaRPPOInvalidModelFile(ILogger logger, string modelPath, Exception ex) => _cvarPpoInvalidModelFile(logger, modelPath, ex);
    
    public static void SACActionSelectionArgumentError(ILogger logger, Exception ex) => _sacActionSelectionArgumentError(logger, ex);
    public static void SACActionSelectionOperationError(ILogger logger, Exception ex) => _sacActionSelectionOperationError(logger, ex);
    public static void SACActionSelectionMemoryError(ILogger logger, Exception ex) => _sacActionSelectionMemoryError(logger, ex);
    public static void SACTrainingDebug(ILogger logger) => _sacTrainingDebug(logger, null);
    public static void SACTrainingArgumentError(ILogger logger, Exception ex) => _sacTrainingArgumentError(logger, ex);
    public static void SACTrainingOperationError(ILogger logger, Exception ex) => _sacTrainingOperationError(logger, ex);
    public static void SACTrainingMemoryError(ILogger logger, Exception ex) => _sacTrainingMemoryError(logger, ex);
    public static void SACTrainingCancelled(ILogger logger, Exception ex) => _sacTrainingCancelled(logger, ex);
    public static void SACTrainingCompleted(ILogger logger, double actorLoss, double criticLoss1, double criticLoss2, double valueLoss, double entropy) => _sacTrainingCompleted(logger, actorLoss, criticLoss1, criticLoss2, valueLoss, entropy, null);
    public static void SACTrainingOperationFailed(ILogger logger, Exception ex) => _sacTrainingOperationFailed(logger, ex);
    public static void SACTrainingMemoryFailed(ILogger logger, Exception ex) => _sacTrainingMemoryFailed(logger, ex);
    
    public static void MetaAdaptationStep(ILogger logger, int step, int totalSteps, double loss) => _metaAdaptationStep(logger, step, totalSteps, loss, null);
    public static void MetaExperienceStored(ILogger logger, string taskId, double reward) => _metaExperienceStored(logger, taskId, reward, null);
    public static void MetaTaskAdaptationArgumentError(ILogger logger, string taskId, Exception ex) => _metaTaskAdaptationArgumentError(logger, taskId, ex);
    public static void MetaTaskAdaptationOperationError(ILogger logger, string taskId, Exception ex) => _metaTaskAdaptationOperationError(logger, taskId, ex);
    public static void MetaTaskAdaptationMemoryError(ILogger logger, string taskId, Exception ex) => _metaTaskAdaptationMemoryError(logger, taskId, ex);
    public static void MetaTrainingInfo(ILogger logger, double loss, int updates, int episodes) => _metaTrainingInfo(logger, loss, updates, episodes, null);
    public static void MetaTrainingArgumentError(ILogger logger, string message, Exception ex) => _metaTrainingArgumentError(logger, message, ex);
    public static void MetaTrainingOperationError(ILogger logger, string message, Exception ex) => _metaTrainingOperationError(logger, message, ex);
    public static void MetaTrainingMemoryError(ILogger logger, string message, Exception ex) => _metaTrainingMemoryError(logger, message, ex);
    
    public static void OnnxGpuAccelerationEnabled(ILogger logger) => _onnxGpuAccelerationEnabled(logger, null);
    public static void OnnxModelValidationDebug(ILogger logger, string modelName, string inputShape, string outputShape) => _onnxModelValidationDebug(logger, modelName, inputShape, outputShape, null);
    public static void OnnxDisposedDebug(ILogger logger) => _onnxDisposedDebug(logger, null);
    
    public static void FeatureEngineeringDebug(ILogger logger, string message) => _featureEngineeringDebug(logger, message, null);
    public static void FeatureStreamingTickProcessed(ILogger logger, string symbol, double price, double volume) => _featureStreamingTickProcessed(logger, symbol, price, volume, null);
    
    public static void ModelHotReloadError(ILogger logger, string message, Exception ex) => _modelHotReloadError(logger, message, ex);
    public static void ModelHotReloadWarning(ILogger logger, string message, Exception ex) => _modelHotReloadWarning(logger, message, ex);
    public static void ModelHotReloadInfo(ILogger logger, string message) => _modelHotReloadInfo(logger, message, null);
    public static void ModelHotReloadDebug(ILogger logger, string message) => _modelHotReloadDebug(logger, message, null);
    public static void ModelHotReloadWarningParam(ILogger logger, string modelPath) => _modelHotReloadWarningParam(logger, modelPath, null);
    public static void ModelHotReloadFailedLoad(ILogger logger, string modelPath) => _modelHotReloadFailedLoad(logger, modelPath, null);
    public static void ModelHotReloadSmokeTestFailed(ILogger logger, string candidateName) => _modelHotReloadSmokeTestFailed(logger, candidateName, null);
    public static void ModelHotReloadCompleted(ILogger logger, string candidateName) => _modelHotReloadCompleted(logger, candidateName, null);
    public static void ModelHotReloadFileNotFound(ILogger logger, string modelPath, Exception ex) => _modelHotReloadFileNotFound(logger, modelPath, ex);
    public static void ModelHotReloadAccessDenied(ILogger logger, string modelPath, Exception ex) => _modelHotReloadAccessDenied(logger, modelPath, ex);
    public static void ModelHotReloadInvalidOperation(ILogger logger, string modelPath, Exception ex) => _modelHotReloadInvalidOperation(logger, modelPath, ex);
    public static void ModelHotReloadOutOfMemory(ILogger logger, string modelPath, Exception ex) => _modelHotReloadOutOfMemory(logger, modelPath, ex);
    public static void ModelHotReloadSmokeTestStart(ILogger logger, string modelName) => _modelHotReloadSmokeTestStart(logger, modelName, null);
    public static void ModelHotReloadSmokeTestConfidenceFailed(ILogger logger, double confidence) => _modelHotReloadSmokeTestConfidenceFailed(logger, confidence, null);
    public static void ModelHotReloadSmokeTestAnomalyFailed(ILogger logger) => _modelHotReloadSmokeTestAnomalyFailed(logger, null);
    public static void ModelHotReloadSmokeTestInvalidResult(ILogger logger, double result) => _modelHotReloadSmokeTestInvalidResult(logger, result, null);
    public static void ModelHotReloadSmokeTestInvalidOperationFailed(ILogger logger, string modelName, Exception ex) => _modelHotReloadSmokeTestInvalidOperationFailed(logger, modelName, ex);
    public static void ModelHotReloadSmokeTestMemoryFailed(ILogger logger, string modelName, Exception ex) => _modelHotReloadSmokeTestMemoryFailed(logger, modelName, ex);
    public static void ModelHotReloadSmokeTestTimeout(ILogger logger, string modelName, Exception ex) => _modelHotReloadSmokeTestTimeout(logger, modelName, ex);
    public static void ModelHotReloadRegistryAccessDenied(ILogger logger, Exception ex) => _modelHotReloadRegistryAccessDenied(logger, ex);
    public static void ModelHotReloadRegistryDirectoryNotFound(ILogger logger, Exception ex) => _modelHotReloadRegistryDirectoryNotFound(logger, ex);
    public static void ModelHotReloadRegistryIOError(ILogger logger, Exception ex) => _modelHotReloadRegistryIOError(logger, ex);
    public static void ModelHotReloadManagerDisposed(ILogger logger) => _modelHotReloadManagerDisposed(logger, null);
}