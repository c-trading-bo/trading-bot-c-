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
            "[META] Adapting to task: {TaskId} with {SupportSize} examples");

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

    // Position Sizing Messages
    private static readonly Action<ILogger, string, double, Exception?> _positionSizeCalculated =
        LoggerMessage.Define<string, double>(LogLevel.Information, new EventId(7001, nameof(PositionSizeCalculated)),
            "[POSITION_SIZING] Calculated position size for {Symbol}: {Size}");

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
    public static void PositionSizeCalculated(ILogger logger, string symbol, double size) => _positionSizeCalculated(logger, symbol, size, null);
}