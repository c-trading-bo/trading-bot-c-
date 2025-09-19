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
    private static readonly Action<ILogger, string, Exception?> _modelHotReloadStarted =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(6001, nameof(ModelHotReloadStarted)),
            "[MODEL_RELOAD] Started monitoring directory: {Directory}");

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
    public static void ModelHotReloadStarted(ILogger logger, string directory) => _modelHotReloadStarted(logger, directory, null);
    public static void PositionSizeCalculated(ILogger logger, string symbol, double size) => _positionSizeCalculated(logger, symbol, size, null);
}