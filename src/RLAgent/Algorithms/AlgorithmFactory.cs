using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using System;
using TradingBot.RLAgent.Algorithms;
using TradingBot.RLAgent.Models;

namespace TradingBot.RLAgent;

/// <summary>
/// Factory for creating RL algorithms based on configuration
/// Supports SAC, Meta-RL, and existing CVaR-PPO algorithms
/// </summary>
public static class AlgorithmFactory
{
    /// <summary>
    /// Create RL algorithm based on environment variable or configuration
    /// </summary>
    public static IRLAlgorithm CreateAlgorithm(ILogger logger, AlgorithmConfig? config = null)
    {
        // Check environment variable for algorithm selection
        var algorithmType = Environment.GetEnvironmentVariable("RL_ALGO")?.ToUpperInvariant() ?? "CVAR_PPO";
        
        config ??= new AlgorithmConfig();
        
        return algorithmType switch
        {
            "SAC" => CreateSACAlgorithm(logger, config),
            "META" or "MAML" => CreateMetaLearningAlgorithm(logger, config),
            "CVAR_PPO" or "PPO" => CreateCVaRPPOAlgorithm(logger, config),
            _ => throw new ArgumentException($"Unknown algorithm type: {algorithmType}")
        };
    }

    /// <summary>
    /// Create Soft Actor-Critic algorithm
    /// </summary>
    private static IRLAlgorithm CreateSACAlgorithm(ILogger logger, AlgorithmConfig config)
    {
        var sacConfig = new Models.SacConfig
        {
            StateDimension = config.StateDimension,
            ActionDimension = config.ActionDimension,
            HiddenDimension = config.HiddenDimension,
            LearningRateActor = config.LearningRateActor,
            LearningRateCritic = config.LearningRateCritic,
            LearningRateValue = config.LearningRateValue,
            Gamma = config.Gamma,
            Tau = config.Tau,
            TemperatureAlpha = config.TemperatureAlpha,
            BufferSize = config.BufferSize,
            MinBufferSize = config.MinBufferSize,
            BatchSize = config.BatchSize,
            ActionLowBound = config.ActionLowBound,
            ActionHighBound = config.ActionHighBound
        };
        
        var sacLogger = logger as ILogger<SoftActorCritic> ?? 
            new NullLogger<SoftActorCritic>();
        
        var sac = new SoftActorCritic(sacLogger, sacConfig);
        
        logger.LogInformation("[ALGORITHM_FACTORY] Created SAC algorithm with config: {Config}", 
            System.Text.Json.JsonSerializer.Serialize(sacConfig));
        
        return new SacAlgorithmWrapper(sac);
    }

    /// <summary>
    /// Create Meta-Learning algorithm
    /// </summary>
    private static IRLAlgorithm CreateMetaLearningAlgorithm(ILogger logger, AlgorithmConfig config)
    {
        var metaConfig = new MetaLearningConfig
        {
            StateDimension = config.StateDimension,
            ActionDimension = config.ActionDimension,
            HiddenDimension = config.HiddenDimension,
            MetaLearningRate = config.MetaLearningRate,
            TaskLearningRate = config.TaskLearningRate,
            AdaptationSteps = config.AdaptationSteps,
            MetaBatchSize = config.MetaBatchSize,
            MinTasksForMetaUpdate = config.MinTasksForMetaUpdate,
            MetaBufferSize = config.MetaBufferSize,
            TemporalDecay = config.TemporalDecay
        };
        
        var metaLogger = logger as ILogger<MetaLearner> ?? 
            new NullLogger<MetaLearner>();
        
        var metaLearner = new MetaLearner(metaLogger, metaConfig);
        
        logger.LogInformation("[ALGORITHM_FACTORY] Created Meta-Learning algorithm with config: {Config}", 
            System.Text.Json.JsonSerializer.Serialize(metaConfig));
        
        return new MetaLearningAlgorithmWrapper(metaLearner);
    }

    /// <summary>
    /// Create CVaR-PPO algorithm (existing implementation)
    /// </summary>
    private static IRLAlgorithm CreateCVaRPPOAlgorithm(ILogger logger, AlgorithmConfig config)
    {
        // Use existing CVaRPPO implementation
        var ppoConfig = new CVaRPPOConfig
        {
            StateSize = config.StateDimension,
            ActionSize = config.ActionDimension,
            HiddenSize = config.HiddenDimension,
            LearningRate = config.LearningRateActor,
            Gamma = config.Gamma,
            Lambda = config.LambdaGae,
            ClipEpsilon = config.EpsilonClip,
            EntropyCoeff = config.EntropyCoeff,
            CVaRAlpha = config.CvarAlpha
        };
        
        var cvarLogger = logger as ILogger<CVaRPPO> ?? 
            new NullLogger<CVaRPPO>();
        
        var cvarPpo = new CVaRPPO(cvarLogger, ppoConfig);
        
        logger.LogInformation("[ALGORITHM_FACTORY] Created CVaR-PPO algorithm with config: {Config}", 
            System.Text.Json.JsonSerializer.Serialize(ppoConfig));
        
        return new CVaRppoAlgorithmWrapper(cvarPpo);
    }

    /// <summary>
    /// Get algorithm description
    /// </summary>
    public static string GetAlgorithmDescription(string algorithmType)
    {
        return algorithmType.ToUpperInvariant() switch
        {
            "SAC" => "Soft Actor-Critic - Off-policy algorithm with continuous actions and automatic entropy tuning",
            "META" or "MAML" => "Model-Agnostic Meta-Learning - Fast adaptation to new market regimes",
            "CVAR_PPO" or "PPO" => "Conditional Value at Risk PPO - Risk-aware policy optimization",
            _ => "Unknown algorithm type"
        };
    }

    /// <summary>
    /// Check if algorithm type is supported
    /// </summary>
    public static bool IsAlgorithmSupported(string algorithmType)
    {
        return algorithmType.ToUpperInvariant() switch
        {
            "SAC" or "META" or "MAML" or "CVAR_PPO" or "PPO" => true,
            _ => false
        };
    }
}

/// <summary>
/// Common interface for all RL algorithms
/// </summary>
public interface IRLAlgorithm
{
    Task<double[]> SelectActionAsync(double[] state, bool isTraining = true);
    void StoreExperience(double[] state, double[] action, double reward, double[] nextState, bool done);
    Task<ITrainingResult> TrainAsync();
    string GetAlgorithmType();
    object GetStatistics();
}

/// <summary>
/// Common interface for training results
/// </summary>
public interface ITrainingResult
{
    bool Success { get; }
    string? Message { get; }
    double Loss { get; }
}

/// <summary>
/// Unified algorithm configuration
/// </summary>
public class AlgorithmConfig
{
    // Common parameters
    public int StateDimension { get; set; } = 20;
    public int ActionDimension { get; set; } = 1;
    public int HiddenDimension { get; set; } = 256;
    public double Gamma { get; set; } = 0.99;
    
    // SAC-specific parameters
    public double LearningRateActor { get; set; } = 3e-4;
    public double LearningRateCritic { get; set; } = 3e-4;
    public double LearningRateValue { get; set; } = 3e-4;
    public double Tau { get; set; } = 0.005;
    public double TemperatureAlpha { get; set; } = 0.2;
    public int BufferSize { get; set; } = 100000;
    public int MinBufferSize { get; set; } = 1000;
    public int BatchSize { get; set; } = 64;
    public double ActionLowBound { get; set; } = -1.0;
    public double ActionHighBound { get; set; } = 1.0;
    
    // Meta-learning parameters
    public double MetaLearningRate { get; set; } = 1e-3;
    public double TaskLearningRate { get; set; } = 1e-2;
    public int AdaptationSteps { get; set; } = 5;
    public int MetaBatchSize { get; set; } = 4;
    public int MinTasksForMetaUpdate { get; set; } = 8;
    public int MetaBufferSize { get; set; } = 10000;
    public double TemporalDecay { get; set; } = 0.99;
    
    // PPO-specific parameters
    public double LambdaGae { get; set; } = 0.95;
    public double EpsilonClip { get; set; } = 0.2;
    public double ValueCoeff { get; set; } = 0.5;
    public double EntropyCoeff { get; set; } = 0.01;
    public double MaxGradNorm { get; set; } = 0.5;
    public double TargetKl { get; set; } = 0.01;
    public double CvarAlpha { get; set; } = 0.05;
}

#region Algorithm Wrappers

/// <summary>
/// Wrapper for SAC algorithm
/// </summary>
public class SacAlgorithmWrapper : IRLAlgorithm
{
    private readonly SoftActorCritic _sac;

    public SacAlgorithmWrapper(SoftActorCritic sac)
    {
        _sac = sac;
    }

    public Task<double[]> SelectActionAsync(double[] state, bool isTraining = true)
    {
        return _sac.SelectActionAsync(state, isTraining);
    }

    public void StoreExperience(double[] state, double[] action, double reward, double[] nextState, bool done)
    {
        _sac.StoreExperience(state, action, reward, nextState, done);
    }

    public async Task<ITrainingResult> TrainAsync()
    {
        var result = await _sac.TrainAsync().ConfigureAwait(false);
        return new SacTrainingResultWrapper(result);
    }

    public string GetAlgorithmType() => "SAC";

    public object GetStatistics() => _sac.GetStatistics();
}

/// <summary>
/// Wrapper for Meta-Learning algorithm
/// </summary>
public class MetaLearningAlgorithmWrapper : IRLAlgorithm
{
    private readonly MetaLearner _metaLearner;
    private string _currentTaskId = "default";

    public MetaLearningAlgorithmWrapper(MetaLearner metaLearner)
    {
        _metaLearner = metaLearner;
    }

    public Task<double[]> SelectActionAsync(double[] state, bool isTraining = true)
    {
        var policy = _metaLearner.GetTaskPolicy(_currentTaskId) ?? _metaLearner.GetMetaPolicy();
        return Task.FromResult(policy.Predict(state));
    }

    public void StoreExperience(double[] state, double[] action, double reward, double[] nextState, bool done)
    {
        _metaLearner.StoreTaskExperience(_currentTaskId, state, action, reward, nextState, done);
    }

    public async Task<ITrainingResult> TrainAsync()
    {
        var result = await _metaLearner.MetaTrainAsync().ConfigureAwait(false);
        return new MetaTrainingResultWrapper(result);
    }

    public string GetAlgorithmType() => "Meta-Learning";

    public object GetStatistics() => _metaLearner.GetStatistics();

    public void SetCurrentTask(string taskId) => _currentTaskId = taskId;
}

/// <summary>
/// Wrapper for CVaR-PPO algorithm
/// </summary>
public class CVaRppoAlgorithmWrapper : IRLAlgorithm
{
    private readonly CVaRPPO _cvarPpo;

    public CVaRppoAlgorithmWrapper(CVaRPPO cvarPpo)
    {
        _cvarPpo = cvarPpo;
    }

    public async Task<double[]> SelectActionAsync(double[] state, bool isTraining = true)
    {
        var actionResult = await _cvarPpo.GetActionAsync(state, deterministic: !isTraining).ConfigureAwait(false);
        return new[] { (double)actionResult.Action };
    }

    public void StoreExperience(double[] state, double[] action, double reward, double[] nextState, bool done)
    {
        // Convert to CVaR-PPO format
        var actionValue = action.Length > 0 ? (int)Math.Round(action[0]) : 0;
        var experience = new Experience
        {
            State = (double[])state.Clone(),
            Action = actionValue,
            Reward = reward,
            NextState = (double[])nextState.Clone(),
            Done = done,
            LogProbability = 0.0, // Would be calculated during action selection
            ValueEstimate = 0.0   // Would be calculated during action selection
        };
        _cvarPpo.AddExperience(experience);
    }

    public async Task<ITrainingResult> TrainAsync()
    {
        var result = await _cvarPpo.TrainAsync().ConfigureAwait(false);
        return new CVaRppoTrainingResultWrapper(result);
    }

    public string GetAlgorithmType() => "CVaR-PPO";

    public object GetStatistics() => _cvarPpo.GetTrainingStatistics();
}

#endregion

#region Training Result Wrappers

public class SacTrainingResultWrapper : ITrainingResult
{
    private readonly Models.SacTrainingResult _result;

    public SacTrainingResultWrapper(Models.SacTrainingResult result)
    {
        _result = result;
    }

    public bool Success => _result.Success;
    public string? Message => _result.Message;
    public double Loss => _result.ActorLoss;
}

public class MetaTrainingResultWrapper : ITrainingResult
{
    private readonly MetaTrainingResult _result;

    public MetaTrainingResultWrapper(MetaTrainingResult result)
    {
        _result = result;
    }

    public bool Success => _result.Success;
    public string? Message => _result.Message;
    public double Loss => _result.MetaLoss;
}

public class CVaRppoTrainingResultWrapper : ITrainingResult
{
    private readonly TrainingResult _result;

    public CVaRppoTrainingResultWrapper(TrainingResult result)
    {
        _result = result;
    }

    public bool Success => _result.Success;
    public string? Message => _result.ErrorMessage;
    public double Loss => _result.PolicyLoss;
}

#endregion