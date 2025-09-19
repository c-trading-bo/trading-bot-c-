using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using System.Threading.Tasks;
using System.Threading;
using TradingBot.RLAgent.Algorithms;
using TradingBot.RLAgent.Models;

namespace TradingBot.RLAgent.Algorithms;

/// <summary>
/// Neural network initialization constants for SAC algorithm
/// </summary>
internal static class NeuralNetworkConstants
{
    public const double WEIGHT_INIT_RANGE = 2.0;
    public const double XAVIER_FACTOR = 2.0;
    public const double EXPLORATION_NOISE_FACTOR = 0.1; // Small exploration noise
    public const double RANDOM_VALUE_MULTIPLIER = 2.0;
    public const double RANDOM_VALUE_OFFSET = 1.0;
}

/// <summary>
/// Soft Actor-Critic (SAC) algorithm implementation for continuous control
/// Designed for position sizing and entry/exit timing in trading systems
/// </summary>
public class SoftActorCritic
{
    private readonly ILogger<SoftActorCritic> _logger;
    private readonly Models.SacConfig _config;
    
    // Neural networks
    private readonly ActorNetwork _actor;
    private readonly CriticNetwork _critic1;
    private readonly CriticNetwork _critic2;
    private readonly CriticNetwork _targetCritic1;
    private readonly CriticNetwork _targetCritic2;
    private readonly ValueNetwork _valueNetwork;
    
    // Experience replay buffer
    private readonly ExperienceReplayBuffer _replayBuffer;
    
    // Training statistics
    private int _totalSteps;
    private double _averageReward;
    private double _entropy;
    
    // Constants for moving average
    private const double RewardMovingAverageDecay = 0.99;
    private const double RewardMovingAverageWeight = 0.01;
    
    public SoftActorCritic(ILogger<SoftActorCritic> logger, Models.SacConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        
        _logger = logger;
        _config = config;
        
        // Initialize networks with proper dimensions
        var stateDim = config.StateDimension;
        var actionDim = config.ActionDimension;
        var hiddenDim = config.HiddenDimension;
        
        _actor = new ActorNetwork(stateDim, actionDim, hiddenDim, config.LearningRateActor);
        _critic1 = new CriticNetwork(stateDim + actionDim, 1, hiddenDim, config.LearningRateCritic);
        _critic2 = new CriticNetwork(stateDim + actionDim, 1, hiddenDim, config.LearningRateCritic);
        _targetCritic1 = new CriticNetwork(stateDim + actionDim, 1, hiddenDim, config.LearningRateCritic);
        _targetCritic2 = new CriticNetwork(stateDim + actionDim, 1, hiddenDim, config.LearningRateCritic);
        _valueNetwork = new ValueNetwork(stateDim, 1, hiddenDim, config.LearningRateValue);
        
        // Initialize experience replay buffer
        _replayBuffer = new ExperienceReplayBuffer(config.BufferSize);
        
        // Copy weights to target networks
        _targetCritic1.CopyWeightsFrom(_critic1);
        _targetCritic2.CopyWeightsFrom(_critic2);
        
        _logger.LogInformation("[SAC] Initialized with state_dim={StateDim}, action_dim={ActionDim}, hidden_dim={HiddenDim}", 
            stateDim, actionDim, hiddenDim);
    }

    /// <summary>
    /// Select action using the current policy (for live trading)
    /// </summary>
    public async Task<double[]> SelectActionAsync(double[] state, bool isTraining = true, CancellationToken cancellationToken = default)
    {
        try
        {
            await Task.CompletedTask.ConfigureAwait(false); // Ensure async pattern compliance
            
            var action = _actor.SampleAction(state, isTraining);
            
            // Clip actions to valid range for trading
            for (int i = 0; i < action.Length; i++)
            {
                action[i] = Math.Max(_config.ActionLowBound, Math.Min(_config.ActionHighBound, action[i]));
            }
            
            if (isTraining)
            {
                _totalSteps++;
            }
            
            return action;
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "[SAC] Invalid arguments for action selection");
            // Return safe default action (no position change)
            return new double[_config.ActionDimension];
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "[SAC] Invalid operation during action selection");
            // Return safe default action (no position change)
            return new double[_config.ActionDimension];
        }
        catch (OutOfMemoryException ex)
        {
            _logger.LogError(ex, "[SAC] Out of memory during action selection");
            throw; // Rethrow memory issues
        }
    }

    /// <summary>
    /// Store experience in replay buffer for training
    /// </summary>
    public void StoreExperience(double[] state, double[] action, double reward, double[] nextState, bool done)
    {
        var experience = new Experience
        {
            State = (double[])state.Clone(),
            Action = (double[])action.Clone(),
            Reward = reward,
            NextState = (double[])nextState.Clone(),
            Done = done
        };
        
        _replayBuffer.Add(experience);
        
        // Update moving average reward
        _averageReward = RewardMovingAverageDecay * _averageReward + RewardMovingAverageWeight * reward;
    }

    /// <summary>
    /// Train the SAC agent using experience replay
    /// </summary>
    public async Task<Models.SacTrainingResult> TrainAsync(CancellationToken cancellationToken = default)
    {
        if (_replayBuffer.Count < _config.MinBufferSize)
        {
            await Task.CompletedTask.ConfigureAwait(false); // Ensure async pattern compliance
            return new Models.SacTrainingResult
            {
                Success = false,
                Message = $"Insufficient experience: {_replayBuffer.Count} < {_config.MinBufferSize}"
            };
        }

        try
        {
            await Task.CompletedTask.ConfigureAwait(false); // Ensure async pattern compliance
            
            var batchSize = Math.Min(_config.BatchSize, _replayBuffer.Count);
            var batch = _replayBuffer.SampleBatch(batchSize);
            
            // Extract batch data
            var states = batch.Select(e => e.State.ToArray()).ToArray();
            var actions = batch.Select(e => e.Action.ToArray()).ToArray();
            var rewards = batch.Select(e => e.Reward).ToArray();
            var nextStates = batch.Select(e => e.NextState.ToArray()).ToArray();
            var dones = batch.Select(e => e.Done).ToArray();
            
            // Train critic networks
            var criticLoss1 = TrainCritic(_critic1, states, actions, rewards, nextStates, dones);
            var criticLoss2 = TrainCritic(_critic2, states, actions, rewards, nextStates, dones);
            
            // Train actor network
            var (actorLoss, entropy) = TrainActor(states);
            _entropy = entropy;
            
            // Train value network
            var valueLoss = TrainValueNetwork(states);
            
            // Soft update target networks
            SoftUpdateTargetNetworks();
            
            var result = new Models.SacTrainingResult
            {
                Success = true,
                ActorLoss = actorLoss,
                CriticLoss1 = criticLoss1,
                CriticLoss2 = criticLoss2,
                ValueLoss = valueLoss,
                Entropy = entropy,
                AverageReward = _averageReward,
                BufferSize = _replayBuffer.Count,
                TotalSteps = _totalSteps
            };
            
            _logger.LogDebug("[SAC] Training completed: Actor={ActorLoss:F4}, Critic1={CriticLoss1:F4}, Critic2={CriticLoss2:F4}, Value={ValueLoss:F4}, Entropy={Entropy:F4}", 
                actorLoss, criticLoss1, criticLoss2, valueLoss, entropy);
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SAC] Training failed");
            return new Models.SacTrainingResult
            {
                Success = false,
                Message = ex.Message
            };
        }
    }

    /// <summary>
    /// Train critic network using TD error
    /// </summary>
    private double TrainCritic(CriticNetwork critic, double[][] states, double[][] actions, double[] rewards, double[][] nextStates, bool[] dones)
    {
        var totalLoss = 0.0;
        var batchSize = states.Length;
        
        for (int i = 0; i < batchSize; i++)
        {
            // Sample next action from current policy
            var nextAction = _actor.SampleAction(nextStates[i], isTraining: true);
            
            // Calculate target Q-value using target critics and value network
            var nextQValue1 = _targetCritic1.Predict(CombineStateAction(nextStates[i], nextAction));
            var nextQValue2 = _targetCritic2.Predict(CombineStateAction(nextStates[i], nextAction));
            var nextQValue = Math.Min(nextQValue1, nextQValue2);
            
            var targetQValue = rewards[i];
            if (!dones[i])
            {
                targetQValue += _config.Gamma * nextQValue;
            }
            
            // Current Q-value
            var currentQValue = critic.Predict(CombineStateAction(states[i], actions[i]));
            
            // TD error
            var tdError = targetQValue - currentQValue;
            var loss = tdError * tdError;
            totalLoss += loss;
            
            // Update critic network
            critic.UpdateWeights(CombineStateAction(states[i], actions[i]), targetQValue);
        }
        
        return totalLoss / batchSize;
    }

    /// <summary>
    /// Train actor network to maximize Q-value and entropy
    /// </summary>
    private (double loss, double entropy) TrainActor(double[][] states)
    {
        var totalLoss = 0.0;
        var totalEntropy = 0.0;
        var batchSize = states.Length;
        
        for (int i = 0; i < batchSize; i++)
        {
            var action = _actor.SampleAction(states[i], isTraining: true);
            var logProb = _actor.GetLogProbability(states[i], action);
            
            // Q-values from both critics
            var qValue1 = _critic1.Predict(CombineStateAction(states[i], action));
            var qValue2 = _critic2.Predict(CombineStateAction(states[i], action));
            var qValue = Math.Min(qValue1, qValue2);
            
            // SAC loss: maximize Q-value and entropy
            var loss = _config.TemperatureAlpha * logProb - qValue;
            totalLoss += loss;
            
            // Calculate entropy for logging
            totalEntropy += -logProb;
            
            // Update actor network
            _actor.UpdateWeights(states[i], -loss); // Negative because we want to minimize loss
        }
        
        return (totalLoss / batchSize, totalEntropy / batchSize);
    }

    /// <summary>
    /// Train value network to predict expected return
    /// </summary>
    private double TrainValueNetwork(double[][] states)
    {
        var totalLoss = 0.0;
        var batchSize = states.Length;
        
        for (int i = 0; i < batchSize; i++)
        {
            var action = _actor.SampleAction(states[i], isTraining: true);
            var logProb = _actor.GetLogProbability(states[i], action);
            
            // Target value: Q-value minus entropy term
            var qValue1 = _critic1.Predict(CombineStateAction(states[i], action));
            var qValue2 = _critic2.Predict(CombineStateAction(states[i], action));
            var qValue = Math.Min(qValue1, qValue2);
            var targetValue = qValue - _config.TemperatureAlpha * logProb;
            
            // Current value
            var currentValue = _valueNetwork.Predict(states[i]);
            
            // Value loss
            var loss = Math.Pow(targetValue - currentValue, 2);
            totalLoss += loss;
            
            // Update value network
            _valueNetwork.UpdateWeights(states[i], targetValue);
        }
        
        return totalLoss / batchSize;
    }

    /// <summary>
    /// Soft update target networks
    /// </summary>
    private void SoftUpdateTargetNetworks()
    {
        _targetCritic1.SoftUpdate(_critic1, _config.Tau);
        _targetCritic2.SoftUpdate(_critic2, _config.Tau);
    }

    /// <summary>
    /// Combine state and action into single input vector
    /// </summary>
    private static double[] CombineStateAction(double[] state, double[] action)
    {
        var combined = new double[state.Length + action.Length];
        Array.Copy(state, 0, combined, 0, state.Length);
        Array.Copy(action, 0, combined, state.Length, action.Length);
        return combined;
    }

    /// <summary>
    /// Get current training statistics
    /// </summary>
    public Models.SacStatistics GetStatistics()
    {
        return new Models.SacStatistics
        {
            TotalSteps = _totalSteps,
            AverageReward = _averageReward,
            Entropy = _entropy,
            BufferSize = _replayBuffer.Count,
            MaxBufferSize = _config.BufferSize
        };
    }
}

#region Supporting Classes

/// <summary>
/// SAC configuration parameters
/// </summary>
public class SacConfig
{
    public int StateDimension { get; set; } = 20;          // Number of features
    public int ActionDimension { get; set; } = 1;          // Position size
    public int HiddenDimension { get; set; } = 256;        // Hidden layer size
    public double LearningRateActor { get; set; } = 3e-4;  // Actor learning rate
    public double LearningRateCritic { get; set; } = 3e-4; // Critic learning rate  
    public double LearningRateValue { get; set; } = 3e-4;  // Value network learning rate
    public double Gamma { get; set; } = 0.99;              // Discount factor
    public double Tau { get; set; } = 0.005;               // Soft update rate
    public double TemperatureAlpha { get; set; } = 0.2;    // Entropy regularization
    public int BufferSize { get; set; } = 100000;          // Experience replay buffer size
    public int MinBufferSize { get; set; } = 1000;         // Minimum experiences before training
    public int BatchSize { get; set; } = 64;               // Training batch size
    public double ActionLowBound { get; set; } = -1.0;     // Minimum action value
    public double ActionHighBound { get; set; } = 1.0;     // Maximum action value
}

/// <summary>
/// Experience for replay buffer
/// </summary>
public class Experience
{
    public IReadOnlyList<double> State { get; set; } = Array.Empty<double>();
    public IReadOnlyList<double> Action { get; set; } = Array.Empty<double>();
    public double Reward { get; set; }
    public IReadOnlyList<double> NextState { get; set; } = Array.Empty<double>();
    public bool Done { get; set; }
}

/// <summary>
/// Training result from SAC update
/// </summary>
public class SacTrainingResult
{
    public bool Success { get; set; }
    public string? Message { get; set; }
    public double ActorLoss { get; set; }
    public double CriticLoss1 { get; set; }
    public double CriticLoss2 { get; set; }
    public double ValueLoss { get; set; }
    public double Entropy { get; set; }
    public double AverageReward { get; set; }
    public int BufferSize { get; set; }
    public int TotalSteps { get; set; }
}

/// <summary>
/// Current SAC statistics
/// </summary>
public class SacStatistics
{
    public int TotalSteps { get; set; }
    public double AverageReward { get; set; }
    public double Entropy { get; set; }
    public int BufferSize { get; set; }
    public int MaxBufferSize { get; set; }
}

/// <summary>
/// Experience replay buffer with random sampling
/// </summary>
public class ExperienceReplayBuffer : IDisposable
{
    private readonly List<Experience> _buffer;
    private readonly int _maxSize;
    private readonly System.Security.Cryptography.RandomNumberGenerator _rng;
    private int _index;

    public ExperienceReplayBuffer(int maxSize)
    {
        _maxSize = maxSize;
        _buffer = new List<Experience>(maxSize);
        _rng = System.Security.Cryptography.RandomNumberGenerator.Create();
    }

    public int Count => _buffer.Count;

    public void Add(Experience experience)
    {
        if (_buffer.Count < _maxSize)
        {
            _buffer.Add(experience);
        }
        else
        {
            // Circular buffer: overwrite oldest experience
            _buffer[_index] = experience;
            _index = (_index + 1) % _maxSize;
        }
    }

    public List<Experience> SampleBatch(int batchSize)
    {
        if (batchSize > _buffer.Count)
            throw new ArgumentException($"Batch size {batchSize} is larger than buffer size {_buffer.Count}");

        var batch = new List<Experience>(batchSize);
        var indices = new HashSet<int>();
        
        while (indices.Count < batchSize)
        {
            var bytes = new byte[4];
            _rng.GetBytes(bytes);
            var randomIndex = BitConverter.ToUInt32(bytes, 0) % _buffer.Count;
            indices.Add((int)randomIndex);
        }
        
        foreach (var index in indices)
        {
            batch.Add(_buffer[index]);
        }
        
        return batch;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _rng?.Dispose();
        }
    }
}

/// <summary>
/// Simple neural network for actor (policy)
/// </summary>
public class ActorNetwork : IDisposable
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _hiddenDim;
    private readonly double _learningRate;
    private readonly System.Security.Cryptography.RandomNumberGenerator _rng;
    private bool _disposed;
    
    // Network weights (simplified implementation)
    private double[][] _weightsInput = null!;
    private double[] _biasHidden = null!;
    private double[][] _weightsOutput = null!;
    private double[] _biasOutput = null!;

    public ActorNetwork(int inputDim, int outputDim, int hiddenDim, double learningRate)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _hiddenDim = hiddenDim;
        _learningRate = learningRate;
        _rng = System.Security.Cryptography.RandomNumberGenerator.Create();
        
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        // Xavier initialization
        var scale = Math.Sqrt(NeuralNetworkConstants.XAVIER_FACTOR / _inputDim);
        
        _weightsInput = new double[_inputDim][];
        for (int i = 0; i < _inputDim; i++)
        {
            _weightsInput[i] = new double[_hiddenDim];
        }
        _biasHidden = new double[_hiddenDim];
        _weightsOutput = new double[_hiddenDim][];
        for (int i = 0; i < _hiddenDim; i++)
        {
            _weightsOutput[i] = new double[_outputDim];
        }
        _biasOutput = new double[_outputDim];
        
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _weightsInput[i][j] = (GetRandomDouble() * NeuralNetworkConstants.WEIGHT_INIT_RANGE - 1) * scale;
            }
        }
        
        scale = Math.Sqrt(NeuralNetworkConstants.XAVIER_FACTOR / _hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                _weightsOutput[i][j] = (GetRandomDouble() * NeuralNetworkConstants.WEIGHT_INIT_RANGE - 1) * scale;
            }
        }
    }

    private double GetRandomDouble()
    {
        var bytes = new byte[8];
        _rng.GetBytes(bytes);
        return Math.Abs(BitConverter.ToDouble(bytes, 0) % 1.0);
    }

    public double[] SampleAction(double[] state, bool isTraining)
    {
        ArgumentNullException.ThrowIfNull(state);
        // Forward pass
        var hidden = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            hidden[i] = _biasHidden[i];
            for (int j = 0; j < _inputDim; j++)
            {
                hidden[i] += state[j] * _weightsInput[j][i];
            }
            hidden[i] = Math.Max(0, hidden[i]); // ReLU activation
        }
        
        var output = new double[_outputDim];
        for (int i = 0; i < _outputDim; i++)
        {
            output[i] = _biasOutput[i];
            for (int j = 0; j < _hiddenDim; j++)
            {
                output[i] += hidden[j] * _weightsOutput[j][i];
            }
            output[i] = Math.Tanh(output[i]); // Tanh for bounded actions
        }
        
        // Add exploration noise if training
        if (isTraining)
        {
            for (int i = 0; i < _outputDim; i++)
            {
                var bytes = new byte[8];
                _rng.GetBytes(bytes);
                var randomValue = Math.Abs(BitConverter.ToDouble(bytes, 0) % 1.0);
                output[i] += (randomValue * NeuralNetworkConstants.RANDOM_VALUE_MULTIPLIER - NeuralNetworkConstants.RANDOM_VALUE_OFFSET) * NeuralNetworkConstants.EXPLORATION_NOISE_FACTOR; // Small exploration noise
            }
        }
        
        return output;
    }

    public double GetLogProbability(double[] state, double[] action)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(action);
        // Simplified log probability calculation
        // In a full implementation, this would use proper probability distributions
        var predictedAction = SampleAction(state, isTraining: false);
        var distance = 0.0;
        
        for (int i = 0; i < action.Length; i++)
        {
            distance += Math.Pow(action[i] - predictedAction[i], NeuralNetworkConstants.RANDOM_VALUE_MULTIPLIER);
        }
        
        return -distance; // Negative distance as log probability
    }

    public void UpdateWeights(double[] state, double loss)
    {
        // Simplified gradient descent update
        // In a full implementation, this would use proper backpropagation
        var gradient = loss * _learningRate;
        
        // Update output biases (simplified)
        for (int i = 0; i < _outputDim; i++)
        {
            _biasOutput[i] -= gradient * NeuralNetworkConstants.EXPLORATION_NOISE_FACTOR;
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _rng?.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Simple neural network for critic (Q-function)
/// </summary>
public class CriticNetwork
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _hiddenDim;
    private readonly double _learningRate;
    private readonly System.Security.Cryptography.RandomNumberGenerator _rng;
    
    // Network weights (simplified implementation)
    private double[][] _weightsInput = null!;
    private double[] _biasHidden = null!;
    private double[][] _weightsOutput = null!;
    private double[] _biasOutput = null!;

    public CriticNetwork(int inputDim, int outputDim, int hiddenDim, double learningRate)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _hiddenDim = hiddenDim;
        _learningRate = learningRate;
        _rng = System.Security.Cryptography.RandomNumberGenerator.Create();
        
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        // Xavier initialization
        var scale = Math.Sqrt(NeuralNetworkConstants.XAVIER_FACTOR / _inputDim);
        
        _weightsInput = new double[_inputDim][];
        for (int i = 0; i < _inputDim; i++)
        {
            _weightsInput[i] = new double[_hiddenDim];
        }
        _biasHidden = new double[_hiddenDim];
        _weightsOutput = new double[_hiddenDim][];
        for (int i = 0; i < _hiddenDim; i++)
        {
            _weightsOutput[i] = new double[_outputDim];
        }
        _biasOutput = new double[_outputDim];
        
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _weightsInput[i][j] = (GetRandomDouble() * NeuralNetworkConstants.WEIGHT_INIT_RANGE - 1) * scale;
            }
        }
        
        scale = Math.Sqrt(NeuralNetworkConstants.XAVIER_FACTOR / _hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                _weightsOutput[i][j] = (GetRandomDouble() * NeuralNetworkConstants.WEIGHT_INIT_RANGE - 1) * scale;
            }
        }
    }

    public double Predict(double[] input)
    {
        ArgumentNullException.ThrowIfNull(input);
        // Forward pass
        var hidden = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            hidden[i] = _biasHidden[i];
            for (int j = 0; j < _inputDim; j++)
            {
                hidden[i] += input[j] * _weightsInput[j][i];
            }
            hidden[i] = Math.Max(0, hidden[i]); // ReLU activation
        }
        
        var output = 0.0;
        for (int j = 0; j < _hiddenDim; j++)
        {
            output += hidden[j] * _weightsOutput[j][0];
        }
        output += _biasOutput[0];
        
        return output;
    }

    public void UpdateWeights(double[] input, double target)
    {
        // Simplified gradient descent update
        var prediction = Predict(input);
        var error = target - prediction;
        var gradient = error * _learningRate;
        
        // Update output bias (simplified)
        _biasOutput[0] += gradient;
    }

    public void CopyWeightsFrom(CriticNetwork source)
    {
        ArgumentNullException.ThrowIfNull(source);
        // Copy weights from source network
        Array.Copy(source._weightsInput, _weightsInput, source._weightsInput.Length);
        Array.Copy(source._biasHidden, _biasHidden, source._biasHidden.Length);
        Array.Copy(source._weightsOutput, _weightsOutput, source._weightsOutput.Length);
        Array.Copy(source._biasOutput, _biasOutput, source._biasOutput.Length);
    }

    public void SoftUpdate(CriticNetwork source, double tau)
    {
        ArgumentNullException.ThrowIfNull(source);
        // Soft update: target = tau * source + (1 - tau) * target
        for (int i = 0; i < _weightsInput.GetLength(0); i++)
        {
            for (int j = 0; j < _weightsInput.GetLength(1); j++)
            {
                _weightsInput[i][j] = tau * source._weightsInput[i][j] + (1 - tau) * _weightsInput[i][j];
            }
        }
        
        for (int i = 0; i < _biasHidden.Length; i++)
        {
            _biasHidden[i] = tau * source._biasHidden[i] + (1 - tau) * _biasHidden[i];
        }
        
        for (int i = 0; i < _weightsOutput.GetLength(0); i++)
        {
            for (int j = 0; j < _weightsOutput.GetLength(1); j++)
            {
                _weightsOutput[i][j] = tau * source._weightsOutput[i][j] + (1 - tau) * _weightsOutput[i][j];
            }
        }
        
        for (int i = 0; i < _biasOutput.Length; i++)
        {
            _biasOutput[i] = tau * source._biasOutput[i] + (1 - tau) * _biasOutput[i];
        }
    }

    private double GetRandomDouble()
    {
        var bytes = new byte[8];
        _rng.GetBytes(bytes);
        return Math.Abs(BitConverter.ToDouble(bytes, 0)) / double.MaxValue;
    }
}

/// <summary>
/// Simple neural network for value function
/// </summary>
public class ValueNetwork
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _hiddenDim;
    private readonly double _learningRate;
    private readonly System.Security.Cryptography.RandomNumberGenerator _rng;
    
    // Network weights (simplified implementation)
    private double[][] _weightsInput = null!;
    private double[] _biasHidden = null!;
    private double[][] _weightsOutput = null!;
    private double[] _biasOutput = null!;

    public ValueNetwork(int inputDim, int outputDim, int hiddenDim, double learningRate)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _hiddenDim = hiddenDim;
        _learningRate = learningRate;
        _rng = System.Security.Cryptography.RandomNumberGenerator.Create();
        
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        // Xavier initialization
        var scale = Math.Sqrt(NeuralNetworkConstants.XAVIER_FACTOR / _inputDim);
        
        _weightsInput = new double[_inputDim][];
        for (int i = 0; i < _inputDim; i++)
        {
            _weightsInput[i] = new double[_hiddenDim];
        }
        _biasHidden = new double[_hiddenDim];
        _weightsOutput = new double[_hiddenDim][];
        for (int i = 0; i < _hiddenDim; i++)
        {
            _weightsOutput[i] = new double[_outputDim];
        }
        _biasOutput = new double[_outputDim];
        
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _weightsInput[i][j] = (GetRandomDouble() * NeuralNetworkConstants.WEIGHT_INIT_RANGE - 1) * scale;
            }
        }
        
        scale = Math.Sqrt(NeuralNetworkConstants.XAVIER_FACTOR / _hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                _weightsOutput[i][j] = (GetRandomDouble() * NeuralNetworkConstants.WEIGHT_INIT_RANGE - 1) * scale;
            }
        }
    }

    public double Predict(double[] input)
    {
        ArgumentNullException.ThrowIfNull(input);
        // Forward pass
        var hidden = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            hidden[i] = _biasHidden[i];
            for (int j = 0; j < _inputDim; j++)
            {
                hidden[i] += input[j] * _weightsInput[j][i];
            }
            hidden[i] = Math.Max(0, hidden[i]); // ReLU activation
        }
        
        var output = 0.0;
        for (int j = 0; j < _hiddenDim; j++)
        {
            output += hidden[j] * _weightsOutput[j][0];
        }
        output += _biasOutput[0];
        
        return output;
    }

    public void UpdateWeights(double[] input, double target)
    {
        // Simplified gradient descent update
        var prediction = Predict(input);
        var error = target - prediction;
        var gradient = error * _learningRate;
        
        // Update output bias (simplified)
        _biasOutput[0] += gradient;
    }

    private double GetRandomDouble()
    {
        var bytes = new byte[8];
        _rng.GetBytes(bytes);
        return Math.Abs(BitConverter.ToDouble(bytes, 0)) / double.MaxValue;
    }
}

#endregion