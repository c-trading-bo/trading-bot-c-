using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TradingBot.RLAgent.Algorithms;

/// <summary>
/// Soft Actor-Critic (SAC) - Advanced RL algorithm with continuous action spaces
/// Maximizes both reward and entropy for better exploration
/// More sample-efficient than PPO and better for continuous action spaces (position sizing)
/// </summary>
public class SoftActorCritic
{
    private readonly ILogger<SoftActorCritic> _logger;
    private readonly SACConfig _config;
    
    // SAC uses two critic networks (double Q-learning) and one actor network
    internal SACActorNetwork _actor = null!;
    internal SACCriticNetwork _critic1 = null!;
    internal SACCriticNetwork _critic2 = null!;
    internal SACCriticNetwork _targetCritic1 = null!;
    internal SACCriticNetwork _targetCritic2 = null!;
    
    // Temperature parameter for entropy regularization (learnable)
    private Tensor _logAlpha = null!;
    private double _targetEntropy;
    
    public SoftActorCritic(ILogger<SoftActorCritic> logger, SACConfig config)
    {
        _logger = logger;
        _config = config;
        
        // Target entropy is -dim(action_space) for continuous actions
        _targetEntropy = -_config.ActionDim;
        
        _logger.LogInformation("SoftActorCritic initialized - StateSize: {State}, ActionDim: {Action}, LR: {LR}",
            _config.StateSize, _config.ActionDim, _config.LearningRate);
    }
    
    public void InitializeNetworks()
    {
        _actor = new SACActorNetwork(_config.StateSize, _config.ActionDim, _config.HiddenSize);
        _critic1 = new SACCriticNetwork(_config.StateSize, _config.ActionDim, _config.HiddenSize);
        _critic2 = new SACCriticNetwork(_config.StateSize, _config.ActionDim, _config.HiddenSize);
        
        // Target networks (slowly updated copies)
        _targetCritic1 = new SACCriticNetwork(_config.StateSize, _config.ActionDim, _config.HiddenSize);
        _targetCritic2 = new SACCriticNetwork(_config.StateSize, _config.ActionDim, _config.HiddenSize);
        
        // Copy weights to target networks
        CopyWeights(_critic1, _targetCritic1);
        CopyWeights(_critic2, _targetCritic2);
        
        // Initialize temperature parameter
        _logAlpha = zeros(1, requires_grad: true);
    }
    
    private void CopyWeights(SACCriticNetwork source, SACCriticNetwork target)
    {
        var sourceParams = source.parameters().ToList();
        var targetParams = target.parameters().ToList();
        
        for (int i = 0; i < sourceParams.Count; i++)
        {
            targetParams[i].copy_(sourceParams[i]);
        }
    }
    
    public double[] GetAction(double[] state, bool deterministic = false)
    {
        using var stateTensor = tensor(state, ScalarType.Float32).reshape(1, -1);
        
        if (deterministic)
        {
            // Use mean action (no sampling)
            var (mean, _) = _actor.forward(stateTensor);
            using (mean)
            {
                return mean.data<float>().ToArray().Select(x => (double)x).ToArray();
            }
        }
        else
        {
            // Sample action with exploration
            var (action, logProb) = _actor.Sample(stateTensor);
            using (action)
            using (logProb)
            {
                return action.data<float>().ToArray().Select(x => (double)x).ToArray();
            }
        }
    }
}

/// <summary>
/// SAC Actor Network - outputs continuous actions with Gaussian distribution
/// </summary>
internal class SACActorNetwork : Module<Tensor, (Tensor, Tensor)>
{
    private readonly Module<Tensor, Tensor> _shared;
    private readonly Module<Tensor, Tensor> _mean;
    private readonly Module<Tensor, Tensor> _logStd;
    private readonly int _actionDim;
    
    public SACActorNetwork(int stateSize, int actionDim, int hiddenSize) : base("SACActorNetwork")
    {
        _actionDim = actionDim;
        
        // Shared layers
        _shared = Sequential(
            ("fc1", Linear(stateSize, hiddenSize)),
            ("relu1", ReLU()),
            ("fc2", Linear(hiddenSize, hiddenSize)),
            ("relu2", ReLU())
        );
        
        // Mean and log_std heads
        _mean = Linear(hiddenSize, actionDim);
        _logStd = Linear(hiddenSize, actionDim);
        
        RegisterComponents();
    }
    
    public override (Tensor, Tensor) forward(Tensor state)
    {
        using var shared = _shared.forward(state);
        var mean = _mean.forward(shared);
        var logStd = _logStd.forward(shared);
        
        // Clamp log_std to prevent numerical instability
        logStd = logStd.clamp(-20, 2);
        
        return (mean, logStd);
    }
    
    public (Tensor, Tensor) Sample(Tensor state)
    {
        var (mean, logStd) = forward(state);
        var std = logStd.exp();
        
        // Sample from Gaussian distribution
        var normal = torch.randn_like(mean);
        var action = mean + std * normal;
        
        // Squash action to [-1, 1] using tanh
        var squashedAction = action.tanh();
        
        // Compute log probability
        var logProb = -0.5 * (normal.pow(2) + 2 * logStd + Math.Log(2 * Math.PI));
        logProb = logProb.sum(dim: -1, keepdim: true);
        
        // Correction for tanh squashing
        var tanhCorrection = (1 - squashedAction.pow(2) + 1e-6).log().sum(dim: -1, keepdim: true);
        logProb = logProb - tanhCorrection;
        
        return (squashedAction, logProb);
    }
    
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _shared?.Dispose();
            _mean?.Dispose();
            _logStd?.Dispose();
        }
        base.Dispose(disposing);
    }
}

/// <summary>
/// SAC Critic Network - estimates Q-value for state-action pairs
/// </summary>
internal class SACCriticNetwork : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _network;
    
    public SACCriticNetwork(int stateSize, int actionDim, int hiddenSize) : base("SACCriticNetwork")
    {
        // Concatenate state and action as input
        _network = Sequential(
            ("fc1", Linear(stateSize + actionDim, hiddenSize)),
            ("relu1", ReLU()),
            ("fc2", Linear(hiddenSize, hiddenSize)),
            ("relu2", ReLU()),
            ("fc3", Linear(hiddenSize, 1))
        );
        
        RegisterComponents();
    }
    
    public override Tensor forward(Tensor input)
    {
        // Input should be concatenated [state, action]
        return _network.forward(input);
    }
    
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _network?.Dispose();
        }
        base.Dispose(disposing);
    }
}

/// <summary>
/// SAC Configuration
/// </summary>
public class SACConfig
{
    public int StateSize { get; set; } = 20;
    public int ActionDim { get; set; } = 1; // Continuous: position size 0-5 contracts
    public int HiddenSize { get; set; } = 256;
    public double LearningRate { get; set; } = 3e-4;
    public double Gamma { get; set; } = 0.99; // Discount factor
    public double Tau { get; set; } = 0.005; // Soft update coefficient for target networks
    public double Alpha { get; set; } = 0.2; // Initial temperature
    public int BatchSize { get; set; } = 256;
    public int BufferSize { get; set; } = 1000000;
}
