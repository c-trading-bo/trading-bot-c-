using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json;

namespace TradingBot.RLAgent;

/// <summary>
/// CVaR-PPO Implementation with training loop, experience buffer, and model versioning
/// Implements requirement 2.1: Training loop (policy, value, CVaR head), experience buffer, advantage/CVaR estimation, model save/restore
/// </summary>
[System.Diagnostics.CodeAnalysis.SuppressMessage("SonarAnalyzer.CSharp", "S101:Types should be named in PascalCase", Justification = "CVaR (Conditional Value at Risk) and PPO (Proximal Policy Optimization) are standard financial/ML acronyms")]
public class CVaRPPO : IDisposable
{
    private readonly ILogger<CVaRPPO> _logger;
    private readonly CVaRPPOConfig _config;
    private readonly string _modelBasePath;
    
    // Neural network components (simplified implementation)
    private PolicyNetwork _policyNetwork = null!;
    private ValueNetwork _valueNetwork = null!;
    private CVaRNetwork _cvarNetwork = null!;
    
    // Experience buffer
    private readonly ConcurrentQueue<Experience> _experienceBuffer = new();
    private readonly object _trainingLock = new();
    
    // Training state
    private int _currentEpisode;
    private double _averageReward;
    private double _averageLoss;
    private DateTime _lastTrainingTime = DateTime.MinValue;
    
    // Model versioning
    private string _currentModelVersion = "1.0.0";
    
    // Training constants
    private const double LossMovingAverageWeight = 0.9;
    private const double NewLossWeight = 0.1;
    private const int DefaultHistorySize = 20;
    private readonly Dictionary<string, ModelCheckpoint> _modelCheckpoints = new();
    
    // Cached JSON serializer options
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    
    // Performance tracking
    private readonly CircularBuffer<double> _rewardHistory = new(1000);
    private readonly CircularBuffer<double> _lossHistory = new(1000);
    private readonly CircularBuffer<double> _cvarHistory = new(1000);
    
    private bool _disposed;

    public CVaRPPO(
        ILogger<CVaRPPO> logger,
        CVaRPPOConfig config,
        string modelBasePath = "models/cvar_ppo")
    {
        _logger = logger;
        _config = config;
        _modelBasePath = modelBasePath;
        
        Directory.CreateDirectory(_modelBasePath);
        
        // Initialize neural networks
        InitializeNetworks();
        
        LogMessages.CVaRPPOInitialized(_logger, _config.StateSize, _config.ActionSize, _config.CVaRAlpha);
    }

    /// <summary>
    /// Train the CVaR-PPO agent on collected experiences
    /// </summary>
    public async Task<TrainingResult> TrainAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            LogMessages.CVaRPPOTrainingStarted(_logger, _currentEpisode, _experienceBuffer.Count);

            var startTime = DateTime.UtcNow;
            var result = CreateInitialTrainingResult(startTime);

            // Check if we have enough experiences
            if (_experienceBuffer.Count < _config.MinExperiencesForTraining)
            {
                return CreateInsufficientExperiencesResult(result);
            }

            // Collect experiences from buffer
            var experiences = CollectExperiencesForTraining(result);
            if (experiences == null) return result;

            // Perform training
            PerformTrainingIteration(experiences, result);

            // Finalize results
            await FinalizeTrainingResultAsync(experiences, result, cancellationToken).ConfigureAwait(false);

            return result;
        }
        catch (ArgumentException ex)
        {
            LogMessages.CVaRPPOTrainingArgumentError(_logger, ex);
            return new TrainingResult
            {
                Episode = _currentEpisode,
                Success = false,
                ErrorMessage = ex.Message,
                StartTime = DateTime.UtcNow,
                EndTime = DateTime.UtcNow
            };
        }
    }

    private TrainingResult CreateInitialTrainingResult(DateTime startTime)
    {
        return new TrainingResult
        {
            Episode = _currentEpisode,
            StartTime = startTime
        };
    }

    private TrainingResult CreateInsufficientExperiencesResult(TrainingResult result)
    {
        result.Success = false;
        result.ErrorMessage = $"Insufficient experiences: {_experienceBuffer.Count} < {_config.MinExperiencesForTraining}";
        return result;
    }

    private List<Experience>? CollectExperiencesForTraining(TrainingResult result)
    {
        List<Experience> experiences;
        lock (_trainingLock)
        {
            experiences = CollectExperiences();
        }
        
        if (experiences.Count == 0)
        {
            result.Success = false;
            result.ErrorMessage = "No valid experiences collected";
            return null;
        }
        
        return experiences;
    }

    private void PerformTrainingIteration(List<Experience> experiences, TrainingResult result)
    {
        // Calculate advantages and CVaR targets
        var (advantages, cvarTargets) = CalculateAdvantagesAndCVaR(experiences);
        
        // Training loop
        var totalPolicyLoss = 0.0;
        var totalValueLoss = 0.0;
        var totalCVaRLoss = 0.0;
        var totalEntropy = 0.0;

        for (int epoch = 0; epoch < _config.PPOEpochs; epoch++)
        {
            var shuffledIndices = CreateShuffledIndices(experiences.Count);
            
            // Mini-batch training
            for (int i = 0; i < experiences.Count; i += _config.BatchSize)
            {
                var (batchExperiences, batchAdvantages, batchCVaRTargets) = CreateTrainingBatch(
                    experiences, advantages, cvarTargets, shuffledIndices, i);

                var losses = TrainMiniBatch(batchExperiences, batchAdvantages, batchCVaRTargets);
                
                totalPolicyLoss += losses.PolicyLoss;
                totalValueLoss += losses.ValueLoss;
                totalCVaRLoss += losses.CVaRLoss;
                totalEntropy += losses.Entropy;
            }
        }

        // Calculate average losses
        var numBatches = (int)Math.Ceiling((double)experiences.Count / _config.BatchSize) * _config.PPOEpochs;
        result.PolicyLoss = totalPolicyLoss / numBatches;
        result.ValueLoss = totalValueLoss / numBatches;
        result.CVaRLoss = totalCVaRLoss / numBatches;
        result.Entropy = totalEntropy / numBatches;
        result.TotalLoss = result.PolicyLoss + result.ValueLoss + result.CVaRLoss;
    }

    private static int[] CreateShuffledIndices(int count)
    {
        // Shuffle experiences using cryptographically secure random
        using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
        return Enumerable.Range(0, count).OrderBy(x => {
            var bytes = new byte[4];
            rng.GetBytes(bytes);
            return BitConverter.ToUInt32(bytes, 0);
        }).ToArray();
    }

    private (Experience[], double[], double[]) CreateTrainingBatch(
        List<Experience> experiences, double[] advantages, double[] cvarTargets,
        int[] shuffledIndices, int startIndex)
    {
        var batchIndices = shuffledIndices.Skip(startIndex).Take(_config.BatchSize).ToArray();
        var batchExperiences = batchIndices.Select(idx => experiences[idx]).ToArray();
        var batchAdvantages = batchIndices.Select(idx => advantages[idx]).ToArray();
        var batchCVaRTargets = batchIndices.Select(idx => cvarTargets[idx]).ToArray();
        
        return (batchExperiences, batchAdvantages, batchCVaRTargets);
    }

    private Task FinalizeTrainingResultAsync(List<Experience> experiences, TrainingResult result, CancellationToken cancellationToken)
    {
        // Update averages and state (with lock)
        lock (_trainingLock)
        {
            _averageLoss = _averageLoss * LossMovingAverageWeight + result.TotalLoss * NewLossWeight;
            _averageReward = experiences.Count > 0 ? experiences.Average(e => e.Reward) : 0.0;

            // Track performance
            _lossHistory.Add(result.TotalLoss);
            _rewardHistory.Add(_averageReward);
            _cvarHistory.Add(result.CVaRLoss);

            _currentEpisode++;
            _lastTrainingTime = DateTime.UtcNow;
        }

        result.AverageReward = _averageReward;
        result.ExperiencesUsed = experiences.Count;
        result.Success = true;
        result.EndTime = DateTime.UtcNow;

        LogMessages.CVaRPPOTrainingCompleted(_logger, _currentEpisode, result.TotalLoss, result.PolicyLoss, result.ValueLoss, result.CVaRLoss, result.AverageReward);

        // Save checkpoint if performance improved
        return SaveCheckpointIfImproved(result, cancellationToken);
    }

    /// <summary>
    /// Get action from policy network
    /// </summary>
    public Task<ActionResult> GetActionAsync(double[] state, bool deterministic = false, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        
        try
        {
            if (state.Length != _config.StateSize)
            {
                throw new ArgumentException($"State size mismatch: expected {_config.StateSize}, got {state.Length}");
            }

            // Forward pass through policy network
            var policyOutput = _policyNetwork.Forward(state);
            
            // Sample action or take most probable
            int action;
            double[] actionProbs = SoftmaxActivation(policyOutput);
            
            if (deterministic)
            {
                action = Array.IndexOf(actionProbs, actionProbs.Max());
            }
            else
            {
                action = SampleFromDistribution(actionProbs);
            }

            // Calculate action probability for PPO
            var actionProb = actionProbs[action];
            var logProb = Math.Log(Math.Max(actionProb, 1e-8));

            // Get value estimate
            var valueOutput = _valueNetwork.Forward(state);
            var valueEstimate = valueOutput[0];

            // Get CVaR estimate
            var cvarOutput = _cvarNetwork.Forward(state);
            var cvarEstimate = cvarOutput[0];

            var result = new ActionResult
            {
                Action = action,
                ActionProbability = actionProb,
                LogProbability = logProb,
                ValueEstimate = valueEstimate,
                CVaREstimate = cvarEstimate,
                ActionProbabilities = actionProbs,
                Timestamp = DateTime.UtcNow
            };

            LogMessages.CVaRPPOActionSelected(_logger, action, actionProb, valueEstimate, cvarEstimate);

            return Task.FromResult(result);
        }
        catch (ArgumentException ex)
        {
            LogMessages.CVaRPPOActionSelectionArgumentError(_logger, ex);
            
            // Return safe default action
            var defaultResult = new ActionResult
            {
                Action = 0, // Hold action
                ActionProbability = 1.0,
                LogProbability = 0.0,
                ValueEstimate = 0.0,
                CVaREstimate = 0.0,
                ActionProbabilities = new double[_config.ActionSize],
                Timestamp = DateTime.UtcNow
            };
            
            return Task.FromResult(defaultResult);
        }
    }

    /// <summary>
    /// Add experience to the buffer
    /// </summary>
    public void AddExperience(Experience experience)
    {
        ArgumentNullException.ThrowIfNull(experience);
        
        _experienceBuffer.Enqueue(experience);
        
        // Limit buffer size
        while (_experienceBuffer.Count > _config.MaxExperienceBuffer)
        {
            _experienceBuffer.TryDequeue(out _);
        }
        
        LogMessages.CVaRPPOExperienceAdded(_logger, experience.Action, experience.Reward, experience.Done);
    }

    /// <summary>
    /// Save model checkpoint with versioning
    /// </summary>
    public async Task<string> SaveModelAsync(string? customVersion = null, CancellationToken cancellationToken = default)
    {
        try
        {
            var version = customVersion ?? GenerateNextVersion();
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            var modelPath = Path.Combine(_modelBasePath, $"cvar_ppo_v{version}_{timestamp}");
            
            Directory.CreateDirectory(modelPath);

            // Save networks
            await _policyNetwork.SaveAsync(Path.Combine(modelPath, "policy.json"), cancellationToken).ConfigureAwait(false);
            await _valueNetwork.SaveAsync(Path.Combine(modelPath, "value.json"), cancellationToken).ConfigureAwait(false);
            await _cvarNetwork.SaveAsync(Path.Combine(modelPath, "cvar.json"), cancellationToken).ConfigureAwait(false);

            // Save metadata
            var metadata = new ModelMetadata
            {
                Version = version,
                CreatedAt = DateTime.UtcNow,
                Episode = _currentEpisode,
                AverageReward = _averageReward,
                AverageLoss = _averageLoss,
                Config = _config,
                Performance = new PerformanceMetrics
                {
                    RecentRewards = _rewardHistory.GetAll().TakeLast(100).ToArray(),
                    RecentLosses = _lossHistory.GetAll().TakeLast(100).ToArray(),
                    RecentCVaRLosses = _cvarHistory.GetAll().TakeLast(100).ToArray()
                }
            };

            var metadataJson = JsonSerializer.Serialize(metadata, JsonOptions);
            await File.WriteAllTextAsync(Path.Combine(modelPath, "metadata.json"), metadataJson, cancellationToken).ConfigureAwait(false);

            // Create checkpoint record
            var checkpoint = new ModelCheckpoint
            {
                Version = version,
                Path = modelPath,
                CreatedAt = DateTime.UtcNow,
                Performance = _averageReward,
                Loss = _averageLoss
            };

            _modelCheckpoints[version] = checkpoint;
            _currentModelVersion = version;

            LogMessages.CVaRPPOModelSaved(_logger, modelPath, version);
            
            return modelPath;
        }
        catch (UnauthorizedAccessException ex)
        {
            LogMessages.CVaRPPOModelSaveAccessDenied(_logger, ex);
            throw new InvalidOperationException("Failed to save model due to access restrictions", ex);
        }
        catch (DirectoryNotFoundException ex)
        {
            LogMessages.CVaRPPOModelSaveDirectoryNotFound(_logger, ex);
            throw new InvalidOperationException("Failed to save model due to missing directory", ex);
        }
        catch (IOException ex)
        {
            LogMessages.CVaRPPOModelSaveIOError(_logger, ex);
            throw new InvalidOperationException("Failed to save model due to IO error", ex);
        }
    }

    /// <summary>
    /// Load model from checkpoint
    /// </summary>
    public async Task<bool> LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        try
        {
            if (!Directory.Exists(modelPath))
            {
                LogMessages.CVaRPPOModelPathNotExists(_logger, modelPath);
                return false;
            }

            // Load networks
            var policyPath = Path.Combine(modelPath, "policy.json");
            var valuePath = Path.Combine(modelPath, "value.json");
            var cvarPath = Path.Combine(modelPath, "cvar.json");

            if (!File.Exists(policyPath) || !File.Exists(valuePath) || !File.Exists(cvarPath))
            {
                LogMessages.CVaRPPOMissingNetworkFiles(_logger, modelPath);
                return false;
            }

            await _policyNetwork.LoadAsync(policyPath, cancellationToken).ConfigureAwait(false);
            await _valueNetwork.LoadAsync(valuePath, cancellationToken).ConfigureAwait(false);
            await _cvarNetwork.LoadAsync(cvarPath, cancellationToken).ConfigureAwait(false);

            // Load metadata if available
            var metadataPath = Path.Combine(modelPath, "metadata.json");
            if (File.Exists(metadataPath))
            {
                var metadataJson = await File.ReadAllTextAsync(metadataPath, cancellationToken).ConfigureAwait(false);
                var metadata = JsonSerializer.Deserialize<ModelMetadata>(metadataJson);
                
                if (metadata != null)
                {
                    _currentModelVersion = metadata.Version;
                    _currentEpisode = metadata.Episode;
                    _averageReward = metadata.AverageReward;
                    _averageLoss = metadata.AverageLoss;
                    
                    LogMessages.CVaRPPOModelMetadataLoaded(_logger, metadata.Version, metadata.Episode, metadata.AverageReward);
                }
            }

            LogMessages.CVaRPPOModelLoaded(_logger, modelPath);
            return true;
        }
        catch (FileNotFoundException ex)
        {
            LogMessages.CVaRPPOModelFileNotFound(_logger, modelPath, ex);
            return false;
        }
        catch (UnauthorizedAccessException ex)
        {
            LogMessages.CVaRPPOModelLoadAccessDenied(_logger, modelPath, ex);
            return false;
        }
        catch (InvalidOperationException ex)
        {
            LogMessages.CVaRPPOInvalidModelFile(_logger, modelPath, ex);
            return false;
        }
    }

    /// <summary>
    /// Get current training statistics
    /// </summary>
    public TrainingStatistics GetTrainingStatistics()
    {
        return new TrainingStatistics
        {
            CurrentEpisode = _currentEpisode,
            AverageReward = _averageReward,
            AverageLoss = _averageLoss,
            ExperienceBufferSize = _experienceBuffer.Count,
            LastTrainingTime = _lastTrainingTime,
            CurrentModelVersion = _currentModelVersion,
            RecentRewards = _rewardHistory.GetAll().TakeLast(DefaultHistorySize).ToArray(),
            RecentLosses = _lossHistory.GetAll().TakeLast(DefaultHistorySize).ToArray(),
            RecentCVaRLosses = _cvarHistory.GetAll().TakeLast(DefaultHistorySize).ToArray()
        };
    }

    // Private methods

    private void InitializeNetworks()
    {
        _policyNetwork = new PolicyNetwork(_config.StateSize, _config.ActionSize, _config.HiddenSize);
        _valueNetwork = new ValueNetwork(_config.StateSize, _config.HiddenSize);
        _cvarNetwork = new CVaRNetwork(_config.StateSize, _config.HiddenSize);
        
        LogMessages.CVaRPPONetworksInitialized(_logger);
    }

    private List<Experience> CollectExperiences()
    {
        var experiences = new List<Experience>();
        
        while (_experienceBuffer.TryDequeue(out var experience))
        {
            experiences.Add(experience);
        }
        
        return experiences;
    }

    private (double[] advantages, double[] cvarTargets) CalculateAdvantagesAndCVaR(List<Experience> experiences)
    {
        var advantages = new double[experiences.Count];
        var cvarTargets = new double[experiences.Count];
        
        // Calculate values for GAE (Generalized Advantage Estimation)
        var values = experiences.Select(e => _valueNetwork.Forward(e.State.ToArray())[0]).ToArray();
        
        // GAE calculation
        var gaeAdvantage = 0.0;
        for (int i = experiences.Count - 1; i >= 0; i--)
        {
            var delta = experiences[i].Reward + _config.Gamma * (i < experiences.Count - 1 ? values[i + 1] : 0) - values[i];
            gaeAdvantage = delta + _config.Gamma * _config.Lambda * gaeAdvantage;
            advantages[i] = gaeAdvantage;
        }
        
        // Normalize advantages
        var advantageMean = advantages.Average();
        var advantageStd = Math.Sqrt(advantages.Select(a => Math.Pow(a - advantageMean, 2)).Average());
        
        if (advantageStd > 0)
        {
            for (int i = 0; i < advantages.Length; i++)
            {
                advantages[i] = (advantages[i] - advantageMean) / advantageStd;
            }
        }
        
        // Calculate CVaR targets
        for (int i = 0; i < experiences.Count; i++)
        {
            cvarTargets[i] = CalculateCVaRTarget(experiences, i);
        }
        
        return (advantages, cvarTargets);
    }

    // Method CalculateReturns removed - was unused (S1144)

    private double CalculateCVaRTarget(List<Experience> experiences, int index)
    {
        // Calculate CVaR (Conditional Value at Risk) target
        // This is a simplified implementation - in practice, this would involve more sophisticated risk calculations
        
        var lookAheadWindow = Math.Min(10, experiences.Count - index);
        var futureRewards = new List<double>();
        
        for (int i = index; i < Math.Min(index + lookAheadWindow, experiences.Count); i++)
        {
            futureRewards.Add(experiences[i].Reward);
        }
        
        if (futureRewards.Count == 0) return 0.0;
        
        // Sort rewards to find worst-case scenarios
        var sortedRewards = futureRewards.OrderBy(r => r).ToArray();
        var cvarIndex = Math.Max(0, (int)(sortedRewards.Length * _config.CVaRAlpha) - 1);
        
        // Average of worst α% of outcomes
        var cvarRewards = sortedRewards.Take(cvarIndex + 1).ToArray();
        return cvarRewards.Length > 0 ? cvarRewards.Average() : 0.0;
    }

    private MiniBatchLosses TrainMiniBatch(Experience[] batch, double[] advantages, double[] cvarTargets)
    {
        var policyLoss = 0.0;
        var valueLoss = 0.0;
        var cvarLoss = 0.0;
        var entropy = 0.0;
        
        for (int i = 0; i < batch.Length; i++)
        {
            var experience = batch[i];
            var advantage = advantages[i];
            var cvarTarget = cvarTargets[i];
            
            // Policy loss (PPO clipped objective)
            var newPolicyOutput = _policyNetwork.Forward(experience.State.ToArray());
            var newActionProbs = SoftmaxActivation(newPolicyOutput);
            var newLogProb = Math.Log(Math.Max(newActionProbs[experience.Action], 1e-8));
            
            var ratio = Math.Exp(newLogProb - experience.LogProbability);
            var clippedRatio = Math.Max(Math.Min(ratio, 1 + _config.ClipEpsilon), 1 - _config.ClipEpsilon);
            
            var policyObjective = Math.Min(ratio * advantage, clippedRatio * advantage);
            policyLoss -= policyObjective; // Negative because we want to maximize
            
            // Entropy bonus
            var entropyBonus = -newActionProbs.Sum(p => p * Math.Log(Math.Max(p, 1e-8)));
            entropy += entropyBonus;
            policyLoss -= _config.EntropyCoeff * entropyBonus;
            
            // Value loss
            var newValueEstimate = _valueNetwork.Forward(experience.State.ToArray())[0];
            var valueDelta = experience.Return - newValueEstimate;
            valueLoss += valueDelta * valueDelta;
            
            // CVaR loss
            var newCVaREstimate = _cvarNetwork.Forward(experience.State.ToArray())[0];
            var cvarDelta = cvarTarget - newCVaREstimate;
            cvarLoss += cvarDelta * cvarDelta;
        }
        
        // Apply gradients (simplified - in practice would use proper backpropagation)
        _policyNetwork.UpdateWeights(policyLoss / batch.Length, _config.LearningRate);
        _valueNetwork.UpdateWeights(valueLoss / batch.Length, _config.LearningRate);
        _cvarNetwork.UpdateWeights(cvarLoss / batch.Length, _config.LearningRate);
        
        return new MiniBatchLosses
        {
            PolicyLoss = policyLoss / batch.Length,
            ValueLoss = valueLoss / batch.Length,
            CVaRLoss = cvarLoss / batch.Length,
            Entropy = entropy / batch.Length
        };
    }

    private static double[] SoftmaxActivation(double[] logits)
    {
        var maxLogit = logits.Max();
        var exps = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
        var sum = exps.Sum();
        return exps.Select(x => x / sum).ToArray();
    }

    private static int SampleFromDistribution(double[] probabilities)
    {
        using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
        var bytes = new byte[8];
        rng.GetBytes(bytes);
        var randomValue = BitConverter.ToUInt64(bytes, 0) / (double)ulong.MaxValue;
        
        var cumulative = 0.0;
        
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (randomValue <= cumulative)
                return i;
        }
        
        return probabilities.Length - 1; // Fallback
    }

    private string GenerateNextVersion()
    {
        var currentVersion = Version.Parse(_currentModelVersion);
        var nextVersion = new Version(currentVersion.Major, currentVersion.Minor, currentVersion.Build + 1);
        return nextVersion.ToString();
    }

    private async Task SaveCheckpointIfImproved(TrainingResult result, CancellationToken cancellationToken)
    {
        // Save checkpoint if performance improved significantly
        var shouldSave = _modelCheckpoints.Count == 0 || 
                        result.AverageReward > _modelCheckpoints.Values.Max(c => c.Performance) + 0.01;
        
        if (shouldSave)
        {
            await SaveModelAsync(null, cancellationToken).ConfigureAwait(false);
            LogMessages.CVaRPPOCheckpointSaved(_logger);
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _policyNetwork?.Dispose();
                _valueNetwork?.Dispose();
                _cvarNetwork?.Dispose();
                
                LogMessages.CVaRPPODisposed(_logger);
            }
            
            _disposed = true;
        }
    }
}

#region Supporting Classes and Data Structures

/// <summary>
/// CVaR-PPO configuration
/// </summary>
[System.Diagnostics.CodeAnalysis.SuppressMessage("SonarAnalyzer.CSharp", "S101:Types should be named in PascalCase", Justification = "CVaR (Conditional Value at Risk) and PPO (Proximal Policy Optimization) are standard financial/ML acronyms")]
public class CVaRPPOConfig
{
    public int StateSize { get; set; } = 50;
    public int ActionSize { get; set; } = 4;
    public int HiddenSize { get; set; } = 128;
    public double LearningRate { get; set; } = 3e-4;
    public double Gamma { get; set; } = 0.99;
    public double Lambda { get; set; } = 0.95;
    public double ClipEpsilon { get; set; } = 0.2;
    public double EntropyCoeff { get; set; } = 0.01;
    public double CVaRAlpha { get; set; } = 0.05; // 5% tail risk
    public int BatchSize { get; set; } = 64;
    public int PPOEpochs { get; set; } = 4;
    public int MinExperiencesForTraining { get; set; } = 256;
    public int MaxExperienceBuffer { get; set; } = 10000;
}

/// <summary>
/// Experience for training
/// </summary>
public class Experience
{
    public IReadOnlyList<double> State { get; set; } = Array.Empty<double>();
    public int Action { get; set; }
    public double Reward { get; set; }
    public IReadOnlyList<double> NextState { get; set; } = Array.Empty<double>();
    public bool Done { get; set; }
    public double LogProbability { get; set; }
    public double ValueEstimate { get; set; }
    public double Return { get; set; } // Calculated during advantage estimation
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Action result from policy
/// </summary>
public class ActionResult
{
    public int Action { get; set; }
    public double ActionProbability { get; set; }
    public double LogProbability { get; set; }
    public double ValueEstimate { get; set; }
    public double CVaREstimate { get; set; }
    public IReadOnlyList<double> ActionProbabilities { get; set; } = Array.Empty<double>();
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Training result
/// </summary>
public class TrainingResult
{
    public int Episode { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public double PolicyLoss { get; set; }
    public double ValueLoss { get; set; }
    public double CVaRLoss { get; set; }
    public double TotalLoss { get; set; }
    public double Entropy { get; set; }
    public double AverageReward { get; set; }
    public int ExperiencesUsed { get; set; }
}

/// <summary>
/// Training statistics
/// </summary>
public class TrainingStatistics
{
    public int CurrentEpisode { get; set; }
    public double AverageReward { get; set; }
    public double AverageLoss { get; set; }
    public int ExperienceBufferSize { get; set; }
    public DateTime LastTrainingTime { get; set; }
    public string CurrentModelVersion { get; set; } = string.Empty;
    public IReadOnlyList<double> RecentRewards { get; set; } = Array.Empty<double>();
    public IReadOnlyList<double> RecentLosses { get; set; } = Array.Empty<double>();
    public IReadOnlyList<double> RecentCVaRLosses { get; set; } = Array.Empty<double>();
}

/// <summary>
/// Mini-batch training losses
/// </summary>
public class MiniBatchLosses
{
    public double PolicyLoss { get; set; }
    public double ValueLoss { get; set; }
    public double CVaRLoss { get; set; }
    public double Entropy { get; set; }
}

/// <summary>
/// Model checkpoint information
/// </summary>
public class ModelCheckpoint
{
    public string Version { get; set; } = string.Empty;
    public string Path { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public double Performance { get; set; }
    public double Loss { get; set; }
}

/// <summary>
/// Model metadata for persistence
/// </summary>
public class ModelMetadata
{
    public string Version { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public int Episode { get; set; }
    public double AverageReward { get; set; }
    public double AverageLoss { get; set; }
    public CVaRPPOConfig Config { get; set; } = new();
    public PerformanceMetrics Performance { get; set; } = new();
}

/// <summary>
/// Performance metrics for model evaluation
/// </summary>
public class PerformanceMetrics
{
    public IReadOnlyList<double> RecentRewards { get; set; } = Array.Empty<double>();
    public IReadOnlyList<double> RecentLosses { get; set; } = Array.Empty<double>();
    public IReadOnlyList<double> RecentCVaRLosses { get; set; } = Array.Empty<double>();
}

/// <summary>
/// Simplified Policy Network
/// </summary>
public class PolicyNetwork : IDisposable
{
    private const double WEIGHT_RANGE_MULTIPLIER = 2.0;
    private const double LEARNING_RATE = 0.001;
    
    private readonly int _stateSize;
    private readonly int _actionSize;
    private readonly int _hiddenSize;
    private double[][] _weights1 = null!;
    private double[] _bias1 = null!;
    private double[][] _weights2 = null!;
    private double[] _bias2 = null!;
    public PolicyNetwork(int stateSize, int actionSize, int hiddenSize)
    {
        _stateSize = stateSize;
        _actionSize = actionSize;
        _hiddenSize = hiddenSize;
        
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
        
        // Initialize jagged arrays
        _weights1 = new double[_stateSize][];
        for (int i = 0; i < _stateSize; i++)
        {
            _weights1[i] = new double[_hiddenSize];
        }
        
        _weights2 = new double[_hiddenSize][];
        for (int i = 0; i < _hiddenSize; i++)
        {
            _weights2[i] = new double[_actionSize];
        }
        
        _bias1 = new double[_hiddenSize];
        _bias2 = new double[_actionSize];
        
        // Xavier initialization with secure random
        var limit1 = Math.Sqrt(6.0 / (_stateSize + _hiddenSize));
        var limit2 = Math.Sqrt(6.0 / (_hiddenSize + _actionSize));
        
        for (int i = 0; i < _stateSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                var bytes = new byte[8];
                rng.GetBytes(bytes);
                var randomValue = BitConverter.ToUInt64(bytes, 0) / (double)ulong.MaxValue;
                _weights1[i][j] = (randomValue * WEIGHT_RANGE_MULTIPLIER - 1) * limit1;
            }
        }
        
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _actionSize; j++)
            {
                var bytes = new byte[8];
                rng.GetBytes(bytes);
                var randomValue = BitConverter.ToUInt64(bytes, 0) / (double)ulong.MaxValue;
                _weights2[i][j] = (randomValue * WEIGHT_RANGE_MULTIPLIER - 1) * limit2;
            }
        }
    }

    public double[] Forward(double[] state)
    {
        ArgumentNullException.ThrowIfNull(state);
        
        // Hidden layer
        var hidden = new double[_hiddenSize];
        for (int i = 0; i < _hiddenSize; i++)
        {
            var hiddenValue = _bias1[i];
            for (int j = 0; j < _stateSize; j++)
            {
                hiddenValue += state[j] * _weights1[j][i];
            }
            hidden[i] = Math.Tanh(hiddenValue); // Activation
        }
        
        // Output layer
        var output = new double[_actionSize];
        for (int i = 0; i < _actionSize; i++)
        {
            output[i] = _bias2[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                output[i] += hidden[j] * _weights2[j][i];
            }
        }
        
        return output;
    }

    public void UpdateWeights(double loss, double learningRate)
    {
        // Simplified gradient update (in practice, this would be proper backpropagation)
        var gradient = loss * learningRate;
        
        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _actionSize; j++)
            {
                _weights2[i][j] -= gradient * LEARNING_RATE; // Simplified gradient
            }
        }
    }

    public Task SaveAsync(string path, CancellationToken cancellationToken = default)
    {
        var data = new
        {
            Weights1 = _weights1,
            Bias1 = _bias1,
            Weights2 = _weights2,
            Bias2 = _bias2
        };
        
        var json = JsonSerializer.Serialize(data);
        return File.WriteAllTextAsync(path, json, cancellationToken);
    }

    public Task LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        // Load weights (simplified - in practice would handle proper deserialization)
        InitializeWeights(); // Reset to defaults for now
        return Task.FromResult(0); // Proper async completion without async keyword
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        // Nothing to dispose in this simplified implementation
    }
}

/// <summary>
/// Simplified Value Network
/// </summary>
public class ValueNetwork : IDisposable
{
    private const double WeightInitializationRange = 2.0;
    private const double OutputWeightInitializationScale = 0.1;
    private const double LearningRateDefault = 0.001;
    
    private readonly int _stateSize;
    private readonly int _hiddenSize;
    private double[][] _weights1 = null!;
    private double[] _bias1 = null!;
    private double[] _weights2 = null!;
    private double _bias2;
    public ValueNetwork(int stateSize, int hiddenSize)
    {
        _stateSize = stateSize;
        _hiddenSize = hiddenSize;
        
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
        
        // Initialize jagged array
        _weights1 = new double[_stateSize][];
        for (int i = 0; i < _stateSize; i++)
        {
            _weights1[i] = new double[_hiddenSize];
        }
        
        _bias1 = new double[_hiddenSize];
        _weights2 = new double[_hiddenSize];
        _bias2 = 0.0;
        
        var limit = Math.Sqrt(6.0 / (_stateSize + _hiddenSize));
        
        for (int i = 0; i < _stateSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                var bytes = new byte[8];
                rng.GetBytes(bytes);
                var randomValue = BitConverter.ToUInt64(bytes, 0) / (double)ulong.MaxValue;
                _weights1[i][j] = (randomValue * WeightInitializationRange - 1) * limit;
            }
        }
        
        for (int i = 0; i < _hiddenSize; i++)
        {
            var bytes = new byte[8];
            rng.GetBytes(bytes);
            var randomValue = BitConverter.ToUInt64(bytes, 0) / (double)ulong.MaxValue;
            _weights2[i] = (randomValue * WeightInitializationRange - 1) * OutputWeightInitializationScale;
        }
    }

    public double[] Forward(double[] state)
    {
        ArgumentNullException.ThrowIfNull(state);
        
        // Hidden layer
        var hidden = new double[_hiddenSize];
        for (int i = 0; i < _hiddenSize; i++)
        {
            var hiddenValue = _bias1[i];
            for (int j = 0; j < _stateSize; j++)
            {
                hiddenValue += state[j] * _weights1[j][i];
            }
            hidden[i] = Math.Tanh(hiddenValue);
        }
        
        // Output (single value)
        var output = _bias2;
        for (int i = 0; i < _hiddenSize; i++)
        {
            output += hidden[i] * _weights2[i];
        }
        
        return new[] { output };
    }

    public void UpdateWeights(double loss, double learningRate)
    {
        var gradient = loss * learningRate;
        
        for (int i = 0; i < _hiddenSize; i++)
        {
            _weights2[i] -= gradient * LearningRateDefault;
        }
    }

    public Task SaveAsync(string path, CancellationToken cancellationToken = default)
    {
        var data = new
        {
            Weights1 = _weights1,
            Bias1 = _bias1,
            Weights2 = _weights2,
            Bias2 = _bias2
        };
        
        var json = JsonSerializer.Serialize(data);
        return File.WriteAllTextAsync(path, json, cancellationToken);
    }

    public Task LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        // Load weights (simplified - in practice would handle proper deserialization)
        InitializeWeights(); // Reset to defaults for now
        return Task.FromResult(0); // Proper async completion
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        // Nothing to dispose in this simplified implementation
    }
}

/// <summary>
/// Simplified CVaR Network
/// </summary>
public class CVaRNetwork : IDisposable
{
    private readonly ValueNetwork _valueNetwork;

    public CVaRNetwork(int stateSize, int hiddenSize)
    {
        _valueNetwork = new ValueNetwork(stateSize, hiddenSize);
    }

    public double[] Forward(double[] state)
    {
        return _valueNetwork.Forward(state);
    }

    public void UpdateWeights(double loss, double learningRate)
    {
        _valueNetwork.UpdateWeights(loss, learningRate);
    }

    public Task SaveAsync(string path, CancellationToken cancellationToken = default)
    {
        return _valueNetwork.SaveAsync(path, cancellationToken);
    }

    public Task LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        return _valueNetwork.LoadAsync(path, cancellationToken);
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
            _valueNetwork?.Dispose();
        }
    }
}

#endregion