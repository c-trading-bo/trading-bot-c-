using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace TradingBot.RLAgent;

/// <summary>
/// CVaR-PPO Trainer - Lab-only component for model training
/// Separated from CVaRPPO.cs to keep Terminal lean (inference only)
/// This component runs ONLY in Lab mode during Sunday training sessions
/// </summary>
[System.Diagnostics.CodeAnalysis.SuppressMessage("SonarAnalyzer.CSharp", "S101:Types should be named in PascalCase", Justification = "CVaR (Conditional Value at Risk) and PPO (Proximal Policy Optimization) are standard financial/ML acronyms")]
public class CVaRPPOTrainer
{
    private readonly ILogger<CVaRPPOTrainer> _logger;
    private readonly CVaRPPOConfig _config;
    private readonly string _modelBasePath;
    private readonly Func<string, Dictionary<string, object>, Task>? _modelRegistrationCallback;
    
    // Neural network components (for training)
    private PolicyNetwork _policyNetwork = null!;
    private ValueNetwork _valueNetwork = null!;
    private CVaRNetwork _cvarNetwork = null!;
    
    // TorchSharp optimizers for real backpropagation
    private TorchSharp.Modules.OptimizerHelper _policyOptimizer = null!;
    private TorchSharp.Modules.OptimizerHelper _valueOptimizer = null!;
    private TorchSharp.Modules.OptimizerHelper _cvarOptimizer = null!;
    
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
    private readonly Dictionary<string, ModelCheckpoint> _modelCheckpoints = new();
    
    // Cached JSON serializer options
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    
    // Performance tracking
    private readonly CircularBuffer<double> _rewardHistory = new(1000);
    private readonly CircularBuffer<double> _lossHistory = new(1000);
    private readonly CircularBuffer<double> _cvarHistory = new(1000);

    public CVaRPPOTrainer(
        ILogger<CVaRPPOTrainer> logger,
        CVaRPPOConfig config,
        string modelBasePath = "models/cvar_ppo",
        Func<string, Dictionary<string, object>, Task>? modelRegistrationCallback = null)
    {
        _logger = logger;
        _config = config;
        _modelBasePath = modelBasePath;
        _modelRegistrationCallback = modelRegistrationCallback;
        
        try
        {
            Directory.CreateDirectory(_modelBasePath);
            
            // Initialize neural networks for training
            InitializeNetworks();
            
            _logger.LogInformation("CVaRPPOTrainer initialized (Lab mode) - StateSize: {StateSize}, ActionSize: {ActionSize}, CVaRAlpha: {CVaRAlpha}, HasModelRegistry: {HasRegistry}",
                _config.StateSize, _config.ActionSize, _config.CVaRAlpha, _modelRegistrationCallback != null);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ [CVAR-PPO-TRAINER] Failed to initialize neural networks (TorchSharp may not be available). Training will be disabled.");
            // Continue without neural networks - training won't work but DI container will build
        }
    }

    /// <summary>
    /// Train from collected experiences (Lab entry point)
    /// This is called by HistoricalTrainingOrchestrator during Sunday training
    /// </summary>
    /// <param name="experiences">Training experiences</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <param name="progressCallback">Optional callback for reporting epoch progress (epoch, totalEpochs, loss)</param>
    public async Task<TrainingResult> TrainFromExperiencesAsync(
        Experience[] experiences, 
        CancellationToken cancellationToken = default,
        Action<int, int, double>? progressCallback = null)
    {
        _logger.LogInformation("🔧 CVaRPPOTrainer starting training from {Count} experiences", experiences.Length);

        var startTime = DateTime.UtcNow;
        var result = CreateInitialTrainingResult(startTime);

        // CRITICAL FIX: Check if neural networks are initialized
        if (_policyNetwork == null || _valueNetwork == null || _cvarNetwork == null)
        {
            _logger.LogError("❌ CVaRPPOTrainer: Neural networks not initialized. TorchSharp may not be available.");
            result.Success = false;
            result.ErrorMessage = "Neural networks not initialized - TorchSharp dependency missing or failed to load";
            result.EndTime = DateTime.UtcNow;
            return result;
        }

        // Check if we have enough experiences
        if (experiences.Length < _config.MinExperiencesForTraining)
        {
            _logger.LogWarning("Insufficient experiences for training: {Count} < {Required}",
                experiences.Length, _config.MinExperiencesForTraining);
            return CreateInsufficientExperiencesResult(result);
        }

        var experiencesList = new List<Experience>(experiences);
        
        try
        {
            // Perform training iterations
            PerformTrainingIteration(experiencesList, result, progressCallback);

            // Finalize result
            await FinalizeTrainingResultAsync(experiencesList, result, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("✅ CVaRPPOTrainer completed training - Episode: {Episode}, AvgReward: {AvgReward:F4}, TotalLoss: {TotalLoss:F4}",
                result.Episode, result.AverageReward, result.TotalLoss);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ CVaRPPOTrainer: Training failed with exception");
            result.Success = false;
            result.ErrorMessage = $"Training failed: {ex.Message}";
            result.EndTime = DateTime.UtcNow;
        }

        return result;
    }

    /// <summary>
    /// Main training loop - separated from inference for Lab-only execution
    /// </summary>
    /// <param name="experienceBuffer">Concurrent queue of experiences</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <param name="progressCallback">Optional callback for reporting epoch progress (epoch, totalEpochs, loss)</param>
    public async Task<TrainingResult> TrainAsync(
        ConcurrentQueue<Experience> experienceBuffer, 
        CancellationToken cancellationToken = default,
        Action<int, int, double>? progressCallback = null)
    {
        _logger.LogInformation("CVaRPPOTrainer starting training - Episode: {Episode}, BufferSize: {BufferSize}",
            _currentEpisode, experienceBuffer.Count);

        var startTime = DateTime.UtcNow;
        var result = CreateInitialTrainingResult(startTime);

        // CRITICAL FIX: Check if neural networks are initialized
        if (_policyNetwork == null || _valueNetwork == null || _cvarNetwork == null)
        {
            _logger.LogError("❌ CVaRPPOTrainer: Neural networks not initialized. TorchSharp may not be available.");
            result.Success = false;
            result.ErrorMessage = "Neural networks not initialized - TorchSharp dependency missing or failed to load";
            result.EndTime = DateTime.UtcNow;
            return result;
        }

        // Check if we have enough experiences
        if (experienceBuffer.Count < _config.MinExperiencesForTraining)
        {
            _logger.LogWarning("Insufficient experiences for training: {Count} < {Required}",
                experienceBuffer.Count, _config.MinExperiencesForTraining);
            return CreateInsufficientExperiencesResult(result);
        }

        // Collect experiences from buffer
        var experiences = CollectExperiencesFromBuffer(experienceBuffer, result);
        if (experiences == null)
        {
            return result;
        }

        try
        {
            // Perform training iterations
            PerformTrainingIteration(experiences, result, progressCallback);

            // Finalize result
            await FinalizeTrainingResultAsync(experiences, result, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("CVaRPPOTrainer completed training - Episode: {Episode}, AvgReward: {AvgReward:F4}, TotalLoss: {TotalLoss:F4}",
                result.Episode, result.AverageReward, result.TotalLoss);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ CVaRPPOTrainer: Training failed with exception");
            result.Success = false;
            result.ErrorMessage = $"Training failed: {ex.Message}";
            result.EndTime = DateTime.UtcNow;
        }

        return result;
    }

    #region Private Training Methods

    private void InitializeNetworks()
    {
        _policyNetwork = new PolicyNetwork(_config.StateSize, _config.HiddenSize, _config.ActionSize);
        _valueNetwork = new ValueNetwork(_config.StateSize, _config.HiddenSize);
        _cvarNetwork = new CVaRNetwork(_config.StateSize, _config.HiddenSize);
        
        // Create Adam optimizers for real backpropagation
        _policyOptimizer = Adam(_policyNetwork.parameters(), lr: _config.LearningRate);
        _valueOptimizer = Adam(_valueNetwork.parameters(), lr: _config.LearningRate);
        _cvarOptimizer = Adam(_cvarNetwork.parameters(), lr: _config.LearningRate);
    }

    private TrainingResult CreateInitialTrainingResult(DateTime startTime)
    {
        _currentEpisode++;
        
        return new TrainingResult
        {
            Episode = _currentEpisode,
            StartTime = startTime,
            Success = false
        };
    }

    private TrainingResult CreateInsufficientExperiencesResult(TrainingResult result)
    {
        result.Success = false;
        result.ErrorMessage = "Insufficient experiences for training";
        result.EndTime = DateTime.UtcNow;
        return result;
    }

    private List<Experience>? CollectExperiencesFromBuffer(ConcurrentQueue<Experience> experienceBuffer, TrainingResult result)
    {
        var experiences = new List<Experience>();
        
        // Drain experience buffer
        while (experienceBuffer.TryDequeue(out var experience))
        {
            experiences.Add(experience);
        }
        
        if (experiences.Count < _config.MinExperiencesForTraining)
        {
            _logger.LogWarning("Collected insufficient experiences: {Count} < {Required}",
                experiences.Count, _config.MinExperiencesForTraining);
            result.Success = false;
            result.ErrorMessage = "Insufficient experiences collected";
            result.EndTime = DateTime.UtcNow;
            return null;
        }
        
        return experiences;
    }

    private void PerformTrainingIteration(List<Experience> experiences, TrainingResult result, Action<int, int, double>? progressCallback = null)
    {
        // Calculate advantages and CVaR targets
        var (advantages, cvarTargets) = CalculateAdvantagesAndCVaR(experiences);
        
        // Update experience returns
        var returns = CalculateReturns(experiences);
        for (int i = 0; i < experiences.Count; i++)
        {
            experiences[i].Return = returns[i];
        }
        
        // Training loop (PPO multi-epoch)
        var totalPolicyLoss = 0.0;
        var totalValueLoss = 0.0;
        var totalCVaRLoss = 0.0;
        var totalEntropy = 0.0;
        var batchCount = 0;
        var epochLosses = new List<double>();
        
        for (int epoch = 0; epoch < _config.PPOEpochs; epoch++)
        {
            var epochPolicyLoss = 0.0;
            var epochValueLoss = 0.0;
            var epochCVaRLoss = 0.0;
            var epochBatchCount = 0;
            
            // Shuffle experiences for each epoch
            var shuffled = experiences.OrderBy(_ => Guid.NewGuid()).ToList();
            
            // Process in mini-batches
            for (int i = 0; i < shuffled.Count; i += _config.BatchSize)
            {
                var batchSize = Math.Min(_config.BatchSize, shuffled.Count - i);
                var (batch, batchAdvantages, batchCVaRTargets) = CreateTrainingBatch(
                    shuffled, advantages, cvarTargets, i, batchSize);
                
                var losses = TrainMiniBatch(batch, batchAdvantages, batchCVaRTargets);
                
                totalPolicyLoss += losses.PolicyLoss;
                totalValueLoss += losses.ValueLoss;
                totalCVaRLoss += losses.CVaRLoss;
                totalEntropy += losses.Entropy;
                batchCount++;
                
                epochPolicyLoss += losses.PolicyLoss;
                epochValueLoss += losses.ValueLoss;
                epochCVaRLoss += losses.CVaRLoss;
                epochBatchCount++;
            }
            
            // Calculate epoch average loss and report progress
            var epochAvgLoss = (epochPolicyLoss + epochValueLoss + epochCVaRLoss) / epochBatchCount;
            epochLosses.Add(epochAvgLoss);
            
            // Report progress if callback provided
            progressCallback?.Invoke(epoch + 1, _config.PPOEpochs, epochAvgLoss);
        }
        
        // Update training statistics
        var avgPolicyLoss = totalPolicyLoss / batchCount;
        var avgValueLoss = totalValueLoss / batchCount;
        var avgCVaRLoss = totalCVaRLoss / batchCount;
        var totalLoss = avgPolicyLoss + avgValueLoss + avgCVaRLoss;
        
        _averageLoss = LossMovingAverageWeight * _averageLoss + NewLossWeight * totalLoss;
        _averageReward = experiences.Average(e => e.Reward);
        
        // Update result
        result.PolicyLoss = avgPolicyLoss;
        result.ValueLoss = avgValueLoss;
        result.CVaRLoss = avgCVaRLoss;
        result.TotalLoss = totalLoss;
        result.AverageReward = _averageReward;
        result.Entropy = totalEntropy / batchCount;
        result.ExperiencesUsed = experiences.Count;
        
        // Track performance history
        _rewardHistory.Add(_averageReward);
        _lossHistory.Add(_averageLoss);
        _cvarHistory.Add(avgCVaRLoss);
    }

    private (Experience[], double[], double[]) CreateTrainingBatch(
        List<Experience> experiences, double[] advantages, double[] cvarTargets, int startIndex, int batchSize)
    {
        var batch = experiences.Skip(startIndex).Take(batchSize).ToArray();
        var batchAdvantages = advantages.Skip(startIndex).Take(batchSize).ToArray();
        var batchCVaRTargets = cvarTargets.Skip(startIndex).Take(batchSize).ToArray();
        return (batch, batchAdvantages, batchCVaRTargets);
    }

    private async Task FinalizeTrainingResultAsync(List<Experience> experiences, TrainingResult result, CancellationToken cancellationToken)
    {
        result.Success = true;
        result.EndTime = DateTime.UtcNow;
        _lastTrainingTime = result.EndTime.Value;
        
        // ALWAYS save model after training (not just when performance improves)
        // This ensures the orchestrator can verify training occurred
        try
        {
            var fullModelPath = await SaveModelAsync(null, cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("[CVaR-PPO] Model saved after training to: {Path}", fullModelPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[CVaR-PPO] Failed to save model, but training completed successfully");
            // Don't fail the whole training just because model save failed
        }
    }

    /// <summary>
    /// Calculate advantages using GAE (Generalized Advantage Estimation)
    /// Core training logic - moved from CVaRPPO.cs
    /// </summary>
    private (double[] advantages, double[] cvarTargets) CalculateAdvantagesAndCVaR(List<Experience> experiences)
    {
        var advantages = new double[experiences.Count];
        var cvarTargets = new double[experiences.Count];
        
        // Calculate values for GAE
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

    private double[] CalculateReturns(List<Experience> experiences)
    {
        var returns = new double[experiences.Count];
        var runningReturn = 0.0;
        
        for (int i = experiences.Count - 1; i >= 0; i--)
        {
            runningReturn = experiences[i].Reward + _config.Gamma * runningReturn;
            returns[i] = runningReturn;
        }
        
        return returns;
    }

    private double CalculateCVaRTarget(List<Experience> experiences, int index)
    {
        // Calculate CVaR (Conditional Value at Risk) target
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

    /// <summary>
    /// Train mini-batch with TorchSharp automatic differentiation and backpropagation
    /// Real gradient computation using chain rule - replaces uniform gradient updates
    /// </summary>
    private MiniBatchLosses TrainMiniBatch(Experience[] batch, double[] advantages, double[] cvarTargets)
    {
        // Convert batch to tensors for TorchSharp
        var states = new float[batch.Length, _config.StateSize];
        var actions = new long[batch.Length];
        var returns = new float[batch.Length];
        var advantageTensors = new float[batch.Length];
        var cvarTargetTensors = new float[batch.Length];
        var oldLogProbs = new float[batch.Length];
        
        for (int i = 0; i < batch.Length; i++)
        {
            var state = batch[i].State.ToArray();
            for (int j = 0; j < _config.StateSize; j++)
            {
                states[i, j] = (float)state[j];
            }
            actions[i] = batch[i].Action;
            returns[i] = (float)batch[i].Return;
            advantageTensors[i] = (float)advantages[i];
            cvarTargetTensors[i] = (float)cvarTargets[i];
            oldLogProbs[i] = (float)batch[i].LogProbability;
        }
        
        using var stateTensor = tensor(states);
        using var actionTensor = tensor(actions);
        using var returnTensor = tensor(returns).reshape(-1, 1);
        using var advantageTensor = tensor(advantageTensors).reshape(-1, 1);
        using var cvarTargetTensor = tensor(cvarTargetTensors).reshape(-1, 1);
        using var oldLogProbTensor = tensor(oldLogProbs);
        
        // ==== Policy Network Training ====
        _policyOptimizer.zero_grad();
        using var policyOutput = _policyNetwork.forward(stateTensor);
        using var logProbs = functional.log_softmax(policyOutput, dim: 1);
        using var newLogProbs = logProbs.gather(1, actionTensor.reshape(-1, 1)).reshape(-1);
        
        // PPO clipped objective
        using var ratio = (newLogProbs - oldLogProbTensor).exp();
        using var clippedRatio = ratio.clamp(1 - _config.ClipEpsilon, 1 + _config.ClipEpsilon);
        using var policyObjective1 = ratio * advantageTensor.reshape(-1);
        using var policyObjective2 = clippedRatio * advantageTensor.reshape(-1);
        using var policyObjective = torch.min(policyObjective1, policyObjective2);
        
        // Entropy bonus for exploration
        using var probs = functional.softmax(policyOutput, dim: 1);
        using var entropy = -(probs * logProbs).sum(dim: 1);
        using var policyLossTensor = -(policyObjective.mean() + _config.EntropyCoeff * entropy.mean());
        
        policyLossTensor.backward();
        _policyOptimizer.step();
        
        // ==== Value Network Training ====
        _valueOptimizer.zero_grad();
        using var valueOutput = _valueNetwork.forward(stateTensor);
        using var valueLossTensor = functional.mse_loss(valueOutput, returnTensor);
        
        valueLossTensor.backward();
        _valueOptimizer.step();
        
        // ==== CVaR Network Training ====
        _cvarOptimizer.zero_grad();
        using var cvarOutput = _cvarNetwork.forward(stateTensor);
        using var cvarLossTensor = functional.mse_loss(cvarOutput, cvarTargetTensor);
        
        cvarLossTensor.backward();
        _cvarOptimizer.step();
        
        // Extract scalar loss values for logging
        var policyLoss = policyLossTensor.ToDouble();
        var valueLoss = valueLossTensor.ToDouble();
        var cvarLoss = cvarLossTensor.ToDouble();
        var entropyValue = entropy.mean().ToDouble();
        
        return new MiniBatchLosses
        {
            PolicyLoss = policyLoss,
            ValueLoss = valueLoss,
            CVaRLoss = cvarLoss,
            Entropy = entropyValue
        };
    }

    /// <summary>
    /// Save trained model as challenger to Model Registry
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

            // Validate model files were created with substantial content
            var policyPath = Path.Combine(modelPath, "policy.json");
            var valuePath = Path.Combine(modelPath, "value.json");
            var cvarPath = Path.Combine(modelPath, "cvar.json");
            
            ValidateModelFile(policyPath, "PolicyNetwork");
            ValidateModelFile(valuePath, "ValueNetwork");
            ValidateModelFile(cvarPath, "CVaRNetwork");

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

            _logger.LogInformation("CVaRPPOTrainer saved model - Path: {Path}, Version: {Version}", modelPath, version);
            
            // Register model in Model Registry via callback
            if (_modelRegistrationCallback != null)
            {
                try
                {
                    var registrationMetadata = new Dictionary<string, object>
                    {
                        ["algorithm"] = "CVaR-PPO",
                        ["version"] = version,
                        ["timestamp"] = timestamp,
                        ["artifact_path"] = modelPath,
                        ["episode"] = _currentEpisode,
                        ["average_reward"] = _averageReward,
                        ["average_loss"] = _averageLoss,
                        ["model_type"] = "CVaR-PPO",
                        ["created_by"] = "CVaRPPOTrainer"
                    };
                    
                    await _modelRegistrationCallback($"cvar-ppo-v{version}-{timestamp}", registrationMetadata).ConfigureAwait(false);
                    _logger.LogInformation("CVaRPPOTrainer registered model in registry - Version: {Version}", version);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "CVaRPPOTrainer failed to register model in registry - Model saved but not tracked");
                }
            }
            else
            {
                _logger.LogWarning("CVaRPPOTrainer: Model registry callback not configured - Model saved but not tracked");
            }
            
            return modelPath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "CVaRPPOTrainer failed to save model");
            throw;
        }
    }

    public TrainingStatistics GetTrainingStatistics()
    {
        return new TrainingStatistics
        {
            CurrentEpisode = _currentEpisode,
            AverageReward = _averageReward,
            AverageLoss = _averageLoss,
            LastTrainingTime = _lastTrainingTime,
            ExperienceBufferSize = 0, // Not applicable in trainer
            CurrentModelVersion = _currentModelVersion,
            RecentRewards = _rewardHistory.GetAll().ToArray(),
            RecentLosses = _lossHistory.GetAll().ToArray(),
            RecentCVaRLosses = _cvarHistory.GetAll().ToArray()
        };
    }

    private async Task SaveCheckpointIfImproved(TrainingResult result, CancellationToken cancellationToken)
    {
        // Save checkpoint if performance improved significantly
        var shouldSave = _modelCheckpoints.Count == 0 || 
                        result.AverageReward > _modelCheckpoints.Values.Max(c => c.Performance) + 0.01;
        
        if (shouldSave)
        {
            await SaveModelAsync(null, cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("CVaRPPOTrainer saved checkpoint - Performance: {Performance:F4}", result.AverageReward);
        }
    }

    private string GenerateNextVersion()
    {
        var currentVersion = Version.Parse(_currentModelVersion);
        var nextVersion = new Version(currentVersion.Major, currentVersion.Minor, currentVersion.Build + 1);
        return nextVersion.ToString();
    }

    private static double[] SoftmaxActivation(double[] logits)
    {
        var maxLogit = logits.Max();
        var exps = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
        var sum = exps.Sum();
        return exps.Select(x => x / sum).ToArray();
    }
    
    private void ValidateModelFile(string path, string modelName)
    {
        if (!File.Exists(path))
        {
            throw new InvalidOperationException($"{modelName} model file was not created at: {path}. TorchSharp save may have failed silently.");
        }
        
        var fileInfo = new FileInfo(path);
        const long minExpectedSize = 1024; // Minimum 1KB - real PyTorch models should be much larger
        
        if (fileInfo.Length < minExpectedSize)
        {
            _logger.LogError("❌ {ModelName} model file is suspiciously small: {Size} bytes at {Path}. Expected at least {MinSize} bytes. " +
                "This indicates TorchSharp may have saved an empty/incomplete file or neural networks failed to initialize.",
                modelName, fileInfo.Length, path, minExpectedSize);
            throw new InvalidOperationException(
                $"{modelName} model file appears to be incomplete or empty ({fileInfo.Length} bytes). " +
                "Real trained models should be at least {minExpectedSize} bytes. " +
                "Check that TorchSharp native libraries are available and neural networks initialized correctly.");
        }
        
        _logger.LogDebug("✅ {ModelName} model file validated: {Size} bytes at {Path}", 
            modelName, fileInfo.Length, path);
    }

    #endregion
}

// MiniBatchLosses, Experience, TrainingResult, CVaRPPOConfig, and other supporting classes
// are defined in CVaRPPO.cs and shared between CVaRPPO (Terminal) and CVaRPPOTrainer (Lab)
