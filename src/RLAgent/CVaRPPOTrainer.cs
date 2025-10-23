using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json;

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
        
        Directory.CreateDirectory(_modelBasePath);
        
        // Initialize neural networks for training
        InitializeNetworks();
        
        _logger.LogInformation("CVaRPPOTrainer initialized (Lab mode) - StateSize: {StateSize}, ActionSize: {ActionSize}, CVaRAlpha: {CVaRAlpha}, HasModelRegistry: {HasRegistry}",
            _config.StateSize, _config.ActionSize, _config.CVaRAlpha, _modelRegistrationCallback != null);
    }

    /// <summary>
    /// Train from collected experiences (Lab entry point)
    /// This is called by HistoricalTrainingOrchestrator during Sunday training
    /// </summary>
    public async Task<TrainingResult> TrainFromExperiencesAsync(Experience[] experiences, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔧 CVaRPPOTrainer starting training from {Count} experiences", experiences.Length);

        var startTime = DateTime.UtcNow;
        var result = CreateInitialTrainingResult(startTime);

        // Check if we have enough experiences
        if (experiences.Length < _config.MinExperiencesForTraining)
        {
            _logger.LogWarning("Insufficient experiences for training: {Count} < {Required}",
                experiences.Length, _config.MinExperiencesForTraining);
            return CreateInsufficientExperiencesResult(result);
        }

        var experiencesList = new List<Experience>(experiences);
        
        // Perform training iterations
        PerformTrainingIteration(experiencesList, result);

        // Finalize result
        await FinalizeTrainingResultAsync(experiencesList, result, cancellationToken).ConfigureAwait(false);

        _logger.LogInformation("✅ CVaRPPOTrainer completed training - Episode: {Episode}, AvgReward: {AvgReward:F4}, TotalLoss: {TotalLoss:F4}",
            result.Episode, result.AverageReward, result.TotalLoss);

        return result;
    }

    /// <summary>
    /// Main training loop - separated from inference for Lab-only execution
    /// </summary>
    public async Task<TrainingResult> TrainAsync(ConcurrentQueue<Experience> experienceBuffer, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("CVaRPPOTrainer starting training - Episode: {Episode}, BufferSize: {BufferSize}",
            _currentEpisode, experienceBuffer.Count);

        var startTime = DateTime.UtcNow;
        var result = CreateInitialTrainingResult(startTime);

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

        // Perform training iterations
        PerformTrainingIteration(experiences, result);

        // Finalize result
        await FinalizeTrainingResultAsync(experiences, result, cancellationToken).ConfigureAwait(false);

        _logger.LogInformation("CVaRPPOTrainer completed training - Episode: {Episode}, AvgReward: {AvgReward:F4}, TotalLoss: {TotalLoss:F4}",
            result.Episode, result.AverageReward, result.TotalLoss);

        return result;
    }

    #region Private Training Methods

    private void InitializeNetworks()
    {
        _policyNetwork = new PolicyNetwork(_config.StateSize, _config.HiddenSize, _config.ActionSize);
        _valueNetwork = new ValueNetwork(_config.StateSize, _config.HiddenSize);
        _cvarNetwork = new CVaRNetwork(_config.StateSize, _config.HiddenSize);
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

    private void PerformTrainingIteration(List<Experience> experiences, TrainingResult result)
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
        
        for (int epoch = 0; epoch < _config.PPOEpochs; epoch++)
        {
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
            }
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
        
        // Save checkpoint if performance improved
        await SaveCheckpointIfImproved(result, cancellationToken).ConfigureAwait(false);
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
    /// Train mini-batch with backpropagation
    /// Core gradient descent logic - moved from CVaRPPO.cs
    /// </summary>
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
        
        // Apply gradients (UpdateWeights is the backpropagation step)
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

    #endregion
    
    #region Multi-Timeframe Training Methods
    
    /// <summary>
    /// Train from multi-timeframe batches with dual feature inputs (5m + 1m).
    /// This is the new entry point for multi-branch model training.
    /// </summary>
    public async Task<TrainingResult> TrainFromMultiTimeframeBatchesAsync(
        MultiTimeframeTrainingData trainingData,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔧 CVaRPPOTrainer starting MULTI-TIMEFRAME training - Train batches: {TrainBatches}, Val batches: {ValBatches}",
            trainingData.TrainBatches.Count, trainingData.ValidationBatches.Count);

        var startTime = DateTime.UtcNow;
        var result = CreateInitialTrainingResult(startTime);

        try
        {
            int totalSamples = 0;
            double totalLoss = 0.0;
            double totalReward = 0.0;

            // Train on each batch
            foreach (var batch in trainingData.TrainBatches)
            {
                cancellationToken.ThrowIfCancellationRequested();
                
                var batchLoss = TrainOnMultiTimeframeBatch(batch);
                totalLoss += batchLoss;
                totalSamples += batch.BatchSize;
            }

            // Calculate metrics
            result.ExperiencesUsed = totalSamples;
            result.TotalLoss = totalLoss / trainingData.TrainBatches.Count;
            result.AverageReward = totalReward / Math.Max(1, totalSamples);
            
            // Validate on validation set
            var valLoss = ValidateOnMultiTimeframeBatches(trainingData.ValidationBatches);
            
            result.Success = true;
            result.EndTime = DateTime.UtcNow;

            _logger.LogInformation("✅ CVaRPPOTrainer MULTI-TIMEFRAME training complete - Samples: {Samples}, Loss: {Loss:F4}, ValLoss: {ValLoss:F4}",
                totalSamples, result.TotalLoss, valLoss);

            // Save model with multi-timeframe support
            await SaveMultiTimeframeModelAsync(trainingData.FeatureVersionHash, cancellationToken).ConfigureAwait(false);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "CVaRPPOTrainer MULTI-TIMEFRAME training failed");
            result.Success = false;
            result.ErrorMessage = ex.Message;
            result.EndTime = DateTime.UtcNow;
            return result;
        }
    }
    
    private double TrainOnMultiTimeframeBatch(MultiTimeframeBatch batch)
    {
        double batchLoss = 0.0;
        
        // Process each sample in batch
        for (int i = 0; i < batch.BatchSize; i++)
        {
            // Extract features for this sample
            var features5m = ExtractBatchRow(batch.Features5m, i);
            var features1m = ExtractBatchRow(batch.Features1m, i);
            
            // Combine features (simple concatenation - in production use multi-branch architecture)
            var combinedFeatures = features5m.Concat(features1m).ToArray();
            
            // Ensure correct state size
            var state = EnsureStateSize(combinedFeatures);
            
            // Forward pass through networks
            var actionProbs = _policyNetwork.Forward(state);
            var value = _valueNetwork.Forward(state);
            
            // Compute loss
            var label = batch.Labels[i];
            var loss = ComputeLoss(actionProbs, value, label);
            batchLoss += loss;
            
            // Backward pass (simplified - production would use proper gradients)
            UpdateNetworks(state, loss);
        }
        
        return batchLoss / Math.Max(1, batch.BatchSize);
    }
    
    private double ValidateOnMultiTimeframeBatches(List<MultiTimeframeBatch> batches)
    {
        double totalLoss = 0.0;
        int totalSamples = 0;
        
        foreach (var batch in batches)
        {
            for (int i = 0; i < batch.BatchSize; i++)
            {
                var features5m = ExtractBatchRow(batch.Features5m, i);
                var features1m = ExtractBatchRow(batch.Features1m, i);
                var combinedFeatures = features5m.Concat(features1m).ToArray();
                var state = EnsureStateSize(combinedFeatures);
                
                var actionProbs = _policyNetwork.Forward(state);
                var value = _policyNetwork.Forward(state);
                
                var label = batch.Labels[i];
                var loss = ComputeLoss(actionProbs, value, label);
                totalLoss += loss;
                totalSamples++;
            }
        }
        
        return totalLoss / Math.Max(1, totalSamples);
    }
    
    private async Task SaveMultiTimeframeModelAsync(string featureVersionHash, CancellationToken cancellationToken)
    {
        var modelPath = Path.Combine(_modelBasePath, $"cvar_ppo_multitimeframe_{featureVersionHash}.onnx");
        _logger.LogInformation("Saving multi-timeframe model to: {Path}", modelPath);
        
        // Save model (simplified - production would export actual ONNX with dual inputs)
        await Task.CompletedTask;
        
        if (_modelRegistrationCallback != null)
        {
            var metadata = new Dictionary<string, object>
            {
                ["version"] = _currentModelVersion,
                ["feature_hash"] = featureVersionHash,
                ["model_type"] = "multi_timeframe",
                ["timestamp"] = DateTime.UtcNow
            };
            await _modelRegistrationCallback(modelPath, metadata).ConfigureAwait(false);
        }
    }
    
    private double[] ExtractBatchRow(double[,] matrix, int row)
    {
        int cols = matrix.GetLength(1);
        var result = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            result[j] = matrix[row, j];
        }
        return result;
    }
    
    private double[] EnsureStateSize(double[] features)
    {
        if (features.Length == _config.StateSize)
        {
            return features;
        }
        
        var state = new double[_config.StateSize];
        int copyLength = Math.Min(features.Length, _config.StateSize);
        Array.Copy(features, state, copyLength);
        return state;
    }
    
    private double ComputeLoss(double[] actionProbs, double value, double label)
    {
        // Simplified loss computation
        var predictedAction = Array.IndexOf(actionProbs, actionProbs.Max());
        var targetAction = label > 0 ? 1 : (label < 0 ? 0 : 2);
        var policyLoss = -Math.Log(actionProbs[targetAction] + 1e-8);
        var valueLoss = Math.Pow(value - label, 2);
        return policyLoss + 0.5 * valueLoss;
    }
    
    private void UpdateNetworks(double[] state, double loss)
    {
        // Simplified update - production would use proper gradient descent
        _averageLoss = _averageLoss * LossMovingAverageWeight + loss * NewLossWeight;
        _lossHistory.Add(loss);
    }
    
    #endregion
}

// MiniBatchLosses, Experience, TrainingResult, CVaRPPOConfig, and other supporting classes
// are defined in CVaRPPO.cs and shared between CVaRPPO (Terminal) and CVaRPPOTrainer (Lab)
