using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.ML;

namespace BotCore.Bandits;

/// <summary>
/// Neural UCB Bandit Trainer - Lab-only component for neural network retraining
/// Separated from NeuralUcbBandit.cs to keep Terminal lean (inference only)
/// 
/// TERMINAL (NeuralUcbBandit.cs): SelectArmAsync, UpdateArmStatisticsAsync (lightweight, milliseconds)
/// LAB (this class): RetrainNetworkAsync (heavy neural network training, 15 minutes)
/// 
/// This component runs ONLY in Lab mode during Sunday training sessions
/// </summary>
public class NeuralUcbBanditTrainer
{
    private readonly ILogger<NeuralUcbBanditTrainer> _logger;
    private readonly NeuralUcbConfig _config;
    
    public NeuralUcbBanditTrainer(
        ILogger<NeuralUcbBanditTrainer> logger,
        NeuralUcbConfig? config = null)
    {
        _logger = logger;
        _config = config ?? new NeuralUcbConfig();
        
        _logger.LogInformation("NeuralUcbBanditTrainer initialized (Lab mode) - MinSamples: {MinSamples}, RetrainingInterval: {Interval}",
            _config.MinSamplesForTraining, _config.RetrainingInterval);
    }

    /// <summary>
    /// Retrain neural network from collected experience data (Lab entry point)
    /// This is called by HistoricalTrainingOrchestrator during Sunday training
    /// </summary>
    public async Task<TrainingResult> RetrainNetworkAsync(
        INeuralNetwork network,
        List<(ContextVector context, decimal reward)> trainingData,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔧 NeuralUcbBanditTrainer starting retraining with {Count} samples", trainingData.Count);

        var startTime = DateTime.UtcNow;
        var result = new TrainingResult
        {
            StartTime = startTime,
            Success = false
        };

        // Validate sufficient data
        if (trainingData.Count < _config.MinSamplesForTraining)
        {
            _logger.LogWarning("Insufficient training data: {Count} < {Required}",
                trainingData.Count, _config.MinSamplesForTraining);
            result.ErrorMessage = $"Insufficient training data: {trainingData.Count} < {_config.MinSamplesForTraining}";
            result.EndTime = DateTime.UtcNow;
            return result;
        }

        try
        {
            _logger.LogInformation("Retraining neural network with {Count} samples", trainingData.Count);

            // Prepare training features and targets
            var features = trainingData.Select(d => d.context.ToArray(_config.InputDimension)).ToArray();
            var targets = trainingData.Select(d => d.reward).ToArray();

            // Train neural network (this is the heavy operation - 15 minutes)
            await network.TrainAsync(features, targets, cancellationToken).ConfigureAwait(false);

            result.Success = true;
            result.EndTime = DateTime.UtcNow;
            result.SampleCount = trainingData.Count;

            _logger.LogInformation("✅ NeuralUcbBanditTrainer completed retraining - Samples: {Count}, Duration: {Duration:F1}s",
                trainingData.Count, (result.EndTime.Value - result.StartTime).TotalSeconds);

            return result;
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "Invalid operation during neural network retraining");
            result.ErrorMessage = $"Invalid operation: {ex.Message}";
            result.EndTime = DateTime.UtcNow;
            return result;
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "Invalid arguments during neural network retraining");
            result.ErrorMessage = $"Invalid arguments: {ex.Message}";
            result.EndTime = DateTime.UtcNow;
            return result;
        }
        catch (OutOfMemoryException ex)
        {
            _logger.LogError(ex, "Out of memory during neural network retraining");
            result.ErrorMessage = $"Out of memory: {ex.Message}";
            result.EndTime = DateTime.UtcNow;
            return result;
        }
        catch (OperationCanceledException)
        {
            _logger.LogWarning("Neural network retraining was cancelled");
            result.ErrorMessage = "Retraining cancelled";
            result.EndTime = DateTime.UtcNow;
            return result;
        }
    }

    /// <summary>
    /// Batch retrain multiple arms (used during Lab training)
    /// </summary>
    public async Task<Dictionary<string, TrainingResult>> RetrainMultipleArmsAsync(
        Dictionary<string, (INeuralNetwork network, List<(ContextVector context, decimal reward)> data)> armsData,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔧 NeuralUcbBanditTrainer retraining {Count} arms", armsData.Count);

        var results = new Dictionary<string, TrainingResult>();

        foreach (var (armId, (network, data)) in armsData)
        {
            _logger.LogInformation("Retraining arm: {ArmId}", armId);
            var result = await RetrainNetworkAsync(network, data, cancellationToken).ConfigureAwait(false);
            results[armId] = result;
        }

        var successCount = results.Values.Count(r => r.Success);
        _logger.LogInformation("✅ NeuralUcbBanditTrainer completed batch retraining - {Success}/{Total} arms successful",
            successCount, armsData.Count);

        return results;
    }

    /// <summary>
    /// Training result for neural UCB retraining
    /// </summary>
    public class TrainingResult
    {
        public DateTime StartTime { get; set; }
        public DateTime? EndTime { get; set; }
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
        public int SampleCount { get; set; }
    }
}
