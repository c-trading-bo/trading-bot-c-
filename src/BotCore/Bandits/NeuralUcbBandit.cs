using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Bandits;

/// <summary>
/// Neural Upper Confidence Bound (NeuralUCB) bandit using neural network function approximation.
/// Uses deep learning to model complex non-linear relationships in high-dimensional contexts.
/// More powerful than LinUCB but requires more data and computation.
/// </summary>
public class NeuralUcbBandit : IFunctionApproximationBandit
{
    private readonly Dictionary<string, NeuralUcbArm> _arms = new();
    private readonly NeuralUcbConfig _config;
    private readonly INeuralNetwork _networkTemplate;
    private readonly object _lock = new();

    public NeuralUcbBandit(
        INeuralNetwork networkTemplate,
        NeuralUcbConfig? config = null)
    {
        _networkTemplate = networkTemplate;
        _config = config ?? new NeuralUcbConfig();
    }

    /// <summary>
    /// Selects arm using NeuralUCB algorithm with neural network predictions.
    /// </summary>
    public async Task<BanditSelection> SelectArmAsync(
        List<string> availableArms,
        ContextVector context,
        CancellationToken ct = default)
    {
        lock (_lock)
        {
            // Ensure all arms exist
            foreach (var armId in availableArms)
            {
                if (!_arms.ContainsKey(armId))
                {
                    _arms[armId] = new NeuralUcbArm(
                        _networkTemplate.Clone(),
                        _config);
                }
            }
        }

        var selections = new List<(string armId, decimal ucbValue, decimal prediction, decimal uncertainty)>();

        foreach (var armId in availableArms)
        {
            var arm = _arms[armId];
            var (prediction, uncertainty) = await arm.PredictWithUncertaintyAsync(context, ct);
            var ucbValue = prediction + _config.ExplorationWeight * uncertainty;

            selections.Add((armId, ucbValue, prediction, uncertainty));
        }

        // Select arm with highest UCB value
        var bestSelection = selections.OrderByDescending(s => s.ucbValue).First();

        Console.WriteLine($"[NEURAL-UCB] Selected {bestSelection.armId}: " +
                        $"pred={bestSelection.prediction:F3} unc={bestSelection.uncertainty:F3} " +
                        $"ucb={bestSelection.ucbValue:F3}");

        return new BanditSelection
        {
            SelectedArm = bestSelection.armId,
            Prediction = bestSelection.prediction,
            Confidence = 1m - bestSelection.uncertainty, // Convert uncertainty to confidence
            UcbValue = bestSelection.ucbValue,
            SelectionReason = $"NeuralUCB: highest UCB among {availableArms.Count} arms",
            ContextFeatures = context.Features.ToDictionary(kvp => kvp.Key, kvp => kvp.Value)
        };
    }

    /// <summary>
    /// Updates the selected arm with observed reward and context.
    /// </summary>
    public async Task UpdateArmAsync(
        string selectedArm,
        ContextVector context,
        decimal reward,
        CancellationToken ct = default)
    {
        NeuralUcbArm arm;
        
        lock (_lock)
        {
            if (!_arms.ContainsKey(selectedArm))
            {
                _arms[selectedArm] = new NeuralUcbArm(
                    _networkTemplate.Clone(),
                    _config);
            }
            arm = _arms[selectedArm];
        }

        await arm.UpdateAsync(context, reward, ct);

        Console.WriteLine($"[NEURAL-UCB] Updated {selectedArm}: reward={reward:F3} " +
                        $"updates={arm.UpdateCount}");
    }

    /// <summary>
    /// Gets current arm statistics for analysis.
    /// </summary>
    public async Task<Dictionary<string, ArmStatistics>> GetArmStatisticsAsync(CancellationToken ct = default)
    {
        var stats = new Dictionary<string, ArmStatistics>();

        foreach (var kvp in _arms.ToList()) // ToList to avoid lock issues
        {
            var armStats = await kvp.Value.GetStatisticsAsync(ct);
            stats[kvp.Key] = new ArmStatistics
            {
                ArmId = kvp.Key,
                UpdateCount = armStats.UpdateCount,
                AverageReward = armStats.AverageReward,
                ModelNorm = armStats.ModelComplexity,
                ConfidenceWidth = armStats.AverageUncertainty,
                LastUpdated = armStats.LastUpdated
            };
        }

        return stats;
    }

    /// <summary>
    /// Analyzes feature importance using neural network gradients.
    /// </summary>
    public async Task<FeatureImportanceReport> AnalyzeFeatureImportanceAsync(CancellationToken ct = default)
    {
        var featureImportance = new Dictionary<string, decimal>();
        var totalWeight = 0m;

        foreach (var arm in _arms.Values.ToList())
        {
            if (arm.UpdateCount == 0) continue;

            var armImportance = await arm.ComputeFeatureImportanceAsync(ct);
            var armWeight = (decimal)arm.UpdateCount;
            totalWeight += armWeight;

            foreach (var kvp in armImportance)
            {
                if (!featureImportance.ContainsKey(kvp.Key))
                    featureImportance[kvp.Key] = 0m;

                featureImportance[kvp.Key] += kvp.Value * armWeight;
            }
        }

        // Normalize by total weight
        if (totalWeight > 0)
        {
            featureImportance = featureImportance.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value / totalWeight
            );
        }

        return new FeatureImportanceReport
        {
            FeatureWeights = featureImportance,
            TotalArms = _arms.Count,
            ActiveArms = _arms.Count(kvp => kvp.Value.UpdateCount > 0),
            GeneratedAt = DateTime.UtcNow
        };
    }
}

/// <summary>
/// Individual NeuralUCB arm with neural network and uncertainty estimation.
/// </summary>
internal class NeuralUcbArm
{
    private readonly INeuralNetwork _network;
    private readonly NeuralUcbConfig _config;
    private readonly List<(ContextVector context, decimal reward)> _trainingData = new();
    private readonly object _dataLock = new();

    public int UpdateCount { get; private set; }
    public decimal AverageReward { get; private set; }
    public DateTime LastUpdated { get; private set; } = DateTime.UtcNow;
    public DateTime LastTraining { get; private set; } = DateTime.MinValue;

    public NeuralUcbArm(INeuralNetwork network, NeuralUcbConfig config)
    {
        _network = network;
        _config = config;
    }

    public async Task<(decimal prediction, decimal uncertainty)> PredictWithUncertaintyAsync(
        ContextVector context,
        CancellationToken ct = default)
    {
        if (UpdateCount < _config.MinSamplesForTraining)
        {
            // Not enough data - return high uncertainty
            return (0.5m, 1m);
        }

        // Get network prediction
        var features = context.ToArray(_config.InputDimension);
        var prediction = await _network.PredictAsync(features, ct);

        // Estimate uncertainty using ensemble or dropout
        var uncertainty = await EstimateUncertaintyAsync(context, ct);

        return (Math.Max(0m, Math.Min(1m, prediction)), uncertainty);
    }

    public async Task UpdateAsync(ContextVector context, decimal reward, CancellationToken ct = default)
    {
        lock (_dataLock)
        {
            _trainingData.Add((context, reward));
            
            // Keep only recent data
            if (_trainingData.Count > _config.MaxTrainingDataSize)
            {
                _trainingData.RemoveAt(0);
            }

            UpdateCount++;
            AverageReward = (AverageReward * (UpdateCount - 1) + reward) / UpdateCount;
            LastUpdated = DateTime.UtcNow;
        }

        // Retrain network periodically
        if (ShouldRetrain())
        {
            await RetrainNetworkAsync(ct);
        }
    }

    public async Task<NeuralArmStatistics> GetStatisticsAsync(CancellationToken ct = default)
    {
        await Task.CompletedTask;

        return new NeuralArmStatistics
        {
            UpdateCount = UpdateCount,
            AverageReward = AverageReward,
            ModelComplexity = await _network.GetComplexityAsync(ct),
            AverageUncertainty = await GetAverageUncertaintyAsync(ct),
            LastUpdated = LastUpdated,
            LastTraining = LastTraining
        };
    }

    public async Task<Dictionary<string, decimal>> ComputeFeatureImportanceAsync(CancellationToken ct = default)
    {
        // Compute feature importance using gradient-based methods
        var importance = new Dictionary<string, decimal>();
        
        if (UpdateCount == 0)
            return importance;

        // Use representative context for gradient computation
        var recentContexts = _trainingData.TakeLast(10).Select(d => d.context).ToList();
        
        foreach (var context in recentContexts)
        {
            var gradients = await _network.ComputeGradientsAsync(context.ToArray(_config.InputDimension), ct);
            
            var featureKeys = context.Features.Keys.OrderBy(k => k).ToList();
            for (int i = 0; i < Math.Min(gradients.Length, featureKeys.Count); i++)
            {
                var featureName = featureKeys[i];
                if (!importance.ContainsKey(featureName))
                    importance[featureName] = 0m;

                importance[featureName] += Math.Abs(gradients[i]);
            }
        }

        // Average over contexts
        if (recentContexts.Count > 0)
        {
            importance = importance.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value / recentContexts.Count
            );
        }

        return importance;
    }

    private async Task<decimal> EstimateUncertaintyAsync(ContextVector context, CancellationToken ct)
    {
        if (UpdateCount < _config.MinSamplesForUncertainty)
        {
            return 1m; // High uncertainty with little data
        }

        // Estimate uncertainty using ensemble prediction variance
        var predictions = new List<decimal>();
        
        for (int i = 0; i < _config.UncertaintyEstimationSamples; i++)
        {
            var features = context.ToArray(_config.InputDimension);
            var prediction = await _network.PredictWithDropoutAsync(features, ct);
            predictions.Add(prediction);
        }

        // Calculate variance as uncertainty measure
        if (predictions.Count <= 1)
            return 0.5m;

        var mean = predictions.Average();
        var variance = predictions.Sum(p => (p - mean) * (p - mean)) / (predictions.Count - 1);
        
        return Math.Min(1m, (decimal)Math.Sqrt((double)variance));
    }

    private bool ShouldRetrain()
    {
        // Retrain conditions
        var timeSinceLastTraining = DateTime.UtcNow - LastTraining;
        var hasEnoughNewData = UpdateCount >= _config.MinSamplesForTraining;
        var enoughTimePassed = timeSinceLastTraining >= _config.RetrainingInterval;
        var significantDataIncrease = UpdateCount > 0 && 
            (DateTime.UtcNow - LastTraining).TotalMinutes > _config.RetrainingInterval.TotalMinutes;

        return hasEnoughNewData && (enoughTimePassed || significantDataIncrease);
    }

    private async Task RetrainNetworkAsync(CancellationToken ct)
    {
        List<(ContextVector context, decimal reward)> trainingData;
        
        lock (_dataLock)
        {
            trainingData = _trainingData.ToList();
        }

        if (trainingData.Count < _config.MinSamplesForTraining)
            return;

        try
        {
            Console.WriteLine($"[NEURAL-UCB] Retraining network with {trainingData.Count} samples");

            var features = trainingData.Select(d => d.context.ToArray(_config.InputDimension)).ToArray();
            var targets = trainingData.Select(d => d.reward).ToArray();

            await _network.TrainAsync(features, targets, ct);
            LastTraining = DateTime.UtcNow;

            Console.WriteLine($"[NEURAL-UCB] Retraining completed");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[NEURAL-UCB] Retraining failed: {ex.Message}");
        }
    }

    private async Task<decimal> GetAverageUncertaintyAsync(CancellationToken ct)
    {
        if (UpdateCount == 0)
            return 1m;

        // Estimate average uncertainty using recent contexts
        var recentContexts = _trainingData.TakeLast(10).Select(d => d.context).ToList();
        
        if (!recentContexts.Any())
            return 0.5m;

        var uncertainties = new List<decimal>();
        foreach (var context in recentContexts)
        {
            var uncertainty = await EstimateUncertaintyAsync(context, ct);
            uncertainties.Add(uncertainty);
        }

        return uncertainties.Average();
    }
}

/// <summary>
/// Configuration for NeuralUCB bandit
/// </summary>
public record NeuralUcbConfig
{
    public decimal ExplorationWeight { get; init; } = 0.1m;
    public int InputDimension { get; init; } = 10;
    public int MinSamplesForTraining { get; init; } = 20;
    public int MinSamplesForUncertainty { get; init; } = 10;
    public int MaxTrainingDataSize { get; init; } = 1000;
    public TimeSpan RetrainingInterval { get; init; } = TimeSpan.FromMinutes(30);
    public int UncertaintyEstimationSamples { get; init; } = 5;
}

/// <summary>
/// Neural arm specific statistics
/// </summary>
internal record NeuralArmStatistics
{
    public int UpdateCount { get; init; }
    public decimal AverageReward { get; init; }
    public decimal ModelComplexity { get; init; }
    public decimal AverageUncertainty { get; init; }
    public DateTime LastUpdated { get; init; }
    public DateTime LastTraining { get; init; }
}

/// <summary>
/// Interface for neural network used in NeuralUCB
/// </summary>
public interface INeuralNetwork
{
    Task<decimal> PredictAsync(decimal[] features, CancellationToken ct = default);
    Task<decimal> PredictWithDropoutAsync(decimal[] features, CancellationToken ct = default);
    Task TrainAsync(decimal[][] features, decimal[] targets, CancellationToken ct = default);
    Task<decimal[]> ComputeGradientsAsync(decimal[] features, CancellationToken ct = default);
    Task<decimal> GetComplexityAsync(CancellationToken ct = default);
    INeuralNetwork Clone();
}

/// <summary>
/// Simple neural network implementation for NeuralUCB (placeholder)
/// In practice, this would integrate with ML.NET, TensorFlow.NET, or ONNX
/// </summary>
public class SimpleNeuralNetwork : INeuralNetwork
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly Random _random = new();
    private decimal[,] _weightsInput;
    private decimal[] _biasHidden;
    private decimal[] _weightsOutput;
    private decimal _biasOutput;

    public SimpleNeuralNetwork(int inputSize = 10, int hiddenSize = 20)
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        InitializeWeights();
    }

    public async Task<decimal> PredictAsync(decimal[] features, CancellationToken ct = default)
    {
        await Task.CompletedTask;
        return ForwardPass(features);
    }

    public async Task<decimal> PredictWithDropoutAsync(decimal[] features, CancellationToken ct = default)
    {
        await Task.CompletedTask;
        // Simple dropout simulation - randomly zero some hidden units
        return ForwardPass(features, dropout: true);
    }

    public async Task TrainAsync(decimal[][] features, decimal[] targets, CancellationToken ct = default)
    {
        await Task.CompletedTask;
        
        // Simple gradient descent training (placeholder)
        var learningRate = 0.01m;
        var epochs = 10;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < features.Length; i++)
            {
                var prediction = ForwardPass(features[i]);
                var error = targets[i] - prediction;
                
                // Very simplified backpropagation
                _biasOutput += learningRate * error;
                for (int j = 0; j < _weightsOutput.Length; j++)
                {
                    _weightsOutput[j] += learningRate * error * 0.1m; // Simplified gradient
                }
            }
        }
    }

    public async Task<decimal[]> ComputeGradientsAsync(decimal[] features, CancellationToken ct = default)
    {
        await Task.CompletedTask;
        
        // Simplified gradient computation
        var gradients = new decimal[features.Length];
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = (decimal)_random.NextDouble() * 0.1m; // Placeholder
        }
        
        return gradients;
    }

    public async Task<decimal> GetComplexityAsync(CancellationToken ct = default)
    {
        await Task.CompletedTask;
        
        // L2 norm of weights as complexity measure
        var complexity = 0m;
        
        for (int i = 0; i < _weightsInput.GetLength(0); i++)
        {
            for (int j = 0; j < _weightsInput.GetLength(1); j++)
            {
                complexity += _weightsInput[i, j] * _weightsInput[i, j];
            }
        }
        
        for (int i = 0; i < _weightsOutput.Length; i++)
        {
            complexity += _weightsOutput[i] * _weightsOutput[i];
        }
        
        return (decimal)Math.Sqrt((double)complexity);
    }

    public INeuralNetwork Clone()
    {
        var clone = new SimpleNeuralNetwork(_inputSize, _hiddenSize);
        
        // Deep copy weights
        Array.Copy(_weightsInput, clone._weightsInput, _weightsInput.Length);
        Array.Copy(_biasHidden, clone._biasHidden, _biasHidden.Length);
        Array.Copy(_weightsOutput, clone._weightsOutput, _weightsOutput.Length);
        clone._biasOutput = _biasOutput;
        
        return clone;
    }

    private void InitializeWeights()
    {
        _weightsInput = new decimal[_inputSize, _hiddenSize];
        _biasHidden = new decimal[_hiddenSize];
        _weightsOutput = new decimal[_hiddenSize];
        _biasOutput = 0m;

        // Xavier initialization
        var stddev = (decimal)Math.Sqrt(2.0 / (_inputSize + _hiddenSize));
        
        for (int i = 0; i < _inputSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                _weightsInput[i, j] = (decimal)_random.NextGaussian() * stddev;
            }
        }

        for (int i = 0; i < _hiddenSize; i++)
        {
            _biasHidden[i] = (decimal)_random.NextGaussian() * stddev;
            _weightsOutput[i] = (decimal)_random.NextGaussian() * stddev;
        }
    }

    private decimal ForwardPass(decimal[] features, bool dropout = false)
    {
        // Input to hidden
        var hidden = new decimal[_hiddenSize];
        for (int j = 0; j < _hiddenSize; j++)
        {
            var sum = _biasHidden[j];
            for (int i = 0; i < Math.Min(_inputSize, features.Length); i++)
            {
                sum += features[i] * _weightsInput[i, j];
            }
            
            // ReLU activation
            hidden[j] = Math.Max(0m, sum);
            
            // Dropout
            if (dropout && _random.NextDouble() < 0.5)
            {
                hidden[j] = 0m;
            }
        }

        // Hidden to output
        var output = _biasOutput;
        for (int i = 0; i < _hiddenSize; i++)
        {
            output += hidden[i] * _weightsOutput[i];
        }

        // Sigmoid activation for output
        return 1m / (1m + (decimal)Math.Exp(-(double)output));
    }
}

/// <summary>
/// Extension methods for Random class
/// </summary>
public static class RandomExtensions
{
    public static double NextGaussian(this Random random)
    {
        // Box-Muller transform
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
