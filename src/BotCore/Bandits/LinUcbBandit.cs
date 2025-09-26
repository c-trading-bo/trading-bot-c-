using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Bandits;

/// <summary>
/// LinUCB (Linear Upper Confidence Bound) bandit with function approximation.
/// Uses linear models to generalize across continuous feature spaces instead of discrete table lookup.
/// Enables smooth generalization to new but similar contexts.
/// </summary>
public class LinUcbBandit : IFunctionApproximationBandit
{
    private readonly Dictionary<string, LinUcbArm> _arms = new();
    private readonly LinUcbConfig _config;
    private readonly object _lock = new();

    public LinUcbBandit(LinUcbConfig? config = null)
    {
        _config = config ?? new LinUcbConfig();
    }

    /// <summary>
    /// Selects arm using LinUCB algorithm with continuous context features.
    /// </summary>
    public async Task<BanditSelection> SelectArmAsync(
        List<string> availableArms,
        ContextVector context,
        CancellationToken ct = default)
    {
        if (availableArms is null) throw new ArgumentNullException(nameof(availableArms));
        if (context is null) throw new ArgumentNullException(nameof(context));
        
        await Task.CompletedTask.ConfigureAwait(false);

        lock (_lock)
        {
            // Ensure all arms exist
            foreach (var armId in availableArms)
            {
                if (!_arms.ContainsKey(armId))
                {
                    _arms[armId] = new LinUcbArm(_config.ContextDimension, _config.Alpha);
                }
            }

            var selections = new List<(string armId, decimal ucbValue, decimal prediction, decimal confidence)>();

            foreach (var armId in availableArms)
            {
                var arm = _arms[armId];
                var (prediction, confidence) = arm.PredictWithConfidence(context);
                var ucbValue = prediction + _config.Alpha * confidence;

                selections.Add((armId, ucbValue, prediction, confidence));
            }

            // Select arm with highest UCB value
            var bestSelection = selections.OrderByDescending(s => s.ucbValue).First();

            Console.WriteLine($"[LINUCB] Selected {bestSelection.armId}: " +
                            $"pred={bestSelection.prediction:F3} conf={bestSelection.confidence:F3} " +
                            $"ucb={bestSelection.ucbValue:F3}");

            return new BanditSelection
            {
                SelectedArm = bestSelection.armId,
                Prediction = bestSelection.prediction,
                Confidence = bestSelection.confidence,
                UcbValue = bestSelection.ucbValue,
                SelectionReason = $"LinUCB: highest UCB among {availableArms.Count} arms",
                ContextFeatures = context.Features.ToDictionary(kvp => kvp.Key, kvp => kvp.Value)
            };
        }
    }

    /// <summary>
    /// Updates the selected arm with the observed reward and context.
    /// </summary>
    public async Task UpdateArmAsync(
        string selectedArm,
        ContextVector context,
        decimal reward,
        CancellationToken ct = default)
    {
        if (selectedArm is null) throw new ArgumentNullException(nameof(selectedArm));
        if (context is null) throw new ArgumentNullException(nameof(context));
        
        await Task.CompletedTask.ConfigureAwait(false);

        lock (_lock)
        {
            if (!_arms.ContainsKey(selectedArm))
            {
                _arms[selectedArm] = new LinUcbArm(_config.ContextDimension, _config.Alpha);
            }

            var arm = _arms[selectedArm];
            arm.Update(context, reward);

            Console.WriteLine($"[LINUCB] Updated {selectedArm}: reward={reward:F3} " +
                            $"updates={arm.UpdateCount}");
        }
    }

    /// <summary>
    /// Gets current arm statistics for analysis.
    /// </summary>
    public async Task<Dictionary<string, ArmStatistics>> GetArmStatisticsAsync(CancellationToken ct = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);

        lock (_lock)
        {
            return _arms.ToDictionary(
                kvp => kvp.Key,
                kvp => new ArmStatistics
                {
                    ArmId = kvp.Key,
                    UpdateCount = kvp.Value.UpdateCount,
                    AverageReward = kvp.Value.AverageReward,
                    ModelNorm = kvp.Value.GetModelNorm(),
                    ConfidenceWidth = kvp.Value.GetAverageConfidenceWidth(),
                    LastUpdated = kvp.Value.LastUpdated
                });
        }
    }

    /// <summary>
    /// Analyzes feature importance across all arms.
    /// </summary>
    public async Task<FeatureImportanceReport> AnalyzeFeatureImportanceAsync(CancellationToken ct = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);

        lock (_lock)
        {
            var featureImportance = new Dictionary<string, decimal>();
            var totalWeight = 0m;

            foreach (var arm in _arms.Values)
            {
                if (arm.UpdateCount == 0) continue;

                var weights = arm.GetFeatureWeights();
                var armWeight = (decimal)arm.UpdateCount;
                totalWeight += armWeight;

                for (int i = 0; i < weights.Length && i < _config.ContextDimension; i++)
                {
                    var featureName = $"feature_{i}";
                    if (!featureImportance.ContainsKey(featureName))
                        featureImportance[featureName] = 0;

                    featureImportance[featureName] += Math.Abs(weights[i]) * armWeight;
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
}

/// <summary>
/// Individual LinUCB arm with linear model and confidence estimation.
/// </summary>
internal sealed class LinUcbArm
{
    private readonly decimal _alpha;
    private readonly int _dimension;
    private readonly decimal[,] _A; // A = I + sum(x_t * x_t^T)
    private readonly decimal[] _b;  // b = sum(r_t * x_t)
    private decimal[,]? _AInverse;
    private bool _inverseNeedsUpdate = true;

    public int UpdateCount { get; private set; }
    public decimal AverageReward { get; private set; }
    public DateTime LastUpdated { get; private set; } = DateTime.UtcNow;

    public LinUcbArm(int dimension, decimal alpha)
    {
        _dimension = dimension;
        _alpha = alpha;
        _A = new decimal[dimension, dimension];
        _b = new decimal[dimension];

        // Initialize A as identity matrix
        for (int i = 0; i < dimension; i++)
        {
            _A[i, i] = 1m;
        }
    }

    public (decimal prediction, decimal confidence) PredictWithConfidence(ContextVector context)
    {
        var x = context.ToArray(_dimension);

        if (_inverseNeedsUpdate)
        {
            UpdateInverse();
        }

        if (_AInverse == null)
        {
            // Fallback if matrix inversion fails
            return (0.5m, 1m);
        }

        // theta = A^(-1) * b
        var theta = MatrixVectorMultiply(_AInverse, _b);

        // prediction = theta^T * x
        var prediction = VectorDotProduct(theta, x);

        // confidence = sqrt(x^T * A^(-1) * x)
        var temp = MatrixVectorMultiply(_AInverse, x);
        var confidence = (decimal)Math.Sqrt((double)VectorDotProduct(x, temp));

        return (Math.Max(0m, Math.Min(1m, prediction)), confidence);
    }

    public void Update(ContextVector context, decimal reward)
    {
        var x = context.ToArray(_dimension);

        // Update A = A + x * x^T
        for (int i = 0; i < _dimension; i++)
        {
            for (int j = 0; j < _dimension; j++)
            {
                _A[i, j] += x[i] * x[j];
            }
        }

        // Update b = b + reward * x
        for (int i = 0; i < _dimension; i++)
        {
            _b[i] += reward * x[i];
        }

        // Update statistics
        UpdateCount++;
        AverageReward = (AverageReward * (UpdateCount - 1) + reward) / UpdateCount;
        LastUpdated = DateTime.UtcNow;
        _inverseNeedsUpdate = true;
    }

    public decimal GetModelNorm()
    {
        if (_inverseNeedsUpdate)
        {
            UpdateInverse();
        }

        if (_AInverse == null) return 0m;

        var theta = MatrixVectorMultiply(_AInverse, _b);
        return (decimal)Math.Sqrt((double)VectorDotProduct(theta, theta));
    }

    public decimal GetAverageConfidenceWidth()
    {
        if (UpdateCount == 0) return 1m;

        // Approximate confidence width using trace of A^(-1)
        if (_inverseNeedsUpdate)
        {
            UpdateInverse();
        }

        if (_AInverse == null) return 1m;

        var trace = 0m;
        for (int i = 0; i < _dimension; i++)
        {
            trace += _AInverse[i, i];
        }

        return (decimal)Math.Sqrt((double)(trace / _dimension));
    }

    public decimal[] GetFeatureWeights()
    {
        if (_inverseNeedsUpdate)
        {
            UpdateInverse();
        }

        if (_AInverse == null)
        {
            return new decimal[_dimension];
        }

        return MatrixVectorMultiply(_AInverse, _b);
    }

    private void UpdateInverse()
    {
        try
        {
            _AInverse = InvertMatrix(_A);
            _inverseNeedsUpdate = false;
        }
        catch
        {
            // Matrix inversion failed - use identity as fallback
            _AInverse = new decimal[_dimension, _dimension];
            for (int i = 0; i < _dimension; i++)
            {
                _AInverse[i, i] = 1m;
            }
        }
    }

    private decimal[,] InvertMatrix(decimal[,] matrix)
    {
        var n = _dimension;
        var augmented = new decimal[n, 2 * n];

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = matrix[i, j];
                augmented[i, j + n] = i == j ? 1m : 0m;
            }
        }

        // Gaussian elimination
        for (int i = 0; i < n; i++)
        {
            // Find pivot
            var maxRow = i;
            for (int k = i + 1; k < n; k++)
            {
                if (Math.Abs(augmented[k, i]) > Math.Abs(augmented[maxRow, i]))
                {
                    maxRow = k;
                }
            }

            // Swap rows
            if (maxRow != i)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    (augmented[i, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[i, j]);
                }
            }

            // Check for singular matrix
            if (Math.Abs(augmented[i, i]) < 1e-10m)
            {
                throw new InvalidOperationException("Matrix is singular");
            }

            // Scale pivot row
            var pivot = augmented[i, i];
            for (int j = 0; j < 2 * n; j++)
            {
                augmented[i, j] /= pivot;
            }

            // Eliminate column
            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    var factor = augmented[k, i];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[k, j] -= factor * augmented[i, j];
                    }
                }
            }
        }

        // Extract inverse matrix
        var inverse = new decimal[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, j + n];
            }
        }

        return inverse;
    }

    private static decimal[] MatrixVectorMultiply(decimal[,] matrix, decimal[] vector)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var result = new decimal[rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols && j < vector.Length; j++)
            {
                result[i] += matrix[i, j] * vector[j];
            }
        }

        return result;
    }

    private static decimal VectorDotProduct(decimal[] a, decimal[] b)
    {
        var result = 0m;
        var length = Math.Min(a.Length, b.Length);

        for (int i = 0; i < length; i++)
        {
            result += a[i] * b[i];
        }

        return result;
    }
}

/// <summary>
/// Configuration for LinUCB bandit
/// </summary>
public record LinUcbConfig
{
    public decimal Alpha { get; init; } = 0.1m; // Exploration parameter
    public int ContextDimension { get; init; } = 10; // Number of context features
    public bool EnableFeatureNormalization { get; init; } = true;
    public decimal RegularizationStrength { get; init; } = 1m;
}

/// <summary>
/// Context vector for function approximation
/// </summary>
public class ContextVector
{
    public Dictionary<string, decimal> Features { get; init; } = new();
    public DateTime Timestamp { get; init; } = DateTime.UtcNow;

    public decimal[] ToArray(int dimension)
    {
        var array = new decimal[dimension];
        var featureKeys = Features.Keys.OrderBy(k => k).ToList();

        for (int i = 0; i < Math.Min(dimension, featureKeys.Count); i++)
        {
            array[i] = Features[featureKeys[i]];
        }

        return array;
    }

    public static ContextVector FromStrategy(
        string strategy,
        string config,
        string regime,
        string session,
        decimal atr,
        decimal spread,
        decimal volume,
        decimal volatility,
        decimal timeOfDay)
    {
        return new ContextVector
        {
            Features = new Dictionary<string, decimal>
            {
                ["strategy_hash"] = (decimal)strategy.GetHashCode() / int.MaxValue,
                ["config_hash"] = (decimal)config.GetHashCode() / int.MaxValue,
                ["regime_bull"] = regime == "BULL" ? 1m : 0m,
                ["regime_bear"] = regime == "BEAR" ? 1m : 0m,
                ["regime_neutral"] = regime == "NEUTRAL" ? 1m : 0m,
                ["session_morning"] = session.Contains("MORNING") ? 1m : 0m,
                ["session_afternoon"] = session.Contains("AFTERNOON") ? 1m : 0m,
                ["atr_zscore"] = Math.Max(-3m, Math.Min(3m, atr)), // Clip to [-3, 3]
                ["spread_bps"] = Math.Min(20m, spread), // Cap at 20 bps
                ["volume_ratio"] = Math.Min(3m, volume), // Cap at 3x normal
                ["volatility"] = Math.Min(0.1m, volatility) // Cap at 10%
            }
        };
    }
}

/// <summary>
/// Result of bandit arm selection
/// </summary>
public record BanditSelection
{
    public string SelectedArm { get; init; } = "";
    public decimal Prediction { get; init; }
    public decimal Confidence { get; init; }
    public decimal UcbValue { get; init; }
    public string SelectionReason { get; init; } = "";
    public Dictionary<string, decimal> ContextFeatures { get; init; } = new();
}

/// <summary>
/// Statistics for a bandit arm
/// </summary>
public record ArmStatistics
{
    public string ArmId { get; init; } = "";
    public int UpdateCount { get; init; }
    public decimal AverageReward { get; init; }
    public decimal ModelNorm { get; init; }
    public decimal ConfidenceWidth { get; init; }
    public DateTime LastUpdated { get; init; }
}

/// <summary>
/// Feature importance analysis
/// </summary>
public record FeatureImportanceReport
{
    public Dictionary<string, decimal> FeatureWeights { get; init; } = new();
    public int TotalArms { get; init; }
    public int ActiveArms { get; init; }
    public DateTime GeneratedAt { get; init; }
}

/// <summary>
/// Interface for function approximation bandits
/// </summary>
public interface IFunctionApproximationBandit
{
    Task<BanditSelection> SelectArmAsync(
        List<string> availableArms,
        ContextVector context,
        CancellationToken ct = default);

    Task UpdateArmAsync(
        string selectedArm,
        ContextVector context,
        decimal reward,
        CancellationToken ct = default);

    Task<Dictionary<string, ArmStatistics>> GetArmStatisticsAsync(CancellationToken ct = default);

    Task<FeatureImportanceReport> AnalyzeFeatureImportanceAsync(CancellationToken ct = default);
}
