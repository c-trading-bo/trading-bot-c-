using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Risk;

/// <summary>
/// Enhanced Bayesian priors with uncertainty quantification and credible intervals.
/// Provides shrinkage across strategy/config/regime dimensions for better generalization.
/// </summary>
public class EnhancedBayesianPriors : IBayesianPriors
{
    private readonly Dictionary<string, BayesianPosterior> _priors = new();
    private readonly Dictionary<string, HierarchicalGroup> _hierarchicalGroups = new();
    private readonly ShrinkageConfiguration _shrinkageConfig;
    private readonly object _lock = new();

    public EnhancedBayesianPriors(ShrinkageConfiguration? shrinkageConfig = null)
    {
        _shrinkageConfig = shrinkageConfig ?? new ShrinkageConfiguration();
        InitializeHierarchicalStructure();
    }

    /// <summary>
    /// Gets enhanced prior with uncertainty quantification and shrinkage
    /// </summary>
    public async Task<BayesianEstimate> GetPriorAsync(
        string strategy,
        string config,
        string regime,
        string session,
        CancellationToken ct = default)
    {
        await Task.CompletedTask.ConfigureAwait(false); // For async compatibility

        lock (_lock)
        {
            var key = CreateKey(strategy, config, regime, session);

            if (!_priors.ContainsKey(key))
            {
                _priors[key] = CreateDefaultPosterior();
            }

            var posterior = _priors[key];
            var shrunkPosterior = ApplyShrinkage(strategy, regime, posterior);

            // Create shrinkage estimate for calculations
            var shrinkageEstimate = new ShrinkageEstimate(
                shrunkPosterior.Alpha,
                shrunkPosterior.Beta,
                BayesianCalculationExtensions.CalculateShrinkageFactor(shrunkPosterior)
            );

            return new BayesianEstimate
            {
                Mean = shrinkageEstimate.CalculateMean(),
                Variance = shrinkageEstimate.CalculateVariance(),
                CredibleInterval = CalculateCredibleInterval(shrunkPosterior, 0.95m),
                EffectiveSampleSize = CalculateEffectiveSampleSize(shrunkPosterior),
                UncertaintyLevel = CalculateUncertainty(shrunkPosterior),
                IsReliable = shrinkageEstimate.Alpha + shrinkageEstimate.Beta > 20, // Sufficient data
                ShrinkageFactor = shrinkageEstimate.Shrinkage,
                LastUpdated = shrunkPosterior.LastUpdated
            };
        }
    }

    /// <summary>
    /// Updates posterior with new outcome and applies hierarchical learning
    /// </summary>
    public async Task UpdatePosteriorAsync(
        string strategy,
        string config,
        string regime,
        string session,
        bool wasSuccessful,
        CancellationToken ct = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);

        lock (_lock)
        {
            var key = CreateKey(strategy, config, regime, session);

            if (!_priors.ContainsKey(key))
            {
                _priors[key] = CreateDefaultPosterior();
            }

            var posterior = _priors[key];

            // Update local posterior
            if (wasSuccessful)
            {
                posterior.Alpha += 1;
            }
            else
            {
                posterior.Beta += 1;
            }

            posterior.LastUpdated = DateTime.UtcNow;
            posterior.TotalObservations++;

            // Update hierarchical groups for learning
            UpdateHierarchicalGroups(strategy, regime, session, wasSuccessful);

            Console.WriteLine($"[ENHANCED-PRIORS] Updated {key}: " +
                            $"α={posterior.Alpha:F1} β={posterior.Beta:F1} " +
                            $"mean={posterior.Alpha / (posterior.Alpha + posterior.Beta):F3}");
        }
    }

    /// <summary>
    /// Gets uncertainty-weighted sampling for bandit selection
    /// </summary>
    public async Task<decimal> SamplePosteriorAsync(
        string strategy,
        string config,
        string regime,
        string session,
        CancellationToken ct = default)
    {
        var estimate = await GetPriorAsync(strategy, config, regime, session, ct).ConfigureAwait(false);

        // Sample from Beta distribution with shrinkage
        var alpha = estimate.EffectiveSampleSize * estimate.Mean;
        var beta = estimate.EffectiveSampleSize * (1 - estimate.Mean);

        return SampleBeta(alpha, beta);
    }

    /// <summary>
    /// Gets all priors with their uncertainty levels for analysis
    /// </summary>
    public async Task<Dictionary<string, BayesianEstimate>> GetAllPriorsAsync(CancellationToken ct = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);

        lock (_lock)
        {
            var result = new Dictionary<string, BayesianEstimate>();

            foreach (var kvp in _priors)
            {
                var parts = kvp.Key.Split('|');
                if (parts.Length == 4)
                {
                    var estimate = GetPriorAsync(parts[0], parts[1], parts[2], parts[3], ct).Result;
                    result[kvp.Key] = estimate;
                }
            }

            return result;
        }
    }

    private BayesianPosterior ApplyShrinkage(
        string strategy,
                string regime,
                BayesianPosterior localPosterior)
    {
        var shrinkageTargets = new List<(decimal weight, BayesianPosterior target)>();

        // Strategy-level shrinkage
        var strategyKey = CreateKey(strategy, "*", "*", "*");
        if (_hierarchicalGroups.ContainsKey(strategyKey))
        {
            shrinkageTargets.Add((_shrinkageConfig.StrategyWeight, _hierarchicalGroups[strategyKey].Posterior));
        }

        // Regime-level shrinkage
        var regimeKey = CreateKey("*", "*", regime, "*");
        if (_hierarchicalGroups.ContainsKey(regimeKey))
        {
            shrinkageTargets.Add((_shrinkageConfig.RegimeWeight, _hierarchicalGroups[regimeKey].Posterior));
        }

        // Global shrinkage
        var globalKey = CreateKey("*", "*", "*", "*");
        if (_hierarchicalGroups.ContainsKey(globalKey))
        {
            shrinkageTargets.Add((_shrinkageConfig.GlobalWeight, _hierarchicalGroups[globalKey].Posterior));
        }

        if (shrinkageTargets.Count == 0)
        {
            localPosterior.ShrinkageFactor = 0;
            return localPosterior;
        }

        // Calculate shrinkage strength based on local data quality
        var localN = localPosterior.Alpha + localPosterior.Beta;
        var shrinkageStrength = _shrinkageConfig.BaseShrinkage *
                               (decimal)Math.Exp(-(double)localN / (double)_shrinkageConfig.ShrinkageDecay);

        shrinkageStrength = Math.Min(0.8m, Math.Max(0.05m, shrinkageStrength));

        // Apply James-Stein shrinkage
        var shrunkAlpha = localPosterior.Alpha;
        var shrunkBeta = localPosterior.Beta;

        foreach (var (weight, target) in shrinkageTargets)
        {
            var targetAlpha = target.Alpha;
            var targetBeta = target.Beta;

            shrunkAlpha += shrinkageStrength * weight * (targetAlpha - localPosterior.Alpha);
            shrunkBeta += shrinkageStrength * weight * (targetBeta - localPosterior.Beta);
        }

        return new BayesianPosterior
        {
            Alpha = Math.Max(0.1m, shrunkAlpha),
            Beta = Math.Max(0.1m, shrunkBeta),
            LastUpdated = localPosterior.LastUpdated,
            TotalObservations = localPosterior.TotalObservations,
            ShrinkageFactor = shrinkageStrength
        };
    }

    private void UpdateHierarchicalGroups(
        string strategy,
                string regime,
        string session,
        bool wasSuccessful)
    {
        var updates = new[]
        {
            CreateKey(strategy, "*", "*", "*"),
            CreateKey("*", "*", regime, "*"),
            CreateKey("*", "*", "*", session),
            CreateKey("*", "*", "*", "*") // Global
        };

        foreach (var key in updates)
        {
            if (!_hierarchicalGroups.ContainsKey(key))
            {
                _hierarchicalGroups[key] = new HierarchicalGroup
                {
                    Posterior = CreateDefaultPosterior()
                };
            }

            var group = _hierarchicalGroups[key];
            if (wasSuccessful)
            {
                group.Posterior.Alpha += 0.1m; // Fractional updates for hierarchical groups
            }
            else
            {
                group.Posterior.Beta += 0.1m;
            }

            group.Posterior.LastUpdated = DateTime.UtcNow;
        }
    }

    private static CredibleInterval CalculateCredibleInterval(BayesianPosterior posterior, decimal confidence)
    {
        // For Beta distribution, use quantile function approximation
        var alpha = posterior.Alpha;
        var beta = posterior.Beta;
        var mean = alpha / (alpha + beta);
        var variance = (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1));
        var stdDev = (decimal)Math.Sqrt((double)variance);

        // Approximate credible interval using normal approximation for large samples
        var z = confidence switch
        {
            0.95m => 1.96m,
            0.90m => 1.645m,
            0.99m => 2.576m,
            _ => 1.96m
        };

        var lower = Math.Max(0m, mean - z * stdDev);
        var upper = Math.Min(1m, mean + z * stdDev);

        return new CredibleInterval(lower, upper, confidence);
    }

    private static decimal CalculateEffectiveSampleSize(BayesianPosterior posterior)
    {
        // Effective sample size accounting for shrinkage
        var baseSampleSize = posterior.Alpha + posterior.Beta;
        var shrinkageDiscount = 1m - posterior.ShrinkageFactor * 0.5m; // Shrinkage reduces effective sample size

        return Math.Max(1m, baseSampleSize * shrinkageDiscount);
    }

    private static UncertaintyLevel CalculateUncertainty(BayesianPosterior posterior)
    {
        var effectiveN = CalculateEffectiveSampleSize(posterior);
        var variance = posterior.Alpha * posterior.Beta /
                      ((posterior.Alpha + posterior.Beta) * (posterior.Alpha + posterior.Beta) *
                       (posterior.Alpha + posterior.Beta + 1));

        return (effectiveN, variance) switch
        {
            ( < 5, _) => UncertaintyLevel.VeryHigh,
            ( < 15, > 0.05m) => UncertaintyLevel.High,
            ( < 30, > 0.02m) => UncertaintyLevel.Medium,
            ( < 100, > 0.01m) => UncertaintyLevel.Low,
            _ => UncertaintyLevel.VeryLow
        };
    }

    private decimal SampleBeta(decimal alpha, decimal beta)
    {
        // Box-Muller transform for Beta sampling
        var random = new Random();

        // Use Gamma sampling to generate Beta
        var x = SampleGamma(alpha, random, CancellationToken.None);
        var y = SampleGamma(beta, random, CancellationToken.None);

        return x / (x + y);
    }

    private decimal SampleGamma(decimal shape, Random random, CancellationToken cancellationToken = default)
    {
        // Simple gamma sampling using acceptance-rejection
        if (shape < 1m)
        {
            return SampleGamma(shape + 1m, random, cancellationToken) * (decimal)Math.Pow(random.NextDouble(), 1.0 / (double)shape);
        }

        var d = shape - 1m / 3m;
        var c = 1m / (decimal)Math.Sqrt(9.0 * (double)d);

        while (!cancellationToken.IsCancellationRequested)
        {
            var x = (decimal)random.NextDouble();
            var y = (decimal)random.NextDouble();

            var z = (decimal)Math.Sqrt(-2.0 * Math.Log((double)x)) * (decimal)Math.Cos(2.0 * Math.PI * (double)y);
            var v = (1m + c * z) * (1m + c * z) * (1m + c * z);

            if (v > 0 && Math.Log((double)y) < 0.5 * (double)(z * z) + (double)d * (1.0 - (double)v + Math.Log((double)v)))
            {
                return d * v;
            }
        }
        
        // Fallback if cancellation was requested
        return d; // Return a reasonable default value
    }

    private void InitializeHierarchicalStructure()
    {
        // Initialize global prior
        var globalKey = CreateKey("*", "*", "*", "*");
        _hierarchicalGroups[globalKey] = new HierarchicalGroup
        {
            Posterior = new BayesianPosterior
            {
                Alpha = _shrinkageConfig.GlobalAlpha,
                Beta = _shrinkageConfig.GlobalBeta,
                LastUpdated = DateTime.UtcNow
            }
        };
    }

    private static BayesianPosterior CreateDefaultPosterior()
    {
        return new BayesianPosterior
        {
            Alpha = 1m, // Weak prior
            Beta = 1m,
            LastUpdated = DateTime.UtcNow,
            TotalObservations = 0,
            ShrinkageFactor = 0m
        };
    }

    private static string CreateKey(string strategy, string config, string regime, string session)
    {
        return $"{strategy}|{config}|{regime}|{session}";
    }
}

/// <summary>
/// Enhanced Bayesian posterior with shrinkage information
/// </summary>
public class BayesianPosterior
{
    public decimal Alpha { get; set; }
    public decimal Beta { get; set; }
    public DateTime LastUpdated { get; set; }
    public int TotalObservations { get; set; }
    public decimal ShrinkageFactor { get; set; }
}

/// <summary>
/// Hierarchical group for shrinkage estimation
/// </summary>
public class HierarchicalGroup
{
    public BayesianPosterior Posterior { get; set; } = null!;
    public int MemberCount { get; set; }
}

/// <summary>
/// Bayesian estimate with uncertainty quantification
/// </summary>
public record BayesianEstimate
{
    public decimal Mean { get; init; }
    public decimal Variance { get; init; }
    public CredibleInterval CredibleInterval { get; init; } = null!;
    public decimal EffectiveSampleSize { get; init; }
    public UncertaintyLevel UncertaintyLevel { get; init; }
    public bool IsReliable { get; init; }
    public decimal ShrinkageFactor { get; init; }
    public DateTime LastUpdated { get; init; }
}

/// <summary>
/// Credible interval for Bayesian estimate
/// </summary>
public record CredibleInterval(decimal Lower, decimal Upper, decimal Confidence);

/// <summary>
/// Uncertainty level classification
/// </summary>
public enum UncertaintyLevel
{
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh
}

/// <summary>
/// Shrinkage configuration parameters
/// </summary>
public record ShrinkageConfiguration
{
    public decimal BaseShrinkage { get; init; } = 0.3m;
    public decimal ShrinkageDecay { get; init; } = 20m;
    public decimal StrategyWeight { get; init; } = 0.4m;
    public decimal RegimeWeight { get; init; } = 0.3m;
    public decimal GlobalWeight { get; init; } = 0.3m;
    public decimal GlobalAlpha { get; init; } = 10m;
    public decimal GlobalBeta { get; init; } = 10m;
}

/// <summary>
/// Interface for Bayesian priors
/// </summary>
public interface IBayesianPriors
{
    Task<BayesianEstimate> GetPriorAsync(
        string strategy,
        string config,
        string regime,
        string session,
        CancellationToken ct = default);

    Task UpdatePosteriorAsync(
        string strategy,
        string config,
        string regime,
        string session,
        bool wasSuccessful,
        CancellationToken ct = default);

    Task<decimal> SamplePosteriorAsync(
        string strategy,
        string config,
        string regime,
        string session,
        CancellationToken ct = default);

    Task<Dictionary<string, BayesianEstimate>> GetAllPriorsAsync(CancellationToken ct = default);
}

/// <summary>
/// James-Stein shrinkage estimate for Bayesian calculations
/// </summary>
public record ShrinkageEstimate(decimal Alpha, decimal Beta, decimal Shrinkage)
{
    public decimal Mean => Alpha / (Alpha + Beta);
    public decimal Variance => (Alpha * Beta) / ((Alpha + Beta) * (Alpha + Beta) * (Alpha + Beta + 1));
}

/// <summary>
/// Helper extension methods for Bayesian calculations
/// </summary>
public static class BayesianCalculationExtensions
{
    // Helper methods for calculations
    public static decimal CalculateMean(this ShrinkageEstimate estimate)
    {
        if (estimate is null) throw new ArgumentNullException(nameof(estimate));
        
        return estimate.Alpha / (estimate.Alpha + estimate.Beta);
    }

    public static decimal CalculateVariance(this ShrinkageEstimate estimate)
    {
        if (estimate is null) throw new ArgumentNullException(nameof(estimate));
        
        var total = estimate.Alpha + estimate.Beta;
        return (estimate.Alpha * estimate.Beta) / (total * total * (total + 1));
    }

    public static decimal CalculateShrinkageFactor(BayesianPosterior posterior)
    {
        if (posterior is null) throw new ArgumentNullException(nameof(posterior));
        
        // Simple shrinkage factor based on sample size
        var n = posterior.Alpha + posterior.Beta;
        return Math.Min(0.9m, Math.Max(0.1m, 1.0m / (1.0m + n / 10.0m)));
    }
}
