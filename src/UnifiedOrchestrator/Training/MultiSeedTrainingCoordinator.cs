using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Multi-Seed Training Coordinator - Trains each component multiple times with different
/// random seeds to verify learning is real and not random luck. Requires majority of seeds
/// to beat champion before promotion.
/// 
/// Process:
/// 1. Train with 5 deterministic seeds (42, 123, 456, 789, 1337)
/// 2. Evaluate each seed's model on TEST set (never shown during training)
/// 3. Compare each seed to current champion
/// 4. Require 3 out of 5 seeds to beat champion (60% success rate)
/// 5. Promote best seed's model if majority succeeds
/// 6. Reject all if majority fails (likely random luck)
/// 
/// This prevents promoting models that got lucky with one random initialization.
/// </summary>
public sealed class MultiSeedTrainingCoordinator
{
    private readonly ILogger<MultiSeedTrainingCoordinator> _logger;
    
    // Five deterministic seeds for reproducible multi-seed training
    private static readonly int[] TrainingSeeds = { 42, 123, 456, 789, 1337 };
    
    // Minimum seeds that must beat champion (3 out of 5 = 60%)
    private const int MinimumSuccessfulSeeds = 3;

    public MultiSeedTrainingCoordinator(ILogger<MultiSeedTrainingCoordinator> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Get the standard training seeds
    /// </summary>
    public int[] GetTrainingSeeds()
    {
        return TrainingSeeds;
    }

    /// <summary>
    /// Make promotion decision based on multi-seed results
    /// </summary>
    /// <param name="componentName">Name of component being trained</param>
    /// <param name="seedResults">Results from each seed</param>
    /// <param name="championTestMetric">Champion's performance on test set</param>
    /// <returns>Promotion decision with best seed if approved</returns>
    public PromotionDecision MakePromotionDecision(
        string componentName,
        List<SeedTrainingResult> seedResults,
        double championTestMetric)
    {
        if (seedResults.Count != TrainingSeeds.Length)
        {
            _logger.LogWarning(
                "[MULTI-SEED] {Component}: Expected {Expected} seed results, got {Actual}",
                componentName, TrainingSeeds.Length, seedResults.Count);
        }

        // Count how many seeds beat champion
        var successfulSeeds = seedResults.Where(r => r.TestMetric > championTestMetric).ToList();
        var successCount = successfulSeeds.Count;
        
        // Log each seed's result
        _logger.LogInformation("[MULTI-SEED] {Component}: Multi-seed training results:", componentName);
        foreach (var result in seedResults)
        {
            var status = result.TestMetric > championTestMetric ? "PASS" : "FAIL";
            _logger.LogInformation(
                "[MULTI-SEED]   Seed {Seed}: {Status} - Test metric {TestMetric:F3} vs champion {Champion:F3}",
                result.Seed, status, result.TestMetric, championTestMetric);
        }

        // Make promotion decision
        var approved = successCount >= MinimumSuccessfulSeeds;
        
        if (approved)
        {
            // Select best seed among successful ones
            var bestSeed = successfulSeeds.OrderByDescending(r => r.TestMetric).First();
            
            _logger.LogInformation(
                "[MULTI-SEED] {Component}: PROMOTION APPROVED - {Success}/{Total} seeds succeeded",
                componentName, successCount, seedResults.Count);
            
            _logger.LogInformation(
                "[MULTI-SEED] {Component}: Promoting seed {Seed} with test metric {Metric:F3}",
                componentName, bestSeed.Seed, bestSeed.TestMetric);
            
            return new PromotionDecision
            {
                Approved = true,
                BestSeed = bestSeed.Seed,
                BestTestMetric = bestSeed.TestMetric,
                SuccessfulSeedCount = successCount,
                TotalSeedCount = seedResults.Count,
                Reason = $"{successCount}/{seedResults.Count} seeds beat champion - real learning detected"
            };
        }
        else
        {
            _logger.LogWarning(
                "[MULTI-SEED] {Component}: PROMOTION REJECTED - Only {Success}/{Total} seeds succeeded, likely random luck",
                componentName, successCount, seedResults.Count);
            
            return new PromotionDecision
            {
                Approved = false,
                BestSeed = null,
                BestTestMetric = 0,
                SuccessfulSeedCount = successCount,
                TotalSeedCount = seedResults.Count,
                Reason = $"Only {successCount}/{seedResults.Count} seeds beat champion - insufficient evidence of real learning"
            };
        }
    }

    /// <summary>
    /// Evaluate a single seed's training result
    /// </summary>
    /// <param name="seed">Random seed used</param>
    /// <param name="testMetric">Performance on test set</param>
    /// <param name="validationMetric">Performance on validation set</param>
    /// <param name="modelPath">Path to trained model</param>
    /// <returns>Seed training result</returns>
    public SeedTrainingResult CreateSeedResult(
        int seed,
        double testMetric,
        double validationMetric,
        string modelPath)
    {
        return new SeedTrainingResult
        {
            Seed = seed,
            TestMetric = testMetric,
            ValidationMetric = validationMetric,
            ModelPath = modelPath,
            Timestamp = DateTime.UtcNow
        };
    }
}

/// <summary>
/// Result from training with a single seed
/// </summary>
public sealed class SeedTrainingResult
{
    public required int Seed { get; init; }
    public required double TestMetric { get; init; }
    public required double ValidationMetric { get; init; }
    public required string ModelPath { get; init; }
    public required DateTime Timestamp { get; init; }
}

/// <summary>
/// Promotion decision based on multi-seed results
/// </summary>
public sealed class PromotionDecision
{
    public required bool Approved { get; init; }
    public required int? BestSeed { get; init; }
    public required double BestTestMetric { get; init; }
    public required int SuccessfulSeedCount { get; init; }
    public required int TotalSeedCount { get; init; }
    public required string Reason { get; init; }
}
