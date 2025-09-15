using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.LearningPipeline;

/// <summary>
/// Interface for stratified evaluation with proper train/validation/test separation
/// </summary>
public interface IStratifiedEvaluationManager
{
    /// <summary>
    /// Create stratified splits for training, validation, and testing
    /// </summary>
    Task<StratifiedSplits> CreateStratifiedSplitsAsync(
        string datasetHash, 
        StratificationStrategy strategy, 
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Evaluate a model on stratified test sets
    /// </summary>
    Task<StratifiedEvaluationResults> EvaluateModelAsync(
        string modelHash, 
        StratifiedSplits splits, 
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Record evaluation results for audit and comparison
    /// </summary>
    Task RecordEvaluationResultsAsync(StratifiedEvaluationResults results, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get historical evaluation results for a model
    /// </summary>
    Task<List<StratifiedEvaluationResults>> GetEvaluationHistoryAsync(string modelHash, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validate that splits maintain proper stratification
    /// </summary>
    Task<StratificationValidationResult> ValidateStratificationAsync(StratifiedSplits splits, CancellationToken cancellationToken = default);
}

/// <summary>
/// Stratified dataset splits for proper evaluation
/// </summary>
public record StratifiedSplits(
    string SplitsId,
    string DatasetHash,
    DatasetSplit TrainingSplit,
    DatasetSplit ValidationSplit,
    DatasetSplit TestSplit,
    StratificationStrategy Strategy,
    DateTime CreatedAt
);

/// <summary>
/// Individual dataset split
/// </summary>
public record DatasetSplit(
    string SplitId,
    List<int> RecordIndices,
    double Proportion,
    Dictionary<string, int> StratificationCounts
);

/// <summary>
/// Stratification strategy configuration
/// </summary>
public record StratificationStrategy(
    string StratificationColumn,
    double TrainProportion = 0.7,
    double ValidationProportion = 0.15,
    double TestProportion = 0.15,
    int? RandomSeed = null,
    bool EnsureMinimumSamples = true,
    int MinimumSamplesPerStratum = 5
);

/// <summary>
/// Results of stratified evaluation
/// </summary>
public record StratifiedEvaluationResults(
    string EvaluationId,
    string ModelHash,
    string SplitsId,
    DateTime EvaluatedAt,
    EvaluationMetrics TrainingMetrics,
    EvaluationMetrics ValidationMetrics,
    EvaluationMetrics TestMetrics,
    Dictionary<string, EvaluationMetrics> StratumMetrics,
    List<string> Warnings
);

/// <summary>
/// Evaluation metrics for a dataset split
/// </summary>
public record EvaluationMetrics(
    double Accuracy,
    double Precision,
    double Recall,
    double F1Score,
    double AUC,
    Dictionary<string, double> ClassMetrics,
    ConfusionMatrix ConfusionMatrix,
    int SampleCount
);

/// <summary>
/// Confusion matrix for classification evaluation
/// </summary>
public record ConfusionMatrix(
    Dictionary<string, Dictionary<string, int>> Matrix,
    List<string> ClassLabels
);

/// <summary>
/// Validation result for stratification quality
/// </summary>
public record StratificationValidationResult(
    bool IsValid,
    List<string> Errors,
    List<string> Warnings,
    Dictionary<string, StratificationQualityMetrics> QualityMetrics
);

/// <summary>
/// Quality metrics for stratification
/// </summary>
public record StratificationQualityMetrics(
    string Stratum,
    double ExpectedProportion,
    double ActualProportion,
    double ProportionDeviation,
    int SampleCount,
    bool MeetsMinimumSamples
);