using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.LearningPipeline;

/// <summary>
/// Interface for A/B shadow evaluation of candidate models
/// </summary>
public interface IShadowEvaluationManager
{
    /// <summary>
    /// Start a shadow evaluation comparing candidate model to live model
    /// </summary>
    Task<string> StartShadowEvaluationAsync(ShadowEvaluationConfiguration config, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Record a shadow prediction for comparison
    /// </summary>
    Task RecordShadowPredictionAsync(string evaluationId, ShadowPrediction prediction, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current shadow evaluation results
    /// </summary>
    Task<ShadowEvaluationResults> GetShadowEvaluationResultsAsync(string evaluationId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Complete a shadow evaluation and generate final report
    /// </summary>
    Task<ShadowEvaluationReport> CompleteShadowEvaluationAsync(string evaluationId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// List active shadow evaluations
    /// </summary>
    Task<List<ShadowEvaluationStatus>> ListActiveShadowEvaluationsAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Cancel a running shadow evaluation
    /// </summary>
    Task CancelShadowEvaluationAsync(string evaluationId, CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for shadow evaluation
/// </summary>
public record ShadowEvaluationConfiguration(
    string EvaluationId,
    string LiveModelHash,
    string CandidateModelHash,
    TimeSpan Duration,
    double TrafficPercentage,
    ShadowEvaluationCriteria Criteria,
    string? Description = null
);

/// <summary>
/// Criteria for shadow evaluation success
/// </summary>
public record ShadowEvaluationCriteria(
    double MinimumAccuracyImprovement = 0.01,
    double MinimumSharpeRatioImprovement = 0.1,
    double MaximumLatencyIncrease = 0.2,
    int MinimumSampleSize = 1000,
    double SignificanceLevel = 0.05
);

/// <summary>
/// Shadow prediction record for comparison
/// </summary>
public record ShadowPrediction(
    string EvaluationId,
    DateTime Timestamp,
    Dictionary<string, object> Features,
    object LivePrediction,
    object CandidatePrediction,
    double? LiveConfidence = null,
    double? CandidateConfidence = null,
    TimeSpan LiveLatency = default,
    TimeSpan CandidateLatency = default,
    object? ActualOutcome = null,
    string? MarketRegime = null
);

/// <summary>
/// Current results of shadow evaluation
/// </summary>
public record ShadowEvaluationResults(
    string EvaluationId,
    DateTime StartTime,
    DateTime LastUpdate,
    int TotalPredictions,
    int PredictionsWithOutcomes,
    ModelPerformanceComparison PerformanceComparison,
    LatencyComparison LatencyComparison,
    StatisticalSignificance Significance,
    EvaluationProgress Progress
);

/// <summary>
/// Performance comparison between live and candidate models
/// </summary>
public record ModelPerformanceComparison(
    ModelPerformanceSummary LiveModel,
    ModelPerformanceSummary CandidateModel,
    double AccuracyDifference,
    double PrecisionDifference,
    double RecallDifference,
    double SharpeRatioDifference
);

/// <summary>
/// Performance summary for a model
/// </summary>
public record ModelPerformanceSummary(
    string ModelHash,
    double Accuracy,
    double Precision,
    double Recall,
    double F1Score,
    double SharpeRatio,
    double AverageConfidence,
    int PredictionCount
);

/// <summary>
/// Latency comparison between models
/// </summary>
public record LatencyComparison(
    TimeSpan LiveModelMedianLatency,
    TimeSpan CandidateModelMedianLatency,
    TimeSpan LiveModelP95Latency,
    TimeSpan CandidateModelP95Latency,
    double LatencyDifferencePercentage
);

/// <summary>
/// Statistical significance of observed differences
/// </summary>
public record StatisticalSignificance(
    double AccuracyPValue,
    double SharpeRatioPValue,
    double LatencyPValue,
    bool IsAccuracySignificant,
    bool IsSharpeRatioSignificant,
    bool IsLatencySignificant
);

/// <summary>
/// Progress of shadow evaluation
/// </summary>
public record EvaluationProgress(
    double CompletionPercentage,
    TimeSpan Elapsed,
    TimeSpan EstimatedRemaining,
    bool HasSufficientSamples,
    List<string> Milestones
);

/// <summary>
/// Final shadow evaluation report
/// </summary>
public record ShadowEvaluationReport(
    string EvaluationId,
    DateTime StartTime,
    DateTime EndTime,
    ShadowEvaluationConfiguration Configuration,
    ShadowEvaluationResults FinalResults,
    EvaluationRecommendation Recommendation,
    List<string> KeyFindings,
    Dictionary<string, object> DetailedAnalysis
);

/// <summary>
/// Recommendation from shadow evaluation
/// </summary>
public record EvaluationRecommendation(
    RecommendationType Type,
    double ConfidenceScore,
    string Reasoning,
    List<string> Requirements,
    List<string> RiskFactors
);

/// <summary>
/// Status of shadow evaluation
/// </summary>
public record ShadowEvaluationStatus(
    string EvaluationId,
    DateTime StartTime,
    EvaluationState State,
    double Progress,
    int PredictionCount,
    string? ErrorMessage = null
);

public enum RecommendationType
{
    PromoteCandidate,
    RejectCandidate,
    ExtendEvaluation,
    RequiresManualReview
}

public enum EvaluationState
{
    Starting,
    Running,
    Completed,
    Cancelled,
    Failed
}