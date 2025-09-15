using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.ModelLifecycle;

/// <summary>
/// Model performance monitoring interface for drift and degradation detection
/// </summary>
public interface IModelPerformanceMonitor
{
    /// <summary>
    /// Record a model prediction and its outcome for performance tracking
    /// </summary>
    Task RecordPredictionAsync(string modelHash, ModelPrediction prediction, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Calculate rolling performance metrics for a model
    /// </summary>
    Task<ModelPerformanceReport> GetPerformanceReportAsync(string modelHash, TimeSpan lookbackPeriod, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if model performance has degraded below thresholds
    /// </summary>
    Task<ModelDegradationAlert?> CheckForDegradationAsync(string modelHash, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Trigger automatic rollback if current model underperforms previous version
    /// </summary>
    Task<bool> TriggerSafeRollbackIfNeededAsync(string currentModelHash, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get comparative performance between two model versions
    /// </summary>
    Task<ModelComparisonReport> CompareModelsAsync(string baselineHash, string candidateHash, CancellationToken cancellationToken = default);
}

/// <summary>
/// Model prediction record for performance tracking
/// </summary>
public record ModelPrediction(
    string ModelHash,
    DateTime Timestamp,
    Dictionary<string, object> Features,
    object Prediction,
    object? ActualOutcome = null,
    double? Confidence = null,
    string? MarketRegime = null,
    string? TradingContext = null
);

/// <summary>
/// Comprehensive model performance report
/// </summary>
public record ModelPerformanceReport(
    string ModelHash,
    TimeSpan ReportPeriod,
    int TotalPredictions,
    double Accuracy,
    double Precision,
    double Recall,
    double F1Score,
    double SharpeRatio,
    double MaxDrawdown,
    Dictionary<string, double> PerformanceByRegime,
    List<PerformanceAlert> Alerts,
    DateTime GeneratedAt
);

/// <summary>
/// Model degradation alert details
/// </summary>
public record ModelDegradationAlert(
    string ModelHash,
    DegradationType Type,
    double CurrentMetric,
    double ThresholdMetric,
    double SeverityScore,
    DateTime DetectedAt,
    string Description
);

/// <summary>
/// Model comparison report for A/B testing and rollback decisions
/// </summary>
public record ModelComparisonReport(
    string BaselineHash,
    string CandidateHash,
    TimeSpan ComparisonPeriod,
    Dictionary<string, ModelMetricComparison> MetricComparisons,
    bool ShouldRollback,
    string Recommendation,
    DateTime GeneratedAt
);

/// <summary>
/// Comparison of a specific metric between models
/// </summary>
public record ModelMetricComparison(
    string MetricName,
    double BaselineValue,
    double CandidateValue,
    double PercentageChange,
    bool IsSignificantDifference,
    double PValue
);

/// <summary>
/// Performance alert levels
/// </summary>
public record PerformanceAlert(
    AlertSeverity Severity,
    string Metric,
    double CurrentValue,
    double ThresholdValue,
    string Message
);

public enum DegradationType
{
    AccuracyDrop,
    PrecisionDrop,
    RecallDrop,
    SharpeRatioDrop,
    DrawdownIncrease,
    PredictionLatency,
    ConfidenceCalibration
}

public enum AlertSeverity
{
    Info,
    Warning,
    Critical
}