using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Phase 4: Post-training validation results
/// Comprehensive validation of trained models before promotion
/// </summary>
public sealed class PostTrainingValidationResult
{
    [JsonPropertyName("sessionId")]
    public string SessionId { get; set; } = string.Empty;

    [JsonPropertyName("validationTime")]
    public DateTime ValidationTime { get; set; } = DateTime.UtcNow;

    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("inferenceTests")]
    public InferenceTestResults InferenceTests { get; set; } = new();

    [JsonPropertyName("baselineComparison")]
    public BaselineComparisonResults BaselineComparison { get; set; } = new();

    [JsonPropertyName("catastrophicForgetting")]
    public CatastrophicForgettingResults CatastrophicForgetting { get; set; } = new();

    [JsonPropertyName("modelIntegrity")]
    public ModelIntegrityResults ModelIntegrity { get; set; } = new();

    [JsonPropertyName("promotionDecision")]
    public PostTrainingPromotionDecision PromotionDecision { get; set; } = new();

    [JsonPropertyName("issues")]
    public List<string> Issues { get; set; } = new();

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; set; } = new();
}

/// <summary>
/// Results from inference testing on validation dataset
/// </summary>
public sealed class InferenceTestResults
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("modelsLoaded")]
    public int ModelsLoaded { get; set; }

    [JsonPropertyName("modelsExpected")]
    public int ModelsExpected { get; set; }

    [JsonPropertyName("avgLatencyMs")]
    public double AverageLatencyMs { get; set; }

    [JsonPropertyName("maxLatencyMs")]
    public double MaxLatencyMs { get; set; }

    [JsonPropertyName("errors")]
    public int ErrorCount { get; set; }

    [JsonPropertyName("modelResults")]
    public List<ModelInferenceResult> ModelResults { get; set; } = new();
}

/// <summary>
/// Inference results for a single model
/// </summary>
public sealed class ModelInferenceResult
{
    [JsonPropertyName("modelName")]
    public string ModelName { get; set; } = string.Empty;

    [JsonPropertyName("modelType")]
    public string ModelType { get; set; } = string.Empty;

    [JsonPropertyName("loaded")]
    public bool Loaded { get; set; }

    [JsonPropertyName("avgLatencyMs")]
    public double AverageLatencyMs { get; set; }

    [JsonPropertyName("validOutputs")]
    public int ValidOutputs { get; set; }

    [JsonPropertyName("totalOutputs")]
    public int TotalOutputs { get; set; }

    [JsonPropertyName("hasNaN")]
    public bool HasNaN { get; set; }

    [JsonPropertyName("hasInf")]
    public bool HasInf { get; set; }
    
    [JsonPropertyName("maxLatencyMs")]
    public double MaxLatencyMs { get; set; }

    [JsonPropertyName("errors")]
    public List<string> Errors { get; set; } = new();
}

/// <summary>
/// Results from comparing new models against baseline
/// </summary>
public sealed class BaselineComparisonResults
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("baselineFound")]
    public bool BaselineFound { get; set; }

    [JsonPropertyName("avgImprovement")]
    public double AverageImprovement { get; set; }

    [JsonPropertyName("regressions")]
    public int RegressionCount { get; set; }

    [JsonPropertyName("modelComparisons")]
    public List<ModelComparison> ModelComparisons { get; set; } = new();
}

/// <summary>
/// Comparison of a single model against baseline
/// </summary>
public sealed class ModelComparison
{
    [JsonPropertyName("modelName")]
    public string ModelName { get; set; } = string.Empty;

    [JsonPropertyName("modelType")]
    public string ModelType { get; set; } = string.Empty;

    [JsonPropertyName("metric")]
    public string Metric { get; set; } = string.Empty;

    [JsonPropertyName("baselineValue")]
    public double BaselineValue { get; set; }

    [JsonPropertyName("newValue")]
    public double NewValue { get; set; }

    [JsonPropertyName("improvement")]
    public double Improvement { get; set; }

    [JsonPropertyName("improvementPercent")]
    public double ImprovementPercent { get; set; }

    [JsonPropertyName("isRegression")]
    public bool IsRegression { get; set; }
}

/// <summary>
/// Results from catastrophic forgetting detection
/// </summary>
public sealed class CatastrophicForgettingResults
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("recentPerformance")]
    public double RecentPerformance { get; set; }

    [JsonPropertyName("midTermPerformance")]
    public double MidTermPerformance { get; set; }

    [JsonPropertyName("longTermPerformance")]
    public double LongTermPerformance { get; set; }

    [JsonPropertyName("degradationPercent")]
    public double DegradationPercent { get; set; }

    [JsonPropertyName("modelsAffected")]
    public List<string> ModelsAffected { get; set; } = new();
}

/// <summary>
/// Results from model file integrity checks
/// </summary>
public sealed class ModelIntegrityResults
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("modelsChecked")]
    public int ModelsChecked { get; set; }

    [JsonPropertyName("checksumVerified")]
    public int ChecksumVerified { get; set; }

    [JsonPropertyName("corruptedModels")]
    public List<string> CorruptedModels { get; set; } = new();
}

/// <summary>
/// Promotion decision after post-training validation (Phase 4)
/// </summary>
public sealed class PostTrainingPromotionDecision
{
    [JsonPropertyName("promoted")]
    public bool Promoted { get; set; }

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;

    [JsonPropertyName("promotedAt")]
    public DateTime? PromotedAt { get; set; }

    [JsonPropertyName("modelsPromoted")]
    public int ModelsPromoted { get; set; }
}

/// <summary>
/// Post-training validation report for human consumption (Phase 4)
/// </summary>
public sealed class PostTrainingValidationReport
{
    [JsonPropertyName("sessionId")]
    public string SessionId { get; set; } = string.Empty;

    [JsonPropertyName("trainingDate")]
    public DateTime TrainingDate { get; set; }

    [JsonPropertyName("completionDate")]
    public DateTime CompletionDate { get; set; }

    [JsonPropertyName("durationSeconds")]
    public int DurationSeconds { get; set; }

    [JsonPropertyName("totalComponents")]
    public int TotalComponents { get; set; }

    [JsonPropertyName("successfulComponents")]
    public int SuccessfulComponents { get; set; }

    [JsonPropertyName("failedComponents")]
    public int FailedComponents { get; set; }

    [JsonPropertyName("validationResults")]
    public PostTrainingValidationResult ValidationResults { get; set; } = new();

    [JsonPropertyName("summary")]
    public string Summary { get; set; } = string.Empty;

    [JsonPropertyName("detailedFindings")]
    public List<string> DetailedFindings { get; set; } = new();

    [JsonPropertyName("recommendations")]
    public List<string> Recommendations { get; set; } = new();
}
