using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Phase 5: Atomic promotion result with backup tracking
/// </summary>
public sealed class AtomicPromotionResult
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }

    [JsonPropertyName("sessionId")]
    public string SessionId { get; set; } = string.Empty;

    [JsonPropertyName("promotionTime")]
    public DateTime PromotionTime { get; set; }

    [JsonPropertyName("modelsPromoted")]
    public int ModelsPromoted { get; set; }

    [JsonPropertyName("backupCreated")]
    public bool BackupCreated { get; set; }

    [JsonPropertyName("backupLocation")]
    public string BackupLocation { get; set; } = string.Empty;

    [JsonPropertyName("totalSizeBytes")]
    public long TotalSizeBytes { get; set; }

    [JsonPropertyName("promotionDurationMs")]
    public double PromotionDurationMs { get; set; }

    [JsonPropertyName("rollbackCapable")]
    public bool RollbackCapable { get; set; }

    [JsonPropertyName("issues")]
    public List<string> Issues { get; set; } = new();

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; set; } = new();
}

/// <summary>
/// Phase 5: Enhanced promotion criteria evaluation result
/// </summary>
public sealed class EnhancedPromotionCriteria
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("trainingSuccess")]
    public TrainingSuccessCriteria TrainingSuccess { get; set; } = new();

    [JsonPropertyName("validationSuccess")]
    public ValidationSuccessCriteria ValidationSuccess { get; set; } = new();

    [JsonPropertyName("performanceCriteria")]
    public PerformanceCriteria PerformanceCriteria { get; set; } = new();

    [JsonPropertyName("technicalCriteria")]
    public TechnicalCriteria TechnicalCriteria { get; set; } = new();

    [JsonPropertyName("operationalCriteria")]
    public OperationalCriteria OperationalCriteria { get; set; } = new();

    [JsonPropertyName("failedCriteria")]
    public List<string> FailedCriteria { get; set; } = new();
}

/// <summary>
/// Training success criteria
/// </summary>
public sealed class TrainingSuccessCriteria
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("componentsExpected")]
    public int ComponentsExpected { get; set; }

    [JsonPropertyName("componentsTrained")]
    public int ComponentsTrained { get; set; }

    [JsonPropertyName("completedWithinTimeWindow")]
    public bool CompletedWithinTimeWindow { get; set; }

    [JsonPropertyName("trainingDurationHours")]
    public double TrainingDurationHours { get; set; }

    [JsonPropertyName("noTrainingCrashes")]
    public bool NoTrainingCrashes { get; set; }

    [JsonPropertyName("allModelsSavedToStaging")]
    public bool AllModelsSavedToStaging { get; set; }
}

/// <summary>
/// Validation success criteria
/// </summary>
public sealed class ValidationSuccessCriteria
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("inferenceTestsPassed")]
    public bool InferenceTestsPassed { get; set; }

    [JsonPropertyName("baselineComparisonPositive")]
    public bool BaselineComparisonPositive { get; set; }

    [JsonPropertyName("noCatastrophicForgetting")]
    public bool NoCatastrophicForgetting { get; set; }

    [JsonPropertyName("modelIntegrityVerified")]
    public bool ModelIntegrityVerified { get; set; }

    [JsonPropertyName("allChecksPassedCount")]
    public int AllChecksPassedCount { get; set; }

    [JsonPropertyName("totalChecksCount")]
    public int TotalChecksCount { get; set; }
}

/// <summary>
/// Performance criteria
/// </summary>
public sealed class PerformanceCriteria
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("averageImprovementPercent")]
    public double AverageImprovementPercent { get; set; }

    [JsonPropertyName("noCriticalRegression")]
    public bool NoCriticalRegression { get; set; }

    [JsonPropertyName("cvarPpoImproved")]
    public bool CVarPpoImproved { get; set; }

    [JsonPropertyName("neuralUcbImproved")]
    public bool NeuralUcbImproved { get; set; }

    [JsonPropertyName("maxRegressionPercent")]
    public double MaxRegressionPercent { get; set; }
}

/// <summary>
/// Technical criteria
/// </summary>
public sealed class TechnicalCriteria
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("totalModelSizeGB")]
    public double TotalModelSizeGB { get; set; }

    [JsonPropertyName("withinSizeLimit")]
    public bool WithinSizeLimit { get; set; }

    [JsonPropertyName("onnxRuntimeCompatible")]
    public bool OnnxRuntimeCompatible { get; set; }

    [JsonPropertyName("noDependencyConflicts")]
    public bool NoDependencyConflicts { get; set; }
}

/// <summary>
/// Operational criteria
/// </summary>
public sealed class OperationalCriteria
{
    [JsonPropertyName("passed")]
    public bool Passed { get; set; }

    [JsonPropertyName("trainingWindowRespected")]
    public bool TrainingWindowRespected { get; set; }

    [JsonPropertyName("systemHealthGood")]
    public bool SystemHealthGood { get; set; }

    [JsonPropertyName("noConcurrentTraining")]
    public bool NoConcurrentTraining { get; set; }

    [JsonPropertyName("lockFileRemoved")]
    public bool LockFileRemoved { get; set; }

    [JsonPropertyName("sufficientDiskSpaceForBackup")]
    public bool SufficientDiskSpaceForBackup { get; set; }
}

/// <summary>
/// Promotion report for Phase 5
/// </summary>
public sealed class PromotionReport
{
    [JsonPropertyName("sessionId")]
    public string SessionId { get; set; } = string.Empty;

    [JsonPropertyName("promotionTime")]
    public DateTime PromotionTime { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty; // SUCCESS, FAILED, ROLLED_BACK

    [JsonPropertyName("criteria")]
    public EnhancedPromotionCriteria Criteria { get; set; } = new();

    [JsonPropertyName("atomicResult")]
    public AtomicPromotionResult AtomicResult { get; set; } = new();

    [JsonPropertyName("summary")]
    public string Summary { get; set; } = string.Empty;

    [JsonPropertyName("modelsPromoted")]
    public List<string> ModelsPromoted { get; set; } = new();

    [JsonPropertyName("rollbackAvailable")]
    public bool RollbackAvailable { get; set; }

    [JsonPropertyName("recommendations")]
    public List<string> Recommendations { get; set; } = new();
}

/// <summary>
/// Rollback result
/// </summary>
public sealed class RollbackResult
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }

    [JsonPropertyName("rollbackTime")]
    public DateTime RollbackTime { get; set; }

    [JsonPropertyName("rollbackDurationMs")]
    public double RollbackDurationMs { get; set; }

    [JsonPropertyName("modelsRestored")]
    public int ModelsRestored { get; set; }

    [JsonPropertyName("backupSource")]
    public string BackupSource { get; set; } = string.Empty;

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;

    [JsonPropertyName("issues")]
    public List<string> Issues { get; set; } = new();
}
