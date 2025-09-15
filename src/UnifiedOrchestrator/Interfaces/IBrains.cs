using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Read-only inference brain interface - no training or parameter mutation allowed
/// </summary>
public interface IInferenceBrain
{
    /// <summary>
    /// Make a trading decision using current champion models (read-only)
    /// </summary>
    Task<TradingDecision> DecideAsync(TradingContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current champion model versions for all algorithms
    /// </summary>
    Task<Dictionary<string, ModelVersion?>> GetChampionVersionsAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if all champion models are loaded and ready
    /// </summary>
    Task<bool> IsReadyAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get inference statistics
    /// </summary>
    Task<InferenceStats> GetStatsAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Write-only training brain interface - produces versioned artifacts
/// </summary>
public interface ITrainingBrain
{
    /// <summary>
    /// Train a new challenger model for an algorithm
    /// </summary>
    Task<TrainingResult> TrainChallengerAsync(string algorithm, TrainingConfig config, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Export a trained model to artifacts
    /// </summary>
    Task<ModelVersion> ExportModelAsync(string algorithm, string modelPath, TrainingMetadata metadata, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get training job status
    /// </summary>
    Task<TrainingStatus> GetTrainingStatusAsync(string jobId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Cancel a training job
    /// </summary>
    Task<bool> CancelTrainingAsync(string jobId, CancellationToken cancellationToken = default);
}

/// <summary>
/// Trading decision output from inference brain
/// </summary>
public class TradingDecision
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Action { get; set; } = "HOLD"; // BUY, SELL, HOLD
    public decimal Size { get; set; }
    public decimal Confidence { get; set; }
    public string Strategy { get; set; } = string.Empty;
    
    // Model attribution
    public string PPOVersionId { get; set; } = string.Empty;
    public string UCBVersionId { get; set; } = string.Empty;
    public string LSTMVersionId { get; set; } = string.Empty;
    public Dictionary<string, string> AlgorithmVersions { get; set; } = new();
    public Dictionary<string, string> AlgorithmHashes { get; set; } = new();
    
    // Decision metadata
    public decimal ProcessingTimeMs { get; set; }
    public Dictionary<string, decimal> AlgorithmConfidences { get; set; } = new();
    public Dictionary<string, object> DecisionMetadata { get; set; } = new();
    
    // Compatibility alias
    public Dictionary<string, object> Metadata => DecisionMetadata;
    
    // Compatibility alias for legacy code
    public Dictionary<string, object> Reasoning => DecisionMetadata;
    
    // Risk assessment
    public bool PassedRiskChecks { get; set; }
    public List<string> RiskWarnings { get; set; } = new();
}

/// <summary>
/// Inference brain statistics
/// </summary>
public class InferenceStats
{
    public int TotalDecisions { get; set; }
    public int DecisionsToday { get; set; }
    public decimal AverageProcessingTimeMs { get; set; }
    public decimal MaxProcessingTimeMs { get; set; }
    public DateTime LastDecisionTime { get; set; }
    
    // Model health
    public Dictionary<string, bool> ModelHealth { get; set; } = new();
    public Dictionary<string, DateTime> ModelLoadTimes { get; set; } = new();
    public Dictionary<string, int> ModelSwapCounts { get; set; } = new();
    
    // Performance
    public decimal TodayWinRate { get; set; }
    public decimal TodayPnL { get; set; }
    public int ErrorCount { get; set; }
    public DateTime StartTime { get; set; }
}

/// <summary>
/// Training configuration
/// </summary>
public class TrainingConfig
{
    public string Algorithm { get; set; } = string.Empty;
    public DateTime DataStartTime { get; set; }
    public DateTime DataEndTime { get; set; }
    public Dictionary<string, object> Parameters { get; set; } = new();
    public string DataSource { get; set; } = string.Empty;
    public int MaxEpochs { get; set; } = 100;
    public decimal LearningRate { get; set; } = 0.001m;
    public int BatchSize { get; set; } = 32;
    public string OutputPath { get; set; } = string.Empty;
}

/// <summary>
/// Training result
/// </summary>
public class TrainingResult
{
    public string JobId { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string ErrorMessage { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public TimeSpan TrainingDuration { get; set; }
    
    // Training metrics
    public decimal FinalLoss { get; set; }
    public decimal BestValidationScore { get; set; }
    public int EpochsCompleted { get; set; }
    public Dictionary<string, decimal> Metrics { get; set; } = new();
    
    // Model metadata
    public TrainingMetadata Metadata { get; set; } = new();
}

/// <summary>
/// Training metadata
/// </summary>
public class TrainingMetadata
{
    public DateTime TrainingStartTime { get; set; }
    public DateTime TrainingEndTime { get; set; }
    public string DataRangeStart { get; set; } = string.Empty;
    public string DataRangeEnd { get; set; } = string.Empty;
    public int DataSamples { get; set; }
    public string GitSha { get; set; } = string.Empty;
    public string CreatedBy { get; set; } = Environment.UserName;
    public Dictionary<string, object> Parameters { get; set; } = new();
    public Dictionary<string, decimal> PerformanceMetrics { get; set; } = new();
}

/// <summary>
/// Training job status
/// </summary>
public class TrainingStatus
{
    public string JobId { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public string Status { get; set; } = "UNKNOWN"; // QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
    public decimal Progress { get; set; } // 0.0 to 1.0
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public string CurrentStage { get; set; } = string.Empty;
    public Dictionary<string, object> StageData { get; set; } = new();
    public List<string> Logs { get; set; } = new();
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// Interface for model training services
/// </summary>
public interface IModelTrainingService
{
    /// <summary>
    /// Train a challenger model based on promising backtest results
    /// </summary>
    Task<string> TrainChallengerAsync(ChallengerTrainingRequest request, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get training status
    /// </summary>
    Task<TrainingStatus> GetTrainingStatusAsync(string trainingId, CancellationToken cancellationToken = default);
}

/// <summary>
/// Request for challenger model training
/// </summary>
public class ChallengerTrainingRequest
{
    public string BaseModelVersion { get; set; } = string.Empty;
    public double TargetSharpe { get; set; }
    public Dictionary<string, object> SuccessfulPatterns { get; set; } = new();
    public TimeSpan TrainingDataPeriod { get; set; }
    public DateTime Timestamp { get; set; }
}