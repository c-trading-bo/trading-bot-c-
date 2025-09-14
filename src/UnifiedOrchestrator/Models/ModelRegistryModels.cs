using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Represents a versioned, immutable model artifact
/// </summary>
public class ModelVersion
{
    public string VersionId { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty; // PPO, UCB, LSTM
    public string ArtifactPath { get; set; } = string.Empty;
    public string ArtifactHash { get; set; } = string.Empty;
    public string GitSha { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public string CreatedBy { get; set; } = string.Empty;
    
    // Training metadata
    public DateTime TrainingStartTime { get; set; }
    public DateTime TrainingEndTime { get; set; }
    public string DataRangeStart { get; set; } = string.Empty;
    public string DataRangeEnd { get; set; } = string.Empty;
    
    // Performance metrics
    public decimal Sharpe { get; set; }
    public decimal Sortino { get; set; }
    public decimal CVaR { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal WinRate { get; set; }
    public int TotalTrades { get; set; }
    
    /// <summary>
    /// Additional performance metrics for production-grade tracking
    /// </summary>
    public decimal CalmarRatio { get; set; }
    public decimal UlcerIndex { get; set; }
    public decimal VAR95 { get; set; }
    public decimal StdDeviation { get; set; }
    public decimal SkewnessRatio { get; set; }
    public decimal KurtosisRatio { get; set; }
    public int ConsecutiveLosses { get; set; }
    public int ConsecutiveWins { get; set; }
    public decimal MaxSingleLoss { get; set; }
    public decimal MaxSingleWin { get; set; }
    
    /// <summary>
    /// Training dataset information for reproducibility
    /// </summary>
    public string? DatasetHash { get; set; }
    public long DatasetSizeBytes { get; set; }
    public int DatasetRecordCount { get; set; }
    public string DatasetSource { get; set; } = string.Empty;
    public DateTime DatasetStartTime { get; set; }
    public DateTime DatasetEndTime { get; set; }
    
    /// <summary>
    /// Market conditions during training for context
    /// </summary>
    public string MarketRegime { get; set; } = string.Empty;
    public decimal MarketVolatility { get; set; }
    public decimal MarketTrend { get; set; }
    public string EconomicConditions { get; set; } = string.Empty;
    
    // Model schema information
    public string SchemaVersion { get; set; } = string.Empty;
    public string ModelType { get; set; } = string.Empty; // ONNX, Pickle, Custom
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    // Validation status
    public bool IsValidated { get; set; }
    public bool IsPromoted { get; set; }
    public DateTime? PromotedAt { get; set; }
}

/// <summary>
/// Records a champion promotion event
/// </summary>
public class PromotionRecord
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Algorithm { get; set; } = string.Empty;
    public string FromVersionId { get; set; } = string.Empty;
    public string ToVersionId { get; set; } = string.Empty;
    public DateTime PromotedAt { get; set; } = DateTime.UtcNow;
    public string PromotedBy { get; set; } = string.Empty;
    public string Reason { get; set; } = string.Empty;
    
    // Validation results
    public string ValidationReportPath { get; set; } = string.Empty;
    public bool PassedValidation { get; set; }
    public decimal PerformanceImprovement { get; set; }
    public decimal RiskReduction { get; set; }
    
    // Rollback information
    public bool WasRolledBack { get; set; }
    public DateTime? RolledBackAt { get; set; }
    public string RollbackReason { get; set; } = string.Empty;
    
    // Context at promotion time
    public bool WasFlat { get; set; }
    public string MarketSession { get; set; } = string.Empty; // PRE_MARKET, OPEN, CLOSE, OVERNIGHT
    public Dictionary<string, object> ContextData { get; set; } = new();
}

/// <summary>
/// Promotion decision with validation details
/// </summary>
public class PromotionDecision
{
    public bool ShouldPromote { get; set; }
    public string Reason { get; set; } = string.Empty;
    public List<string> ValidationErrors { get; set; } = new();
    public List<string> RiskConcerns { get; set; } = new();
    
    // Statistical validation
    public decimal PValue { get; set; }
    public decimal ConfidenceInterval { get; set; }
    public bool StatisticallySignificant { get; set; }
    
    // Performance comparison
    public decimal SharpeImprovement { get; set; }
    public decimal SortinoImprovement { get; set; }
    public decimal CVaRImprovement { get; set; }
    public decimal DrawdownImprovement { get; set; }
    
    // Timing validation
    public bool IsInSafeWindow { get; set; }
    public bool IsFlat { get; set; }
    public string NextSafeWindow { get; set; } = string.Empty;
    
    // Resource validation
    public bool HasSufficientMemory { get; set; }
    public bool PassedSchemaValidation { get; set; }
    public bool PassedBehaviorAlignment { get; set; }
}

/// <summary>
/// Shadow testing validation report
/// </summary>
public class ValidationReport
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string ChallengerVersionId { get; set; } = string.Empty;
    public string ChampionVersionId { get; set; } = string.Empty;
    public DateTime TestStartTime { get; set; }
    public DateTime TestEndTime { get; set; }
    
    // Test configuration
    public int MinTrades { get; set; }
    public int MinSessions { get; set; }
    public int ActualTrades { get; set; }
    public int ActualSessions { get; set; }
    
    // Performance comparison
    public decimal ChampionSharpe { get; set; }
    public decimal ChallengerSharpe { get; set; }
    public decimal ChampionSortino { get; set; }
    public decimal ChallengerSortino { get; set; }
    public decimal ChampionCVaR { get; set; }
    public decimal ChallengerCVaR { get; set; }
    public decimal ChampionMaxDrawdown { get; set; }
    public decimal ChallengerMaxDrawdown { get; set; }
    
    // Statistical tests
    public decimal PValue { get; set; }
    public decimal TStatistic { get; set; }
    public bool StatisticallySignificant { get; set; }
    
    // Behavior alignment
    public decimal DecisionAlignment { get; set; } // % of same decisions
    public decimal TimingAlignment { get; set; } // % of similar timing
    public decimal SizeAlignment { get; set; } // % of similar position sizes
    
    // Risk metrics
    public decimal LatencyP95 { get; set; }
    public decimal LatencyP99 { get; set; }
    public decimal MaxMemoryUsage { get; set; }
    public int ErrorCount { get; set; }
    
    // Final assessment
    public bool PassedAllGates { get; set; }
    public List<string> FailureReasons { get; set; } = new();
    public string RecommendedAction { get; set; } = string.Empty;
}