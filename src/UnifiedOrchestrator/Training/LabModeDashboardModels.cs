using System;
using System.Collections.Generic;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Strategy-specific training metrics for the Lab Mode dashboard
/// Tracks win rate, PnL, and detailed performance during training
/// </summary>
public sealed class StrategyTrainingMetrics
{
    public string StrategyName { get; set; } = string.Empty;
    public decimal WinRate { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal TotalWon { get; set; }
    public decimal TotalLost { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public int TotalTrades { get; set; }
    public int EpochsCompleted { get; set; }
    public int TotalEpochs { get; set; }
    public double CurrentLoss { get; set; }
    public string ModelVersion { get; set; } = string.Empty;
    public TrainingPhaseStatus Status { get; set; } = TrainingPhaseStatus.Pending;
}

/// <summary>
/// Training phase status for visual indicators
/// </summary>
public enum TrainingPhaseStatus
{
    Pending,
    InProgress,
    Complete,
    Failed
}

/// <summary>
/// Component-level training details
/// </summary>
public sealed class ComponentTrainingDetails
{
    public string ComponentName { get; set; } = string.Empty;
    public string Phase { get; set; } = string.Empty;
    public int EpochsCompleted { get; set; }
    public int TotalEpochs { get; set; }
    public double CurrentLoss { get; set; }
    public double ProgressPercentage { get; set; }
    public string Status { get; set; } = string.Empty;
    public Dictionary<string, object> AdditionalMetrics { get; set; } = new();
}

/// <summary>
/// Complete dashboard state for Lab Mode training session
/// </summary>
public sealed class LabModeDashboardState
{
    public string SessionId { get; set; } = string.Empty;
    public DateTimeOffset SessionStartTime { get; set; }
    public TimeSpan Elapsed { get; set; }
    public TimeSpan EstimatedTimeRemaining { get; set; }
    public string CurrentPhase { get; set; } = string.Empty;
    public double OverallProgress { get; set; }
    public int ComponentsCompleted { get; set; }
    public int TotalComponents { get; set; }
    public int ComponentsRemaining { get; set; }
    
    // Phase-specific details
    public PhaseDetails HeavyPhase { get; set; } = new();
    public PhaseDetails MediumPhase { get; set; } = new();
    public PhaseDetails LightPhase { get; set; } = new();
    
    // Strategy metrics
    public List<StrategyTrainingMetrics> StrategyMetrics { get; set; } = new();
    
    // Current activity
    public ComponentTrainingDetails? CurrentComponent { get; set; }
    
    // System resources
    public ResourceMetrics Resources { get; set; } = new();
    
    // Recent activity
    public List<ActivityLogEntry> RecentActivity { get; set; } = new();
    
    // Alerts (warnings/errors)
    public List<DashboardAlert> ActiveAlerts { get; set; } = new();
}

/// <summary>
/// Details for each training phase (Heavy, Medium, Light)
/// </summary>
public sealed class PhaseDetails
{
    public string PhaseName { get; set; } = string.Empty;
    public TrainingPhaseStatus Status { get; set; } = TrainingPhaseStatus.Pending;
    public TimeSpan? Duration { get; set; }
    public int TotalComponents { get; set; }
    public int CompletedComponents { get; set; }
    public int FailedComponents { get; set; }
    public List<ComponentSummary> Components { get; set; } = new();
    public List<string> QueuedComponentNames { get; set; } = new(); // For pending phases
}

/// <summary>
/// Summary of an individual component's training
/// </summary>
public sealed class ComponentSummary
{
    public string ComponentName { get; set; } = string.Empty;
    public int ComponentNumber { get; set; } // 1-based index (1/11, 2/11, etc.)
    public double ProgressPercentage { get; set; }
    public int EpochsCompleted { get; set; }
    public int TotalEpochs { get; set; }
    public double FinalLoss { get; set; }
    public string Status { get; set; } = string.Empty;
    public TimeSpan Duration { get; set; } // Time taken to train this component
    public int ExperienceCount { get; set; } // Number of experiences used for training
    public Dictionary<string, string> Metrics { get; set; } = new();
}

/// <summary>
/// System resource usage metrics
/// </summary>
public sealed class ResourceMetrics
{
    public double CpuUsagePercent { get; set; }
    public long MemoryUsedMb { get; set; }
    public long MemoryTotalMb { get; set; }
    public double DiskReadMbPerSec { get; set; }
    public double DiskWriteMbPerSec { get; set; }
    public int ActiveProcesses { get; set; }
}

/// <summary>
/// Activity log entry for recent activity section
/// </summary>
public sealed class ActivityLogEntry
{
    public DateTimeOffset Timestamp { get; set; }
    public string LogLevel { get; set; } = string.Empty;
    public string Source { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
}

/// <summary>
/// Alert entry for warnings and errors (displayed in dashboard)
/// </summary>
public sealed class DashboardAlert
{
    public DateTimeOffset Timestamp { get; set; }
    public AlertLevel Level { get; set; }
    public string Source { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public bool IsDismissed { get; set; }
}

/// <summary>
/// Alert severity level
/// </summary>
public enum AlertLevel
{
    Warning,
    Error,
    Critical
}
