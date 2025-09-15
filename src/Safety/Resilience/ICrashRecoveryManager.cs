using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Resilience;

/// <summary>
/// Interface for automated crash recovery and system health checks
/// </summary>
public interface ICrashRecoveryManager
{
    /// <summary>
    /// Start crash recovery monitoring
    /// </summary>
    Task StartMonitoringAsync(CrashRecoveryConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Perform comprehensive system health check
    /// </summary>
    Task<SystemHealthReport> PerformHealthCheckAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Execute automated recovery sequence
    /// </summary>
    Task<RecoveryResult> ExecuteRecoverySequenceAsync(CrashRecoveryReason reason, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register a subsystem for health monitoring
    /// </summary>
    Task RegisterSubsystemAsync(SubsystemHealthCheck subsystem);
    
    /// <summary>
    /// Get recovery history and statistics
    /// </summary>
    Task<List<RecoveryEvent>> GetRecoveryHistoryAsync(TimeSpan lookbackPeriod, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Force system restart with specified reason
    /// </summary>
    Task<bool> ForceSystemRestartAsync(string reason, bool emergency = false, CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for crash recovery behavior
/// </summary>
public record CrashRecoveryConfiguration(
    TimeSpan HealthCheckInterval,
    int MaxRecoveryAttempts = 3,
    TimeSpan RecoveryTimeout = default,
    bool EnableAutoRestart = true,
    bool EnableSubsystemRecovery = true,
    Dictionary<string, object>? CustomRecoveryActions = null
)
{
    public TimeSpan RecoveryTimeout { get; init; } = RecoveryTimeout == default ? TimeSpan.FromMinutes(5) : RecoveryTimeout;
}

/// <summary>
/// Comprehensive system health report
/// </summary>
public record SystemHealthReport(
    DateTime GeneratedAt,
    SystemHealthStatus OverallStatus,
    Dictionary<string, SubsystemHealthResult> SubsystemResults,
    List<HealthIssue> CriticalIssues,
    List<HealthIssue> Warnings,
    SystemResourceMetrics ResourceMetrics,
    bool RequiresRecovery,
    string? RecommendedAction = null
);

/// <summary>
/// Overall system health status
/// </summary>
public enum SystemHealthStatus
{
    Healthy,
    Degraded,
    Critical,
    Failed
}

/// <summary>
/// Health check result for individual subsystem
/// </summary>
public record SubsystemHealthResult(
    string SubsystemName,
    SubsystemHealthStatus Status,
    DateTime LastChecked,
    TimeSpan ResponseTime,
    List<string> Issues,
    Dictionary<string, object>? Metrics = null
);

/// <summary>
/// Health status for subsystems
/// </summary>
public enum SubsystemHealthStatus
{
    Online,
    Degraded,
    Offline,
    Unknown
}

/// <summary>
/// Health issue details
/// </summary>
public record HealthIssue(
    string Category,
    string Description,
    HealthIssueSeverity Severity,
    DateTime DetectedAt,
    string? RecommendedAction = null,
    Dictionary<string, object>? Context = null
);

/// <summary>
/// Severity levels for health issues
/// </summary>
public enum HealthIssueSeverity
{
    Info,
    Warning,
    Error,
    Critical
}

/// <summary>
/// System resource metrics
/// </summary>
public record SystemResourceMetrics(
    double CpuUsagePercentage,
    long MemoryUsageBytes,
    long MemoryAvailableBytes,
    double DiskUsagePercentage,
    int ThreadCount,
    int HandleCount,
    TimeSpan Uptime
);

/// <summary>
/// Result of recovery operation
/// </summary>
public record RecoveryResult(
    bool Success,
    TimeSpan Duration,
    List<RecoveryAction> ActionsPerformed,
    string? ErrorMessage = null,
    Dictionary<string, object>? RecoveryMetrics = null
);

/// <summary>
/// Individual recovery action
/// </summary>
public record RecoveryAction(
    string ActionName,
    bool Success,
    TimeSpan Duration,
    string? Output = null,
    string? Error = null
);

/// <summary>
/// Recovery event for history tracking
/// </summary>
public record RecoveryEvent(
    DateTime Timestamp,
    CrashRecoveryReason Reason,
    RecoveryResult Result,
    SystemHealthReport PreRecoveryHealth,
    SystemHealthReport? PostRecoveryHealth = null
);

/// <summary>
/// Reasons for crash recovery
/// </summary>
public enum CrashRecoveryReason
{
    SystemCrash,
    MemoryLeak,
    ConnectivityLoss,
    SubsystemFailure,
    PerformanceDegradation,
    ManualTrigger,
    ScheduledMaintenance
}

/// <summary>
/// Subsystem health check definition
/// </summary>
public record SubsystemHealthCheck(
    string Name,
    Func<CancellationToken, Task<SubsystemHealthResult>> HealthCheckFunction,
    TimeSpan Timeout,
    bool IsCritical = true,
    Dictionary<string, object>? Configuration = null
);