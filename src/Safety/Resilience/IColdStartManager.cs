using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Resilience;

/// <summary>
/// Interface for managing cold start warm-up phases with gating controls
/// </summary>
public interface IColdStartManager
{
    /// <summary>
    /// Start the cold start process with defined phases
    /// </summary>
    Task StartColdStartAsync(ColdStartConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current cold start status
    /// </summary>
    Task<ColdStartStatus> GetColdStartStatusAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Force progression to next phase (admin override)
    /// </summary>
    Task ForcePhaseProgressionAsync(ColdStartPhase targetPhase, string reason, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if system is ready for trading based on cold start gates
    /// </summary>
    Task<bool> IsReadyForTradingAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register a readiness check for a specific subsystem
    /// </summary>
    Task RegisterReadinessCheckAsync(string subsystemName, Func<CancellationToken, Task<ReadinessCheckResult>> readinessCheck);
    
    /// <summary>
    /// Get detailed readiness report for all subsystems
    /// </summary>
    Task<ReadinessReport> GetReadinessReportAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for cold start process
/// </summary>
public record ColdStartConfiguration(
    TimeSpan WarmupDuration,
    TimeSpan ObservationDuration,
    int MinimumDataPoints,
    double MinimumAccuracyThreshold,
    bool RequireModelValidation = true,
    bool RequireDataQualityChecks = true,
    bool RequireConnectivityChecks = true,
    Dictionary<string, object>? CustomGates = null
);

/// <summary>
/// Current status of cold start process
/// </summary>
public record ColdStartStatus(
    ColdStartPhase CurrentPhase,
    DateTime StartTime,
    DateTime? PhaseStartTime,
    TimeSpan Elapsed,
    double ProgressPercentage,
    List<GateStatus> GateStatuses,
    List<string> PendingRequirements,
    bool CanProgressToNextPhase,
    string? BlockingReason = null
);

/// <summary>
/// Cold start phases
/// </summary>
public enum ColdStartPhase
{
    Initializing,
    Warmup,
    Observing,
    ReadyToTrade,
    TradingActive
}

/// <summary>
/// Status of individual gates
/// </summary>
public record GateStatus(
    string GateName,
    bool IsOpen,
    DateTime LastChecked,
    string Status,
    List<string> Requirements,
    double CompletionPercentage
);

/// <summary>
/// Result of a readiness check
/// </summary>
public record ReadinessCheckResult(
    string SubsystemName,
    bool IsReady,
    double HealthScore,
    List<string> Issues,
    Dictionary<string, object>? Metrics = null,
    DateTime CheckedAt = default
);

/// <summary>
/// Comprehensive readiness report
/// </summary>
public record ReadinessReport(
    DateTime GeneratedAt,
    bool OverallReady,
    double OverallHealthScore,
    Dictionary<string, ReadinessCheckResult> SubsystemResults,
    List<string> CriticalIssues,
    List<string> Warnings,
    TimeSpan? EstimatedTimeToReady = null
);