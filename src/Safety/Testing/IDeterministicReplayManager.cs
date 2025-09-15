using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Testing;

/// <summary>
/// Interface for deterministic replay testing for regression validation
/// </summary>
public interface IDeterministicReplayManager
{
    /// <summary>
    /// Record execution session for later replay
    /// </summary>
    Task<string> StartRecordingSessionAsync(RecordingConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Stop recording and finalize session
    /// </summary>
    Task<RecordingResult> StopRecordingSessionAsync(string sessionId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Replay recorded session deterministically
    /// </summary>
    Task<ReplayResult> ReplaySessionAsync(ReplayConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Compare replay results with original recording
    /// </summary>
    Task<ReplayComparisonResult> CompareReplayResultsAsync(string originalSessionId, string replaySessionId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// List available recorded sessions
    /// </summary>
    Task<List<RecordedSession>> ListRecordedSessionsAsync(SessionFilter? filter = null, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validate session integrity and completeness
    /// </summary>
    Task<SessionValidationResult> ValidateSessionAsync(string sessionId, CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for recording sessions
/// </summary>
public record RecordingConfiguration(
    string SessionId,
    RecordingScope Scope,
    TimeSpan MaxDuration,
    bool RecordMarketData = true,
    bool RecordDecisions = true,
    bool RecordOrders = true,
    bool RecordSystemState = true,
    Dictionary<string, object>? CustomFilters = null
);

/// <summary>
/// Scope of data to record
/// </summary>
public enum RecordingScope
{
    MarketDataOnly,
    TradingDecisions,
    FullSystem,
    Custom
}

/// <summary>
/// Result of recording session
/// </summary>
public record RecordingResult(
    string SessionId,
    DateTime StartTime,
    DateTime EndTime,
    long TotalEvents,
    long DataSizeBytes,
    string StoragePath,
    bool Success,
    string? Error = null
);

/// <summary>
/// Configuration for replay
/// </summary>
public record ReplayConfiguration(
    string OriginalSessionId,
    string ReplaySessionId,
    ReplayMode Mode,
    double? SpeedMultiplier = 1.0,
    bool EnableComparison = true,
    bool StopOnDiscrepancy = false,
    Dictionary<string, object>? Overrides = null
);

/// <summary>
/// Replay execution modes
/// </summary>
public enum ReplayMode
{
    ExactTiming,
    FastForward,
    StepByStep,
    Interactive
}

/// <summary>
/// Result of replay execution
/// </summary>
public record ReplayResult(
    string ReplaySessionId,
    string OriginalSessionId,
    DateTime StartTime,
    DateTime EndTime,
    long EventsReplayed,
    bool Success,
    List<ReplayEvent> Events,
    List<ReplayDiscrepancy> Discrepancies,
    string? Error = null
);

/// <summary>
/// Individual event during replay
/// </summary>
public record ReplayEvent(
    DateTime Timestamp,
    string EventType,
    Dictionary<string, object> Data,
    bool MatchesOriginal,
    string? Discrepancy = null
);

/// <summary>
/// Discrepancy found during replay
/// </summary>
public record ReplayDiscrepancy(
    DateTime Timestamp,
    string EventType,
    object OriginalValue,
    object ReplayValue,
    double Difference,
    DiscrepancySeverity Severity
);

/// <summary>
/// Severity of replay discrepancies
/// </summary>
public enum DiscrepancySeverity
{
    Minor,
    Moderate,
    Significant,
    Critical
}

/// <summary>
/// Comparison result between original and replay
/// </summary>
public record ReplayComparisonResult(
    string OriginalSessionId,
    string ReplaySessionId,
    DateTime ComparedAt,
    bool IsExactMatch,
    double SimilarityScore,
    List<ReplayDiscrepancy> Discrepancies,
    ComparisonMetrics Metrics,
    string Summary
);

/// <summary>
/// Metrics from replay comparison
/// </summary>
public record ComparisonMetrics(
    int TotalEventsCompared,
    int ExactMatches,
    int MinorDiscrepancies,
    int SignificantDiscrepancies,
    double AverageTimingAccuracy,
    double DataAccuracy
);

/// <summary>
/// Recorded session metadata
/// </summary>
public record RecordedSession(
    string SessionId,
    DateTime RecordedAt,
    TimeSpan Duration,
    RecordingScope Scope,
    long EventCount,
    long DataSizeBytes,
    List<string> DataTypes,
    bool IsValid,
    string? Description = null
);

/// <summary>
/// Filter for session queries
/// </summary>
public record SessionFilter(
    DateTime? StartDate = null,
    DateTime? EndDate = null,
    RecordingScope? Scope = null,
    TimeSpan? MinDuration = null,
    TimeSpan? MaxDuration = null,
    List<string>? DataTypes = null
);

/// <summary>
/// Result of session validation
/// </summary>
public record SessionValidationResult(
    string SessionId,
    bool IsValid,
    bool IsComplete,
    List<string> Issues,
    List<string> Warnings,
    SessionIntegrityMetrics IntegrityMetrics
);

/// <summary>
/// Integrity metrics for session validation
/// </summary>
public record SessionIntegrityMetrics(
    bool HasAllExpectedEvents,
    bool TimestampsConsistent,
    bool DataIntegrityValid,
    double CompletenessScore,
    int MissingEvents,
    int CorruptedEvents
);