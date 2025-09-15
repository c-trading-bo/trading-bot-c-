using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Legal;

/// <summary>
/// Interface for trading halt awareness and market status integration
/// </summary>
public interface ITradingHaltManager
{
    /// <summary>
    /// Initialize trading halt monitoring with market data feeds
    /// </summary>
    Task InitializeAsync(TradingHaltConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if trading is currently allowed for specific symbol
    /// </summary>
    Task<bool> IsTradingAllowedAsync(string symbol, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current market status for all monitored symbols
    /// </summary>
    Task<MarketStatusReport> GetMarketStatusAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register callback for trading halt events
    /// </summary>
    void RegisterHaltCallback(Func<TradingHaltEvent, Task> callback);
    
    /// <summary>
    /// Force trading suspension for specific symbol or all symbols
    /// </summary>
    Task SuspendTradingAsync(string? symbol, string reason, TimeSpan? duration = null, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Resume trading after suspension
    /// </summary>
    Task ResumeTradingAsync(string? symbol, string reason, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get trading halt history and statistics
    /// </summary>
    Task<TradingHaltHistoryReport> GetHaltHistoryAsync(TimeSpan lookbackPeriod, CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for trading halt monitoring
/// </summary>
public record TradingHaltConfiguration(
    List<string> MonitoredSymbols,
    List<MarketDataSource> DataSources,
    TimeSpan StatusCheckInterval,
    bool EnableAutomaticSuspension = true,
    bool EnablePreMarketMonitoring = true,
    bool EnableAfterHoursMonitoring = false,
    Dictionary<string, object>? CustomHaltRules = null
);

/// <summary>
/// Market data source for halt information
/// </summary>
public record MarketDataSource(
    string SourceName,
    string EndpointUrl,
    MarketDataSourceType Type,
    TimeSpan Timeout,
    Dictionary<string, string>? Headers = null
);

/// <summary>
/// Types of market data sources
/// </summary>
public enum MarketDataSourceType
{
    REST_API,
    WebSocket,
    RSS_Feed,
    FTP,
    Email
}

/// <summary>
/// Trading halt event
/// </summary>
public record TradingHaltEvent(
    string Symbol,
    TradingHaltType HaltType,
    DateTime HaltTime,
    DateTime? ResumeTime,
    string Reason,
    string Source,
    TradingHaltSeverity Severity
);

/// <summary>
/// Types of trading halts
/// </summary>
public enum TradingHaltType
{
    RegulatoryHalt,
    CircuitBreaker,
    VolatilityHalt,
    NewsHalt,
    SystemHalt,
    ManualSuspension,
    MarketClosure
}

/// <summary>
/// Severity of trading halt
/// </summary>
public enum TradingHaltSeverity
{
    Info,
    Warning,
    Critical,
    Emergency
}

/// <summary>
/// Market status report
/// </summary>
public record MarketStatusReport(
    DateTime GeneratedAt,
    MarketSession CurrentSession,
    Dictionary<string, SymbolTradingStatus> SymbolStatuses,
    List<ActiveTradingHalt> ActiveHalts,
    List<string> SystemMessages
);

/// <summary>
/// Market session information
/// </summary>
public enum MarketSession
{
    PreMarket,
    Regular,
    PostMarket,
    Closed,
    Holiday
}

/// <summary>
/// Trading status for individual symbol
/// </summary>
public record SymbolTradingStatus(
    string Symbol,
    TradingStatus Status,
    DateTime LastUpdated,
    string? HaltReason = null,
    DateTime? HaltStartTime = null,
    DateTime? ExpectedResumeTime = null
);

/// <summary>
/// Trading status enumeration
/// </summary>
public enum TradingStatus
{
    Open,
    Halted,
    Suspended,
    Paused,
    Closed,
    Unknown
}

/// <summary>
/// Active trading halt information
/// </summary>
public record ActiveTradingHalt(
    string Symbol,
    TradingHaltType Type,
    DateTime StartTime,
    TimeSpan Duration,
    string Reason,
    bool IsResolutionExpected,
    DateTime? ExpectedResolution = null
);

/// <summary>
/// Trading halt history report
/// </summary>
public record TradingHaltHistoryReport(
    DateTime GeneratedAt,
    TimeSpan ReportPeriod,
    List<HistoricalTradingHalt> Halts,
    TradingHaltStatistics Statistics,
    List<string> Patterns
);

/// <summary>
/// Historical trading halt record
/// </summary>
public record HistoricalTradingHalt(
    string Symbol,
    TradingHaltType Type,
    DateTime StartTime,
    DateTime? EndTime,
    TimeSpan Duration,
    string Reason,
    decimal? ImpactOnPrice = null
);

/// <summary>
/// Statistics for trading halts
/// </summary>
public record TradingHaltStatistics(
    int TotalHalts,
    TimeSpan TotalHaltDuration,
    TimeSpan AverageHaltDuration,
    Dictionary<TradingHaltType, int> HaltsByType,
    Dictionary<string, int> HaltsBySymbol,
    int FalseAlarms
);

/// <summary>
/// Interface for deployment freeze window management
/// </summary>
public interface IDeploymentFreezeManager
{
    /// <summary>
    /// Initialize freeze window monitoring
    /// </summary>
    Task InitializeAsync(FreezeWindowConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if deployments are currently allowed
    /// </summary>
    Task<bool> IsDeploymentAllowedAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current freeze window status
    /// </summary>
    Task<FreezeWindowStatus> GetFreezeStatusAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register a high-impact event that triggers freeze window
    /// </summary>
    Task RegisterHighImpactEventAsync(HighImpactEvent impactEvent, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Force freeze window activation
    /// </summary>
    Task ActivateFreezeWindowAsync(string reason, TimeSpan duration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Request emergency deployment during freeze window
    /// </summary>
    Task<EmergencyDeploymentResult> RequestEmergencyDeploymentAsync(EmergencyDeploymentRequest request, CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for freeze window management
/// </summary>
public record FreezeWindowConfiguration(
    List<FreezeWindowRule> FreezeRules,
    TimeSpan DefaultFreezeDuration,
    bool EnableAutomaticFreezing = true,
    List<string> EmergencyApprovers = default!,
    Dictionary<string, TimeSpan>? EventSpecificDurations = null
);

/// <summary>
/// Rule for freeze window activation
/// </summary>
public record FreezeWindowRule(
    string RuleName,
    EventType TriggerEvent,
    TimeSpan FreezeDuration,
    TimeSpan PreEventBuffer,
    TimeSpan PostEventBuffer,
    int Priority
);

/// <summary>
/// High-impact event types
/// </summary>
public enum EventType
{
    FOMC_Meeting,
    EarningsAnnouncement,
    MarketOpen,
    MarketClose,
    VolatilitySpike,
    SystemMaintenance,
    RegulatoryAnnouncement,
    GeopoliticalEvent,
    Custom
}

/// <summary>
/// High-impact event definition
/// </summary>
public record HighImpactEvent(
    string EventId,
    EventType Type,
    DateTime ScheduledTime,
    TimeSpan EstimatedDuration,
    ImpactLevel Impact,
    string Description,
    List<string> AffectedSymbols
);

/// <summary>
/// Impact level of events
/// </summary>
public enum ImpactLevel
{
    Low,
    Medium,
    High,
    Critical
}

/// <summary>
/// Current freeze window status
/// </summary>
public record FreezeWindowStatus(
    bool IsActive,
    DateTime? FreezeStartTime,
    DateTime? FreezeEndTime,
    string? Reason,
    List<ActiveFreezeWindow> ActiveFreezeWindows,
    List<UpcomingFreezeWindow> UpcomingFreezeWindows
);

/// <summary>
/// Active freeze window
/// </summary>
public record ActiveFreezeWindow(
    string WindowId,
    string Reason,
    DateTime StartTime,
    DateTime EndTime,
    EventType? TriggerEvent = null
);

/// <summary>
/// Upcoming freeze window
/// </summary>
public record UpcomingFreezeWindow(
    string WindowId,
    DateTime ScheduledStart,
    DateTime ScheduledEnd,
    string Reason,
    EventType TriggerEvent
);

/// <summary>
/// Emergency deployment request
/// </summary>
public record EmergencyDeploymentRequest(
    string RequestId,
    string Requestor,
    string DeploymentDescription,
    EmergencyJustification Justification,
    List<string> RequiredApprovers,
    TimeSpan RequestedDuration
);

/// <summary>
/// Justification for emergency deployment
/// </summary>
public record EmergencyJustification(
    EmergencyType Type,
    string BusinessImpact,
    string TechnicalRationale,
    string RiskMitigation,
    List<string> SupportingEvidence
);

/// <summary>
/// Types of emergencies
/// </summary>
public enum EmergencyType
{
    SystemOutage,
    SecurityVulnerability,
    DataCorruption,
    RegulatoryRequirement,
    BusinessCritical
}

/// <summary>
/// Result of emergency deployment request
/// </summary>
public record EmergencyDeploymentResult(
    string RequestId,
    bool Approved,
    List<string> Approvers,
    DateTime? ApprovedUntil,
    string? DenialReason = null,
    List<string> Conditions = default!
);