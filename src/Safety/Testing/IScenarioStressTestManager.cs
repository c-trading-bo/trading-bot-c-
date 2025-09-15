using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Testing;

/// <summary>
/// Interface for scenario stress testing with volatility spikes and outlier events
/// </summary>
public interface IScenarioStressTestManager
{
    /// <summary>
    /// Execute a comprehensive stress test scenario
    /// </summary>
    Task<StressTestResult> ExecuteStressTestAsync(StressTestConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register a custom stress test scenario
    /// </summary>
    Task RegisterScenarioAsync(StressTestScenario scenario, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get all available stress test scenarios
    /// </summary>
    Task<List<StressTestScenario>> GetAvailableScenariosAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Run stress test on historical data with backfill simulation
    /// </summary>
    Task<HistoricalStressTestResult> RunHistoricalStressTestAsync(HistoricalStressTestConfig config, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Monitor live system during stress conditions
    /// </summary>
    Task<LiveStressMonitoringResult> MonitorLiveStressAsync(LiveStressMonitoringConfig config, CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for stress testing
/// </summary>
public record StressTestConfiguration(
    string TestId,
    List<string> ScenarioIds,
    TimeSpan TestDuration,
    StressTestIntensity Intensity,
    bool EnableRealTimeMonitoring = true,
    Dictionary<string, object>? CustomParameters = null
);

/// <summary>
/// Stress test intensity levels
/// </summary>
public enum StressTestIntensity
{
    Low,
    Medium,
    High,
    Extreme
}

/// <summary>
/// Definition of a stress test scenario
/// </summary>
public record StressTestScenario(
    string ScenarioId,
    string Name,
    string Description,
    StressTestType Type,
    Dictionary<string, object> Parameters,
    TimeSpan Duration,
    List<string> ExpectedImpacts
);

/// <summary>
/// Types of stress tests
/// </summary>
public enum StressTestType
{
    VolatilitySpike,
    AbnormalGaps,
    LiquidityDrought,
    HighFrequencyErrors,
    NetworkLatency,
    BackfillStress,
    MarketCrash,
    FlashCrash,
    OutlierEvents
}

/// <summary>
/// Result of stress test execution
/// </summary>
public record StressTestResult(
    string TestId,
    DateTime StartTime,
    DateTime EndTime,
    bool Success,
    List<ScenarioResult> ScenarioResults,
    SystemPerformanceMetrics Performance,
    List<StressTestFinding> Findings,
    string? Summary = null
);

/// <summary>
/// Result of individual scenario
/// </summary>
public record ScenarioResult(
    string ScenarioId,
    bool Passed,
    TimeSpan Duration,
    Dictionary<string, object> Metrics,
    List<string> Issues,
    StressTestSeverity WorstIssue
);

/// <summary>
/// Severity of stress test issues
/// </summary>
public enum StressTestSeverity
{
    Info,
    Warning,
    Error,
    Critical
}

/// <summary>
/// System performance during stress test
/// </summary>
public record SystemPerformanceMetrics(
    TimeSpan AverageResponseTime,
    TimeSpan MaxResponseTime,
    double ThroughputPerSecond,
    double ErrorRate,
    long MemoryPeakUsage,
    double CpuPeakUsage,
    int ActiveThreadsPeak
);

/// <summary>
/// Key finding from stress test
/// </summary>
public record StressTestFinding(
    StressTestSeverity Severity,
    string Category,
    string Description,
    string Recommendation,
    List<string> AffectedComponents
);

/// <summary>
/// Configuration for historical stress testing
/// </summary>
public record HistoricalStressTestConfig(
    DateTime StartDate,
    DateTime EndDate,
    List<string> Symbols,
    List<StressTestType> StressTypes,
    bool IncludeBackfillStress = true
);

/// <summary>
/// Result of historical stress test
/// </summary>
public record HistoricalStressTestResult(
    DateTime StartDate,
    DateTime EndDate,
    int TotalDataPoints,
    int StressEvents,
    List<HistoricalStressEvent> Events,
    Dictionary<string, object> PerformanceMetrics
);

/// <summary>
/// Historical stress event details
/// </summary>
public record HistoricalStressEvent(
    DateTime Timestamp,
    StressTestType Type,
    string Description,
    double Magnitude,
    TimeSpan SystemResponseTime,
    List<string> SystemReactions
);

/// <summary>
/// Configuration for live stress monitoring
/// </summary>
public record LiveStressMonitoringConfig(
    TimeSpan MonitoringDuration,
    Dictionary<StressTestType, double> ThresholdLevels,
    bool EnableAutomaticRecovery = true
);

/// <summary>
/// Result of live stress monitoring
/// </summary>
public record LiveStressMonitoringResult(
    DateTime StartTime,
    DateTime EndTime,
    List<DetectedStressEvent> DetectedEvents,
    bool SystemStabilityMaintained,
    List<string> RecoveryActions
);

/// <summary>
/// Detected stress event during live monitoring
/// </summary>
public record DetectedStressEvent(
    DateTime DetectedAt,
    StressTestType Type,
    double Magnitude,
    bool WasHandledCorrectly,
    TimeSpan RecoveryTime
);