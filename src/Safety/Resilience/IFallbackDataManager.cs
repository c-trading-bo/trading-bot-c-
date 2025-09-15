using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Resilience;

/// <summary>
/// Interface for fallback data sources and cached snapshots
/// </summary>
public interface IFallbackDataManager
{
    /// <summary>
    /// Register a fallback data source
    /// </summary>
    Task RegisterFallbackSourceAsync(FallbackDataSource source, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get data with automatic fallback if primary source fails
    /// </summary>
    Task<DataResult<T>> GetDataWithFallbackAsync<T>(DataRequest request, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Update cached snapshot for fallback use
    /// </summary>
    Task UpdateCachedSnapshotAsync<T>(string key, T data, TimeSpan cacheExpiry, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get cached snapshot if available
    /// </summary>
    Task<CachedData<T>?> GetCachedSnapshotAsync<T>(string key, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check health of all data sources
    /// </summary>
    Task<DataSourceHealthReport> CheckDataSourceHealthAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Force failover to specific data source
    /// </summary>
    Task<bool> ForceFailoverAsync(string targetSourceId, string reason, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get data source statistics and performance metrics
    /// </summary>
    Task<Dictionary<string, DataSourceMetrics>> GetDataSourceMetricsAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for a fallback data source
/// </summary>
public record FallbackDataSource(
    string SourceId,
    string Name,
    int Priority,
    DataSourceType Type,
    Dictionary<string, object> Configuration,
    TimeSpan Timeout,
    int MaxRetries = 3,
    bool IsHealthCheckEnabled = true,
    TimeSpan HealthCheckInterval = default
)
{
    public TimeSpan HealthCheckInterval { get; init; } = HealthCheckInterval == default ? TimeSpan.FromMinutes(1) : HealthCheckInterval;
}

/// <summary>
/// Types of data sources
/// </summary>
public enum DataSourceType
{
    Primary,
    Secondary,
    Cache,
    StaticSnapshot,
    ExternalAPI
}

/// <summary>
/// Request for data with fallback options
/// </summary>
public record DataRequest(
    string DataKey,
    Dictionary<string, object> Parameters,
    TimeSpan MaxAge,
    bool AllowCachedData = true,
    List<string>? PreferredSources = null
);

/// <summary>
/// Result of data request with source information
/// </summary>
public record DataResult<T>(
    T? Data,
    bool Success,
    string SourceId,
    DateTime RetrievedAt,
    TimeSpan Latency,
    string? Error = null,
    bool WasFromCache = false
);

/// <summary>
/// Cached data with metadata
/// </summary>
public record CachedData<T>(
    T Data,
    DateTime CachedAt,
    TimeSpan ExpiresIn,
    string SourceId,
    bool IsExpired
);

/// <summary>
/// Health report for all data sources
/// </summary>
public record DataSourceHealthReport(
    DateTime GeneratedAt,
    bool OverallHealthy,
    Dictionary<string, DataSourceHealth> SourceHealth,
    string? ActivePrimarySource,
    List<string> FailedSources,
    List<string> DegradedSources
);

/// <summary>
/// Health status of individual data source
/// </summary>
public record DataSourceHealth(
    string SourceId,
    DataSourceHealthStatus Status,
    DateTime LastChecked,
    TimeSpan AverageLatency,
    double SuccessRate,
    string? LastError = null,
    DateTime? LastSuccessfulRequest = null
);

/// <summary>
/// Health status enumeration
/// </summary>
public enum DataSourceHealthStatus
{
    Healthy,
    Degraded,
    Unhealthy,
    Unknown
}

/// <summary>
/// Performance metrics for data sources
/// </summary>
public record DataSourceMetrics(
    string SourceId,
    int TotalRequests,
    int SuccessfulRequests,
    int FailedRequests,
    TimeSpan AverageLatency,
    TimeSpan MaxLatency,
    TimeSpan MinLatency,
    double SuccessRate,
    DateTime LastRequest,
    long DataBytesTransferred
);