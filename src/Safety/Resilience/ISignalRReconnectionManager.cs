using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Resilience;

/// <summary>
/// Interface for robust SignalR reconnection with jitter and backoff
/// </summary>
public interface ISignalRReconnectionManager
{
    /// <summary>
    /// Start monitoring SignalR connection with automatic reconnection
    /// </summary>
    Task StartMonitoringAsync(SignalRReconnectionConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Stop monitoring and cleanup resources
    /// </summary>
    Task StopMonitoringAsync();
    
    /// <summary>
    /// Get current connection status and statistics
    /// </summary>
    Task<SignalRConnectionStatus> GetConnectionStatusAsync();
    
    /// <summary>
    /// Force a reconnection attempt
    /// </summary>
    Task ForceReconnectionAsync(string reason);
    
    /// <summary>
    /// Register a callback for connection state changes
    /// </summary>
    void RegisterConnectionStateCallback(Func<SignalRConnectionState, Task> callback);
    
    /// <summary>
    /// Get reconnection statistics and health metrics
    /// </summary>
    Task<ReconnectionStatistics> GetReconnectionStatisticsAsync();
}

/// <summary>
/// Configuration for SignalR reconnection behavior
/// </summary>
public record SignalRReconnectionConfiguration(
    TimeSpan InitialRetryDelay = default,
    TimeSpan MaxRetryDelay = default,
    double BackoffMultiplier = 2.0,
    double JitterFactor = 0.1,
    int MaxRetryAttempts = 20,
    TimeSpan ConnectionTimeout = default,
    TimeSpan HeartbeatInterval = default,
    bool EnableCircuitBreaker = true,
    int CircuitBreakerFailureThreshold = 5,
    TimeSpan CircuitBreakerResetTimeout = default
)
{
    public TimeSpan InitialRetryDelay { get; init; } = InitialRetryDelay == default ? TimeSpan.FromSeconds(1) : InitialRetryDelay;
    public TimeSpan MaxRetryDelay { get; init; } = MaxRetryDelay == default ? TimeSpan.FromMinutes(5) : MaxRetryDelay;
    public TimeSpan ConnectionTimeout { get; init; } = ConnectionTimeout == default ? TimeSpan.FromSeconds(30) : ConnectionTimeout;
    public TimeSpan HeartbeatInterval { get; init; } = HeartbeatInterval == default ? TimeSpan.FromSeconds(15) : HeartbeatInterval;
    public TimeSpan CircuitBreakerResetTimeout { get; init; } = CircuitBreakerResetTimeout == default ? TimeSpan.FromMinutes(2) : CircuitBreakerResetTimeout;
}

/// <summary>
/// Current SignalR connection status
/// </summary>
public record SignalRConnectionStatus(
    SignalRConnectionState State,
    DateTime LastConnected,
    DateTime? LastDisconnected,
    TimeSpan? ConnectionDuration,
    int ReconnectionAttempts,
    string? LastError,
    bool IsCircuitBreakerOpen,
    double ConnectionQuality
);

/// <summary>
/// SignalR connection states
/// </summary>
public enum SignalRConnectionState
{
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed,
    CircuitBreakerOpen
}

/// <summary>
/// Reconnection statistics for monitoring
/// </summary>
public record ReconnectionStatistics(
    int TotalReconnectionAttempts,
    int SuccessfulReconnections,
    int FailedReconnections,
    TimeSpan TotalDowntime,
    TimeSpan LongestDowntime,
    TimeSpan AverageReconnectionTime,
    double ConnectionUptime,
    List<ReconnectionEvent> RecentEvents
);

/// <summary>
/// Individual reconnection event
/// </summary>
public record ReconnectionEvent(
    DateTime Timestamp,
    SignalRConnectionState FromState,
    SignalRConnectionState ToState,
    TimeSpan Duration,
    string? Reason = null,
    bool WasSuccessful = true
);