using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace Trading.Safety;

/// <summary>
/// Monitors system health including hub connections, error rates, and latency
/// Suspends trading operations when health degrades below acceptable thresholds
/// </summary>
public interface IHealthMonitor
{
    event Action<HealthStatus> OnHealthChanged;
    Task StartMonitoringAsync(CancellationToken cancellationToken = default);
    void RecordHubConnection(string hubName, bool isConnected);
    void RecordApiCall(string operation, TimeSpan duration, bool success);
    void RecordError(string source, Exception error);
    HealthStatus GetCurrentHealth();
    bool IsTradingAllowed { get; }
}

public record HealthStatus(
    bool IsHealthy,
    bool TradingAllowed,
    int ConnectedHubs,
    int TotalHubs,
    double ErrorRate,
    double AverageLatencyMs,
    string StatusMessage
);

public class HealthMonitor : TradingBot.Abstractions.IHealthMonitor
{
    private readonly ILogger<HealthMonitor> _logger;
    private readonly AppOptions _config;
    private readonly ConcurrentDictionary<string, bool> _hubConnections = new();
    private readonly ConcurrentQueue<ApiCallMetric> _apiCalls = new();
    private readonly ConcurrentQueue<ErrorMetric> _errors = new();
    
    private bool _isHealthy = true;
    private bool _tradingAllowed = true;
    private DateTime _lastHealthCheck = DateTime.UtcNow;
    private int _reconnectAttempts = 0;

    public event Action<TradingBot.Abstractions.HealthStatus>? OnHealthChanged;
    public event Action<TradingBot.Abstractions.HealthStatus>? HealthStatusChanged;
    public bool IsTradingAllowed => _tradingAllowed;

    // Health thresholds
    private const double MaxErrorRate = 0.1; // 10% error rate threshold
    private const double MaxLatencyMs = 5000; // 5 second latency threshold
    private const int RequiredHubConnections = 2; // User and Market hubs
    private const int MaxReconnectAttempts = 5;
    private const int HealthCheckIntervalMs = 5000; // Check every 5 seconds

    public HealthMonitor(ILogger<HealthMonitor> logger, IOptions<AppOptions> config)
    {
        _logger = logger;
        _config = config.Value;
    }

    public async Task<TradingBot.Abstractions.HealthStatus> GetHealthStatusAsync(string componentName)
    {
        var currentHealth = GetCurrentHealth();
        return await Task.FromResult(new TradingBot.Abstractions.HealthStatus
        {
            ComponentName = componentName,
            IsHealthy = currentHealth.IsHealthy,
            Status = currentHealth.IsHealthy ? "HEALTHY" : "UNHEALTHY",
            TradingAllowed = currentHealth.TradingAllowed,
            ConnectedHubs = currentHealth.ConnectedHubs,
            TotalHubs = currentHealth.TotalHubs,
            ErrorRate = currentHealth.ErrorRate,
            AverageLatencyMs = currentHealth.AverageLatencyMs,
            StatusMessage = currentHealth.StatusMessage
        });
    }

    public async Task StartMonitoringAsync()
    {
        await StartMonitoringAsync(CancellationToken.None);
    }

    public async Task StartMonitoringAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[HEALTH] Starting health monitoring");

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                await PerformHealthCheckAsync();
                await Task.Delay(HealthCheckIntervalMs, cancellationToken);
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("[HEALTH] Health monitoring cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH] Error in health monitoring");
            throw;
        }
    }

    public void RecordHubConnection(string hubName, bool isConnected)
    {
        _hubConnections.AddOrUpdate(hubName, isConnected, (key, oldValue) => isConnected);
        
        _logger.LogDebug("[HEALTH] Hub connection updated: {Hub} = {Connected}", hubName, isConnected);
        
        // Trigger immediate health check if connection lost
        if (!isConnected)
        {
            _ = Task.Run(PerformHealthCheckAsync);
        }
    }

    public void RecordApiCall(string operation, TimeSpan duration, bool success)
    {
        var metric = new ApiCallMetric(operation, duration.TotalMilliseconds, success, DateTime.UtcNow);
        _apiCalls.Enqueue(metric);
        
        // Keep only last 100 calls for rolling metrics
        while (_apiCalls.Count > 100)
        {
            _apiCalls.TryDequeue(out _);
        }

        _logger.LogDebug("[HEALTH] API call recorded: {Operation} - {Duration}ms - {Success}", 
            operation, duration.TotalMilliseconds, success);
    }

    public void RecordError(string source, Exception error)
    {
        var metric = new ErrorMetric(source, error.GetType().Name, error.Message, DateTime.UtcNow);
        _errors.Enqueue(metric);
        
        // Keep only last 50 errors for analysis
        while (_errors.Count > 50)
        {
            _errors.TryDequeue(out _);
        }

        _logger.LogWarning("[HEALTH] Error recorded from {Source}: {ErrorType}", source, error.GetType().Name);
        
        // Trigger immediate health check on error
        _ = Task.Run(PerformHealthCheckAsync);
    }

    private async Task PerformHealthCheckAsync()
    {
        try
        {
            var previouslyHealthy = _isHealthy;
            var previouslyAllowedTrading = _tradingAllowed;

            // Check hub connections
            var connectedHubs = 0;
            var totalHubs = _hubConnections.Count;
            foreach (var connection in _hubConnections.Values)
            {
                if (connection) connectedHubs++;
            }

            var hubsHealthy = connectedHubs >= RequiredHubConnections;

            // Calculate error rate from recent calls
            var recentCalls = GetRecentApiCalls();
            var errorRate = CalculateErrorRate(recentCalls);
            var avgLatency = CalculateAverageLatency(recentCalls);

            // Determine overall health
            var latencyHealthy = avgLatency < MaxLatencyMs;
            var errorRateHealthy = errorRate < MaxErrorRate;
            
            _isHealthy = hubsHealthy && latencyHealthy && errorRateHealthy;
            _tradingAllowed = _isHealthy;

            var status = new HealthStatus(
                _isHealthy,
                _tradingAllowed,
                connectedHubs,
                totalHubs,
                errorRate,
                avgLatency,
                BuildStatusMessage(hubsHealthy, latencyHealthy, errorRateHealthy)
            );

            // Log health changes
            if (previouslyHealthy != _isHealthy || previouslyAllowedTrading != _tradingAllowed)
            {
                var level = _isHealthy ? LogLevel.Information : LogLevel.Warning;
                _logger.Log(level, "[HEALTH] Health status changed: {Status}", status.StatusMessage);
                
                // Convert to Abstractions HealthStatus for event
                var abstractionsStatus = new TradingBot.Abstractions.HealthStatus
                {
                    ComponentName = "HealthMonitor",
                    IsHealthy = status.IsHealthy,
                    Status = status.IsHealthy ? "HEALTHY" : "UNHEALTHY",
                    TradingAllowed = status.TradingAllowed,
                    ConnectedHubs = status.ConnectedHubs,
                    TotalHubs = status.TotalHubs,
                    ErrorRate = status.ErrorRate,
                    AverageLatencyMs = status.AverageLatencyMs,
                    StatusMessage = status.StatusMessage
                };
                OnHealthChanged?.Invoke(abstractionsStatus);
                HealthStatusChanged?.Invoke(abstractionsStatus);
            }

            // Handle degraded health
            if (!_isHealthy)
            {
                await HandleDegradedHealthAsync(status);
            }
            else
            {
                _reconnectAttempts = 0; // Reset on recovery
            }

            _lastHealthCheck = DateTime.UtcNow;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH] Error during health check");
        }
    }

    private async Task HandleDegradedHealthAsync(HealthStatus status)
    {
        _logger.LogWarning("[HEALTH] 🟡 Degraded health detected - Trading suspended");
        
        _reconnectAttempts++;
        
        if (_reconnectAttempts <= MaxReconnectAttempts)
        {
            _logger.LogInformation("[HEALTH] Attempting reconnection {Attempt}/{Max}", 
                _reconnectAttempts, MaxReconnectAttempts);
            
            // Exponential backoff
            var delayMs = Math.Min(1000 * Math.Pow(2, _reconnectAttempts - 1), 30000);
            await Task.Delay(TimeSpan.FromMilliseconds(delayMs));
            
            // Trigger reconnection logic for SignalR hubs
            await TriggerHubReconnectionAsync();
        }
        else
        {
            _logger.LogCritical("[HEALTH] 🔴 Maximum reconnection attempts exceeded - Manual intervention required");
        }
    }

    private List<ApiCallMetric> GetRecentApiCalls()
    {
        var cutoff = DateTime.UtcNow.AddMinutes(-5); // Last 5 minutes
        return _apiCalls.Where(call => call.Timestamp > cutoff).ToList();
    }

    private double CalculateErrorRate(List<ApiCallMetric> calls)
    {
        if (calls.Count == 0) return 0;
        
        var errors = calls.Count(c => !c.Success);
        return (double)errors / calls.Count;
    }

    private double CalculateAverageLatency(List<ApiCallMetric> calls)
    {
        if (calls.Count == 0) return 0;
        
        return calls.Average(c => c.DurationMs);
    }

    private string BuildStatusMessage(bool hubsHealthy, bool latencyHealthy, bool errorRateHealthy)
    {
        var issues = new List<string>();
        
        if (!hubsHealthy) issues.Add("hub connections");
        if (!latencyHealthy) issues.Add("high latency");
        if (!errorRateHealthy) issues.Add("high error rate");

        return issues.Count == 0 
            ? "All systems healthy" 
            : $"Issues detected: {string.Join(", ", issues)}";
    }

    public HealthStatus GetCurrentHealth()
    {
        var recentCalls = GetRecentApiCalls();
        var connectedHubs = _hubConnections.Values.Count(c => c);
        
        return new HealthStatus(
            _isHealthy,
            _tradingAllowed,
            connectedHubs,
            _hubConnections.Count,
            CalculateErrorRate(recentCalls),
            CalculateAverageLatency(recentCalls),
            _isHealthy ? "Healthy" : "Degraded"
        );
    }

    private async Task TriggerHubReconnectionAsync()
    {
        try
        {
            _logger.LogInformation("[HEALTH] Triggering hub reconnection attempts");
            
            // Get list of disconnected hubs
            var disconnectedHubs = _hubConnections
                .Where(kv => !kv.Value)
                .Select(kv => kv.Key)
                .ToList();

            if (disconnectedHubs.Count == 0)
            {
                _logger.LogDebug("[HEALTH] No disconnected hubs found");
                return;
            }

            foreach (var hubName in disconnectedHubs)
            {
                _logger.LogInformation("[HEALTH] Attempting to reconnect hub: {HubName}", hubName);
                
                try
                {
                    // Signal to the hub management system to attempt reconnection
                    // This approach uses events rather than direct coupling
                    await NotifyHubReconnectionRequiredAsync(hubName);
                    
                    // Wait a moment before processing next hub
                    await Task.Delay(100);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[HEALTH] Failed to trigger reconnection for hub: {HubName}", hubName);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH] Error during hub reconnection trigger");
        }
    }

    private async Task NotifyHubReconnectionRequiredAsync(string hubName)
    {
        try
        {
            // Use a standardized approach to signal reconnection need
            // This follows the observer pattern used elsewhere in the system
            var reconnectionData = new
            {
                HubName = hubName,
                Timestamp = DateTime.UtcNow,
                AttemptNumber = _reconnectAttempts,
                Reason = "Health check detected disconnection"
            };

            // Log the reconnection attempt with structured data for monitoring
            _logger.LogInformation("[HEALTH] Hub reconnection triggered: {ReconnectionData}", 
                System.Text.Json.JsonSerializer.Serialize(reconnectionData));

            // In a production system, this would:
            // 1. Send a message to the hub management service
            // 2. Update reconnection metrics
            // 3. Trigger circuit breaker reset if appropriate
            // 4. Notify monitoring systems
            
            // For now, this provides the infrastructure for hub management
            // The actual SignalR connection management should subscribe to these events
            await Task.Delay(50); // Simulate async notification
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH] Failed to notify reconnection requirement for hub: {HubName}", hubName);
        }
    }
}

internal record ApiCallMetric(string Operation, double DurationMs, bool Success, DateTime Timestamp);
internal record ErrorMetric(string Source, string ErrorType, string Message, DateTime Timestamp);