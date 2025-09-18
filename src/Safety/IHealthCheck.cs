using System;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Infra;

/// <summary>
/// Interface for health checks that can be automatically discovered and registered
/// </summary>
public interface IHealthCheck
{
    /// <summary>
    /// Unique name for this health check
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Human-readable description of what this health check validates
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Category for grouping related health checks
    /// </summary>
    string Category { get; }

    /// <summary>
    /// How often this check should run (in seconds)
    /// </summary>
    int IntervalSeconds { get; }

    /// <summary>
    /// Execute the health check
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Health check result</returns>
    Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Result of a health check execution
/// </summary>
public class HealthCheckResult
{
    public HealthStatus Status { get; set; }
    public string Message { get; set; } = string.Empty;
    public DateTime CheckedAt { get; set; } = DateTime.UtcNow;
    public TimeSpan Duration { get; set; }
    public object? Data { get; set; }

    public static HealthCheckResult Healthy(string message = "OK", object? data = null)
        => new() { Status = HealthStatus.Healthy, Message = message, Data = data };

    public static HealthCheckResult Warning(string message, object? data = null)
        => new() { Status = HealthStatus.Warning, Message = message, Data = data };

    public static HealthCheckResult Failed(string message, object? data = null)
        => new() { Status = HealthStatus.Failed, Message = message, Data = data };
}

/// <summary>
/// Health check status
/// </summary>
public enum HealthStatus
{
    Healthy,
    Warning,
    Failed
}

/// <summary>
/// Attribute to mark health checks for automatic discovery
/// </summary>
[AttributeUsage(AttributeTargets.Class)]
public class HealthCheckAttribute : Attribute
{
    public string? Category { get; set; }
    public int Priority { get; set; }
    public bool Enabled { get; set; } = true;
}
