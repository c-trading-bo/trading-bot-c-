using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for monitoring model drift and performance degradation
/// </summary>
internal interface IDriftMonitor
{
    /// <summary>
    /// Check for model drift in performance
    /// </summary>
    Task<DriftDetectionResult> CheckPerformanceDriftAsync(string algorithm, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check for data drift in input features
    /// </summary>
    Task<DriftDetectionResult> CheckDataDriftAsync(string algorithm, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get drift monitoring status
    /// </summary>
    Task<DriftMonitoringStatus> GetMonitoringStatusAsync(string algorithm, CancellationToken cancellationToken = default);
}

/// <summary>
/// Interface for health gates and risk monitoring
/// </summary>
internal interface IHealthGate
{
    /// <summary>
    /// Check if algorithm is healthy and safe to trade
    /// </summary>
    Task<HealthGateResult> CheckHealthAsync(string algorithm, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current health status for all algorithms
    /// </summary>
    Task<Dictionary<string, HealthGateResult>> GetAllHealthStatusAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register a health check failure
    /// </summary>
    Task RegisterHealthFailureAsync(string algorithm, string reason, CancellationToken cancellationToken = default);
}

/// <summary>
/// Drift detection result
/// </summary>
internal class DriftDetectionResult
{
    public string Algorithm { get; set; } = string.Empty;
    public string DriftType { get; set; } = string.Empty; // PERFORMANCE, DATA, CONCEPT
    public bool DriftDetected { get; set; }
    public decimal DriftScore { get; set; } // 0.0 = no drift, 1.0 = significant drift
    public decimal Threshold { get; set; }
    public DateTime DetectionTime { get; set; } = DateTime.UtcNow;
    public string Description { get; set; } = string.Empty;
    public Dictionary<string, object> Metrics { get; } = new();
    public List<string> RecommendedActions { get; } = new();
}

/// <summary>
/// Drift monitoring status
/// </summary>
internal class DriftMonitoringStatus
{
    public string Algorithm { get; set; } = string.Empty;
    public bool IsMonitoring { get; set; }
    public DateTime LastCheck { get; set; }
    public int ChecksPerformed { get; set; }
    public int DriftEventsDetected { get; set; }
    public List<DriftDetectionResult> RecentDriftEvents { get; } = new();
}

/// <summary>
/// Health gate result
/// </summary>
internal class HealthGateResult
{
    public string Algorithm { get; set; } = string.Empty;
    public bool IsHealthy { get; set; }
    public decimal HealthScore { get; set; } // 0.0 = unhealthy, 1.0 = perfectly healthy
    public DateTime LastCheck { get; set; } = DateTime.UtcNow;
    public List<string> HealthIssues { get; } = new();
    public List<string> Warnings { get; } = new();
    public Dictionary<string, object> HealthMetrics { get; } = new();
    public bool SafeToTrade { get; set; }
}