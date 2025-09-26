using System;
using System.Collections.Generic;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Training intensity levels for adaptive learning
/// </summary>
internal enum TrainingIntensity
{
    /// <summary>
    /// Light learning during market hours (60 min intervals)
    /// </summary>
    Light,
    
    /// <summary>
    /// Intensive learning during market closed (15 min intervals)
    /// </summary>
    Intensive
}

/// <summary>
/// Telemetry data for cloud integration
/// </summary>
internal class TelemetryData
{
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Source { get; set; } = string.Empty;
    public Dictionary<string, object> Metrics { get; } = new();
    public string SessionId { get; set; } = string.Empty;
}

/// <summary>
/// Trade data model
/// </summary>
internal class TradeData
{
    public string TradeId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Status { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; } = new();
}

/// <summary>
/// System status information
/// </summary>
internal class SystemStatus
{
    public bool IsHealthy { get; set; }
    public Dictionary<string, string> ComponentStatuses { get; } = new();
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
    public string OverallStatus { get; set; } = "Unknown";
}

/// <summary>
/// Cloud metrics information
/// </summary>
internal class CloudMetrics
{
    public DateTime LastSync { get; set; }
    public string Status { get; set; } = string.Empty;
    public TimeSpan Latency { get; set; }
    public Dictionary<string, object> AdditionalMetrics { get; } = new();
}

/// <summary>
/// Workflow definition model
/// </summary>
internal class WorkflowDefinition
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public Dictionary<string, object> Parameters { get; } = new();
    public bool Enabled { get; set; } = true;
}

/// <summary>
/// Workflow status model
/// </summary>
internal class WorkflowStatus
{
    public string WorkflowId { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty;
    public DateTime LastRun { get; set; }
    public DateTime NextRun { get; set; }
    public Dictionary<string, object> Metadata { get; } = new();
}

/// <summary>
/// Workflow execution result
/// </summary>
internal class WorkflowResult
{
    public string WorkflowId { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string? Error { get; set; }
    public DateTime CompletedAt { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object>? Results { get; set; }
}

/// <summary>
/// Position status information
/// </summary>
internal class PositionStatus
{
    public string Symbol { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal AveragePrice { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public bool IsOpen { get; set; }
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}