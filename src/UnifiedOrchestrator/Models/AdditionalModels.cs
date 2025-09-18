using System;
using System.Collections.Generic;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Telemetry data for cloud integration
/// </summary>
public class TelemetryData
{
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Source { get; set; } = string.Empty;
    public Dictionary<string, object> Metrics { get; } = new();
    public string SessionId { get; set; } = string.Empty;
}

/// <summary>
/// Trade data model
/// </summary>
public class TradeData
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
public class SystemStatus
{
    public bool IsHealthy { get; set; }
    public Dictionary<string, string> ComponentStatuses { get; } = new();
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
    public string OverallStatus { get; set; } = "Unknown";
}

/// <summary>
/// Cloud metrics information
/// </summary>
public class CloudMetrics
{
    public DateTime LastSync { get; set; }
    public string Status { get; set; } = string.Empty;
    public TimeSpan Latency { get; set; }
    public Dictionary<string, object> AdditionalMetrics { get; } = new();
}

/// <summary>
/// Workflow definition model
/// </summary>
public class WorkflowDefinition
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
public class WorkflowStatus
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
public class WorkflowResult
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
public class PositionStatus
{
    public string Symbol { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal AveragePrice { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public bool IsOpen { get; set; }
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}