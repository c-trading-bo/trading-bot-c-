using System;
using System.Collections.Generic;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Rollback drill configuration
/// </summary>
public class RollbackDrillConfig
{
    public LoadLevel LoadLevel { get; set; } = LoadLevel.Medium;
    public int TestDurationSeconds { get; set; } = 15;
    public int DecisionsPerSecond { get; set; } = 20;
    public bool EnableMetrics { get; set; } = true;
    public bool EnableContextPreservation { get; set; } = true;
}

/// <summary>
/// Load level enumeration
/// </summary>
public enum LoadLevel
{
    Low,
    Medium,
    High,
    Extreme
}

/// <summary>
/// Rollback drill result
/// </summary>
public class RollbackDrillResult
{
    public string DrillId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public double TotalDurationMs { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public RollbackDrillConfig Config { get; set; } = null!;
    public List<RollbackEvent> Events { get; set; } = new();
    public RollbackMetrics Metrics { get; set; } = null!;
}

/// <summary>
/// Rollback event during drill
/// </summary>
public class RollbackEvent
{
    public DateTime Timestamp { get; set; }
    public string EventType { get; set; } = string.Empty;
    public string Details { get; set; } = string.Empty;
    public bool Success { get; set; }
    public double LatencyMs { get; set; }
}

/// <summary>
/// Rollback metrics
/// </summary>
public class RollbackMetrics
{
    public double BaselineLatencyMs { get; set; }
    public int BaselineDecisionsCount { get; set; }
    public double PromotionLatencyMs { get; set; }
    public double LoadTestLatencyMs { get; set; }
    public int LoadTestDecisionsCount { get; set; }
    public double LoadTestSuccessRate { get; set; }
    public double RollbackLatencyMs { get; set; }
    public bool RollbackSuccess { get; set; }
    public int ConcurrentDecisionsDuringRollback { get; set; }
    public double PostRollbackLatencyMs { get; set; }
    public int PostRollbackDecisionsCount { get; set; }
    public double LatencyDegradationPercent { get; set; }
    public bool ContextPreserved { get; set; }
}

/// <summary>
/// Rollback drill summary across all drills
/// </summary>
public class RollbackDrillSummary
{
    public int TotalDrills { get; set; }
    public int SuccessfulDrills { get; set; }
    public double SuccessRate { get; set; }
    public double AverageRollbackTimeMs { get; set; }
    public double MaxRollbackTimeMs { get; set; }
    public DateTime LastDrillTime { get; set; }
    public string Message { get; set; } = string.Empty;
}