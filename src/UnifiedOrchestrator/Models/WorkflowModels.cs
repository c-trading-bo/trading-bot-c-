using System.Text.Json.Serialization;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Unified workflow definition that consolidates all orchestrator features
/// </summary>
public class UnifiedWorkflow
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public int Priority { get; set; } = 3; // 1=Critical, 2=High, 3=Normal
    public int BudgetAllocation { get; set; } = 0; // Minutes per month
    public WorkflowSchedule Schedule { get; set; } = new();
    public string[] Actions { get; set; } = Array.Empty<string>();
    public WorkflowType Type { get; set; } = WorkflowType.Standard;
    public bool Enabled { get; set; } = true;
    public Dictionary<string, object> Configuration { get; set; } = new();
    public WorkflowMetrics Metrics { get; set; } = new();
}

/// <summary>
/// Unified schedule configuration supporting all timing patterns
/// </summary>
public class WorkflowSchedule
{
    public string? MarketHours { get; set; }        // During market hours
    public string? ExtendedHours { get; set; }      // Extended trading hours
    public string? Overnight { get; set; }          // Overnight sessions
    public string? CoreHours { get; set; }          // Core trading hours
    public string? FirstHour { get; set; }          // First hour of trading
    public string? LastHour { get; set; }           // Last hour of trading
    public string? Regular { get; set; }            // Regular intervals
    public string? Global { get; set; }             // 24/7 global
    public string? Weekends { get; set; }           // Weekend only
    public string? Disabled { get; set; }           // When to disable
    
    // Helper to get active schedule based on current time
    public string? GetActiveSchedule(DateTime utcNow)
    {
        var et = TimeZoneInfo.ConvertTimeFromUtc(utcNow, TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
        var isWeekend = et.DayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday;
        var hour = et.Hour;
        
        if (isWeekend && !string.IsNullOrEmpty(Weekends)) return Weekends;
        if (hour >= 9 && hour <= 16 && !string.IsNullOrEmpty(MarketHours)) return MarketHours;
        if (hour >= 9 && hour <= 11 && !string.IsNullOrEmpty(FirstHour)) return FirstHour;
        if (hour >= 15 && hour <= 16 && !string.IsNullOrEmpty(LastHour)) return LastHour;
        if (hour >= 9 && hour <= 11 || hour >= 14 && hour <= 16 && !string.IsNullOrEmpty(CoreHours)) return CoreHours;
        if ((hour >= 17 || hour <= 8) && !string.IsNullOrEmpty(Overnight)) return Overnight;
        if (!string.IsNullOrEmpty(ExtendedHours)) return ExtendedHours;
        if (!string.IsNullOrEmpty(Global)) return Global;
        return Regular;
    }
}

/// <summary>
/// Types of workflows in the unified system
/// </summary>
public enum WorkflowType
{
    Standard,
    Trading,
    Intelligence,
    RiskManagement,
    DataCollection,
    MachineLearning,
    Portfolio,
    Analytics
}

/// <summary>
/// Metrics tracking for workflow execution
/// </summary>
public class WorkflowMetrics
{
    public int ExecutionCount { get; set; } = 0;
    public int SuccessCount { get; set; } = 0;
    public int FailureCount { get; set; } = 0;
    public TimeSpan TotalExecutionTime { get; set; } = TimeSpan.Zero;
    public DateTime LastExecution { get; set; } = DateTime.MinValue;
    public DateTime LastSuccess { get; set; } = DateTime.MinValue;
    public DateTime LastFailure { get; set; } = DateTime.MinValue;
    public string? LastError { get; set; }
    public double SuccessRate => ExecutionCount > 0 ? (double)SuccessCount / ExecutionCount : 0.0;
    public TimeSpan AverageExecutionTime => ExecutionCount > 0 ? TimeSpan.FromTicks(TotalExecutionTime.Ticks / ExecutionCount) : TimeSpan.Zero;
}

/// <summary>
/// Execution context for workflow runs
/// </summary>
public class WorkflowExecutionContext
{
    public string WorkflowId { get; set; } = string.Empty;
    public string ExecutionId { get; set; } = Guid.NewGuid().ToString();
    public DateTime StartTime { get; set; } = DateTime.UtcNow;
    public DateTime? EndTime { get; set; }
    public WorkflowExecutionStatus Status { get; set; } = WorkflowExecutionStatus.Running;
    public Dictionary<string, object> Parameters { get; set; } = new();
    public List<string> Logs { get; set; } = new();
    public string? ErrorMessage { get; set; }
    public TimeSpan Duration => (EndTime ?? DateTime.UtcNow) - StartTime;
}

/// <summary>
/// Status of workflow execution
/// </summary>
public enum WorkflowExecutionStatus
{
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout
}

/// <summary>
/// Result of workflow execution
/// </summary>
public class WorkflowExecutionResult
{
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public Dictionary<string, object> Results { get; set; } = new();
    public TimeSpan Duration { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}