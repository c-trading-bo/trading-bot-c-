using System.Text.Json.Serialization;

namespace TradingBot.Abstractions;

/// <summary>
/// Central Trading Brain State - The "ONE BRAIN" that knows everything
/// Contains real-time state of all trading components working together
/// </summary>
public class TradingBrainState
{
    public bool IsActive { get; set; }
    public bool IsSystemHealthy { get; set; } = true;
    public string CurrentMarketRegime { get; set; } = "UNKNOWN";
    public DateTime LastUpdate { get; set; } = DateTime.UtcNow;
    public List<string> ConnectedComponents { get; } = new();
    public List<string> ActiveStrategies { get; } = new();
    public Dictionary<string, decimal> ActivePositions { get; } = new();
    public decimal DailyPnL { get; set; } = 0m;
    public string MarketRegime { get; set; } = "UNKNOWN";
    public decimal RiskLevel { get; set; } = 0m;
    public Dictionary<string, object> ComponentStates { get; } = new();
    
    // Intelligence system state
    public MLSystemState MLState { get; set; } = new();
    public RiskSystemState RiskState { get; set; } = new();
    public TradingSystemState TradingState { get; set; } = new();
    public DataSystemState DataState { get; set; } = new();
    
    public TradingBrainState Clone()
    {
        var clone = new TradingBrainState
        {
            IsActive = IsActive,
            LastUpdate = LastUpdate,
            DailyPnL = DailyPnL,
            MarketRegime = MarketRegime,
            RiskLevel = RiskLevel,
            MLState = MLState.Clone(),
            RiskState = RiskState.Clone(),
            TradingState = TradingState.Clone(),
            DataState = DataState.Clone()
        };
        
        // Populate read-only collections
        foreach (var item in ConnectedComponents)
            clone.ConnectedComponents.Add(item);
        foreach (var item in ActiveStrategies)
            clone.ActiveStrategies.Add(item);
        foreach (var kvp in ActivePositions)
            clone.ActivePositions.Add(kvp.Key, kvp.Value);
        foreach (var kvp in ComponentStates)
            clone.ComponentStates.Add(kvp.Key, kvp.Value);
            
        return clone;
    }
}

/// <summary>
/// ML/RL System State
/// </summary>
public class MLSystemState
{
    public bool IsActive { get; set; } = false;
    public DateTime LastTraining { get; set; } = DateTime.MinValue;
    public Dictionary<string, decimal> ModelConfidences { get; } = new();
    public string ActiveModel { get; set; } = "NEURAL_BANDIT";
    public Dictionary<string, decimal> FeatureImportance { get; } = new();
    public string LastPrediction { get; set; } = string.Empty;
    public decimal PredictionConfidence { get; set; } = 0m;
    public List<string> ActiveFeatures { get; } = new();
    
    public MLSystemState Clone()
    {
        var clone = new MLSystemState
        {
            IsActive = IsActive,
            LastTraining = LastTraining,
            ActiveModel = ActiveModel,
            LastPrediction = LastPrediction,
            PredictionConfidence = PredictionConfidence
        };
        
        // Populate read-only collections
        foreach (var kvp in ModelConfidences)
            clone.ModelConfidences.Add(kvp.Key, kvp.Value);
        foreach (var kvp in FeatureImportance)
            clone.FeatureImportance.Add(kvp.Key, kvp.Value);
        foreach (var item in ActiveFeatures)
            clone.ActiveFeatures.Add(item);
            
        return clone;
    }
}

/// <summary>
/// Risk Management System State
/// </summary>
public class RiskSystemState
{
    public bool IsActive { get; set; } = false;
    public decimal CurrentRisk { get; set; } = 0m;
    public decimal MaxDailyRisk { get; set; } = 1000m;
    public decimal RiskUtilization { get; set; } = 0m;
    public Dictionary<string, decimal> PositionRisks { get; } = new();
    public List<string> RiskWarnings { get; } = new();
    public DateTime LastRiskCheck { get; set; } = DateTime.MinValue;
    
    public RiskSystemState Clone()
    {
        var clone = new RiskSystemState
        {
            IsActive = IsActive,
            CurrentRisk = CurrentRisk,
            MaxDailyRisk = MaxDailyRisk,
            RiskUtilization = RiskUtilization,
            LastRiskCheck = LastRiskCheck
        };
        
        // Populate read-only collections
        foreach (var kvp in PositionRisks)
            clone.PositionRisks.Add(kvp.Key, kvp.Value);
        foreach (var item in RiskWarnings)
            clone.RiskWarnings.Add(item);
            
        return clone;
    }
}

/// <summary>
/// Trading Execution System State
/// </summary>
public class TradingSystemState
{
    public bool IsConnected { get; set; }
    public bool CanTrade { get; set; }
    public int ActiveOrders { get; set; }
    public int FilledOrders { get; set; }
    public Dictionary<string, decimal> Positions { get; set; } = new();
    public Dictionary<string, string> ContractIds { get; set; } = new();
    public DateTime LastOrderTime { get; set; } = DateTime.MinValue;
    public string LastOrderStatus { get; set; } = string.Empty;
    
    public TradingSystemState Clone()
    {
        return new TradingSystemState
        {
            IsConnected = IsConnected,
            CanTrade = CanTrade,
            ActiveOrders = ActiveOrders,
            FilledOrders = FilledOrders,
            Positions = new Dictionary<string, decimal>(Positions),
            ContractIds = new Dictionary<string, string>(ContractIds),
            LastOrderTime = LastOrderTime,
            LastOrderStatus = LastOrderStatus
        };
    }
}

/// <summary>
/// Data Collection System State
/// </summary>
public class DataSystemState
{
    public bool IsActive { get; set; } = false;
    public DateTime LastDataUpdate { get; set; } = DateTime.MinValue;
    public Dictionary<string, DateTime> LastSymbolUpdate { get; set; } = new();
    public int TotalDataPoints { get; set; } = 0;
    public string DataQuality { get; set; } = "UNKNOWN";
    public List<string> DataSources { get; set; } = new();
    
    public DataSystemState Clone()
    {
        return new DataSystemState
        {
            IsActive = IsActive,
            LastDataUpdate = LastDataUpdate,
            LastSymbolUpdate = new Dictionary<string, DateTime>(LastSymbolUpdate),
            TotalDataPoints = TotalDataPoints,
            DataQuality = DataQuality,
            DataSources = new List<string>(DataSources)
        };
    }
}

/// <summary>
/// Trading message for the message bus
/// </summary>
public class TradingMessage
{
    public string Id { get; set; } = string.Empty;
    public string Topic { get; set; } = string.Empty;
    public string Payload { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string MessageType { get; set; } = string.Empty;
    public Dictionary<string, string> Headers { get; set; } = new();
}

/// <summary>
/// Trading signal input
/// </summary>
public class TradingSignal
{
    public string Symbol { get; set; } = string.Empty;
    public string Direction { get; set; } = string.Empty; // LONG/SHORT
    public decimal Strength { get; set; } = 0m; // 0-1
    public string Strategy { get; set; } = string.Empty;
    public decimal EntryPrice { get; set; } = 0m;
    public decimal StopLoss { get; set; } = 0m;
    public decimal TakeProfit { get; set; } = 0m;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object> Metadata { get; } = new();
}

/// <summary>
/// Trading decision output
/// </summary>
public class TradingDecision
{
    public string DecisionId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public TradeSide Side { get; set; } = TradeSide.Hold;
    public decimal Quantity { get; set; } = 0m;
    public decimal Price { get; set; } = 0m;
    public TradingSignal Signal { get; set; } = new();
    public TradingAction Action { get; set; } = TradingAction.Hold;
    public decimal Confidence { get; set; } = 0m;
    public decimal MLConfidence { get; set; } = 0m;
    public string MLStrategy { get; set; } = string.Empty;
    public decimal RiskScore { get; set; } = 0m;
    public decimal MaxPositionSize { get; set; } = 0m;
    public string MarketRegime { get; set; } = string.Empty;
    public decimal RegimeConfidence { get; set; } = 0m;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object> Reasoning { get; } = new();
}

/// <summary>
/// Trade side enum
/// </summary>
public enum TradeSide
{
    Hold,
    Buy,
    Sell,
    Close
}

/// <summary>
/// Trading action enum
/// </summary>
public enum TradingAction
{
    Hold,
    Buy,
    Sell,
    BuySmall,
    SellSmall,
    Close,
    ReducePosition
}

/// <summary>
/// Risk Assessment
/// </summary>
public class RiskAssessment
{
    public decimal RiskScore { get; set; } = 0m; // 0-1
    public decimal MaxPositionSize { get; set; } = 0m;
    public decimal CurrentExposure { get; set; } = 0m;
    public decimal VaR { get; set; } = 0m; // Value at Risk
    public string RiskLevel { get; set; } = "LOW"; // LOW/MEDIUM/HIGH
    public List<string> Warnings { get; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Market Regime
/// </summary>
public class MarketRegime
{
    public string CurrentRegime { get; set; } = "UNKNOWN"; // TRENDING/RANGING/CHOPPY/VOLATILE
    public decimal Confidence { get; set; } = 0m;
    public Dictionary<string, decimal> RegimeScores { get; } = new();
    public string Trend { get; set; } = "SIDEWAYS"; // UP/DOWN/SIDEWAYS
    public decimal Volatility { get; set; } = 0m;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Cloud trading recommendation from 27 GitHub workflows intelligence
/// </summary>
public class CloudTradingRecommendation
{
    public string Symbol { get; set; } = "";
    public DateTime Timestamp { get; set; }
    public string Signal { get; set; } = "NEUTRAL"; // BUY, SELL, NEUTRAL, ERROR
    public double Confidence { get; set; }
    public string Reasoning { get; set; } = "";
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
    public Dictionary<string, object> Results { get; } = new();
    public TimeSpan Duration { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

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
/// Trading session types - expanded to support all trading hours
/// </summary>
public enum SessionType
{
    Regular,    // Regular Trading Hours (9:30 AM - 4:00 PM ET)
    Extended,   // Extended Trading Hours (4:00 PM - 9:30 AM ET)
    Overnight,  // Overnight trading (10:00 PM - 6:00 AM ET)
    RTH = Regular,  // Alias for Regular Trading Hours
    ETH = Extended  // Alias for Extended Trading Hours
}

/// <summary>
/// Workflow execution status
/// </summary>
public enum WorkflowStatus
{
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled
}

/// <summary>
/// Unified schedule configuration supporting all timing patterns
/// </summary>
public class WorkflowSchedule
{
    // Market timing constants
    private const int SundayMarketOpenHour = 18;  // 6 PM ET
    private const int FridayMarketCloseHour = 17; // 5 PM ET
    private const int MaintenanceStartHour = 17;  // 5 PM ET - start of maintenance break
    private const int MaintenanceEndHour = 18;    // 6 PM ET - end of maintenance break
    
    // Standard market hours
    private const int MarketOpenHour = 9;    // 9 AM ET
    private const int MarketCloseHour = 16;  // 4 PM ET
    private const int FirstHourEnd = 10;     // 10 AM ET
    private const int LastHourStart = 15;    // 3 PM ET
    private const int CoreMorningEnd = 11;   // 11 AM ET
    private const int CoreAfternoonStart = 14; // 2 PM ET
    private const int OvernightStart = 18;   // 6 PM ET
    private const int OvernightEnd = 8;      // 8 AM ET
    
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
    
    // CME Futures Session Configuration
    public string? SessionOpen { get; set; }        // Sunday 6:00 PM ET session open
    public string? SessionClose { get; set; }       // Friday 5:00 PM ET session close
    public string? DailyBreakStart { get; set; }    // 5:00 PM ET daily break start
    public string? DailyBreakEnd { get; set; }      // 6:00 PM ET daily break end
    
    // Helper to get active schedule based on current time with CME futures sessions
    public string? GetActiveSchedule(DateTime utcNow)
    {
        try
        {
            // Convert to Eastern Time (handles DST automatically)
            var et = TimeZoneInfo.ConvertTimeFromUtc(utcNow, TimeZoneInfo.FindSystemTimeZoneById("America/New_York"));
            
            if (IsWeekend(et))
                return GetWeekendSchedule(et);
                
            return GetWeekdaySchedule(et);
        }
        catch
        {
            // Fallback to regular schedule on any timezone conversion errors
            return Regular ?? Global;
        }
    }

    private static bool IsWeekend(DateTime et)
    {
        return et.DayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday;
    }

    private string? GetWeekendSchedule(DateTime et)
    {
        var hour = et.Hour;
        var dayOfWeek = et.DayOfWeek;
        
        // Saturday: markets closed
        if (dayOfWeek == DayOfWeek.Saturday)
            return Disabled ?? Weekends;
            
        // Sunday: market opens at 6 PM ET
        if (dayOfWeek == DayOfWeek.Sunday)
        {
            if (hour >= SundayMarketOpenHour)
                return MarketHours ?? Regular;
            else
                return Disabled ?? Weekends;
        }
            
        return Weekends;
    }

    private string? GetWeekdaySchedule(DateTime et)
    {
        var hour = et.Hour;
        var dayOfWeek = et.DayOfWeek;
        
        // Friday: market closes at 5 PM ET
        if (dayOfWeek == DayOfWeek.Friday && hour >= FridayMarketCloseHour)
            return Disabled ?? Weekends;
        
        // Check for maintenance break (Monday-Thursday 5-6 PM ET)
        var maintenanceSchedule = GetMaintenanceSchedule(dayOfWeek, hour);
        if (maintenanceSchedule != null)
            return maintenanceSchedule;
        
        // Regular trading hours
        if (IsRegularTradingTime(dayOfWeek, hour))
            return GetTradingHoursSchedule(hour);
            
        // Global schedule for 24/7 operations
        if (!string.IsNullOrEmpty(Global)) 
            return Global;
            
        return Regular;
    }

    private string? GetMaintenanceSchedule(DayOfWeek dayOfWeek, int hour)
    {
        // Monday-Thursday: daily maintenance break 5-6 PM ET
        if (dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Thursday)
        {
            if (hour == MaintenanceStartHour) // 5 PM ET - start of maintenance break
                return Disabled;
            if (hour == MaintenanceEndHour) // 6 PM ET - end of maintenance break, session resumes
                return MarketHours ?? Regular;
        }
        return null;
    }

    private static bool IsRegularTradingTime(DayOfWeek dayOfWeek, int hour)
    {
        return (dayOfWeek == DayOfWeek.Sunday && hour >= SundayMarketOpenHour) || 
               (dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Thursday) ||
               (dayOfWeek == DayOfWeek.Friday && hour < FridayMarketCloseHour);
    }

    private string? GetTradingHoursSchedule(int hour)
    {
        // Check core market hours first
        var coreSchedule = GetCoreMarketSchedule(hour);
        if (coreSchedule != null)
            return coreSchedule;
            
        // Check extended/overnight hours
        return GetExtendedHoursSchedule(hour);
    }

    private string? GetCoreMarketSchedule(int hour)
    {
        // Check standard market hours first
        if (IsStandardMarketHour(hour))
            return MarketHours;
            
        // Check special trading periods
        return GetSpecialTradingPeriod(hour);
    }

    private bool IsStandardMarketHour(int hour)
    {
        return hour >= MarketOpenHour && hour <= MarketCloseHour && !string.IsNullOrEmpty(MarketHours);
    }

    private string? GetSpecialTradingPeriod(int hour)
    {
        // First hour of traditional market
        if (hour >= MarketOpenHour && hour <= FirstHourEnd && !string.IsNullOrEmpty(FirstHour)) 
            return FirstHour;
            
        // Last hour of traditional market  
        if (hour >= LastHourStart && hour <= MarketCloseHour && !string.IsNullOrEmpty(LastHour)) 
            return LastHour;
            
        // Core hours (morning and afternoon peaks)
        if (IsCoreHour(hour) && !string.IsNullOrEmpty(CoreHours)) 
            return CoreHours;
            
        return null;
    }

    private static bool IsCoreHour(int hour)
    {
        return (hour >= MarketOpenHour && hour <= CoreMorningEnd) || (hour >= CoreAfternoonStart && hour <= MarketCloseHour);
    }

    private string? GetExtendedHoursSchedule(int hour)
    {
        // Overnight/extended hours
        if ((hour >= OvernightStart || hour <= OvernightEnd) && !string.IsNullOrEmpty(Overnight)) 
            return Overnight;
            
        // Extended hours
        if (!string.IsNullOrEmpty(ExtendedHours)) 
            return ExtendedHours;
            
        // Fall back to regular market hours schedule
        return MarketHours ?? Regular;
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
    Analytics,
    CloudIntegration
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

// MLRecommendation Model
public class MLRecommendation
{
    public string Strategy { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public decimal TargetPrice { get; set; }
    public DateTime Timestamp { get; set; }
}

// Risk Breach Model
public class RiskBreach
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Type { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public decimal CurrentValue { get; set; }
    public decimal Limit { get; set; }
    public decimal Severity { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object> Details { get; set; } = new();
}

// Health Status Model
public class HealthStatus
{
    public string ComponentName { get; set; } = string.Empty;
    public bool IsHealthy { get; set; }
    public string Status { get; set; } = string.Empty;
    public bool TradingAllowed { get; set; }
    public int ConnectedHubs { get; set; }
    public int TotalHubs { get; set; }
    public double ErrorRate { get; set; }
    public double AverageLatencyMs { get; set; }
    public string StatusMessage { get; set; } = string.Empty;
    public DateTime LastCheck { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object> Metrics { get; set; } = new();
    public List<string> Issues { get; set; } = new();
}

// TopstepX SignalR message models
public class GatewayUserOrder
{
    public string AccountId { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty; // "New", "Open", "Filled", "Cancelled", "Rejected"
    public string Reason { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public int Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; }
}

public class GatewayUserTrade
{
    public string AccountId { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public decimal FillPrice { get; set; }
    public int Quantity { get; set; }
    public DateTime Time { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
}

public class OrderBookData
{
    public string Symbol { get; set; } = string.Empty;
    public decimal BidPrice { get; set; }
    public int BidSize { get; set; }
    public decimal AskPrice { get; set; }
    public int AskSize { get; set; }
    public DateTime Timestamp { get; set; }
}

public class TradeConfirmation
{
    public string OrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public int Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; }
}