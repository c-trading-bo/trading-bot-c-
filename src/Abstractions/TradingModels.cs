using System.Text.Json.Serialization;

namespace TradingBot.Abstractions;

/// <summary>
/// Central Trading Brain State - The "ONE BRAIN" that knows everything
/// Contains real-time state of all trading components working together
/// </summary>
public class TradingBrainState
{
    public bool IsActive { get; set; } = false;
    public DateTime LastUpdate { get; set; } = DateTime.UtcNow;
    public List<string> ConnectedComponents { get; set; } = new();
    public List<string> ActiveStrategies { get; set; } = new();
    public Dictionary<string, decimal> ActivePositions { get; set; } = new();
    public decimal DailyPnL { get; set; } = 0m;
    public string MarketRegime { get; set; } = "UNKNOWN";
    public decimal RiskLevel { get; set; } = 0m;
    public Dictionary<string, object> ComponentStates { get; set; } = new();
    
    // Intelligence system state
    public MLSystemState MLState { get; set; } = new();
    public RiskSystemState RiskState { get; set; } = new();
    public TradingSystemState TradingState { get; set; } = new();
    public DataSystemState DataState { get; set; } = new();
    
    public TradingBrainState Clone()
    {
        return new TradingBrainState
        {
            IsActive = IsActive,
            LastUpdate = LastUpdate,
            ConnectedComponents = new List<string>(ConnectedComponents),
            ActiveStrategies = new List<string>(ActiveStrategies),
            ActivePositions = new Dictionary<string, decimal>(ActivePositions),
            DailyPnL = DailyPnL,
            MarketRegime = MarketRegime,
            RiskLevel = RiskLevel,
            ComponentStates = new Dictionary<string, object>(ComponentStates),
            MLState = MLState.Clone(),
            RiskState = RiskState.Clone(),
            TradingState = TradingState.Clone(),
            DataState = DataState.Clone()
        };
    }
}

/// <summary>
/// ML/RL System State
/// </summary>
public class MLSystemState
{
    public bool IsActive { get; set; } = false;
    public DateTime LastTraining { get; set; } = DateTime.MinValue;
    public Dictionary<string, decimal> ModelConfidences { get; set; } = new();
    public string ActiveModel { get; set; } = "NEURAL_BANDIT";
    public Dictionary<string, decimal> FeatureImportance { get; set; } = new();
    public string LastPrediction { get; set; } = string.Empty;
    public decimal PredictionConfidence { get; set; } = 0m;
    public List<string> ActiveFeatures { get; set; } = new();
    
    public MLSystemState Clone()
    {
        return new MLSystemState
        {
            IsActive = IsActive,
            LastTraining = LastTraining,
            ModelConfidences = new Dictionary<string, decimal>(ModelConfidences),
            ActiveModel = ActiveModel,
            FeatureImportance = new Dictionary<string, decimal>(FeatureImportance),
            LastPrediction = LastPrediction,
            PredictionConfidence = PredictionConfidence,
            ActiveFeatures = new List<string>(ActiveFeatures)
        };
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
    public Dictionary<string, decimal> PositionRisks { get; set; } = new();
    public List<string> RiskWarnings { get; set; } = new();
    public DateTime LastRiskCheck { get; set; } = DateTime.MinValue;
    
    public RiskSystemState Clone()
    {
        return new RiskSystemState
        {
            IsActive = IsActive,
            CurrentRisk = CurrentRisk,
            MaxDailyRisk = MaxDailyRisk,
            RiskUtilization = RiskUtilization,
            PositionRisks = new Dictionary<string, decimal>(PositionRisks),
            RiskWarnings = new List<string>(RiskWarnings),
            LastRiskCheck = LastRiskCheck
        };
    }
}

/// <summary>
/// Trading Execution System State
/// </summary>
public class TradingSystemState
{
    public bool IsConnected { get; set; } = false;
    public bool CanTrade { get; set; } = false;
    public int ActiveOrders { get; set; } = 0;
    public int FilledOrders { get; set; } = 0;
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
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Trading decision output
/// </summary>
public class TradingDecision
{
    public string DecisionId { get; set; } = string.Empty;
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
    public Dictionary<string, object> Reasoning { get; set; } = new();
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
    public List<string> Warnings { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Market Regime
/// </summary>
public class MarketRegime
{
    public string CurrentRegime { get; set; } = "UNKNOWN"; // TRENDING/RANGING/CHOPPY/VOLATILE
    public decimal Confidence { get; set; } = 0m;
    public Dictionary<string, decimal> RegimeScores { get; set; } = new();
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
    public Dictionary<string, object> Results { get; set; } = new();
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