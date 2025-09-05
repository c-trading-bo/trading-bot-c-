using System.Text.Json.Serialization;

namespace TradingBot.UnifiedOrchestrator.Models;

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
/// ML Recommendation
/// </summary>
public class MLRecommendation
{
    public string RecommendedStrategy { get; set; } = string.Empty;
    public decimal Confidence { get; set; } = 0m;
    public Dictionary<string, decimal> StrategyScores { get; set; } = new();
    public string[] Features { get; set; } = Array.Empty<string>();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
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
/// Strategy Performance
/// </summary>
public class StrategyPerformance
{
    public string StrategyName { get; set; } = string.Empty;
    public decimal Returns { get; set; } = 0m;
    public decimal Sharpe { get; set; } = 0m;
    public decimal MaxDrawdown { get; set; } = 0m;
    public int TotalTrades { get; set; } = 0;
    public decimal WinRate { get; set; } = 0m;
    public bool IsActive { get; set; } = false;
    public DateTime LastTrade { get; set; } = DateTime.MinValue;
}

/// <summary>
/// Component Health Status
/// </summary>
public class ComponentHealth
{
    public string ComponentName { get; set; } = string.Empty;
    public bool IsHealthy { get; set; } = false;
    public string Status { get; set; } = "UNKNOWN";
    public Dictionary<string, object> Metrics { get; set; } = new();
    public List<string> Errors { get; set; } = new();
    public DateTime LastCheck { get; set; } = DateTime.UtcNow;
    public TimeSpan Uptime { get; set; } = TimeSpan.Zero;
}