using System;

namespace Trading.Safety.Models;

/// <summary>
/// Comprehensive risk limit configuration for production trading
/// Supports daily, session, and absolute limits with escalation thresholds
/// </summary>
public class RiskLimits
{
    // Daily Risk Limits
    public decimal MaxDailyLoss { get; set; } = -1000m;
    public decimal MaxDailyProfit { get; set; } = 5000m; // Profit cap to prevent overconfidence
    public int MaxDailyTrades { get; set; } = 50;
    public decimal MaxDailyVolume { get; set; } = 100000m;
    
    // Session Risk Limits  
    public decimal MaxSessionLoss { get; set; } = -500m;
    public int MaxSessionTrades { get; set; } = 20;
    public TimeSpan MaxSessionDuration { get; set; } = TimeSpan.FromHours(8);
    
    // Position Risk Limits
    public decimal MaxPositionSize { get; set; } = 5000m;
    public decimal MaxPortfolioExposure { get; set; } = 25000m;
    public int MaxOpenPositions { get; set; } = 10;
    
    // Drawdown Limits
    public decimal MaxDrawdownAmount { get; set; } = -2000m;
    public decimal MaxDrawdownPercent { get; set; } = 0.20m; // 20% of peak
    
    // Risk Escalation Thresholds (% of limit before warnings)
    public decimal WarningThreshold { get; set; } = 0.80m; // 80%
    public decimal CriticalThreshold { get; set; } = 0.95m; // 95%
    
    // Risk Measurement Intervals
    public TimeSpan RiskEvaluationInterval { get; set; } = TimeSpan.FromMinutes(1);
    public TimeSpan PositionMonitoringInterval { get; set; } = TimeSpan.FromSeconds(30);
    
    // Risk Reset Schedule
    public TimeOnly DailyRiskResetTime { get; set; } = new TimeOnly(9, 30); // Market open
    public TimeOnly SessionBreakTime { get; set; } = new TimeOnly(12, 0); // Lunch break
    
    // Emergency Controls
    public bool EnableEmergencyStop { get; set; } = true;
    public bool EnableLossLimitStop { get; set; } = true;
    public bool EnablePositionLimitStop { get; set; } = true;
    public bool EnableTradeCountLimitStop { get; set; } = true;
    
    // Market Condition Adjustments
    public decimal VolatilityAdjustmentFactor { get; set; } = 0.8m; // Reduce limits in high vol
    public decimal LowVolatilityBoostFactor { get; set; } = 1.2m; // Increase limits in low vol
    public decimal NewsEventAdjustmentFactor { get; set; } = 0.6m; // Reduce during news
    
    // Validation
    public bool IsValid()
    {
        return MaxDailyLoss < 0 &&
               MaxSessionLoss < 0 &&
               MaxPositionSize > 0 &&
               MaxDailyTrades > 0 &&
               MaxSessionTrades > 0 &&
               WarningThreshold > 0 && WarningThreshold < 1 &&
               CriticalThreshold > WarningThreshold && CriticalThreshold < 1;
    }
}

/// <summary>
/// Real-time risk state tracking with persistence
/// </summary>
public record RiskState
{
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
    public DateTime DailyResetTime { get; set; }
    public DateTime SessionStartTime { get; set; }
    
    // Daily Metrics
    public decimal DailyPnL { get; set; }
    public decimal DailyPeak { get; set; }
    public decimal DailyDrawdown { get; set; }
    public int DailyTradeCount { get; set; }
    public decimal DailyVolume { get; set; }
    
    // Session Metrics
    public decimal SessionPnL { get; set; }
    public decimal SessionPeak { get; set; }
    public decimal SessionDrawdown { get; set; }
    public int SessionTradeCount { get; set; }
    public decimal SessionVolume { get; set; }
    
    // Position Metrics
    public decimal CurrentPositionSize { get; set; }
    public decimal MaxPositionSize { get; set; }
    public decimal TotalExposure { get; set; }
    public int OpenPositionCount { get; set; }
    
    // Risk Status
    public bool IsRiskBreached { get; set; }
    public List<string> ActiveBreaches { get; } = new();
    public List<string> ActiveWarnings { get; } = new();
    
    // Risk Level Assessment
    public string RiskLevel { get; set; } = "LOW"; // LOW, MEDIUM, HIGH, CRITICAL
    public decimal CompositeRiskScore { get; set; }
    
    // Market Condition Context
    public string MarketRegime { get; set; } = "NORMAL"; // NORMAL, VOLATILE, NEWS, ILLIQUID
    public decimal VolatilityAdjustment { get; set; } = 1.0m;
    
    // Reset daily metrics - create new instance for records
    public RiskState ResetDaily()
    {
        return this with
        {
            DailyPnL = 0,
            DailyPeak = 0,
            DailyDrawdown = 0,
            DailyTradeCount = 0,
            DailyVolume = 0,
            DailyResetTime = DateTime.UtcNow,
            LastUpdated = DateTime.UtcNow
        };
    }
    
    // Reset session metrics - create new instance for records
    public RiskState ResetSession()
    {
        return this with
        {
            SessionPnL = 0,
            SessionPeak = 0,
            SessionDrawdown = 0,
            SessionTradeCount = 0,
            SessionVolume = 0,
            SessionStartTime = DateTime.UtcNow,
            LastUpdated = DateTime.UtcNow
        };
    }
}

/// <summary>
/// Risk breach event with escalation level
/// </summary>
public class RiskBreachEvent
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string BreachType { get; set; } = string.Empty;
    public string Severity { get; set; } = string.Empty; // WARNING, CRITICAL, EMERGENCY
    public string Description { get; set; } = string.Empty;
    public decimal CurrentValue { get; set; }
    public decimal LimitValue { get; set; }
    public decimal UtilizationPercent { get; set; }
    public Dictionary<string, object> Context { get; } = new();
    public bool RequiresImmediateAction { get; set; }
    public List<string> RecommendedActions { get; } = new();
}