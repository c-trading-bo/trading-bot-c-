namespace TradingBot.Abstractions;

// Kill Switch Interface
public interface IKillSwitchWatcher
{
    Task<bool> IsKillSwitchActiveAsync();
    bool IsKillSwitchActive { get; }
    event EventHandler<KillSwitchToggledEventArgs> KillSwitchToggled;
    event EventHandler? OnKillSwitchActivated;
    Task StartWatchingAsync();
}

// Risk Management Interface  
public interface IRiskManager  
{
    Task<RiskAssessment> AssessRiskAsync(TradingDecision decision);
    bool IsRiskBreached { get; }
    event EventHandler<RiskBreachEventArgs> RiskBreachDetected;
    event EventHandler<RiskBreachEventArgs> OnRiskBreach;
}

// Health Monitoring Interface
public interface IHealthMonitor
{
    Task<HealthStatus> GetHealthStatusAsync(string componentName);
    bool IsTradingAllowed { get; }
    event EventHandler<HealthStatusChangedEventArgs> HealthStatusChanged;
    event EventHandler<HealthStatusChangedEventArgs> OnHealthChanged;
    Task StartMonitoringAsync();
}

public class MarketContext
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public double Price { get; set; }
    public double Volume { get; set; }
    public double Bid { get; set; }
    public double Ask { get; set; }
    public Dictionary<string, double> TechnicalIndicators { get; } = new();
    public RegimeState? CurrentRegime { get; set; }
    
    // Additional properties for intelligence integration
    public string Regime { get; set; } = string.Empty;
    public double ModelConfidence { get; set; }
    public string PrimaryBias { get; set; } = string.Empty;
    public bool IsFomcDay { get; set; }
    public bool IsCpiDay { get; set; }
    public double NewsIntensity { get; set; }
    
    // Required properties mentioned in production readiness requirements
    public double SignalStrength { get; set; }
    public double ConfidenceLevel { get; set; }
    public Dictionary<string, double> Features { get; } = new();
}

public class MarketData
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double Volume { get; set; }
    public double Bid { get; set; }
    public double Ask { get; set; }
}

public class IntelligenceEventArgs : EventArgs
{
    public string EventType { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object> Data { get; } = new();
}

/// <summary>
/// Event arguments for kill switch toggle events
/// </summary>
public class KillSwitchToggledEventArgs : EventArgs
{
    public bool IsActive { get; }

    public KillSwitchToggledEventArgs(bool isActive)
    {
        IsActive = isActive;
    }
}

/// <summary>
/// Event arguments for risk breach events
/// </summary>
public class RiskBreachEventArgs : EventArgs
{
    public RiskBreach RiskBreach { get; }

    public RiskBreachEventArgs(RiskBreach riskBreach)
    {
        RiskBreach = riskBreach;
    }
}

/// <summary>
/// Event arguments for health status change events
/// </summary>
public class HealthStatusChangedEventArgs : EventArgs
{
    public HealthStatus HealthStatus { get; }

    public HealthStatusChangedEventArgs(HealthStatus healthStatus)
    {
        HealthStatus = healthStatus;
    }
}
