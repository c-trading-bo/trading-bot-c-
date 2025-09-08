namespace TradingBot.Abstractions;

// Kill Switch Interface
public interface IKillSwitchWatcher
{
    Task<bool> IsKillSwitchActiveAsync();
    bool IsKillSwitchActive { get; }
    event Action<bool> KillSwitchToggled;
    event Action OnKillSwitchActivated;
    Task StartWatchingAsync();
}

// Risk Management Interface  
public interface IRiskManager  
{
    Task<RiskAssessment> AssessRiskAsync(TradingDecision decision);
    bool IsRiskBreached { get; }
    event Action<RiskBreach> RiskBreachDetected;
    event Action<RiskBreach> OnRiskBreach;
}

// Health Monitoring Interface
public interface IHealthMonitor
{
    Task<HealthStatus> GetHealthStatusAsync(string componentName);
    bool IsTradingAllowed { get; }
    event Action<HealthStatus> HealthStatusChanged;
    event Action<HealthStatus> OnHealthChanged;
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
    public Dictionary<string, double> TechnicalIndicators { get; set; } = new();
    public RegimeState? CurrentRegime { get; set; }
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
    public Dictionary<string, object> Data { get; set; } = new();
}
