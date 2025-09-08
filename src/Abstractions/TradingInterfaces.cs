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
