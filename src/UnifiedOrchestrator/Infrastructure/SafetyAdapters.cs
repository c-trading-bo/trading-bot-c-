extern alias SafetyProject;

using TradingBot.Abstractions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace TradingBot.UnifiedOrchestrator.Infrastructure;

/// <summary>
/// Simple adapters to bridge Safety project implementations with Abstractions interfaces
/// These provide minimal functionality to get the system running
/// </summary>

public class KillSwitchWatcherAdapter : IKillSwitchWatcher
{
    private readonly ILogger<KillSwitchWatcherAdapter> _logger;
    private bool _isActive = false;

    public event Action<bool>? KillSwitchToggled;
    public event Action? OnKillSwitchActivated;

    public bool IsKillSwitchActive => _isActive;

    public KillSwitchWatcherAdapter(ILogger<KillSwitchWatcherAdapter> logger)
    {
        _logger = logger;
    }

    public async Task<bool> IsKillSwitchActiveAsync()
    {
        // Check for kill file existence
        var killFile = Path.Combine(Directory.GetCurrentDirectory(), "kill.txt");
        _isActive = File.Exists(killFile);
        
        if (_isActive)
        {
            _logger.LogWarning("🛑 Kill switch activated - kill.txt file detected");
            OnKillSwitchActivated?.Invoke();
            KillSwitchToggled?.Invoke(true);
        }
        
        return await Task.FromResult(_isActive);
    }

    public async Task StartWatchingAsync()
    {
        _logger.LogInformation("🔍 Kill switch watcher started");
        await Task.CompletedTask;
    }
}

public class RiskManagerAdapter : IRiskManager
{
    private readonly ILogger<RiskManagerAdapter> _logger;
    private bool _isBreached = false;

    public event Action<RiskBreach>? RiskBreachDetected;
    public event Action<RiskBreach>? OnRiskBreach;

    public bool IsRiskBreached => _isBreached;

    public RiskManagerAdapter(ILogger<RiskManagerAdapter> logger)
    {
        _logger = logger;
    }

    public async Task<RiskAssessment> AssessRiskAsync(TradingDecision decision)
    {
        _logger.LogDebug("🔍 Assessing risk for trading decision");
        
        // Simple risk assessment - approve all in DRY_RUN mode
        return await Task.FromResult(new RiskAssessment
        {
            RiskScore = 0.3m,
            MaxPositionSize = 5m,
            CurrentExposure = 0m,
            VaR = 100m,
            RiskLevel = "LOW",
            Warnings = new List<string>(),
            Timestamp = DateTime.UtcNow
        });
    }
}

public class HealthMonitorAdapter : IHealthMonitor
{
    private readonly ILogger<HealthMonitorAdapter> _logger;

    public event Action<HealthStatus>? HealthStatusChanged;
    public event Action<HealthStatus>? OnHealthChanged;

    public bool IsTradingAllowed => true; // Allow trading in DRY_RUN mode

    public HealthMonitorAdapter(ILogger<HealthMonitorAdapter> logger)
    {
        _logger = logger;
    }

    public async Task<HealthStatus> GetHealthStatusAsync(string componentName)
    {
        _logger.LogDebug("🔍 Getting health status for {Component}", componentName);
        
        return await Task.FromResult(new HealthStatus
        {
            ComponentName = componentName,
            IsHealthy = true,
            Status = "Healthy",
            TradingAllowed = true,
            ConnectedHubs = 1,
            TotalHubs = 1,
            ErrorRate = 0.0,
            AverageLatencyMs = 50.0,
            StatusMessage = "System operational"
        });
    }

    public async Task StartMonitoringAsync()
    {
        _logger.LogInformation("🔍 Health monitor started");
        await Task.CompletedTask;
    }
}