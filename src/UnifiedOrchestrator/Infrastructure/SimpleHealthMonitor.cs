using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Infrastructure;

/// <summary>
/// Simple health monitor implementation
/// </summary>
public class SimpleHealthMonitor : IHealthMonitor
{
    private readonly ILogger<SimpleHealthMonitor> _logger;

    public SimpleHealthMonitor(ILogger<SimpleHealthMonitor> logger)
    {
        _logger = logger;
    }

    public bool IsTradingAllowed { get; private set; } = true;

    public event Action<HealthStatus>? HealthStatusChanged;
    public event Action<HealthStatus>? OnHealthChanged;

    public async Task<HealthStatus> GetHealthStatusAsync(string componentName)
    {
        await Task.Delay(1); // Make it async
        
        return new HealthStatus
        {
            IsHealthy = true,
            ComponentName = componentName,
            StatusMessage = "Component is healthy",
            TradingAllowed = true
        };
    }

    public async Task StartMonitoringAsync()
    {
        await Task.Delay(1); // Make it async
        _logger.LogInformation("Health monitor started");
    }
}