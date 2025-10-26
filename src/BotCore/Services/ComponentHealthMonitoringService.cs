using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using BotCore.Health;

namespace BotCore.Services;

/// <summary>
/// Background service that continuously monitors all bot components
/// and reports health issues in plain English.
/// This is the ORCHESTRATOR that makes the self-awareness system actually work.
/// </summary>
public sealed class ComponentHealthMonitoringService : BackgroundService
{
    private readonly ILogger<ComponentHealthMonitoringService> _logger;
    private readonly ComponentDiscoveryService _discoveryService;
    private readonly GenericHealthCheckService _healthCheckService;
    private readonly OllamaClient? _ollamaClient;

    // Health check interval
    private static readonly TimeSpan HealthCheckInterval = TimeSpan.FromMinutes(5);
    private static readonly TimeSpan InitialDelay = TimeSpan.FromSeconds(30);
    
    // Maximum number of metrics to display in health check logs
    private const int MaxMetricsToDisplay = 3;

    public ComponentHealthMonitoringService(
        ILogger<ComponentHealthMonitoringService> logger,
        ComponentDiscoveryService discoveryService,
        GenericHealthCheckService healthCheckService,
        OllamaClient? ollamaClient = null)
    {
        _logger = logger;
        _discoveryService = discoveryService;
        _healthCheckService = healthCheckService;
        _ollamaClient = ollamaClient;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            // Wait for other services to start up
            await Task.Delay(InitialDelay, stoppingToken).ConfigureAwait(false);

            _logger.LogInformation("🏥 [HEALTH-MONITOR] Starting component health monitoring service...");

            // Discover all components once at startup
            var components = await _discoveryService.DiscoverAllComponentsAsync(stoppingToken).ConfigureAwait(false);
            
            if (components.Count == 0)
            {
                _logger.LogWarning("⚠️ [HEALTH-MONITOR] No components discovered - monitoring disabled");
                // FIXED: Wait indefinitely instead of returning to prevent shutdown signal
                await Task.Delay(Timeout.Infinite, stoppingToken).ConfigureAwait(false);
                return;
            }

            _logger.LogInformation("🏥 [HEALTH-MONITOR] Monitoring {Count} components every {Interval} minutes", 
                components.Count, HealthCheckInterval.TotalMinutes);

            // Continuous monitoring loop
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await PerformHealthCheckCycleAsync(components, stoppingToken).ConfigureAwait(false);
                    
                    // Wait before next check
                    await Task.Delay(HealthCheckInterval, stoppingToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (InvalidOperationException ex)
                {
                    _logger.LogError(ex, "❌ [HEALTH-MONITOR] Error in health check cycle");
                    await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken).ConfigureAwait(false);
                }
            }

            _logger.LogInformation("🏥 [HEALTH-MONITOR] Component health monitoring stopped");
        }
        catch (OperationCanceledException ex)
        {
            _logger.LogInformation(ex, "🏥 [HEALTH-MONITOR] Monitoring cancelled during startup");
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogCritical(ex, "💥 [HEALTH-MONITOR] Critical error in health monitoring service");
        }
    }

    private async Task PerformHealthCheckCycleAsync(
        System.Collections.Generic.List<DiscoveredComponent> components,
        CancellationToken cancellationToken)
    {
        var unhealthyCount = 0;
        var degradedCount = 0;
        var healthyCount = 0;

        _logger.LogDebug("🔍 [HEALTH-MONITOR] Starting health check cycle for {Count} components", components.Count);

        foreach (var component in components)
        {
            try
            {
                var healthResult = await _healthCheckService.CheckComponentHealthAsync(component, cancellationToken)
                    .ConfigureAwait(false);

                // Update component status
                component.LastChecked = DateTime.UtcNow;
                component.LastStatus = healthResult.Status;

                // Count and log based on status
                if (healthResult.Status == "Unhealthy")
                {
                    unhealthyCount++;
                    await ReportUnhealthyComponentAsync(component, healthResult, cancellationToken).ConfigureAwait(false);
                }
                else if (healthResult.Status == "Degraded")
                {
                    degradedCount++;
                    await ReportDegradedComponentAsync(component, healthResult, cancellationToken).ConfigureAwait(false);
                }
                else
                {
                    healthyCount++;
                    _logger.LogTrace("✅ {Component}: {Status}", component.Name, healthResult.Status);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [HEALTH-MONITOR] Failed to check health of {Component}", component.Name);
                unhealthyCount++;
            }
        }

        // Summary log
        if (unhealthyCount > 0 || degradedCount > 0)
        {
            _logger.LogWarning("🏥 [HEALTH-MONITOR] Health Check Summary: ✅ {Healthy} Healthy, ⚠️ {Degraded} Degraded, ❌ {Unhealthy} Unhealthy",
                healthyCount, degradedCount, unhealthyCount);
        }
        else
        {
            _logger.LogInformation("🏥 [HEALTH-MONITOR] All {Count} components are healthy ✅", healthyCount);
        }
    }

    private async Task ReportUnhealthyComponentAsync(
        DiscoveredComponent component,
        HealthCheckResult healthResult,
        CancellationToken cancellationToken)
    {
        // Add relevant metrics to the message
        var metricsStr = string.Empty;
        if (healthResult.Metrics.Count > 0)
        {
            metricsStr = " (" + string.Join(", ", healthResult.Metrics.Take(MaxMetricsToDisplay).Select(kvp => $"{kvp.Key}={kvp.Value}")) + ")";
        }

        _logger.LogWarning("❌ {ComponentName}: {Status} - {Description}{Metrics}", component.Name, healthResult.Status, healthResult.Description, metricsStr);

        // If Ollama is available and enabled, generate a plain English explanation
        var selfAwarenessEnabled = Environment.GetEnvironmentVariable("BOT_SELF_AWARENESS_ENABLED")?.ToUpperInvariant() == "TRUE";
        if (_ollamaClient != null && selfAwarenessEnabled)
        {
            try
            {
                var prompt = $@"I am a trading bot. One of my components has FAILED:

Component: {component.Name}
Type: {component.Type}
Status: {healthResult.Status}
Issue: {healthResult.Description}
Metrics: {string.Join(", ", healthResult.Metrics.Select(kvp => $"{kvp.Key}={kvp.Value}"))}

Explain in one sentence what this means for my operation and what action should be taken. Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                if (!string.IsNullOrEmpty(response))
                {
                    _logger.LogWarning("🤖 [SELF-AWARENESS] {Response}", response);
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Failed to generate AI explanation for unhealthy component");
            }
        }
    }

    private async Task ReportDegradedComponentAsync(
        DiscoveredComponent component,
        HealthCheckResult healthResult,
        CancellationToken cancellationToken)
    {
        // Add relevant metrics
        var metricsStr = string.Empty;
        if (healthResult.Metrics.Count > 0)
        {
            metricsStr = " (" + string.Join(", ", healthResult.Metrics.Take(MaxMetricsToDisplay).Select(kvp => $"{kvp.Key}={kvp.Value}")) + ")";
        }

        _logger.LogInformation("⚠️ {ComponentName}: {Status} - {Description}{Metrics}", component.Name, healthResult.Status, healthResult.Description, metricsStr);

        // Generate AI explanation for degraded components if enabled
        var selfAwarenessEnabled = Environment.GetEnvironmentVariable("BOT_SELF_AWARENESS_ENABLED")?.ToUpperInvariant() == "TRUE";
        if (_ollamaClient != null && selfAwarenessEnabled)
        {
            try
            {
                var prompt = $@"I am a trading bot. One of my components is DEGRADED:

Component: {component.Name}
Type: {component.Type}
Status: {healthResult.Status}
Issue: {healthResult.Description}

Explain in one sentence what this means and if I should be concerned. Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                if (!string.IsNullOrEmpty(response))
                {
                    _logger.LogInformation("🤖 [SELF-AWARENESS] {Response}", response);
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Failed to generate AI explanation for degraded component");
            }
        }
    }
}
