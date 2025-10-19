using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using BotCore.Health;

namespace BotCore.Services;

/// <summary>
/// 🎯 BOT SELF-AWARENESS SERVICE - PHASE 4
/// 
/// Main background service that orchestrates all self-awareness and monitoring capabilities.
/// This is the comprehensive orchestrator that ties together component discovery, health checking,
/// change detection, and intelligent reporting with plain English explanations.
/// 
/// Key Features:
/// - Discovers all bot components at startup
/// - Continuously monitors component health
/// - Detects health changes and reports them immediately
/// - Generates periodic status reports
/// - Provides AI-powered explanations for issues
/// - Maintains health history for trend analysis
/// </summary>
public sealed class BotSelfAwarenessService : BackgroundService
{
    private readonly ILogger<BotSelfAwarenessService> _logger;
    private readonly ComponentDiscoveryService _discoveryService;
    private readonly GenericHealthCheckService _healthCheckService;
    private readonly BotAlertService? _alertService;
    
    private List<DiscoveredComponent> _discoveredComponents = new();
    private readonly Dictionary<string, HealthCheckResult> _healthHistory = new();
    private DateTime _lastStatusReport = DateTime.MinValue;
    
    // Configuration
    private readonly bool _selfAwarenessEnabled;
    private readonly TimeSpan _healthCheckInterval;
    private readonly TimeSpan _statusReportInterval;
    private readonly TimeSpan _initialDelay = TimeSpan.FromSeconds(30);
    
    public BotSelfAwarenessService(
        ILogger<BotSelfAwarenessService> logger,
        ComponentDiscoveryService discoveryService,
        GenericHealthCheckService healthCheckService,
        IConfiguration configuration,
        BotAlertService? alertService = null)
    {
        _logger = logger;
        _discoveryService = discoveryService;
        _healthCheckService = healthCheckService;
        _alertService = alertService;
        
        ArgumentNullException.ThrowIfNull(configuration);
        
        // Read configuration
        _selfAwarenessEnabled = configuration["BOT_SELF_AWARENESS_ENABLED"]?.ToUpperInvariant() == "TRUE";
        
        var healthCheckMinutes = int.TryParse(configuration["BOT_HEALTH_CHECK_INTERVAL_MINUTES"], out var hcm) ? hcm : 5;
        _healthCheckInterval = TimeSpan.FromMinutes(healthCheckMinutes);
        
        var statusReportMinutes = int.TryParse(configuration["BOT_STATUS_REPORT_INTERVAL_MINUTES"], out var srm) ? srm : 60;
        _statusReportInterval = TimeSpan.FromMinutes(statusReportMinutes);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            // Check if self-awareness is enabled
            if (!_selfAwarenessEnabled)
            {
                _logger.LogInformation("🤖 [SELF-AWARENESS] Self-awareness system is disabled. Set BOT_SELF_AWARENESS_ENABLED=true to enable.");
                return;
            }
            
            // Wait for other services to start up
            await Task.Delay(_initialDelay, stoppingToken).ConfigureAwait(false);
            
            _logger.LogInformation("🤖 [SELF-AWARENESS] Bot self-awareness system starting...");
            _logger.LogInformation("🤖 [SELF-AWARENESS] Health check interval: {Interval} minutes", _healthCheckInterval.TotalMinutes);
            _logger.LogInformation("🤖 [SELF-AWARENESS] Status report interval: {Interval} minutes", _statusReportInterval.TotalMinutes);
            
            // Discover all components at startup
            _discoveredComponents = await _discoveryService.DiscoverAllComponentsAsync(stoppingToken).ConfigureAwait(false);
            
            if (_discoveredComponents.Count == 0)
            {
                _logger.LogWarning("⚠️ [SELF-AWARENESS] No components discovered - monitoring disabled");
                return;
            }
            
            _logger.LogInformation("✅ [SELF-AWARENESS] Discovered {Count} components to monitor", _discoveredComponents.Count);
            
            // Initialize health history
            try
            {
                foreach (var component in _discoveredComponents)
                {
                    if (component == null || string.IsNullOrWhiteSpace(component.Name))
                    {
                        _logger.LogWarning("⚠️ [SELF-AWARENESS] Skipping component with null or empty name");
                        continue;
                    }
                    
                    // Avoid duplicate keys
                    if (!_healthHistory.ContainsKey(component.Name))
                    {
                        _healthHistory[component.Name] = HealthCheckResult.Healthy("Initial state");
                    }
                    else
                    {
                        _logger.LogDebug("⚠️ [SELF-AWARENESS] Component {Name} already exists in health history", component.Name);
                    }
                }
                _logger.LogInformation("✅ [SELF-AWARENESS] Initialized health history for {Count} components", _healthHistory.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [SELF-AWARENESS] Error initializing health history");
                throw;
            }
            
            // Start monitoring loop
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await ExecuteMonitoringCycleAsync(stoppingToken).ConfigureAwait(false);
                    
                    // Wait for next health check interval
                    await Task.Delay(_healthCheckInterval, stoppingToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [SELF-AWARENESS] Error in monitoring cycle");
                    await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken).ConfigureAwait(false);
                }
            }
            
            _logger.LogInformation("🤖 [SELF-AWARENESS] Self-awareness system stopped");
        }
        catch (OperationCanceledException ex)
        {
            _logger.LogInformation(ex, "🤖 [SELF-AWARENESS] Self-awareness cancelled during startup");
        }
        catch (Exception ex)
        {
            _logger.LogCritical(ex, "💥 [SELF-AWARENESS] Critical error in self-awareness service");
        }
    }

    /// <summary>
    /// Execute a single monitoring cycle: check health, detect changes, report issues
    /// </summary>
    private async Task ExecuteMonitoringCycleAsync(CancellationToken cancellationToken)
    {
        // Step 1: Check health of all components
        var healthResults = await CheckAllComponentsHealthAsync(cancellationToken).ConfigureAwait(false);
        
        // Step 2: Detect health changes
        var healthChanges = DetectHealthChanges(healthResults);
        
        // Step 3: Report health changes immediately
        if (healthChanges.Count > 0)
        {
            await ReportHealthChangesAsync(healthChanges, cancellationToken).ConfigureAwait(false);
        }
        
        // Step 4: Update health history
        foreach (var result in healthResults)
        {
            _healthHistory[result.Key] = result.Value;
        }
        
        // Step 5: Generate periodic status report if needed
        if (ShouldGenerateStatusReport())
        {
            await GenerateStatusReportAsync(healthResults, cancellationToken).ConfigureAwait(false);
            _lastStatusReport = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Check health of all discovered components
    /// </summary>
    private async Task<Dictionary<string, HealthCheckResult>> CheckAllComponentsHealthAsync(CancellationToken cancellationToken)
    {
        var results = new Dictionary<string, HealthCheckResult>();
        
        foreach (var component in _discoveredComponents)
        {
            try
            {
                var healthResult = await _healthCheckService.CheckComponentHealthAsync(component, cancellationToken)
                    .ConfigureAwait(false);
                
                healthResult.ComponentName = component.Name;
                results[component.Name] = healthResult;
                
                // Update component status
                component.LastChecked = DateTime.UtcNow;
                component.LastStatus = healthResult.Status;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [SELF-AWARENESS] Failed to check health of {Component}", component.Name);
                results[component.Name] = HealthCheckResult.Unhealthy(
                    $"Health check failed: {ex.Message}",
                    new Dictionary<string, object> { ["Error"] = ex.Message });
            }
        }
        
        return results;
    }

    /// <summary>
    /// Detect changes in component health compared to previous state
    /// </summary>
    private List<HealthChange> DetectHealthChanges(Dictionary<string, HealthCheckResult> currentHealth)
    {
        var changes = new List<HealthChange>();
        
        foreach (var current in currentHealth)
        {
            if (!_healthHistory.TryGetValue(current.Key, out var previous))
            {
                continue;
            }
            
            // Detect status changes
            if (previous.Status != current.Value.Status)
            {
                var changeType = DetermineChangeType(previous.Status, current.Value.Status);
                
                changes.Add(new HealthChange
                {
                    ComponentName = current.Key,
                    PreviousStatus = previous.Status,
                    CurrentStatus = current.Value.Status,
                    ChangeType = changeType,
                    CurrentResult = current.Value,
                    DetectedAt = DateTime.UtcNow
                });
                
                _logger.LogWarning("🔄 [SELF-AWARENESS] Health change detected: {Component} {Previous} → {Current}",
                    current.Key, previous.Status, current.Value.Status);
            }
        }
        
        return changes;
    }

    /// <summary>
    /// Determine the type of health change
    /// </summary>
    private static HealthChangeType DetermineChangeType(string previousStatus, string currentStatus)
    {
        // Improvements
        if (previousStatus == "Unhealthy" && currentStatus == "Degraded")
            return HealthChangeType.Improvement;
        if (previousStatus == "Unhealthy" && currentStatus == "Healthy")
            return HealthChangeType.Recovered;
        if (previousStatus == "Degraded" && currentStatus == "Healthy")
            return HealthChangeType.Recovered;
        
        // Degradations
        if (previousStatus == "Healthy" && currentStatus == "Degraded")
            return HealthChangeType.Degraded;
        if (previousStatus == "Healthy" && currentStatus == "Unhealthy")
            return HealthChangeType.Failed;
        if (previousStatus == "Degraded" && currentStatus == "Unhealthy")
            return HealthChangeType.Failed;
        
        return HealthChangeType.Unknown;
    }

    /// <summary>
    /// Report health changes to alert service and logs
    /// </summary>
    private async Task ReportHealthChangesAsync(List<HealthChange> changes, CancellationToken cancellationToken)
    {
        foreach (var change in changes)
        {
            var emoji = GetChangeEmoji(change.ChangeType);
            var message = FormatHealthChangeMessage(change);
            
            _logger.LogWarning("{Emoji} [HEALTH-CHANGE] {Message}", emoji, message);
            
            // Send alert for critical changes
            if (_alertService != null && (change.ChangeType == HealthChangeType.Failed || change.ChangeType == HealthChangeType.Recovered))
            {
                try
                {
                    var alertType = change.ChangeType == HealthChangeType.Failed 
                        ? $"Component Failed: {change.ComponentName}" 
                        : $"Component Recovered: {change.ComponentName}";
                    
                    await _alertService.AlertSystemHealthAsync(alertType, message).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to send alert for health change");
                }
            }
        }
    }

    /// <summary>
    /// Format a health change into a plain English message
    /// </summary>
    private static string FormatHealthChangeMessage(HealthChange change)
    {
        var description = change.CurrentResult.Description ?? "No details available";
        
        return change.ChangeType switch
        {
            HealthChangeType.Failed => $"{change.ComponentName} has FAILED: {description}",
            HealthChangeType.Degraded => $"{change.ComponentName} is now DEGRADED: {description}",
            HealthChangeType.Recovered => $"{change.ComponentName} has RECOVERED: {description}",
            HealthChangeType.Improvement => $"{change.ComponentName} is IMPROVING: {description}",
            _ => $"{change.ComponentName} changed from {change.PreviousStatus} to {change.CurrentStatus}: {description}"
        };
    }

    /// <summary>
    /// Get emoji for health change type
    /// </summary>
    private static string GetChangeEmoji(HealthChangeType changeType)
    {
        return changeType switch
        {
            HealthChangeType.Failed => "❌",
            HealthChangeType.Degraded => "⚠️",
            HealthChangeType.Recovered => "✅",
            HealthChangeType.Improvement => "📈",
            _ => "🔄"
        };
    }

    /// <summary>
    /// Check if it's time to generate a periodic status report
    /// </summary>
    private bool ShouldGenerateStatusReport()
    {
        if (_lastStatusReport == DateTime.MinValue)
        {
            // Generate first report after one interval
            var firstComponentTime = _discoveredComponents.FirstOrDefault()?.DiscoveredAt ?? DateTime.UtcNow;
            return (DateTime.UtcNow - firstComponentTime) > _statusReportInterval;
        }
        
        return DateTime.UtcNow - _lastStatusReport >= _statusReportInterval;
    }

    /// <summary>
    /// Generate and log a comprehensive status report
    /// </summary>
    private Task GenerateStatusReportAsync(Dictionary<string, HealthCheckResult> currentHealth, CancellationToken cancellationToken)
    {
        var healthyCount = currentHealth.Count(h => h.Value.Status == "Healthy");
        var degradedCount = currentHealth.Count(h => h.Value.Status == "Degraded");
        var unhealthyCount = currentHealth.Count(h => h.Value.Status == "Unhealthy");
        
        _logger.LogInformation("📊 [STATUS-REPORT] ═══════════════════════════════════════");
        _logger.LogInformation("📊 [STATUS-REPORT] Bot Self-Awareness Status Report");
        _logger.LogInformation("📊 [STATUS-REPORT] Time: {Time:yyyy-MM-dd HH:mm:ss UTC}", DateTime.UtcNow);
        _logger.LogInformation("📊 [STATUS-REPORT] ═══════════════════════════════════════");
        _logger.LogInformation("📊 [STATUS-REPORT] Total Components: {Total}", currentHealth.Count);
        _logger.LogInformation("📊 [STATUS-REPORT] ✅ Healthy: {Healthy}", healthyCount);
        _logger.LogInformation("📊 [STATUS-REPORT] ⚠️ Degraded: {Degraded}", degradedCount);
        _logger.LogInformation("📊 [STATUS-REPORT] ❌ Unhealthy: {Unhealthy}", unhealthyCount);
        
        // Report unhealthy components
        if (unhealthyCount > 0)
        {
            _logger.LogWarning("📊 [STATUS-REPORT] ❌ UNHEALTHY COMPONENTS:");
            foreach (var component in currentHealth.Where(h => h.Value.Status == "Unhealthy"))
            {
                _logger.LogWarning("📊 [STATUS-REPORT]   - {Component}: {Description}", 
                    component.Key, component.Value.Description);
            }
        }
        
        // Report degraded components
        if (degradedCount > 0)
        {
            _logger.LogWarning("📊 [STATUS-REPORT] ⚠️ DEGRADED COMPONENTS:");
            foreach (var component in currentHealth.Where(h => h.Value.Status == "Degraded"))
            {
                _logger.LogWarning("📊 [STATUS-REPORT]   - {Component}: {Description}", 
                    component.Key, component.Value.Description);
            }
        }
        
        _logger.LogInformation("📊 [STATUS-REPORT] ═══════════════════════════════════════");
        
        return Task.CompletedTask;
    }

    /// <summary>
    /// Represents a detected change in component health
    /// </summary>
    private sealed class HealthChange
    {
        public string ComponentName { get; set; } = string.Empty;
        public string PreviousStatus { get; set; } = string.Empty;
        public string CurrentStatus { get; set; } = string.Empty;
        public HealthChangeType ChangeType { get; set; }
        public HealthCheckResult CurrentResult { get; set; } = HealthCheckResult.Healthy();
        public DateTime DetectedAt { get; set; }
    }

    /// <summary>
    /// Types of health changes
    /// </summary>
    private enum HealthChangeType
    {
        Unknown,
        Failed,         // Healthy/Degraded → Unhealthy
        Degraded,       // Healthy → Degraded
        Recovered,      // Unhealthy/Degraded → Healthy
        Improvement     // Unhealthy → Degraded
    }
}
