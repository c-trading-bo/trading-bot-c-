using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Scheduling;

/// <summary>
/// Maintenance Scheduler - Coordinates all periodic cleanup and maintenance tasks
/// Day 22: Phase 6 - MaintenanceScheduler Updates
/// 
/// Coordinates:
/// - LogRetentionService (daily at 2 AM)
/// - DataRetentionService (daily at 3 AM)
/// 
/// Features:
/// - Centralized monitoring of cleanup task execution
/// - Alerting for cleanup failures
/// - Health checks for maintenance services
/// - Graceful shutdown handling
/// </summary>
internal sealed class MaintenanceScheduler : BackgroundService
{
    private readonly ILogger<MaintenanceScheduler> _logger;
    private readonly LogRetentionService _logRetention;
    private readonly DataRetentionService _dataRetention;
    private readonly TrainingAlertService _alertService;
    private readonly PeriodicTimer _monitoringTimer;
    private DateTime _lastLogRetentionRun = DateTime.MinValue;
    private DateTime _lastDataRetentionRun = DateTime.MinValue;

    public MaintenanceScheduler(
        ILogger<MaintenanceScheduler> logger,
        LogRetentionService logRetention,
        DataRetentionService dataRetention,
        TrainingAlertService alertService)
    {
        _logger = logger;
        _logRetention = logRetention;
        _dataRetention = dataRetention;
        _alertService = alertService;
        _monitoringTimer = new PeriodicTimer(TimeSpan.FromHours(1)); // Check every hour
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[MAINTENANCE] Maintenance Scheduler starting - Coordinating cleanup services");
        _logger.LogInformation("[MAINTENANCE]   - LogRetentionService: Daily at 2:00 AM (30-day retention)");
        _logger.LogInformation("[MAINTENANCE]   - DataRetentionService: Daily at 3:00 AM (configurable retention)");

        try
        {
            // Initial health check
            await PerformHealthCheckAsync(stoppingToken).ConfigureAwait(false);

            // Monitoring loop
            while (await _monitoringTimer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
            {
                try
                {
                    await PerformHealthCheckAsync(stoppingToken).ConfigureAwait(false);
                }
                catch (Exception ex) when (!(ex is OperationCanceledException))
                {
                    _logger.LogError(ex, "[MAINTENANCE] Health check failed: {Error}", ex.Message);
                }
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("[MAINTENANCE] Maintenance Scheduler stopping - shutdown requested");
        }
        finally
        {
            _logger.LogInformation("[MAINTENANCE] Maintenance Scheduler stopped");
        }
    }

    /// <summary>
    /// Perform periodic health checks on cleanup services
    /// Monitors:
    /// - Last execution time (detect stuck/failed services)
    /// - Service health status
    /// - Alert on failures
    /// </summary>
    private async Task PerformHealthCheckAsync(CancellationToken cancellationToken)
    {
        var now = DateTime.UtcNow;
        var issues = new System.Collections.Generic.List<string>();

        // Check LogRetentionService health
        // Expected to run daily at 2 AM - alert if not run in 26 hours
        if (_lastLogRetentionRun != DateTime.MinValue)
        {
            var timeSinceLastRun = now - _lastLogRetentionRun;
            if (timeSinceLastRun.TotalHours > 26)
            {
                issues.Add($"LogRetentionService not run in {timeSinceLastRun.TotalHours:F1} hours (expected daily)");
            }
        }

        // Check DataRetentionService health
        // Expected to run daily at 3 AM - alert if not run in 26 hours
        if (_lastDataRetentionRun != DateTime.MinValue)
        {
            var timeSinceLastRun = now - _lastDataRetentionRun;
            if (timeSinceLastRun.TotalHours > 26)
            {
                issues.Add($"DataRetentionService not run in {timeSinceLastRun.TotalHours:F1} hours (expected daily)");
            }
        }

        // Log health status
        if (issues.Count == 0)
        {
            // Only log every 24 hours when healthy to reduce noise
            if ((now - _lastLogRetentionRun).TotalHours >= 24 || _lastLogRetentionRun == DateTime.MinValue)
            {
                _logger.LogInformation("[MAINTENANCE] All cleanup services healthy");
            }
        }
        else
        {
            _logger.LogWarning("[MAINTENANCE] Cleanup service issues detected: {Issues}",
                string.Join("; ", issues));

            // Send alert for maintenance failures
            await _alertService.AlertHealthCheckFailureAsync(
                "Maintenance Services",
                string.Join("; ", issues),
                cancellationToken).ConfigureAwait(false);
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Report successful log retention execution
    /// Called by LogRetentionService after successful cleanup
    /// </summary>
    public void ReportLogRetentionSuccess()
    {
        _lastLogRetentionRun = DateTime.UtcNow;
        _logger.LogInformation("[MAINTENANCE] LogRetentionService executed successfully at {Time}",
            _lastLogRetentionRun.ToString("yyyy-MM-dd HH:mm:ss UTC"));
    }

    /// <summary>
    /// Report successful data retention execution
    /// Called by DataRetentionService after successful cleanup
    /// </summary>
    public void ReportDataRetentionSuccess()
    {
        _lastDataRetentionRun = DateTime.UtcNow;
        _logger.LogInformation("[MAINTENANCE] DataRetentionService executed successfully at {Time}",
            _lastDataRetentionRun.ToString("yyyy-MM-dd HH:mm:ss UTC"));
    }

    /// <summary>
    /// Report log retention failure
    /// Called by LogRetentionService if cleanup fails
    /// </summary>
    public async Task ReportLogRetentionFailureAsync(string error, CancellationToken cancellationToken)
    {
        _logger.LogError("[MAINTENANCE] LogRetentionService failed: {Error}", error);

        await _alertService.AlertHealthCheckFailureAsync(
            "LogRetentionService",
            error,
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Report data retention failure
    /// Called by DataRetentionService if cleanup fails
    /// </summary>
    public async Task ReportDataRetentionFailureAsync(string error, CancellationToken cancellationToken)
    {
        _logger.LogError("[MAINTENANCE] DataRetentionService failed: {Error}", error);

        await _alertService.AlertHealthCheckFailureAsync(
            "DataRetentionService",
            error,
            cancellationToken).ConfigureAwait(false);
    }

    public override void Dispose()
    {
        _monitoringTimer?.Dispose();
        base.Dispose();
    }
}
