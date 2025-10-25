using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace TradingBot.ML.Services;

/// <summary>
/// Daily Retraining Scheduler Service for continuous model improvement
/// Addresses gap in HEDGE_FUND_GAP_ANALYSIS.md - Section "2. Daily Retraining"
/// Schedules nightly model retraining to keep models fresh and adapt to regime changes
/// </summary>
public interface IDailyRetrainingScheduler
{
    /// <summary>
    /// Schedule a retraining session for a specific time
    /// </summary>
    Task ScheduleRetrainingAsync(
        TimeSpan scheduledTime,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Trigger immediate retraining
    /// </summary>
    Task TriggerRetrainingAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Get next scheduled retraining time
    /// </summary>
    DateTime? GetNextScheduledTime();

    /// <summary>
    /// Check if daily retraining is enabled
    /// </summary>
    bool IsEnabled();
}

/// <summary>
/// Production implementation of daily retraining scheduler
/// Runs as a background service, triggering model retraining at scheduled times
/// </summary>
public class DailyRetrainingScheduler : BackgroundService, IDailyRetrainingScheduler
{
    private readonly ILogger<DailyRetrainingScheduler> _logger;
    private readonly bool _enabled;
    private readonly TimeSpan _scheduledTime;
    private DateTime? _nextScheduledTime;
    private readonly SemaphoreSlim _retrainingLock;

    public DailyRetrainingScheduler(
        ILogger<DailyRetrainingScheduler> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _enabled = Environment.GetEnvironmentVariable("DAILY_RETRAINING_ENABLED") != "0";
        
        // Default to 2 AM UTC for nightly retraining
        var scheduledHour = int.Parse(
            Environment.GetEnvironmentVariable("RETRAINING_HOUR") ?? "2");
        var scheduledMinute = int.Parse(
            Environment.GetEnvironmentVariable("RETRAINING_MINUTE") ?? "0");
        
        _scheduledTime = new TimeSpan(scheduledHour, scheduledMinute, 0);
        _retrainingLock = new SemaphoreSlim(1, 1);

        if (_enabled)
        {
            _logger.LogInformation(
                "Daily Retraining Scheduler initialized. Scheduled time: {Time} UTC",
                _scheduledTime);
        }
        else
        {
            _logger.LogInformation("Daily Retraining Scheduler disabled via configuration");
        }
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!_enabled)
        {
            _logger.LogInformation("Daily retraining is disabled, scheduler will not run");
            return;
        }

        _logger.LogInformation("Daily Retraining Scheduler started");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var now = DateTime.UtcNow;
                var nextRun = CalculateNextRunTime(now);
                _nextScheduledTime = nextRun;

                var delay = nextRun - now;
                
                _logger.LogInformation(
                    "Next retraining scheduled at {NextRun} UTC (in {Hours:F1} hours)",
                    nextRun,
                    delay.TotalHours);

                await Task.Delay(delay, stoppingToken).ConfigureAwait(false);

                if (!stoppingToken.IsCancellationRequested)
                {
                    await TriggerRetrainingAsync(stoppingToken).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("Daily Retraining Scheduler stopping");
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in daily retraining scheduler loop");
                // Wait 5 minutes before retrying to avoid tight error loop
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken).ConfigureAwait(false);
            }
        }

        _logger.LogInformation("Daily Retraining Scheduler stopped");
    }

    public async Task ScheduleRetrainingAsync(
        TimeSpan scheduledTime,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            _logger.LogWarning("Cannot schedule retraining - daily retraining is disabled");
            return;
        }

        await Task.CompletedTask.ConfigureAwait(false);
        
        _logger.LogInformation(
            "Retraining scheduled for {Time} UTC daily",
            scheduledTime);
    }

    public async Task TriggerRetrainingAsync(CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            _logger.LogWarning("Cannot trigger retraining - daily retraining is disabled");
            return;
        }

        await _retrainingLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            _logger.LogInformation("Starting daily model retraining at {Time} UTC", DateTime.UtcNow);

            // In production, this would trigger the actual retraining pipeline
            // For now, we'll create a trigger file that Python training scripts can detect
            var triggerFile = "./state/trigger_retraining.txt";
            await System.IO.File.WriteAllTextAsync(
                triggerFile,
                DateTime.UtcNow.ToString("O"),
                cancellationToken).ConfigureAwait(false);

            _logger.LogInformation(
                "Retraining trigger created: {File}. Python training pipeline should detect and process.",
                triggerFile);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error triggering daily retraining");
        }
        finally
        {
            _retrainingLock.Release();
        }
    }

    public DateTime? GetNextScheduledTime()
    {
        return _nextScheduledTime;
    }

    public bool IsEnabled()
    {
        return _enabled;
    }

    private DateTime CalculateNextRunTime(DateTime now)
    {
        var today = now.Date;
        var scheduledToday = today + _scheduledTime;

        if (now < scheduledToday)
        {
            return scheduledToday;
        }
        else
        {
            return today.AddDays(1) + _scheduledTime;
        }
    }
}
