using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Scheduling;

/// <summary>
/// Internal Training Scheduler - Self-contained scheduling system for Lab training
/// Runs automatically on Sunday 12:00 PM - 5:45 PM ET without external schedulers
/// No Task Scheduler, no cron jobs - just pure self-contained scheduling
/// </summary>
internal sealed class InternalScheduler : BackgroundService
{
    private readonly ILogger<InternalScheduler> _logger;
    private readonly HistoricalTrainingOrchestrator _trainingOrchestrator;
    private bool _trainingInProgress = false;
    private bool _idleLogged = false;
    private DateTime? _lastTrainingStart = null;

    // Training window configuration (Eastern Time)
    private readonly TimeSpan TrainingWindowStart = new(12, 0, 0);  // 12:00 PM ET
    private readonly TimeSpan TrainingWindowEnd = new(17, 45, 0);   // 5:45 PM ET
    private readonly DayOfWeek TrainingDay = DayOfWeek.Sunday;

    // Optional: Daily maintenance window (5:00-5:15 PM ET Monday-Thursday)
    private readonly TimeSpan MaintenanceWindowStart = new(17, 0, 0);  // 5:00 PM ET
    private readonly TimeSpan MaintenanceWindowEnd = new(17, 15, 0);   // 5:15 PM ET

    public InternalScheduler(
        ILogger<InternalScheduler> logger,
        HistoricalTrainingOrchestrator trainingOrchestrator)
    {
        _logger = logger;
        _trainingOrchestrator = trainingOrchestrator;
        
        _logger.LogInformation("[LAB] Internal scheduler initialized - No external Task Scheduler needed");
    }

    /// <summary>
    /// Main scheduler loop - runs continuously while process is alive
    /// Checks clock every few minutes and starts training when it's time
    /// </summary>
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[LAB] Scheduler starting - will train every Sunday 12:00 PM - 5:45 PM ET");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Step 1: Get current time in Eastern Time
                var currentTime = GetEasternTime();
                
                // Step 2: Check if it's training time
                var isTrainingTime = IsTrainingTime(currentTime);

                if (isTrainingTime)
                {
                    // Reset idle flag when entering training time
                    if (_idleLogged)
                    {
                        _idleLogged = false;
                    }

                    // Step 3: Start training if not already in progress
                    if (!_trainingInProgress)
                    {
                        _logger.LogInformation("[LAB] Training window OPEN - Starting training session");
                        _trainingInProgress = true;
                        _lastTrainingStart = DateTime.UtcNow;

                        // Start training in background task so scheduler loop can continue
                        _ = Task.Run(async () =>
                        {
                            try
                            {
                                await _trainingOrchestrator.RunTrainingSessionAsync(stoppingToken).ConfigureAwait(false);
                                
                                _logger.LogInformation("[LAB] Training session complete - Entering idle mode");
                            }
                            catch (Exception ex)
                            {
                                _logger.LogError(ex, "[LAB] ERROR: Training session failed - {Error}", ex.Message);
                            }
                            finally
                            {
                                _trainingInProgress = false;
                            }
                        }, stoppingToken);
                    }
                    else
                    {
                        // Step 4: Training is in progress - log status periodically
                        var elapsed = DateTime.UtcNow - (_lastTrainingStart ?? DateTime.UtcNow);
                        _logger.LogInformation("[LAB] Training in progress: {Elapsed:F0} minutes elapsed", 
                            elapsed.TotalMinutes);
                    }

                    // Step 5: Sleep for 5 minutes during training time to check progress
                    await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken).ConfigureAwait(false);
                }
                else
                {
                    // Step 6: Not training time - enter idle mode
                    if (!_idleLogged)
                    {
                        var nextTraining = GetNextTrainingWindow(currentTime);
                        _logger.LogInformation("[LAB] Lab idle - next training: {NextTraining}", 
                            nextTraining.ToString("dddd MMM dd, h:mm tt") + " ET");
                        _idleLogged = true;
                    }

                    // Step 7: Sleep for 1 hour during idle to prevent CPU burn
                    await Task.Delay(TimeSpan.FromHours(1), stoppingToken).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during shutdown
                _logger.LogInformation("[LAB] Scheduler stopping - shutdown requested");
                break;
            }
            catch (Exception ex)
            {
                // Step 8: Handle errors gracefully - do not crash scheduler
                _logger.LogError(ex, "[LAB] ERROR: Scheduler encountered error - {Error}", ex.Message);
                _logger.LogError(ex, "[LAB] ERROR: Stack trace - {StackTrace}", ex.StackTrace);
                
                // Sleep 10 seconds after error to prevent rapid failure loop
                await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken).ConfigureAwait(false);
            }
        }

        _logger.LogInformation("[LAB] Scheduler stopped");
    }

    /// <summary>
    /// Check if current time falls within training window
    /// Training window: Sunday 12:00 PM - 5:45 PM ET
    /// </summary>
    private bool IsTrainingTime(DateTime easternTime)
    {
        var dayOfWeek = easternTime.DayOfWeek;
        var timeOfDay = easternTime.TimeOfDay;

        // All three conditions must be true for training time:
        // 1. Day is Sunday
        // 2. Time is >= 12:00 PM (noon)
        // 3. Time is < 5:45 PM
        return dayOfWeek == TrainingDay &&
               timeOfDay >= TrainingWindowStart &&
               timeOfDay < TrainingWindowEnd;
    }

    /// <summary>
    /// Check if current time is in daily maintenance window (OPTIONAL)
    /// Maintenance window: Monday-Thursday 5:00-5:15 PM ET
    /// </summary>
    private bool IsMaintenanceTime(DateTime easternTime)
    {
        var dayOfWeek = easternTime.DayOfWeek;
        var timeOfDay = easternTime.TimeOfDay;

        // Skip Friday (market closes at 5 PM Friday)
        if (dayOfWeek == DayOfWeek.Friday || dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return false;
        }

        return timeOfDay >= MaintenanceWindowStart &&
               timeOfDay < MaintenanceWindowEnd;
    }

    /// <summary>
    /// Calculate next Sunday training window
    /// </summary>
    private DateTime GetNextTrainingWindow(DateTime currentEasternTime)
    {
        var currentDate = currentEasternTime.Date;
        var timeOfDay = currentEasternTime.TimeOfDay;

        // If today is Sunday and before noon, next training is today at noon
        if (currentEasternTime.DayOfWeek == DayOfWeek.Sunday && timeOfDay < TrainingWindowStart)
        {
            return currentDate.Add(TrainingWindowStart);
        }

        // If today is Sunday and after 6 PM, next training is next Sunday
        if (currentEasternTime.DayOfWeek == DayOfWeek.Sunday && timeOfDay >= new TimeSpan(18, 0, 0))
        {
            return currentDate.AddDays(7).Add(TrainingWindowStart);
        }

        // Calculate days until next Sunday
        var daysUntilSunday = ((int)DayOfWeek.Sunday - (int)currentEasternTime.DayOfWeek + 7) % 7;
        if (daysUntilSunday == 0)
        {
            daysUntilSunday = 7; // Next Sunday, not today
        }

        return currentDate.AddDays(daysUntilSunday).Add(TrainingWindowStart);
    }

    /// <summary>
    /// Get current time in Eastern Time (handles DST properly)
    /// </summary>
    private DateTime GetEasternTime()
    {
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, easternZone);
        }
        catch
        {
            // Fallback to UTC-5 (EST) if timezone not found
            return DateTime.UtcNow.AddHours(-5);
        }
    }
}
