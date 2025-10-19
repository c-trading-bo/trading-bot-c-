using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Scheduling;

/// <summary>
/// Internal Training Scheduler - Production-grade scheduling system for Lab training
/// Runs automatically on Sunday 12:00 PM - 5:45 PM America/New_York timezone
/// Features: DST handling, lock files, health checks, watchdog, proper event-driven architecture
/// </summary>
internal sealed class InternalScheduler : BackgroundService
{
    private readonly ILogger<InternalScheduler> _logger;
    private readonly HistoricalTrainingOrchestrator _trainingOrchestrator;
    private readonly SemaphoreSlim _trainingLock = new(1, 1);
    private bool _idleLogged = false;
    private DateTime? _lastTrainingStart = null;
    private readonly string _lockFilePath;
    private readonly TimeZoneInfo _easternTimeZone;

    // Training window configuration (Eastern Time)
    private readonly TimeSpan TrainingWindowStart = new(12, 0, 0);  // 12:00 PM ET
    private readonly TimeSpan TrainingWindowEnd = new(17, 45, 0);   // 5:45 PM ET
    private readonly DayOfWeek TrainingDay = DayOfWeek.Sunday;
    private readonly TimeSpan MaxTrainingDuration = TimeSpan.FromHours(5); // 5 hour watchdog

    public InternalScheduler(
        ILogger<InternalScheduler> logger,
        HistoricalTrainingOrchestrator trainingOrchestrator)
    {
        _logger = logger;
        _trainingOrchestrator = trainingOrchestrator;
        _lockFilePath = Path.Combine(Path.GetTempPath(), "qbot_lab_training.lock");
        
        // Initialize Eastern timezone - handles DST automatically
        try
        {
            _easternTimeZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            _logger.LogInformation("[LAB] Scheduler initialized with America/New_York timezone (DST-aware)");
        }
        catch
        {
            // Fallback for systems without timezone database
            _easternTimeZone = TimeZoneInfo.CreateCustomTimeZone(
                "Eastern", TimeSpan.FromHours(-5), "Eastern Time", "EST");
            _logger.LogWarning("[LAB] Using fallback timezone (EST -5). Install tzdata for proper DST handling.");
        }
        
        // Clean up stale lock files on startup
        CleanupStaleLockFile();
        
        _logger.LogInformation("[LAB] Scheduler initialized - Production-grade with lock files, health checks, watchdog");
    }

    /// <summary>
    /// Main scheduler loop - event-driven with proper shutdown handling
    /// Uses WaitHandle for efficient sleeping instead of busy loop
    /// </summary>
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[LAB] Scheduler starting - Training Sunday 12:00 PM - 5:45 PM America/New_York");

        using var timer = new PeriodicTimer(TimeSpan.FromMinutes(5)); // Event-driven, not busy loop

        try
        {
            do
            {
                try
                {
                    var easternTime = GetEasternTime();
                    var isTrainingTime = IsTrainingTime(easternTime);

                    if (isTrainingTime)
                    {
                        if (_idleLogged)
                        {
                            _idleLogged = false;
                        }

                        // Use semaphore for proper concurrency control
                        if (await _trainingLock.WaitAsync(0, stoppingToken).ConfigureAwait(false))
                        {
                            try
                            {
                                // Check lock file - prevent concurrent runs
                                if (File.Exists(_lockFilePath))
                                {
                                    var lockInfo = await ReadLockFileAsync().ConfigureAwait(false);
                                    if (IsLockStale(lockInfo))
                                    {
                                        _logger.LogWarning("[LAB] Stale lock detected - cleaning up and proceeding");
                                        CleanupStaleLockFile();
                                    }
                                    else
                                    {
                                        _logger.LogInformation("[LAB] Training already in progress (lock file exists)");
                                        continue;
                                    }
                                }

                                // Pre-training health checks
                                if (!await RunHealthChecksAsync(stoppingToken).ConfigureAwait(false))
                                {
                                    _logger.LogError("[LAB] Health checks failed - skipping training");
                                    continue;
                                }

                                _logger.LogInformation("[LAB] Training window OPEN - Starting training with watchdog");
                                
                                // Create lock file
                                await CreateLockFileAsync().ConfigureAwait(false);
                                _lastTrainingStart = DateTime.UtcNow;

                                // Run training with watchdog timeout
                                using var trainingCts = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken);
                                trainingCts.CancelAfter(MaxTrainingDuration);

                                try
                                {
                                    await _trainingOrchestrator.RunTrainingSessionAsync(trainingCts.Token).ConfigureAwait(false);
                                    _logger.LogInformation("[LAB] Training completed successfully");
                                }
                                catch (OperationCanceledException) when (trainingCts.IsCancellationRequested && !stoppingToken.IsCancellationRequested)
                                {
                                    _logger.LogError("[LAB] Training TIMEOUT - exceeded {Hours} hour maximum", MaxTrainingDuration.TotalHours);
                                }
                                catch (Exception ex)
                                {
                                    _logger.LogError(ex, "[LAB] Training failed: {Error}", ex.Message);
                                }
                                finally
                                {
                                    // Always clean up lock file
                                    CleanupStaleLockFile();
                                }
                            }
                            finally
                            {
                                _trainingLock.Release();
                            }
                        }
                        else
                        {
                            var elapsed = DateTime.UtcNow - (_lastTrainingStart ?? DateTime.UtcNow);
                            _logger.LogInformation("[LAB] Training in progress: {Elapsed:F0} minutes", elapsed.TotalMinutes);
                        }
                    }
                    else
                    {
                        // Idle mode - log once and use efficient waiting
                        if (!_idleLogged)
                        {
                            var nextTraining = GetNextTrainingWindow(easternTime);
                            _logger.LogInformation("[LAB] Idle - next training: {NextTraining}",
                                nextTraining.ToString("dddd MMM dd, h:mm tt") + " ET");
                            _idleLogged = true;
                        }
                    }
                }
                catch (Exception ex) when (!(ex is OperationCanceledException))
                {
                    _logger.LogError(ex, "[LAB] Scheduler error: {Error}", ex.Message);
                    await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken).ConfigureAwait(false);
                }
            }
            while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false));
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("[LAB] Scheduler stopping - shutdown requested");
        }
        finally
        {
            CleanupStaleLockFile();
            _logger.LogInformation("[LAB] Scheduler stopped");
        }
    }

    /// <summary>
    /// Run pre-training health checks - must pass before starting training
    /// </summary>
    private async Task<bool> RunHealthChecksAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("[LAB] Running pre-training health checks...");
        
        try
        {
            // Check 1: Sufficient disk space (require at least 10GB free)
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
            if (Directory.Exists(dataPath))
            {
                var drive = new DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
                var freeSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
                if (freeSpaceGB < 10)
                {
                    _logger.LogError("[LAB] Health check FAILED: Insufficient disk space ({Free:F1} GB < 10 GB required)", freeSpaceGB);
                    return false;
                }
                _logger.LogInformation("[LAB] ✓ Disk space: {Free:F1} GB available", freeSpaceGB);
            }

            // Check 2: Model registry accessible
            var modelRegistry = Path.Combine(Directory.GetCurrentDirectory(), "model_registry");
            Directory.CreateDirectory(modelRegistry); // Ensure it exists
            var testFile = Path.Combine(modelRegistry, ".health_check");
            try
            {
                await File.WriteAllTextAsync(testFile, DateTime.UtcNow.ToString(), cancellationToken).ConfigureAwait(false);
                File.Delete(testFile);
                _logger.LogInformation("[LAB] ✓ Model registry writable");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LAB] Health check FAILED: Model registry not writable");
                return false;
            }

            // Check 3: Historical data directory exists
            var historicalDataPath = Path.Combine(dataPath, "historical");
            if (!Directory.Exists(historicalDataPath))
            {
                _logger.LogWarning("[LAB] Historical data directory missing - will be created");
                Directory.CreateDirectory(historicalDataPath);
            }
            _logger.LogInformation("[LAB] ✓ Historical data directory accessible");

            // Check 4: Experiences database/directory exists
            var experiencesPath = Path.Combine("state", "learning");
            if (!Directory.Exists(experiencesPath))
            {
                _logger.LogWarning("[LAB] Experiences directory missing - training may have limited data");
                Directory.CreateDirectory(experiencesPath);
            }
            _logger.LogInformation("[LAB] ✓ Experiences directory accessible");

            _logger.LogInformation("[LAB] All health checks passed");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LAB] Health checks failed with exception: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Create lock file to prevent concurrent training runs
    /// </summary>
    private async Task CreateLockFileAsync()
    {
        var lockInfo = new
        {
            PID = Environment.ProcessId,
            StartTime = DateTime.UtcNow,
            MachineName = Environment.MachineName
        };
        
        var lockContent = System.Text.Json.JsonSerializer.Serialize(lockInfo);
        await File.WriteAllTextAsync(_lockFilePath, lockContent).ConfigureAwait(false);
        _logger.LogInformation("[LAB] Lock file created: {LockFile}", _lockFilePath);
    }

    /// <summary>
    /// Read lock file information
    /// </summary>
    private async Task<(int PID, DateTime StartTime)?> ReadLockFileAsync()
    {
        try
        {
            if (!File.Exists(_lockFilePath))
                return null;

            var content = await File.ReadAllTextAsync(_lockFilePath).ConfigureAwait(false);
            using var doc = System.Text.Json.JsonDocument.Parse(content);
            var root = doc.RootElement;
            
            return (
                root.GetProperty("PID").GetInt32(),
                root.GetProperty("StartTime").GetDateTime()
            );
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Check if lock file is stale (process no longer running or > 6 hours old)
    /// </summary>
    private bool IsLockStale((int PID, DateTime StartTime)? lockInfo)
    {
        if (!lockInfo.HasValue)
            return true;

        // Check if lock is older than 6 hours (training should never take this long)
        if ((DateTime.UtcNow - lockInfo.Value.StartTime).TotalHours > 6)
        {
            _logger.LogWarning("[LAB] Lock file is stale (> 6 hours old)");
            return true;
        }

        // Check if process is still running
        try
        {
            var process = System.Diagnostics.Process.GetProcessById(lockInfo.Value.PID);
            return false; // Process exists, lock is valid
        }
        catch
        {
            _logger.LogWarning("[LAB] Lock file process {PID} no longer running", lockInfo.Value.PID);
            return true; // Process doesn't exist, lock is stale
        }
    }

    /// <summary>
    /// Clean up stale lock file
    /// </summary>
    private void CleanupStaleLockFile()
    {
        try
        {
            if (File.Exists(_lockFilePath))
            {
                File.Delete(_lockFilePath);
                _logger.LogInformation("[LAB] Lock file cleaned up");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LAB] Failed to clean up lock file: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Check if current time falls within training window
    /// Training window: Sunday 12:00 PM - 5:45 PM America/New_York (DST-aware)
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
    /// Calculate next Sunday training window in Eastern Time
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
    /// Get current time in America/New_York timezone (handles DST automatically)
    /// </summary>
    private DateTime GetEasternTime()
    {
        return TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, _easternTimeZone);
    }

    public override void Dispose()
    {
        _trainingLock?.Dispose();
        CleanupStaleLockFile();
        base.Dispose();
    }
}
