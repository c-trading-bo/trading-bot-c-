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
    private readonly Training.TrainingOrchestratorService? _enhancedOrchestrator;
    private readonly ResourcePreCheckService _resourceChecker;
    private readonly TrainingAlertService _alertService;
    private readonly SemaphoreSlim _trainingLock = new(1, 1);
    private bool _idleLogged = false;
    private DateTime? _lastTrainingStart = null;
    private readonly string _lockFilePath;
    private readonly string _checkpointFilePath;
    private readonly TimeZoneInfo _easternTimeZone;
    private CancellationTokenSource? _currentTrainingCts;
    private DateTime _lastIdleHealthCheck = DateTime.MinValue;
    private DateTime _lastIdleCountdownDisplay = DateTime.MinValue;

    // Training window configuration (Eastern Time)
    private readonly TimeSpan TrainingWindowStart = new(12, 0, 0);  // 12:00 PM ET
    private readonly TimeSpan TrainingWindowEnd = new(17, 45, 0);   // 5:45 PM ET
    private readonly DayOfWeek TrainingDay = DayOfWeek.Sunday;
    private readonly TimeSpan MaxTrainingDuration = TimeSpan.FromHours(5); // 5 hour watchdog
    private readonly TimeSpan PreWarmTime = new(11, 55, 0);  // 11:55 AM ET (5 min before training)

    public InternalScheduler(
        ILogger<InternalScheduler> logger,
        HistoricalTrainingOrchestrator trainingOrchestrator,
        ResourcePreCheckService resourceChecker,
        TrainingAlertService alertService,
        Training.TrainingOrchestratorService? enhancedOrchestrator = null)
    {
        _logger = logger;
        _trainingOrchestrator = trainingOrchestrator;
        _enhancedOrchestrator = enhancedOrchestrator;
        _resourceChecker = resourceChecker;
        _alertService = alertService;
        _lockFilePath = Path.Combine(Path.GetTempPath(), "qbot_lab_training.lock");
        _checkpointFilePath = Path.Combine(Directory.GetCurrentDirectory(), "state", "training_checkpoint.json");

        if (_enhancedOrchestrator != null)
        {
            _logger.LogInformation("[LAB] Using enhanced TrainingOrchestratorService with progress tracking");
        }

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

        // Check for incomplete training runs
        CheckForIncompleteTrainingAsync().ConfigureAwait(false);

        _logger.LogInformation("[LAB] Scheduler initialized - Production-grade with lock files, health checks, watchdog, graceful shutdown");
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

                                // Pre-training health checks (basic)
                                if (!await RunHealthChecksAsync(stoppingToken).ConfigureAwait(false))
                                {
                                    _logger.LogError("[LAB] Health checks failed - skipping training");
                                    await _alertService.AlertHealthCheckFailureAsync(
                                        "Basic health checks",
                                        "Pre-training validation failed",
                                        stoppingToken).ConfigureAwait(false);
                                    continue;
                                }

                                // Resource pre-checks
                                var (resourcesOk, failedChecks) = await _resourceChecker.RunAllChecksAsync(stoppingToken).ConfigureAwait(false);
                                if (!resourcesOk)
                                {
                                    _logger.LogError("[LAB] Resource checks failed: {Checks}", string.Join(", ", failedChecks));
                                    await _alertService.AlertHealthCheckFailureAsync(
                                        "Resource checks",
                                        $"Failed: {string.Join(", ", failedChecks)}",
                                        stoppingToken).ConfigureAwait(false);
                                    continue;
                                }

                                _logger.LogInformation("[LAB] Training window OPEN - Starting training with watchdog");

                                // NOTE: Lock file is created by TrainingOrchestratorService.StartTrainingSessionAsync()
                                // Do NOT create it here or we'll conflict with ourselves
                                _lastTrainingStart = DateTime.UtcNow;

                                // Run training with watchdog timeout (5 hours = 18,000,000 milliseconds)
                                _currentTrainingCts = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken);
                                // BUGFIX: Explicitly convert to milliseconds to avoid any ambiguity in overload resolution
                                // MaxTrainingDuration = TimeSpan.FromHours(5) = 18,000,000 ms
                                var timeoutMilliseconds = (int)MaxTrainingDuration.TotalMilliseconds;
                                _logger.LogInformation("[LAB] Setting training timeout to {Hours} hours ({Milliseconds:N0} ms)", 
                                    MaxTrainingDuration.TotalHours, timeoutMilliseconds);
                                _currentTrainingCts.CancelAfter(timeoutMilliseconds);

                                try
                                {
                                    // Use enhanced orchestrator if available, fallback to legacy
                                    if (_enhancedOrchestrator != null)
                                    {
                                        _logger.LogInformation("[LAB] Starting enhanced training session with progress tracking");

                                        // Start session
                                        var session = await _enhancedOrchestrator.StartTrainingSessionAsync(_currentTrainingCts.Token).ConfigureAwait(false);

                                        // Alert training started
                                        await _alertService.AlertTrainingStartedAsync(
                                            session.SessionId,
                                            "N/A",
                                            new Dictionary<string, object>
                                            {
                                                ["TotalComponents"] = session.ComponentsTotal
                                            },
                                            stoppingToken).ConfigureAwait(false);

                                        // Run health checks
                                        if (!await _enhancedOrchestrator.RunPreTrainingHealthChecksAsync(session, _currentTrainingCts.Token).ConfigureAwait(false))
                                        {
                                            throw new InvalidOperationException("Pre-training health checks failed");
                                        }

                                        // Execute training phases
                                        // Heavy phase contains actual model training (CVaRPPO, NeuralUCB, LSTM, etc.)
                                        await _enhancedOrchestrator.ExecuteTrainingPhaseAsync(session, Training.TrainingPhase.Heavy, _currentTrainingCts.Token).ConfigureAwait(false);
                                        
                                        // Medium and Light phases execute but components are runtime optimization, not training
                                        await _enhancedOrchestrator.ExecuteTrainingPhaseAsync(session, Training.TrainingPhase.Medium, _currentTrainingCts.Token).ConfigureAwait(false);
                                        await _enhancedOrchestrator.ExecuteTrainingPhaseAsync(session, Training.TrainingPhase.Light, _currentTrainingCts.Token).ConfigureAwait(false);

                                        // Run validation and promotion
                                        await _enhancedOrchestrator.RunPostTrainingValidationAsync(session, _currentTrainingCts.Token).ConfigureAwait(false);
                                        await _enhancedOrchestrator.EvaluateAndPromoteModelsAsync(session, _currentTrainingCts.Token).ConfigureAwait(false);

                                        // Generate summary
                                        var summary = await _enhancedOrchestrator.GenerateSessionSummaryAsync(session, _currentTrainingCts.Token).ConfigureAwait(false);

                                        // Cleanup
                                        await _enhancedOrchestrator.CleanupAndFinalizeAsync(session, _currentTrainingCts.Token).ConfigureAwait(false);

                                        _logger.LogInformation("[LAB] Enhanced training session completed successfully");
                                    }
                                    else
                                    {
                                        // Fallback to legacy orchestrator
                                        await _alertService.AlertTrainingStartedAsync(
                                            "training_session",
                                            "N/A",
                                            new Dictionary<string, object>(),
                                            stoppingToken).ConfigureAwait(false);

                                        await _trainingOrchestrator.RunTrainingSessionAsync(_currentTrainingCts.Token).ConfigureAwait(false);
                                        _logger.LogInformation("[LAB] Training completed successfully");

                                        // Alert success
                                        await _alertService.AlertTrainingSuccessAsync(
                                            "training_session",
                                            (DateTime.UtcNow - _lastTrainingStart.Value).TotalMinutes,
                                            0, 0,
                                            new Dictionary<string, object>(),
                                            stoppingToken).ConfigureAwait(false);
                                    }
                                }
                                catch (OperationCanceledException) when (_currentTrainingCts.IsCancellationRequested && !stoppingToken.IsCancellationRequested)
                                {
                                    _logger.LogError("[LAB] Training TIMEOUT - exceeded {Hours} hour maximum", MaxTrainingDuration.TotalHours);
                                    await _alertService.AlertTrainingTimeoutAsync(
                                        "training_session",
                                        MaxTrainingDuration.TotalHours,
                                        stoppingToken).ConfigureAwait(false);
                                }
                                catch (Exception ex)
                                {
                                    _logger.LogError(ex, "[LAB] Training failed: {Error}", ex.Message);
                                    await _alertService.AlertTrainingFailureAsync(
                                        "training_session",
                                        ex.Message,
                                        new List<string> { ex.GetType().Name },
                                        stoppingToken).ConfigureAwait(false);
                                }
                                finally
                                {
                                    // Always clean up lock file
                                    CleanupStaleLockFile();
                                    _currentTrainingCts?.Dispose();
                                    _currentTrainingCts = null;
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
                        // Idle mode - enter enhanced idle state management
                        await EnterIdleStateAsync(easternTime, stoppingToken).ConfigureAwait(false);
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

    #region Idle State Management (Phase 10)

    /// <summary>
    /// Enter enhanced idle state loop with health monitoring and countdown display
    /// Phase 10: Idle State Management
    /// </summary>
    private async Task EnterIdleStateAsync(DateTime easternTime, CancellationToken cancellationToken)
    {
        // Display idle state message once when entering
        if (!_idleLogged)
        {
            var nextTraining = GetNextTrainingWindow(easternTime);
            var timeUntilTraining = CalculateTimeUntilNextTraining(easternTime);

            _logger.LogInformation(@"
╔═══════════════════════════════════════════════════════════════════════════╗
║                        LAB MODE - IDLE STATE                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Status:               IDLE - Waiting for next Sunday training             ║
║ Current Time:         {CurrentTime,-50} ║
║ Next Training:        {NextTraining,-50} ║
║ Countdown:            {Countdown,-50} ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Watchdog:             Active (will wake automatically)                    ║
║ Health Checks:        Running hourly (ensuring system readiness)          ║
║ Lock File:            Cleared (no concurrent session prevention)          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Market Status:        {MarketStatus,-50} ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Press Ctrl+C to exit gracefully                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝",
                easternTime.ToString("dddd, MMM dd yyyy, h:mm:ss tt") + " ET",
                nextTraining.ToString("dddd, MMM dd yyyy, h:mm tt") + " ET",
                FormatCountdown(timeUntilTraining),
                GetMarketStatus(easternTime));

            DisplayWatchdogStatus();
            _idleLogged = true;
            _lastIdleCountdownDisplay = DateTime.UtcNow;
        }

        // Check if we need to display hourly countdown update
        var timeSinceLastCountdown = DateTime.UtcNow - _lastIdleCountdownDisplay;
        if (timeSinceLastCountdown.TotalHours >= 1)
        {
            await DisplayIdleCountdownAsync(easternTime, cancellationToken).ConfigureAwait(false);
            _lastIdleCountdownDisplay = DateTime.UtcNow;
        }

        // Run hourly health checks during idle
        var timeSinceLastHealthCheck = DateTime.UtcNow - _lastIdleHealthCheck;
        if (timeSinceLastHealthCheck.TotalHours >= 1 || _lastIdleHealthCheck == DateTime.MinValue)
        {
            await RunIdleHealthCheckAsync(cancellationToken).ConfigureAwait(false);
            _lastIdleHealthCheck = DateTime.UtcNow;
        }

        // Check if we're within 5 minutes of training start - pre-warm systems
        var timeUntilTrainingNow = CalculateTimeUntilNextTraining(easternTime);
        if (timeUntilTrainingNow.TotalMinutes <= 5 && timeUntilTrainingNow.TotalMinutes > 0)
        {
            await PreWarmSystemsAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Calculate time until next Sunday 12:00 PM ET training session
    /// Phase 10: Idle State Management
    /// </summary>
    private TimeSpan CalculateTimeUntilNextTraining(DateTime currentEasternTime)
    {
        var nextTraining = GetNextTrainingWindow(currentEasternTime);
        var nowUtc = DateTime.UtcNow;

        // Convert next training ET to UTC for accurate calculation
        try
        {
            var nextTrainingUtc = TimeZoneInfo.ConvertTimeToUtc(nextTraining, _easternTimeZone);
            return nextTrainingUtc - nowUtc;
        }
        catch
        {
            // Fallback - assume EST (UTC-5)
            var nextTrainingUtc = nextTraining.AddHours(5);
            return nextTrainingUtc - nowUtc;
        }
    }

    /// <summary>
    /// Display countdown update every hour during idle state
    /// Phase 10: Idle State Management
    /// </summary>
    private async Task DisplayIdleCountdownAsync(DateTime easternTime, CancellationToken cancellationToken)
    {
        var nextTraining = GetNextTrainingWindow(easternTime);
        var timeUntilTraining = CalculateTimeUntilNextTraining(easternTime);

        _logger.LogInformation("[LAB] Next Training: {NextTraining} (in {Countdown}) - Current: {CurrentTime}",
            nextTraining.ToString("dddd, MMM dd yyyy, h:mm tt") + " ET",
            FormatCountdown(timeUntilTraining),
            easternTime.ToString("h:mm:ss tt") + " ET");

        _logger.LogDebug("[LAB] Watchdog monitoring active - System ready for next session");

        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Run hourly health checks during idle state to ensure system stays ready
    /// Phase 10: Idle State Management
    /// </summary>
    private async Task RunIdleHealthCheckAsync(CancellationToken cancellationToken)
    {
        _logger.LogDebug("[LAB] Running hourly health check during idle state...");

        try
        {
            var issues = new List<string>();

            // Check 1: Disk space (critical below 20GB)
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
            if (Directory.Exists(dataPath))
            {
                var drive = new DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
                var freeSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);

                if (freeSpaceGB < 20)
                {
                    issues.Add($"Low disk space: {freeSpaceGB:F1} GB (critical below 20 GB)");
                }
            }

            // Check 2: Historical data files still exist and readable
            var historicalDataPath = Path.Combine(dataPath, "historical");
            if (!Directory.Exists(historicalDataPath))
            {
                issues.Add("Historical data directory missing");
            }

            // Check 3: Model registry directory writable
            var modelRegistry = Path.Combine(Directory.GetCurrentDirectory(), "model_registry");
            if (!Directory.Exists(modelRegistry))
            {
                Directory.CreateDirectory(modelRegistry);
            }

            var testFile = Path.Combine(modelRegistry, ".health_check_idle");
            try
            {
                await File.WriteAllTextAsync(testFile, DateTime.UtcNow.ToString(), cancellationToken).ConfigureAwait(false);
                File.Delete(testFile);
            }
            catch (Exception ex)
            {
                issues.Add($"Model registry not writable: {ex.Message}");
            }

            // Check 4: Experience database accessible
            var experiencesPath = Path.Combine("state", "learning");
            if (!Directory.Exists(experiencesPath))
            {
                issues.Add("Experiences directory missing (will impact training quality)");
            }

            // Check 5: Clean up any stale lock files
            if (File.Exists(_lockFilePath))
            {
                var lockInfo = await ReadLockFileAsync().ConfigureAwait(false);
                if (IsLockStale(lockInfo))
                {
                    _logger.LogWarning("[LAB] Cleaning up stale lock file during idle health check");
                    CleanupStaleLockFile();
                }
            }

            // Log results
            if (issues.Count == 0)
            {
                _logger.LogDebug("[LAB] Hourly health check: All systems nominal");
            }
            else if (issues.Count <= 2)
            {
                _logger.LogWarning("[LAB] Hourly health check: Issues detected - {Issues}",
                    string.Join("; ", issues));

                // Send alert for issues
                await _alertService.AlertHealthCheckFailureAsync(
                    "Idle health check",
                    string.Join("; ", issues),
                    cancellationToken).ConfigureAwait(false);
            }
            else
            {
                _logger.LogError("[LAB] System unhealthy - training may fail - {Issues}",
                    string.Join("; ", issues));

                // Send critical alert
                await _alertService.AlertHealthCheckFailureAsync(
                    "CRITICAL: Idle health check",
                    string.Join("; ", issues),
                    cancellationToken).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LAB] Hourly health check failed: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Pre-warm systems 5 minutes before training window starts
    /// Phase 10: Idle State Management
    /// </summary>
    private async Task PreWarmSystemsAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("[LAB] Pre-warming systems (5 minutes before training window)...");

        try
        {
            // Pre-warm 1: Initialize data directory access (warm filesystem cache)
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
            if (Directory.Exists(dataPath))
            {
                _ = Directory.GetFiles(dataPath, "*", SearchOption.TopDirectoryOnly);
                _logger.LogDebug("[LAB] ✓ Data directory warmed");
            }

            // Pre-warm 2: Open database connection pool (if using SQLite)
            var experiencesPath = Path.Combine("state", "learning");
            if (Directory.Exists(experiencesPath))
            {
                _ = Directory.GetFiles(experiencesPath, "*.db", SearchOption.TopDirectoryOnly);
                _logger.LogDebug("[LAB] ✓ Experience database paths cached");
            }

            // Pre-warm 3: Initialize model registry
            var modelRegistry = Path.Combine(Directory.GetCurrentDirectory(), "model_registry");
            if (Directory.Exists(modelRegistry))
            {
                _ = Directory.GetFiles(modelRegistry, "*.onnx", SearchOption.TopDirectoryOnly);
                _logger.LogDebug("[LAB] ✓ Model registry warmed");
            }

            // Pre-warm 4: Allocate memory buffers (force GC and compact)
            GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
            _logger.LogDebug("[LAB] ✓ Memory compacted and ready");

            _logger.LogInformation("[LAB] System pre-warming complete - ready for training");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LAB] System pre-warming encountered issues: {Error}", ex.Message);
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Display watchdog status during idle state
    /// Phase 10: Idle State Management
    /// </summary>
    private void DisplayWatchdogStatus()
    {
        var nextHealthCheck = _lastIdleHealthCheck == DateTime.MinValue
            ? DateTime.UtcNow
            : _lastIdleHealthCheck.AddHours(1);

        _logger.LogDebug("[LAB] Watchdog Status:");
        _logger.LogDebug("[LAB]   - Active: YES (will wake for next session automatically)");
        _logger.LogDebug("[LAB]   - Health checks: Every 1 hour (ensuring readiness)");
        _logger.LogDebug("[LAB]   - Lock file: Cleared");
        _logger.LogDebug("[LAB]   - Next check: {NextCheck}",
            nextHealthCheck.ToString("yyyy-MM-dd HH:mm:ss UTC"));
    }

    /// <summary>
    /// Get current market status (for display during idle state)
    /// Phase 10: Idle State Management
    /// </summary>
    private string GetMarketStatus(DateTime easternTime)
    {
        var dayOfWeek = easternTime.DayOfWeek;
        var timeOfDay = easternTime.TimeOfDay;

        // Weekend
        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return "Closed (Weekend)";
        }

        // Pre-market: 4:00 AM - 9:30 AM ET
        if (timeOfDay >= new TimeSpan(4, 0, 0) && timeOfDay < new TimeSpan(9, 30, 0))
        {
            return "Pre-Market (4:00 AM - 9:30 AM ET)";
        }

        // Regular Trading Hours: 9:30 AM - 4:00 PM ET
        if (timeOfDay >= new TimeSpan(9, 30, 0) && timeOfDay < new TimeSpan(16, 0, 0))
        {
            return "Regular Trading Hours (9:30 AM - 4:00 PM ET)";
        }

        // After-hours: 4:00 PM - 8:00 PM ET
        if (timeOfDay >= new TimeSpan(16, 0, 0) && timeOfDay < new TimeSpan(20, 0, 0))
        {
            return "After-Hours (4:00 PM - 8:00 PM ET)";
        }

        // Overnight/Closed
        return "Closed (Outside Trading Hours)";
    }

    /// <summary>
    /// Format countdown duration as "X days Xh Xm"
    /// </summary>
    private string FormatCountdown(TimeSpan duration)
    {
        if (duration.TotalDays >= 1)
        {
            return $"{(int)duration.TotalDays} days {duration.Hours}h {duration.Minutes}m";
        }
        else if (duration.TotalHours >= 1)
        {
            return $"{duration.Hours}h {duration.Minutes}m";
        }
        else
        {
            return $"{duration.Minutes}m {duration.Seconds}s";
        }
    }

    #endregion

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
        // 🚀 DEBUG: Log every check to see if this is even being called
        _logger.LogWarning("[LAB-DEBUG] ⏰ IsTrainingTime() called at {Time}", easternTime.ToString("yyyy-MM-dd HH:mm:ss"));

        // 🚀 FORCE_LAB_NOW: Bypass Sunday schedule for immediate testing
        var forceLab = Environment.GetEnvironmentVariable("FORCE_LAB_NOW") == "1";
        if (forceLab)
        {
            _logger.LogInformation("[LAB-DEBUG] FORCE_LAB_NOW=1 detected - forcing training to START NOW");
            return true; // Always return true to run immediately
        }

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

    /// <summary>
    /// Check for incomplete training runs on startup
    /// </summary>
    private async Task CheckForIncompleteTrainingAsync()
    {
        try
        {
            if (File.Exists(_checkpointFilePath))
            {
                var content = await File.ReadAllTextAsync(_checkpointFilePath).ConfigureAwait(false);
                using var doc = System.Text.Json.JsonDocument.Parse(content);
                var root = doc.RootElement;

                var runId = root.GetProperty("RunId").GetString();
                var startTime = root.GetProperty("StartTime").GetDateTime();

                _logger.LogWarning("[LAB] Detected incomplete training run: {RunId} started {Time}",
                    runId, startTime);
                _logger.LogInformation("[LAB] Incomplete run will be discarded (no resume capability yet)");

                // Delete checkpoint file
                File.Delete(_checkpointFilePath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LAB] Failed to check for incomplete training: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Save checkpoint during training for graceful shutdown
    /// </summary>
    private async Task SaveCheckpointAsync(string runId, CancellationToken cancellationToken)
    {
        try
        {
            var checkpoint = new
            {
                RunId = runId,
                StartTime = _lastTrainingStart ?? DateTime.UtcNow,
                CheckpointTime = DateTime.UtcNow
            };

            var json = System.Text.Json.JsonSerializer.Serialize(checkpoint);
            await File.WriteAllTextAsync(_checkpointFilePath, json, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[LAB] Checkpoint saved for run: {RunId}", runId);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LAB] Failed to save checkpoint: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Handle graceful shutdown - save checkpoint if training in progress
    /// Phase 10: Enhanced to handle idle state
    /// </summary>
    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        if (_currentTrainingCts != null && !_currentTrainingCts.IsCancellationRequested)
        {
            _logger.LogWarning("[LAB] Training in progress - saving checkpoint before shutdown");
            await SaveCheckpointAsync("training_session", cancellationToken).ConfigureAwait(false);

            // Request cancellation and wait a bit for cleanup
            _currentTrainingCts.Cancel();
            await Task.Delay(TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        }
        else
        {
            // Idle state shutdown
            var easternTime = GetEasternTime();
            var nextTraining = GetNextTrainingWindow(easternTime);
            _logger.LogInformation("[LAB] Shutdown requested during idle state");
            _logger.LogInformation("[LAB] Lab Mode shutdown complete - next session: {NextTraining}",
                nextTraining.ToString("dddd, MMM dd yyyy, h:mm tt") + " ET");
        }

        await base.StopAsync(cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("[LAB] Graceful shutdown complete");
    }

    public override void Dispose()
    {
        _trainingLock?.Dispose();
        _currentTrainingCts?.Dispose();
        CleanupStaleLockFile();
        base.Dispose();
    }
}
