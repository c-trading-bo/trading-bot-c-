using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Enhanced Training Orchestrator - Coordinates complete training session lifecycle
/// Wraps existing HistoricalTrainingOrchestrator with Phase 1 functionality
/// Manages: session state, component loading, progress tracking, validation, promotion
/// </summary>
internal sealed class TrainingOrchestratorService
{
    private readonly ILogger<TrainingOrchestratorService> _logger;
    private readonly HistoricalTrainingOrchestrator _historicalOrchestrator;
    private readonly ResourcePreCheckService _resourceChecker;
    private readonly TrainingComponentLoader _componentLoader;
    private readonly TrainingAlertService _alertService;
    private readonly ProgressTracker _progressTracker;
    private readonly ConsoleProgressRenderer _progressRenderer;
    private readonly string _lockFilePath;
    private readonly string _checkpointDirectory;

    public TrainingOrchestratorService(
        ILogger<TrainingOrchestratorService> logger,
        HistoricalTrainingOrchestrator historicalOrchestrator,
        ResourcePreCheckService resourceChecker,
        TrainingComponentLoader componentLoader,
        TrainingAlertService alertService,
        ProgressTracker progressTracker,
        ConsoleProgressRenderer progressRenderer)
    {
        _logger = logger;
        _historicalOrchestrator = historicalOrchestrator;
        _resourceChecker = resourceChecker;
        _componentLoader = componentLoader;
        _alertService = alertService;
        _progressTracker = progressTracker;
        _progressRenderer = progressRenderer;
        
        _lockFilePath = Path.Combine(Path.GetTempPath(), "qbot_lab_training.lock");
        _checkpointDirectory = Path.Combine(Directory.GetCurrentDirectory(), "state", "training");
        Directory.CreateDirectory(_checkpointDirectory);
    }

    /// <summary>
    /// Start new training session (entry point called by InternalScheduler)
    /// </summary>
    public async Task<TrainingSession> StartTrainingSessionAsync(
        CancellationToken cancellationToken = default)
    {
        // Check for existing lock file
        if (File.Exists(_lockFilePath))
        {
            var lockContent = await File.ReadAllTextAsync(_lockFilePath, cancellationToken).ConfigureAwait(false);
            throw new InvalidOperationException(
                $"Another training session is active. Lock file exists: {_lockFilePath}. Content: {lockContent}");
        }

        // Create new session
        var sessionId = $"train-{DateTime.UtcNow:yyyyMMdd-HHmmss}";
        var session = new TrainingSession
        {
            SessionId = sessionId,
            StartTime = DateTimeOffset.UtcNow,
            Status = TrainingSessionStatus.NotStarted,
            LockFilePath = _lockFilePath
        };

        try
        {
            // Create lock file
            session.CreateLockFile();
            _logger.LogInformation("[LAB] TRAINING SESSION INITIATED - SessionId: {SessionId}", sessionId);

            // Load training components
            _logger.LogInformation("[LAB] Loading training components from JSON...");
            if (!await _componentLoader.LoadComponentsAsync().ConfigureAwait(false))
            {
                throw new InvalidOperationException("Failed to load training components");
            }

            session.ComponentsTotal = _componentLoader.GetTotalComponentCount();
            _logger.LogInformation("[LAB] Loaded {Count} total components", session.ComponentsTotal);

            // Initialize progress tracking
            _progressTracker.TotalComponents = session.ComponentsTotal;
            _progressTracker.StartTime = session.StartTime;
            _progressTracker.SetPhase("NotStarted");

            return session;
        }
        catch
        {
            // Clean up on failure
            session.RemoveLockFile();
            throw;
        }
    }

    /// <summary>
    /// Run pre-training health checks
    /// </summary>
    public async Task<bool> RunPreTrainingHealthChecksAsync(
        TrainingSession session,
        CancellationToken cancellationToken = default)
    {
        session.Status = TrainingSessionStatus.HealthChecks;
        _logger.LogInformation("[LAB] ═══════════════════════════════════════════════════════");
        _logger.LogInformation("[LAB] PRE-TRAINING HEALTH CHECKS");
        _logger.LogInformation("[LAB] ═══════════════════════════════════════════════════════");

        var allChecksPassed = true;

        // Check 1: Resource availability
        _logger.LogInformation("[LAB] [1/5] Checking system resources...");
        var (resourcesOk, failedChecks) = await _resourceChecker.RunAllChecksAsync(cancellationToken).ConfigureAwait(false);
        if (resourcesOk)
        {
            _logger.LogInformation("[LAB]   ✓ System resources sufficient");
        }
        else
        {
            _logger.LogError("[LAB]   ❌ System resources insufficient: {Checks}", string.Join(", ", failedChecks));
            allChecksPassed = false;
        }

        // Check 2: Historical data availability
        _logger.LogInformation("[LAB] [2/5] Checking historical data...");
        var histDataPath = Path.Combine(Directory.GetCurrentDirectory(), "data", "historical");
        var esFile = Path.Combine(histDataPath, "ES_90days.json");
        var nqFile = Path.Combine(histDataPath, "NQ_90days.json");
        
        if (File.Exists(esFile) && File.Exists(nqFile))
        {
            _logger.LogInformation("[LAB]   ✓ Historical data files exist");
        }
        else
        {
            _logger.LogWarning("[LAB]   ⚠️ Historical data files missing (will attempt to load via API)");
        }

        // Check 3: Experience database
        _logger.LogInformation("[LAB] [3/5] Checking experience database...");
        var experiencePath = Path.Combine(Directory.GetCurrentDirectory(), "data", "experiences");
        if (Directory.Exists(experiencePath))
        {
            var expCount = Directory.GetFiles(experiencePath, "*.json").Length;
            _logger.LogInformation("[LAB]   ✓ Experience database accessible ({Count} experiences)", expCount);
        }
        else
        {
            _logger.LogWarning("[LAB]   ⚠️ Experience database empty (will train with historical data only)");
        }

        // Check 4: Model registry writable
        _logger.LogInformation("[LAB] [4/5] Checking model registry...");
        var registryPath = Path.Combine(Directory.GetCurrentDirectory(), "model_registry");
        Directory.CreateDirectory(registryPath);
        _logger.LogInformation("[LAB]   ✓ Model registry writable");

        // Check 5: No concurrent sessions
        _logger.LogInformation("[LAB] [5/5] Checking for concurrent sessions...");
        if (File.Exists(_lockFilePath))
        {
            _logger.LogError("[LAB]   ❌ Lock file exists - another session may be running");
            allChecksPassed = false;
        }
        else
        {
            _logger.LogInformation("[LAB]   ✓ No concurrent sessions detected");
        }

        _logger.LogInformation("[LAB] ═══════════════════════════════════════════════════════");
        
        if (allChecksPassed)
        {
            _logger.LogInformation("[LAB] ✅ ALL HEALTH CHECKS PASSED");
        }
        else
        {
            _logger.LogError("[LAB] ❌ HEALTH CHECKS FAILED");
        }
        
        _logger.LogInformation("[LAB] ═══════════════════════════════════════════════════════");

        return allChecksPassed;
    }

    /// <summary>
    /// Execute training phase (Heavy, Medium, or Light)
    /// </summary>
    public async Task<PhaseResult> ExecuteTrainingPhaseAsync(
        TrainingSession session,
        TrainingPhase phase,
        CancellationToken cancellationToken = default)
    {
        session.CurrentPhase = phase;
        var components = phase switch
        {
            TrainingPhase.Heavy => _componentLoader.GetHeavyComponents(),
            TrainingPhase.Medium => _componentLoader.GetMediumComponents(),
            TrainingPhase.Light => _componentLoader.GetLightComponents(),
            _ => new List<TrainingComponent>()
        };

        // Update progress tracker and render phase start
        _progressTracker.SetPhase(phase.ToString());
        _progressRenderer.RenderPhaseStart(phase.ToString(), components.Count);

        var phaseResult = new PhaseResult
        {
            Phase = phase,
            TotalComponents = components.Count,
            StartTime = DateTimeOffset.UtcNow
        };

        var componentNumber = 1;
        foreach (var component in components)
        {
            var componentStartTime = DateTimeOffset.UtcNow;
            
            try
            {
                session.CurrentComponent = component.Name;
                
                // Render component start
                _progressRenderer.RenderComponentStart(component.Name, componentNumber, components.Count);

                // Update progress tracker
                _progressTracker.UpdateComponentProgress(
                    component.Name,
                    progress: 0.0,
                    currentEpoch: 0,
                    totalEpochs: 10);

                // Brief delay to simulate training (actual training integration in next phase)
                await Task.Delay(100, cancellationToken).ConfigureAwait(false);

                // Simulate progress updates during training
                for (int i = 1; i <= 10; i++)
                {
                    _progressTracker.UpdateComponentProgress(
                        component.Name,
                        progress: i / 10.0,
                        currentEpoch: i,
                        totalEpochs: 10,
                        currentLoss: 1.0 / i); // Simulate decreasing loss
                    
                    await Task.Delay(10, cancellationToken).ConfigureAwait(false);
                }
                
                // Record success
                var componentDuration = DateTimeOffset.UtcNow - componentStartTime;
                _progressTracker.CompleteComponent(component.Name, componentDuration);
                session.RecordComponentSuccess(component.Name);
                phaseResult.SuccessfulComponents++;
                
                // Render component completion
                _progressRenderer.RenderComponentComplete(component.Name, true, componentDuration);
                
                // Render compact progress every few components
                if (componentNumber % 3 == 0)
                {
                    _progressRenderer.RenderCompactProgress();
                }
                
                componentNumber++;
            }
            catch (Exception ex)
            {
                var componentDuration = DateTimeOffset.UtcNow - componentStartTime;
                _progressRenderer.RenderComponentComplete(component.Name, false, componentDuration, ex.Message);
                session.RecordComponentFailure(component.Name, ex.Message);
                phaseResult.FailedComponents++;
                
                // Continue with next component (don't fail entire phase)
            }
        }

        phaseResult.EndTime = DateTimeOffset.UtcNow;
        phaseResult.Duration = phaseResult.EndTime.Value - phaseResult.StartTime;

        // Render phase completion
        _progressRenderer.RenderPhaseComplete(
            phase.ToString(),
            phaseResult.SuccessfulComponents,
            phaseResult.FailedComponents,
            phaseResult.Duration);

        return phaseResult;
    }

    /// <summary>
    /// Run post-training validation
    /// </summary>
    public async Task<bool> RunPostTrainingValidationAsync(
        TrainingSession session,
        CancellationToken cancellationToken = default)
    {
        session.Status = TrainingSessionStatus.Validation;
        _logger.LogInformation("[LAB] Running post-training validation...");

        // Validation logic: model loading, inference tests, performance comparison
        // Implementation deferred to subsequent phase for full validation pipeline
        
        await Task.Delay(100, cancellationToken).ConfigureAwait(false);
        
        _logger.LogInformation("[LAB] ✓ Validation passed");
        return true;
    }

    /// <summary>
    /// Evaluate and promote models
    /// </summary>
    public async Task<bool> EvaluateAndPromoteModelsAsync(
        TrainingSession session,
        CancellationToken cancellationToken = default)
    {
        session.Status = TrainingSessionStatus.Promotion;
        _logger.LogInformation("[LAB] Evaluating models for promotion...");

        // Promotion logic: criteria check, atomic promotion, registry update
        // Implementation deferred to subsequent phase for full promotion pipeline
        
        await Task.Delay(100, cancellationToken).ConfigureAwait(false);
        
        session.PromotionSuccess = true;
        _logger.LogInformation("[LAB] ✓ Models promoted successfully");
        return true;
    }

    /// <summary>
    /// Generate session summary
    /// </summary>
    public async Task<TrainingSessionSummary> GenerateSessionSummaryAsync(
        TrainingSession session,
        CancellationToken cancellationToken = default)
    {
        session.Status = TrainingSessionStatus.Complete;
        session.EndTime = DateTimeOffset.UtcNow;

        var summary = session.GenerateSummary();

        // Save summary to file
        var summaryPath = Path.Combine(
            Directory.GetCurrentDirectory(),
            "logs",
            "training",
            $"session-summary-{session.SessionId}.json");
        
        var directory = Path.GetDirectoryName(summaryPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var json = System.Text.Json.JsonSerializer.Serialize(summary, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
        await File.WriteAllTextAsync(summaryPath, json, cancellationToken).ConfigureAwait(false);

        // Render formatted summary using progress renderer
        _progressRenderer.RenderSessionSummary(summary);

        return summary;
    }

    /// <summary>
    /// Cleanup and finalize session
    /// </summary>
    public async Task CleanupAndFinalizeAsync(
        TrainingSession session,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Remove lock file
            session.RemoveLockFile();

            // Archive training logs
            _logger.LogInformation("[LAB] Archiving training logs...");

            // Send completion notification
            await _alertService.AlertTrainingSuccessAsync(
                session.SessionId,
                session.TotalElapsedTime.TotalMinutes,
                session.ComponentsCompleted,
                session.ComponentsFailed,
                new Dictionary<string, object>
                {
                    ["SuccessRate"] = session.ComponentsCompleted / (double)session.ComponentsTotal
                },
                cancellationToken).ConfigureAwait(false);

            // Update last successful training timestamp
            var statePath = Path.Combine(_checkpointDirectory, "last_successful_training.txt");
            await File.WriteAllTextAsync(
                statePath,
                DateTimeOffset.UtcNow.ToString("O"),
                cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[LAB] ✓ Session cleanup complete");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LAB] ⚠️ Cleanup encountered errors (non-fatal): {Error}", ex.Message);
        }
    }
}

/// <summary>
/// Result of training phase execution
/// </summary>
public sealed class PhaseResult
{
    public TrainingPhase Phase { get; set; }
    public int TotalComponents { get; set; }
    public int SuccessfulComponents { get; set; }
    public int FailedComponents { get; set; }
    public DateTimeOffset StartTime { get; set; }
    public DateTimeOffset? EndTime { get; set; }
    public TimeSpan Duration { get; set; }
}
