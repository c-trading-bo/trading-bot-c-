using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Training Failure Handler - Phase 13: Failure Handling & Recovery
/// Manages component failures, retries, and abort logic during training sessions
/// Ensures training resilience and prevents one failure from killing entire session
/// </summary>
internal sealed class TrainingFailureHandler
{
    private readonly ILogger<TrainingFailureHandler> _logger;
    private readonly TrainingAlertService _alertService;
    private readonly TrainingCheckpointService _checkpointService;

    public TrainingFailureHandler(
        ILogger<TrainingFailureHandler> logger,
        TrainingAlertService alertService,
        TrainingCheckpointService checkpointService)
    {
        _logger = logger;
        _alertService = alertService;
        _checkpointService = checkpointService;
    }

    /// <summary>
    /// Classify failure type to determine retry strategy
    /// Phase 13.4: Failure Classification
    /// </summary>
    public string ClassifyFailure(Exception exception)
    {
        var exceptionType = exception.GetType().Name;
        var message = exception.Message.ToLowerInvariant();

        // Transient failures (retry likely to succeed)
        if (exception is OutOfMemoryException ||
            message.Contains("out of memory") ||
            message.Contains("gc pressure"))
        {
            return "Transient_Memory";
        }

        if (message.Contains("timeout") ||
            message.Contains("network") ||
            message.Contains("connection"))
        {
            return "Transient_Network";
        }

        if (message.Contains("file lock") ||
            message.Contains("locked") ||
            message.Contains("in use"))
        {
            return "Transient_FileLock";
        }

        if (message.Contains("database") ||
            message.Contains("pool exhausted"))
        {
            return "Transient_Database";
        }

        // Permanent failures (retry unlikely to help)
        if (exception is NullReferenceException ||
            exception is ArgumentNullException ||
            exception is IndexOutOfRangeException)
        {
            return "Permanent_CodeBug";
        }

        if (message.Contains("missing data") ||
            message.Contains("no data") ||
            message.Contains("insufficient data"))
        {
            return "Permanent_MissingData";
        }

        if (message.Contains("serialization") ||
            message.Contains("cannot convert") ||
            message.Contains("double[,]"))
        {
            return "Permanent_Serialization";
        }

        if (message.Contains("onnx") ||
            message.Contains("model architecture") ||
            message.Contains("incompatible"))
        {
            return "Permanent_ModelArchitecture";
        }

        // Resource failures (system constrained)
        if (message.Contains("disk space") ||
            message.Contains("no space left"))
        {
            return "Resource_DiskSpace";
        }

        if (message.Contains("memory limit") ||
            message.Contains("exceeded memory"))
        {
            return "Resource_Memory";
        }

        if (message.Contains("timeout exceeded") ||
            message.Contains("training timeout"))
        {
            return "Resource_Timeout";
        }

        return "Unknown";
    }

    /// <summary>
    /// Retry component training with backoff strategy
    /// Phase 13.5: Retry Logic with Backoff
    /// </summary>
    public async Task<ComponentTrainingResult> RetryComponentTrainingAsync(
        string componentId,
        Func<CancellationToken, Task> trainingFunc,
        int maxAttempts,
        CancellationToken cancellationToken)
    {
        var result = new ComponentTrainingResult
        {
            ComponentId = componentId,
            Success = false
        };

        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                _logger.LogInformation("[RETRY] Component {Component}: Attempt {Attempt}/{Max}",
                    componentId, attempt, maxAttempts);

                var startTime = DateTime.UtcNow;
                await trainingFunc(cancellationToken).ConfigureAwait(false);
                var duration = DateTime.UtcNow - startTime;

                result.Success = true;
                result.Duration = duration;
                result.RetryCount = attempt - 1; // First attempt doesn't count as retry
                
                if (attempt > 1)
                {
                    _logger.LogInformation("[RETRY] Component {Component}: SUCCESS after {Retries} retries",
                        componentId, attempt - 1);
                }

                return result;
            }
            catch (Exception ex)
            {
                var failureType = ClassifyFailure(ex);
                result.ErrorMessage = ex.Message;
                result.FailureType = failureType;
                result.RetryCount = attempt;

                _logger.LogWarning("[RETRY] Component {Component}: Attempt {Attempt} FAILED - {Type}: {Error}",
                    componentId, attempt, failureType, ex.Message);

                // Check if we should retry based on failure type
                if (failureType.StartsWith("Permanent_"))
                {
                    _logger.LogError("[RETRY] Permanent failure detected - no retry");
                    return result;
                }

                if (failureType.StartsWith("Resource_"))
                {
                    _logger.LogError("[RETRY] Resource failure detected - skipping component");
                    return result;
                }

                // Transient failure - apply backoff before retry
                if (attempt < maxAttempts)
                {
                    var backoffSeconds = attempt switch
                    {
                        1 => 0,    // Immediate retry
                        2 => 30,   // Wait 30 seconds, force GC
                        _ => 60    // Wait 60 seconds
                    };

                    if (backoffSeconds > 0)
                    {
                        _logger.LogInformation("[RETRY] Waiting {Seconds}s before retry...", backoffSeconds);
                        
                        if (attempt == 2)
                        {
                            // Attempt 2: Force full GC
                            _logger.LogDebug("[RETRY] Forcing full garbage collection");
                            GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
                        }

                        await Task.Delay(TimeSpan.FromSeconds(backoffSeconds), cancellationToken).ConfigureAwait(false);
                    }
                }
            }
        }

        _logger.LogError("[RETRY] Component {Component}: FAILED after {Attempts} attempts",
            componentId, maxAttempts);
        
        return result;
    }

    /// <summary>
    /// Retry component training with backoff strategy - Generic result version
    /// Captures and returns the training result from the trainer
    /// </summary>
    public async Task<ComponentTrainingResult<T>> RetryComponentTrainingAsync<T>(
        string componentId,
        Func<CancellationToken, Task<T>> trainingFunc,
        int maxAttempts,
        CancellationToken cancellationToken)
    {
        var result = new ComponentTrainingResult<T>
        {
            ComponentId = componentId,
            Success = false
        };

        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                _logger.LogInformation("[RETRY] Component {Component}: Attempt {Attempt}/{Max}",
                    componentId, attempt, maxAttempts);

                var startTime = DateTime.UtcNow;
                var trainerResult = await trainingFunc(cancellationToken).ConfigureAwait(false);
                var duration = DateTime.UtcNow - startTime;

                result.TrainerResult = trainerResult;
                result.Success = true;
                result.Duration = duration;
                result.RetryCount = attempt - 1;
                
                if (attempt > 1)
                {
                    _logger.LogInformation("[RETRY] Component {Component}: SUCCESS after {Retries} retries",
                        componentId, attempt - 1);
                }

                return result;
            }
            catch (Exception ex)
            {
                var failureType = ClassifyFailure(ex);
                result.ErrorMessage = ex.Message;
                result.FailureType = failureType;
                result.RetryCount = attempt;

                _logger.LogWarning("[RETRY] Component {Component}: Attempt {Attempt} FAILED - {Type}: {Error}",
                    componentId, attempt, failureType, ex.Message);

                // Check if we should retry based on failure type
                if (failureType.StartsWith("Permanent_"))
                {
                    _logger.LogError("[RETRY] Permanent failure detected - no retry");
                    return result;
                }

                if (failureType.StartsWith("Resource_"))
                {
                    _logger.LogError("[RETRY] Resource failure detected - skipping component");
                    return result;
                }

                // Transient failure - apply backoff before retry
                if (attempt < maxAttempts)
                {
                    var backoffSeconds = attempt switch
                    {
                        1 => 0,    // Immediate retry
                        2 => 30,   // Wait 30 seconds, force GC
                        _ => 60    // Wait 60 seconds
                    };

                    if (backoffSeconds > 0)
                    {
                        _logger.LogInformation("[RETRY] Waiting {Seconds}s before retry...", backoffSeconds);
                        
                        if (attempt == 2)
                        {
                            // Attempt 2: Force full GC
                            _logger.LogDebug("[RETRY] Forcing full garbage collection");
                            GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
                        }

                        await Task.Delay(TimeSpan.FromSeconds(backoffSeconds), cancellationToken).ConfigureAwait(false);
                    }
                }
            }
        }

        _logger.LogError("[RETRY] Component {Component}: FAILED after {Attempts} attempts",
            componentId, maxAttempts);
        
        return result;
    }

    /// <summary>
    /// Check if training session should be aborted
    /// Phase 13.6: Critical Failure Abort Logic
    /// </summary>
    public bool ShouldAbortSession(TrainingSessionState state, double windowClosingInMinutes)
    {
        var failureRate = state.TotalComponents > 0
            ? state.ComponentsFailed.Count / (double)state.TotalComponents
            : 0;

        // Abort if more than 25% of components failed
        if (failureRate > 0.25)
        {
            _logger.LogError("[ABORT] Failure rate {Rate:P0} exceeds 25% threshold - aborting session",
                failureRate);
            return true;
        }

        // Abort if all Heavy components failed (something fundamentally broken)
        var heavyComponentsFailed = state.ComponentsFailed
            .Where(f => f.ComponentId.Contains("Heavy"))
            .Count();
        if (heavyComponentsFailed >= 67) // All 67 heavy components
        {
            _logger.LogError("[ABORT] All Heavy components failed - fundamental issue detected");
            return true;
        }

        // Abort if training window closing soon and less than 50% complete
        var percentComplete = state.TotalComponents > 0
            ? state.ComponentsCompleted.Count / (double)state.TotalComponents
            : 0;
        
        if (windowClosingInMinutes < 30 && percentComplete < 0.5)
        {
            _logger.LogError("[ABORT] Training window closing in {Minutes}min and only {Percent:P0} complete",
                windowClosingInMinutes, percentComplete);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Handle session abort - save checkpoint and cleanup
    /// Phase 13.6: Critical Failure Abort Logic
    /// </summary>
    public async Task AbortSessionAsync(
        TrainingSessionState state,
        string reason,
        CancellationToken cancellationToken)
    {
        _logger.LogError("[ABORT] Training session aborted - Reason: {Reason}", reason);

        // Save final checkpoint
        state.CheckpointTime = DateTime.UtcNow;
        await _checkpointService.SaveCheckpointAsync(state, cancellationToken).ConfigureAwait(false);

        // Send alert notification
        await _alertService.AlertTrainingFailureAsync(
            state.SessionId,
            $"Session aborted: {reason}",
            state.ComponentsFailed.Select(f => f.ComponentId).ToList(),
            cancellationToken).ConfigureAwait(false);

        // Log summary
        _logger.LogInformation("[ABORT] Session summary:");
        _logger.LogInformation("[ABORT] - Components completed: {Completed}", state.ComponentsCompleted.Count);
        _logger.LogInformation("[ABORT] - Components failed: {Failed}", state.ComponentsFailed.Count);
        _logger.LogInformation("[ABORT] - Components pending: {Pending}", state.ComponentsPending.Count);
        _logger.LogInformation("[ABORT] - Training time: {Minutes:F1} minutes", state.TotalTrainingTimeMinutes);
    }

    /// <summary>
    /// Check component timeout
    /// Phase 13.9: Component Timeout Watchdog
    /// </summary>
    public int GetComponentTimeoutMinutes(string phase)
    {
        return phase switch
        {
            "Heavy" => 15,
            "Medium" => 5,
            "Light" => 2,
            _ => 10
        };
    }

    /// <summary>
    /// Calculate time remaining until training window closes
    /// Phase 13.8: Training Timeout Watchdog
    /// </summary>
    public double CalculateWindowClosingInMinutes(DateTime startTime)
    {
        // Training window: Sunday 12:00 PM - 5:45 PM ET (5 hours 45 minutes)
        var maxDuration = TimeSpan.FromHours(5).Add(TimeSpan.FromMinutes(45));
        var elapsed = DateTime.UtcNow - startTime;
        var remaining = maxDuration - elapsed;
        
        return remaining.TotalMinutes;
    }

    /// <summary>
    /// Estimate remaining components completion time
    /// </summary>
    public double EstimateRemainingTimeMinutes(
        TrainingSessionState state,
        int avgComponentMinutes)
    {
        var remainingComponents = state.TotalComponents - state.ComponentsCompleted.Count - state.ComponentsFailed.Count;
        return remainingComponents * avgComponentMinutes;
    }
}

/// <summary>
/// Component training result returned from retry handler
/// </summary>
public class ComponentTrainingResult
{
    public string ComponentId { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public string? FailureType { get; set; }
    public int RetryCount { get; set; }
    public TimeSpan Duration { get; set; }
}

/// <summary>
/// Generic component training result that captures trainer-specific result
/// </summary>
/// <typeparam name="T">Type of training result from the trainer</typeparam>
public class ComponentTrainingResult<T>
{
    public string ComponentId { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public string? FailureType { get; set; }
    public int RetryCount { get; set; }
    public TimeSpan Duration { get; set; }
    public T? TrainerResult { get; set; }
}
