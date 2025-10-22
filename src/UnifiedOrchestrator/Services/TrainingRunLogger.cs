using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Training Run Logger - Logs every epoch of every model to JSONL file
/// Creates one line per epoch with full training metrics
/// Step 4 from Integration Plan
/// </summary>
internal sealed class TrainingRunLogger
{
    private readonly ILogger<TrainingRunLogger> _logger;
    private readonly string _logsDirectory;
    private string? _currentRunId;
    private StreamWriter? _currentLogStream;
    private readonly SemaphoreSlim _writeLock = new(1, 1);

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false // One line per epoch
    };

    public TrainingRunLogger(ILogger<TrainingRunLogger> logger)
    {
        _logger = logger;
        _logsDirectory = Path.Combine(Directory.GetCurrentDirectory(), "state", "training_logs");
        Directory.CreateDirectory(_logsDirectory);
    }

    /// <summary>
    /// Start a new training run - creates new JSONL file
    /// </summary>
    public async Task StartRunAsync(string runId, CancellationToken cancellationToken = default)
    {
        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Close any existing stream
            if (_currentLogStream != null)
            {
                await _currentLogStream.DisposeAsync().ConfigureAwait(false);
            }

            _currentRunId = runId;
            var logFilePath = Path.Combine(_logsDirectory, $"{runId}_epochs.jsonl");

            _currentLogStream = new StreamWriter(logFilePath, append: false);
            _logger.LogInformation("[EPOCH-LOGGER] Started epoch logging for run {RunId} -> {Path}",
                runId, logFilePath);

            // Write header event
            await LogEventAsync(new
            {
                type = "RUN_START",
                runId,
                timestamp = DateTime.UtcNow,
                message = "Training run started"
            }, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    /// <summary>
    /// Log a single training epoch
    /// </summary>
    public async Task LogEpochAsync(
        string modelName,
        int epoch,
        double trainLoss,
        double? valLoss = null,
        double? valWinRate = null,
        int? tradeCount = null,
        Dictionary<string, object>? additionalMetrics = null,
        CancellationToken cancellationToken = default)
    {
        if (_currentLogStream == null)
        {
            _logger.LogWarning("[EPOCH-LOGGER] No active log stream - call StartRunAsync first");
            return;
        }

        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var epochData = new Dictionary<string, object>
            {
                ["type"] = "TRAIN_EPOCH",
                ["runId"] = _currentRunId ?? "unknown",
                ["model"] = modelName,
                ["epoch"] = epoch,
                ["trainLoss"] = trainLoss,
                ["timestamp"] = DateTime.UtcNow
            };

            if (valLoss.HasValue)
                epochData["valLoss"] = valLoss.Value;

            if (valWinRate.HasValue)
                epochData["valWinRate"] = valWinRate.Value;

            if (tradeCount.HasValue)
                epochData["tradeCount"] = tradeCount.Value;

            if (additionalMetrics != null)
            {
                foreach (var kvp in additionalMetrics)
                {
                    epochData[kvp.Key] = kvp.Value;
                }
            }

            await LogEventAsync(epochData, cancellationToken).ConfigureAwait(false);

            // Log to console every 10 epochs
            if (epoch % 10 == 0)
            {
                _logger.LogInformation("[EPOCH-LOGGER] {Model} Epoch {Epoch}: TrainLoss={TrainLoss:F4}, ValLoss={ValLoss:F4}, WinRate={WinRate:F2}%",
                    modelName, epoch, trainLoss, valLoss ?? 0, (valWinRate ?? 0) * 100);
            }
        }
        finally
        {
            _writeLock.Release();
        }
    }

    /// <summary>
    /// Log model training completion
    /// </summary>
    public async Task LogModelCompletionAsync(
        string modelName,
        int totalEpochs,
        double finalTrainLoss,
        double finalValLoss,
        double finalWinRate,
        TimeSpan duration,
        CancellationToken cancellationToken = default)
    {
        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await LogEventAsync(new
            {
                type = "MODEL_COMPLETE",
                runId = _currentRunId ?? "unknown",
                model = modelName,
                totalEpochs,
                finalTrainLoss,
                finalValLoss,
                finalWinRate,
                durationSeconds = duration.TotalSeconds,
                timestamp = DateTime.UtcNow
            }, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[EPOCH-LOGGER] ✅ {Model} completed: {Epochs} epochs, WinRate={WinRate:F2}%, Duration={Duration:F1}s",
                modelName, totalEpochs, finalWinRate * 100, duration.TotalSeconds);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    /// <summary>
    /// Log training run completion
    /// </summary>
    public async Task CompleteRunAsync(
        bool success,
        string? errorMessage = null,
        Dictionary<string, object>? summary = null,
        CancellationToken cancellationToken = default)
    {
        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var completionData = new Dictionary<string, object>
            {
                ["type"] = "RUN_COMPLETE",
                ["runId"] = _currentRunId ?? "unknown",
                ["success"] = success,
                ["timestamp"] = DateTime.UtcNow
            };

            if (errorMessage != null)
                completionData["errorMessage"] = errorMessage;

            if (summary != null)
            {
                foreach (var kvp in summary)
                {
                    completionData[kvp.Key] = kvp.Value;
                }
            }

            await LogEventAsync(completionData, cancellationToken).ConfigureAwait(false);

            // Close the stream
            if (_currentLogStream != null)
            {
                await _currentLogStream.FlushAsync().ConfigureAwait(false);
                await _currentLogStream.DisposeAsync().ConfigureAwait(false);
                _currentLogStream = null;
            }

            var status = success ? "✅ SUCCESS" : "❌ FAILED";
            _logger.LogInformation("[EPOCH-LOGGER] {Status} - Training run {RunId} completed",
                status, _currentRunId);
            _currentRunId = null;
        }
        finally
        {
            _writeLock.Release();
        }
    }

    /// <summary>
    /// Write a JSON event to the log file
    /// </summary>
    private async Task LogEventAsync(object eventData, CancellationToken cancellationToken)
    {
        if (_currentLogStream == null) return;

        var json = JsonSerializer.Serialize(eventData, JsonOptions);
        await _currentLogStream.WriteLineAsync(json).ConfigureAwait(false);
        await _currentLogStream.FlushAsync().ConfigureAwait(false);
    }

    public void Dispose()
    {
        _currentLogStream?.Dispose();
        _writeLock.Dispose();
    }
}
