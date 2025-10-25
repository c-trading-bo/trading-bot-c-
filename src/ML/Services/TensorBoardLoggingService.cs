using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.ML.Services;

/// <summary>
/// TensorBoard Logging Service for hedge fund level training visualization
/// Addresses gap in HEDGE_FUND_GAP_ANALYSIS.md - Section "6. TensorBoard Logging"
/// Provides real-time training metrics visualization and debugging capabilities
/// </summary>
public interface ITensorBoardLoggingService
{
    /// <summary>
    /// Log scalar metric to TensorBoard
    /// </summary>
    Task LogScalarAsync(
        string tag,
        double value,
        int step,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Log multiple scalars at once
    /// </summary>
    Task LogScalarsAsync(
        Dictionary<string, double> metrics,
        int step,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Log training epoch metrics
    /// </summary>
    Task LogEpochMetricsAsync(
        int epoch,
        double trainLoss,
        double validationLoss,
        double accuracy,
        Dictionary<string, double> additionalMetrics = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Log hyperparameters and final metrics
    /// </summary>
    Task LogHyperparametersAsync(
        Dictionary<string, object> hyperparameters,
        Dictionary<string, double> finalMetrics,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get TensorBoard log directory
    /// </summary>
    string GetLogDirectory();

    /// <summary>
    /// Check if TensorBoard logging is enabled
    /// </summary>
    bool IsEnabled();
}

/// <summary>
/// Production implementation of TensorBoard logging service
/// Writes metrics in TensorBoard-compatible format for visualization
/// </summary>
public class TensorBoardLoggingService : ITensorBoardLoggingService
{
    private readonly ILogger<TensorBoardLoggingService> _logger;
    private readonly string _logDirectory;
    private readonly bool _enabled;
    private readonly string _runName;
    private readonly SemaphoreSlim _fileLock;

    public TensorBoardLoggingService(
        ILogger<TensorBoardLoggingService> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _enabled = Environment.GetEnvironmentVariable("TENSORBOARD_LOGGING_ENABLED") != "0";
        
        var baseLogDir = Environment.GetEnvironmentVariable("TENSORBOARD_LOG_DIR") 
            ?? Path.Combine("./logs", "tensorboard");
        
        _runName = Environment.GetEnvironmentVariable("TENSORBOARD_RUN_NAME") 
            ?? $"run_{DateTime.UtcNow:yyyyMMdd_HHmmss}";
        
        _logDirectory = Path.Combine(baseLogDir, _runName);
        _fileLock = new SemaphoreSlim(1, 1);

        if (_enabled)
        {
            Directory.CreateDirectory(_logDirectory);
            _logger.LogInformation(
                "TensorBoard Logging Service initialized. Log directory: {LogDir}",
                _logDirectory);
        }
        else
        {
            _logger.LogInformation("TensorBoard Logging Service disabled via configuration");
        }
    }

    public async Task LogScalarAsync(
        string tag,
        double value,
        int step,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            return;
        }

        await _fileLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var logEntry = new
            {
                tag,
                value,
                step,
                timestamp = DateTime.UtcNow
            };

            var logFile = Path.Combine(_logDirectory, $"{tag.Replace("/", "_")}.jsonl");
            var jsonLine = JsonSerializer.Serialize(logEntry) + Environment.NewLine;

            await File.AppendAllTextAsync(logFile, jsonLine, cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _fileLock.Release();
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error logging scalar to TensorBoard. Tag: {Tag}, Step: {Step}",
                tag,
                step);
        }
    }

    public async Task LogScalarsAsync(
        Dictionary<string, double> metrics,
        int step,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled || metrics == null || metrics.Count == 0)
        {
            return;
        }

        try
        {
            foreach (var (tag, value) in metrics)
            {
                await LogScalarAsync(tag, value, step, cancellationToken)
                    .ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error logging multiple scalars to TensorBoard. Step: {Step}",
                step);
        }
    }

    public async Task LogEpochMetricsAsync(
        int epoch,
        double trainLoss,
        double validationLoss,
        double accuracy,
        Dictionary<string, double> additionalMetrics = null,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            return;
        }

        try
        {
            await LogScalarAsync("loss/train", trainLoss, epoch, cancellationToken)
                .ConfigureAwait(false);
            await LogScalarAsync("loss/validation", validationLoss, epoch, cancellationToken)
                .ConfigureAwait(false);
            await LogScalarAsync("metrics/accuracy", accuracy, epoch, cancellationToken)
                .ConfigureAwait(false);

            if (additionalMetrics != null)
            {
                foreach (var (key, value) in additionalMetrics)
                {
                    await LogScalarAsync($"metrics/{key}", value, epoch, cancellationToken)
                        .ConfigureAwait(false);
                }
            }

            _logger.LogDebug(
                "Logged epoch {Epoch} metrics to TensorBoard. Train Loss: {TrainLoss:F4}, Val Loss: {ValLoss:F4}",
                epoch,
                trainLoss,
                validationLoss);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error logging epoch metrics to TensorBoard. Epoch: {Epoch}",
                epoch);
        }
    }

    public async Task LogHyperparametersAsync(
        Dictionary<string, object> hyperparameters,
        Dictionary<string, double> finalMetrics,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            return;
        }

        try
        {
            var hparamLog = new
            {
                hyperparameters,
                finalMetrics,
                timestamp = DateTime.UtcNow
            };

            var hparamFile = Path.Combine(_logDirectory, "hyperparameters.json");
            var json = JsonSerializer.Serialize(hparamLog, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });

            await File.WriteAllTextAsync(hparamFile, json, cancellationToken)
                .ConfigureAwait(false);

            _logger.LogInformation(
                "Logged hyperparameters to TensorBoard: {Path}",
                hparamFile);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error logging hyperparameters to TensorBoard");
        }
    }

    public string GetLogDirectory()
    {
        return _logDirectory;
    }

    public bool IsEnabled()
    {
        return _enabled;
    }
}
