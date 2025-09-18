using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.Globalization;

namespace BotCore
{
    /// <summary>
    /// DEPRECATED: Local automated RL trainer - replaced by 100% cloud-based learning.
    /// This component is no longer used as all training now happens in the cloud via GitHub Actions.
    /// The bot only needs to run for trading, not for learning.
    /// Use CloudDataUploader + CloudRlTrainerEnhanced for cloud-based learning instead.
    /// </summary>
    [Obsolete("Local training is deprecated. Use 100% cloud-based learning instead.", false)]
    public sealed class AutoRlTrainer : IDisposable
    {
        private readonly ILogger _log;
        private readonly Timer? _timer;
        private readonly string? _dataDir;
        private readonly string? _modelDir;
        private readonly string? _pythonScriptDir;
        private bool _disposed;
        private DateTime _lastTrainingAttempt = DateTime.MinValue;
        private int _consecutiveFailures = 0;

        private const int MaxConsecutiveFailures = 3;
        private const int MinTrainingDays = 7;

        public AutoRlTrainer(ILogger logger)
        {
            _log = logger;

            // Log deprecation warning
            _log.LogWarning("[AutoRlTrainer] DEPRECATED: Local training is disabled in favor of 100% cloud-based learning. Use CloudDataUploader + CloudRlTrainerEnhanced instead.");

            // Don't start the timer - this component is deprecated
            _log.LogInformation("[AutoRlTrainer] Local training disabled - all learning now happens in cloud every 30 minutes");

            // Initialize as null since this is deprecated
            _timer = null;
            _dataDir = null;
            _modelDir = null;
            _pythonScriptDir = null;
        }

        private async void CheckAndTrain(object? state)
        {
            try
            {
                // Prevent too frequent attempts if failures occur
                if (_consecutiveFailures >= MaxConsecutiveFailures)
                {
                    var backoffHours = Math.Pow(2, _consecutiveFailures - MaxConsecutiveFailures) * 6;
                    if (DateTime.UtcNow - _lastTrainingAttempt < TimeSpan.FromHours(backoffHours))
                    {
                        _log.LogWarning("[AutoRlTrainer] Backing off training attempts due to {Failures} consecutive failures", _consecutiveFailures);
                        return;
                    }
                }

                _lastTrainingAttempt = DateTime.UtcNow;

                if (!HasSufficientTrainingData())
                {
                    _log.LogDebug("[AutoRlTrainer] Insufficient training data - need {MinDays}+ days", MinTrainingDays);
                    return;
                }

                _log.LogInformation("[AutoRlTrainer] Starting automated training - sufficient data available");
                await RunTrainingPipelineAsync().ConfigureAwait(false);

                _consecutiveFailures = 0;
                _log.LogInformation("[AutoRlTrainer] ✅ Automated training complete! New model deployed");
            }
            catch (Exception ex)
            {
                _consecutiveFailures++;
                _log.LogError(ex, "[AutoRlTrainer] Training failed (attempt {Failures}/{Max})", _consecutiveFailures, MaxConsecutiveFailures);
            }
        }

        private bool HasSufficientTrainingData()
        {
            try
            {
                var dataDir = _dataDir ?? "data";
                if (!Directory.Exists(dataDir)) return false;

                var files = Directory.GetFiles(dataDir, "*.jsonl");
                if (!files.Any()) return false;

                // Check if we have data spanning at least MinTrainingDays
                var oldestFile = files.Min(f => File.GetCreationTime(f));
                var dataSpan = DateTime.Now - oldestFile;

                return dataSpan.TotalDays >= MinTrainingDays;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoRlTrainer] Error checking training data sufficiency");
                return false;
            }
        }

        private async Task RunTrainingPipelineAsync()
        {
            // Step 1: Export training data
            var csvFile = await ExportTrainingDataAsync().ConfigureAwait(false).ConfigureAwait(false);
            if (string.IsNullOrEmpty(csvFile))
            {
                throw new InvalidOperationException("Failed to export training data");
            }

            // Step 2: Train new model via Python
            var modelFile = await TrainModelAsync(csvFile).ConfigureAwait(false).ConfigureAwait(false);
            if (string.IsNullOrEmpty(modelFile))
            {
                throw new InvalidOperationException("Failed to train new model");
            }

            // Step 3: Deploy model hot
            await DeployModelAsync(modelFile).ConfigureAwait(false);
        }

        private async Task<string> ExportTrainingDataAsync()
        {
            try
            {
                var endDate = DateTime.UtcNow;
                var startDate = endDate.AddDays(-30); // Export last 30 days

                var fileName = $"training_data_{startDate:yyyyMMdd}_{endDate:yyyyMMdd}.csv";
                var csvPath = Path.Combine(_dataDir ?? "data", fileName);

                // Use MultiStrategyRlCollector to export data for all strategies
                var strategies = new[] {
                    MultiStrategyRlCollector.StrategyType.EmaCross,
                    MultiStrategyRlCollector.StrategyType.MeanReversion,
                    MultiStrategyRlCollector.StrategyType.Breakout,
                    MultiStrategyRlCollector.StrategyType.Momentum
                };

                bool hasData = false;
                foreach (var strategy in strategies)
                {
                    try
                    {
                        var strategyData = await MultiStrategyRlCollector.ExportStrategyData(_log, strategy, startDate).ConfigureAwait(false).ConfigureAwait(false);
                        if (!string.IsNullOrEmpty(strategyData))
                        {
                            hasData = true;
                        }
                    }
                    catch (Exception ex)
                    {
                        _log.LogWarning(ex, "[AutoRlTrainer] Failed to export data for strategy {Strategy}", strategy);
                    }
                }

                if (!hasData)
                {
                    _log.LogWarning("[AutoRlTrainer] No training data exported for any strategy");
                    return string.Empty;
                }

                var fileInfo = new FileInfo(csvPath);
                if (fileInfo.Exists)
                {
                    _log.LogInformation("[AutoRlTrainer] Exported training data: {File} ({Size:F1} KB)",
                        fileName, fileInfo.Length / 1024.0);
                    return csvPath;
                }

                return string.Empty;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoRlTrainer] Failed to export training data");
                return string.Empty;
            }
        }

        private async Task<string> TrainModelAsync(string csvFile)
        {
            try
            {
                var pythonScriptDir = _pythonScriptDir ?? "ml";
                var pythonScript = Path.Combine(pythonScriptDir, "train_cvar_ppo.py");
                if (!File.Exists(pythonScript))
                {
                    _log.LogError("[AutoRlTrainer] Python training script not found: {Script}", pythonScript);
                    return string.Empty;
                }

                var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
                var modelFileName = $"rl_sizer_{timestamp}.onnx";
                var modelDir = _modelDir ?? "models";
                var modelPath = Path.Combine(modelDir, modelFileName);

                var processInfo = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{pythonScript}\" --auto --input \"{csvFile}\" --output \"{modelPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(pythonScript)
                };

                _log.LogInformation("[AutoRlTrainer] Training: {Command} {Args}", processInfo.FileName, processInfo.Arguments);

                using var process = Process.Start(processInfo);
                if (process == null)
                {
                    throw new InvalidOperationException("Failed to start Python training process");
                }

                var output = await process.StandardOutput.ReadToEndAsync().ConfigureAwait(false).ConfigureAwait(false);
                var error = await process.StandardError.ReadToEndAsync().ConfigureAwait(false).ConfigureAwait(false);
                await process.WaitForExitAsync().ConfigureAwait(false);

                if (process.ExitCode == 0 && File.Exists(modelPath))
                {
                    _log.LogInformation("[AutoRlTrainer] Training successful: {Model}", modelFileName);
                    if (!string.IsNullOrEmpty(output))
                    {
                        _log.LogDebug("[AutoRlTrainer] Training output: {Output}", output);
                    }
                    return modelPath;
                }
                else
                {
                    _log.LogError("[AutoRlTrainer] Training failed (exit code {ExitCode}): {Error}", process.ExitCode, error);
                    return string.Empty;
                }
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoRlTrainer] Error running training script");
                return string.Empty;
            }
        }

        private Task DeployModelAsync(string modelPath)
        {
            try
            {
                var latestModelPath = Path.Combine(_modelDir ?? "models", "latest_rl_sizer.onnx");
                var tempModelPath = Path.Combine(_modelDir ?? "models", $"latest_rl_sizer_{Guid.NewGuid():N}.tmp");

                // Backup existing model
                string? backupPath = null;
                if (File.Exists(latestModelPath))
                {
                    var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
                    backupPath = Path.Combine(_modelDir ?? "models", $"backup_rl_sizer_{timestamp}.onnx");
                    // Note: We'll move the existing file to backup as part of atomic operation
                    _log.LogDebug("[AutoRlTrainer] Will backup existing model: {Backup}", Path.GetFileName(backupPath));
                }

                // Deploy new model atomically using copy-to-temp + File.Replace pattern
                try
                {
                    // Step 1: Copy new model to temporary location
                    File.Copy(modelPath, tempModelPath, false);
                    _log.LogDebug("[AutoRlTrainer] Copied model to temporary location: {TempPath}", Path.GetFileName(tempModelPath));

                    // Step 2: Atomic replace using File.Replace (moves existing to backup, replaces with new)
                    if (File.Exists(latestModelPath) && !string.IsNullOrEmpty(backupPath))
                    {
                        // File.Replace is atomic: replaces destination with source, moves destination to backup
                        File.Replace(tempModelPath, latestModelPath, backupPath);
                        _log.LogInformation("[AutoRlTrainer] Model deployed atomically with backup: {Model}, backup: {Backup}", 
                            Path.GetFileName(latestModelPath), Path.GetFileName(backupPath));
                    }
                    else
                    {
                        // No existing file, use atomic File.Move
                        File.Move(tempModelPath, latestModelPath);
                        _log.LogInformation("[AutoRlTrainer] Model deployed atomically: {Model}", Path.GetFileName(latestModelPath));
                    }
                }
                catch
                {
                    // Cleanup temp file on error
                    if (File.Exists(tempModelPath))
                    {
                        try { File.Delete(tempModelPath); } catch { }
                    }
                    throw;
                }

                // Cleanup old backups (keep last 5)
                CleanupOldBackups();

                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoRlTrainer] Failed to deploy model atomically");
                throw;
            }
        }

        private void CleanupOldBackups()
        {
            try
            {
                var modelDir = _modelDir ?? "models";
                if (!Directory.Exists(modelDir)) return;

                var backupFiles = Directory.GetFiles(modelDir, "backup_rl_sizer_*.onnx")
                    .Select(f => new FileInfo(f))
                    .OrderByDescending(f => f.CreationTime)
                    .Skip(5) // Keep last 5 backups
                    .ToList();

                foreach (var file in backupFiles)
                {
                    file.Delete();
                    _log.LogDebug("[AutoRlTrainer] Cleaned up old backup: {File}", file.Name);
                }

                if (backupFiles.Any())
                {
                    _log.LogInformation("[AutoRlTrainer] Cleaned up {Count} old backup(s)", backupFiles.Count);
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[AutoRlTrainer] Failed to cleanup old backups");
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _timer?.Dispose();
            _log.LogInformation("[AutoRlTrainer] Stopped");
        }
    }
}
