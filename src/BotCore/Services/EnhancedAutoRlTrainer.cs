using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace BotCore
{
    /// <summary>
    /// Enhanced Automated RL Trainer that monitors training data and triggers training when conditions are met.
    /// Integrates with the new emergency data generation and live collection systems.
    /// </summary>
    public sealed class EnhancedAutoRlTrainer : IDisposable
    {
        private readonly ILogger<EnhancedAutoRlTrainer> _logger;
        private readonly IEnhancedTrainingDataService _trainingDataService;
        private readonly Timer _timer;
        private readonly string _dataDir;
        private readonly string _modelDir;
        private readonly string _pythonPath;
        private bool _disposed;

        public EnhancedAutoRlTrainer(
            ILogger<EnhancedAutoRlTrainer> logger,
            IEnhancedTrainingDataService trainingDataService)
        {
            _logger = logger;
            _trainingDataService = trainingDataService;
            _dataDir = Path.Combine(AppContext.BaseDirectory, "data", "rl_training");
            _modelDir = Path.Combine(AppContext.BaseDirectory, "models", "rl");

            Directory.CreateDirectory(_dataDir);
            Directory.CreateDirectory(_modelDir);

            // Find Python executable
            _pythonPath = FindPythonExecutable();

            // Check for training data and train every 2 hours (faster than previous 6 hours)
            var checkInterval = TimeSpan.FromHours(2);
            _timer = new Timer(CheckAndTrain, null, TimeSpan.Zero, checkInterval);

            _logger.LogInformation("[EnhancedAutoRlTrainer] Started - checking every {Interval} for training opportunities", checkInterval);
        }

        private async void CheckAndTrain(object? state)
        {
            if (_disposed) return;

            try
            {
                // Check total training samples across all sources
                var totalSamples = await GetTotalTrainingSampleCountAsync().ConfigureAwait(false);

                if (totalSamples < 100)
                {
                    _logger.LogDebug("[EnhancedAutoRlTrainer] Insufficient data for training: {SampleCount} samples (need 100+)", totalSamples);
                    return;
                }

                _logger.LogInformation("[EnhancedAutoRlTrainer] Starting automated training - {SampleCount} samples available", totalSamples);

                // Export latest training data
                var exportPath = await _trainingDataService.ExportTrainingDataAsync(50).ConfigureAwait(false);
                if (exportPath == null)
                {
                    _logger.LogWarning("[EnhancedAutoRlTrainer] Failed to export training data");
                    return;
                }

                // Train the model
                var success = await TrainModelAsync(exportPath).ConfigureAwait(false);

                if (success)
                {
                    _logger.LogInformation("[EnhancedAutoRlTrainer] ✅ Automated training complete! New model deployed");

                    // Cleanup old data
                    await _trainingDataService.CleanupOldDataAsync(7).ConfigureAwait(false);
                }
                else
                {
                    _logger.LogError("[EnhancedAutoRlTrainer] ❌ Training failed");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[EnhancedAutoRlTrainer] Error during automated training check");
            }
        }

        private async Task<int> GetTotalTrainingSampleCountAsync()
        {
            var count;

            // Count from enhanced training data service
            count += await _trainingDataService.GetTrainingSampleCountAsync().ConfigureAwait(false);

            // Count from traditional MultiStrategyRlCollector
            count += MultiStrategyRlCollector.GetTotalTrainingSampleCount();

            // Count emergency generated data
            try
            {
                var emergencyFiles = Directory.GetFiles(_dataDir, "emergency_training_*.jsonl");
                foreach (var file in emergencyFiles)
                {
                    var lines = await File.ReadAllLinesAsync(file).ConfigureAwait(false);
                    count += lines.Length;
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "[EnhancedAutoRlTrainer] Error counting emergency data");
            }

            return count;
        }

        private async Task<bool> TrainModelAsync(string dataPath)
        {
            try
            {
                var mlDir = Path.Combine(AppContext.BaseDirectory, "ml");
                var trainScript = Path.Combine(mlDir, "rl", "train_cvar_ppo.py");

                if (!File.Exists(trainScript))
                {
                    _logger.LogError("[EnhancedAutoRlTrainer] Training script not found: {ScriptPath}", trainScript);
                    return false;
                }

                var outputModel = Path.Combine(_modelDir, $"rl_sizer_{DateTime.UtcNow:yyyyMMdd_HHmmss}.onnx");
                var latestModel = Path.Combine(_modelDir, "latest_rl_sizer.onnx");

                // Build Python command with enhanced parameters
                var arguments = $"\"{trainScript}\" --auto --data \"{dataPath}\" --output_model \"{outputModel}\"";

                _logger.LogInformation("[EnhancedAutoRlTrainer] Training: {Python} {Arguments}", _pythonPath, arguments);

                using var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = _pythonPath,
                        Arguments = arguments,
                        WorkingDirectory = mlDir,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                var outputLines = new List<string>();
                var errorLines = new List<string>();

                process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        outputLines.Add(e.Data);
                        _logger.LogDebug("[Training] {Output}", e.Data);
                    }
                };

                process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        errorLines.Add(e.Data);
                        _logger.LogWarning("[Training] {Error}", e.Data);
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                // Wait for training to complete (with timeout)
                var completed = await Task.Run(() => process.WaitForExit(600000)).ConfigureAwait(false); // 10 minutes timeout

                if (!completed)
                {
                    _logger.LogError("[EnhancedAutoRlTrainer] Training timed out after 10 minutes");
                    process.Kill();
                    return false;
                }

                if (process.ExitCode == 0 && File.Exists(outputModel))
                {
                    // Backup existing model if it exists
                    if (File.Exists(latestModel))
                    {
                        var backupModel = Path.Combine(_modelDir, $"backup_rl_sizer_{DateTime.UtcNow:yyyyMMdd_HHmmss}.onnx");
                        File.Move(latestModel, backupModel);
                        _logger.LogInformation("[EnhancedAutoRlTrainer] Backed up existing model: {BackupModel}", Path.GetFileName(backupModel));
                    }

                    // Deploy new model
                    File.Copy(outputModel, latestModel, true);
                    _logger.LogInformation("[EnhancedAutoRlTrainer] Model deployed: {ModelPath}", latestModel);

                    // Log training summary
                    LogTrainingSummary(outputLines);

                    return true;
                }
                else
                {
                    _logger.LogError("[EnhancedAutoRlTrainer] Training failed with exit code: {ExitCode}", process.ExitCode);
                    foreach (var error in errorLines)
                    {
                        _logger.LogError("[Training Error] {Error}", error);
                    }
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[EnhancedAutoRlTrainer] Exception during model training");
                return false;
            }
        }

        private void LogTrainingSummary(List<string> outputLines)
        {
            // Extract key metrics from training output
            foreach (var line in outputLines.TakeLast(20))
            {
                if (line.Contains("Final average return") ||
                    line.Contains("CVaR") ||
                    line.Contains("Training complete"))
                {
                    _logger.LogInformation("[Training Summary] {Line}", line);
                }
            }
        }

        private static string FindPythonExecutable()
        {
            // Try common Python paths
            var pythonPaths = new[]
            {
                "python",
                "python3",
                "/usr/bin/python3",
                "/usr/bin/python",
                @"C:\Python39\python.exe",
                @"C:\Python310\python.exe",
                @"C:\Python311\python.exe",
                @"C:\Python312\python.exe"
            };

            foreach (var path in pythonPaths)
            {
                try
                {
                    using var process = Process.Start(new ProcessStartInfo
                    {
                        FileName = path,
                        Arguments = "--version",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    });

                    if (process != null && process.WaitForExit(5000) && process.ExitCode == 0)
                    {
                        return path;
                    }
                }
                catch
                {
                    // Continue to next path
                }
            }

            return "python"; // Fallback
        }

        public void Dispose()
        {
            if (_disposed) return;

            _disposed = true;
            _timer?.Dispose();
            _logger.LogInformation("[EnhancedAutoRlTrainer] Disposed");
        }
    }
}