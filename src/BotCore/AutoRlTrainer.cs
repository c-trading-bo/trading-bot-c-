using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore;

/// <summary>
/// Fully automated RL training pipeline - no manual intervention required!
/// Monitors training data, automatically exports, trains models, and deploys them.
/// </summary>
public sealed class AutoRlTrainer : IDisposable
{
    private readonly ILogger _log;
    private readonly Timer _timer;
    private readonly string _dataDir;
    private readonly string _modelDir;
    private readonly string _pythonExe;
    private bool _disposed;

    public AutoRlTrainer(ILogger logger)
    {
        _log = logger;
        _dataDir = Path.Combine(AppContext.BaseDirectory, "data", "rl_training");
        _modelDir = Path.Combine(AppContext.BaseDirectory, "models", "rl");
        _pythonExe = FindPythonExecutable();
        
        Directory.CreateDirectory(_dataDir);
        Directory.CreateDirectory(_modelDir);
        
        // Check every 30 minutes for training opportunities
        _timer = new Timer(CheckAndTrain, null, TimeSpan.Zero, TimeSpan.FromMinutes(30));
        _log.LogInformation("[AutoRlTrainer] Started - checking every 30 minutes for training data");
    }

    private string FindPythonExecutable()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ml", "rl_env", "Scripts", "python.exe"),
            Path.Combine(AppContext.BaseDirectory, "ml", "rl_env", "bin", "python"),
            "python.exe",
            "python"
        };

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate) || IsCommandAvailable(candidate))
            {
                _log.LogInformation("[AutoRlTrainer] Found Python: {Python}", candidate);
                return candidate;
            }
        }

        _log.LogWarning("[AutoRlTrainer] Python not found - training will be skipped");
        return "";
    }

    private static bool IsCommandAvailable(string command)
    {
        try
        {
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = command,
                    Arguments = "--version",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };
            process.Start();
            process.WaitForExit(5000);
            return process.ExitCode == 0;
        }
        catch { return false; }
    }

    private async void CheckAndTrain(object? state)
    {
        if (_disposed || string.IsNullOrEmpty(_pythonExe)) return;

        try
        {
            await CheckAndTrainAsync();
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[AutoRlTrainer] Training check failed");
        }
    }

    private async Task CheckAndTrainAsync()
    {
        // 1. Check if we have enough training data (at least 14 days)
        var minDate = DateTime.UtcNow.AddDays(-14);
        var featureFiles = Directory.GetFiles(_dataDir, "features_*.jsonl");
        
        var validFiles = 0;
        foreach (var file in featureFiles)
        {
            var fileName = Path.GetFileNameWithoutExtension(file);
            if (fileName.StartsWith("features_") && 
                DateTime.TryParseExact(fileName["features_".Length..], "yyyyMMdd", 
                    null, System.Globalization.DateTimeStyles.None, out var fileDate) &&
                fileDate >= minDate)
            {
                validFiles++;
            }
        }

        if (validFiles < 7) // Need at least 7 days of data
        {
            _log.LogDebug("[AutoRlTrainer] Not enough training data: {Days} days, need 7+", validFiles);
            return;
        }

        // 2. Check if model is recent (don't retrain too frequently)
        var latestModelPath = Path.Combine(_modelDir, "latest_rl_sizer.onnx");
        if (File.Exists(latestModelPath))
        {
            var lastTrained = File.GetLastWriteTimeUtc(latestModelPath);
            if (DateTime.UtcNow - lastTrained < TimeSpan.FromHours(12))
            {
                _log.LogDebug("[AutoRlTrainer] Model recent: {Age:F1} hours, waiting", 
                    (DateTime.UtcNow - lastTrained).TotalHours);
                return;
            }
        }

        _log.LogInformation("[AutoRlTrainer] Starting automated training - {Files} days of data available", validFiles);

        // 3. Export training data
        var csvPath = await ExportTrainingDataAsync();
        if (string.IsNullOrEmpty(csvPath))
        {
            _log.LogWarning("[AutoRlTrainer] Data export failed - skipping training");
            return;
        }

        // 4. Train new model
        var modelPath = await TrainModelAsync(csvPath);
        if (string.IsNullOrEmpty(modelPath))
        {
            _log.LogWarning("[AutoRlTrainer] Model training failed");
            return;
        }

        // 5. Deploy model (atomic replacement)
        await DeployModelAsync(modelPath);
        
        _log.LogInformation("[AutoRlTrainer] 🎯 Automated training complete! New model deployed: {Model}", 
            latestModelPath);
    }

    private Task<string?> ExportTrainingDataAsync()
    {
        try
        {
            var endDate = DateTime.UtcNow;
            var startDate = endDate.AddDays(-30);
            
            var csvPath = RlTrainingDataCollector.ExportToCsv(_log, startDate, endDate);
            
            if (File.Exists(csvPath))
            {
                var fileInfo = new FileInfo(csvPath);
                _log.LogInformation("[AutoRlTrainer] Exported training data: {Path} ({Size:F1} KB)", 
                    csvPath, fileInfo.Length / 1024.0);
                return Task.FromResult<string?>(csvPath);
            }
            
            return Task.FromResult<string?>(null);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[AutoRlTrainer] Failed to export training data");
            return Task.FromResult<string?>(null);
        }
    }

    private async Task<string?> TrainModelAsync(string csvPath)
    {
        try
        {
            var scriptPath = Path.Combine(AppContext.BaseDirectory, "ml", "rl", "train_cvar_ppo.py");
            var outputPath = Path.Combine(_modelDir, $"rl_sizer_{DateTime.UtcNow:yyyyMMdd_HHmmss}.onnx");
            
            var args = $"\"{scriptPath}\" --data \"{csvPath}\" --output_model \"{outputPath}\" --auto";
            
            _log.LogInformation("[AutoRlTrainer] Training: {Python} {Args}", _pythonExe, args);
            
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = _pythonExe,
                    Arguments = args,
                    WorkingDirectory = Path.Combine(AppContext.BaseDirectory, "ml"),
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            var output = "";
            var error = "";
            
            process.OutputDataReceived += (s, e) => { if (e.Data != null) output += e.Data + "\n"; };
            process.ErrorDataReceived += (s, e) => { if (e.Data != null) error += e.Data + "\n"; };
            
            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            
            // Allow up to 10 minutes for training
            if (!process.WaitForExit(600000))
            {
                process.Kill();
                _log.LogError("[AutoRlTrainer] Training timeout after 10 minutes");
                return null;
            }

            if (process.ExitCode == 0 && File.Exists(outputPath))
            {
                _log.LogInformation("[AutoRlTrainer] Training successful: {Output}", outputPath);
                return outputPath;
            }
            else
            {
                _log.LogError("[AutoRlTrainer] Training failed (exit {Code}): {Error}", 
                    process.ExitCode, error.Trim());
                return null;
            }
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[AutoRlTrainer] Training process failed");
            return null;
        }
    }

    private Task DeployModelAsync(string modelPath)
    {
        try
        {
            var latestPath = Path.Combine(_modelDir, "latest_rl_sizer.onnx");
            var backupPath = Path.Combine(_modelDir, $"backup_rl_sizer_{DateTime.UtcNow:yyyyMMdd_HHmmss}.onnx");
            
            // Backup existing model
            if (File.Exists(latestPath))
            {
                File.Move(latestPath, backupPath);
                _log.LogInformation("[AutoRlTrainer] Backed up existing model: {Backup}", backupPath);
            }
            
            // Atomic deployment
            File.Move(modelPath, latestPath);
            
            // Cleanup old backups (keep last 5)
            var backups = Directory.GetFiles(_modelDir, "backup_rl_sizer_*.onnx");
            if (backups.Length > 5)
            {
                Array.Sort(backups);
                for (int i = 0; i < backups.Length - 5; i++)
                {
                    File.Delete(backups[i]);
                }
            }
            
            _log.LogInformation("[AutoRlTrainer] Model deployed: {Latest}", latestPath);
            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[AutoRlTrainer] Model deployment failed");
            return Task.CompletedTask;
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
