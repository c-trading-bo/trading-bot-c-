using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;

namespace BotCore.Infra
{
    /// <summary>
    /// Monitors the health of the 24/7 ML/RL learning pipeline.
    /// Tracks data collection, model freshness, training success, and alerts on issues.
    /// </summary>
    public sealed class MlPipelineHealthMonitor : IDisposable
    {
        private readonly ILogger _log;
        private readonly Timer _checkTimer;
        private readonly string _dataDir;
        private readonly string _modelDir;
        private bool _disposed;

        // Health metrics
        private DateTime _lastDataCollection = DateTime.MinValue;
        private DateTime _lastModelUpdate = DateTime.MinValue;
        private DateTime _lastTrainingAttempt = DateTime.MinValue;
        private int _consecutiveDataCollectionFailures = 0;
        private int _consecutiveTrainingFailures = 0;
        private bool _lastHealthCheckPassed = true;

        // Thresholds
        private const int MaxHoursWithoutData = 2;
        private const int MaxHoursWithoutModelUpdate = 48;
        private const int MaxConsecutiveFailures = 5;

        public MlPipelineHealthMonitor(ILogger logger)
        {
            _log = logger;
            _dataDir = Path.Combine(AppContext.BaseDirectory, "data", "rl_training");
            _modelDir = Path.Combine(AppContext.BaseDirectory, "models", "rl");

            Directory.CreateDirectory(_dataDir);
            Directory.CreateDirectory(_modelDir);

            // Check health every 30 minutes
            var interval = TimeSpan.FromMinutes(30);
            _checkTimer = new Timer(CheckHealthAsync, null, TimeSpan.Zero, interval);
            
            _log.LogInformation("[ML-Health] Started monitoring pipeline health every {Interval}", interval);
        }

        private async void CheckHealthAsync(object? state)
        {
            try
            {
                await PerformHealthCheckAsync();
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[ML-Health] Error during health check");
            }
        }

        private async Task PerformHealthCheckAsync()
        {
            var issues = new List<string>();
            var warnings = new List<string>();

            // Check data collection
            CheckDataCollection(issues, warnings);

            // Check model freshness
            CheckModelFreshness(issues, warnings);

            // Check training activity
            CheckTrainingActivity(issues, warnings);

            // Check GitHub Actions pipeline
            await CheckGitHubPipelineAsync(issues, warnings);

            // Check file system health
            CheckFileSystemHealth(issues, warnings);

            // Report results
            if (issues.Any())
            {
                _lastHealthCheckPassed = false;
                _log.LogError("[ML-Health] ❌ Pipeline health check failed with {Count} critical issues: {Issues}", 
                    issues.Count, string.Join("; ", issues));
                
                // Attempt automatic recovery for known issues
                await AttemptAutomaticRecoveryAsync(issues);
            }
            else if (warnings.Any())
            {
                _lastHealthCheckPassed = true;
                _log.LogWarning("[ML-Health] ⚠️ Pipeline health check passed with {Count} warnings: {Warnings}", 
                    warnings.Count, string.Join("; ", warnings));
            }
            else
            {
                _lastHealthCheckPassed = true;
                _log.LogInformation("[ML-Health] ✅ Pipeline health check passed - all systems operational");
            }

            // Log health metrics
            LogHealthMetrics();
        }

        private void CheckDataCollection(List<string> issues, List<string> warnings)
        {
            try
            {
                // Check for recent data files
                var recentFiles = Directory.GetFiles(_dataDir, "*.jsonl")
                    .Where(f => File.GetLastWriteTime(f) > DateTime.Now.AddHours(-MaxHoursWithoutData))
                    .ToList();

                if (!recentFiles.Any())
                {
                    _consecutiveDataCollectionFailures++;
                    issues.Add($"No data collection in last {MaxHoursWithoutData} hours");
                    
                    if (_consecutiveDataCollectionFailures >= MaxConsecutiveFailures)
                    {
                        issues.Add($"Data collection has failed {_consecutiveDataCollectionFailures} times consecutively");
                    }
                }
                else
                {
                    _consecutiveDataCollectionFailures = 0;
                    _lastDataCollection = recentFiles.Max(f => File.GetLastWriteTime(f));
                    
                    // Check data quality
                    var totalSize = recentFiles.Sum(f => new FileInfo(f).Length);
                    if (totalSize < 1024) // Less than 1KB suggests minimal data
                    {
                        warnings.Add("Data collection volume is very low");
                    }
                }
            }
            catch (Exception ex)
            {
                issues.Add($"Failed to check data collection: {ex.Message}");
            }
        }

        private void CheckModelFreshness(List<string> issues, List<string> warnings)
        {
            try
            {
                var latestModel = Path.Combine(_modelDir, "latest_rl_sizer.onnx");
                
                if (!File.Exists(latestModel))
                {
                    issues.Add("No latest RL model found");
                    return;
                }

                var modelAge = DateTime.Now - File.GetLastWriteTime(latestModel);
                _lastModelUpdate = File.GetLastWriteTime(latestModel);

                if (modelAge.TotalHours > MaxHoursWithoutModelUpdate)
                {
                    warnings.Add($"Model is {modelAge.TotalHours:F1} hours old (threshold: {MaxHoursWithoutModelUpdate}h)");
                }

                // Check model file integrity
                var modelInfo = new FileInfo(latestModel);
                if (modelInfo.Length < 1024) // Very small model suggests corruption
                {
                    issues.Add("Latest model file appears corrupted (too small)");
                }
            }
            catch (Exception ex)
            {
                issues.Add($"Failed to check model freshness: {ex.Message}");
            }
        }

        private void CheckTrainingActivity(List<string> issues, List<string> warnings)
        {
            try
            {
                // Look for training logs or backup files as evidence of training activity
                var backupFiles = Directory.GetFiles(_modelDir, "backup_rl_sizer_*.onnx");
                
                if (backupFiles.Any())
                {
                    var latestBackup = backupFiles.Max(f => File.GetCreationTime(f));
                    _lastTrainingAttempt = latestBackup;
                    
                    var timeSinceTraining = DateTime.Now - latestBackup;
                    if (timeSinceTraining.TotalHours > 8) // Should train every 6 hours
                    {
                        warnings.Add($"No training activity detected in {timeSinceTraining.TotalHours:F1} hours");
                    }
                }
                else
                {
                    warnings.Add("No evidence of recent training activity");
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to check training activity: {ex.Message}");
            }
        }

        private async Task CheckGitHubPipelineAsync(List<string> issues, List<string> warnings)
        {
            try
            {
                // This would ideally check GitHub Actions API, but for now just check if cloud trainer is configured
                var manifestUrl = Environment.GetEnvironmentVariable("MODEL_MANIFEST_URL");
                var hmacKey = Environment.GetEnvironmentVariable("MANIFEST_HMAC_KEY");
                
                if (string.IsNullOrEmpty(manifestUrl) || string.IsNullOrEmpty(hmacKey))
                {
                    warnings.Add("Cloud training pipeline not configured (missing environment variables)");
                }
                
                // TODO: Add actual GitHub Actions status check via API
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to check GitHub pipeline: {ex.Message}");
            }
        }

        private void CheckFileSystemHealth(List<string> issues, List<string> warnings)
        {
            try
            {
                // Check disk space
                var dataDir = new DirectoryInfo(_dataDir);
                var modelDir = new DirectoryInfo(_modelDir);
                
                var drive = new DriveInfo(dataDir.Root.FullName);
                var freeSpaceGB = drive.AvailableFreeSpace / (1024 * 1024 * 1024);
                
                if (freeSpaceGB < 1)
                {
                    issues.Add($"Low disk space: {freeSpaceGB:F1} GB remaining");
                }
                else if (freeSpaceGB < 5)
                {
                    warnings.Add($"Disk space getting low: {freeSpaceGB:F1} GB remaining");
                }

                // Check for too many files (cleanup needed)
                var dataFileCount = Directory.GetFiles(_dataDir, "*.jsonl").Length;
                var modelFileCount = Directory.GetFiles(_modelDir, "*.onnx").Length;
                
                if (dataFileCount > 1000)
                {
                    warnings.Add($"Many data files ({dataFileCount}) - cleanup may be needed");
                }
                
                if (modelFileCount > 50)
                {
                    warnings.Add($"Many model files ({modelFileCount}) - cleanup may be needed");
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to check file system health: {ex.Message}");
            }
        }

        private async Task AttemptAutomaticRecoveryAsync(List<string> issues)
        {
            foreach (var issue in issues)
            {
                try
                {
                    if (issue.Contains("No data collection"))
                    {
                        _log.LogInformation("[ML-Health] Attempting to restart data collection...");
                        // This would trigger data collection restart if we had a reference to the collectors
                        // For now, just log the attempt
                    }
                    else if (issue.Contains("corrupted"))
                    {
                        _log.LogInformation("[ML-Health] Attempting to restore model from backup...");
                        await RestoreModelFromBackupAsync();
                    }
                }
                catch (Exception ex)
                {
                    _log.LogError(ex, "[ML-Health] Failed to recover from issue: {Issue}", issue);
                }
            }
        }

        private async Task RestoreModelFromBackupAsync()
        {
            try
            {
                var backupFiles = Directory.GetFiles(_modelDir, "backup_rl_sizer_*.onnx")
                    .OrderByDescending(f => File.GetCreationTime(f))
                    .ToList();

                if (backupFiles.Any())
                {
                    var latestBackup = backupFiles.First();
                    var latestModel = Path.Combine(_modelDir, "latest_rl_sizer.onnx");
                    
                    File.Copy(latestBackup, latestModel, true);
                    _log.LogInformation("[ML-Health] Restored model from backup: {Backup}", Path.GetFileName(latestBackup));
                }
                else
                {
                    _log.LogWarning("[ML-Health] No backup models available for restoration");
                }
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[ML-Health] Failed to restore model from backup");
            }
        }

        private void LogHealthMetrics()
        {
            try
            {
                var metrics = new
                {
                    timestamp = DateTime.UtcNow,
                    lastDataCollection = _lastDataCollection,
                    lastModelUpdate = _lastModelUpdate,
                    lastTrainingAttempt = _lastTrainingAttempt,
                    consecutiveDataFailures = _consecutiveDataCollectionFailures,
                    consecutiveTrainingFailures = _consecutiveTrainingFailures,
                    healthCheckPassed = _lastHealthCheckPassed,
                    dataFileCount = Directory.Exists(_dataDir) ? Directory.GetFiles(_dataDir, "*.jsonl").Length : 0,
                    modelFileCount = Directory.Exists(_modelDir) ? Directory.GetFiles(_modelDir, "*.onnx").Length : 0
                };

                var metricsPath = Path.Combine(_dataDir, "health_metrics.jsonl");
                var json = JsonSerializer.Serialize(metrics);
                File.AppendAllText(metricsPath, json + Environment.NewLine);

                // Keep only last 1000 health records
                var lines = File.ReadAllLines(metricsPath);
                if (lines.Length > 1000)
                {
                    File.WriteAllLines(metricsPath, lines.TakeLast(1000));
                }
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[ML-Health] Failed to log health metrics");
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _checkTimer?.Dispose();
            _log.LogInformation("[ML-Health] Health monitoring stopped");
        }
    }
}