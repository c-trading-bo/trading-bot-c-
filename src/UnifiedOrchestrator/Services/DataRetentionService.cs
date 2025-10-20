using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Runtime;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Phase 10: Comprehensive Data Retention Service
/// Prevents bloat by cleaning up old data across all system components
/// Runs daily at 3 AM with configurable retention policies
/// </summary>
internal sealed class DataRetentionService : BackgroundService
{
    private readonly ILogger<DataRetentionService> _logger;
    private readonly FileModelRegistry _modelRegistry;
    private readonly Timer _dailyCleanupTimer;

    // Retention policies (configurable via environment variables)
    private readonly int _promotionRetentionDays;
    private readonly int _checkpointRetentionDays;
    private readonly int _trainingArtifactRetentionDays;
    private readonly int _tempFileRetentionHours;
    private readonly int _modelVersionsToKeep;
    private readonly int _experienceRetentionDays;
    private readonly int _liveTrainingDataRetentionDays;
    private readonly int _sessionBackupRetentionDays;

    public DataRetentionService(
        ILogger<DataRetentionService> logger,
        FileModelRegistry modelRegistry)
    {
        _logger = logger;
        _modelRegistry = modelRegistry;

        // Read retention policies from environment (defaults provided)
        _promotionRetentionDays = GetEnvInt("PROMOTION_RETENTION_DAYS", 90);
        _checkpointRetentionDays = GetEnvInt("CHECKPOINT_RETENTION_DAYS", 30);
        _trainingArtifactRetentionDays = GetEnvInt("TRAINING_ARTIFACT_RETENTION_DAYS", 14);
        _tempFileRetentionHours = GetEnvInt("TEMP_FILE_RETENTION_HOURS", 24);
        _modelVersionsToKeep = GetEnvInt("MODEL_VERSIONS_TO_KEEP", 10);
        _experienceRetentionDays = GetEnvInt("EXPERIENCE_RETENTION_DAYS", 90);
        _liveTrainingDataRetentionDays = GetEnvInt("LIVE_TRAINING_DATA_RETENTION_DAYS", 7);
        _sessionBackupRetentionDays = GetEnvInt("SESSION_BACKUP_RETENTION_DAYS", 30);

        // Schedule daily cleanup at 3 AM
        var now = DateTime.Now;
        var next3AM = now.Date.AddDays(1).AddHours(3);
        if (now.Hour < 3)
        {
            next3AM = now.Date.AddHours(3); // Run today at 3 AM if not yet passed
        }
        var timeUntil3AM = next3AM - now;

        _dailyCleanupTimer = new Timer(
            callback: _ => { _ = RunDailyCleanupAsync(); },
            state: null,
            dueTime: (int)timeUntil3AM.TotalMilliseconds,
            period: (int)TimeSpan.FromDays(1).TotalMilliseconds);

        _logger.LogInformation(
            "[DATA-RETENTION] Service initialized - Next cleanup at {NextRun}",
            next3AM);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[DATA-RETENTION] Service started - monitoring for bloat prevention");

        // Run initial cleanup on startup (non-blocking)
        _ = Task.Run(async () =>
        {
            await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken).ConfigureAwait(false);
            await RunDailyCleanupAsync().ConfigureAwait(false);
        }, stoppingToken);

        // Keep service alive
        while (!stoppingToken.IsCancellationRequested)
        {
            await Task.Delay(TimeSpan.FromHours(1), stoppingToken).ConfigureAwait(false);
        }
    }

    private async Task RunDailyCleanupAsync()
    {
        var startTime = DateTime.UtcNow;
        _logger.LogInformation("[DATA-RETENTION] ========== Daily Cleanup Started ==========");

        var stats = new CleanupStats();

        try
        {
            // 1. Clean up old promotion records (audit trail bloat)
            await CleanupPromotionRecordsAsync(stats).ConfigureAwait(false);

            // 2. Clean up old model versions (keep recent + champion)
            await CleanupOldModelVersionsAsync(stats).ConfigureAwait(false);

            // 3. Clean up training checkpoints
            await CleanupTrainingCheckpointsAsync(stats).ConfigureAwait(false);

            // 4. Clean up training artifacts (manifests, summaries, reports)
            await CleanupTrainingArtifactsAsync(stats).ConfigureAwait(false);

            // 5. Clean up temp files (.tmp, .lock, staging)
            await CleanupTempFilesAsync(stats).ConfigureAwait(false);

            // 6. Clean up old historical data cache
            await CleanupHistoricalDataCacheAsync(stats).ConfigureAwait(false);

            // 7. Clean up old validation reports
            await CleanupValidationReportsAsync(stats).ConfigureAwait(false);

            // 8. Clean up old experience files (Terminal trading experiences)
            await CleanupExperienceFilesAsync(stats).ConfigureAwait(false);

            // 9. Clean up live training data (JSONL files)
            await CleanupLiveTrainingDataAsync(stats).ConfigureAwait(false);

            // 10. Clean up position state session backups
            await CleanupPositionSessionBackupsAsync(stats).ConfigureAwait(false);

            var duration = DateTime.UtcNow - startTime;
            _logger.LogInformation(
                "[DATA-RETENTION] ========== Daily Cleanup Complete ========== " +
                "Duration: {Duration:F1}s | Files Removed: {FilesRemoved} | Space Freed: {SpaceMB:F2} MB",
                duration.TotalSeconds, stats.FilesRemoved, stats.BytesFreed / (1024.0 * 1024.0));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA-RETENTION] Daily cleanup failed");
        }
    }

    /// <summary>
    /// Clean up old promotion records (keep recent N days for audit)
    /// </summary>
    private async Task CleanupPromotionRecordsAsync(CleanupStats stats)
    {
        try
        {
            var promotionsPath = Path.Combine(Directory.GetCurrentDirectory(), "model_registry", "promotions");
            if (!Directory.Exists(promotionsPath))
            {
                return;
            }

            var cutoffDate = DateTime.UtcNow.AddDays(-_promotionRetentionDays);
            var files = Directory.GetFiles(promotionsPath, "*.json");
            var removedCount = 0;
            var removedSize = 0L;

            foreach (var file in files)
            {
                var fileInfo = new FileInfo(file);
                if (fileInfo.CreationTimeUtc < cutoffDate)
                {
                    removedSize += fileInfo.Length;
                    File.Delete(file);
                    removedCount++;
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} promotion records older than {Days} days ({SizeMB:F2} MB)",
                    removedCount, _promotionRetentionDays, removedSize / (1024.0 * 1024.0));
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup promotion records");
        }
    }

    /// <summary>
    /// Clean up old model versions (keep recent N versions per algorithm)
    /// </summary>
    private async Task CleanupOldModelVersionsAsync(CleanupStats stats)
    {
        try
        {
            var modelsPath = Path.Combine(Directory.GetCurrentDirectory(), "model_registry", "models");
            if (!Directory.Exists(modelsPath))
            {
                return;
            }

            // Get all algorithms (distinct prefixes)
            var algorithms = Directory.GetFiles(modelsPath, "*.json")
                .Select(f => Path.GetFileNameWithoutExtension(f))
                .Select(name => name.Split('_')[0])
                .Distinct()
                .ToList();

            foreach (var algorithm in algorithms)
            {
                await _modelRegistry.CleanupOldModelsAsync(algorithm, _modelVersionsToKeep).ConfigureAwait(false);
            }

            _logger.LogInformation(
                "[DATA-RETENTION] Cleaned up old model versions (kept {Keep} per algorithm)",
                _modelVersionsToKeep);

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup old model versions");
        }
    }

    /// <summary>
    /// Clean up old training checkpoints
    /// </summary>
    private async Task CleanupTrainingCheckpointsAsync(CleanupStats stats)
    {
        try
        {
            var checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoints");
            if (!Directory.Exists(checkpointPath))
            {
                return;
            }

            var cutoffDate = DateTime.UtcNow.AddDays(-_checkpointRetentionDays);
            var files = Directory.GetFiles(checkpointPath, "*.json", SearchOption.AllDirectories);
            var removedCount = 0;
            var removedSize = 0L;

            foreach (var file in files)
            {
                var fileInfo = new FileInfo(file);
                if (fileInfo.CreationTimeUtc < cutoffDate)
                {
                    removedSize += fileInfo.Length;
                    File.Delete(file);
                    removedCount++;
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} training checkpoints older than {Days} days",
                    removedCount, _checkpointRetentionDays);
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup training checkpoints");
        }
    }

    /// <summary>
    /// Clean up old training artifacts (manifests, summaries, reports)
    /// </summary>
    private async Task CleanupTrainingArtifactsAsync(CleanupStats stats)
    {
        try
        {
            var artifactsPath = Path.Combine(Directory.GetCurrentDirectory(), "training_artifacts");
            if (!Directory.Exists(artifactsPath))
            {
                return;
            }

            var cutoffDate = DateTime.UtcNow.AddDays(-_trainingArtifactRetentionDays);
            var patterns = new[] { "*.json", "*.md", "*.txt" };
            var removedCount = 0;
            var removedSize = 0L;

            foreach (var pattern in patterns)
            {
                var files = Directory.GetFiles(artifactsPath, pattern, SearchOption.AllDirectories);
                foreach (var file in files)
                {
                    // Keep GitHub backup manifests (they're small and useful for audit)
                    if (file.Contains("github_backup_manifest"))
                    {
                        continue;
                    }

                    var fileInfo = new FileInfo(file);
                    if (fileInfo.CreationTimeUtc < cutoffDate)
                    {
                        removedSize += fileInfo.Length;
                        File.Delete(file);
                        removedCount++;
                    }
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} training artifacts older than {Days} days",
                    removedCount, _trainingArtifactRetentionDays);
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup training artifacts");
        }
    }

    /// <summary>
    /// Clean up temp files (.tmp, .lock, staging directories)
    /// </summary>
    private async Task CleanupTempFilesAsync(CleanupStats stats)
    {
        try
        {
            var cutoffDate = DateTime.UtcNow.AddHours(-_tempFileRetentionHours);
            var rootPath = Directory.GetCurrentDirectory();
            var tempPatterns = new[] { "*.tmp", "*.lock", "*.staging" };
            var removedCount = 0;
            var removedSize = 0L;

            foreach (var pattern in tempPatterns)
            {
                var files = Directory.GetFiles(rootPath, pattern, SearchOption.AllDirectories);
                foreach (var file in files)
                {
                    try
                    {
                        var fileInfo = new FileInfo(file);
                        if (fileInfo.LastWriteTimeUtc < cutoffDate)
                        {
                            removedSize += fileInfo.Length;
                            File.Delete(file);
                            removedCount++;
                        }
                    }
                    catch (IOException)
                    {
                        // File may be in use, skip
                    }
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} temp files older than {Hours} hours",
                    removedCount, _tempFileRetentionHours);
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup temp files");
        }
    }

    /// <summary>
    /// Clean up old historical data cache (keep recent 30 days)
    /// </summary>
    private async Task CleanupHistoricalDataCacheAsync(CleanupStats stats)
    {
        try
        {
            var historicalPath = Path.Combine("data", "historical");
            if (!Directory.Exists(historicalPath))
            {
                return;
            }

            // Keep last 30 days of historical data cache
            var cutoffDate = DateTime.UtcNow.AddDays(-30);
            var files = Directory.GetFiles(historicalPath, "*.parquet", SearchOption.AllDirectories);
            var removedCount = 0;
            var removedSize = 0L;

            foreach (var file in files)
            {
                var fileInfo = new FileInfo(file);
                if (fileInfo.CreationTimeUtc < cutoffDate)
                {
                    removedSize += fileInfo.Length;
                    File.Delete(file);
                    removedCount++;
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} cached historical data files (freed {SizeMB:F2} MB)",
                    removedCount, removedSize / (1024.0 * 1024.0));
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup historical data cache");
        }
    }

    /// <summary>
    /// Clean up old validation reports
    /// </summary>
    private async Task CleanupValidationReportsAsync(CleanupStats stats)
    {
        try
        {
            var reportsPath = Path.Combine(Directory.GetCurrentDirectory(), "validation_reports");
            if (!Directory.Exists(reportsPath))
            {
                return;
            }

            var cutoffDate = DateTime.UtcNow.AddDays(-_trainingArtifactRetentionDays);
            var files = Directory.GetFiles(reportsPath, "*.json", SearchOption.AllDirectories);
            var removedCount = 0;
            var removedSize = 0L;

            foreach (var file in files)
            {
                var fileInfo = new FileInfo(file);
                if (fileInfo.CreationTimeUtc < cutoffDate)
                {
                    removedSize += fileInfo.Length;
                    File.Delete(file);
                    removedCount++;
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} validation reports older than {Days} days",
                    removedCount, _trainingArtifactRetentionDays);
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup validation reports");
        }
    }

    /// <summary>
    /// Clean up old experience files from Terminal trading
    /// </summary>
    private async Task CleanupExperienceFilesAsync(CleanupStats stats)
    {
        try
        {
            var experiencesPath = Path.Combine("data", "experiences");
            if (!Directory.Exists(experiencesPath))
            {
                return;
            }

            var cutoffDate = DateTime.UtcNow.AddDays(-_experienceRetentionDays);
            var files = Directory.GetFiles(experiencesPath, "*.json", SearchOption.TopDirectoryOnly);
            var removedCount = 0;
            var removedSize = 0L;

            foreach (var file in files)
            {
                try
                {
                    var filename = Path.GetFileName(file);
                    var timestampStr = filename.Split('_')[0];

                    if (DateTime.TryParse(timestampStr, out var fileDate))
                    {
                        if (fileDate < cutoffDate.Date)
                        {
                            var fileInfo = new FileInfo(file);
                            removedSize += fileInfo.Length;
                            File.Delete(file);
                            removedCount++;
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[DATA-RETENTION] Failed to delete experience file: {File}", file);
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} experience files older than {Days} days ({SizeMB:F2} MB)",
                    removedCount, _experienceRetentionDays, removedSize / (1024.0 * 1024.0));
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup experience files");
        }
    }

    /// <summary>
    /// Clean up old live training data (JSONL files)
    /// </summary>
    private async Task CleanupLiveTrainingDataAsync(CleanupStats stats)
    {
        try
        {
            var liveDataPath = Path.Combine("data", "live_trades");
            if (!Directory.Exists(liveDataPath))
            {
                return;
            }

            var cutoffDate = DateTime.UtcNow.AddDays(-_liveTrainingDataRetentionDays);
            var files = Directory.GetFiles(liveDataPath, "live_trades_*.jsonl", SearchOption.TopDirectoryOnly);
            var removedCount = 0;
            var removedSize = 0L;

            foreach (var file in files)
            {
                try
                {
                    var fileInfo = new FileInfo(file);
                    if (fileInfo.CreationTimeUtc < cutoffDate)
                    {
                        removedSize += fileInfo.Length;
                        File.Delete(file);
                        removedCount++;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[DATA-RETENTION] Failed to delete live training data file: {File}", file);
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} live training data files older than {Days} days ({SizeMB:F2} MB)",
                    removedCount, _liveTrainingDataRetentionDays, removedSize / (1024.0 * 1024.0));
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup live training data");
        }
    }

    /// <summary>
    /// Clean up old position state session backups
    /// </summary>
    private async Task CleanupPositionSessionBackupsAsync(CleanupStats stats)
    {
        try
        {
            var baseStatePath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "TradingBot",
                "State");

            var sessionsPath = Path.Combine(baseStatePath, "Sessions");
            var backupsPath = Path.Combine(baseStatePath, "Backups");

            var cutoffDate = DateTime.UtcNow.AddDays(-_sessionBackupRetentionDays);
            var removedCount = 0;
            var removedSize = 0L;

            // Clean up old session directories
            if (Directory.Exists(sessionsPath))
            {
                var sessionDirs = Directory.GetDirectories(sessionsPath);
                foreach (var dir in sessionDirs)
                {
                    try
                    {
                        var dirInfo = new DirectoryInfo(dir);
                        if (dirInfo.CreationTimeUtc < cutoffDate)
                        {
                            var size = dirInfo.EnumerateFiles("*", SearchOption.AllDirectories)
                                .Sum(fi => fi.Length);
                            removedSize += size;

                            Directory.Delete(dir, recursive: true);
                            removedCount += Directory.GetFiles(dir, "*", SearchOption.AllDirectories).Length;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "[DATA-RETENTION] Failed to delete session directory: {Dir}", dir);
                    }
                }
            }

            // Clean up old backup directories (full_backup_*)
            if (Directory.Exists(backupsPath))
            {
                var backupDirs = Directory.GetDirectories(backupsPath, "full_backup_*");
                foreach (var dir in backupDirs)
                {
                    try
                    {
                        var dirInfo = new DirectoryInfo(dir);
                        if (dirInfo.CreationTimeUtc < cutoffDate)
                        {
                            var size = dirInfo.EnumerateFiles("*", SearchOption.AllDirectories)
                                .Sum(fi => fi.Length);
                            removedSize += size;

                            Directory.Delete(dir, recursive: true);
                            removedCount += 1; // Count directories
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "[DATA-RETENTION] Failed to delete backup directory: {Dir}", dir);
                    }
                }
            }

            stats.FilesRemoved += removedCount;
            stats.BytesFreed += removedSize;

            if (removedCount > 0)
            {
                _logger.LogInformation(
                    "[DATA-RETENTION] Cleaned up {Count} position state backups older than {Days} days ({SizeMB:F2} MB)",
                    removedCount, _sessionBackupRetentionDays, removedSize / (1024.0 * 1024.0));
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DATA-RETENTION] Failed to cleanup position session backups");
        }
    }

    public override void Dispose()
    {
        _dailyCleanupTimer?.Dispose();
        base.Dispose();
    }

    private static int GetEnvInt(string key, int defaultValue)
    {
        var value = Environment.GetEnvironmentVariable(key);
        return int.TryParse(value, out var result) ? result : defaultValue;
    }

    private sealed class CleanupStats
    {
        public int FilesRemoved { get; set; }
        public long BytesFreed { get; set; }
    }
}
