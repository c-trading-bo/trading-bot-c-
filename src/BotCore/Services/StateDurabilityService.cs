using System;
using System.IO;
using System.IO.Compression;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// State durability service with daily backups and N-copy retention
    /// Backs up order ledger, UCB tables, position epochs, microstructure caps
    /// </summary>
    public class StateDurabilityService : BackgroundService
    {
        private readonly ILogger<StateDurabilityService> _logger;
        private readonly IPathConfig _pathConfig;
        private readonly TimeSpan _backupInterval = TimeSpan.FromHours(24); // Daily backups
        private const int MaxBackupCopies = 30; // Keep 30 days of backups

        public StateDurabilityService(ILogger<StateDurabilityService> logger, IPathConfig pathConfig)
        {
            _logger = logger;
            _pathConfig = pathConfig;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🏛️ [STATE-DURABILITY] Starting daily backup service");

            // Perform initial backup on startup
            await PerformBackupAsync().ConfigureAwait(false);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(_backupInterval, stoppingToken).ConfigureAwait(false);
                    await PerformBackupAsync().ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // Expected when cancellation is requested
                    break;
                }
                catch (IOException ex)
                {
                    _logger.LogError(ex, "🚨 [STATE-DURABILITY] I/O error during backup");
                    await Task.Delay(TimeSpan.FromHours(1), stoppingToken).ConfigureAwait(false);
                }
                catch (UnauthorizedAccessException ex)
                {
                    _logger.LogError(ex, "🚨 [STATE-DURABILITY] Access denied during backup");
                    await Task.Delay(TimeSpan.FromHours(1), stoppingToken).ConfigureAwait(false);
                }
                catch (InvalidOperationException ex)
                {
                    _logger.LogError(ex, "🚨 [STATE-DURABILITY] Invalid operation during backup");
                    await Task.Delay(TimeSpan.FromMinutes(30), stoppingToken).ConfigureAwait(false);
                }
                catch (Exception ex) when (!(ex is SystemException))
                {
                    _logger.LogError(ex, "🚨 [STATE-DURABILITY] Application error in backup loop");
                    // Continue running even if backup fails - retry in 1 hour
                    await Task.Delay(TimeSpan.FromHours(1), stoppingToken).ConfigureAwait(false);
                }
            }

            _logger.LogInformation("🏛️ [STATE-DURABILITY] Backup service stopped");
        }

        /// <summary>
        /// Perform immediate backup of all critical state
        /// </summary>
        public async Task PerformBackupAsync()
        {
            var backupStart = DateTime.UtcNow;
            var backupTimestamp = backupStart.ToString("yyyyMMdd_HHmmss");
            
            try
            {
                _logger.LogInformation("📦 [STATE-DURABILITY] Starting backup at {Timestamp}", backupStart);

                var statePath = _pathConfig.GetStatePath();
                var backupDir = Path.Combine(_pathConfig.GetDataRootPath(), "backups");
                var backupFile = Path.Combine(backupDir, $"state_backup_{backupTimestamp}.zip");

                // Ensure backup directory exists
                Directory.CreateDirectory(backupDir);

                // Create backup archive
                await CreateBackupArchiveAsync(statePath, backupFile).ConfigureAwait(false);

                // Cleanup old backups
                CleanupOldBackups(backupDir);

                var duration = DateTime.UtcNow - backupStart;
                _logger.LogInformation("✅ [STATE-DURABILITY] Backup completed in {Duration:mm\\:ss}: {BackupFile}", 
                    duration, backupFile);

            }
            catch (DirectoryNotFoundException ex)
            {
                _logger.LogError(ex, "🚨 [STATE-DURABILITY] State directory not found during backup at {Timestamp}", backupStart);
                await CreateBackupFailureAlertAsync(ex).ConfigureAwait(false);
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "🚨 [STATE-DURABILITY] Access denied during backup at {Timestamp}", backupStart);
                await CreateBackupFailureAlertAsync(ex).ConfigureAwait(false);
            }
            catch (IOException ex)
            {
                _logger.LogError(ex, "🚨 [STATE-DURABILITY] I/O error during backup at {Timestamp}", backupStart);
                await CreateBackupFailureAlertAsync(ex).ConfigureAwait(false);
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "🚨 [STATE-DURABILITY] Invalid operation during backup at {Timestamp}", backupStart);
                await CreateBackupFailureAlertAsync(ex).ConfigureAwait(false);
            }
        }

        /// <summary>
        /// Restore state from a specific backup
        /// </summary>
        public Task RestoreFromBackupAsync(string backupFileName)
        {
            return Task.Run(() =>
            {
                try
                {
                    _logger.LogWarning("🔄 [STATE-DURABILITY] Starting restore from backup: {BackupFile}", backupFileName);

                    var backupDir = Path.Combine(_pathConfig.GetDataRootPath(), "backups");
                    var backupFile = Path.Combine(backupDir, backupFileName);
                    var statePath = _pathConfig.GetStatePath();

                    if (!File.Exists(backupFile))
                    {
                        throw new FileNotFoundException($"Backup file not found: {backupFile}");
                    }

                    // Create restore directory
                    var restoreTemp = Path.Combine(_pathConfig.GetTempPath(), $"restore_{DateTime.UtcNow:yyyyMMdd_HHmmss}");
                    Directory.CreateDirectory(restoreTemp);

                    // Extract backup
                    ZipFile.ExtractToDirectory(backupFile, restoreTemp);

                    // Move current state to backup
                    var currentStateBackup = Path.Combine(backupDir, $"state_pre_restore_{DateTime.UtcNow:yyyyMMdd_HHmmss}");
                    if (Directory.Exists(statePath))
                    {
                        Directory.Move(statePath, currentStateBackup);
                        _logger.LogInformation("📂 [STATE-DURABILITY] Current state backed up to: {BackupPath}", currentStateBackup);
                    }

                    // Restore from backup
                    Directory.Move(Path.Combine(restoreTemp, "state"), statePath);

                    _logger.LogInformation("✅ [STATE-DURABILITY] Restore completed from: {BackupFile}", backupFileName);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "🚨 [STATE-DURABILITY] Restore failed from backup: {BackupFile}", backupFileName);
                    throw;
                }
            });
        }

        /// <summary>
        /// List available backups
        /// </summary>
        public string[] GetAvailableBackups()
        {
            try
            {
                var backupDir = Path.Combine(_pathConfig.GetDataRootPath(), "backups");
                if (!Directory.Exists(backupDir))
                    return Array.Empty<string>();

                var backupFiles = Directory.GetFiles(backupDir, "state_backup_*.zip");
                Array.Sort(backupFiles, StringComparer.OrdinalIgnoreCase);
                Array.Reverse(backupFiles); // Most recent first

                return Array.ConvertAll(backupFiles, file => Path.GetFileName(file) ?? "unknown");
            }
            catch (DirectoryNotFoundException ex)
            {
                _logger.LogWarning(ex, "Backup directory not found when listing available backups");
                return Array.Empty<string>();
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "Access denied when listing available backups");
                return Array.Empty<string>();
            }
            catch (IOException ex)
            {
                _logger.LogError(ex, "I/O error when listing available backups");
                return Array.Empty<string>();
            }
            catch (ArgumentException ex)
            {
                _logger.LogError(ex, "Invalid path when listing available backups");
                return Array.Empty<string>();
            }
        }

        private async Task CreateBackupArchiveAsync(string statePath, string backupFile)
        {
            if (!Directory.Exists(statePath))
            {
                _logger.LogWarning("📂 [STATE-DURABILITY] State directory does not exist, creating empty backup");
                Directory.CreateDirectory(statePath);
            }

            // Create backup with compression
            await Task.Run(() =>
            {
                ZipFile.CreateFromDirectory(statePath, backupFile, CompressionLevel.Optimal, false);
            }).ConfigureAwait(false);

            // Verify backup file
            var backupInfo = new FileInfo(backupFile);
            _logger.LogInformation("📊 [STATE-DURABILITY] Backup size: {Size:N0} bytes", backupInfo.Length);

            // Verify backup integrity
            try
            {
                using var archive = ZipFile.OpenRead(backupFile);
                _logger.LogInformation("📋 [STATE-DURABILITY] Backup contains {Count} entries", archive.Entries.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [STATE-DURABILITY] Backup integrity check failed");
                throw;
            }
        }

        private void CleanupOldBackups(string backupDir)
        {
            try
            {
                var backupFiles = Directory.GetFiles(backupDir, "state_backup_*.zip");
                Array.Sort(backupFiles, StringComparer.OrdinalIgnoreCase);
                
                if (backupFiles.Length > MaxBackupCopies)
                {
                    var filesToDelete = backupFiles[..^MaxBackupCopies]; // Keep only the last MaxBackupCopies
                    
                    foreach (var oldBackup in filesToDelete)
                    {
                        File.Delete(oldBackup);
                        _logger.LogInformation("🗑️ [STATE-DURABILITY] Deleted old backup: {BackupFile}", Path.GetFileName(oldBackup));
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error cleaning up old backups");
            }
        }

        private async Task CreateBackupFailureAlertAsync(Exception ex)
        {
            try
            {
                var alertPath = $"CRITICAL_ALERT_BACKUP_FAILURE_{DateTime.UtcNow:yyyyMMdd_HHmmss}.txt";
                var alertContent = $"{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC - STATE BACKUP FAILURE\n\n{ex}";
                
                await File.WriteAllTextAsync(alertPath, alertContent).ConfigureAwait(false);
                _logger.LogCritical("🚨 [STATE-DURABILITY] CRITICAL ALERT: Backup failure alert created at {AlertPath}", alertPath);
            }
            catch (Exception alertEx)
            {
                _logger.LogError(alertEx, "Failed to create backup failure alert");
            }
        }
    }
}