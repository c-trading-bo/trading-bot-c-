using System;
using System.IO;
using System.Security;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TopstepX.Bot.Abstractions;

namespace TopstepX.Bot.Core.Services
{
    /// <summary>
    /// Emergency Stop System - Critical Safety Component
    /// Monitors kill.txt file and provides emergency shutdown capabilities
    /// </summary>
    public class EmergencyStopSystem : BackgroundService
    {
        private readonly ILogger<EmergencyStopSystem> _logger;
        private readonly string _killFilePath;
        private readonly CancellationTokenSource _emergencyStopSource;
        private FileSystemWatcher? _fileWatcher;
        private volatile bool _isEmergencyStop;
        
        public event EventHandler<EmergencyStopEventArgs>? EmergencyStopTriggered;
        
        public bool IsEmergencyStop => _isEmergencyStop;
        public CancellationToken EmergencyStopToken => _emergencyStopSource.Token;
        
        public EmergencyStopSystem(ILogger<EmergencyStopSystem> logger)
        {
            _logger = logger;
            _killFilePath = Path.Combine(Directory.GetCurrentDirectory(), "kill.txt");
            _emergencyStopSource = new CancellationTokenSource();
        }
        
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            try
            {
                // Initial check for kill.txt
                CheckKillFile();
                
                // Setup file system watcher
                SetupFileWatcher();
                
                _logger.LogInformation("🛡️ Emergency Stop System initialized - monitoring {KillFile}", _killFilePath);
                
                // Keep monitoring until cancellation
                while (!stoppingToken.IsCancellationRequested)
                {
                    await Task.Delay(1000, stoppingToken).ConfigureAwait(false);
                    
                    // Periodic check in case file watcher fails
                    if (!_isEmergencyStop)
                    {
                        CheckKillFile();
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during shutdown
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "❌ Emergency Stop System failed - access denied");
            }
            catch (DirectoryNotFoundException ex)
            {
                _logger.LogError(ex, "❌ Emergency Stop System failed - directory not found");
            }
            catch (IOException ex)
            {
                _logger.LogError(ex, "❌ Emergency Stop System failed - I/O error");
            }
            catch (SecurityException ex)
            {
                _logger.LogError(ex, "❌ Emergency Stop System failed - security error");
            }
            finally
            {
                _fileWatcher?.Dispose();
            }
        }
        
        private void SetupFileWatcher()
        {
            try
            {
                var directory = Path.GetDirectoryName(_killFilePath) ?? Directory.GetCurrentDirectory();
                _fileWatcher = new FileSystemWatcher(directory, "kill.txt");
                
                _fileWatcher.Created += OnKillFileChanged;
                _fileWatcher.Changed += OnKillFileChanged;
                _fileWatcher.EnableRaisingEvents = true;
                
                _logger.LogDebug("📂 File watcher setup for {Directory}", directory);
            }
            catch (ArgumentException ex)
            {
                _logger.LogError(ex, "❌ Failed to setup file watcher - invalid argument");
            }
            catch (DirectoryNotFoundException ex)
            {
                _logger.LogError(ex, "❌ Failed to setup file watcher - directory not found");
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "❌ Failed to setup file watcher - access denied");
            }
            catch (NotSupportedException ex)
            {
                _logger.LogError(ex, "❌ Failed to setup file watcher - not supported");
            }
        }
        
        private void OnKillFileChanged(object sender, FileSystemEventArgs e)
        {
            _logger.LogWarning("🚨 Kill file detected: {EventType}", e.ChangeType);
            CheckKillFile();
        }
        
        private void CheckKillFile()
        {
            try
            {
                if (File.Exists(_killFilePath))
                {
                    TriggerEmergencyStop("kill.txt file detected");
                }
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "❌ Error checking kill file - access denied");
            }
            catch (DirectoryNotFoundException ex)
            {
                _logger.LogError(ex, "❌ Error checking kill file - directory not found");
            }
            catch (IOException ex)
            {
                _logger.LogError(ex, "❌ Error checking kill file - I/O error");
            }
        }
        
        /// <summary>
        /// Manually trigger emergency stop
        /// </summary>
        public void TriggerEmergencyStop(string reason)
        {
            if (_isEmergencyStop) return;
            
            _isEmergencyStop = true;
            _emergencyStopSource.Cancel();
            
            _logger.LogCritical("🛑 EMERGENCY STOP TRIGGERED: {Reason}", reason);
            
            var eventArgs = new EmergencyStopEventArgs
            {
                Reason = reason,
                Timestamp = DateTime.UtcNow
            };
            
            EmergencyStopTriggered?.Invoke(this, eventArgs);
            
            // Create emergency log file
            CreateEmergencyLog(reason);
        }
        
        private void CreateEmergencyLog(string reason)
        {
            try
            {
                var logPath = Path.Combine(Directory.GetCurrentDirectory(), $"emergency_stop_{DateTime.UtcNow:yyyyMMdd_HHmmss}.log");
                var logContent = $"""
                    EMERGENCY STOP EVENT
                    ====================
                    Timestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC
                    Reason: {reason}
                    Process ID: {Environment.ProcessId}
                    Machine: {Environment.MachineName}
                    User: {Environment.UserName}
                    
                    ACTIONS REQUIRED:
                    - Verify all positions are closed
                    - Check for pending orders
                    - Review trading logs
                    - Investigate root cause before restart
                    """;
                    
                File.WriteAllText(logPath, logContent);
                _logger.LogInformation("📋 Emergency log created: {LogPath}", logPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to create emergency log");
            }
        }
        
        /// <summary>
        /// Reset emergency stop (requires manual intervention)
        /// </summary>
        public async Task<bool> ResetEmergencyStopAsync()
        {
            try
            {
                // Remove kill.txt if it exists
                if (File.Exists(_killFilePath))
                {
                    File.Delete(_killFilePath);
                    _logger.LogInformation("🗑️ kill.txt removed");
                }
                
                // Wait a moment
                await Task.Delay(1000).ConfigureAwait(false);
                
                // Reset state
                _isEmergencyStop = false;
                
                _logger.LogWarning("🔄 Emergency stop reset - system ready");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to reset emergency stop");
                return false;
            }
        }
        
        public override void Dispose()
        {
            _fileWatcher?.Dispose();
            _emergencyStopSource?.Dispose();
            base.Dispose();
        }
    }
}