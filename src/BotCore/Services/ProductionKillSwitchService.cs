using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Services;

/// <summary>
/// Production-ready kill switch service that enforces DRY_RUN mode when kill.txt exists
/// Following agent guardrails: "kill.txt always forces DRY_RUN"
/// </summary>
public class ProductionKillSwitchService : IHostedService, IDisposable
{
    private readonly ILogger<ProductionKillSwitchService> _logger;
    private readonly FileSystemWatcher _fileWatcher;
    private readonly Timer _periodicCheck;
    private volatile bool _disposed;
    
    private const string KILL_FILE_NAME = "kill.txt";
    private const int CHECK_INTERVAL_MS = 1000; // Check every second

    public ProductionKillSwitchService(ILogger<ProductionKillSwitchService> logger)
    {
        _logger = logger;
        
        // Watch current directory for kill.txt
        _fileWatcher = new FileSystemWatcher(".", KILL_FILE_NAME)
        {
            NotifyFilter = NotifyFilters.CreationTime | NotifyFilters.LastWrite | NotifyFilters.FileName,
            EnableRaisingEvents = true
        };
        
        _fileWatcher.Created += OnKillFileDetected;
        _fileWatcher.Changed += OnKillFileDetected;
        
        // Periodic check as backup in case file watcher fails
        _periodicCheck = new Timer(PeriodicKillFileCheck, null, TimeSpan.Zero, TimeSpan.FromMilliseconds(CHECK_INTERVAL_MS));
        
        _logger.LogInformation("🛡️ [KILL-SWITCH] Production kill switch service initialized - monitoring for {File}", KILL_FILE_NAME);
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🟢 [KILL-SWITCH] Kill switch monitoring started");
        CheckKillFileOnStartup();
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔴 [KILL-SWITCH] Kill switch monitoring stopped");
        return Task.CompletedTask;
    }

    private void CheckKillFileOnStartup()
    {
        if (File.Exists(KILL_FILE_NAME))
        {
            _logger.LogCritical("🔴 [KILL-SWITCH] KILL FILE DETECTED ON STARTUP - Forcing DRY_RUN mode");
            EnforceDryRunMode("Startup detection");
        }
    }

    private void OnKillFileDetected(object sender, FileSystemEventArgs e)
    {
        _logger.LogCritical("🔴 [KILL-SWITCH] KILL FILE DETECTED - {EventType} - Forcing DRY_RUN mode", e.ChangeType);
        EnforceDryRunMode($"File event: {e.ChangeType}");
    }

    private void PeriodicKillFileCheck(object? state)
    {
        try
        {
            if (File.Exists(KILL_FILE_NAME))
            {
                _logger.LogCritical("🔴 [KILL-SWITCH] KILL FILE DETECTED (periodic check) - Forcing DRY_RUN mode");
                EnforceDryRunMode("Periodic check");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [KILL-SWITCH] Error during periodic kill file check");
        }
    }

    private void EnforceDryRunMode(string detectionMethod)
    {
        try
        {
            // Force DRY_RUN environment variable
            Environment.SetEnvironmentVariable("DRY_RUN", "true");
            Environment.SetEnvironmentVariable("EXECUTE", "false");
            Environment.SetEnvironmentVariable("AUTO_EXECUTE", "false");
            
            _logger.LogCritical("🛡️ [KILL-SWITCH] DRY_RUN MODE ENFORCED - Detection: {Method}", detectionMethod);
            _logger.LogCritical("🛡️ [KILL-SWITCH] All execution flags disabled for safety");
            
            // Log kill file contents if available for debugging
            LogKillFileContents();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [KILL-SWITCH] Critical error while enforcing DRY_RUN mode");
        }
    }

    private void LogKillFileContents()
    {
        try
        {
            if (File.Exists(KILL_FILE_NAME))
            {
                var contents = File.ReadAllText(KILL_FILE_NAME);
                if (!string.IsNullOrWhiteSpace(contents))
                {
                    _logger.LogInformation("📝 [KILL-SWITCH] Kill file contents: {Contents}", contents.Trim());
                }
                
                var fileInfo = new FileInfo(KILL_FILE_NAME);
                _logger.LogInformation("📅 [KILL-SWITCH] Kill file created: {Created}, modified: {Modified}", 
                    fileInfo.CreationTime, fileInfo.LastWriteTime);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ [KILL-SWITCH] Could not read kill file contents");
        }
    }

    /// <summary>
    /// Check if kill.txt exists (for use by other services)
    /// </summary>
    public static bool IsKillSwitchActive()
    {
        return File.Exists(KILL_FILE_NAME);
    }

    /// <summary>
    /// Get the current execution mode based on environment and kill switch
    /// Following guardrails: DRY_RUN precedence
    /// </summary>
    public static bool IsDryRunMode()
    {
        // Kill switch always forces DRY_RUN
        if (IsKillSwitchActive())
        {
            return true;
        }
        
        // Check environment variables with DRY_RUN precedence
        var dryRun = Environment.GetEnvironmentVariable("DRY_RUN");
        if (dryRun?.ToLowerInvariant() == "true")
        {
            return true;
        }
        
        var execute = Environment.GetEnvironmentVariable("EXECUTE");
        var autoExecute = Environment.GetEnvironmentVariable("AUTO_EXECUTE");
        
        // Default to DRY_RUN if execution flags are not explicitly true
        return execute?.ToLowerInvariant() != "true" && autoExecute?.ToLowerInvariant() != "true";
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            try
            {
                _fileWatcher?.Dispose();
                _periodicCheck?.Dispose();
                
                _logger.LogDebug("🗑️ [KILL-SWITCH] Kill switch service disposed");
            }
            catch (ObjectDisposedException)
            {
                // Expected during shutdown - ignore
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🗑️ [KILL-SWITCH] Error disposing resources");
            }
            finally
            {
                _disposed = true;
            }
        }
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}