using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using BotCore.Configuration;
using TradingBot.Abstractions;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Services;

/// <summary>
/// Production kill switch service - monitors for critical system failures
/// Kill switch is MANUAL ONLY or triggered by critical system failures
/// When activated, it forces shutdown of live trading for safety
/// Implements IKillSwitchWatcher for compatibility with legacy Safety module integrations
/// </summary>
public sealed class ProductionKillSwitchService : IHostedService, IKillSwitchWatcher, IDisposable
{
    private readonly ILogger<ProductionKillSwitchService> _logger;
    private readonly KillSwitchConfiguration _config;
    private readonly FileSystemWatcher _fileWatcher;
    private readonly Timer _periodicCheck;
    private volatile bool _disposed;
    
    // Default kill file path for static method fallback
    private const string DefaultKillFilePath = "kill.txt";

    // IKillSwitchWatcher implementation - for compatibility with legacy Safety module integrations
    public event EventHandler<KillSwitchToggledEventArgs>? KillSwitchToggled;
    public event EventHandler? OnKillSwitchActivated;
    
    // IKillSwitchWatcher property implementation (explicit to avoid conflict with static method)
    bool IKillSwitchWatcher.IsKillSwitchActive => IsKillSwitchActive();

    public ProductionKillSwitchService(ILogger<ProductionKillSwitchService> logger, IOptions<KillSwitchConfiguration> config)
    {
        _logger = logger;
        _config = config?.Value ?? throw new ArgumentNullException(nameof(config));
        
        // Validate configuration
        _config.Validate();
        
        // Resolve kill file path (support relative and absolute paths)
        var killFilePath = ResolveKillFilePath(_config.FilePath);
        var killFileDirectory = Path.GetDirectoryName(killFilePath) ?? Directory.GetCurrentDirectory();
        var killFileName = Path.GetFileName(killFilePath);
        
        // Ensure directory exists
        Directory.CreateDirectory(killFileDirectory);
        
        // Watch for kill file
        _fileWatcher = new FileSystemWatcher(killFileDirectory, killFileName)
        {
            NotifyFilter = NotifyFilters.CreationTime | NotifyFilters.LastWrite | NotifyFilters.FileName,
            EnableRaisingEvents = true
        };
        
        _fileWatcher.Created += OnKillFileDetected;
        _fileWatcher.Changed += OnKillFileDetected;
        
        // Periodic check as backup in case file watcher fails
        _periodicCheck = new Timer(PeriodicKillFileCheck, null, TimeSpan.Zero, TimeSpan.FromMilliseconds(_config.CheckIntervalMs));
        
        _logger.LogInformation("🛡️ [KILL-SWITCH] Production kill switch service initialized - monitoring for critical failures at {File}", killFilePath);
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
        var killFilePath = ResolveKillFilePath(_config.FilePath);
        if (File.Exists(killFilePath))
        {
            _logger.LogCritical("🔴 [KILL-SWITCH] CRITICAL FAILURE - Kill file detected on startup - Forcing shutdown");
            EnforceDryRunMode("Critical failure detected on startup");
        }
    }

    private void OnKillFileDetected(object sender, FileSystemEventArgs e)
    {
        _logger.LogCritical("🔴 [KILL-SWITCH] CRITICAL SYSTEM FAILURE - Kill file {EventType} - Forcing shutdown", e.ChangeType);
        EnforceDryRunMode($"Critical system failure: {e.ChangeType}");
    }

    private void PeriodicKillFileCheck(object? state)
    {
        try
        {
            var killFilePath = ResolveKillFilePath(_config.FilePath);
            if (File.Exists(killFilePath))
            {
                _logger.LogCritical("🔴 [KILL-SWITCH] CRITICAL FAILURE DETECTED (periodic check) - Forcing shutdown");
                EnforceDryRunMode("Critical failure - periodic check");
            }
        }
        catch (IOException ex)
        {
            _logger.LogError(ex, "❌ [KILL-SWITCH] I/O error during periodic kill file check");
        }
        catch (UnauthorizedAccessException ex)
        {
            _logger.LogError(ex, "❌ [KILL-SWITCH] Access denied during periodic kill file check");
        }
        catch (NotSupportedException ex)
        {
            _logger.LogError(ex, "❌ [KILL-SWITCH] Operation not supported during periodic kill file check");
        }
    }

    public void EnforceDryRunMode(string detectionMethod)
    {
        try
        {
            // Force DRY_RUN=1 environment variable (paper trading with live data)
            Environment.SetEnvironmentVariable("DRY_RUN", "1");
            
            _logger.LogCritical("🛡️ [KILL-SWITCH] DRY_RUN MODE ENFORCED - Detection: {Method}", detectionMethod);
            _logger.LogCritical("🛡️ [KILL-SWITCH] Bot switched to paper trading mode - continues with live data but simulates trades");
            
            // Create DRY_RUN marker file if enabled
            if (_config.CreateDryRunMarker)
            {
                CreateDryRunMarker(detectionMethod);
            }
            
            // Log kill file contents if available for debugging
            LogKillFileContents();
            
            // AUDIT-CLEAN: Publish guardrail metric for monitoring alerting per audit requirements
            PublishGuardrailMetric("kill_switch_activated", 1.0, detectionMethod);
            
            // Fire IKillSwitchWatcher events for compatibility with legacy Safety module integrations
            try
            {
                KillSwitchToggled?.Invoke(this, new KillSwitchToggledEventArgs(true));
                OnKillSwitchActivated?.Invoke(this, EventArgs.Empty);
            }
            catch (InvalidOperationException eventEx)
            {
                _logger.LogError(eventEx, "❌ [KILL-SWITCH] Invalid operation firing kill switch events");
            }
            catch (ArgumentException eventEx)
            {
                _logger.LogError(eventEx, "❌ [KILL-SWITCH] Invalid argument firing kill switch events");
            }
        }
        catch (System.Security.SecurityException ex)
        {
            _logger.LogError(ex, "❌ [KILL-SWITCH] Security error while enforcing DRY_RUN mode");
        }
        catch (UnauthorizedAccessException ex)
        {
            _logger.LogError(ex, "❌ [KILL-SWITCH] Access denied while enforcing DRY_RUN mode");
        }
    }

    private void CreateDryRunMarker(string reason)
    {
        try
        {
            var markerPath = ResolveKillFilePath(_config.DryRunMarkerPath);
            var markerDirectory = Path.GetDirectoryName(markerPath);
            if (!string.IsNullOrEmpty(markerDirectory))
            {
                Directory.CreateDirectory(markerDirectory);
            }
            
            var markerContent = $"""
                DRY_RUN MODE ENFORCED
                =====================
                Timestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC
                Reason: {reason}
                Process ID: {Environment.ProcessId}
                Kill File: {ResolveKillFilePath(_config.FilePath)}
                """;
                
            File.WriteAllText(markerPath, markerContent);
            _logger.LogInformation("📝 [KILL-SWITCH] DRY_RUN marker created: {MarkerPath}", markerPath);
        }
        catch (IOException ex)
        {
            _logger.LogWarning(ex, "⚠️ [KILL-SWITCH] I/O error creating DRY_RUN marker file");
        }
        catch (UnauthorizedAccessException ex)
        {
            _logger.LogWarning(ex, "⚠️ [KILL-SWITCH] Access denied creating DRY_RUN marker file");
        }
        catch (NotSupportedException ex)
        {
            _logger.LogWarning(ex, "⚠️ [KILL-SWITCH] Operation not supported creating DRY_RUN marker file");
        }
    }

    private void LogKillFileContents()
    {
        try
        {
            var killFilePath = ResolveKillFilePath(_config.FilePath);
            if (File.Exists(killFilePath))
            {
                var contents = File.ReadAllText(killFilePath);
                if (!string.IsNullOrWhiteSpace(contents))
                {
                    _logger.LogInformation("📝 [KILL-SWITCH] Kill file contents: {Contents}", contents.Trim());
                }
                
                var fileInfo = new FileInfo(killFilePath);
                _logger.LogInformation("📅 [KILL-SWITCH] Kill file created: {Created}, modified: {Modified}", 
                    fileInfo.CreationTime, fileInfo.LastWriteTime);
            }
        }
        catch (IOException ex)
        {
            _logger.LogWarning(ex, "⚠️ [KILL-SWITCH] I/O error reading kill file contents");
        }
        catch (UnauthorizedAccessException ex)
        {
            _logger.LogWarning(ex, "⚠️ [KILL-SWITCH] Access denied reading kill file contents");
        }
        catch (NotSupportedException ex)
        {
            _logger.LogWarning(ex, "⚠️ [KILL-SWITCH] Operation not supported reading kill file contents");
        }
    }

    /// <summary>
    /// Resolve kill file path (support relative and absolute paths)
    /// </summary>
    private static string ResolveKillFilePath(string configuredPath)
    {
        if (Path.IsPathRooted(configuredPath))
        {
            return configuredPath;
        }
        return Path.Combine(Directory.GetCurrentDirectory(), configuredPath);
    }

    /// <summary>
    /// Check if kill file exists (for use by other services)
    /// Uses default kill.txt file path when called statically
    /// </summary>
    public static bool IsKillSwitchActive()
    {
        // Static method uses default kill file path
        return File.Exists(DefaultKillFilePath);
    }
    
    /// <summary>
    /// Check if kill file exists using custom configuration
    /// </summary>
    public static bool IsKillSwitchActive(string killFilePath)
    {
        var resolvedPath = ResolveKillFilePath(killFilePath);
        return File.Exists(resolvedPath);
    }

    /// <summary>
    /// Get the current execution mode based on DRY_RUN environment variable
    /// DRY_RUN=1: Paper trading with live data (simulated trades)
    /// DRY_RUN=0: Live trading with real executions
    /// </summary>
    public static bool IsDryRunMode()
    {
        // Check DRY_RUN environment variable (defaults to true for safety)
        var dryRun = Environment.GetEnvironmentVariable("DRY_RUN");
        
        // Default to DRY_RUN if not explicitly set to 0/false
        if (string.IsNullOrEmpty(dryRun))
        {
            return true;
        }
        
        // DRY_RUN=0 or DRY_RUN=false means live trading
        return !(dryRun == "0" || dryRun.Equals("false", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Get whether the bot is running in historical replay mode
    /// HISTORICAL_MODE=1: Historical data replay mode (no API calls, local files only)
    /// HISTORICAL_MODE=0: Live or DRY-RUN mode (uses TopstepX API)
    /// </summary>
    public static bool IsHistoricalMode()
    {
        // Check HISTORICAL_MODE environment variable (defaults to false)
        var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE");
        
        // Default to false if not set
        if (string.IsNullOrEmpty(historicalMode))
        {
            return false;
        }
        
        // HISTORICAL_MODE=1 or HISTORICAL_MODE=true means historical replay mode
        return historicalMode == "1" || historicalMode.Equals("true", StringComparison.OrdinalIgnoreCase);
    }

    // IKillSwitchWatcher interface implementation
    public Task<bool> IsKillSwitchActiveAsync()
    {
        return Task.FromResult(IsKillSwitchActive());
    }

    public Task StartWatchingAsync()
    {
        // Already started in StartAsync, this is for interface compatibility
        return Task.CompletedTask;
    }

    /// <summary>
    /// Publish guardrail metrics for monitoring and alerting per audit requirements
    /// AUDIT-CLEAN: Ensure metrics push to central observability stack
    /// </summary>
    private void PublishGuardrailMetric(string metricName, double value, string context)
    {
        try
        {
            // Use structured logging with metric tags for observability systems to pick up
            _logger.LogCritical("📊 [GUARDRAIL-METRIC] {MetricName}={Value} Context={Context} Timestamp={Timestamp}", 
                $"guardrail.{metricName}", value, context, DateTimeOffset.UtcNow.ToUnixTimeSeconds());
                
            // Additional structured log for time-series systems
            _logger.LogInformation("METRIC: guardrail.{MetricName} {Value} {UnixTimestamp} tags=context:{Context},component:kill_switch", 
                metricName, value, DateTimeOffset.UtcNow.ToUnixTimeSeconds(), context);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [GUARDRAIL-METRIC] Invalid operation publishing metric {MetricName}", metricName);
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "❌ [GUARDRAIL-METRIC] Invalid argument publishing metric {MetricName}", metricName);
        }
    }

    private void Dispose(bool disposing)
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
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "🗑️ [KILL-SWITCH] Invalid operation disposing resources");
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