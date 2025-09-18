using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace Trading.Safety;

/// <summary>
/// Monitors kill.txt file and immediately halts all trading operations when detected
/// Ensures safe system shutdown and state persistence
/// </summary>
public class KillSwitchWatcher : TradingBot.Abstractions.IKillSwitchWatcher, IDisposable
{
    private readonly ILogger<KillSwitchWatcher> _logger;
    private readonly AppOptions _config;
    private readonly FileSystemWatcher _fileWatcher;
    private bool _isActive = false;
    private bool _disposed = false;

    /// <summary>
    /// Raised when the kill switch is toggled. The bool parameter indicates whether it is now active (true) or inactive (false).
    /// </summary>
    public event Action<bool>? KillSwitchToggled;
    public event Action? OnKillSwitchActivated;
    public bool IsKillSwitchActive => _isActive;

    public KillSwitchWatcher(ILogger<KillSwitchWatcher> logger, IOptions<AppOptions> config)
    {
        _logger = logger;
        _config = config.Value;
        
        // Setup file system watcher for kill file
        var killFileDir = Path.GetDirectoryName(_config.KillFile) ?? Directory.GetCurrentDirectory();
        var killFileName = Path.GetFileName(_config.KillFile);
        
        _fileWatcher = new FileSystemWatcher(killFileDir, killFileName)
        {
            NotifyFilter = NotifyFilters.CreationTime | NotifyFilters.LastWrite | NotifyFilters.Size,
            EnableRaisingEvents = false
        };
        
        _fileWatcher.Created += OnKillFileDetected;
        _fileWatcher.Changed += OnKillFileDetected;
    }

    public async Task<bool> IsKillSwitchActiveAsync()
    {
        return await Task.FromResult(_isActive).ConfigureAwait(false).ConfigureAwait(false);
    }

    public async Task StartWatchingAsync()
    {
        await StartWatchingAsync(CancellationToken.None).ConfigureAwait(false);
    }

    public async Task StartWatchingAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Check if kill file already exists
            if (File.Exists(_config.KillFile))
            {
                _logger.LogWarning("[KILL_SWITCH] Kill file already exists at startup - activating kill switch");
                ActivateKillSwitch();
                return;
            }

            // Start file system monitoring
            _fileWatcher.EnableRaisingEvents = true;
            _logger.LogInformation("[KILL_SWITCH] Started monitoring kill file: {KillFile}", _config.KillFile);

            // Keep monitoring until cancellation requested
            while (!cancellationToken.IsCancellationRequested && !_isActive)
            {
                await Task.Delay(1000, cancellationToken).ConfigureAwait(false);
                
                // Double-check file existence every second (backup to file watcher)
                if (File.Exists(_config.KillFile) && !_isActive)
                {
                    _logger.LogWarning("[KILL_SWITCH] Kill file detected via polling");
                    ActivateKillSwitch();
                }
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("[KILL_SWITCH] Monitoring cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[KILL_SWITCH] Error in kill switch monitoring");
            throw;
        }
    }

    private void OnKillFileDetected(object sender, FileSystemEventArgs e)
    {
        if (!_isActive)
        {
            _logger.LogWarning("[KILL_SWITCH] Kill file detected via file watcher: {FilePath}", e.FullPath);
            ActivateKillSwitch();
        }
    }

    private void ActivateKillSwitch()
    {
        if (_isActive) return;
        
        _isActive = true;
        _logger.LogCritical("[KILL_SWITCH] 🔴 KILL SWITCH ACTIVATED - All trading operations will be halted");
        
        try
        {
            // Trigger events to notify all subscribers
            OnKillSwitchActivated?.Invoke();
            KillSwitchToggled?.Invoke(true);
            
            // Log activation to persistent state
            var stateFile = Path.Combine(Directory.GetCurrentDirectory(), "state", "kill_switch_activation.log");
            Directory.CreateDirectory(Path.GetDirectoryName(stateFile)!);
            
            var logEntry = $"{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC} - Kill switch activated{Environment.NewLine}";
            File.AppendAllText(stateFile, logEntry);
            
            _logger.LogInformation("[KILL_SWITCH] Kill switch activation logged to state file");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[KILL_SWITCH] Error during kill switch activation");
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _fileWatcher?.Dispose();
            _disposed = true;
        }
    }
}