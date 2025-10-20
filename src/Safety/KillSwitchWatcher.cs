using System;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace Trading.Safety;

/// <summary>
/// Legacy KillSwitchWatcher for backward compatibility
/// Wraps ProductionKillSwitchService to provide the interface expected by tests
/// </summary>
[Obsolete("Use ProductionKillSwitchService via dependency injection instead. This class is provided for backward compatibility only.")]
public class KillSwitchWatcher : IKillSwitchWatcher
{
    private readonly BotCore.Services.ProductionKillSwitchService? _productionService;
    private readonly string _killFilePath;

    public event EventHandler<KillSwitchToggledEventArgs>? KillSwitchToggled;
    public event EventHandler? OnKillSwitchActivated;

    public bool IsKillSwitchActive => IsProductionKillSwitchActive();

    public KillSwitchWatcher(string killFilePath = "kill.txt")
    {
        _killFilePath = killFilePath;
        _productionService = null; // Static methods will be used
    }

    public KillSwitchWatcher(BotCore.Services.ProductionKillSwitchService productionService)
    {
        _productionService = productionService ?? throw new ArgumentNullException(nameof(productionService));
        _killFilePath = "kill.txt";
        
        // Wire up events from production service
        if (_productionService is IKillSwitchWatcher watcher)
        {
            watcher.KillSwitchToggled += (sender, args) => KillSwitchToggled?.Invoke(this, args);
            watcher.OnKillSwitchActivated += (sender, args) => OnKillSwitchActivated?.Invoke(this, args);
        }
    }

    /// <summary>
    /// Check if kill switch is active asynchronously
    /// </summary>
    public Task<bool> IsKillSwitchActiveAsync()
    {
        return Task.FromResult(IsProductionKillSwitchActive());
    }

    /// <summary>
    /// Start watching for kill switch changes (no-op for file-based implementation)
    /// </summary>
    public Task StartWatchingAsync()
    {
        // File-based implementation doesn't need active watching
        // Production service handles this if available
        return Task.CompletedTask;
    }

    /// <summary>
    /// Check if production kill switch is active (uses production service if available, otherwise checks file)
    /// </summary>
    public bool IsProductionKillSwitchActive()
    {
        if (_productionService != null && _productionService is IKillSwitchWatcher watcher)
        {
            return watcher.IsKillSwitchActive;
        }

        // Fallback to static file check
        return File.Exists(_killFilePath);
    }

    /// <summary>
    /// Check if system should force DRY_RUN mode
    /// </summary>
    public bool ShouldForceDryRun()
    {
        // Always force dry run if kill switch is active
        if (IsProductionKillSwitchActive())
            return true;

        // Check if we're in a safe environment (not production)
        var environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production";
        if (environment.Equals("Development", StringComparison.OrdinalIgnoreCase) ||
            environment.Equals("Test", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        // Check if live trading is explicitly disabled
        var allowLiveTrading = Environment.GetEnvironmentVariable("SAFETY_ALLOW_LIVE_TRADING");
        if (string.IsNullOrEmpty(allowLiveTrading) || 
            !bool.TryParse(allowLiveTrading, out var allowed) || 
            !allowed)
        {
            return true;
        }

        return false;
    }

    /// <summary>
    /// Activate the kill switch (creates kill file)
    /// </summary>
    public void Activate()
    {
        File.WriteAllText(_killFilePath, $"Kill switch activated at {DateTime.UtcNow:O}");
        OnKillSwitchActivated?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>
    /// Deactivate the kill switch (removes kill file)
    /// </summary>
    public void Deactivate()
    {
        if (File.Exists(_killFilePath))
        {
            File.Delete(_killFilePath);
        }
    }
}
