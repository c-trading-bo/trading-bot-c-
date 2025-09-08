using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Infrastructure;

/// <summary>
/// Simple kill switch implementation that monitors for kill.txt file
/// </summary>
public class SimpleKillSwitchWatcher : IKillSwitchWatcher
{
    private readonly ILogger<SimpleKillSwitchWatcher> _logger;
    private readonly string _killFilePath;

    public SimpleKillSwitchWatcher(ILogger<SimpleKillSwitchWatcher> logger)
    {
        _logger = logger;
        _killFilePath = Path.Combine(Environment.CurrentDirectory, "kill.txt");
    }

    public bool IsKillSwitchActive => File.Exists(_killFilePath);

    public event Action<bool>? KillSwitchToggled;
    public event Action? OnKillSwitchActivated;

    public async Task<bool> IsKillSwitchActiveAsync()
    {
        await Task.Delay(1); // Make it async
        return IsKillSwitchActive;
    }

    public async Task StartWatchingAsync()
    {
        await Task.Delay(1); // Make it async
        _logger.LogInformation("Kill switch watcher started");
    }
}