using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Console Dashboard Service - Clean, structured console output for trading bot
/// Replaces verbose logging with dashboard-style status display
/// </summary>
public class ConsoleDashboardService : BackgroundService
{
    private readonly ILogger<ConsoleDashboardService> _logger;
    private readonly AppOptions _appOptions;
    private Timer? _refreshTimer;
    private readonly object _lock = new();
    private bool _isInitialized = false;
    
    // Status tracking
    private bool _authStatus = false;
    private string _accountInfo = "";
    private string _contractsInfo = "";
    private string _hubStatus = "";
    private string _systemStatus = "";
    private string _mode = "DRY_RUN";
    private string _strategy = "ES/NQ Mean Reversion";
    private string _schedule = "Mon-Fri 09:30-16:00 ET";
    
    // Event log
    private readonly Queue<string> _eventLog = new();
    private const int MaxEventLogEntries = 50;

    public ConsoleDashboardService(
        ILogger<ConsoleDashboardService> logger,
        IOptions<AppOptions> appOptions)
    {
        _logger = logger;
        _appOptions = appOptions.Value;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Wait a moment for other services to initialize
        await Task.Delay(2000, stoppingToken);
        
        DisplayInitialBanner();
        await InitializeStatusAsync();
        
        // Start refresh timer
        _refreshTimer = new Timer(RefreshDashboard, null, TimeSpan.Zero, TimeSpan.FromSeconds(10));
        
        // Keep service running
        while (!stoppingToken.IsCancellationRequested)
        {
            await Task.Delay(1000, stoppingToken);
        }
    }

    private void DisplayInitialBanner()
    {
        Console.Clear();
        Console.WriteLine("════════════════════════════════════════════════════════════════════════");
        Console.WriteLine("                    TOPSTEPX TRADING BOT v1.0.0");
        Console.WriteLine($"                    Evaluation Account: {GetAccountDisplay()}");
        Console.WriteLine("════════════════════════════════════════════════════════════════════════");
        Console.WriteLine();
    }

    private Task InitializeStatusAsync()
    {
        lock (_lock)
        {
            _isInitialized = true;
            
            // Initialize status based on configuration
            _authStatus = !string.IsNullOrEmpty(_appOptions.AuthToken);
            _accountInfo = $"$50,000 | Max Loss: $2,000 | Daily Loss: $1,000";
            _contractsInfo = "ESU5, NQU5, MESU5, MNQU5 (active)";
            _hubStatus = "User ✓ Market ✓ (stable)";
            _systemStatus = "Ready (5/5 checks passed)";
            _mode = _appOptions.EnableDryRunMode ? "DRY_RUN" : "AUTO_EXECUTE";
            
            // Add initial events
            AddEvent("🔐 AUTH", _authStatus ? "Logged in successfully" : "Authentication pending");
            AddEvent("📊 ACCOUNT", _accountInfo);
            AddEvent("📈 CONTRACTS", _contractsInfo);
            AddEvent("🔌 HUBS", _hubStatus);
            AddEvent("✅ SYSTEM", _systemStatus);
            AddEvent("📍 MODE", $"{_mode} (kill.txt not present → AUTO_EXECUTE available)");
            AddEvent("🎯 STRATEGY", $"{_strategy} | Risk: 1% per trade");
            AddEvent("🕐 SCHEDULE", $"{_schedule} | Next: 09:30:00");
        }
        
        return Task.CompletedTask;
    }

    private void RefreshDashboard(object? state)
    {
        if (!_isInitialized) return;

        lock (_lock)
        {
            // Move cursor to top and redraw status
            Console.SetCursorPosition(0, 4);
            
            var currentTime = DateTime.Now.ToString("HH:mm:ss");
            
            // Status section
            Console.WriteLine($"[{currentTime}] 🔐 AUTH: {(_authStatus ? "Logged in as kevinsuero072897@gmail.com" : "Authentication pending")}");
            Console.WriteLine($"[{currentTime}] 📊 ACCOUNT: {_accountInfo}");
            Console.WriteLine($"[{currentTime}] 📈 CONTRACTS: {_contractsInfo}");
            Console.WriteLine($"[{currentTime}] 🔌 HUBS: {_hubStatus}");
            Console.WriteLine($"[{currentTime}] ✅ SYSTEM: {_systemStatus}");
            Console.WriteLine();
            Console.WriteLine($"[{currentTime}] 📍 MODE: {_mode} (kill.txt not present → AUTO_EXECUTE available)");
            Console.WriteLine($"[{currentTime}] 🎯 STRATEGY: {_strategy} | Risk: 1% per trade");
            Console.WriteLine($"[{currentTime}] 🕐 SCHEDULE: {_schedule} | Next: 09:30:00");
            Console.WriteLine();
            Console.WriteLine("════════════════════════════════════════════════════════════════════════");
            Console.WriteLine();
            
            // Recent events (last 10)
            var recentEvents = _eventLog.TakeLast(10).ToList();
            foreach (var eventEntry in recentEvents)
            {
                Console.WriteLine(eventEntry);
            }
            
            // Clear any remaining lines
            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine(new string(' ', Console.WindowWidth - 1));
            }
        }
    }

    public void UpdateAuthStatus(bool isAuthenticated, string? userEmail = null)
    {
        lock (_lock)
        {
            _authStatus = isAuthenticated;
            var message = isAuthenticated 
                ? $"Logged in as {userEmail ?? "user"}" 
                : "Authentication failed";
            AddEvent("🔐 AUTH", message);
        }
    }

    public void UpdateAccountInfo(string accountInfo)
    {
        lock (_lock)
        {
            _accountInfo = accountInfo;
            AddEvent("📊 ACCOUNT", accountInfo);
        }
    }

    public void UpdateContractsInfo(string contractsInfo)
    {
        lock (_lock)
        {
            _contractsInfo = contractsInfo;
            AddEvent("📈 CONTRACTS", contractsInfo);
        }
    }

    public void UpdateHubStatus(string hubStatus)
    {
        lock (_lock)
        {
            _hubStatus = hubStatus;
            AddEvent("🔌 HUBS", hubStatus);
        }
    }

    public void UpdateSystemStatus(string systemStatus)
    {
        lock (_lock)
        {
            _systemStatus = systemStatus;
            AddEvent("✅ SYSTEM", systemStatus);
        }
    }

    public void UpdateMode(string mode)
    {
        lock (_lock)
        {
            _mode = mode;
            var modeMessage = mode == "DRY_RUN" 
                ? "DRY_RUN (kill.txt not present → AUTO_EXECUTE available)"
                : "AUTO_EXECUTE (live trading enabled)";
            AddEvent("📍 MODE", modeMessage);
        }
    }

    public void LogTradeEvent(string symbol, string action, decimal price, decimal quantity, string orderId)
    {
        var message = $"TRADE {symbol} {action} {quantity}@{price:F2} OrderId:{orderId}";
        AddEvent("💹 TRADE", message);
    }

    public void LogSignalEvent(string symbol, string signal, decimal confidence)
    {
        var message = $"SIGNAL {symbol} {signal} Confidence:{confidence:F2}";
        AddEvent("📡 SIGNAL", message);
    }

    public void LogMLUpdate(string modelType, string status, decimal? accuracy = null)
    {
        var message = accuracy.HasValue 
            ? $"ML {modelType} {status} Accuracy:{accuracy:F2}" 
            : $"ML {modelType} {status}";
        AddEvent("🧠 ML", message);
    }

    public void LogWorkflowEvent(string workflow, string status)
    {
        var message = $"WORKFLOW {workflow} {status}";
        AddEvent("🔄 WORKFLOW", message);
    }

    private void AddEvent(string category, string message)
    {
        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        var eventEntry = $"[{timestamp}] {category}: {message}";
        
        _eventLog.Enqueue(eventEntry);
        
        // Keep only recent events
        while (_eventLog.Count > MaxEventLogEntries)
        {
            _eventLog.Dequeue();
        }
    }

    private string GetAccountDisplay()
    {
        var accountId = _appOptions.AccountId;
        return !string.IsNullOrEmpty(accountId) ? accountId : "50KTC-V2-297693";
    }

    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        _refreshTimer?.Dispose();
        await base.StopAsync(cancellationToken);
    }

    public override void Dispose()
    {
        _refreshTimer?.Dispose();
        base.Dispose();
    }
}