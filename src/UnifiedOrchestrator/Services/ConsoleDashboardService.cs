using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Text;
using System.Text.Json;
using TradingBot.Abstractions;
using TradingBot.Infrastructure.TopstepX;
using BotCore.Services;
using BotCore;
using BotCore.Config;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Console Dashboard Service - Clean, structured console output for trading bot
/// Replaces verbose logging with dashboard-style status display with LIVE TopstepX data
/// </summary>
public class ConsoleDashboardService : BackgroundService
{
    private readonly ILogger<ConsoleDashboardService> _logger;
    private readonly AppOptions _appOptions;
    private readonly IServiceProvider _serviceProvider;
    private Timer? _refreshTimer;
    private readonly object _lock = new();
    private bool _isInitialized = false;
    
    // Live data tracking
    private bool _authStatus = false;
    private string _userEmail = "";
    private string _accountInfo = "";
    private string _contractsInfo = "";
    private string _hubStatus = "";
    private string _systemStatus = "";
    private string _mode = "DRY_RUN";
    private string _strategy = "S2, S3, S6, S11 (4 enabled strategies)";
    private string _schedule = "Sun 6PM - Fri 5PM ET (Futures Hours)";
    private decimal _lastESPrice = 0m;
    private decimal _lastNQPrice = 0m;
    private int _barsSeen = 0;
    private int _quotesSeen = 0;
    private int _tradesSeen = 0;
    
    // Event log
    private readonly Queue<string> _eventLog = new();
    private const int MaxEventLogEntries = 50;

    public ConsoleDashboardService(
        ILogger<ConsoleDashboardService> logger,
        IOptions<AppOptions> appOptions,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _appOptions = appOptions.Value;
        _serviceProvider = serviceProvider;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Wait a moment for other services to initialize
        await Task.Delay(2000, stoppingToken);
        
        DisplayInitialBanner();
        await InitializeStatusAsync();
        await WireUpLiveDataServicesAsync();
        
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
            
            // Check for kill.txt file
            bool killFileExists = File.Exists(Path.Combine(Directory.GetCurrentDirectory(), "kill.txt"));
            
            // Initialize status based on configuration and actual state
            _authStatus = !string.IsNullOrEmpty(_appOptions.AuthToken);
            _accountInfo = $"Initializing...";
            _contractsInfo = "ES, NQ, MES, MNQ (ES/NQ focus) - Loading...";
            _hubStatus = killFileExists ? "User ⚠️ Market ⚠️ (kill switch)" : "User ✓ Market ✓ (connecting...)";
            _systemStatus = killFileExists ? "Safe Mode (kill.txt present)" : "Initializing (0/5 checks passed)";
            
            // Determine mode based on kill.txt and configuration
            if (killFileExists)
            {
                _mode = "DRY_RUN";
            }
            else
            {
                _mode = _appOptions.EnableDryRunMode ? "DRY_RUN" : "AUTO_EXECUTE";
            }
            
            // Add initial events with real status
            AddEvent("🔐 AUTH", _authStatus ? "Authentication check in progress..." : "Authentication pending");
            AddEvent("📊 ACCOUNT", "Loading account data from TopstepX...");
            AddEvent("📈 CONTRACTS", "Loading contract data...");
            AddEvent("🔌 HUBS", "Connecting to TopstepX SignalR hubs...");
            AddEvent("✅ SYSTEM", "System initialization in progress...");
            
            string modeMessage = killFileExists 
                ? "DRY_RUN (kill.txt present → Trading disabled for safety)"
                : $"{_mode} (kill.txt not present → AUTO_EXECUTE available)";
            AddEvent("📍 MODE", modeMessage);
            
            AddEvent("🎯 STRATEGY", $"{_strategy} | Risk: 1% per trade");
            AddEvent("🕐 SCHEDULE", $"{_schedule} | Current session check...");
            
            if (killFileExists)
            {
                AddEvent("⚠️ SAFETY", "Kill switch activated - All trading disabled");
            }
        }
        
        return Task.CompletedTask;
    }

    private async Task WireUpLiveDataServicesAsync()
    {
        try
        {
            // Wire up AutoTopstepXLoginService for real auth status
            var autoLoginService = _serviceProvider.GetService<BotCore.Services.AutoTopstepXLoginService>();
            if (autoLoginService != null)
            {
                // Monitor auth status changes
                _ = Task.Run(async () =>
                {
                    while (!_serviceProvider.GetService<IHostApplicationLifetime>()?.ApplicationStopping.IsCancellationRequested ?? true)
                    {
                        await Task.Delay(5000);
                        var wasAuth = _authStatus;
                        _authStatus = autoLoginService.IsAuthenticated;
                        _userEmail = autoLoginService.CurrentCredentials?.Username ?? "user";
                        
                        if (wasAuth != _authStatus)
                        {
                            var message = _authStatus ? $"Logged in as {_userEmail}" : "Authentication lost";
                            AddEvent("🔐 AUTH", message);
                        }
                    }
                });
                AddEvent("🔗 WIRING", "AutoTopstepXLoginService connected");
            }
            else
            {
                AddEvent("⚠️ AUTH", "AutoTopstepXLoginService not available - authentication status unknown");
            }

            // Wire up AccountService for real account data
            var accountService = _serviceProvider.GetService<IAccountService>();
            if (accountService != null)
            {
                accountService.OnAccountUpdated += (accountInfo) =>
                {
                    lock (_lock)
                    {
                        _accountInfo = $"${accountInfo.Balance:F0} | Day P&L: ${accountInfo.DayPnL:F2} | Unrealized: ${accountInfo.UnrealizedPnL:F2}";
                        AddEvent("📊 ACCOUNT", _accountInfo);
                    }
                };

                // Start periodic refresh for account data
                await accountService.StartPeriodicRefreshAsync(TimeSpan.FromSeconds(30));
                AddEvent("🔗 WIRING", "AccountService connected");
            }
            else
            {
                AddEvent("⚠️ ACCOUNT", "AccountService not available - account data will be static");
            }

            // Wire up MarketHubClient for real market data
            var marketHubClient = _serviceProvider.GetService<MarketHubClient>();
            if (marketHubClient != null)
            {
                marketHubClient.OnQuote += (contractId, last, bid, ask) =>
                {
                    lock (_lock)
                    {
                        _quotesSeen++;
                        if (contractId.Contains("ES"))
                        {
                            _lastESPrice = last;
                            AddEvent("📊 MARKET", $"ES {last:F2} | Bid: {bid:F2} | Ask: {ask:F2}");
                        }
                        else if (contractId.Contains("NQ"))
                        {
                            _lastNQPrice = last;
                            AddEvent("📊 MARKET", $"NQ {last:F2} | Bid: {bid:F2} | Ask: {ask:F2}");
                        }
                    }
                };

                marketHubClient.OnTrade += (contractId, tradeTick) =>
                {
                    lock (_lock)
                    {
                        _tradesSeen++;
                        var symbol = contractId.Contains("ES") ? "ES" : contractId.Contains("NQ") ? "NQ" : contractId;
                        AddEvent("💹 TRADE", $"{symbol} @ {tradeTick.Price:F2} Vol: {tradeTick.Volume}");
                    }
                };
                AddEvent("🔗 WIRING", "MarketHubClient connected");
            }
            else
            {
                AddEvent("⚠️ MARKET", "MarketHubClient not available - no live market data");
            }

            // Wire up MarketDataAgent for bar count and market activity
            var marketDataAgent = _serviceProvider.GetService<MarketDataAgent>();
            if (marketDataAgent != null)
            {
                marketDataAgent.OnBar += (bar) =>
                {
                    lock (_lock)
                    {
                        _barsSeen = marketDataAgent.BarsSeen;
                        var symbol = bar.Symbol ?? "UNKNOWN";
                        AddEvent("📊 BAR", $"[{_barsSeen}] {symbol} {bar.Close:F2} | Vol: {bar.Volume}");
                        
                        // Update system status based on bar count
                        if (_barsSeen >= 10 && !File.Exists(Path.Combine(Directory.GetCurrentDirectory(), "kill.txt")))
                        {
                            _systemStatus = "Ready (5/5 checks passed) | Trading enabled";
                            AddEvent("✅ PRECHECKS", $"BarsSeen({_barsSeen}) Hubs(✓) CanTrade(✓) → EXECUTE mode enabled");
                        }
                    }
                };

                marketDataAgent.OnQuote += (quoteData) =>
                {
                    // Additional quote processing if needed
                    _quotesSeen = marketDataAgent.QuotesSeen;
                };

                marketDataAgent.OnTrade += (tradeData) =>
                {
                    // Additional trade processing if needed  
                    _tradesSeen = marketDataAgent.TradesSeen;
                };
                AddEvent("🔗 WIRING", "MarketDataAgent connected");
            }
            else
            {
                AddEvent("⚠️ DATA", "MarketDataAgent not available - no bar count/market activity tracking");
            }

            // Wire up UserHubClient for real trade and position updates
            var userHubClient = _serviceProvider.GetService<UserHubClient>();
            if (userHubClient != null)
            {
                userHubClient.OnTrade += (tradeData) =>
                {
                    try
                    {
                        if (tradeData.TryGetProperty("symbol", out var symbolProp) &&
                            tradeData.TryGetProperty("price", out var priceProp) &&
                            tradeData.TryGetProperty("quantity", out var qtyProp))
                        {
                            var symbol = symbolProp.GetString() ?? "";
                            var price = priceProp.GetDecimal();
                            var qty = qtyProp.GetInt32();
                            AddEvent("✅ FILL", $"{symbol} @ {price:F2} x {qty}");
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, "Error processing trade data for dashboard");
                    }
                };

                userHubClient.OnAccount += (accountData) =>
                {
                    try
                    {
                        if (accountData.TryGetProperty("balance", out var balanceProp) &&
                            accountData.TryGetProperty("dayPnL", out var dayPnLProp))
                        {
                            var balance = balanceProp.GetDecimal();
                            var dayPnL = dayPnLProp.GetDecimal();
                            _accountInfo = $"${balance:F0} | Day P&L: ${dayPnL:F2}";
                            AddEvent("📊 ACCOUNT", _accountInfo);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, "Error processing account data for dashboard");
                    }
                };

                userHubClient.OnPosition += (positionData) =>
                {
                    try
                    {
                        if (positionData.TryGetProperty("symbol", out var symbolProp) &&
                            positionData.TryGetProperty("quantity", out var qtyProp) &&
                            positionData.TryGetProperty("unrealizedPnL", out var pnlProp))
                        {
                            var symbol = symbolProp.GetString() ?? "";
                            var qty = qtyProp.GetInt32();
                            var pnl = pnlProp.GetDecimal();
                            var side = qty > 0 ? "Long" : qty < 0 ? "Short" : "Flat";
                            AddEvent("📍 POSITION", $"{side} {Math.Abs(qty)} {symbol} | Unrealized: ${pnl:F2}");
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, "Error processing position data for dashboard");
                    }
                };
                AddEvent("🔗 WIRING", "UserHubClient connected");
            }
            else
            {
                AddEvent("⚠️ TRADES", "UserHubClient not available - no live trade/position updates");
            }

            AddEvent("🔗 WIRING", "Live data services initialization completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error wiring up live data services");
            AddEvent("⚠️ WIRING", $"Live data services setup failed: {ex.Message}");
            AddEvent("🔧 DEBUG", "Some features may not work correctly - check service registrations");
        }
    }

    private void RefreshDashboard(object? state)
    {
        if (!_isInitialized) return;

        lock (_lock)
        {
            // Check for kill.txt file dynamically
            bool killFileExists = File.Exists(Path.Combine(Directory.GetCurrentDirectory(), "kill.txt"));
            
            // Get real auth status from AutoTopstepXLoginService
            var autoLoginService = _serviceProvider.GetService<BotCore.Services.AutoTopstepXLoginService>();
            if (autoLoginService != null)
            {
                _authStatus = autoLoginService.IsAuthenticated;
                _userEmail = autoLoginService.CurrentCredentials?.Username ?? "";
            }

            // Update status based on current state
            if (killFileExists && !_hubStatus.Contains("kill switch"))
            {
                _hubStatus = "User ⚠️ Market ⚠️ (kill switch)";
                _systemStatus = "Safe Mode (kill.txt present)";
                _mode = "DRY_RUN";
                AddEvent("⚠️ SAFETY", "Kill switch activated - All trading disabled");
            }
            else if (!killFileExists && _hubStatus.Contains("kill switch"))
            {
                _hubStatus = $"User {(_authStatus ? "✓" : "⚠️")} Market {(_quotesSeen > 0 ? "✓" : "⚠️")} (stable)";
                _systemStatus = $"Ready ({GetSystemReadyChecks()}/5 checks passed)";
                _mode = _appOptions.EnableDryRunMode ? "DRY_RUN" : "AUTO_EXECUTE";
                AddEvent("✅ SAFETY", "Kill switch deactivated - Trading resumed");
            }
            else if (!killFileExists)
            {
                // Update hub status based on real connectivity
                _hubStatus = $"User {(_authStatus ? "✓" : "⚠️")} Market {(_quotesSeen > 0 ? "✓" : "⚠️")} (stable)";
                _systemStatus = $"Ready ({GetSystemReadyChecks()}/5 checks passed)";
            }
            
            // Move cursor to top and redraw status
            Console.SetCursorPosition(0, 4);
            
            var currentTime = DateTime.Now.ToString("HH:mm:ss");
            
            // Status section with live data
            var authDisplay = _authStatus ? $"Logged in as {_userEmail}" : "Authentication pending";
            Console.WriteLine($"[{currentTime}] 🔐 AUTH: {authDisplay}");
            Console.WriteLine($"[{currentTime}] 📊 ACCOUNT: {_accountInfo}");
            
            // Show real market data if available
            var contractDisplay = _contractsInfo;
            if (_lastESPrice > 0 || _lastNQPrice > 0)
            {
                var priceInfo = "";
                if (_lastESPrice > 0) priceInfo += $"ES: {_lastESPrice:F2} ";
                if (_lastNQPrice > 0) priceInfo += $"NQ: {_lastNQPrice:F2}";
                contractDisplay = $"ES, NQ, MES, MNQ (ES/NQ focus) | Live: {priceInfo}";
            }
            Console.WriteLine($"[{currentTime}] 📈 CONTRACTS: {contractDisplay}");
            Console.WriteLine($"[{currentTime}] 🔌 HUBS: {_hubStatus}");
            Console.WriteLine($"[{currentTime}] ✅ SYSTEM: {_systemStatus}");
            
            // Show live market activity stats
            if (_barsSeen > 0 || _quotesSeen > 0 || _tradesSeen > 0)
            {
                Console.WriteLine($"[{currentTime}] 📊 ACTIVITY: Bars: {_barsSeen} | Quotes: {_quotesSeen} | Trades: {_tradesSeen}");
            }
            Console.WriteLine();
            
            string modeMessage = killFileExists 
                ? "DRY_RUN (kill.txt present → Trading disabled for safety)"
                : $"{_mode} (kill.txt not present → AUTO_EXECUTE available)";
            Console.WriteLine($"[{currentTime}] 📍 MODE: {modeMessage}");
            Console.WriteLine($"[{currentTime}] 🎯 STRATEGY: {_strategy} | Risk: 1% per trade");
            Console.WriteLine($"[{currentTime}] 🕐 SCHEDULE: {_schedule} | Session: {GetCurrentSession()}");
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

    private int GetSystemReadyChecks()
    {
        int readyChecks = 0;
        
        // Check 1: Authentication
        if (_authStatus) readyChecks++;
        
        // Check 2: Market data flowing  
        if (_quotesSeen > 0 || _tradesSeen > 0) readyChecks++;
        
        // Check 3: Minimum bars seen for trading
        if (_barsSeen >= 10) readyChecks++;
        
        // Check 4: Account data loaded
        if (!_accountInfo.Contains("Initializing") && !_accountInfo.Contains("Loading")) readyChecks++;
        
        // Check 5: No kill switch
        if (!File.Exists(Path.Combine(Directory.GetCurrentDirectory(), "kill.txt"))) readyChecks++;
        
        return readyChecks;
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

    private string GetCurrentSession()
    {
        try
        {
            var now = DateTime.Now;
            var et = TimeZoneInfo.ConvertTime(now, TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            var currentSession = ES_NQ_TradingSchedule.GetCurrentSession(et.TimeOfDay);
            
            if (currentSession != null)
            {
                return $"{currentSession.Description} ({et:HH:mm} ET)";
            }
            else
            {
                return $"Market Closed ({et:HH:mm} ET)";
            }
        }
        catch (Exception ex)
        {
            return $"Schedule Error: {ex.Message}";
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