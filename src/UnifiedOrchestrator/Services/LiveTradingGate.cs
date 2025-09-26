using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Gate interface for controlling live trading access
/// </summary>
internal interface ILiveTradingGate
{
    bool IsLiveTradingAllowed { get; }
    Task<bool> CheckLiveTradingPermissionAsync(CancellationToken cancellationToken = default);
    void EnableLiveTrading();
    void DisableLiveTrading();
}

/// <summary>
/// Live trading gate implementation with multiple safety checks
/// </summary>
internal sealed class LiveTradingGate : ILiveTradingGate
{
    private readonly ILogger<LiveTradingGate> _logger;
    private readonly IConfiguration _configuration;
    private volatile bool _liveTradingEnabled;
    
    public LiveTradingGate(ILogger<LiveTradingGate> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        
        // Default to disabled, must be explicitly enabled
        _liveTradingEnabled = false;
        
        // Check configuration for explicit enabling
        var liveOrders = _configuration.GetValue("LIVE_ORDERS", "0");
        var allowTopstepLive = _configuration.GetValue("ALLOW_TOPSTEP_LIVE", "0");
        var instantAllowLive = _configuration.GetValue("INSTANT_ALLOW_LIVE", "0");
        
        if (liveOrders == "1" && allowTopstepLive == "1" && instantAllowLive == "1")
        {
            _liveTradingEnabled = true;
            _logger.LogWarning("Live trading enabled via configuration");
        }
    }
    
    public bool IsLiveTradingAllowed => _liveTradingEnabled && !IsKillSwitchActive();
    
    public Task<bool> CheckLiveTradingPermissionAsync(CancellationToken cancellationToken = default)
    {
        // Check kill switch first
        if (IsKillSwitchActive())
        {
            _logger.LogWarning("Live trading blocked by kill switch");
            return Task.FromResult(false);
        }
        
        // Check DRY_RUN environment variable
        if (IsDryRunForced())
        {
            _logger.LogInformation("Live trading blocked by DRY_RUN mode");
            return Task.FromResult(false);
        }
        
        return Task.FromResult(_liveTradingEnabled);
    }
    
    public void EnableLiveTrading()
    {
        if (IsKillSwitchActive())
        {
            _logger.LogError("Cannot enable live trading: kill switch is active");
            return;
        }
        
        _liveTradingEnabled = true;
        _logger.LogWarning("Live trading enabled manually");
    }
    
    public void DisableLiveTrading()
    {
        _liveTradingEnabled = false;
        _logger.LogInformation("Live trading disabled");
    }
    
    private bool IsKillSwitchActive()
    {
        return System.IO.File.Exists("state/kill.txt");
    }
    
    private bool IsDryRunForced()
    {
        var dryRun = _configuration.GetValue("DRY_RUN", "1");
        var enableDryRun = _configuration.GetValue("ENABLE_DRY_RUN", "true");
        
        return dryRun == "1" || enableDryRun.Equals("true", StringComparison.OrdinalIgnoreCase);
    }
}