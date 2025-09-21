using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.BotCore.Services;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Enhanced Kill Switch Protection with three-level enforcement
/// Production-grade emergency stop system for institutional trading
/// </summary>
public class EnhancedKillSwitchService : IHostedService, IDisposable
{
    private readonly ILogger<EnhancedKillSwitchService> _logger;
    private readonly Timer _monitoringTimer;
    private volatile bool _disposed;
    
    // Three-level kill switch state
    private volatile bool _level1QuoteSubscriptionBlocked;
    private volatile bool _level2OrderPlacementBlocked;
    private volatile bool _level3FillAttributionBlocked;
    
    private const string KILL_FILE_NAME = "kill.txt";
    private const string LEVEL1_FILE_NAME = "kill_quotes.txt";
    private const string LEVEL2_FILE_NAME = "kill_orders.txt";
    private const string LEVEL3_FILE_NAME = "kill_fills.txt";
    private const int MONITORING_INTERVAL_MS = 1000; // Check every second

    public EnhancedKillSwitchService(ILogger<EnhancedKillSwitchService> logger)
    {
        _logger = logger;
        
        // Start with comprehensive check
        CheckAllKillSwitchLevels();
        
        // Set up continuous monitoring
        _monitoringTimer = new Timer(MonitorKillSwitchFiles, null, 
            TimeSpan.FromMilliseconds(MONITORING_INTERVAL_MS), 
            TimeSpan.FromMilliseconds(MONITORING_INTERVAL_MS));
            
        _logger.LogInformation("🛡️ [ENHANCED-KILL-SWITCH] Three-level kill switch monitoring started");
        _logger.LogInformation("📁 [ENHANCED-KILL-SWITCH] Monitoring files: {GeneralKill}, {Level1}, {Level2}, {Level3}",
            KILL_FILE_NAME, LEVEL1_FILE_NAME, LEVEL2_FILE_NAME, LEVEL3_FILE_NAME);
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🟢 [ENHANCED-KILL-SWITCH] Enhanced kill switch service started");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔴 [ENHANCED-KILL-SWITCH] Enhanced kill switch service stopped");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Level 1: Check if market data subscription should be blocked
    /// </summary>
    public bool IsQuoteSubscriptionAllowed()
    {
        if (_level1QuoteSubscriptionBlocked || IsGeneralKillSwitchActive())
        {
            _logger.LogDebug("🚫 [ENHANCED-KILL-SWITCH] Level 1: Quote subscription BLOCKED");
            return false;
        }
        return true;
    }

    /// <summary>
    /// Level 2: Check if order placement should be blocked
    /// </summary>
    public bool IsOrderPlacementAllowed()
    {
        if (_level2OrderPlacementBlocked || _level1QuoteSubscriptionBlocked || IsGeneralKillSwitchActive())
        {
            _logger.LogDebug("🚫 [ENHANCED-KILL-SWITCH] Level 2: Order placement BLOCKED");
            return false;
        }
        return true;
    }

    /// <summary>
    /// Level 3: Check if fill attribution should be blocked
    /// </summary>
    public bool IsFillAttributionAllowed()
    {
        if (_level3FillAttributionBlocked || _level2OrderPlacementBlocked || 
            _level1QuoteSubscriptionBlocked || IsGeneralKillSwitchActive())
        {
            _logger.LogDebug("🚫 [ENHANCED-KILL-SWITCH] Level 3: Fill attribution BLOCKED");
            return false;
        }
        return true;
    }

    /// <summary>
    /// Check general kill switch (backwards compatibility)
    /// </summary>
    public static bool IsGeneralKillSwitchActive()
    {
        return ProductionKillSwitchService.IsKillSwitchActive();
    }

    /// <summary>
    /// Get current kill switch status for all levels
    /// </summary>
    public EnhancedKillSwitchStatus GetStatus()
    {
        return new EnhancedKillSwitchStatus
        {
            GeneralKillSwitch = IsGeneralKillSwitchActive(),
            Level1QuotesBlocked = _level1QuoteSubscriptionBlocked,
            Level2OrdersBlocked = _level2OrderPlacementBlocked,
            Level3FillsBlocked = _level3FillAttributionBlocked,
            AnyLevelActive = IsGeneralKillSwitchActive() || _level1QuoteSubscriptionBlocked || 
                            _level2OrderPlacementBlocked || _level3FillAttributionBlocked,
            LastCheckedUtc = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Force activation of specific kill switch level (for emergency use)
    /// </summary>
    public void ForceKillSwitchLevel(int level, string reason)
    {
        switch (level)
        {
            case 1:
                _level1QuoteSubscriptionBlocked = true;
                File.WriteAllText(LEVEL1_FILE_NAME, $"Emergency activation: {reason}\nTime: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
                _logger.LogCritical("🔴 [ENHANCED-KILL-SWITCH] LEVEL 1 FORCE ACTIVATED: {Reason}", reason);
                break;
            case 2:
                _level2OrderPlacementBlocked = true;
                File.WriteAllText(LEVEL2_FILE_NAME, $"Emergency activation: {reason}\nTime: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
                _logger.LogCritical("🔴 [ENHANCED-KILL-SWITCH] LEVEL 2 FORCE ACTIVATED: {Reason}", reason);
                break;
            case 3:
                _level3FillAttributionBlocked = true;
                File.WriteAllText(LEVEL3_FILE_NAME, $"Emergency activation: {reason}\nTime: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
                _logger.LogCritical("🔴 [ENHANCED-KILL-SWITCH] LEVEL 3 FORCE ACTIVATED: {Reason}", reason);
                break;
            default:
                _logger.LogError("❌ [ENHANCED-KILL-SWITCH] Invalid kill switch level: {Level}", level);
                break;
        }

        // Force environment variables for additional safety
        Environment.SetEnvironmentVariable("DRY_RUN", "true");
        Environment.SetEnvironmentVariable("EXECUTE", "false");
        Environment.SetEnvironmentVariable("AUTO_EXECUTE", "false");
    }

    /// <summary>
    /// Monitor kill switch files continuously
    /// </summary>
    private void MonitorKillSwitchFiles(object? state)
    {
        if (_disposed) return;
        
        try
        {
            CheckAllKillSwitchLevels();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [ENHANCED-KILL-SWITCH] Error during kill switch monitoring");
        }
    }

    /// <summary>
    /// Check all kill switch levels
    /// </summary>
    private void CheckAllKillSwitchLevels()
    {
        // Check Level 1: Quote subscription blocking
        var newLevel1State = File.Exists(LEVEL1_FILE_NAME) || IsGeneralKillSwitchActive();
        if (newLevel1State != _level1QuoteSubscriptionBlocked)
        {
            _level1QuoteSubscriptionBlocked = newLevel1State;
            if (newLevel1State)
            {
                _logger.LogCritical("🔴 [ENHANCED-KILL-SWITCH] LEVEL 1 ACTIVATED: Quote subscription blocked");
                EnforceDryRunMode("Level 1 Kill Switch");
            }
            else
            {
                _logger.LogInformation("🟢 [ENHANCED-KILL-SWITCH] LEVEL 1 DEACTIVATED: Quote subscription allowed");
            }
        }

        // Check Level 2: Order placement blocking
        var newLevel2State = File.Exists(LEVEL2_FILE_NAME) || _level1QuoteSubscriptionBlocked;
        if (newLevel2State != _level2OrderPlacementBlocked)
        {
            _level2OrderPlacementBlocked = newLevel2State;
            if (newLevel2State)
            {
                _logger.LogCritical("🔴 [ENHANCED-KILL-SWITCH] LEVEL 2 ACTIVATED: Order placement blocked");
                EnforceDryRunMode("Level 2 Kill Switch");
            }
            else
            {
                _logger.LogInformation("🟢 [ENHANCED-KILL-SWITCH] LEVEL 2 DEACTIVATED: Order placement allowed");
            }
        }

        // Check Level 3: Fill attribution blocking
        var newLevel3State = File.Exists(LEVEL3_FILE_NAME) || _level2OrderPlacementBlocked;
        if (newLevel3State != _level3FillAttributionBlocked)
        {
            _level3FillAttributionBlocked = newLevel3State;
            if (newLevel3State)
            {
                _logger.LogCritical("🔴 [ENHANCED-KILL-SWITCH] LEVEL 3 ACTIVATED: Fill attribution blocked");
                EnforceDryRunMode("Level 3 Kill Switch");
            }
            else
            {
                _logger.LogInformation("🟢 [ENHANCED-KILL-SWITCH] LEVEL 3 DEACTIVATED: Fill attribution allowed");
            }
        }
    }

    /// <summary>
    /// Enforce DRY_RUN mode for safety
    /// </summary>
    private void EnforceDryRunMode(string reason)
    {
        try
        {
            Environment.SetEnvironmentVariable("DRY_RUN", "true");
            Environment.SetEnvironmentVariable("EXECUTE", "false");
            Environment.SetEnvironmentVariable("AUTO_EXECUTE", "false");
            
            _logger.LogCritical("🛡️ [ENHANCED-KILL-SWITCH] DRY_RUN MODE ENFORCED - Reason: {Reason}", reason);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [ENHANCED-KILL-SWITCH] Failed to enforce DRY_RUN mode");
        }
    }

    /// <summary>
    /// Check for VPN/remote desktop connections (additional safety)
    /// </summary>
    public bool IsRemoteConnectionDetected()
    {
        try
        {
            // Check for common VPN indicators
            var vpnIndicators = new[]
            {
                Environment.GetEnvironmentVariable("VPN_CONNECTED"),
                Environment.GetEnvironmentVariable("REMOTE_DESKTOP_SESSION"),
                Environment.GetEnvironmentVariable("CITRIX_SESSION"),
                Environment.GetEnvironmentVariable("RDP_SESSION")
            };

            var remoteDetected = vpnIndicators.Any(indicator => 
                !string.IsNullOrEmpty(indicator) && 
                (indicator.Equals("true", StringComparison.OrdinalIgnoreCase) || 
                 indicator.Equals("1", StringComparison.OrdinalIgnoreCase)));

            if (remoteDetected)
            {
                _logger.LogWarning("🌐 [ENHANCED-KILL-SWITCH] Remote connection detected - additional safety measures may apply");
            }

            return remoteDetected;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to check remote connection status");
            return false; // Default to false if check fails
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _monitoringTimer?.Dispose();
        _disposed = true;
        
        _logger.LogInformation("🗑️ [ENHANCED-KILL-SWITCH] Enhanced kill switch service disposed");
    }
}

/// <summary>
/// Enhanced kill switch status for monitoring
/// </summary>
public class EnhancedKillSwitchStatus
{
    public bool GeneralKillSwitch { get; set; }
    public bool Level1QuotesBlocked { get; set; }
    public bool Level2OrdersBlocked { get; set; }
    public bool Level3FillsBlocked { get; set; }
    public bool AnyLevelActive { get; set; }
    public DateTime LastCheckedUtc { get; set; }
    
    public string GetStatusSummary()
    {
        if (!AnyLevelActive)
            return "All systems operational";
            
        var activeFeatures = new List<string>();
        if (GeneralKillSwitch) activeFeatures.Add("General");
        if (Level1QuotesBlocked) activeFeatures.Add("L1-Quotes");
        if (Level2OrdersBlocked) activeFeatures.Add("L2-Orders");
        if (Level3FillsBlocked) activeFeatures.Add("L3-Fills");
        
        return $"ACTIVE: {string.Join(", ", activeFeatures)}";
    }
}