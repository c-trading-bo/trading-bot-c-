using System;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using BotCore.Services;

namespace TradingBot.References;

/// <summary>
/// Reference integration showing how to use SessionAwareRuntimeGates
/// for 24×5 ES/NQ futures trading with proper session awareness
/// </summary>
public class SessionAwareIntegrationReference
{
    private readonly SessionAwareRuntimeGates _sessionGates;
    private readonly ILogger<SessionAwareIntegrationReference> _logger;

    public SessionAwareIntegrationReference(
        SessionAwareRuntimeGates sessionGates,
        ILogger<SessionAwareIntegrationReference> logger)
    {
        _sessionGates = sessionGates;
        _logger = logger;
    }

    /// <summary>
    /// Reference of session-aware trading decision logic
    /// </summary>
    public async Task<bool> ShouldExecuteTradeAsync(string symbol, decimal entryPrice, string strategy)
    {
        try
        {
            // Check basic trading permission
            var isTradingAllowed = await _sessionGates.IsTradingAllowedAsync(symbol);
            if (!isTradingAllowed)
            {
                _logger.LogDebug("Trade blocked by session gates for {Symbol}", symbol);
                return false;
            }

            // Get detailed session status for enhanced decision making
            var sessionStatus = _sessionGates.GetSessionStatus();
            
            // Log session details for monitoring
            _logger.LogInformation(
                "Trade evaluation for {Symbol}: Session={Session}, RTH={IsRth}, ETH={IsEth}, TradingAllowed={TradingAllowed}",
                symbol, sessionStatus.CurrentSession, sessionStatus.IsRth, sessionStatus.IsEth, sessionStatus.TradingAllowed);

            // Reference: Different risk parameters for RTH vs ETH
            if (sessionStatus.IsRth)
            {
                // Regular trading hours - normal risk parameters
                _logger.LogDebug("RTH session: Using normal risk parameters for {Symbol}", symbol);
                return await ValidateRthTradeAsync(symbol, entryPrice, strategy);
            }
            else if (sessionStatus.IsEth)
            {
                // Extended hours - more conservative approach
                _logger.LogDebug("ETH session: Using conservative risk parameters for {Symbol}", symbol);
                return await ValidateEthTradeAsync(symbol, entryPrice, strategy);
            }

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in session-aware trade evaluation for {Symbol}", symbol);
            return false; // Fail safe
        }
    }

    /// <summary>
    /// Reference monitoring method for session changes
    /// </summary>
    public async Task MonitorSessionChangesAsync()
    {
        var sessionStatus = _sessionGates.GetSessionStatus();
        
        _logger.LogInformation(
            "Current Market Session Status: " +
            "Session={CurrentSession}, " +
            "TradingAllowed={TradingAllowed}, " +
            "NextChange={NextSessionChange}, " +
            "EasternTime={EasternTime:yyyy-MM-dd HH:mm:ss} ET",
            sessionStatus.CurrentSession,
            sessionStatus.TradingAllowed,
            sessionStatus.NextSessionChange?.ToString("yyyy-MM-dd HH:mm:ss") ?? "N/A",
            sessionStatus.EasternTime);

        // Alert on special conditions
        if (sessionStatus.IsSundayReopenCurb)
        {
            _logger.LogWarning("ALERT: Sunday reopen stabilization period active");
        }

        if (sessionStatus.IsEthFirstMinsCurb)
        {
            _logger.LogWarning("ALERT: ETH first minutes curb active");
        }

        if (sessionStatus.IsMaintenanceBreak)
        {
            _logger.LogInformation("INFO: Daily maintenance break (17:00-18:00 ET)");
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Reference session-specific strategy selection
    /// </summary>
    public string SelectStrategyBySession()
    {
        var sessionStatus = _sessionGates.GetSessionStatus();
        
        return sessionStatus.CurrentSession switch
        {
            "RTH" => "AggressiveScalping",     // Higher volume, tighter spreads
            "ETH" => "ConservativeTrend",      // Lower volume, wider spreads
            "MAINTENANCE" => "NoTrading",      // Market closed
            "CLOSED" => "NoTrading",           // Weekend
            _ => "ConservativeTrend"           // Default fallback
        };
    }

    #region Private Helper Methods

    private async Task<bool> ValidateRthTradeAsync(string symbol, decimal entryPrice, string strategy)
    {
        await Task.CompletedTask;
        
        // RTH validation logic (normal risk parameters)
        // - Standard position sizing
        // - Normal volatility expectations
        // - Higher liquidity assumptions
        
        _logger.LogDebug("RTH validation passed for {Symbol} at {Price} using {Strategy}", 
            symbol, entryPrice, strategy);
        
        return true; // Simplified for reference
    }

    private async Task<bool> ValidateEthTradeAsync(string symbol, decimal entryPrice, string strategy)
    {
        await Task.CompletedTask;
        
        // ETH validation logic (conservative risk parameters)
        // - Reduced position sizing
        // - Higher volatility expectations  
        // - Lower liquidity assumptions
        // - Wider stop losses
        
        _logger.LogDebug("ETH validation passed for {Symbol} at {Price} using {Strategy}", 
            symbol, entryPrice, strategy);
        
        return true; // Simplified for reference
    }

    #endregion
}

/// <summary>
/// Extension method for easy integration into existing trading systems
/// </summary>
public static class SessionAwareExtensions
{
    /// <summary>
    /// Add session awareness to service collection
    /// </summary>
    public static IServiceCollection AddSessionAwareRuntimeGates(this IServiceCollection services)
    {
        services.AddSingleton<SessionAwareRuntimeGates>();
        services.AddSingleton<SessionAwareIntegrationReference>();
        return services;
    }

    /// <summary>
    /// Quick session check extension method
    /// </summary>
    public static async Task<bool> IsCurrentlyTradingTimeAsync(this SessionAwareRuntimeGates gates, string symbol = "ES")
    {
        return await gates.IsTradingAllowedAsync(symbol);
    }

    /// <summary>
    /// Get simple session name for logging
    /// </summary>
    public static string GetCurrentSessionName(this SessionAwareRuntimeGates gates)
    {
        return gates.GetCurrentSession();
    }
}