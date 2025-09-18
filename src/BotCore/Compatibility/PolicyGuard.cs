using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace BotCore.Compatibility;

/// <summary>
/// Policy guard that adds environment-based protection working alongside 
/// your existing ProductionRuleEnforcementAnalyzer
/// 
/// Safety Layer Enhancement: Provides defense in depth with both rule-based 
/// validation and environment detection blocking unauthorized live trading.
/// </summary>
public class PolicyGuard
{
    private readonly ILogger<PolicyGuard> _logger;
    private readonly PolicyGuardConfig _config;
    private readonly Dictionary<string, PolicyRule> _policyRules = new();
    
    public PolicyGuard(ILogger<PolicyGuard> logger, PolicyGuardConfig config)
    {
        _logger = logger;
        _config = config;
        
        InitializePolicyRules();
        
        _logger.LogInformation("PolicyGuard initialized with {RuleCount} policy rules", 
            _policyRules.Count);
    }
    
    private void InitializePolicyRules()
    {
        // Environment-based rules
        _policyRules["environment_check"] = new PolicyRule
        {
            Name = "Environment Check",
            Description = "Block live trading in development/test environments",
            IsActive = true,
            CheckFunction = CheckEnvironmentSafety
        };
        
        // Symbol-based rules
        _policyRules["authorized_symbols"] = new PolicyRule
        {
            Name = "Authorized Symbols",
            Description = "Only allow trading on approved symbols",
            IsActive = true,
            CheckFunction = CheckAuthorizedSymbols
        };
        
        // Time-based rules
        _policyRules["trading_hours"] = new PolicyRule
        {
            Name = "Trading Hours",
            Description = "Enforce trading hour restrictions",
            IsActive = _config.EnforceTradingHours,
            CheckFunction = CheckTradingHours
        };
        
        // Kill switch rule
        _policyRules["kill_switch"] = new PolicyRule
        {
            Name = "Kill Switch",
            Description = "Emergency stop mechanism",
            IsActive = true,
            CheckFunction = CheckKillSwitch
        };
    }
    
    /// <summary>
    /// Check if trading is authorized for the given symbol
    /// </summary>
    public async Task<bool> IsAuthorizedForTradingAsync(
        string symbol, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            foreach (var rule in _policyRules.Values)
            {
                if (!rule.IsActive) continue;
                
                var result = await rule.CheckFunction(symbol, cancellationToken);
                if (!result.IsAuthorized)
                {
                    _logger.LogWarning("PolicyGuard blocked trading for {Symbol}: {Rule} - {Reason}", 
                        symbol, rule.Name, result.Reason);
                    return false;
                }
            }
            
            _logger.LogDebug("PolicyGuard authorized trading for {Symbol}", symbol);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in policy guard authorization check for {Symbol}", symbol);
            return false; // Fail safe
        }
    }
    
    private async Task<PolicyCheckResult> CheckEnvironmentSafety(
        string symbol, 
        CancellationToken cancellationToken)
    {
        // Check environment variables
        var environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Development";
        var isProduction = environment.Equals("Production", StringComparison.OrdinalIgnoreCase);
        
        // Check for debug/development indicators
        var isDevelopment = System.Diagnostics.Debugger.IsAttached ||
                           environment.Contains("Dev", StringComparison.OrdinalIgnoreCase) ||
                           environment.Contains("Test", StringComparison.OrdinalIgnoreCase);
        
        if (isDevelopment && _config.BlockDevelopmentTrading)
        {
            return new PolicyCheckResult
            {
                IsAuthorized = false,
                Reason = $"Development environment detected: {environment}"
            };
        }
        
        return new PolicyCheckResult { IsAuthorized = true };
    }
    
    private async Task<PolicyCheckResult> CheckAuthorizedSymbols(
        string symbol, 
        CancellationToken cancellationToken)
    {
        if (_config.AuthorizedSymbols?.Count > 0)
        {
            var isAuthorized = _config.AuthorizedSymbols.Contains(symbol);
            if (!isAuthorized)
            {
                return new PolicyCheckResult
                {
                    IsAuthorized = false,
                    Reason = $"Symbol {symbol} not in authorized list"
                };
            }
        }
        
        return new PolicyCheckResult { IsAuthorized = true };
    }
    
    private async Task<PolicyCheckResult> CheckTradingHours(
        string symbol, 
        CancellationToken cancellationToken)
    {
        var now = DateTime.UtcNow;
        var currentHour = now.Hour;
        
        // Basic trading hours check (can be enhanced with market-specific hours)
        var isWithinTradingHours = currentHour >= _config.TradingStartHour && 
                                  currentHour < _config.TradingEndHour;
        
        if (!isWithinTradingHours)
        {
            return new PolicyCheckResult
            {
                IsAuthorized = false,
                Reason = $"Outside trading hours: {currentHour}:00 UTC"
            };
        }
        
        return new PolicyCheckResult { IsAuthorized = true };
    }
    
    private async Task<PolicyCheckResult> CheckKillSwitch(
        string symbol, 
        CancellationToken cancellationToken)
    {
        // Check for kill.txt file
        var killSwitchPath = Path.Combine(Directory.GetCurrentDirectory(), "kill.txt");
        if (File.Exists(killSwitchPath))
        {
            return new PolicyCheckResult
            {
                IsAuthorized = false,
                Reason = "Kill switch activated (kill.txt exists)"
            };
        }
        
        // Check emergency stop flag
        if (_config.EmergencyStop)
        {
            return new PolicyCheckResult
            {
                IsAuthorized = false,
                Reason = "Emergency stop flag activated"
            };
        }
        
        return new PolicyCheckResult { IsAuthorized = true };
    }
}

/// <summary>
/// Policy rule definition
/// </summary>
public class PolicyRule
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public bool IsActive { get; set; }
    public Func<string, CancellationToken, Task<PolicyCheckResult>> CheckFunction { get; set; } = null!;
}

/// <summary>
/// Result of a policy check
/// </summary>
public class PolicyCheckResult
{
    public bool IsAuthorized { get; set; }
    public string Reason { get; set; } = string.Empty;
}

/// <summary>
/// Configuration for policy guard
/// </summary>
public class PolicyGuardConfig
{
    public bool BlockDevelopmentTrading { get; set; } = true;
    public List<string> AuthorizedSymbols { get; set; } = new() { "ES", "NQ", "MES", "MNQ" };
    public bool EnforceTradingHours { get; set; } = false;
    public int TradingStartHour { get; set; } = 14; // 2 PM UTC (9 AM EST)
    public int TradingEndHour { get; set; } = 21;   // 9 PM UTC (4 PM EST)
    public bool EmergencyStop { get; set; } = false;
}