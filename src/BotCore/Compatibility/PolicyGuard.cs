using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
    
    public PolicyGuard(ILogger<PolicyGuard> logger, StructuredConfigurationManager configManager)
    {
        _logger = logger;
        
        // Load configuration from the main configuration
        var mainConfig = configManager.LoadMainConfiguration();
        _config = new PolicyGuardConfig
        {
            Environment = mainConfig.Environment.Mode,
            AuthorizedSymbols = mainConfig.Environment.AuthorizedSymbols,
            KillSwitchEnabled = mainConfig.Environment.KillSwitchEnabled,
            TradingHours = new Dictionary<string, string>
            {
                ["Start"] = "09:30",
                ["End"] = "16:00",
                ["Timezone"] = "EST"
            }
        };
        
        InitializePolicyRules();
        
        _logger.LogInformation("PolicyGuard initialized with {RuleCount} policy rules for environment {Environment}", 
            _policyRules.Count, _config.Environment);
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
        // Environment detection blocking unauthorized live trading
        var environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Development";
        var configEnvironment = _config.Environment?.ToLowerInvariant() ?? "development";
        
        // Check for ALLOW_TOPSTEP_LIVE override - coordinated with ProductionRuleEnforcementAnalyzer
        var allowTopstepLive = Environment.GetEnvironmentVariable("ALLOW_TOPSTEP_LIVE");
        var isTopstepLiveAllowed = string.Equals(allowTopstepLive, "true", StringComparison.OrdinalIgnoreCase);
        
        _logger.LogDebug("Environment safety check: Config={ConfigEnv}, AspNetCore={AspNetCoreEnv}, ALLOW_TOPSTEP_LIVE={AllowLive}", 
            configEnvironment, environment, allowTopstepLive ?? "null");
        
        // Block trading if not in production mode (unless explicitly overridden)
        if (configEnvironment != "production" && !isTopstepLiveAllowed)
        {
            return new PolicyCheckResult
            {
                IsAuthorized = false,
                Reason = $"Trading blocked in {configEnvironment} environment. Use 'production' mode or set ALLOW_TOPSTEP_LIVE=true for live trading."
            };
        }
        
        // Additional safety checks (can be overridden with ALLOW_TOPSTEP_LIVE)
        var isDevelopment = System.Diagnostics.Debugger.IsAttached ||
                           environment.Contains("Dev", StringComparison.OrdinalIgnoreCase) ||
                           environment.Contains("Test", StringComparison.OrdinalIgnoreCase);
        
        if (isDevelopment && !isTopstepLiveAllowed)
        {
            return new PolicyCheckResult
            {
                IsAuthorized = false,
                Reason = $"Development environment detected: {environment}. Debugger attached: {System.Diagnostics.Debugger.IsAttached}. Set ALLOW_TOPSTEP_LIVE=true to override."
            };
        }
        
        // Log successful authorization for audit trail
        if (isTopstepLiveAllowed)
        {
            _logger.LogWarning("⚠️ Live trading authorized via ALLOW_TOPSTEP_LIVE override in {Environment} environment", environment);
        }
        
        // Check kill switch
        if (_config.KillSwitchEnabled && File.Exists("kill.txt"))
        {
            return new PolicyCheckResult
            {
                IsAuthorized = false,
                Reason = "Kill switch activated (kill.txt file found)"
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
        // Kill switch integration - moved to CheckEnvironmentSafety for consolidation
        // This method kept for backward compatibility
        return new PolicyCheckResult { IsAuthorized = true };
    }
    
    /// <summary>
    /// Get unified safety dashboard showing status of all safety mechanisms
    /// Coordinates with ProductionRuleEnforcementAnalyzer for comprehensive coverage
    /// </summary>
    public SafetyDashboard GetUnifiedSafetyDashboard()
    {
        var dashboard = new SafetyDashboard
        {
            Timestamp = DateTime.UtcNow,
            PolicyGuardStatus = GetPolicyGuardStatus(),
            EnvironmentChecks = GetEnvironmentStatus(),
            SafetyHierarchy = GetSafetyHierarchy()
        };
        
        _logger.LogInformation("Generated unified safety dashboard: {PolicyGuardActive} policy rules active, Environment: {Environment}", 
            _policyRules.Count(r => r.Value.IsActive), _config.Environment);
        
        return dashboard;
    }
    
    private PolicyGuardStatus GetPolicyGuardStatus()
    {
        return new PolicyGuardStatus
        {
            IsActive = true,
            ActiveRules = _policyRules.Where(r => r.Value.IsActive).Select(r => r.Value.Name).ToList(),
            InactiveRules = _policyRules.Where(r => !r.Value.IsActive).Select(r => r.Value.Name).ToList(),
            Configuration = new Dictionary<string, object>
            {
                ["Environment"] = _config.Environment,
                ["KillSwitchEnabled"] = _config.KillSwitchEnabled,
                ["AuthorizedSymbols"] = _config.AuthorizedSymbols,
                ["EnforceTradingHours"] = _config.EnforceTradingHours
            }
        };
    }
    
    private EnvironmentStatus GetEnvironmentStatus()
    {
        var aspNetCoreEnv = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Development";
        var allowTopstepLive = Environment.GetEnvironmentVariable("ALLOW_TOPSTEP_LIVE");
        var isDebuggerAttached = System.Diagnostics.Debugger.IsAttached;
        
        return new EnvironmentStatus
        {
            AspNetCoreEnvironment = aspNetCoreEnv,
            ConfiguredEnvironment = _config.Environment,
            AllowTopstepLive = allowTopstepLive,
            IsDebuggerAttached = isDebuggerAttached,
            KillSwitchActive = _config.KillSwitchEnabled && File.Exists("kill.txt"),
            IsProductionReady = _config.Environment == "production" || 
                               string.Equals(allowTopstepLive, "true", StringComparison.OrdinalIgnoreCase)
        };
    }
    
    private SafetyHierarchy GetSafetyHierarchy()
    {
        return new SafetyHierarchy
        {
            Description = "Safety system precedence and coordination",
            Levels = new List<SafetyLevel>
            {
                new SafetyLevel 
                { 
                    Priority = 1, 
                    Name = "Kill Switch", 
                    Description = "Emergency stop mechanism (kill.txt file)",
                    IsActive = _config.KillSwitchEnabled
                },
                new SafetyLevel 
                { 
                    Priority = 2, 
                    Name = "Environment Detection", 
                    Description = "ALLOW_TOPSTEP_LIVE environment variable override",
                    IsActive = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ALLOW_TOPSTEP_LIVE"))
                },
                new SafetyLevel 
                { 
                    Priority = 3, 
                    Name = "PolicyGuard Rules", 
                    Description = "Environment checks, symbol authorization, trading hours",
                    IsActive = _policyRules.Any(r => r.Value.IsActive)
                },
                new SafetyLevel 
                { 
                    Priority = 4, 
                    Name = "ProductionRuleEnforcementAnalyzer", 
                    Description = "Compile-time business rule validation (external)",
                    IsActive = true // Always active during compilation
                }
            }
        };
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
    public string Environment { get; set; } = "development";
    public List<string> AuthorizedSymbols { get; set; } = new() { "ES", "NQ", "MES", "MNQ" };
    public bool KillSwitchEnabled { get; set; } = true;
    public Dictionary<string, string> TradingHours { get; set; } = new();
    public bool EnforceTradingHours { get; set; } = false;
    public int TradingStartHour { get; set; } = 14; // 2 PM UTC (9 AM EST)
    public int TradingEndHour { get; set; } = 21;   // 9 PM UTC (4 PM EST)
}

/// <summary>
/// Unified safety dashboard showing status of all safety mechanisms
/// </summary>
public class SafetyDashboard
{
    public DateTime Timestamp { get; set; }
    public PolicyGuardStatus PolicyGuardStatus { get; set; } = new();
    public EnvironmentStatus EnvironmentChecks { get; set; } = new();
    public SafetyHierarchy SafetyHierarchy { get; set; } = new();
}

/// <summary>
/// PolicyGuard status information
/// </summary>
public class PolicyGuardStatus
{
    public bool IsActive { get; set; }
    public List<string> ActiveRules { get; set; } = new();
    public List<string> InactiveRules { get; set; } = new();
    public Dictionary<string, object> Configuration { get; set; } = new();
}

/// <summary>
/// Environment status information
/// </summary>
public class EnvironmentStatus
{
    public string AspNetCoreEnvironment { get; set; } = string.Empty;
    public string ConfiguredEnvironment { get; set; } = string.Empty;
    public string? AllowTopstepLive { get; set; }
    public bool IsDebuggerAttached { get; set; }
    public bool KillSwitchActive { get; set; }
    public bool IsProductionReady { get; set; }
}

/// <summary>
/// Safety hierarchy showing precedence between safety systems
/// </summary>
public class SafetyHierarchy
{
    public string Description { get; set; } = string.Empty;
    public List<SafetyLevel> Levels { get; set; } = new();
}

/// <summary>
/// Individual safety level in the hierarchy
/// </summary>
public class SafetyLevel
{
    public int Priority { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public bool IsActive { get; set; }
}