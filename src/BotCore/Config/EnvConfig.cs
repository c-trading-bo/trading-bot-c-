using System;
using System.Collections;
using System.IO;
using Microsoft.Extensions.Logging;

namespace BotCore.Config;

/// <summary>
/// Environment configuration manager with OS env precedence over .env files
/// Implements AUTO_EXECUTE prechecks and BOT_QUICK_EXIT logic
/// </summary>
public static class EnvConfig
{
    private static readonly ILogger _logger = CreateLogger();
    private static readonly Dictionary<string, string> _envCache = new();
    private static readonly object _lock = new();
    
    static EnvConfig()
    {
        LoadEnvironmentVariables();
    }

    /// <summary>
    /// Load environment variables with OS env precedence over .env files
    /// </summary>
    private static void LoadEnvironmentVariables()
    {
        lock (_lock)
        {
            _envCache.Clear();

            // Load from .env file first (lower precedence)
            LoadFromEnvFile();

            // Load from OS environment variables (higher precedence - overwrites .env)
            foreach (DictionaryEntry entry in Environment.GetEnvironmentVariables())
            {
                var key = entry.Key?.ToString();
                var value = entry.Value?.ToString();
                if (!string.IsNullOrEmpty(key) && value != null)
                {
                    _envCache[key] = value;
                }
            }

            _logger.LogDebug("Environment configuration loaded with {Count} variables", _envCache.Count);
        }
    }

    /// <summary>
    /// Load variables from .env file (fills missing keys only)
    /// </summary>
    private static void LoadFromEnvFile()
    {
        var envFilePaths = new[] { ".env", ".env.local", ".env.sample.local" };
        
        foreach (var envPath in envFilePaths)
        {
            if (File.Exists(envPath))
            {
                try
                {
                    var lines = File.ReadAllLines(envPath);
                    foreach (var line in lines)
                    {
                        var trimmed = line.Trim();
                        if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#"))
                            continue;

                        var separatorIndex = trimmed.IndexOf('=');
                        if (separatorIndex > 0)
                        {
                            var key = trimmed[..separatorIndex].Trim();
                            var value = trimmed[(separatorIndex + 1)..].Trim().Trim('"');
                            
                            // Only set if not already in cache (OS env has precedence)
                            if (!_envCache.ContainsKey(key))
                            {
                                _envCache[key] = value;
                            }
                        }
                    }
                    _logger.LogDebug("Loaded environment variables from {EnvFile}", envPath);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to load .env file: {EnvFile}", envPath);
                }
            }
        }
    }

    /// <summary>
    /// Get environment variable with fallback
    /// </summary>
    public static string Get(string key, string defaultValue = "")
    {
        lock (_lock)
        {
            return _envCache.GetValueOrDefault(key, defaultValue);
        }
    }

    /// <summary>
    /// Get boolean environment variable
    /// </summary>
    public static bool GetBool(string key, bool defaultValue = false)
    {
        var value = Get(key);
        if (string.IsNullOrEmpty(value))
            return defaultValue;

        return value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
               value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
               value.Equals("yes", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Get integer environment variable
    /// </summary>
    public static int GetInt(string key, int defaultValue = 0)
    {
        var value = Get(key);
        return int.TryParse(value, out var result) ? result : defaultValue;
    }

    /// <summary>
    /// Get decimal environment variable
    /// </summary>
    public static decimal GetDecimal(string key, decimal defaultValue = 0m)
    {
        var value = Get(key);
        return decimal.TryParse(value, out var result) ? result : defaultValue;
    }

    /// <summary>
    /// Check if kill.txt exists (forces DRY_RUN mode)
    /// </summary>
    public static bool IsKillFilePresent()
    {
        return File.Exists("kill.txt");
    }

    /// <summary>
    /// Determine if system is in AUTO_EXECUTE mode based on prechecks
    /// </summary>
    public static ExecutionMode GetExecutionMode(ExecutionContext context)
    {
        try
        {
            // kill.txt always forces DRY_RUN
            if (IsKillFilePresent())
            {
                _logger.LogWarning("kill.txt detected - forcing DRY_RUN mode");
                return ExecutionMode.DryRun;
            }

            // Check EXECUTE environment variable
            var executeFlag = GetBool("EXECUTE", false);
            if (!executeFlag)
            {
                _logger.LogInformation("EXECUTE=false or not set - using DRY_RUN mode");
                return ExecutionMode.DryRun;
            }

            // AUTO_EXECUTE prechecks
            var prechecksResult = ValidateAutoExecutePrechecks(context);
            if (!prechecksResult.IsValid)
            {
                _logger.LogWarning("AUTO_EXECUTE prechecks failed: {Reason} - forcing DRY_RUN", 
                    prechecksResult.FailureReason);
                
                var prechecksData = new
                {
                    timestamp = DateTime.UtcNow,
                    component = "env_config",
                    operation = "auto_execute_prechecks",
                    success = false,
                    reason = prechecksResult.FailureReason,
                    bars_seen = context.BarsSeen,
                    hubs_connected = context.HubsConnected,
                    can_trade = context.CanTrade,
                    contract_id_present = !string.IsNullOrEmpty(context.ContractId)
                };

                _logger.LogError("PRECHECK_FAILED: {PrechecksData}", 
                    System.Text.Json.JsonSerializer.Serialize(prechecksData));

                return ExecutionMode.DryRun;
            }

            _logger.LogInformation("All AUTO_EXECUTE prechecks passed - enabling live trading");
            return ExecutionMode.AutoExecute;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error determining execution mode - defaulting to DRY_RUN");
            return ExecutionMode.DryRun;
        }
    }

    /// <summary>
    /// Check BOT_QUICK_EXIT logic
    /// </summary>
    public static bool ShouldQuickExit()
    {
        try
        {
            var quickExit = GetBool("BOT_QUICK_EXIT", false);
            if (quickExit)
            {
                _logger.LogInformation("BOT_QUICK_EXIT=1 detected - quick exit enabled");
                
                var exitData = new
                {
                    timestamp = DateTime.UtcNow,
                    component = "env_config",
                    operation = "quick_exit_check",
                    quick_exit_enabled = true
                };

                _logger.LogInformation("QUICK_EXIT_ENABLED: {ExitData}", 
                    System.Text.Json.JsonSerializer.Serialize(exitData));
            }

            return quickExit;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking BOT_QUICK_EXIT - defaulting to false");
            return false;
        }
    }

    /// <summary>
    /// Validate AUTO_EXECUTE prechecks: BarsSeen ≥ 10, hubs connected, canTrade==true, contractId present
    /// </summary>
    private static PrecheckResult ValidateAutoExecutePrechecks(ExecutionContext context)
    {
        // Check BarsSeen ≥ 10
        if (context.BarsSeen < 10)
        {
            return new PrecheckResult 
            { 
                IsValid = false, 
                FailureReason = $"BarsSeen ({context.BarsSeen}) < 10" 
            };
        }

        // Check hubs connected
        if (!context.HubsConnected)
        {
            return new PrecheckResult 
            { 
                IsValid = false, 
                FailureReason = "SignalR hubs not connected" 
            };
        }

        // Check canTrade flag
        if (!context.CanTrade)
        {
            return new PrecheckResult 
            { 
                IsValid = false, 
                FailureReason = "canTrade flag is false" 
            };
        }

        // Check contractId present
        if (string.IsNullOrEmpty(context.ContractId))
        {
            return new PrecheckResult 
            { 
                IsValid = false, 
                FailureReason = "contractId is null or empty" 
            };
        }

        return new PrecheckResult { IsValid = true };
    }

    /// <summary>
    /// Reload environment configuration
    /// </summary>
    public static void Reload()
    {
        _logger.LogInformation("Reloading environment configuration...");
        LoadEnvironmentVariables();
    }

    /// <summary>
    /// Get all environment variables for debugging (with secrets redacted)
    /// </summary>
    public static Dictionary<string, string> GetAllRedacted()
    {
        var secretKeys = new[] { "TOKEN", "PASSWORD", "SECRET", "KEY", "AUTH", "JWT" };
        
        lock (_lock)
        {
            return _envCache.ToDictionary(
                kvp => kvp.Key,
                kvp => secretKeys.Any(secret => kvp.Key.Contains(secret, StringComparison.OrdinalIgnoreCase)) 
                    ? "[REDACTED]" 
                    : kvp.Value
            );
        }
    }

    private static ILogger CreateLogger()
    {
        // Create a minimal logger for bootstrap
        using var factory = LoggerFactory.Create(builder => builder.AddConsole());
        return factory.CreateLogger("EnvConfig");
    }
}

public enum ExecutionMode
{
    DryRun,
    AutoExecute
}

public class ExecutionContext
{
    public int BarsSeen { get; set; }
    public bool HubsConnected { get; set; }
    public bool CanTrade { get; set; }
    public string ContractId { get; set; } = "";
}

public class PrecheckResult
{
    public bool IsValid { get; set; }
    public string FailureReason { get; set; } = "";
}