using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace BotCore.Compatibility;

/// <summary>
/// Structured configuration manager that replaces hardcoded values with learnable parameter bundles
/// 
/// Configuration System Enhancement: Add structured JSON configuration files that replace 
/// your hardcoded values with learnable parameter bundles. Your existing business logic 
/// validation can work alongside the new bundle-based system.
/// </summary>
public class StructuredConfigurationManager
{
    private readonly ILogger<StructuredConfigurationManager> _logger;
    private readonly List<string> _configPaths;
    private readonly Dictionary<string, StrategyConfiguration> _strategyConfigs = new();
    private readonly object _configLock = new();
    
    public StructuredConfigurationManager(ILogger<StructuredConfigurationManager> logger)
    {
        _logger = logger;
        _configPaths = new List<string>
        {
            "config/compatibility-kit.json",
            "config/bundles.stage.json",
            "config/bounds.json",
            "config/strategies/S2.json",
            "config/strategies/S3.json",
            "config/strategies/S6.json",
            "config/strategies/S11.json"
        };
        
        LoadConfigurationsAsync().GetAwaiter().GetResult();
        
        _logger.LogInformation("StructuredConfigurationManager loaded configurations from {PathCount} paths", 
            _configPaths.Count);
    }
    
    /// <summary>
    /// Load main compatibility kit configuration with schema validation
    /// </summary>
    public CompatibilityKitConfiguration LoadMainConfiguration()
    {
        try
        {
            var configPath = "config/compatibility-kit.json";
            if (!File.Exists(configPath))
            {
                throw new FileNotFoundException($"Main configuration file not found: {configPath}");
            }
            
            var jsonContent = File.ReadAllText(configPath);
            var config = JsonSerializer.Deserialize<CompatibilityKitConfiguration>(jsonContent);
            
            // Schema validation
            if (config == null)
            {
                throw new InvalidOperationException("Failed to deserialize main configuration");
            }
            
            if (config.Environment == null)
            {
                throw new InvalidOperationException("Environment configuration is required");
            }
            
            if (config.BundleSelection == null)
            {
                throw new InvalidOperationException("BundleSelection configuration is required");
            }
            
            _logger.LogInformation("✅ Main configuration loaded and validated from {ConfigPath}", configPath);
            return config;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to load main configuration");
            throw;
        }
    }
    
    /// <summary>
    /// Get configuration-driven parameters for a strategy
    /// </summary>
    public async Task<ConfigurationSource> GetParametersForStrategyAsync(
        string strategy,
        CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_configLock)
            {
                if (_strategyConfigs.TryGetValue(strategy, out var config))
                {
                    return new ConfigurationSource
                    {
                        Strategy = strategy,
                        BaseParameters = config.BaseParameters,
                        RiskParameters = config.RiskParameters,
                        MarketConditionOverrides = config.MarketConditionOverrides,
                        LoadedFrom = config.LoadedFrom,
                        LastUpdated = config.LastUpdated
                    };
                }
            }
            
            // Return default configuration if not found
            return CreateDefaultConfigurationSource(strategy);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting parameters for strategy {Strategy}", strategy);
            return CreateDefaultConfigurationSource(strategy);
        }
    }
    
    /// <summary>
    /// Load parameter bundles configuration for stage-based learning
    /// </summary>
    public ParameterBundlesConfiguration LoadParameterBundles()
    {
        try
        {
            var configPath = "config/bundles.stage.json";
            if (!File.Exists(configPath))
            {
                throw new FileNotFoundException($"Parameter bundles configuration file not found: {configPath}");
            }
            
            var jsonContent = File.ReadAllText(configPath);
            var config = JsonSerializer.Deserialize<ParameterBundlesConfiguration>(jsonContent);
            
            if (config == null)
            {
                throw new InvalidOperationException("Failed to deserialize parameter bundles configuration");
            }
            
            _logger.LogInformation("✅ Parameter bundles loaded from {ConfigPath}", configPath);
            return config;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to load parameter bundles configuration");
            throw;
        }
    }
    
    /// <summary>
    /// Load parameter bounds configuration for validation
    /// </summary>
    public ParameterBoundsConfiguration LoadParameterBounds()
    {
        try
        {
            var configPath = "config/bounds.json";
            if (!File.Exists(configPath))
            {
                throw new FileNotFoundException($"Parameter bounds configuration file not found: {configPath}");
            }
            
            var jsonContent = File.ReadAllText(configPath);
            var config = JsonSerializer.Deserialize<ParameterBoundsConfiguration>(jsonContent);
            
            if (config == null)
            {
                throw new InvalidOperationException("Failed to deserialize parameter bounds configuration");
            }
            
            _logger.LogInformation("✅ Parameter bounds loaded from {ConfigPath}", configPath);
            return config;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to load parameter bounds configuration");
            throw;
        }
    }
    
    /// <summary>
    /// Load all strategy configurations from JSON files
    /// </summary>
    private async Task LoadConfigurationsAsync()
    {
        try
        {
            foreach (var configPath in _configPaths)
            {
                if (Directory.Exists(configPath))
                {
                    await LoadConfigurationsFromDirectoryAsync(configPath);
                }
                else if (File.Exists(configPath))
                {
                    await LoadConfigurationFromFileAsync(configPath);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading configurations");
        }
    }
    
    private async Task LoadConfigurationsFromDirectoryAsync(string directoryPath)
    {
        var jsonFiles = Directory.GetFiles(directoryPath, "*.json", SearchOption.AllDirectories);
        
        foreach (var jsonFile in jsonFiles)
        {
            await LoadConfigurationFromFileAsync(jsonFile);
        }
    }
    
    private async Task LoadConfigurationFromFileAsync(string filePath)
    {
        try
        {
            var json = await File.ReadAllTextAsync(filePath);
            var config = JsonSerializer.Deserialize<StrategyConfiguration>(json);
            
            if (config != null && !string.IsNullOrEmpty(config.Strategy))
            {
                config.LoadedFrom = filePath;
                config.LastUpdated = DateTime.UtcNow;
                
                lock (_configLock)
                {
                    _strategyConfigs[config.Strategy] = config;
                }
                
                _logger.LogDebug("Loaded configuration for strategy {Strategy} from {FilePath}", 
                    config.Strategy, filePath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading configuration from {FilePath}", filePath);
        }
    }
    
    private ConfigurationSource CreateDefaultConfigurationSource(string strategy)
    {
        // No static defaults - all parameters must come from config or bundle selection
        _logger.LogWarning("⚠️ Using emergency fallback for strategy {Strategy} - configuration should be externalized", strategy);
        
        return new ConfigurationSource
        {
            Strategy = strategy,
            BaseParameters = new Dictionary<string, object>
            {
                // Emergency safe defaults - should never be used in production
                ["DefaultMultiplier"] = GetFromEnvironmentOrThrow("DEFAULT_MULTIPLIER", 1.0m),
                ["DefaultThreshold"] = GetFromEnvironmentOrThrow("DEFAULT_THRESHOLD", 0.65m),
                ["MaxPositionSize"] = GetFromEnvironmentOrThrow("MAX_POSITION_SIZE", 5),
                ["MinConfidence"] = GetFromEnvironmentOrThrow("MIN_CONFIDENCE", 0.6m)
            },
            RiskParameters = new Dictionary<string, object>
            {
                ["MaxDrawdown"] = GetFromEnvironmentOrThrow("MAX_DRAWDOWN", 0.05m),
                ["StopLossMultiplier"] = GetFromEnvironmentOrThrow("STOP_LOSS_MULTIPLIER", 2.0m),
                ["TakeProfitMultiplier"] = GetFromEnvironmentOrThrow("TAKE_PROFIT_MULTIPLIER", 1.5m)
            },
            MarketConditionOverrides = new Dictionary<string, Dictionary<string, object>>(),
            LoadedFrom = "EmergencyFallback",
            LastUpdated = DateTime.UtcNow
        };
    }
    
    /// <summary>
    /// Get configuration value from environment or throw if not found (no static defaults)
    /// </summary>
    private T GetFromEnvironmentOrThrow<T>(string key, T emergencyDefault) where T : struct
    {
        var envValue = Environment.GetEnvironmentVariable(key);
        if (!string.IsNullOrEmpty(envValue))
        {
            try
            {
                return (T)Convert.ChangeType(envValue, typeof(T));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to parse environment variable {Key}={Value}", key, envValue);
            }
        }
        
        _logger.LogWarning("Environment variable {Key} not found, using emergency default {Default}", key, emergencyDefault);
        return emergencyDefault;
    }
    
    /// <summary>
    /// Save updated configuration for a strategy
    /// </summary>
    public async Task SaveStrategyConfigurationAsync(
        string strategy,
        StrategyConfiguration config,
        CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_configLock)
            {
                _strategyConfigs[strategy] = config;
            }
            
            // Save to first available config path
            if (_configPaths.Count > 0)
            {
                var configDirectory = _configPaths[0];
                Directory.CreateDirectory(configDirectory);
                
                var filePath = Path.Combine(configDirectory, $"{strategy}.json");
                var json = JsonSerializer.Serialize(config, new JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });
                
                await File.WriteAllTextAsync(filePath, json, cancellationToken);
                
                _logger.LogInformation("Saved configuration for strategy {Strategy} to {FilePath}", 
                    strategy, filePath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving configuration for strategy {Strategy}", strategy);
        }
    }
}

/// <summary>
/// Configuration types for the compatibility kit
/// </summary>
public class CompatibilityKitConfiguration
{
    public EnvironmentConfig Environment { get; set; } = new();
    public BundleSelectionConfig BundleSelection { get; set; } = new();
    public SafetyConfig Safety { get; set; } = new();
}

public class EnvironmentConfig
{
    public string Mode { get; set; } = "development"; // development, staging, production
    public List<string> AuthorizedSymbols { get; set; } = new();
    public bool KillSwitchEnabled { get; set; } = true;
}

public class BundleSelectionConfig  
{
    public bool Enabled { get; set; } = true;
    public int MaxBundles { get; set; } = 36;
    public double LearningRate { get; set; } = 0.1;
}

public class SafetyConfig
{
    public double MaxDrawdown { get; set; } = 0.05;
    public int MaxPositionSize { get; set; } = 5;
    public bool RequireEnvironmentValidation { get; set; } = true;
}

/// <summary>
/// Strategy configuration loaded from JSON files
/// </summary>
public class StrategyConfiguration
{
    public string Strategy { get; set; } = string.Empty;
    public Dictionary<string, object> BaseParameters { get; set; } = new();
    public Dictionary<string, object> RiskParameters { get; set; } = new();
    public Dictionary<string, Dictionary<string, object>> MarketConditionOverrides { get; set; } = new();
    public string LoadedFrom { get; set; } = string.Empty;
    public DateTime LastUpdated { get; set; }
}

/// <summary>
/// Configuration source with all parameters for a strategy
/// </summary>
public class ConfigurationSource
{
    public string Strategy { get; set; } = string.Empty;
    public Dictionary<string, object> BaseParameters { get; set; } = new();
    public Dictionary<string, object> RiskParameters { get; set; } = new();
    public Dictionary<string, Dictionary<string, object>> MarketConditionOverrides { get; set; } = new();
    public string LoadedFrom { get; set; } = string.Empty;
    public DateTime LastUpdated { get; set; }
    
    /// <summary>
    /// Get parameter value with fallback to default
    /// </summary>
    public T GetParameter<T>(string parameterName, T defaultValue = default!)
    {
        if (BaseParameters.TryGetValue(parameterName, out var value))
        {
            try
            {
                if (value is JsonElement jsonElement)
                {
                    return JsonSerializer.Deserialize<T>(jsonElement.GetRawText()) ?? defaultValue;
                }
                
                return (T)Convert.ChangeType(value, typeof(T)) ?? defaultValue;
            }
            catch
            {
                return defaultValue;
            }
        }
        
        return defaultValue;
    }
    
    /// <summary>
    /// Get risk parameter value with fallback to default
    /// </summary>
    public T GetRiskParameter<T>(string parameterName, T defaultValue = default!)
    {
        if (RiskParameters.TryGetValue(parameterName, out var value))
        {
            try
            {
                if (value is JsonElement jsonElement)
                {
                    return JsonSerializer.Deserialize<T>(jsonElement.GetRawText()) ?? defaultValue;
                }
                
                return (T)Convert.ChangeType(value, typeof(T)) ?? defaultValue;
            }
            catch
            {
                return defaultValue;
            }
        }
        
        return defaultValue;
    }
    
    /// <summary>
    /// Get market condition specific parameters
    /// </summary>
    public Dictionary<string, object> GetMarketConditionOverrides(string marketCondition)
    {
        return MarketConditionOverrides.GetValueOrDefault(marketCondition, new Dictionary<string, object>());
    }
}

/// <summary>
/// Parameter bundles configuration for stage-based learning
/// </summary>
public class ParameterBundlesConfiguration
{
    public string Version { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public DateTime LastUpdated { get; set; }
    public Dictionary<string, ParameterBundle> Bundles { get; set; } = new();
    public GlobalBundleSettings GlobalSettings { get; set; } = new();
}

public class ParameterBundle
{
    public string Stage { get; set; } = string.Empty;
    public bool Enabled { get; set; }
    public Dictionary<string, object> Parameters { get; set; } = new();
    public Dictionary<string, Dictionary<string, object>> MarketConditions { get; set; } = new();
}

public class GlobalBundleSettings
{
    public int MaxBundlesPerStrategy { get; set; } = 3;
    public double LearningRate { get; set; } = 0.1;
    public double ExplorationWeight { get; set; } = 0.15;
    public int MinSamplesForUpdate { get; set; } = 20;
}

/// <summary>
/// Parameter bounds configuration for validation
/// </summary>
public class ParameterBoundsConfiguration
{
    public string Version { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public DateTime LastUpdated { get; set; }
    public Dictionary<string, ParameterBound> ParameterBounds { get; set; } = new();
    public ValidationRules ValidationRules { get; set; } = new();
    public Dictionary<string, MarketConditionBounds> MarketConditionBounds { get; set; } = new();
}

public class ParameterBound
{
    public double Min { get; set; }
    public double Max { get; set; }
    public double Default { get; set; }
    public double Step { get; set; }
    public string Description { get; set; } = string.Empty;
}

public class ValidationRules
{
    public GlobalConstraints GlobalConstraints { get; set; } = new();
    public Dictionary<string, StrategyConstraints> StrategyConstraints { get; set; } = new();
}

public class GlobalConstraints
{
    public int MaxConcurrentPositions { get; set; } = 5;
    public double MaxDailyLoss { get; set; } = 1000;
    public double MinConfidenceForLive { get; set; } = 0.6;
    public bool RequireStopLoss { get; set; } = true;
}

public class StrategyConstraints
{
    public int MaxPositionSize { get; set; }
    public double MinConfidence { get; set; }
    public double MaxRiskPerTrade { get; set; }
}

public class MarketConditionBounds
{
    public double Threshold { get; set; }
    public Dictionary<string, ParameterBoundAdjustment> Adjustments { get; set; } = new();
}

public class ParameterBoundAdjustment
{
    public double Min { get; set; }
    public double Max { get; set; }
}