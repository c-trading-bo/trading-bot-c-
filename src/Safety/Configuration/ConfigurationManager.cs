using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Hosting;
using System.Collections.Concurrent;
using System.Text.Json;
using System.Security.Cryptography;
using System.Text;

namespace Trading.Safety.Configuration;

/// <summary>
/// Production-grade configuration and change management system
/// Provides central config registry, feature flags, rollout strategies,
/// and configuration validation with severity levels
/// </summary>
public interface IConfigurationManager
{
    Task<T?> GetConfigAsync<T>(string key) where T : class;
    Task SetConfigAsync<T>(string key, T value, string? changeReason = null) where T : class;
    Task<bool> GetFeatureFlagAsync(string flagName);
    Task SetFeatureFlagAsync(string flagName, bool enabled, string? changeReason = null);
    Task<RolloutStatus> GetRolloutStatusAsync(string featureName);
    Task SetRolloutPercentageAsync(string featureName, double percentage, string? changeReason = null);
    Task<List<ConfigurationChange>> GetConfigurationHistoryAsync(DateTime? from = null);
    Task ValidateConfigurationAsync();
    Task ExportConfigurationAsync(string filePath);
    Task ImportConfigurationAsync(string filePath, bool validateFirst = true);
    event Action<ConfigurationChange> OnConfigurationChanged;
    event Action<ValidationResult> OnValidationCompleted;
}

public class ConfigurationManager : IConfigurationManager, IHostedService
{
    private readonly ILogger<ConfigurationManager> _logger;
    private readonly ConfigurationManagerConfig _config;
    private readonly ConcurrentDictionary<string, ConfigurationItem> _configurations = new();
    private readonly ConcurrentDictionary<string, FeatureFlag> _featureFlags = new();
    private readonly ConcurrentDictionary<string, RolloutConfig> _rollouts = new();
    private readonly List<ConfigurationChange> _changeHistory = new();
    private readonly Timer _validationTimer;
    private readonly Timer _persistenceTimer;
    private readonly object _changeLock = new object();

    public event Action<ConfigurationChange> OnConfigurationChanged = delegate { };
    public event Action<ValidationResult> OnValidationCompleted = delegate { };

    public ConfigurationManager(
        ILogger<ConfigurationManager> logger,
        IOptions<ConfigurationManagerConfig> config)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _config = config?.Value ?? throw new ArgumentNullException(nameof(config));
        
        _validationTimer = new Timer(ValidationCallback, null, Timeout.Infinite, Timeout.Infinite);
        _persistenceTimer = new Timer(PersistenceCallback, null, Timeout.Infinite, Timeout.Infinite);
        
        InitializeDefaultConfigurations();
    }

    public async Task<T?> GetConfigAsync<T>(string key) where T : class
    {
        try
        {
            if (_configurations.TryGetValue(key, out var item))
            {
                var json = item.Value;
                var result = JsonSerializer.Deserialize<T>(json, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });

                _logger.LogDebug("[CONFIG_MANAGER] Retrieved config: {Key} of type {Type}", key, typeof(T).Name);
                return result;
            }

            _logger.LogDebug("[CONFIG_MANAGER] Config not found: {Key}", key);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error retrieving config: {Key}", key);
            return null;
        }
    }

    public Task SetConfigAsync<T>(string key, T value, string? changeReason = null) where T : class
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        
        try
        {
            var json = JsonSerializer.Serialize(value, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true
            });

            var oldItem = _configurations.GetValueOrDefault(key);
            var newItem = new ConfigurationItem
            {
                Key = key,
                Value = json,
                Type = typeof(T).FullName ?? typeof(T).Name,
                LastModified = DateTime.UtcNow,
                ModifiedBy = Environment.UserName,
                Version = (oldItem?.Version ?? 0) + 1,
                Hash = CalculateHash(json)
            };

            _configurations.AddOrUpdate(key, newItem, (k, existing) => newItem);

            var change = new ConfigurationChange
            {
                Id = Guid.NewGuid().ToString("N"),
                Key = key,
                ChangeType = ConfigurationChangeType.Updated,
                OldValue = oldItem?.Value,
                NewValue = json,
                Reason = changeReason ?? "Programmatic update",
                Timestamp = DateTime.UtcNow,
                ModifiedBy = Environment.UserName,
                CorrelationId = correlationId
            };

            lock (_changeLock)
            {
                _changeHistory.Add(change);
                
                // Trim history if too long
                while (_changeHistory.Count > _config.MaxHistoryEntries)
                {
                    _changeHistory.RemoveAt(0);
                }
            }

            OnConfigurationChanged.Invoke(change);

            _logger.LogInformation("[CONFIG_MANAGER] Configuration updated: {Key} v{Version} - {Reason} [CorrelationId: {CorrelationId}]",
                key, newItem.Version, changeReason ?? "No reason", correlationId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error setting config: {Key} [CorrelationId: {CorrelationId}]", 
                key, correlationId);
            throw;
        }

        return Task.CompletedTask;
    }

    public async Task<bool> GetFeatureFlagAsync(string flagName)
    {
        try
        {
            if (_featureFlags.TryGetValue(flagName, out var flag))
            {
                var isEnabled = flag.Enabled && IsWithinRolloutPercentage(flagName);
                
                _logger.LogDebug("[CONFIG_MANAGER] Feature flag checked: {FlagName} = {Enabled}", flagName, isEnabled);
                return isEnabled;
            }

            _logger.LogDebug("[CONFIG_MANAGER] Feature flag not found: {FlagName}, defaulting to false", flagName);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error checking feature flag: {FlagName}", flagName);
            return false;
        }
    }

    public Task SetFeatureFlagAsync(string flagName, bool enabled, string? changeReason = null)
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        
        try
        {
            var oldFlag = _featureFlags.GetValueOrDefault(flagName);
            var newFlag = new FeatureFlag
            {
                Name = flagName,
                Enabled = enabled,
                LastModified = DateTime.UtcNow,
                ModifiedBy = Environment.UserName,
                Description = oldFlag?.Description ?? $"Feature flag for {flagName}"
            };

            _featureFlags.AddOrUpdate(flagName, newFlag, (k, existing) => newFlag);

            var change = new ConfigurationChange
            {
                Id = Guid.NewGuid().ToString("N"),
                Key = $"FeatureFlag:{flagName}",
                ChangeType = ConfigurationChangeType.FeatureFlagToggled,
                OldValue = oldFlag?.Enabled.ToString(),
                NewValue = enabled.ToString(),
                Reason = changeReason ?? "Feature flag toggle",
                Timestamp = DateTime.UtcNow,
                ModifiedBy = Environment.UserName,
                CorrelationId = correlationId
            };

            lock (_changeLock)
            {
                _changeHistory.Add(change);
            }

            OnConfigurationChanged.Invoke(change);

            _logger.LogInformation("[CONFIG_MANAGER] Feature flag updated: {FlagName} = {Enabled} - {Reason} [CorrelationId: {CorrelationId}]",
                flagName, enabled, changeReason ?? "No reason", correlationId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error setting feature flag: {FlagName} [CorrelationId: {CorrelationId}]", 
                flagName, correlationId);
            throw;
        }

        return Task.CompletedTask;
    }

    public async Task<RolloutStatus> GetRolloutStatusAsync(string featureName)
    {
        try
        {
            if (_rollouts.TryGetValue(featureName, out var rollout))
            {
                var status = new RolloutStatus
                {
                    FeatureName = featureName,
                    CurrentPercentage = rollout.Percentage,
                    Strategy = rollout.Strategy,
                    IsActive = rollout.IsActive,
                    StartTime = rollout.StartTime,
                    LastModified = rollout.LastModified
                };

                return status;
            }

            return new RolloutStatus { FeatureName = featureName, CurrentPercentage = 0 };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error getting rollout status: {FeatureName}", featureName);
            return new RolloutStatus { FeatureName = featureName, CurrentPercentage = 0 };
        }
    }

    public Task SetRolloutPercentageAsync(string featureName, double percentage, string? changeReason = null)
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        
        try
        {
            percentage = Math.Max(0, Math.Min(100, percentage)); // Clamp to 0-100%
            
            var oldRollout = _rollouts.GetValueOrDefault(featureName);
            var newRollout = new RolloutConfig
            {
                FeatureName = featureName,
                Percentage = percentage,
                Strategy = oldRollout?.Strategy ?? RolloutStrategy.Percentage,
                IsActive = percentage > 0,
                StartTime = oldRollout?.StartTime ?? DateTime.UtcNow,
                LastModified = DateTime.UtcNow,
                ModifiedBy = Environment.UserName
            };

            _rollouts.AddOrUpdate(featureName, newRollout, (k, existing) => newRollout);

            var change = new ConfigurationChange
            {
                Id = Guid.NewGuid().ToString("N"),
                Key = $"Rollout:{featureName}",
                ChangeType = ConfigurationChangeType.RolloutChanged,
                OldValue = oldRollout?.Percentage.ToString(),
                NewValue = percentage.ToString(),
                Reason = changeReason ?? "Rollout percentage change",
                Timestamp = DateTime.UtcNow,
                ModifiedBy = Environment.UserName,
                CorrelationId = correlationId
            };

            lock (_changeLock)
            {
                _changeHistory.Add(change);
            }

            OnConfigurationChanged.Invoke(change);

            _logger.LogInformation("[CONFIG_MANAGER] Rollout updated: {FeatureName} = {Percentage}% - {Reason} [CorrelationId: {CorrelationId}]",
                featureName, percentage, changeReason ?? "No reason", correlationId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error setting rollout percentage: {FeatureName} [CorrelationId: {CorrelationId}]", 
                featureName, correlationId);
            throw;
        }

        return Task.CompletedTask;
    }

    public async Task<List<ConfigurationChange>> GetConfigurationHistoryAsync(DateTime? from = null)
    {
        try
        {
            lock (_changeLock)
            {
                var query = _changeHistory.AsEnumerable();
                
                if (from.HasValue)
                {
                    query = query.Where(c => c.Timestamp >= from.Value);
                }
                
                return query.OrderByDescending(c => c.Timestamp).ToList();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error retrieving configuration history");
            return new List<ConfigurationChange>();
        }
    }

    public async Task ValidateConfigurationAsync()
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        var result = new ValidationResult
        {
            ValidationId = correlationId,
            Timestamp = DateTime.UtcNow,
            Issues = new List<ValidationIssue>()
        };

        try
        {
            _logger.LogInformation("[CONFIG_MANAGER] Starting configuration validation [CorrelationId: {CorrelationId}]", correlationId);

            // Validate critical configurations
            await ValidateCriticalConfigurationsAsync(result).ConfigureAwait(false);
            
            // Validate feature flag consistency
            await ValidateFeatureFlagsAsync(result).ConfigureAwait(false);
            
            // Validate rollout configurations
            await ValidateRolloutsAsync(result).ConfigureAwait(false);
            
            // Check for configuration drift
            await CheckConfigurationDriftAsync(result).ConfigureAwait(false);

            var criticalIssueCount = result.Issues.Count(i => i.Severity == ValidationSeverity.Critical);
            var warningIssueCount = result.Issues.Count(i => i.Severity == ValidationSeverity.Warning);

            if (criticalIssueCount > 0)
            {
                _logger.LogCritical("[CONFIG_MANAGER] Configuration validation FAILED: {CriticalIssues} critical, {Warnings} warnings [CorrelationId: {CorrelationId}]",
                    criticalIssueCount, warningIssueCount, correlationId);
                result.IsValid;
            }
            else if (warningIssueCount > 0)
            {
                _logger.LogWarning("[CONFIG_MANAGER] Configuration validation passed with warnings: {Warnings} warnings [CorrelationId: {CorrelationId}]",
                    warningIssueCount, correlationId);
                result.IsValid = true;
            }
            else
            {
                _logger.LogInformation("[CONFIG_MANAGER] Configuration validation passed [CorrelationId: {CorrelationId}]", correlationId);
                result.IsValid = true;
            }

            OnValidationCompleted.Invoke(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error during configuration validation [CorrelationId: {CorrelationId}]", correlationId);
            result.IsValid;
            result.Issues.Add(new ValidationIssue
            {
                Key = "VALIDATION_ERROR",
                Message = $"Validation process failed: {ex.Message}",
                Severity = ValidationSeverity.Critical
            });
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task ExportConfigurationAsync(string filePath)
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        
        try
        {
            var export = new ConfigurationExport
            {
                ExportId = correlationId,
                Timestamp = DateTime.UtcNow,
                ExportedBy = Environment.UserName,
                Configurations = _configurations.Values.ToList(),
                FeatureFlags = _featureFlags.Values.ToList(),
                Rollouts = _rollouts.Values.ToList()
            };

            var json = JsonSerializer.Serialize(export, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true
            });

            await File.WriteAllTextAsync(filePath, json).ConfigureAwait(false);

            _logger.LogInformation("[CONFIG_MANAGER] Configuration exported to: {FilePath} [CorrelationId: {CorrelationId}]", 
                filePath, correlationId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error exporting configuration [CorrelationId: {CorrelationId}]", correlationId);
            throw;
        }
    }

    public async Task ImportConfigurationAsync(string filePath, bool validateFirst = true)
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        
        try
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Configuration file not found: {filePath}");
            }

            var json = await File.ReadAllTextAsync(filePath).ConfigureAwait(false);
            var import = JsonSerializer.Deserialize<ConfigurationExport>(json, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            if (import == null)
            {
                throw new InvalidOperationException("Failed to deserialize configuration export");
            }

            if (validateFirst)
            {
                // Validate import before applying - check for required keys, valid values, and security
                _logger.LogInformation("[CONFIG_MANAGER] Validating configuration import [CorrelationId: {CorrelationId}]", correlationId);
                
                var configDictionary = import.Configurations.ToDictionary(c => c.Key, c => JsonSerializer.Deserialize<object>(c.Value) ?? c.Value);
                await ValidateConfigurationImport(configDictionary, correlationId).ConfigureAwait(false);
            }

            // Apply configurations
            foreach (var config in import.Configurations)
            {
                _configurations.AddOrUpdate(config.Key, config, (k, existing) => config);
            }

            foreach (var flag in import.FeatureFlags)
            {
                _featureFlags.AddOrUpdate(flag.Name, flag, (k, existing) => flag);
            }

            foreach (var rollout in import.Rollouts)
            {
                _rollouts.AddOrUpdate(rollout.FeatureName, rollout, (k, existing) => rollout);
            }

            _logger.LogInformation("[CONFIG_MANAGER] Configuration imported from: {FilePath} - {ConfigCount} configs, {FlagCount} flags, {RolloutCount} rollouts [CorrelationId: {CorrelationId}]",
                filePath, import.Configurations.Count, import.FeatureFlags.Count, import.Rollouts.Count, correlationId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error importing configuration [CorrelationId: {CorrelationId}]", correlationId);
            throw;
        }
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _validationTimer.Change(TimeSpan.Zero, _config.ValidationInterval);
        _persistenceTimer.Change(_config.PersistenceInterval, _config.PersistenceInterval);
        
        // Load persisted configuration if available
        await LoadPersistedConfigurationAsync().ConfigureAwait(false);
        
        _logger.LogInformation("[CONFIG_MANAGER] Started with validation interval: {ValidationInterval}", _config.ValidationInterval);
        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _validationTimer.Change(Timeout.Infinite, Timeout.Infinite);
        _persistenceTimer.Change(Timeout.Infinite, Timeout.Infinite);
        
        // Persist configuration before shutdown
        await PersistConfigurationAsync().ConfigureAwait(false);
        
        _logger.LogInformation("[CONFIG_MANAGER] Stopped");
    }

    // Private implementation methods
    private void InitializeDefaultConfigurations()
    {
        // Initialize with default critical configurations
        var defaultConfigs = new Dictionary<string, object>
        {
            ["TradingEnabled"] = true,
            ["MaxDailyLoss"] = 1000m,
            ["MaxPositionSize"] = 10,
            ["RiskTolerancePercent"] = 2.0,
            ["DataFeedUrl"] = "wss://default-feed.example.com",
            ["OrderTimeoutSeconds"] = 300
        };

        foreach (var kvp in defaultConfigs)
        {
            var json = JsonSerializer.Serialize(kvp.Value);
            _configurations.TryAdd(kvp.Key, new ConfigurationItem
            {
                Key = kvp.Key,
                Value = json,
                Type = kvp.Value.GetType().Name,
                LastModified = DateTime.UtcNow,
                ModifiedBy = "System",
                Version = 1,
                Hash = CalculateHash(json)
            });
        }

        _logger.LogDebug("[CONFIG_MANAGER] Initialized with {ConfigCount} default configurations", defaultConfigs.Count);
    }

    private bool IsWithinRolloutPercentage(string featureName)
    {
        if (!_rollouts.TryGetValue(featureName, out var rollout) || !rollout.IsActive)
        {
            return true; // No rollout config means 100% rollout
        }

        // Simple percentage-based rollout (in production, use more sophisticated logic)
        var userHash = CalculateUserHash(Environment.MachineName + Environment.UserName);
        var userPercentile = Math.Abs(userHash.GetHashCode()) % 100;
        
        return userPercentile < rollout.Percentage;
    }

    private async Task ValidateConfigurationImport(Dictionary<string, object> newConfig, string correlationId)
    {
        var validationErrors = new List<string>();

        foreach (var kvp in newConfig)
        {
            // Check required keys
            if (string.IsNullOrWhiteSpace(kvp.Key))
            {
                validationErrors.Add("Configuration key cannot be empty");
                continue;
            }

            // Validate critical configuration values
            await ValidateConfigurationValue(kvp.Key, kvp.Value, validationErrors).ConfigureAwait(false);
        }

        if (validationErrors.Any())
        {
            var errorMessage = $"Configuration validation failed: {string.Join(", ", validationErrors)}";
            _logger.LogError("[CONFIG_MANAGER] {ErrorMessage} [CorrelationId: {CorrelationId}]", errorMessage, correlationId);
            throw new InvalidOperationException(errorMessage);
        }

        _logger.LogInformation("[CONFIG_MANAGER] Configuration validation passed [CorrelationId: {CorrelationId}]", correlationId);
    }

    private async Task ValidateConfigurationValue(string key, object value, List<string> validationErrors)
    {
        await Task.CompletedTask.ConfigureAwait(false); // For async consistency

        switch (key.ToUpperInvariant())
        {
            case "MAXDAILYLOSS":
                if (value is decimal loss && loss <= 0)
                    validationErrors.Add("MaxDailyLoss must be positive");
                break;

            case "MAXPOSITIONSIZE":
                if (value is int size && size <= 0)
                    validationErrors.Add("MaxPositionSize must be positive");
                break;

            case "RISKTOLERANCEPERCENT":
                if (value is double risk && (risk <= 0 || risk > 100))
                    validationErrors.Add("RiskTolerancePercent must be between 0 and 100");
                break;

            case "ORDERTIMEOUTSECONDS":
                if (value is int timeout && timeout <= 0)
                    validationErrors.Add("OrderTimeoutSeconds must be positive");
                break;

            case "TRADINGPWD":
            case "APIKEY":
            case "SECRET":
                if (value is string secret && string.IsNullOrWhiteSpace(secret))
                    validationErrors.Add($"{key} cannot be empty for security reasons");
                break;
        }
    }

    private string CalculateHash(string input)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(input));
        return Convert.ToBase64String(hashBytes);
    }

    private string CalculateUserHash(string user)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(user));
        return Convert.ToBase64String(hashBytes);
    }

    private Task ValidateCriticalConfigurationsAsync(ValidationResult result)
    {
        var criticalKeys = _config.CriticalConfigurationKeys;
        
        foreach (var key in criticalKeys)
        {
            if (!_configurations.ContainsKey(key))
            {
                result.Issues.Add(new ValidationIssue
                {
                    Key = key,
                    Message = $"Critical configuration missing: {key}",
                    Severity = ValidationSeverity.Critical
                });
            }
        }

        return Task.CompletedTask;
    }

    private Task ValidateFeatureFlagsAsync(ValidationResult result)
    {
        // Check for orphaned feature flags or conflicting states
        foreach (var flag in _featureFlags.Values)
        {
            if (flag.Enabled && _rollouts.TryGetValue(flag.Name, out var rollout) && !rollout.IsActive)
            {
                result.Issues.Add(new ValidationIssue
                {
                    Key = flag.Name,
                    Message = $"Feature flag enabled but rollout is inactive: {flag.Name}",
                    Severity = ValidationSeverity.Warning
                });
            }
        }

        return Task.CompletedTask;
    }

    private Task ValidateRolloutsAsync(ValidationResult result)
    {
        foreach (var rollout in _rollouts.Values)
        {
            if (rollout.Percentage < 0 || rollout.Percentage > 100)
            {
                result.Issues.Add(new ValidationIssue
                {
                    Key = rollout.FeatureName,
                    Message = $"Invalid rollout percentage: {rollout.Percentage}% (must be 0-100%)",
                    Severity = ValidationSeverity.Critical
                });
            }
        }

        return Task.CompletedTask;
    }

    private Task CheckConfigurationDriftAsync(ValidationResult result)
    {
        // Check for configuration changes without proper approval
        var recentChanges = _changeHistory
            .Where(c => c.Timestamp > DateTime.UtcNow.AddHours(-1))
            .Where(c => string.IsNullOrEmpty(c.Reason))
            .ToList();

        if (recentChanges.Any())
        {
            result.Issues.Add(new ValidationIssue
            {
                Key = "CONFIGURATION_DRIFT",
                Message = $"{recentChanges.Count} recent configuration changes without documented reason",
                Severity = ValidationSeverity.Warning
            });
        }

        return Task.CompletedTask;
    }

    private async Task LoadPersistedConfigurationAsync()
    {
        try
        {
            var configFile = Path.Combine(_config.PersistenceDirectory, "configuration.json");
            if (File.Exists(configFile))
            {
                await ImportConfigurationAsync(configFile, false).ConfigureAwait(false);
                _logger.LogInformation("[CONFIG_MANAGER] Loaded persisted configuration from: {ConfigFile}", configFile);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[CONFIG_MANAGER] Failed to load persisted configuration - using defaults");
        }
    }

    private async Task PersistConfigurationAsync()
    {
        try
        {
            Directory.CreateDirectory(_config.PersistenceDirectory);
            var configFile = Path.Combine(_config.PersistenceDirectory, "configuration.json");
            await ExportConfigurationAsync(configFile).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Failed to persist configuration");
        }
    }

    private void ValidationCallback(object? state)
    {
        try
        {
            _ = Task.Run(async () => await ValidateConfigurationAsync().ConfigureAwait(false)).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error in validation callback");
        }
    }

    private void PersistenceCallback(object? state)
    {
        try
        {
            _ = Task.Run(async () => await PersistConfigurationAsync().ConfigureAwait(false)).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CONFIG_MANAGER] Error in persistence callback");
        }
    }

    public void Dispose()
    {
        _validationTimer?.Dispose();
        _persistenceTimer?.Dispose();
    }
}

// Data models and supporting classes
public class ConfigurationItem
{
    public string Key { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
    public DateTime LastModified { get; set; }
    public string ModifiedBy { get; set; } = string.Empty;
    public int Version { get; set; }
    public string Hash { get; set; } = string.Empty;
}

public class FeatureFlag
{
    public string Name { get; set; } = string.Empty;
    public bool Enabled { get; set; }
    public string Description { get; set; } = string.Empty;
    public DateTime LastModified { get; set; }
    public string ModifiedBy { get; set; } = string.Empty;
}

public class RolloutConfig
{
    public string FeatureName { get; set; } = string.Empty;
    public double Percentage { get; set; }
    public RolloutStrategy Strategy { get; set; }
    public bool IsActive { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime LastModified { get; set; }
    public string ModifiedBy { get; set; } = string.Empty;
}

public class RolloutStatus
{
    public string FeatureName { get; set; } = string.Empty;
    public double CurrentPercentage { get; set; }
    public RolloutStrategy Strategy { get; set; }
    public bool IsActive { get; set; }
    public DateTime? StartTime { get; set; }
    public DateTime? LastModified { get; set; }
}

public class ConfigurationChange
{
    public string Id { get; set; } = string.Empty;
    public string Key { get; set; } = string.Empty;
    public ConfigurationChangeType ChangeType { get; set; }
    public string? OldValue { get; set; }
    public string? NewValue { get; set; }
    public string Reason { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string ModifiedBy { get; set; } = string.Empty;
    public string CorrelationId { get; set; } = string.Empty;
}

public class ValidationResult
{
    public string ValidationId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public bool IsValid { get; set; }
    public List<ValidationIssue> Issues { get; } = new();
}

public class ValidationIssue
{
    public string Key { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public ValidationSeverity Severity { get; set; }
}

public class ConfigurationExport
{
    public string ExportId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string ExportedBy { get; set; } = string.Empty;
    public List<ConfigurationItem> Configurations { get; } = new();
    public List<FeatureFlag> FeatureFlags { get; } = new();
    public List<RolloutConfig> Rollouts { get; } = new();
}

public enum ConfigurationChangeType
{
    Created,
    Updated,
    Deleted,
    FeatureFlagToggled,
    RolloutChanged
}

public enum RolloutStrategy
{
    Percentage,
    Canary,
    Shadow,
    BlueGreen
}

public enum ValidationSeverity
{
    Info,
    Warning,
    Critical
}

public class ConfigurationManagerConfig
{
    public TimeSpan ValidationInterval { get; set; } = TimeSpan.FromMinutes(15);
    public TimeSpan PersistenceInterval { get; set; } = TimeSpan.FromMinutes(5);
    public string PersistenceDirectory { get; set; } = "./config";
    public int MaxHistoryEntries { get; set; } = 10000;
    
    public List<string> CriticalConfigurationKeys { get; set; } = new()
    {
        "TradingEnabled",
        "MaxDailyLoss",
        "MaxPositionSize",
        "RiskTolerancePercent"
    };
}