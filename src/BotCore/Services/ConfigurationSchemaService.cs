using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Schema versioning service for JSON configuration validation and migration
    /// Ensures compatibility when configuration formats change
    /// </summary>
    public class ConfigurationSchemaService
    {
        private readonly ILogger<ConfigurationSchemaService> _logger;
        private readonly Dictionary<string, IConfigurationMigrator> _migrators = new();

        public ConfigurationSchemaService(ILogger<ConfigurationSchemaService> logger)
        {
            _logger = logger;
            RegisterDefaultMigrators();
        }

        /// <summary>
        /// Validate and migrate configuration JSON to current version
        /// </summary>
        public T ValidateAndMigrate<T>(string jsonContent, string configType) where T : IVersionedConfiguration, new()
        {
            try
            {
                // Parse JSON to get version info
                var jsonDoc = JsonDocument.Parse(jsonContent);
                var versionElement = jsonDoc.RootElement.GetProperty("SchemaVersion");
                var currentVersion = versionElement.GetString() ?? "1.0";

                _logger.LogInformation("🔍 [SCHEMA] Validating {ConfigType} configuration, version: {Version}", 
                    configType, currentVersion);

                // Get target version for type T
                var targetInstance = new T();
                var targetVersion = targetInstance.SchemaVersion;

                if (currentVersion == targetVersion)
                {
                    // No migration needed
                    var result = JsonSerializer.Deserialize<T>(jsonContent);
                    if (result is null)
                        throw new InvalidOperationException($"Failed to deserialize {configType} configuration");
                        
                    _logger.LogInformation("✅ [SCHEMA] Configuration validated, no migration needed");
                    return result;
                }

                // Migration needed
                _logger.LogInformation("🔄 [SCHEMA] Migration required: {CurrentVersion} → {TargetVersion}", 
                    currentVersion, targetVersion);

                var migratedJson = MigrateConfiguration(jsonContent, configType, currentVersion, targetVersion);
                var migratedResult = JsonSerializer.Deserialize<T>(migratedJson);
                
                if (migratedResult is null)
                    throw new InvalidOperationException($"Failed to deserialize migrated {configType} configuration");

                _logger.LogInformation("✅ [SCHEMA] Configuration migrated successfully");
                return migratedResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [SCHEMA] Schema validation failed for {ConfigType}", configType);
                throw new ConfigurationSchemaException($"Schema validation failed for {configType}: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Create a new configuration with current schema version
        /// </summary>
        public static T CreateDefault<T>() where T : IVersionedConfiguration, new()
        {
            var instance = new T();
            instance.SchemaVersion = GetCurrentVersion<T>();
            return instance;
        }

        /// <summary>
        /// Save configuration with schema version
        /// </summary>
        public async Task SaveConfigurationAsync<T>(T configuration, string filePath) where T : IVersionedConfiguration, new()
        {
            try
            {
                configuration.SchemaVersion = GetCurrentVersion<T>();
                configuration.LastModified = DateTime.UtcNow;

                var json = JsonSerializer.Serialize(configuration, new JsonSerializerOptions 
                { 
                    WriteIndented = true,
                    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
                });

                await File.WriteAllTextAsync(filePath, json).ConfigureAwait(false);
                
                _logger.LogInformation("💾 [SCHEMA] Configuration saved: {FilePath}, version: {Version}", 
                    filePath, configuration.SchemaVersion);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [SCHEMA] Failed to save configuration: {FilePath}", filePath);
                throw;
            }
        }

        private string MigrateConfiguration(string jsonContent, string configType, string fromVersion, string toVersion)
        {
            if (!_migrators.TryGetValue(configType, out var migrator))
            {
                throw new NotSupportedException($"No migrator available for configuration type: {configType}");
            }

            return migrator.Migrate(jsonContent, fromVersion, toVersion);
        }

        private static string GetCurrentVersion<T>() where T : IVersionedConfiguration, new()
        {
            return new T().SchemaVersion;
        }

        private void RegisterDefaultMigrators()
        {
            // Register migrators for different configuration types
            _migrators["MLConfiguration"] = new MLConfigurationMigrator(_logger);
            _migrators["RiskConfiguration"] = new RiskConfigurationMigrator(_logger);
            _migrators["ExecutionConfiguration"] = new ExecutionConfigurationMigrator(_logger);
        }
    }

    /// <summary>
    /// Interface for versioned configuration objects
    /// </summary>
    public interface IVersionedConfiguration
    {
        string SchemaVersion { get; set; }
        DateTime LastModified { get; set; }
    }

    /// <summary>
    /// Interface for configuration migrators
    /// </summary>
    public interface IConfigurationMigrator
    {
        string Migrate(string jsonContent, string fromVersion, string toVersion);
    }

    /// <summary>
    /// Exception thrown when configuration schema validation fails
    /// </summary>
    public class ConfigurationSchemaException : Exception
    {
        public ConfigurationSchemaException(string message) : base(message) { }
        public ConfigurationSchemaException(string message, Exception innerException) : base(message, innerException) { }

        public ConfigurationSchemaException()
        {
        }
    }

    /// <summary>
    /// Sample versioned configuration for ML settings
    /// </summary>
    public class VersionedMLConfiguration : IVersionedConfiguration
    {
        public string SchemaVersion { get; set; } = "2.0";
        public DateTime LastModified { get; set; }
        
        public double AIConfidenceThreshold { get; set; } = 0.7;
        public double MinimumConfidence { get; set; } = 0.5;
        public double PositionSizeMultiplier { get; set; } = 1.0;
        public double RegimeDetectionThreshold { get; set; } = 0.8;
    }

    /// <summary>
    /// ML Configuration migrator
    /// </summary>
    public class MLConfigurationMigrator : IConfigurationMigrator
    {
        private readonly ILogger _logger;

        public MLConfigurationMigrator(ILogger logger)
        {
            _logger = logger;
        }

        public string Migrate(string jsonContent, string fromVersion, string toVersion)
        {
            _logger.LogInformation("🔄 [ML-MIGRATOR] Migrating from {FromVersion} to {ToVersion}", fromVersion, toVersion);

            var jsonDoc = JsonDocument.Parse(jsonContent);
            var root = jsonDoc.RootElement;
            var migrated = new Dictionary<string, object>();

            // Copy all existing properties
            foreach (var prop in root.EnumerateObject())
            {
                migrated[prop.Name] = prop.Value.GetRawText();
            }

            // Apply version-specific migrations
            switch ((fromVersion, toVersion))
            {
                case ("1.0", "2.0"):
                    // Migration from 1.0 to 2.0: add RegimeDetectionThreshold if missing
                    if (!migrated.ContainsKey("RegimeDetectionThreshold"))
                    {
                        migrated["RegimeDetectionThreshold"] = 0.8;
                        _logger.LogInformation("✅ [ML-MIGRATOR] Added RegimeDetectionThreshold default");
                    }
                    break;
            }

            // Update schema version
            migrated["SchemaVersion"] = $"\"{toVersion}\"";
            migrated["LastModified"] = $"\"{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\"";

            return JsonSerializer.Serialize(migrated, new JsonSerializerOptions { WriteIndented = true });
        }
    }

    /// <summary>
    /// Risk Configuration migrator
    /// </summary>
    public class RiskConfigurationMigrator : IConfigurationMigrator
    {
        private readonly ILogger _logger;

        public RiskConfigurationMigrator(ILogger logger)
        {
            _logger = logger;
        }

        public string Migrate(string jsonContent, string fromVersion, string toVersion)
        {
            _logger.LogInformation("🔄 [RISK-MIGRATOR] Migrating from {FromVersion} to {ToVersion}", fromVersion, toVersion);
            
            // Simple pass-through for now - extend as needed
            var jsonDoc = JsonDocument.Parse(jsonContent);
            var root = jsonDoc.RootElement;
            var migrated = new Dictionary<string, object>();

            foreach (var prop in root.EnumerateObject())
            {
                migrated[prop.Name] = prop.Value.GetRawText();
            }

            migrated["SchemaVersion"] = $"\"{toVersion}\"";
            migrated["LastModified"] = $"\"{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\"";

            return JsonSerializer.Serialize(migrated, new JsonSerializerOptions { WriteIndented = true });
        }
    }

    /// <summary>
    /// Execution Configuration migrator
    /// </summary>
    public class ExecutionConfigurationMigrator : IConfigurationMigrator
    {
        private readonly ILogger _logger;

        public ExecutionConfigurationMigrator(ILogger logger)
        {
            _logger = logger;
        }

        public string Migrate(string jsonContent, string fromVersion, string toVersion)
        {
            _logger.LogInformation("🔄 [EXEC-MIGRATOR] Migrating from {FromVersion} to {ToVersion}", fromVersion, toVersion);
            
            // Simple pass-through for now - extend as needed
            var jsonDoc = JsonDocument.Parse(jsonContent);
            var root = jsonDoc.RootElement;
            var migrated = new Dictionary<string, object>();

            foreach (var prop in root.EnumerateObject())
            {
                migrated[prop.Name] = prop.Value.GetRawText();
            }

            migrated["SchemaVersion"] = $"\"{toVersion}\"";
            migrated["LastModified"] = $"\"{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\"";

            return JsonSerializer.Serialize(migrated, new JsonSerializerOptions { WriteIndented = true });
        }
    }
}