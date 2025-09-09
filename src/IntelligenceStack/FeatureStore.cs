using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.Security.Cryptography;
using System.Text;
using System.IO;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Feature store v1 with versioned schemas and validation
/// Ensures online/offline parity and rejects invalid batches
/// </summary>
public class FeatureStore : IFeatureStore
{
    private readonly ILogger<FeatureStore> _logger;
    private readonly string _basePath;
    private readonly Dictionary<string, FeatureSchema> _schemaCache = new();
    private readonly object _lock = new();

    private const double MaxMissingnessPct = 0.5;
    private const double MaxOutOfRangePct = 1.0;

    public FeatureStore(ILogger<FeatureStore> logger, string basePath = "data/features")
    {
        _logger = logger;
        _basePath = basePath;
        Directory.CreateDirectory(_basePath);
        Directory.CreateDirectory(Path.Combine(_basePath, "schemas"));
        Directory.CreateDirectory(Path.Combine(_basePath, "features"));
    }

    public async Task<FeatureSet> GetFeaturesAsync(string symbol, DateTime fromTime, DateTime toTime, CancellationToken cancellationToken = default)
    {
        try
        {
            var featuresPath = Path.Combine(_basePath, "features", symbol);
            if (!Directory.Exists(featuresPath))
            {
                _logger.LogWarning("[FEATURES] No feature data found for symbol: {Symbol}", symbol);
                return new FeatureSet { Symbol = symbol };
            }

            // Find feature files in time range
            var featureFiles = Directory.GetFiles(featuresPath, "*.json")
                .Where(f => IsFileInTimeRange(f, fromTime, toTime))
                .OrderBy(f => f);

            var features = new Dictionary<string, double>();
            var metadata = new Dictionary<string, object>();
            string? version = null;
            string? checksum = null;

            foreach (var file in featureFiles)
            {
                var content = await File.ReadAllTextAsync(file, cancellationToken);
                var featureSet = JsonSerializer.Deserialize<FeatureSet>(content);
                
                if (featureSet != null)
                {
                    // Merge features (later timestamps override earlier ones)
                    foreach (var kvp in featureSet.Features)
                    {
                        features[kvp.Key] = kvp.Value;
                    }
                    
                    version = featureSet.Version;
                    checksum = featureSet.SchemaChecksum;
                }
            }

            var result = new FeatureSet
            {
                Symbol = symbol,
                Timestamp = fromTime,
                Version = version ?? "unknown",
                SchemaChecksum = checksum ?? "unknown",
                Features = features,
                Metadata = metadata
            };

            _logger.LogDebug("[FEATURES] Retrieved {Count} features for {Symbol} ({From} to {To})", 
                features.Count, symbol, fromTime, toTime);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURES] Failed to get features for {Symbol}", symbol);
            throw;
        }
    }

    public async Task SaveFeaturesAsync(FeatureSet features, CancellationToken cancellationToken = default)
    {
        try
        {
            // Validate before saving
            var isValid = await ValidateSchemaAsync(features, cancellationToken);
            if (!isValid)
            {
                throw new InvalidOperationException($"Feature validation failed for {features.Symbol}");
            }

            // Calculate checksum for the data
            features.SchemaChecksum = CalculateChecksum(features);

            var symbolPath = Path.Combine(_basePath, "features", features.Symbol);
            Directory.CreateDirectory(symbolPath);

            var fileName = $"{features.Timestamp:yyyy-MM-dd_HH-mm-ss}.json";
            var filePath = Path.Combine(symbolPath, fileName);

            var json = JsonSerializer.Serialize(features, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            await File.WriteAllTextAsync(filePath, json, cancellationToken);

            _logger.LogDebug("[FEATURES] Saved {Count} features for {Symbol} at {Timestamp}", 
                features.Features.Count, features.Symbol, features.Timestamp);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURES] Failed to save features for {Symbol}", features.Symbol);
            throw;
        }
    }

    public async Task<bool> ValidateSchemaAsync(FeatureSet features, CancellationToken cancellationToken = default)
    {
        try
        {
            var schema = await GetSchemaAsync(features.Version, cancellationToken);
            if (schema == null)
            {
                _logger.LogWarning("[FEATURES] Schema not found for version: {Version}", features.Version);
                return false;
            }

            var validationResult = ValidateFeatureSet(features, schema);
            
            if (!validationResult.IsValid)
            {
                _logger.LogWarning("[FEATURES] Validation failed for {Symbol}: {Reason}", 
                    features.Symbol, validationResult.FailureReason);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURES] Schema validation error for {Symbol}", features.Symbol);
            return false;
        }
    }

    public async Task<FeatureSchema> GetSchemaAsync(string version, CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_lock)
            {
                if (_schemaCache.TryGetValue(version, out var cachedSchema))
                {
                    return cachedSchema;
                }
            }

            var schemaPath = Path.Combine(_basePath, "schemas", $"{version}.json");
            if (!File.Exists(schemaPath))
            {
                // Create default schema if not found
                var defaultSchema = CreateDefaultSchema(version);
                await SaveSchemaAsync(defaultSchema, cancellationToken);
                return defaultSchema;
            }

            var content = await File.ReadAllTextAsync(schemaPath, cancellationToken);
            var schema = JsonSerializer.Deserialize<FeatureSchema>(content);

            if (schema != null)
            {
                lock (_lock)
                {
                    _schemaCache[version] = schema;
                }
            }

            return schema ?? CreateDefaultSchema(version);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURES] Failed to get schema for version: {Version}", version);
            return CreateDefaultSchema(version);
        }
    }

    public async Task SaveSchemaAsync(FeatureSchema schema, CancellationToken cancellationToken = default)
    {
        try
        {
            schema.Checksum = CalculateSchemaChecksum(schema);
            schema.CreatedAt = DateTime.UtcNow;

            var schemaPath = Path.Combine(_basePath, "schemas", $"{schema.Version}.json");
            var json = JsonSerializer.Serialize(schema, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            await File.WriteAllTextAsync(schemaPath, json, cancellationToken);

            lock (_lock)
            {
                _schemaCache[schema.Version] = schema;
            }

            _logger.LogInformation("[FEATURES] Saved schema version: {Version} with {Count} features", 
                schema.Version, schema.Features.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURES] Failed to save schema: {Version}", schema.Version);
            throw;
        }
    }

    private ValidationResult ValidateFeatureSet(FeatureSet features, FeatureSchema schema)
    {
        var totalFeatures = schema.Features.Count;
        var missingCount = 0;
        var outOfRangeCount = 0;
        var typeErrorCount = 0;

        foreach (var schemaFeature in schema.Features.Values)
        {
            if (!features.Features.TryGetValue(schemaFeature.Name, out var value))
            {
                if (schemaFeature.Required)
                {
                    missingCount++;
                }
                continue;
            }

            // Type validation
            if (schemaFeature.DataType == typeof(double))
            {
                if (!double.IsFinite(value))
                {
                    typeErrorCount++;
                    continue;
                }
            }

            // Range validation
            if (schemaFeature.MinValue.HasValue && value < schemaFeature.MinValue.Value)
            {
                outOfRangeCount++;
            }
            
            if (schemaFeature.MaxValue.HasValue && value > schemaFeature.MaxValue.Value)
            {
                outOfRangeCount++;
            }
        }

        var missingnessPct = totalFeatures > 0 ? (double)missingCount / totalFeatures : 0.0;
        var outOfRangePct = totalFeatures > 0 ? (double)outOfRangeCount / totalFeatures : 0.0;

        var isValid = missingnessPct <= MaxMissingnessPct && 
                      outOfRangePct <= MaxOutOfRangePct && 
                      typeErrorCount == 0;

        var reason = !isValid 
            ? $"Missingness: {missingnessPct:P2}, Out-of-range: {outOfRangePct:P2}, Type errors: {typeErrorCount}"
            : "Valid";

        return new ValidationResult
        {
            IsValid = isValid,
            FailureReason = reason,
            MissingnessPct = missingnessPct,
            OutOfRangePct = outOfRangePct,
            TypeErrorCount = typeErrorCount
        };
    }

    private FeatureSchema CreateDefaultSchema(string version)
    {
        return new FeatureSchema
        {
            Version = version,
            Features = new Dictionary<string, FeatureDefinition>
            {
                ["price"] = new() { Name = "price", DataType = typeof(double), MinValue = 0, Required = true },
                ["volume"] = new() { Name = "volume", DataType = typeof(double), MinValue = 0, Required = true },
                ["volatility"] = new() { Name = "volatility", DataType = typeof(double), MinValue = 0, Required = false },
                ["trend_strength"] = new() { Name = "trend_strength", DataType = typeof(double), MinValue = -1, MaxValue = 1, Required = false }
            }
        };
    }

    private string CalculateChecksum(FeatureSet features)
    {
        var content = JsonSerializer.Serialize(features.Features);
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(content));
        return Convert.ToHexString(hash)[..16]; // First 16 chars
    }

    private string CalculateSchemaChecksum(FeatureSchema schema)
    {
        var content = JsonSerializer.Serialize(schema.Features);
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(content));
        return Convert.ToHexString(hash)[..16]; // First 16 chars
    }

    private bool IsFileInTimeRange(string filePath, DateTime fromTime, DateTime toTime)
    {
        var fileName = Path.GetFileNameWithoutExtension(filePath);
        if (DateTime.TryParseExact(fileName, "yyyy-MM-dd_HH-mm-ss", null, System.Globalization.DateTimeStyles.None, out var fileTime))
        {
            return fileTime >= fromTime && fileTime <= toTime;
        }
        return false;
    }

    private class ValidationResult
    {
        public bool IsValid { get; set; }
        public string FailureReason { get; set; } = string.Empty;
        public double MissingnessPct { get; set; }
        public double OutOfRangePct { get; set; }
        public int TypeErrorCount { get; set; }
    }
}