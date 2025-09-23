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

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    private const double MaxMissingnessPct = 0.5;
    private const double MaxOutOfRangePct = 1.0;
    private const int ChecksumLength = 16;

    // LoggerMessage delegates for CA1848 compliance - FeatureStore  
    private static readonly Action<ILogger, string, Exception?> NoFeatureDataFound =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(3001, "NoFeatureDataFound"),
            "[FEATURES] No feature data found for symbol: {Symbol}");
            
    private static readonly Action<ILogger, int, string, DateTime, DateTime, Exception?> FeaturesRetrieved =
        LoggerMessage.Define<int, string, DateTime, DateTime>(LogLevel.Debug, new EventId(3002, "FeaturesRetrieved"),
            "[FEATURES] Retrieved {Count} features for {Symbol} ({From} to {To})");
            
    private static readonly Action<ILogger, string, DateTime, DateTime, Exception?> FeatureRetrievalFailed =
        LoggerMessage.Define<string, DateTime, DateTime>(LogLevel.Error, new EventId(3003, "FeatureRetrievalFailed"),
            "[FEATURES] Failed to get features for {Symbol} from {FromTime} to {ToTime}");
            
    private static readonly Action<ILogger, int, string, DateTime, Exception?> FeaturesSaved =
        LoggerMessage.Define<int, string, DateTime>(LogLevel.Debug, new EventId(3004, "FeaturesSaved"),
            "[FEATURES] Saved {Count} features for {Symbol} at {Timestamp}");
            
    private static readonly Action<ILogger, string, DateTime, Exception?> FeatureSavingFailed =
        LoggerMessage.Define<string, DateTime>(LogLevel.Error, new EventId(3005, "FeatureSavingFailed"),
            "[FEATURES] Failed to save features for {Symbol} at {Timestamp}");
            
    private static readonly Action<ILogger, string, Exception?> SchemaNotFound =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(3006, "SchemaNotFound"),
            "[FEATURES] Schema not found for version: {Version}");
            
    private static readonly Action<ILogger, string, string, Exception?> ValidationFailed =
        LoggerMessage.Define<string, string>(LogLevel.Warning, new EventId(3007, "ValidationFailed"),
            "[FEATURES] Validation failed for {Symbol}: {Reason}");
            
    private static readonly Action<ILogger, string, Exception?> SchemaValidationError =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3008, "SchemaValidationError"),
            "[FEATURES] Schema validation error for {Symbol}");
            
    private static readonly Action<ILogger, string, int, Exception?> SchemaSaved =
        LoggerMessage.Define<string, int>(LogLevel.Information, new EventId(3009, "SchemaSaved"),
            "[FEATURES] Saved schema version: {Version} with {Count} features");
            
    private static readonly Action<ILogger, string, Exception?> SchemaSaveFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3010, "SchemaSaveFailed"),
            "[FEATURES] Failed to save schema: {Version}");
            
    private static readonly Action<ILogger, string, Exception?> SchemaRetrievalFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3011, "SchemaRetrievalFailed"),
            "[FEATURES] Failed to get schema for version: {Version}");

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
                NoFeatureDataFound(_logger, symbol, null);
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
                var content = await File.ReadAllTextAsync(file, cancellationToken).ConfigureAwait(false);
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
                SchemaChecksum = checksum ?? "unknown"
            };
            
            // Populate read-only collections
            if (features.Count > 0)
            {
                foreach (var kvp in features)
                {
                    result.Features[kvp.Key] = kvp.Value;
                }
            }
            if (metadata.Count > 0)
            {
                foreach (var kvp in metadata)
                {
                    result.Metadata[kvp.Key] = kvp.Value;
                }
            }

            FeaturesRetrieved(_logger, features.Count, symbol, fromTime, toTime, null);

            return result;
        }
        catch (Exception ex)
        {
            FeatureRetrievalFailed(_logger, symbol, fromTime, toTime, ex);
            throw new InvalidOperationException($"Feature retrieval failed for symbol {symbol}", ex);
        }
    }

    public async Task SaveFeaturesAsync(FeatureSet features, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(features);

        try
        {
            // Validate before saving
            var isValid = await ValidateSchemaAsync(features, cancellationToken).ConfigureAwait(false);
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

            var json = JsonSerializer.Serialize(features, JsonOptions);

            await File.WriteAllTextAsync(filePath, json, cancellationToken).ConfigureAwait(false);

            FeaturesSaved(_logger, features.Features.Count, features.Symbol, features.Timestamp, null);
        }
        catch (Exception ex)
        {
            FeatureSavingFailed(_logger, features.Symbol, features.Timestamp, ex);
            throw new InvalidOperationException($"Feature saving failed for symbol {features.Symbol}", ex);
        }
    }

    public async Task<bool> ValidateSchemaAsync(FeatureSet features, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(features);

        try
        {
            var schema = await GetSchemaAsync(features.Version, cancellationToken).ConfigureAwait(false);
            if (schema == null)
            {
                SchemaNotFound(_logger, features.Version, null);
                return false;
            }

            var validationResult = ValidateFeatureSet(features, schema);
            
            if (!validationResult.IsValid)
            {
                ValidationFailed(_logger, features.Symbol, validationResult.FailureReason, null);
                return false;
            }

            return true;
        }
        catch (ArgumentException ex)
        {
            SchemaValidationError(_logger, features.Symbol, ex);
            return false;
        }
        catch (InvalidOperationException ex)
        {
            SchemaValidationError(_logger, features.Symbol, ex);
            return false;
        }
        catch (System.IO.IOException ex)
        {
            SchemaValidationError(_logger, features.Symbol, ex);
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
                await SaveSchemaAsync(defaultSchema, cancellationToken).ConfigureAwait(false);
                return defaultSchema;
            }

            var content = await File.ReadAllTextAsync(schemaPath, cancellationToken).ConfigureAwait(false);
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
        catch (ArgumentException ex)
        {
            SchemaRetrievalFailed(_logger, version, ex);
            return CreateDefaultSchema(version);
        }
        catch (InvalidOperationException ex)
        {
            SchemaRetrievalFailed(_logger, version, ex);
            return CreateDefaultSchema(version);
        }
        catch (System.IO.IOException ex)
        {
            SchemaRetrievalFailed(_logger, version, ex);
            return CreateDefaultSchema(version);
        }
    }

    public async Task SaveSchemaAsync(FeatureSchema schema, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(schema);

        try
        {
            schema.Checksum = CalculateSchemaChecksum(schema);
            schema.CreatedAt = DateTime.UtcNow;

            var schemaPath = Path.Combine(_basePath, "schemas", $"{schema.Version}.json");
            var json = JsonSerializer.Serialize(schema, JsonOptions);

            await File.WriteAllTextAsync(schemaPath, json, cancellationToken).ConfigureAwait(false);

            lock (_lock)
            {
                _schemaCache[schema.Version] = schema;
            }

            SchemaSaved(_logger, schema.Version, schema.Features.Count, null);
        }
        catch (Exception ex)
        {
            SchemaSaveFailed(_logger, schema.Version, ex);
            throw new InvalidOperationException($"Schema save failed for version {schema.Version}", ex);
        }
    }

    /// <summary>
    /// Optimizes storage by compacting old feature files and removing duplicates
    /// </summary>
    public async Task OptimizeStorageAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var featureDir = Path.Combine(_basePath, "features");
            if (!Directory.Exists(featureDir))
            {
                return;
            }

            var cutoffDate = DateTime.UtcNow.AddDays(-7); // Keep last 7 days uncompacted
            var symbolDirs = Directory.GetDirectories(featureDir);
            
            foreach (var symbolDir in symbolDirs)
            {
                var files = Directory.GetFiles(symbolDir, "*.json")
                    .Where(f => IsOldFile(f, cutoffDate))
                    .ToArray();
                
                if (files.Length > 100) // Only optimize if we have many files
                {
                    await CompactFeatureFilesAsync(symbolDir, files, cancellationToken).ConfigureAwait(false);
                }
            }
            
            _logger.LogInformation("[FEATURES] Storage optimization completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURES] Storage optimization failed");
        }
    }

    private async Task CompactFeatureFilesAsync(string symbolDir, string[] files, CancellationToken cancellationToken)
    {
        try
        {
            var symbol = Path.GetFileName(symbolDir);
            var compactedFeatures = new List<FeatureSet>();
            
            foreach (var file in files.Take(50)) // Compact in batches
            {
                try
                {
                    var content = await File.ReadAllTextAsync(file, cancellationToken).ConfigureAwait(false);
                    var features = JsonSerializer.Deserialize<FeatureSet>(content);
                    if (features != null)
                    {
                        compactedFeatures.Add(features);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[FEATURES] Failed to read feature file for compaction: {File}", file);
                }
            }
            
            if (compactedFeatures.Count > 0)
            {
                var compactedFile = Path.Combine(symbolDir, $"compacted_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
                var json = JsonSerializer.Serialize(compactedFeatures, JsonOptions);
                await File.WriteAllTextAsync(compactedFile, json, cancellationToken).ConfigureAwait(false);
                
                // Remove individual files after successful compaction
                foreach (var file in files.Take(50))
                {
                    try
                    {
                        File.Delete(file);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "[FEATURES] Failed to delete compacted file: {File}", file);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURES] Failed to compact feature files in {SymbolDir}", symbolDir);
        }
    }

    private static bool IsOldFile(string filePath, DateTime cutoffDate)
    {
        try
        {
            var fileName = Path.GetFileNameWithoutExtension(filePath);
            if (fileName.StartsWith("compacted_"))
            {
                return false; // Don't re-compact already compacted files
            }
            
            if (DateTime.TryParseExact(fileName, "yyyy-MM-dd_HH-mm-ss", null, System.Globalization.DateTimeStyles.None, out var fileTime))
            {
                return fileTime < cutoffDate;
            }
        }
        catch
        {
            // If we can't parse the date, consider it old
            return true;
        }
        return false;
    }

    private static ValidationResult ValidateFeatureSet(FeatureSet features, FeatureSchema schema)
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
            if (schemaFeature.DataType == typeof(double) && !double.IsFinite(value))
            {
                typeErrorCount++;
                continue;
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

    private static FeatureSchema CreateDefaultSchema(string version)
    {
        var schema = new FeatureSchema
        {
            Version = version
        };

        // Populate the read-only Features dictionary
        schema.Features["price"] = new() { Name = "price", DataType = typeof(double), MinValue = 0, Required = true };
        schema.Features["volume"] = new() { Name = "volume", DataType = typeof(double), MinValue = 0, Required = true };
        schema.Features["volatility"] = new() { Name = "volatility", DataType = typeof(double), MinValue = 0, Required = false };
        schema.Features["trend_strength"] = new() { Name = "trend_strength", DataType = typeof(double), MinValue = -1, MaxValue = 1, Required = false };

        return schema;
    }

    private static string CalculateChecksum(FeatureSet features)
    {
        var content = JsonSerializer.Serialize(features.Features);
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(content));
        return Convert.ToHexString(hash)[..ChecksumLength]; // First 16 chars
    }

    private static string CalculateSchemaChecksum(FeatureSchema schema)
    {
        var content = JsonSerializer.Serialize(schema.Features);
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(content));
        return Convert.ToHexString(hash)[..ChecksumLength]; // First 16 chars
    }

    private static bool IsFileInTimeRange(string filePath, DateTime fromTime, DateTime toTime)
    {
        var fileName = Path.GetFileNameWithoutExtension(filePath);
        if (DateTime.TryParseExact(fileName, "yyyy-MM-dd_HH-mm-ss", null, System.Globalization.DateTimeStyles.None, out var fileTime))
        {
            return fileTime >= fromTime && fileTime <= toTime;
        }
        return false;
    }

    private sealed class ValidationResult
    {
        public bool IsValid { get; set; }
        public string FailureReason { get; set; } = string.Empty;
        public double MissingnessPct { get; set; }
        public double OutOfRangePct { get; set; }
        public int TypeErrorCount { get; set; }
    }
}