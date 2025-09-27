using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Security.Cryptography;
using System.Security;
using System.Text;
using System.Text.Json;
using System.IO;
using System.Globalization;
using System.Linq;
using TradingBot.Abstractions.Helpers;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Model registry with governance and promotion thresholds
/// Manages model versioning, lineage, and automatic promotion
/// </summary>
public class ModelRegistry : IModelRegistry
{
    private readonly ILogger<ModelRegistry> _logger;
    private readonly string _basePath;
    private readonly PromotionCriteria _defaultCriteria;
    private readonly Dictionary<string, ModelArtifact> _modelCache = new();
    private readonly object _lock = new();
    private DateTime _lastPromotion = DateTime.MinValue;
    private readonly TimeSpan _promotionCooldown = TimeSpan.FromMinutes(60);

    // S109 Magic Number Constants - Hash and ID Processing
    private const int ChecksumHashLength = 16;
    private const int RuntimeSignatureLength = 16;

    // LoggerMessage delegates for CA1848 compliance - ModelRegistry
    private static readonly Action<ILogger, string, double, Exception?> ModelRetrieved =
        LoggerMessage.Define<string, double>(LogLevel.Debug, new EventId(5001, "ModelRetrieved"),
            "[REGISTRY] Retrieved model: {ModelId} (AUC: {AUC:F3})");

    private static readonly Action<ILogger, string, string, Exception?> FailedToGetModel =
        LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(5002, "FailedToGetModel"),
            "[REGISTRY] Failed to get model: {Family}_{Version}");

    private static readonly Action<ILogger, string, double, double, Exception?> ModelRegistered =
        LoggerMessage.Define<string, double, double>(LogLevel.Information, new EventId(5003, "ModelRegistered"),
            "[REGISTRY] Registered model: {ModelId} (AUC: {AUC:F3}, PR@10: {PR:F3})");

    private static readonly Action<ILogger, string, Exception?> FailedToRegisterModel =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(5004, "FailedToRegisterModel"),
            "[REGISTRY] Failed to register model: {Family}");

    private static readonly Action<ILogger, string, Exception?> PromotionBlockedByCooldown =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(5005, "PromotionBlockedByCooldown"),
            "[REGISTRY] Promotion blocked by cooldown: {ModelId}");

    private static readonly Action<ILogger, string, Exception?> ModelNotFoundForPromotion =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(5006, "ModelNotFoundForPromotion"),
            "[REGISTRY] Model not found for promotion: {ModelId}");

    private static readonly Action<ILogger, string, Exception?> ModelDoesNotMeetPromotionCriteria =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(5007, "ModelDoesNotMeetPromotionCriteria"),
            "[REGISTRY] Model does not meet promotion criteria: {ModelId}");

    private static readonly Action<ILogger, string, Exception?> ModelPromoted =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(5008, "ModelPromoted"),
            "[REGISTRY] Promoted model: {ModelId} to production");

    private static readonly Action<ILogger, string, Exception?> FailedToPromoteModel =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(5009, "FailedToPromoteModel"),
            "[REGISTRY] Failed to promote model: {ModelId}");

    private static readonly Action<ILogger, string, Exception?> FailedToGetMetrics =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(5010, "FailedToGetMetrics"),
            "[REGISTRY] Failed to get metrics for model: {ModelId}");

    private static readonly Action<ILogger, string, Exception?> FailedToParseModelMetadata =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(5011, "FailedToParseModelMetadata"),
            "[REGISTRY] Failed to parse model metadata: {File}");
            
    // Additional CA1848 compliance delegates for cleanup operations
    private static readonly Action<ILogger, Exception?> FailedToReadModelsDirectory =
        LoggerMessage.Define(LogLevel.Error, new EventId(5012, "FailedToReadModelsDirectory"),
            "[REGISTRY] Failed to read models directory");
            
    private static readonly Action<ILogger, Exception?> AccessDeniedToModelsDirectory =
        LoggerMessage.Define(LogLevel.Error, new EventId(5013, "AccessDeniedToModelsDirectory"),
            "[REGISTRY] Access denied to models directory");
            
    private static readonly Action<ILogger, string, Exception?> ModelCleanedUp =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(5014, "ModelCleanedUp"),
            "[REGISTRY] Cleaned up expired model: {ModelId}");
            
    private static readonly Action<ILogger, string, Exception?> FailedToCleanupModel =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(5015, "FailedToCleanupModel"),
            "[REGISTRY] Failed to cleanup model: {ModelId}");
            
    private static readonly Action<ILogger, string, Exception?> AccessDeniedWhenCleaningUpModel =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(5016, "AccessDeniedWhenCleaningUpModel"),
            "[REGISTRY] Access denied when cleaning up model: {ModelId}");
            
    private static readonly Action<ILogger, Exception?> IOErrorDuringCleanup =
        LoggerMessage.Define(LogLevel.Error, new EventId(5017, "IOErrorDuringCleanup"),
            "[REGISTRY] IO error during cleanup of expired models");
            
    private static readonly Action<ILogger, Exception?> AccessDeniedDuringCleanup =
        LoggerMessage.Define(LogLevel.Error, new EventId(5018, "AccessDeniedDuringCleanup"),
            "[REGISTRY] Access denied during cleanup of expired models");
            
    private static readonly Action<ILogger, Exception?> SecurityErrorDuringCleanup =
        LoggerMessage.Define(LogLevel.Error, new EventId(5019, "SecurityErrorDuringCleanup"),
            "[REGISTRY] Security error during cleanup of expired models");

    public ModelRegistry(
        ILogger<ModelRegistry> logger, 
        PromotionsConfig config,
        string basePath = "data/models")
    {
        _logger = logger;
        _basePath = basePath;
        ArgumentNullException.ThrowIfNull(config);
        _defaultCriteria = new PromotionCriteria
        {
            MinAuc = config.Auto.PromoteIf.MinAuc,
            MinPrAt10 = config.Auto.PromoteIf.MinPrAt10,
            MaxEce = config.Auto.PromoteIf.MaxEce,
            MinEdgeBps = config.Auto.PromoteIf.MinEdgeBps
        };
        
        Directory.CreateDirectory(_basePath);
        Directory.CreateDirectory(Path.Combine(_basePath, "artifacts"));
        Directory.CreateDirectory(Path.Combine(_basePath, "metadata"));
    }

    public async Task<ModelArtifact> GetModelAsync(string familyName, string version = "latest", CancellationToken cancellationToken = default)
    {
        try
        {
            var cacheKey = $"{familyName}_{version}";
            
            lock (_lock)
            {
                if (_modelCache.TryGetValue(cacheKey, out var cachedModel))
                {
                    return cachedModel;
                }
            }

            string modelId;
            if (version == "latest")
            {
                modelId = await FindLatestModelAsync(familyName, cancellationToken).ConfigureAwait(false);
                if (string.IsNullOrEmpty(modelId))
                {
                    throw new FileNotFoundException($"No models found for family: {familyName}");
                }
            }
            else
            {
                modelId = $"{familyName}_{version}";
            }

            var metadataPath = Path.Combine(_basePath, "metadata", $"{modelId}.json");
            if (!File.Exists(metadataPath))
            {
                throw new FileNotFoundException($"Model metadata not found: {modelId}");
            }

            var metadataContent = await File.ReadAllTextAsync(metadataPath, cancellationToken).ConfigureAwait(false);
            var model = JsonSerializationHelper.Deserialize<ModelArtifact>(metadataContent);
            
            if (model == null)
            {
                throw new InvalidDataException($"Invalid model metadata: {modelId}");
            }

            // Load model data if needed
            var artifactPath = Path.Combine(_basePath, "artifacts", $"{modelId}.dat");
            if (File.Exists(artifactPath))
            {
                model.ModelData = await File.ReadAllBytesAsync(artifactPath, cancellationToken).ConfigureAwait(false);
            }

            // Verify checksum
            if (!VerifyModelChecksum(model))
            {
                throw new InvalidDataException($"Model checksum verification failed: {modelId}");
            }

            lock (_lock)
            {
                _modelCache[cacheKey] = model;
            }

            ModelRetrieved(_logger, model.Id, model.Metrics.AUC, null);

            return model;
        }
        catch (FileNotFoundException ex)
        {
            FailedToGetModel(_logger, familyName, version, ex);
            throw new InvalidOperationException($"Model retrieval failed for {familyName}:{version}", ex);
        }
        catch (InvalidDataException ex)
        {
            FailedToGetModel(_logger, familyName, version, ex);
            throw new InvalidOperationException($"Model retrieval failed for {familyName}:{version}", ex);
        }
        catch (JsonException ex)
        {
            FailedToGetModel(_logger, familyName, version, ex);
            throw new InvalidOperationException($"Model retrieval failed for {familyName}:{version}", ex);
        }
        catch (IOException ex)
        {
            FailedToGetModel(_logger, familyName, version, ex);
            throw new InvalidOperationException($"Model retrieval failed for {familyName}:{version}", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            FailedToGetModel(_logger, familyName, version, ex);
            throw new InvalidOperationException($"Model retrieval failed for {familyName}:{version}", ex);
        }
    }

    public async Task<ModelArtifact> RegisterModelAsync(ModelRegistration registration, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(registration);
        try
        {
            var modelId = GenerateModelId(registration.FamilyName);
            var version = ExtractVersionFromId(modelId);

            var model = new ModelArtifact
            {
                Id = modelId,
                Version = version,
                CreatedAt = DateTime.UtcNow,
                TrainingWindow = registration.TrainingWindow,
                FeaturesVersion = registration.FeaturesVersion,
                SchemaChecksum = CalculateSchemaChecksum(registration.FeaturesVersion),
                Metrics = registration.Metrics,
                CalibrationMapId = $"{modelId}_calibration",
                RuntimeSignature = CalculateRuntimeSignature(registration.ModelData),
                ModelData = registration.ModelData
            };

            // Calculate and set checksum
            model.Checksum = CalculateModelChecksum(model);

            // Save metadata
            var metadataPath = Path.Combine(_basePath, "metadata", $"{modelId}.json");
            var metadataJson = JsonSerializationHelper.SerializePretty(model);
            await File.WriteAllTextAsync(metadataPath, metadataJson, cancellationToken).ConfigureAwait(false);

            // Save model artifact
            var artifactPath = Path.Combine(_basePath, "artifacts", $"{modelId}.dat");
            await File.WriteAllBytesAsync(artifactPath, registration.ModelData, cancellationToken).ConfigureAwait(false);

            // Update cache
            lock (_lock)
            {
                _modelCache[$"{registration.FamilyName}_latest"] = model;
                _modelCache[$"{registration.FamilyName}_{version}"] = model;
            }

            ModelRegistered(_logger, modelId, model.Metrics.AUC, model.Metrics.PrAt10, null);

            // Check for automatic promotion
            if (ShouldAutoPromote(model))
            {
                await PromoteModelAsync(modelId, _defaultCriteria, cancellationToken).ConfigureAwait(false);
            }

            return model;
        }
        catch (IOException ex)
        {
            FailedToRegisterModel(_logger, registration.FamilyName, ex);
            throw new InvalidOperationException($"Model registration failed for {registration.FamilyName}", ex);
        }
        catch (JsonException ex)
        {
            FailedToRegisterModel(_logger, registration.FamilyName, ex);
            throw new InvalidOperationException($"Model registration failed for {registration.FamilyName}", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            FailedToRegisterModel(_logger, registration.FamilyName, ex);
            throw new InvalidOperationException($"Model registration failed for {registration.FamilyName}", ex);
        }
        catch (ArgumentException ex)
        {
            FailedToRegisterModel(_logger, registration.FamilyName, ex);
            throw new InvalidOperationException($"Model registration failed for {registration.FamilyName}", ex);
        }
    }

    public async Task<bool> PromoteModelAsync(string modelId, PromotionCriteria criteria, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelId);
        ArgumentNullException.ThrowIfNull(criteria);
        try
        {
            // Check cooldown
            if (DateTime.UtcNow - _lastPromotion < _promotionCooldown)
            {
                PromotionBlockedByCooldown(_logger, modelId, null);
                return false;
            }

            var model = await GetModelByIdAsync(modelId, cancellationToken).ConfigureAwait(false);
            if (model == null)
            {
                ModelNotFoundForPromotion(_logger, modelId, null);
                return false;
            }

            // Check promotion criteria
            if (!MeetsPromotionCriteria(model, criteria))
            {
                ModelDoesNotMeetPromotionCriteria(_logger, modelId, null);
                return false;
            }

            // Perform promotion
            var familyName = ExtractFamilyFromId(modelId);
            var promotedPath = Path.Combine(_basePath, "promoted", $"{familyName}_promoted.json");
            var promotedDir = Path.GetDirectoryName(promotedPath);
            if (!string.IsNullOrEmpty(promotedDir))
            {
                Directory.CreateDirectory(promotedDir);
            }

            var promotionRecord = new PromotionRecord
            {
                ModelId = modelId,
                PromotedAt = DateTime.UtcNow,
                Criteria = criteria,
                Metrics = model.Metrics,
                PromotedBy = "auto-promotion"
            };

            var json = JsonSerializationHelper.SerializePretty(promotionRecord);
            await File.WriteAllTextAsync(promotedPath, json, cancellationToken).ConfigureAwait(false);

            _lastPromotion = DateTime.UtcNow;

            ModelPromoted(_logger, modelId, null);
            return true;
        }
        catch (DirectoryNotFoundException ex)
        {
            FailedToPromoteModel(_logger, modelId, ex);
            return false;
        }
        catch (UnauthorizedAccessException ex)
        {
            FailedToPromoteModel(_logger, modelId, ex);
            return false;
        }
        catch (IOException ex)
        {
            FailedToPromoteModel(_logger, modelId, ex);
            return false;
        }
        catch (JsonException ex)
        {
            FailedToPromoteModel(_logger, modelId, ex);
            return false;
        }
    }

    public async Task<ModelMetrics> GetModelMetricsAsync(string modelId, CancellationToken cancellationToken = default)
    {
        try
        {
            var model = await GetModelByIdAsync(modelId, cancellationToken).ConfigureAwait(false);
            return model?.Metrics ?? new ModelMetrics();
        }
        catch (FileNotFoundException ex)
        {
            FailedToGetMetrics(_logger, modelId, ex);
            return new ModelMetrics();
        }
        catch (IOException ex)
        {
            FailedToGetMetrics(_logger, modelId, ex);
            return new ModelMetrics();
        }
        catch (JsonException ex)
        {
            FailedToGetMetrics(_logger, modelId, ex);
            return new ModelMetrics();
        }
        catch (UnauthorizedAccessException ex)
        {
            FailedToGetMetrics(_logger, modelId, ex);
            return new ModelMetrics();
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return new ModelMetrics();
        }
    }

    /// <summary>
    /// Gets all active models from the registry
    /// </summary>
    public async Task<IEnumerable<ModelArtifact>> GetActiveModelsAsync(CancellationToken cancellationToken = default)
    {
        var activeModels = new List<ModelArtifact>();
        
        try
        {
            var metadataDir = Path.Combine(_basePath, "metadata");
            if (!Directory.Exists(metadataDir))
            {
                return activeModels;
            }

            var files = Directory.GetFiles(metadataDir, "*.json");
            
            foreach (var file in files)
            {
                try
                {
                    var content = await File.ReadAllTextAsync(file, cancellationToken).ConfigureAwait(false);
                    var model = JsonSerializationHelper.Deserialize<ModelArtifact>(content);
                    
                    if (model != null)
                    {
                        activeModels.Add(model);
                    }
                }
                catch (IOException ex)
                {
                    FailedToParseModelMetadata(_logger, file, ex);
                }
                catch (JsonException ex)
                {
                    FailedToParseModelMetadata(_logger, file, ex);
                }
                catch (UnauthorizedAccessException ex)
                {
                    FailedToParseModelMetadata(_logger, file, ex);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    break; // Stop processing files on cancellation
                }
            }
        }
        catch (IOException ex)
        {
            FailedToReadModelsDirectory(_logger, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            AccessDeniedToModelsDirectory(_logger, ex);
        }
        
        return activeModels;
    }

    /// <summary>
    /// Cleans up expired models that are past their retention period
    /// </summary>
    public async Task CleanupExpiredModelsAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var activeModels = await GetActiveModelsAsync(cancellationToken).ConfigureAwait(false);
            var retentionPeriod = TimeSpan.FromDays(30); // Keep models for 30 days
            var cutoffDate = DateTime.UtcNow - retentionPeriod;
            
            // Get expired models first to avoid complex LINQ chains with side effects
            var expiredModels = activeModels.Where(m => m.CreatedAt < cutoffDate).ToList();
            
            // Process expired models for cleanup - no side effects in LINQ chain
            var cleanupTasks = expiredModels.Select(model => 
            {
                var metadataPath = Path.Combine(_basePath, "metadata", $"{model.Id}.json");
                var artifactPath = Path.Combine(_basePath, "artifacts", $"{model.Id}.dat");
                
                return new { Model = model, MetadataPath = metadataPath, ArtifactPath = artifactPath };
            }).ToList();
            
            foreach (var item in cleanupTasks)
            {
                try
                {
                    if (File.Exists(item.MetadataPath))
                    {
                        File.Delete(item.MetadataPath);
                    }
                    
                    if (File.Exists(item.ArtifactPath))
                    {
                        File.Delete(item.ArtifactPath);
                    }
                    
                    lock (_lock)
                    {
                        _modelCache.Remove(item.Model.Id);
                    }
                    
                    ModelCleanedUp(_logger, item.Model.Id, null);
                }
                catch (IOException ex)
                {
                    FailedToCleanupModel(_logger, item.Model.Id, ex);
                }
                catch (UnauthorizedAccessException ex)
                {
                    AccessDeniedWhenCleaningUpModel(_logger, item.Model.Id, ex);
                }
            }
        }
        catch (IOException ex)
        {
            IOErrorDuringCleanup(_logger, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            AccessDeniedDuringCleanup(_logger, ex);
        }
        catch (SecurityException ex)
        {
            SecurityErrorDuringCleanup(_logger, ex);
        }
    }

    private async Task<ModelArtifact?> GetModelByIdAsync(string modelId, CancellationToken cancellationToken)
    {
        var metadataPath = Path.Combine(_basePath, "metadata", $"{modelId}.json");
        if (!File.Exists(metadataPath))
        {
            return null;
        }

        var content = await File.ReadAllTextAsync(metadataPath, cancellationToken).ConfigureAwait(false);
        return JsonSerializationHelper.Deserialize<ModelArtifact>(content);
    }

    private async Task<string> FindLatestModelAsync(string familyName, CancellationToken cancellationToken)
    {
        var metadataDir = Path.Combine(_basePath, "metadata");
        if (!Directory.Exists(metadataDir))
        {
            return string.Empty;
        }

        var pattern = $"{familyName}_*.json";
        var files = Directory.GetFiles(metadataDir, pattern);
        
        ModelArtifact? latestModel = null;
        string latestId = string.Empty;

        foreach (var file in files)
        {
            try
            {
                var content = await File.ReadAllTextAsync(file, cancellationToken).ConfigureAwait(false);
                var model = JsonSerializationHelper.Deserialize<ModelArtifact>(content);
                
                if (model != null && (latestModel == null || model.CreatedAt > latestModel.CreatedAt))
                {
                    latestModel = model;
                    latestId = model.Id;
                }
            }
            catch (IOException ex)
            {
                FailedToParseModelMetadata(_logger, file, ex);
            }
            catch (JsonException ex)
            {
                FailedToParseModelMetadata(_logger, file, ex);
            }
            catch (UnauthorizedAccessException ex)
            {
                FailedToParseModelMetadata(_logger, file, ex);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                break; // Stop processing files on cancellation
            }
        }

        return latestId;
    }

    private static bool MeetsPromotionCriteria(ModelArtifact model, PromotionCriteria criteria)
    {
        return model.Metrics.AUC >= criteria.MinAuc &&
               model.Metrics.PrAt10 >= criteria.MinPrAt10 &&
               model.Metrics.ECE <= criteria.MaxEce &&
               model.Metrics.EdgeBps >= criteria.MinEdgeBps;
    }

    private bool ShouldAutoPromote(ModelArtifact model)
    {
        return MeetsPromotionCriteria(model, _defaultCriteria) &&
               DateTime.UtcNow - _lastPromotion >= _promotionCooldown;
    }

    private static string GenerateModelId(string familyName)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        return $"{familyName}_{timestamp}";
    }

    private static string ExtractVersionFromId(string modelId)
    {
        var parts = modelId.Split('_');
        return parts.Length > 1 ? parts[^1] : "v1";
    }

    private static string ExtractFamilyFromId(string modelId)
    {
        var lastUnderscore = modelId.LastIndexOf('_');
        return lastUnderscore > 0 ? modelId[..lastUnderscore] : modelId;
    }

    private static string CalculateModelChecksum(ModelArtifact model)
    {
        var data = model.ModelData ?? Array.Empty<byte>();
        var hash = SHA256.HashData(data);
        return Convert.ToHexString(hash)[..ChecksumHashLength];
    }

    private static string CalculateSchemaChecksum(string featuresVersion)
    {
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(featuresVersion));
        return Convert.ToHexString(hash)[..ChecksumHashLength];
    }

    private static string CalculateRuntimeSignature(byte[] modelData)
    {
        var hash = SHA256.HashData(modelData);
        return Convert.ToBase64String(hash)[..RuntimeSignatureLength];
    }

    private static bool VerifyModelChecksum(ModelArtifact model)
    {
        if (model.ModelData == null) return true;
        
        var calculatedChecksum = CalculateModelChecksum(model);
        return string.Equals(calculatedChecksum, model.Checksum, StringComparison.OrdinalIgnoreCase);
    }

    private sealed class PromotionRecord
    {
        public string ModelId { get; set; } = string.Empty;
        public DateTime PromotedAt { get; set; }
        public PromotionCriteria Criteria { get; set; } = new();
        public ModelMetrics Metrics { get; set; } = new();
        public string PromotedBy { get; set; } = string.Empty;
    }
}