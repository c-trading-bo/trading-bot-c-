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

    public ModelRegistry(
        ILogger<ModelRegistry> logger, 
        PromotionsConfig config,
        string basePath = "data/models")
    {
        _logger = logger;
        _basePath = basePath;
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
                modelId = await FindLatestModelAsync(familyName, cancellationToken);
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

            var metadataContent = await File.ReadAllTextAsync(metadataPath, cancellationToken);
            var model = JsonSerializer.Deserialize<ModelArtifact>(metadataContent);
            
            if (model == null)
            {
                throw new InvalidDataException($"Invalid model metadata: {modelId}");
            }

            // Load model data if needed
            var artifactPath = Path.Combine(_basePath, "artifacts", $"{modelId}.dat");
            if (File.Exists(artifactPath))
            {
                model.ModelData = await File.ReadAllBytesAsync(artifactPath, cancellationToken);
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

            _logger.LogDebug("[REGISTRY] Retrieved model: {ModelId} (AUC: {AUC:F3})", 
                model.Id, model.Metrics.AUC);

            return model;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REGISTRY] Failed to get model: {Family}_{Version}", familyName, version);
            throw;
        }
    }

    public async Task<ModelArtifact> RegisterModelAsync(ModelRegistration registration, CancellationToken cancellationToken = default)
    {
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
            var metadataJson = JsonSerializer.Serialize(model, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            await File.WriteAllTextAsync(metadataPath, metadataJson, cancellationToken);

            // Save model artifact
            var artifactPath = Path.Combine(_basePath, "artifacts", $"{modelId}.dat");
            await File.WriteAllBytesAsync(artifactPath, registration.ModelData, cancellationToken);

            // Update cache
            lock (_lock)
            {
                _modelCache[$"{registration.FamilyName}_latest"] = model;
                _modelCache[$"{registration.FamilyName}_{version}"] = model;
            }

            _logger.LogInformation("[REGISTRY] Registered model: {ModelId} (AUC: {AUC:F3}, PR@10: {PR:F3})", 
                modelId, model.Metrics.AUC, model.Metrics.PrAt10);

            // Check for automatic promotion
            if (ShouldAutoPromote(model))
            {
                await PromoteModelAsync(modelId, _defaultCriteria, cancellationToken);
            }

            return model;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REGISTRY] Failed to register model: {Family}", registration.FamilyName);
            throw;
        }
    }

    public async Task<bool> PromoteModelAsync(string modelId, PromotionCriteria criteria, CancellationToken cancellationToken = default)
    {
        try
        {
            // Check cooldown
            if (DateTime.UtcNow - _lastPromotion < _promotionCooldown)
            {
                _logger.LogWarning("[REGISTRY] Promotion blocked by cooldown: {ModelId}", modelId);
                return false;
            }

            var model = await GetModelByIdAsync(modelId, cancellationToken);
            if (model == null)
            {
                _logger.LogWarning("[REGISTRY] Model not found for promotion: {ModelId}", modelId);
                return false;
            }

            // Check promotion criteria
            if (!MeetsPromotionCriteria(model, criteria))
            {
                _logger.LogWarning("[REGISTRY] Model does not meet promotion criteria: {ModelId}", modelId);
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

            var json = JsonSerializer.Serialize(promotionRecord, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            await File.WriteAllTextAsync(promotedPath, json, cancellationToken);

            _lastPromotion = DateTime.UtcNow;

            _logger.LogInformation("[REGISTRY] Promoted model: {ModelId} to production", modelId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REGISTRY] Failed to promote model: {ModelId}", modelId);
            return false;
        }
    }

    public async Task<ModelMetrics> GetModelMetricsAsync(string modelId, CancellationToken cancellationToken = default)
    {
        try
        {
            var model = await GetModelByIdAsync(modelId, cancellationToken);
            return model?.Metrics ?? new ModelMetrics();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REGISTRY] Failed to get metrics for model: {ModelId}", modelId);
            return new ModelMetrics();
        }
    }

    private async Task<ModelArtifact?> GetModelByIdAsync(string modelId, CancellationToken cancellationToken)
    {
        var metadataPath = Path.Combine(_basePath, "metadata", $"{modelId}.json");
        if (!File.Exists(metadataPath))
        {
            return null;
        }

        var content = await File.ReadAllTextAsync(metadataPath, cancellationToken);
        return JsonSerializer.Deserialize<ModelArtifact>(content);
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
                var content = await File.ReadAllTextAsync(file, cancellationToken);
                var model = JsonSerializer.Deserialize<ModelArtifact>(content);
                
                if (model != null && (latestModel == null || model.CreatedAt > latestModel.CreatedAt))
                {
                    latestModel = model;
                    latestId = model.Id;
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[REGISTRY] Failed to parse model metadata: {File}", file);
            }
        }

        return latestId;
    }

    private bool MeetsPromotionCriteria(ModelArtifact model, PromotionCriteria criteria)
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

    private string GenerateModelId(string familyName)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        return $"{familyName}_{timestamp}";
    }

    private string ExtractVersionFromId(string modelId)
    {
        var parts = modelId.Split('_');
        return parts.Length > 1 ? parts[^1] : "v1";
    }

    private string ExtractFamilyFromId(string modelId)
    {
        var lastUnderscore = modelId.LastIndexOf('_');
        return lastUnderscore > 0 ? modelId[..lastUnderscore] : modelId;
    }

    private string CalculateModelChecksum(ModelArtifact model)
    {
        var data = model.ModelData ?? Array.Empty<byte>();
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(data);
        return Convert.ToHexString(hash)[..16];
    }

    private string CalculateSchemaChecksum(string featuresVersion)
    {
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(featuresVersion));
        return Convert.ToHexString(hash)[..16];
    }

    private string CalculateRuntimeSignature(byte[] modelData)
    {
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(modelData);
        return Convert.ToBase64String(hash)[..16];
    }

    private bool VerifyModelChecksum(ModelArtifact model)
    {
        if (model.ModelData == null) return true;
        
        var calculatedChecksum = CalculateModelChecksum(model);
        return string.Equals(calculatedChecksum, model.Checksum, StringComparison.OrdinalIgnoreCase);
    }

    private class PromotionRecord
    {
        public string ModelId { get; set; } = string.Empty;
        public DateTime PromotedAt { get; set; }
        public PromotionCriteria Criteria { get; set; } = new();
        public ModelMetrics Metrics { get; set; } = new();
        public string PromotedBy { get; set; } = string.Empty;
    }
}