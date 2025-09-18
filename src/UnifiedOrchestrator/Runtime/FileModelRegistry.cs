using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// File-based model registry with atomic operations and checksums
/// </summary>
public class FileModelRegistry : IModelRegistry
{
    private readonly ILogger<FileModelRegistry> _logger;
    private readonly string _registryPath;
    private readonly string _artifactsPath;
    private readonly SemaphoreSlim _registryLock = new(1, 1);
    
    // Champion pointers for each algorithm
    private readonly Dictionary<string, string> _championPointers = new();
    private readonly object _championLock = new object();

    public FileModelRegistry(ILogger<FileModelRegistry> logger, string? registryPath = null)
    {
        _logger = logger;
        _registryPath = registryPath ?? Path.Combine(Directory.GetCurrentDirectory(), "model_registry");
        _artifactsPath = Path.Combine(_registryPath, "artifacts");
        
        // Ensure directories exist
        Directory.CreateDirectory(_registryPath);
        Directory.CreateDirectory(_artifactsPath);
        Directory.CreateDirectory(Path.Combine(_registryPath, "models"));
        Directory.CreateDirectory(Path.Combine(_registryPath, "promotions"));
        
        // Load champion pointers on startup
        LoadChampionPointersAsync().GetAwaiter().GetResult();
        
        _logger.LogInformation("FileModelRegistry initialized at {RegistryPath}", _registryPath);
    }

    /// <summary>
    /// Register a new model version with atomic temp->final move
    /// </summary>
    public async Task<string> RegisterModelAsync(ModelVersion model, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(model.VersionId))
        {
            model.VersionId = GenerateVersionId();
        }

        await _registryLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Validate artifact exists and compute hash
            if (!string.IsNullOrEmpty(model.ArtifactPath) && File.Exists(model.ArtifactPath))
            {
                model.ArtifactHash = await ComputeFileHashAsync(model.ArtifactPath, cancellationToken).ConfigureAwait(false);
                
                // Copy artifact to registry with atomic move
                var artifactName = $"{model.Algorithm}_{model.VersionId}_{Path.GetFileName(model.ArtifactPath)}";
                var finalArtifactPath = Path.Combine(_artifactsPath, artifactName);
                var tempArtifactPath = finalArtifactPath + ".tmp";
                
                File.Copy(model.ArtifactPath, tempArtifactPath, true);
                File.Move(tempArtifactPath, finalArtifactPath);
                
                model.ArtifactPath = finalArtifactPath;
                _logger.LogDebug("Artifact copied to registry: {Path}", finalArtifactPath);
            }

            // Write model metadata with atomic temp->final move
            var modelMetadataPath = Path.Combine(_registryPath, "models", $"{model.Algorithm}_{model.VersionId}.json");
            var tempMetadataPath = modelMetadataPath + ".tmp";
            
            var json = JsonSerializer.Serialize(model, new JsonSerializerOptions 
            { 
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            
            await File.WriteAllTextAsync(tempMetadataPath, json, cancellationToken).ConfigureAwait(false);
            File.Move(tempMetadataPath, modelMetadataPath);
            
            _logger.LogInformation("Registered model {Algorithm} version {VersionId} with hash {Hash}", 
                model.Algorithm, model.VersionId, model.ArtifactHash[..8]);
            
            return model.VersionId;
        }
        finally
        {
            _registryLock.Release();
        }
    }

    /// <summary>
    /// Get the current champion model for an algorithm
    /// </summary>
    public async Task<ModelVersion?> GetChampionAsync(string algorithm, CancellationToken cancellationToken = default)
    {
        string? championVersionId;
        lock (_championLock)
        {
            _championPointers.TryGetValue(algorithm, out championVersionId);
        }

        if (string.IsNullOrEmpty(championVersionId))
        {
            return null;
        }

        return await GetModelAsync(championVersionId, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Get a specific model version
    /// </summary>
    public async Task<ModelVersion?> GetModelAsync(string versionId, CancellationToken cancellationToken = default)
    {
        var modelFiles = Directory.GetFiles(Path.Combine(_registryPath, "models"), $"*_{versionId}.json");
        if (modelFiles.Length == 0)
        {
            return null;
        }

        try
        {
            var json = await File.ReadAllTextAsync(modelFiles[0], cancellationToken).ConfigureAwait(false);
            return JsonSerializer.Deserialize<ModelVersion>(json, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to deserialize model version {VersionId}", versionId);
            return null;
        }
    }

    /// <summary>
    /// Get all model versions for an algorithm
    /// </summary>
    public async Task<IReadOnlyList<ModelVersion>> GetModelsAsync(string algorithm, CancellationToken cancellationToken = default)
    {
        var models = new List<ModelVersion>();
        var modelFiles = Directory.GetFiles(Path.Combine(_registryPath, "models"), $"{algorithm}_*.json");
        
        foreach (var file in modelFiles)
        {
            try
            {
                var json = await File.ReadAllTextAsync(file, cancellationToken).ConfigureAwait(false);
                var model = JsonSerializer.Deserialize<ModelVersion>(json, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
                
                if (model != null)
                {
                    models.Add(model);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load model from {File}", file);
            }
        }
        
        return models.OrderByDescending(m => m.CreatedAt).ToList();
    }

    /// <summary>
    /// Promote a challenger to champion with atomic operation
    /// </summary>
    public async Task<bool> PromoteToChampionAsync(string algorithm, string challengerVersionId, PromotionRecord promotionRecord, CancellationToken cancellationToken = default)
    {
        await _registryLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Verify challenger model exists
            var challengerModel = await GetModelAsync(challengerVersionId, cancellationToken).ConfigureAwait(false);
            if (challengerModel == null)
            {
                _logger.LogError("Cannot promote non-existent challenger {VersionId} for {Algorithm}", challengerVersionId, algorithm);
                return false;
            }

            // Get current champion
            var currentChampion = await GetChampionAsync(algorithm, cancellationToken).ConfigureAwait(false);
            promotionRecord.FromVersionId = currentChampion?.VersionId ?? "none";
            promotionRecord.ToVersionId = challengerVersionId;

            // Write promotion record
            var promotionPath = Path.Combine(_registryPath, "promotions", $"{algorithm}_{promotionRecord.Id}.json");
            var tempPromotionPath = promotionPath + ".tmp";
            
            var promotionJson = JsonSerializer.Serialize(promotionRecord, new JsonSerializerOptions 
            { 
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            
            await File.WriteAllTextAsync(tempPromotionPath, promotionJson, cancellationToken).ConfigureAwait(false);
            File.Move(tempPromotionPath, promotionPath);

            // Update champion pointer atomically
            var championPointerPath = Path.Combine(_registryPath, $"{algorithm}_champion.txt");
            var tempChampionPath = championPointerPath + ".tmp";
            
            await File.WriteAllTextAsync(tempChampionPath, challengerVersionId, cancellationToken).ConfigureAwait(false);
            File.Move(tempChampionPath, championPointerPath);

            // Update in-memory champion pointer
            lock (_championLock)
            {
                _championPointers[algorithm] = challengerVersionId;
            }

            // Mark challenger as promoted
            challengerModel.IsPromoted = true;
            challengerModel.PromotedAt = DateTime.UtcNow;
            await UpdateModelMetadataAsync(challengerModel, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("Promoted {Algorithm} champion: {From} → {To}", 
                algorithm, promotionRecord.FromVersionId, challengerVersionId);
            
            return true;
        }
        finally
        {
            _registryLock.Release();
        }
    }

    /// <summary>
    /// Rollback to previous champion
    /// </summary>
    public async Task<bool> RollbackToPreviousAsync(string algorithm, string reason, CancellationToken cancellationToken = default)
    {
        var promotionHistory = await GetPromotionHistoryAsync(algorithm, cancellationToken).ConfigureAwait(false);
        var lastPromotion = promotionHistory.FirstOrDefault();
        
        if (lastPromotion == null || string.IsNullOrEmpty(lastPromotion.FromVersionId) || lastPromotion.FromVersionId == "none")
        {
            _logger.LogError("Cannot rollback {Algorithm}: no previous champion found", algorithm);
            return false;
        }

        // Create rollback promotion record
        var rollbackRecord = new PromotionRecord
        {
            Algorithm = algorithm,
            FromVersionId = lastPromotion.ToVersionId,
            ToVersionId = lastPromotion.FromVersionId,
            Reason = $"ROLLBACK: {reason}",
            PromotedBy = "SYSTEM_ROLLBACK",
            WasFlat = true, // Assume rollback is safe
            MarketSession = "ROLLBACK"
        };

        // Mark previous promotion as rolled back
        lastPromotion.WasRolledBack = true;
        lastPromotion.RolledBackAt = DateTime.UtcNow;
        lastPromotion.RollbackReason = reason;
        await UpdatePromotionRecordAsync(lastPromotion, cancellationToken).ConfigureAwait(false);

        return await PromoteToChampionAsync(algorithm, lastPromotion.FromVersionId, rollbackRecord, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Get promotion history for an algorithm
    /// </summary>
    public async Task<IReadOnlyList<PromotionRecord>> GetPromotionHistoryAsync(string algorithm, CancellationToken cancellationToken = default)
    {
        var promotions = new List<PromotionRecord>();
        var promotionFiles = Directory.GetFiles(Path.Combine(_registryPath, "promotions"), $"{algorithm}_*.json");
        
        foreach (var file in promotionFiles)
        {
            try
            {
                var json = await File.ReadAllTextAsync(file, cancellationToken).ConfigureAwait(false);
                var promotion = JsonSerializer.Deserialize<PromotionRecord>(json, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
                
                if (promotion != null)
                {
                    promotions.Add(promotion);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load promotion record from {File}", file);
            }
        }
        
        return promotions.OrderByDescending(p => p.PromotedAt).ToList();
    }

    /// <summary>
    /// Validate model artifact integrity using checksum
    /// </summary>
    public async Task<bool> ValidateArtifactAsync(string versionId, CancellationToken cancellationToken = default)
    {
        var model = await GetModelAsync(versionId, cancellationToken).ConfigureAwait(false);
        if (model == null || string.IsNullOrEmpty(model.ArtifactPath))
        {
            return false;
        }

        if (!File.Exists(model.ArtifactPath))
        {
            _logger.LogError("Artifact file missing for version {VersionId}: {Path}", versionId, model.ArtifactPath);
            return false;
        }

        var currentHash = await ComputeFileHashAsync(model.ArtifactPath, cancellationToken).ConfigureAwait(false);
        var isValid = currentHash.Equals(model.ArtifactHash, StringComparison.OrdinalIgnoreCase);
        
        if (!isValid)
        {
            _logger.LogError("Artifact hash mismatch for version {VersionId}: expected {Expected}, got {Actual}", 
                versionId, model.ArtifactHash, currentHash);
        }
        
        return isValid;
    }

    /// <summary>
    /// Clean up old model versions (keep recent ones)
    /// </summary>
    public async Task CleanupOldModelsAsync(string algorithm, int keepCount = 10, CancellationToken cancellationToken = default)
    {
        var models = await GetModelsAsync(algorithm, cancellationToken).ConfigureAwait(false);
        var champion = await GetChampionAsync(algorithm, cancellationToken).ConfigureAwait(false);
        
        // Always keep the champion, then keep the most recent ones
        var toKeep = models.Take(keepCount).ToList();
        if (champion != null && !toKeep.Any(m => m.VersionId == champion.VersionId))
        {
            toKeep.Add(champion);
        }

        var toDelete = models.Except(toKeep).ToList();
        
        foreach (var model in toDelete)
        {
            try
            {
                // Delete artifact
                if (!string.IsNullOrEmpty(model.ArtifactPath) && File.Exists(model.ArtifactPath))
                {
                    File.Delete(model.ArtifactPath);
                }
                
                // Delete metadata
                var metadataPath = Path.Combine(_registryPath, "models", $"{algorithm}_{model.VersionId}.json");
                if (File.Exists(metadataPath))
                {
                    File.Delete(metadataPath);
                }
                
                _logger.LogDebug("Cleaned up old model version {VersionId}", model.VersionId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to clean up model version {VersionId}", model.VersionId);
            }
        }
        
        if (toDelete.Count > 0)
        {
            _logger.LogInformation("Cleaned up {Count} old model versions for {Algorithm}", toDelete.Count, algorithm);
        }
    }

    #region Private Methods

    private async Task LoadChampionPointersAsync()
    {
        var championFiles = Directory.GetFiles(_registryPath, "*_champion.txt");
        
        foreach (var file in championFiles)
        {
            try
            {
                var algorithm = Path.GetFileNameWithoutExtension(file).Replace("_champion", "");
                var versionId = await File.ReadAllTextAsync(file).ConfigureAwait(false);
                
                lock (_championLock)
                {
                    _championPointers[algorithm] = versionId.Trim();
                }
                
                _logger.LogDebug("Loaded champion pointer for {Algorithm}: {VersionId}", algorithm, versionId.Trim());
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load champion pointer from {File}", file);
            }
        }
    }

    private async Task<string> ComputeFileHashAsync(string filePath, CancellationToken cancellationToken = default)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);
        var hash = await sha256.ComputeHashAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexString(hash);
    }

    private string GenerateVersionId()
    {
        return $"v{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
    }

    private async Task UpdateModelMetadataAsync(ModelVersion model, CancellationToken cancellationToken = default)
    {
        var modelMetadataPath = Path.Combine(_registryPath, "models", $"{model.Algorithm}_{model.VersionId}.json");
        var tempMetadataPath = modelMetadataPath + ".tmp";
        
        var json = JsonSerializer.Serialize(model, new JsonSerializerOptions 
        { 
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
        
        await File.WriteAllTextAsync(tempMetadataPath, json, cancellationToken).ConfigureAwait(false);
        File.Move(tempMetadataPath, modelMetadataPath);
    }

    private async Task UpdatePromotionRecordAsync(PromotionRecord record, CancellationToken cancellationToken = default)
    {
        var promotionPath = Path.Combine(_registryPath, "promotions", $"{record.Algorithm}_{record.Id}.json");
        var tempPromotionPath = promotionPath + ".tmp";
        
        var json = JsonSerializer.Serialize(record, new JsonSerializerOptions 
        { 
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
        
        await File.WriteAllTextAsync(tempPromotionPath, json, cancellationToken).ConfigureAwait(false);
        File.Move(tempPromotionPath, promotionPath);
    }

    #endregion
}