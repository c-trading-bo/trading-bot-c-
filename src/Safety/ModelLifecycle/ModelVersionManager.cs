using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.Safety.ModelLifecycle;

/// <summary>
/// File-based model version manager with integrity validation and rollback capabilities
/// </summary>
public class ModelVersionManager : IModelVersionManager, IDisposable
{
    private readonly ILogger<ModelVersionManager> _logger;
    private readonly string _modelsDirectory;
    private readonly string _metadataFile;
    private readonly ConcurrentDictionary<string, ModelMetadata> _modelCache;
    private readonly SemaphoreSlim _semaphore;
    private bool _disposed;
    
    public ModelVersionManager(ILogger<ModelVersionManager> logger, string? modelsDirectory = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _modelsDirectory = modelsDirectory ?? Path.Combine(Environment.CurrentDirectory, "models");
        _metadataFile = Path.Combine(_modelsDirectory, "model_registry.json");
        _modelCache = new ConcurrentDictionary<string, ModelMetadata>();
        _semaphore = new SemaphoreSlim(1, 1);
        
        Directory.CreateDirectory(_modelsDirectory);
        _ = Task.Run(LoadModelRegistryAsync);
    }

    public async Task<ModelMetadata?> LoadModelAsync(string modelHash, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (_modelCache.TryGetValue(modelHash, out var cached))
            {
                // Validate integrity before returning
                if (await ValidateModelIntegrityAsync(modelHash, cancellationToken))
                {
                    _logger.LogInformation("Loaded model {ModelHash} from cache with integrity validation", modelHash);
                    return cached;
                }
                
                _logger.LogWarning("Model {ModelHash} failed integrity validation, removing from cache", modelHash);
                _modelCache.TryRemove(modelHash, out _);
            }
            
            await LoadModelRegistryAsync();
            
            if (_modelCache.TryGetValue(modelHash, out var metadata))
            {
                if (await ValidateModelIntegrityAsync(modelHash, cancellationToken))
                {
                    _logger.LogInformation("Loaded model {ModelHash} with version {Version}", modelHash, metadata.Version);
                    return metadata;
                }
            }
            
            _logger.LogWarning("Model {ModelHash} not found or failed validation", modelHash);
            return null;
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async Task<string> RegisterModelAsync(ModelMetadata metadata, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            // Calculate file hash for integrity
            var hash = await CalculateFileHashAsync(metadata.FilePath);
            var newMetadata = metadata with { Hash = hash };
            
            _modelCache[hash] = newMetadata;
            await SaveModelRegistryAsync();
            
            _logger.LogInformation("Registered model {ModelHash} version {Version} at {FilePath}", 
                hash, metadata.Version, metadata.FilePath);
            
            return hash;
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async Task<ModelMetadata?> GetActiveModelAsync(CancellationToken cancellationToken = default)
    {
        await LoadModelRegistryAsync();
        
        var activeModel = _modelCache.Values
            .Where(m => m.Status == "active")
            .OrderByDescending(m => m.CreatedAt)
            .FirstOrDefault();
            
        if (activeModel != null)
        {
            _logger.LogDebug("Active model: {ModelHash} version {Version}", activeModel.Hash, activeModel.Version);
        }
        
        return activeModel;
    }

    public async Task SetActiveModelAsync(string modelHash, bool isPending = false, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (!_modelCache.TryGetValue(modelHash, out var model))
            {
                _logger.LogError("Cannot set active model: {ModelHash} not found", modelHash);
                throw new ArgumentException($"Model {modelHash} not found", nameof(modelHash));
            }
            
            // Validate integrity before activation
            if (!await ValidateModelIntegrityAsync(modelHash, cancellationToken))
            {
                _logger.LogError("Cannot set active model: {ModelHash} failed integrity validation", modelHash);
                throw new InvalidOperationException($"Model {modelHash} failed integrity validation");
            }
            
            // Mark previous active models as rollback candidates
            foreach (var existingModel in _modelCache.Values.Where(m => m.Status == "active").ToList())
            {
                _modelCache[existingModel.Hash] = existingModel with { Status = "rollback" };
            }
            
            // Set new active model
            var status = isPending ? "pending" : "active";
            _modelCache[modelHash] = model with { Status = status };
            
            await SaveModelRegistryAsync();
            
            _logger.LogInformation("Set model {ModelHash} version {Version} as {Status}", 
                modelHash, model.Version, status);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async Task<List<ModelMetadata>> ListModelsAsync(CancellationToken cancellationToken = default)
    {
        await LoadModelRegistryAsync();
        return _modelCache.Values.OrderByDescending(m => m.CreatedAt).ToList();
    }

    public async Task<bool> ValidateModelIntegrityAsync(string modelHash, CancellationToken cancellationToken = default)
    {
        if (!_modelCache.TryGetValue(modelHash, out var metadata))
        {
            return false;
        }
        
        if (!File.Exists(metadata.FilePath))
        {
            _logger.LogWarning("Model file not found: {FilePath}", metadata.FilePath);
            return false;
        }
        
        try
        {
            var currentHash = await CalculateFileHashAsync(metadata.FilePath);
            var isValid = currentHash == modelHash;
            
            if (!isValid)
            {
                _logger.LogError("Model integrity validation failed for {ModelHash}. Expected: {Expected}, Actual: {Actual}", 
                    modelHash, modelHash, currentHash);
            }
            
            return isValid;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating model integrity for {ModelHash}", modelHash);
            return false;
        }
    }
    
    private async Task<string> CalculateFileHashAsync(string filePath)
    {
        using var sha256 = SHA256.Create();
        await using var stream = File.OpenRead(filePath);
        var hashBytes = await sha256.ComputeHashAsync(stream);
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }
    
    private async Task LoadModelRegistryAsync()
    {
        if (!File.Exists(_metadataFile))
        {
            return;
        }
        
        try
        {
            var json = await File.ReadAllTextAsync(_metadataFile);
            var models = JsonSerializer.Deserialize<List<ModelMetadata>>(json);
            
            if (models != null)
            {
                _modelCache.Clear();
                foreach (var model in models)
                {
                    _modelCache[model.Hash] = model;
                }
                
                _logger.LogDebug("Loaded {Count} models from registry", models.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading model registry from {MetadataFile}", _metadataFile);
        }
    }
    
    private async Task SaveModelRegistryAsync()
    {
        try
        {
            var models = _modelCache.Values.ToList();
            var json = JsonSerializer.Serialize(models, new JsonSerializerOptions { WriteIndented = true });
            
            // Atomic write
            var tempFile = _metadataFile + ".tmp";
            await File.WriteAllTextAsync(tempFile, json);
            File.Move(tempFile, _metadataFile, true);
            
            _logger.LogDebug("Saved {Count} models to registry", models.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving model registry to {MetadataFile}", _metadataFile);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _semaphore?.Dispose();
            _disposed = true;
        }
    }
}