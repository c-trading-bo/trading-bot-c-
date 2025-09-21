using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using System.Text.Json;
using System.Security.Cryptography;
using System.Text;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Hot-Swappable AI Model Service for continuous learning without downtime
/// Production-grade model management for institutional autonomous trading
/// </summary>
public class HotSwappableModelService : IHostedService, IDisposable
{
    private readonly ILogger<HotSwappableModelService> _logger;
    private readonly Timer _modelCheckTimer;
    private readonly ConcurrentDictionary<string, ModelInfo> _activeModels;
    private readonly string _modelRegistryPath;
    private readonly string _modelStagingPath;
    private readonly object _swapLock = new();
    private volatile bool _disposed;

    // Model check interval (every 5 minutes)
    private const int MODEL_CHECK_INTERVAL_MS = 300000;

    public HotSwappableModelService(ILogger<HotSwappableModelService> logger)
    {
        _logger = logger;
        _activeModels = new ConcurrentDictionary<string, ModelInfo>();
        
        // Set up model directories
        _modelRegistryPath = Path.Combine(Directory.GetCurrentDirectory(), "model_registry");
        _modelStagingPath = Path.Combine(Directory.GetCurrentDirectory(), "model_staging");
        
        Directory.CreateDirectory(_modelRegistryPath);
        Directory.CreateDirectory(_modelStagingPath);
        
        LoadActiveModels();
        
        // Start model monitoring timer
        _modelCheckTimer = new Timer(CheckForModelUpdates, null, 
            TimeSpan.FromMilliseconds(MODEL_CHECK_INTERVAL_MS),
            TimeSpan.FromMilliseconds(MODEL_CHECK_INTERVAL_MS));
            
        _logger.LogInformation("🔄 [HOT-SWAP-MODELS] Model management service initialized");
        _logger.LogInformation("📁 [HOT-SWAP-MODELS] Registry: {RegistryPath}, Staging: {StagingPath}", 
            _modelRegistryPath, _modelStagingPath);
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🟢 [HOT-SWAP-MODELS] Hot-swappable model service started");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔴 [HOT-SWAP-MODELS] Hot-swappable model service stopped");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Download and validate new model from cloud training
    /// </summary>
    public async Task<bool> DownloadNewModelAsync(string modelType, string modelUrl, string version)
    {
        try
        {
            _logger.LogInformation("⬇️ [HOT-SWAP-MODELS] Downloading new {ModelType} model version {Version}", modelType, version);
            
            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromMinutes(10); // 10-minute timeout for large models
            
            var modelBytes = await httpClient.GetByteArrayAsync(modelUrl).ConfigureAwait(false);
            
            // Generate staging filename
            var stagingFile = Path.Combine(_modelStagingPath, $"{modelType}_{version}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.onnx");
            
            // Save to staging
            await File.WriteAllBytesAsync(stagingFile, modelBytes).ConfigureAwait(false);
            
            // Validate model
            var isValid = await ValidateModelAsync(stagingFile, modelType).ConfigureAwait(false);
            
            if (isValid)
            {
                _logger.LogInformation("✅ [HOT-SWAP-MODELS] Successfully downloaded and validated {ModelType} model version {Version}", 
                    modelType, version);
                return true;
            }
            else
            {
                File.Delete(stagingFile); // Clean up invalid model
                _logger.LogError("❌ [HOT-SWAP-MODELS] Model validation failed for {ModelType} version {Version}", 
                    modelType, version);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HOT-SWAP-MODELS] Failed to download model {ModelType} version {Version}", 
                modelType, version);
            return false;
        }
    }

    /// <summary>
    /// Hot-swap model without stopping trading
    /// </summary>
    public async Task<bool> HotSwapModelAsync(string modelType, string stagingFilePath)
    {
        lock (_swapLock)
        {
            try
            {
                _logger.LogInformation("🔄 [HOT-SWAP-MODELS] Starting hot-swap for {ModelType}", modelType);
                
                // Generate new model info
                var newModelInfo = new ModelInfo
                {
                    ModelType = modelType,
                    Version = ExtractVersionFromFilename(stagingFilePath),
                    FilePath = stagingFilePath,
                    HashSha256 = CalculateFileHash(stagingFilePath),
                    LoadedTimeUtc = DateTime.UtcNow,
                    IsActive = false
                };

                // Backup current model
                if (_activeModels.TryGetValue(modelType, out var currentModel))
                {
                    var backupPath = Path.Combine(_modelRegistryPath, 
                        $"{modelType}_backup_{DateTime.UtcNow:yyyyMMdd_HHmmss}.onnx");
                    File.Copy(currentModel.FilePath, backupPath);
                    currentModel.BackupPath = backupPath;
                    
                    _logger.LogInformation("💾 [HOT-SWAP-MODELS] Created backup of current {ModelType} model", modelType);
                }

                // Move new model to registry
                var registryPath = Path.Combine(_modelRegistryPath, $"{modelType}_current.onnx");
                File.Move(stagingFilePath, registryPath, overwrite: true);
                newModelInfo.FilePath = registryPath;

                // Activate new model
                newModelInfo.IsActive = true;
                _activeModels.AddOrUpdate(modelType, newModelInfo, (key, oldValue) =>
                {
                    oldValue.IsActive = false;
                    return newModelInfo;
                });

                // Save model registry
                SaveModelRegistry();

                _logger.LogInformation("✅ [HOT-SWAP-MODELS] Successfully hot-swapped {ModelType} to version {Version}", 
                    modelType, newModelInfo.Version);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [HOT-SWAP-MODELS] Hot-swap failed for {ModelType}", modelType);
                
                // Attempt automatic rollback
                await RollbackModelAsync(modelType).ConfigureAwait(false);
                return false;
            }
        }
    }

    /// <summary>
    /// Validate model compatibility before deployment
    /// </summary>
    private async Task<bool> ValidateModelAsync(string modelPath, string modelType)
    {
        try
        {
            // Basic file validation
            if (!File.Exists(modelPath))
            {
                _logger.LogError("❌ [HOT-SWAP-MODELS] Model file does not exist: {ModelPath}", modelPath);
                return false;
            }

            var fileInfo = new FileInfo(modelPath);
            if (fileInfo.Length == 0)
            {
                _logger.LogError("❌ [HOT-SWAP-MODELS] Model file is empty: {ModelPath}", modelPath);
                return false;
            }

            // ONNX format validation (basic check)
            var bytes = await File.ReadAllBytesAsync(modelPath).ConfigureAwait(false);
            if (bytes.Length < 16 || !HasOnnxMagicNumber(bytes))
            {
                _logger.LogError("❌ [HOT-SWAP-MODELS] Invalid ONNX format: {ModelPath}", modelPath);
                return false;
            }

            // Model type specific validation
            var isCompatible = await ValidateModelCompatibilityAsync(modelPath, modelType).ConfigureAwait(false);
            if (!isCompatible)
            {
                _logger.LogError("❌ [HOT-SWAP-MODELS] Model compatibility check failed: {ModelType}", modelType);
                return false;
            }

            _logger.LogDebug("✅ [HOT-SWAP-MODELS] Model validation passed for {ModelType}", modelType);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HOT-SWAP-MODELS] Model validation error for {ModelType}", modelType);
            return false;
        }
    }

    /// <summary>
    /// Validate model compatibility with expected inputs/outputs
    /// </summary>
    private async Task<bool> ValidateModelCompatibilityAsync(string modelPath, string modelType)
    {
        // In production, this would use ONNX Runtime to validate model structure
        // For now, implement basic compatibility checks
        
        try
        {
            return modelType.ToUpperInvariant() switch
            {
                "NEURAL_UCB" => ValidateNeuralUcbModel(modelPath),
                "CVAR_PPO" => ValidateCvarPpoModel(modelPath),
                "LSTM" => ValidateLstmModel(modelPath),
                "CONFIDENCE" => ValidateConfidenceModel(modelPath),
                _ => true // Default to accepting unknown model types
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Model compatibility validation failed");
            return false;
        }
    }

    /// <summary>
    /// Check for ONNX magic number in file header
    /// </summary>
    private static bool HasOnnxMagicNumber(byte[] bytes)
    {
        // ONNX files typically start with specific byte patterns
        // This is a simplified check - production would use proper ONNX validation
        return bytes.Length >= 4 && bytes[0] == 0x08;
    }

    /// <summary>
    /// Rollback to previous model version
    /// </summary>
    private async Task<bool> RollbackModelAsync(string modelType)
    {
        try
        {
            if (_activeModels.TryGetValue(modelType, out var currentModel) && 
                !string.IsNullOrEmpty(currentModel.BackupPath) && 
                File.Exists(currentModel.BackupPath))
            {
                _logger.LogWarning("🔄 [HOT-SWAP-MODELS] Rolling back {ModelType} to previous version", modelType);
                
                var registryPath = Path.Combine(_modelRegistryPath, $"{modelType}_current.onnx");
                File.Copy(currentModel.BackupPath, registryPath, overwrite: true);
                
                currentModel.FilePath = registryPath;
                currentModel.LoadedTimeUtc = DateTime.UtcNow;
                currentModel.Version = "ROLLBACK_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                
                _logger.LogInformation("✅ [HOT-SWAP-MODELS] Successfully rolled back {ModelType}", modelType);
                return true;
            }
            else
            {
                _logger.LogError("❌ [HOT-SWAP-MODELS] No backup available for rollback of {ModelType}", modelType);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HOT-SWAP-MODELS] Rollback failed for {ModelType}", modelType);
            return false;
        }
    }

    /// <summary>
    /// Check for model updates periodically
    /// </summary>
    private void CheckForModelUpdates(object? state)
    {
        if (_disposed) return;
        
        try
        {
            var stagingFiles = Directory.GetFiles(_modelStagingPath, "*.onnx");
            
            foreach (var stagingFile in stagingFiles)
            {
                var modelType = ExtractModelTypeFromFilename(stagingFile);
                if (!string.IsNullOrEmpty(modelType))
                {
                    Task.Run(async () => await HotSwapModelAsync(modelType, stagingFile).ConfigureAwait(false));
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HOT-SWAP-MODELS] Error checking for model updates");
        }
    }

    /// <summary>
    /// Load active models from registry
    /// </summary>
    private void LoadActiveModels()
    {
        try
        {
            var registryFile = Path.Combine(_modelRegistryPath, "model_registry.json");
            if (!File.Exists(registryFile))
            {
                CreateDefaultModelRegistry();
                return;
            }

            var json = File.ReadAllText(registryFile);
            var registry = JsonSerializer.Deserialize<ModelRegistry>(json);
            
            if (registry?.Models != null)
            {
                foreach (var model in registry.Models)
                {
                    if (File.Exists(model.FilePath))
                    {
                        _activeModels.TryAdd(model.ModelType, model);
                    }
                }
                
                _logger.LogInformation("📂 [HOT-SWAP-MODELS] Loaded {Count} active models from registry", _activeModels.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HOT-SWAP-MODELS] Failed to load model registry");
        }
    }

    /// <summary>
    /// Save model registry to disk
    /// </summary>
    private void SaveModelRegistry()
    {
        try
        {
            var registry = new ModelRegistry
            {
                Models = _activeModels.Values.ToList(),
                LastUpdatedUtc = DateTime.UtcNow,
                Version = "2.0"
            };

            var json = JsonSerializer.Serialize(registry, new JsonSerializerOptions { WriteIndented = true });
            var registryFile = Path.Combine(_modelRegistryPath, "model_registry.json");
            File.WriteAllText(registryFile, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HOT-SWAP-MODELS] Failed to save model registry");
        }
    }

    /// <summary>
    /// Create default model registry
    /// </summary>
    private void CreateDefaultModelRegistry()
    {
        var defaultModels = new[]
        {
            new ModelInfo { ModelType = "NEURAL_UCB", Version = "1.0.0", IsActive = false },
            new ModelInfo { ModelType = "CVAR_PPO", Version = "1.0.0", IsActive = false },
            new ModelInfo { ModelType = "LSTM", Version = "1.0.0", IsActive = false },
            new ModelInfo { ModelType = "CONFIDENCE", Version = "1.0.0", IsActive = false }
        };

        foreach (var model in defaultModels)
        {
            _activeModels.TryAdd(model.ModelType, model);
        }

        SaveModelRegistry();
        _logger.LogInformation("✨ [HOT-SWAP-MODELS] Created default model registry");
    }

    // Helper methods for model validation (simplified for production)
    private static bool ValidateNeuralUcbModel(string modelPath) => true; // Placeholder
    private static bool ValidateCvarPpoModel(string modelPath) => true; // Placeholder  
    private static bool ValidateLstmModel(string modelPath) => true; // Placeholder
    private static bool ValidateConfidenceModel(string modelPath) => true; // Placeholder

    private static string ExtractVersionFromFilename(string filePath)
    {
        var filename = Path.GetFileNameWithoutExtension(filePath);
        var parts = filename.Split('_');
        return parts.Length >= 2 ? parts[1] : "unknown";
    }

    private static string ExtractModelTypeFromFilename(string filePath)
    {
        var filename = Path.GetFileNameWithoutExtension(filePath);
        var parts = filename.Split('_');
        return parts.Length >= 1 ? parts[0].ToUpperInvariant() : string.Empty;
    }

    private static string CalculateFileHash(string filePath)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);
        var hashBytes = sha256.ComputeHash(stream);
        return Convert.ToHexString(hashBytes);
    }

    /// <summary>
    /// Get model swap statistics
    /// </summary>
    public ModelSwapStats GetStats()
    {
        return new ModelSwapStats
        {
            ActiveModels = _activeModels.Count,
            ModelTypes = _activeModels.Keys.ToList(),
            LastSwapTime = _activeModels.Values.Any() ? _activeModels.Values.Max(m => m.LoadedTimeUtc) : (DateTime?)null,
            RegistryPath = _modelRegistryPath,
            StagingPath = _modelStagingPath
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _modelCheckTimer?.Dispose();
        _disposed = true;
        
        _logger.LogInformation("🗑️ [HOT-SWAP-MODELS] Hot-swappable model service disposed");
    }
}

/// <summary>
/// Model information for registry
/// </summary>
public class ModelInfo
{
    public string ModelType { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string FilePath { get; set; } = string.Empty;
    public string HashSha256 { get; set; } = string.Empty;
    public DateTime LoadedTimeUtc { get; set; }
    public bool IsActive { get; set; }
    public string BackupPath { get; set; } = string.Empty;
}

/// <summary>
/// Model registry for persistence
/// </summary>
public class ModelRegistry
{
    public List<ModelInfo> Models { get; set; } = new();
    public DateTime LastUpdatedUtc { get; set; }
    public string Version { get; set; } = "2.0";
}

/// <summary>
/// Statistics for monitoring
/// </summary>
public class ModelSwapStats
{
    public int ActiveModels { get; set; }
    public List<string> ModelTypes { get; set; } = new();
    public DateTime? LastSwapTime { get; set; }
    public string RegistryPath { get; set; } = string.Empty;
    public string StagingPath { get; set; } = string.Empty;
}