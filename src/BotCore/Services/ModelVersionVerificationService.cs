using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using BotCore.Configuration;

namespace BotCore.Services
{
    /// <summary>
    /// Model version verification service ensures each training run produces different weights
    /// Implements comprehensive model versioning, validation, and comparison
    /// </summary>
    public interface IModelVersionVerificationService
    {
        Task<ModelVersionResult> VerifyModelVersionAsync(string modelPath, ModelMetadata metadata);
        Task<bool> CompareModelVersionsAsync(string modelPath1, string modelPath2);
        Task LogTrainingMetadataAsync(TrainingMetadata metadata);
        Task<bool> ValidateModelIntegrityAsync(string modelPath);
        Task BackupModelAsync(string modelPath, string version);
        Task<List<ModelVersionInfo>> GetModelVersionHistoryAsync(string modelName);
    }

    /// <summary>
    /// Comprehensive model version verification service implementation
    /// </summary>
    public class ModelVersionVerificationService : IModelVersionVerificationService
    {
        private readonly ILogger<ModelVersionVerificationService> _logger;
        private readonly ModelVersioningConfiguration _config;
        private readonly string _versionRegistryPath;

        public ModelVersionVerificationService(
            ILogger<ModelVersionVerificationService> logger,
            IOptions<ModelVersioningConfiguration> config)
        {
            _logger = logger;
            _config = config.Value;
            _versionRegistryPath = Path.Combine(_config.ModelRegistry.BaseDirectory, "version_registry.json");
            
            // Ensure directories exist
            Directory.CreateDirectory(_config.ModelRegistry.BaseDirectory);
            Directory.CreateDirectory(_config.ModelRegistry.BackupDirectory);
        }

        /// <summary>
        /// Verify model version and ensure it's different from previous versions
        /// </summary>
        public async Task<ModelVersionResult> VerifyModelVersionAsync(string modelPath, ModelMetadata metadata)
        {
            try
            {
                _logger.LogInformation("[MODEL-VERSION] Verifying model version for {ModelPath}", modelPath);

                var result = new ModelVersionResult
                {
                    ModelPath = modelPath,
                    Metadata = metadata,
                    VerificationTime = DateTime.UtcNow
                };

                if (!File.Exists(modelPath))
                {
                    result.IsValid = false;
                    result.ValidationErrors.Add("Model file does not exist");
                    return result;
                }

                // Calculate model hash
                result.ModelHash = await CalculateModelHashAsync(modelPath).ConfigureAwait(false);
                metadata.ModelHash = result.ModelHash;

                // Load version registry
                var registry = await LoadVersionRegistryAsync().ConfigureAwait(false);

                // Check if this is a duplicate model
                var existingVersion = registry.Versions.FirstOrDefault(v => v.ModelHash == result.ModelHash);
                if (existingVersion != null)
                {
                    result.IsValid = false;
                    result.IsDuplicate = true;
                    result.ValidationErrors.Add($"Duplicate model detected. Identical to version {existingVersion.Version} created at {existingVersion.CreatedAt}");
                    
                    _logger.LogWarning("[MODEL-VERSION] Duplicate model detected: {ModelPath} matches {ExistingVersion}",
                        modelPath, existingVersion.Version);
                    return result;
                }

                // Generate new version
                result.Version = GenerateVersion();
                metadata.Version = result.Version;

                // Validate model integrity
                result.IntegrityValid = await ValidateModelIntegrityAsync(modelPath).ConfigureAwait(false);
                if (!result.IntegrityValid)
                {
                    result.ValidationErrors.Add("Model integrity validation failed");
                }

                // Compare with previous versions if required
                if (_config.RequireVersionDifference && registry.Versions.Any())
                {
                    var isSignificantlyDifferent = await ValidateSignificantDifferenceAsync(modelPath, registry).ConfigureAwait(false);
                    if (!isSignificantlyDifferent)
                    {
                        result.IsValid = false;
                        result.ValidationErrors.Add($"Model does not meet minimum difference threshold of {_config.MinWeightChangePct}%");
                    }
                }

                // If all validations pass
                if (!result.ValidationErrors.Any() && result.IntegrityValid)
                {
                    result.IsValid = true;
                    
                    // Create version info
                    var versionInfo = new ModelVersionInfo
                    {
                        Version = result.Version,
                        ModelPath = modelPath,
                        ModelHash = result.ModelHash,
                        CreatedAt = DateTime.UtcNow,
                        Metadata = metadata,
                        TrainingMetrics = metadata.TrainingMetrics
                    };

                    // Add to registry
                    registry.Versions.Add(versionInfo);
                    registry.LastUpdated = DateTime.UtcNow;

                    // Backup model if configured
                    if (_config.ModelRegistry.AutoBackupModels)
                    {
                        await BackupModelAsync(modelPath, result.Version).ConfigureAwait(false);
                    }

                    // Clean old versions if needed
                    await CleanOldVersionsAsync(registry).ConfigureAwait(false);

                    // Save updated registry
                    await SaveVersionRegistryAsync(registry).ConfigureAwait(false);

                    _logger.LogInformation("[MODEL-VERSION] ✅ Model version verified successfully: {Version} (Hash: {Hash})",
                        result.Version, result.ModelHash[..8]);
                }
                else
                {
                    _logger.LogWarning("[MODEL-VERSION] ❌ Model version verification failed: {Errors}",
                        string.Join(", ", result.ValidationErrors));
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL-VERSION] Error verifying model version for {ModelPath}", modelPath);
                return new ModelVersionResult
                {
                    ModelPath = modelPath,
                    IsValid = false,
                    ValidationErrors = new List<string> { $"Verification error: {ex.Message}" }
                };
            }
        }

        /// <summary>
        /// Compare two model versions to determine if they are significantly different
        /// </summary>
        public async Task<bool> CompareModelVersionsAsync(string modelPath1, string modelPath2)
        {
            try
            {
                if (!File.Exists(modelPath1) || !File.Exists(modelPath2))
                {
                    _logger.LogWarning("[MODEL-COMPARE] One or both model files do not exist: {Path1}, {Path2}", modelPath1, modelPath2);
                    return false;
                }

                var hash1 = await CalculateModelHashAsync(modelPath1).ConfigureAwait(false);
                var hash2 = await CalculateModelHashAsync(modelPath2).ConfigureAwait(false);

                var areIdentical = hash1 == hash2;
                
                if (_config.VersionComparisonLogging)
                {
                    _logger.LogInformation("[MODEL-COMPARE] Model comparison: {Path1} vs {Path2} - Identical: {Identical}",
                        Path.GetFileName(modelPath1), Path.GetFileName(modelPath2), areIdentical);
                }

                return !areIdentical;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL-COMPARE] Error comparing model versions");
                return false;
            }
        }

        /// <summary>
        /// Log comprehensive training metadata
        /// </summary>
        public async Task LogTrainingMetadataAsync(TrainingMetadata metadata)
        {
            try
            {
                if (!_config.TrainingMetadataLogging)
                    return;

                var metadataJson = JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true });
                var logPath = Path.Combine(_config.ModelRegistry.BaseDirectory, $"training_metadata_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
                
                await File.WriteAllTextAsync(logPath, metadataJson).ConfigureAwait(false);
                
                _logger.LogInformation("[TRAINING-METADATA] Training metadata logged: Strategy={Strategy}, Seed={Seed}, Duration={Duration}",
                    metadata.StrategyName, metadata.RandomSeed, metadata.TrainingDuration);

                _logger.LogInformation("[TRAINING-METADATA] Model metrics: Loss={Loss:F6}, Accuracy={Accuracy:F4}, ValidationScore={ValidationScore:F4}",
                    metadata.TrainingMetrics.FinalLoss, metadata.TrainingMetrics.Accuracy, metadata.TrainingMetrics.ValidationScore);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TRAINING-METADATA] Error logging training metadata");
            }
        }

        /// <summary>
        /// Validate model file integrity
        /// </summary>
        public Task<bool> ValidateModelIntegrityAsync(string modelPath)
        {
            try
            {
                if (!_config.ModelHashValidation)
                    return Task.FromResult(true);

                if (!File.Exists(modelPath))
                    return Task.FromResult(false);

                // Basic file validation
                var fileInfo = new FileInfo(modelPath);
                if (fileInfo.Length == 0)
                {
                    _logger.LogWarning("[MODEL-INTEGRITY] Model file is empty: {ModelPath}", modelPath);
                    return Task.FromResult(false);
                }

                // For ONNX models, verify file format
                if (modelPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                {
                    var header = new byte[8];
                    using var fs = File.OpenRead(modelPath);
                    fs.Read(header, 0, 8);
                    
                    // ONNX files should start with protobuf magic bytes or model structure
                    // This is a simplified check - in production you might use ONNX runtime validation
                    if (header[0] == 0x08 || header[0] == 0x0A) // Common protobuf prefixes
                    {
                        return Task.FromResult(true);
                    }
                }

                // For other model types, just verify it's not corrupted
                return Task.FromResult(fileInfo.Length > 100); // Minimum reasonable model size
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL-INTEGRITY] Error validating model integrity for {ModelPath}", modelPath);
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Backup model with version
        /// </summary>
        public async Task BackupModelAsync(string modelPath, string version)
        {
            try
            {
                var fileName = Path.GetFileNameWithoutExtension(modelPath);
                var extension = Path.GetExtension(modelPath);
                var backupFileName = $"{fileName}_v{version}_{DateTime.UtcNow:yyyyMMdd_HHmmss}{extension}";
                var backupPath = Path.Combine(_config.ModelRegistry.BackupDirectory, backupFileName);

                using var sourceStream = File.OpenRead(modelPath);
                using var destStream = File.Create(backupPath);
                await sourceStream.CopyToAsync(destStream).ConfigureAwait(false);
                
                _logger.LogInformation("[MODEL-BACKUP] Model backed up: {BackupPath}", backupPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL-BACKUP] Error backing up model {ModelPath}", modelPath);
            }
        }

        /// <summary>
        /// Get model version history
        /// </summary>
        public Task<List<ModelVersionInfo>> GetModelVersionHistoryAsync(string modelName)
        {
            try
            {
                var registry = LoadVersionRegistryAsync().Result;
                var result = registry.Versions
                    .Where(v => v.ModelPath.Contains(modelName, StringComparison.OrdinalIgnoreCase))
                    .OrderByDescending(v => v.CreatedAt)
                    .ToList();
                return Task.FromResult(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL-HISTORY] Error getting model version history for {ModelName}", modelName);
                return Task.FromResult(new List<ModelVersionInfo>());
            }
        }

        #region Private Methods

        /// <summary>
        /// Calculate SHA256 hash of model file
        /// </summary>
        private async Task<string> CalculateModelHashAsync(string modelPath)
        {
            using var sha256 = SHA256.Create();
            using var stream = File.OpenRead(modelPath);
            var hash = await Task.Run(() => sha256.ComputeHash(stream)).ConfigureAwait(false);
            return Convert.ToHexString(hash);
        }

        /// <summary>
        /// Generate new version string
        /// </summary>
        private string GenerateVersion()
        {
            return $"v{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
        }

        /// <summary>
        /// Validate that model is significantly different from previous versions
        /// </summary>
        private async Task<bool> ValidateSignificantDifferenceAsync(string modelPath, ModelVersionRegistry registry)
        {
            try
            {
                // For now, we'll use hash comparison as a proxy for weight differences
                // In a more sophisticated implementation, you could load model weights and compare numerically
                var currentHash = await CalculateModelHashAsync(modelPath).ConfigureAwait(false);
                
                // Check if any recent versions have the same hash
                var recentVersions = registry.Versions
                    .Where(v => v.CreatedAt > DateTime.UtcNow.AddDays(-7)) // Last 7 days
                    .ToList();

                foreach (var version in recentVersions)
                {
                    if (version.ModelHash == currentHash)
                    {
                        _logger.LogWarning("[MODEL-DIFF] Model is identical to recent version {Version}", version.Version);
                        return false;
                    }
                }

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL-DIFF] Error validating model difference");
                return false;
            }
        }

        /// <summary>
        /// Load version registry from disk
        /// </summary>
        private async Task<ModelVersionRegistry> LoadVersionRegistryAsync()
        {
            try
            {
                if (!File.Exists(_versionRegistryPath))
                {
                    return new ModelVersionRegistry();
                }

                var json = await File.ReadAllTextAsync(_versionRegistryPath).ConfigureAwait(false);
                return JsonSerializer.Deserialize<ModelVersionRegistry>(json) ?? new ModelVersionRegistry();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[REGISTRY] Error loading version registry, creating new one");
                return new ModelVersionRegistry();
            }
        }

        /// <summary>
        /// Save version registry to disk
        /// </summary>
        private async Task SaveVersionRegistryAsync(ModelVersionRegistry registry)
        {
            try
            {
                var json = JsonSerializer.Serialize(registry, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(_versionRegistryPath, json).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[REGISTRY] Error saving version registry");
            }
        }

        /// <summary>
        /// Clean old versions to maintain maximum history
        /// </summary>
        private Task CleanOldVersionsAsync(ModelVersionRegistry registry)
        {
            try
            {
                if (registry.Versions.Count > _config.ModelRegistry.MaxVersionHistory)
                {
                    var versionsToRemove = registry.Versions
                        .OrderBy(v => v.CreatedAt)
                        .Take(registry.Versions.Count - _config.ModelRegistry.MaxVersionHistory)
                        .ToList();

                    foreach (var version in versionsToRemove)
                    {
                        registry.Versions.Remove(version);
                        _logger.LogInformation("[REGISTRY] Removed old version from registry: {Version}", version.Version);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[REGISTRY] Error cleaning old versions");
            }
            
            return Task.CompletedTask;
        }

        #endregion
    }

    #region Supporting Models

    /// <summary>
    /// Model metadata for version tracking
    /// </summary>
    public class ModelMetadata
    {
        public string ModelName { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public string ModelHash { get; set; } = string.Empty;
        public string StrategyName { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public TrainingMetrics TrainingMetrics { get; set; } = new();
        public Dictionary<string, object> AdditionalData { get; } = new();
    }

    /// <summary>
    /// Training metadata for comprehensive logging
    /// </summary>
    public class TrainingMetadata
    {
        public string StrategyName { get; set; } = string.Empty;
        public int RandomSeed { get; set; }
        public DateTime TrainingStart { get; set; }
        public DateTime TrainingEnd { get; set; }
        public TimeSpan TrainingDuration => TrainingEnd - TrainingStart;
        public TrainingMetrics TrainingMetrics { get; set; } = new();
        public Dictionary<string, object> Hyperparameters { get; } = new();
        public string DatasetHash { get; set; } = string.Empty;
        public string Environment { get; set; } = string.Empty;
    }

    /// <summary>
    /// Training metrics for model evaluation
    /// </summary>
    public class TrainingMetrics
    {
        public double FinalLoss { get; set; }
        public double Accuracy { get; set; }
        public double ValidationScore { get; set; }
        public double SharpeRatio { get; set; }
        public double MaxDrawdown { get; set; }
        public int TotalEpochs { get; set; }
        public TimeSpan TrainingTime { get; set; }
    }

    /// <summary>
    /// Model version verification result
    /// </summary>
    public class ModelVersionResult
    {
        public string ModelPath { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public string ModelHash { get; set; } = string.Empty;
        public bool IsValid { get; set; }
        public bool IsDuplicate { get; set; }
        public bool IntegrityValid { get; set; }
        public List<string> ValidationErrors { get; } = new();
        public ModelMetadata? Metadata { get; set; }
        public DateTime VerificationTime { get; set; }
    }

    /// <summary>
    /// Model version information
    /// </summary>
    public class ModelVersionInfo
    {
        public string Version { get; set; } = string.Empty;
        public string ModelPath { get; set; } = string.Empty;
        public string ModelHash { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        public ModelMetadata? Metadata { get; set; }
        public TrainingMetrics? TrainingMetrics { get; set; }
    }

    /// <summary>
    /// Model version registry
    /// </summary>
    public class ModelVersionRegistry
    {
        public List<ModelVersionInfo> Versions { get; } = new();
        public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
        public string RegistryVersion { get; set; } = "1.0";
    }

    #endregion
}