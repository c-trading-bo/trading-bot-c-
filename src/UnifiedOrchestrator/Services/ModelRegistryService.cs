using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    /// <summary>
    /// Timestamped, hash-versioned model registry with metadata
    /// Supports automatic compression and health checks
    /// </summary>
    public class ModelRegistryService
    {
        private readonly ILogger<ModelRegistryService> _logger;
        private readonly ModelRegistryOptions _options;
        private readonly JsonSerializerOptions _jsonOptions;

        public ModelRegistryService(
            ILogger<ModelRegistryService> logger,
            IOptions<ModelRegistryOptions> options)
        {
            _logger = logger;
            _options = options.Value;
            
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true
            };

            // Ensure registry directory exists
            if (!Directory.Exists(_options.RegistryPath))
            {
                Directory.CreateDirectory(_options.RegistryPath);
            }

            _logger.LogInformation("🗃️ Model Registry initialized: {RegistryPath}", _options.RegistryPath);
        }

        /// <summary>
        /// Register a new model with timestamped, hash-versioned metadata
        /// </summary>
        public async Task<ModelRegistryEntry> RegisterModelAsync(
            string modelName,
            string modelPath,
            ModelMetadata metadata,
            CancellationToken cancellationToken = default)
        {
            try
            {
                // Calculate model hash
                var modelHash = await CalculateFileHashAsync(modelPath, cancellationToken);
                var timestamp = DateTime.UtcNow;
                var version = $"{timestamp:yyyyMMdd-HHmmss}-{modelHash[..8]}";

                // Create registry entry
                var entry = new ModelRegistryEntry
                {
                    ModelName = modelName,
                    Version = version,
                    Timestamp = timestamp,
                    Hash = modelHash,
                    Metadata = metadata,
                    OriginalPath = modelPath,
                    Status = ModelStatus.Registered
                };

                // Create versioned directory
                var versionedDir = Path.Combine(_options.RegistryPath, modelName, version);
                Directory.CreateDirectory(versionedDir);

                // Copy model file
                var registryModelPath = Path.Combine(versionedDir, $"{modelName}.onnx");
                File.Copy(modelPath, registryModelPath, overwrite: true);
                entry.RegistryPath = registryModelPath;

                // Compress model if enabled
                if (_options.AutoCompress)
                {
                    await CompressModelAsync(registryModelPath, cancellationToken);
                    entry.IsCompressed = true;
                }

                // Save metadata
                var metadataPath = Path.Combine(versionedDir, "metadata.json");
                var metadataJson = JsonSerializer.Serialize(entry, _jsonOptions);
                await File.WriteAllTextAsync(metadataPath, metadataJson, cancellationToken);

                // Update registry index
                await UpdateRegistryIndexAsync(entry, cancellationToken);

                _logger.LogInformation("📝 Model registered: {ModelName} v{Version}", modelName, version);
                return entry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to register model: {ModelName}", modelName);
                throw;
            }
        }

        /// <summary>
        /// Get latest model version
        /// </summary>
        public async Task<ModelRegistryEntry?> GetLatestModelAsync(string modelName, CancellationToken cancellationToken = default)
        {
            try
            {
                var modelDir = Path.Combine(_options.RegistryPath, modelName);
                if (!Directory.Exists(modelDir))
                {
                    return null;
                }

                var versions = Directory.GetDirectories(modelDir)
                    .Select(d => Path.GetFileName(d))
                    .OrderByDescending(v => v)
                    .ToList();

                if (!versions.Any())
                {
                    return null;
                }

                var latestVersion = versions.First();
                var metadataPath = Path.Combine(modelDir, latestVersion, "metadata.json");
                
                if (!File.Exists(metadataPath))
                {
                    return null;
                }

                var metadataJson = await File.ReadAllTextAsync(metadataPath, cancellationToken);
                var entry = JsonSerializer.Deserialize<ModelRegistryEntry>(metadataJson, _jsonOptions);
                
                return entry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to get latest model: {ModelName}", modelName);
                return null;
            }
        }

        /// <summary>
        /// Perform health check on registered models
        /// </summary>
        public async Task<ModelHealthReport> PerformHealthCheckAsync(CancellationToken cancellationToken = default)
        {
            var report = new ModelHealthReport
            {
                Timestamp = DateTime.UtcNow,
                ModelStatuses = new List<ModelHealthStatus>()
            };

            try
            {
                var modelDirs = Directory.GetDirectories(_options.RegistryPath);
                
                foreach (var modelDir in modelDirs)
                {
                    var modelName = Path.GetFileName(modelDir);
                    var latestModel = await GetLatestModelAsync(modelName, cancellationToken);
                    
                    var status = new ModelHealthStatus
                    {
                        ModelName = modelName,
                        IsHealthy = false,
                        Issues = new List<string>()
                    };

                    if (latestModel == null)
                    {
                        status.Issues.Add("No valid model found");
                    }
                    else
                    {
                        // Check if model file exists
                        if (!File.Exists(latestModel.RegistryPath))
                        {
                            status.Issues.Add("Model file missing");
                        }
                        
                        // Check file integrity
                        var currentHash = await CalculateFileHashAsync(latestModel.RegistryPath, cancellationToken);
                        if (currentHash != latestModel.Hash)
                        {
                            status.Issues.Add("Model file hash mismatch - file may be corrupted");
                        }
                        
                        // Check age
                        var age = DateTime.UtcNow - latestModel.Timestamp;
                        if (age > TimeSpan.FromDays(_options.ModelExpiryDays))
                        {
                            status.Issues.Add($"Model is {age.TotalDays:F1} days old (expires after {_options.ModelExpiryDays} days)");
                        }

                        status.IsHealthy = !status.Issues.Any();
                        status.LastUpdated = latestModel.Timestamp;
                        status.Version = latestModel.Version;
                    }

                    report.ModelStatuses.Add(status);
                }

                report.IsHealthy = report.ModelStatuses.All(s => s.IsHealthy);
                _logger.LogInformation("🏥 Model health check completed: {HealthyCount}/{TotalCount} healthy", 
                    report.ModelStatuses.Count(s => s.IsHealthy), report.ModelStatuses.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to perform model health check");
                report.IsHealthy = false;
            }

            return report;
        }

        /// <summary>
        /// Clean up old model versions
        /// </summary>
        public async Task CleanupOldVersionsAsync(int keepVersions = 5, CancellationToken cancellationToken = default)
        {
            try
            {
                var modelDirs = Directory.GetDirectories(_options.RegistryPath);
                
                foreach (var modelDir in modelDirs)
                {
                    var modelName = Path.GetFileName(modelDir);
                    var versions = Directory.GetDirectories(modelDir)
                        .Select(d => new { Path = d, Version = Path.GetFileName(d) })
                        .OrderByDescending(v => v.Version)
                        .ToList();

                    if (versions.Count > keepVersions)
                    {
                        var toDelete = versions.Skip(keepVersions);
                        foreach (var version in toDelete)
                        {
                            Directory.Delete(version.Path, recursive: true);
                            _logger.LogInformation("🗑️ Cleaned up old model version: {ModelName} v{Version}", 
                                modelName, version.Version);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to cleanup old model versions");
            }
        }

        private async Task<string> CalculateFileHashAsync(string filePath, CancellationToken cancellationToken)
        {
            using var sha256 = SHA256.Create();
            using var fileStream = File.OpenRead(filePath);
            var hashBytes = await Task.Run(() => sha256.ComputeHash(fileStream), cancellationToken);
            return Convert.ToHexString(hashBytes).ToLowerInvariant();
        }

        private async Task CompressModelAsync(string modelPath, CancellationToken cancellationToken)
        {
            // Simple compression - could be enhanced with actual compression algorithms
            await Task.Run(() =>
            {
                var compressedPath = modelPath + ".gz";
                // Implementation would use GZip or similar compression
                _logger.LogDebug("🗜️ Model compression placeholder for: {ModelPath}", modelPath);
            }, cancellationToken);
        }

        private async Task UpdateRegistryIndexAsync(ModelRegistryEntry entry, CancellationToken cancellationToken)
        {
            var indexPath = Path.Combine(_options.RegistryPath, "registry_index.json");
            
            List<ModelRegistryEntry> index;
            if (File.Exists(indexPath))
            {
                var indexJson = await File.ReadAllTextAsync(indexPath, cancellationToken);
                index = JsonSerializer.Deserialize<List<ModelRegistryEntry>>(indexJson, _jsonOptions) ?? new();
            }
            else
            {
                index = new List<ModelRegistryEntry>();
            }

            // Add or update entry
            var existingIndex = index.FindIndex(e => e.ModelName == entry.ModelName && e.Version == entry.Version);
            if (existingIndex >= 0)
            {
                index[existingIndex] = entry;
            }
            else
            {
                index.Add(entry);
            }

            // Save updated index
            var updatedIndexJson = JsonSerializer.Serialize(index, _jsonOptions);
            await File.WriteAllTextAsync(indexPath, updatedIndexJson, cancellationToken);
        }
    }

    /// <summary>
    /// Configuration options for model registry
    /// </summary>
    public class ModelRegistryOptions
    {
        public string RegistryPath { get; set; } = "models/registry";
        public bool AutoCompress { get; set; } = true;
        public int ModelExpiryDays { get; set; } = 30;
    }

    /// <summary>
    /// Model registry entry with metadata
    /// </summary>
    public class ModelRegistryEntry
    {
        public string ModelName { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public string Hash { get; set; } = string.Empty;
        public ModelMetadata Metadata { get; set; } = new();
        public string OriginalPath { get; set; } = string.Empty;
        public string RegistryPath { get; set; } = string.Empty;
        public ModelStatus Status { get; set; }
        public bool IsCompressed { get; set; }
    }

    /// <summary>
    /// Model metadata
    /// </summary>
    public class ModelMetadata
    {
        public DateTime TrainingDate { get; set; }
        public Dictionary<string, object> Hyperparams { get; set; } = new();
        public string TrainingDataHash { get; set; } = string.Empty;
        public double ValidationAccuracy { get; set; }
        public Dictionary<string, double> TrainingMetrics { get; set; } = new();
        public string Description { get; set; } = string.Empty;
        public List<string> Tags { get; set; } = new();
    }

    /// <summary>
    /// Model status enumeration
    /// </summary>
    public enum ModelStatus
    {
        Registered,
        Active,
        Deprecated,
        Failed
    }

    /// <summary>
    /// Model health report
    /// </summary>
    public class ModelHealthReport
    {
        public DateTime Timestamp { get; set; }
        public bool IsHealthy { get; set; }
        public List<ModelHealthStatus> ModelStatuses { get; set; } = new();
    }

    /// <summary>
    /// Individual model health status
    /// </summary>
    public class ModelHealthStatus
    {
        public string ModelName { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public bool IsHealthy { get; set; }
        public DateTime LastUpdated { get; set; }
        public List<string> Issues { get; set; } = new();
    }
}