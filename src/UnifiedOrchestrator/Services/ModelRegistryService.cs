using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    public class ModelRegistryOptions
    {
        public string RegistryPath { get; set; } = "models/registry";
        public bool AutoCompress { get; set; } = true;
        public int ModelExpiryDays { get; set; } = 90;
        public int MaxModelsPerName { get; set; } = 10;
    }

    public class ModelMetadata
    {
        public DateTime TrainingDate { get; set; }
        public Dictionary<string, object> Hyperparams { get; set; } = new();
        public double ValidationAccuracy { get; set; }
        public string Description { get; set; } = "";
        public string Version { get; set; } = "";
        public double F1Score { get; set; }
        public DateTime LastUpdated { get; set; }
    }

    public class ModelRegistryEntry
    {
        public string ModelName { get; set; } = "";
        public string ModelPath { get; set; } = "";
        public ModelMetadata Metadata { get; set; } = new();
        public DateTime RegisteredAt { get; set; }
        public string RegistryId { get; set; } = "";
    }

    public class ModelRegistryService
    {
        private readonly ILogger<ModelRegistryService> _logger;
        private readonly ModelRegistryOptions _options;

        public ModelRegistryService(ILogger<ModelRegistryService> logger, IOptions<ModelRegistryOptions> options)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            
            Directory.CreateDirectory(_options.RegistryPath);
        }

        public async Task<ModelRegistryEntry> RegisterModelAsync(string modelName, string modelPath, ModelMetadata metadata, CancellationToken cancellationToken)
        {
            try
            {
                var registryId = Guid.NewGuid().ToString();
                var timestamp = DateTime.UtcNow;
                
                // Create model directory
                var modelDir = Path.Combine(_options.RegistryPath, modelName, registryId);
                Directory.CreateDirectory(modelDir);

                // Copy model file
                var targetModelPath = Path.Combine(modelDir, Path.GetFileName(modelPath));
                using (var sourceStream = new FileStream(modelPath, FileMode.Open, FileAccess.Read))
                using (var destinationStream = new FileStream(targetModelPath, FileMode.Create, FileAccess.Write))
                {
                    await sourceStream.CopyToAsync(destinationStream, cancellationToken).ConfigureAwait(false);
                }

                // Update metadata
                metadata.LastUpdated = timestamp;

                // Create registry entry
                var entry = new ModelRegistryEntry
                {
                    ModelName = modelName,
                    ModelPath = targetModelPath,
                    Metadata = metadata,
                    RegisteredAt = timestamp,
                    RegistryId = registryId
                };

                // Save metadata
                var metadataPath = Path.Combine(modelDir, "metadata.json");
                var metadataJson = JsonSerializer.Serialize(entry, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(metadataPath, metadataJson, cancellationToken).ConfigureAwait(false);

                _logger.LogInformation("Registered model {ModelName} with ID {RegistryId}", modelName, registryId);
                
                // Cleanup old models if needed
                await CleanupOldModelsAsync(modelName, cancellationToken).ConfigureAwait(false);

                return entry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to register model {ModelName}", modelName);
                throw;
            }
        }

        public async Task<ModelRegistryEntry?> GetLatestModelAsync(string modelName, CancellationToken cancellationToken)
        {
            try
            {
                var modelDir = Path.Combine(_options.RegistryPath, modelName);
                
                if (!Directory.Exists(modelDir))
                {
                    _logger.LogWarning("Model directory not found for {ModelName}", modelName);
                    return null;
                }

                ModelRegistryEntry? latestEntry = null!;
                var latestTime = DateTime.MinValue;

                foreach (var dir in Directory.GetDirectories(modelDir))
                {
                    var metadataPath = Path.Combine(dir, "metadata.json");
                    if (File.Exists(metadataPath))
                    {
                        var metadataJson = await File.ReadAllTextAsync(metadataPath, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                        var entry = JsonSerializer.Deserialize<ModelRegistryEntry>(metadataJson);
                        
                        if (entry != null && entry.RegisteredAt > latestTime)
                        {
                            latestTime = entry.RegisteredAt;
                            latestEntry = entry;
                        }
                    }
                }

                if (latestEntry != null)
                {
                    _logger.LogDebug("Found latest model {ModelName} registered at {RegisteredAt}", 
                        modelName, latestEntry.RegisteredAt);
                }

                return latestEntry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to get latest model {ModelName}", modelName);
                throw;
            }
        }

        private async Task CleanupOldModelsAsync(string modelName, CancellationToken cancellationToken)
        {
            try
            {
                var modelDir = Path.Combine(_options.RegistryPath, modelName);
                var dirs = Directory.GetDirectories(modelDir);
                
                if (dirs.Length <= _options.MaxModelsPerName)
                    return;

                var entries = new List<(string dir, DateTime registeredAt)>();
                
                foreach (var dir in dirs)
                {
                    var metadataPath = Path.Combine(dir, "metadata.json");
                    if (File.Exists(metadataPath))
                    {
                        var metadataJson = await File.ReadAllTextAsync(metadataPath, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                        var entry = JsonSerializer.Deserialize<ModelRegistryEntry>(metadataJson);
                        if (entry != null)
                        {
                            entries.Add((dir, entry.RegisteredAt));
                        }
                    }
                }

                // Sort by date and remove oldest
                entries.Sort((a, b) => b.registeredAt.CompareTo(a.registeredAt));
                
                for (int i = _options.MaxModelsPerName; i < entries.Count; i++)
                {
                    Directory.Delete(entries[i].dir, true);
                    _logger.LogInformation("Cleaned up old model version in {Dir}", entries[i].dir);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to cleanup old models for {ModelName}", modelName);
            }
        }
    }
}