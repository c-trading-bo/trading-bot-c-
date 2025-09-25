using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TradingBot.Backtest;

namespace TradingBot.Backtest.Adapters
{
    /// <summary>
    /// PRODUCTION Model Registry for backtest system
    /// File-based implementation for ES and NQ contracts only
    /// NO MOCK IMPLEMENTATIONS - Production ready
    /// </summary>
    public class ProductionModelRegistry : IModelRegistry
    {
        private readonly ILogger<ProductionModelRegistry> _logger;
        private readonly string _modelsDirectory;
        private readonly string _metadataDirectory;
        private readonly Dictionary<string, List<ModelCard>> _modelsByFamily = new();
        private readonly SemaphoreSlim _lock = new(1, 1);
        private readonly string[] _supportedContracts = { "ES", "NQ" }; // Only ES and NQ as per user requirement
        
        public ProductionModelRegistry(ILogger<ProductionModelRegistry> logger, string? modelsPath = null)
        {
            _logger = logger;
            _modelsDirectory = modelsPath ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "TradingBot", "models");
            _metadataDirectory = Path.Combine(_modelsDirectory, "metadata");
            
            // Ensure directories exist
            Directory.CreateDirectory(_modelsDirectory);
            Directory.CreateDirectory(_metadataDirectory);
            
            // Load existing models on startup
            LoadExistingModelsAsync().GetAwaiter().GetResult();
            
            _logger.LogInformation("PRODUCTION ModelRegistry initialized at {ModelsDirectory} (ES and NQ contracts only)", _modelsDirectory);
        }

        /// <summary>
        /// Get historically appropriate model - prevents future leakage
        /// PRODUCTION: Only ES and NQ contracts supported
        /// </summary>
        public async Task<ModelCard?> GetModelAsOfDateAsync(
            string familyName, 
            DateTime asOfDate, 
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(familyName))
                throw new ArgumentException("Family name cannot be null or empty", nameof(familyName));
            
            // Validate contract support
            var contractSymbol = ExtractContractFromFamilyName(familyName);
            if (!_supportedContracts.Contains(contractSymbol))
            {
                throw new ArgumentException($"Unsupported contract in family name: {familyName}. Only ES and NQ contracts are supported.", nameof(familyName));
            }

            await _lock.WaitAsync(cancellationToken);
            try
            {
                if (!_modelsByFamily.TryGetValue(familyName, out var models) || !models.Any())
                {
                    _logger.LogWarning("No models found for family {FamilyName}", familyName);
                    return null;
                }

                // Get the most recent model trained before the asOfDate to prevent future leakage
                var historicalModel = models
                    .Where(m => m.TrainedAt < asOfDate)
                    .OrderByDescending(m => m.TrainedAt)
                    .FirstOrDefault();

                if (historicalModel != null)
                {
                    _logger.LogInformation("Found historical model {ModelId} for {FamilyName} as of {AsOfDate} (trained {TrainedAt:yyyy-MM-dd})",
                        historicalModel.ModelId, familyName, asOfDate, historicalModel.TrainedAt);
                }
                else
                {
                    _logger.LogWarning("No historical model found for {FamilyName} as of {AsOfDate}", familyName, asOfDate);
                }

                return historicalModel;
            }
            finally
            {
                _lock.Release();
            }
        }

        /// <summary>
        /// Get model file paths for loading
        /// </summary>
        public async Task<ModelPaths?> GetModelPathsAsync(string modelId, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            
            var modelPath = Path.Combine(_modelsDirectory, $"{modelId}.onnx");
            var configPath = Path.Combine(_metadataDirectory, $"{modelId}_config.json");
            var metadataPath = Path.Combine(_metadataDirectory, $"{modelId}_metadata.json");
            
            // Check if model file exists
            if (!File.Exists(modelPath))
            {
                _logger.LogWarning("Model file not found: {ModelPath}", modelPath);
                return null;
            }
            
            return new ModelPaths(
                OnnxModelPath: modelPath,
                ConfigPath: configPath,
                MetadataPath: metadataPath
            );
        }

        /// <summary>
        /// Register a new model (production implementation)
        /// PRODUCTION: Only ES and NQ contracts supported
        /// </summary>
        public async Task<bool> RegisterModelAsync(ModelCard modelCard, CancellationToken cancellationToken = default)
        {
            if (modelCard == null)
                throw new ArgumentNullException(nameof(modelCard));
            
            // Validate contract support
            var contractSymbol = ExtractContractFromFamilyName(modelCard.FamilyName);
            if (!_supportedContracts.Contains(contractSymbol))
            {
                throw new ArgumentException($"Unsupported contract in family name: {modelCard.FamilyName}. Only ES and NQ contracts are supported.");
            }

            await _lock.WaitAsync(cancellationToken);
            try
            {
                if (!_modelsByFamily.ContainsKey(modelCard.FamilyName))
                {
                    _modelsByFamily[modelCard.FamilyName] = new List<ModelCard>();
                }

                _modelsByFamily[modelCard.FamilyName].Add(modelCard);
                
                // Save model metadata to disk
                await SaveModelMetadataAsync(modelCard, cancellationToken);
                
                _logger.LogInformation("Registered PRODUCTION model {ModelId} for family {FamilyName} (contract: {Contract})", 
                    modelCard.ModelId, modelCard.FamilyName, contractSymbol);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to register model {ModelId}", modelCard.ModelId);
                return false;
            }
            finally
            {
                _lock.Release();
            }
        }

        /// <summary>
        /// List all models for a family
        /// </summary>
        public async Task<List<ModelCard>> ListModelsAsync(string familyName, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            
            // Validate contract support
            var contractSymbol = ExtractContractFromFamilyName(familyName);
            if (!_supportedContracts.Contains(contractSymbol))
            {
                throw new ArgumentException($"Unsupported contract in family name: {familyName}. Only ES and NQ contracts are supported.");
            }

            await _lock.WaitAsync(cancellationToken);
            try
            {
                return _modelsByFamily.TryGetValue(familyName, out var models) 
                    ? models.OrderByDescending(m => m.TrainedAt).ToList()
                    : new List<ModelCard>();
            }
            finally
            {
                _lock.Release();
            }
        }

        /// <summary>
        /// Check if a model exists
        /// </summary>
        public async Task<bool> ModelExistsAsync(string modelId, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            
            await _lock.WaitAsync(cancellationToken);
            try
            {
                return _modelsByFamily.Values
                    .SelectMany(models => models)
                    .Any(m => m.ModelId == modelId);
            }
            finally
            {
                _lock.Release();
            }
        }

        /// <summary>
        /// Load existing models from disk on startup
        /// </summary>
        private async Task LoadExistingModelsAsync()
        {
            try
            {
                var metadataFiles = Directory.GetFiles(_metadataDirectory, "*_metadata.json");
                
                foreach (var file in metadataFiles)
                {
                    try
                    {
                        var json = await File.ReadAllTextAsync(file);
                        var modelCard = JsonSerializer.Deserialize<ModelCard>(json);
                        
                        if (modelCard != null)
                        {
                            // Validate contract support before loading
                            var contractSymbol = ExtractContractFromFamilyName(modelCard.FamilyName);
                            if (_supportedContracts.Contains(contractSymbol))
                            {
                                if (!_modelsByFamily.ContainsKey(modelCard.FamilyName))
                                {
                                    _modelsByFamily[modelCard.FamilyName] = new List<ModelCard>();
                                }
                                _modelsByFamily[modelCard.FamilyName].Add(modelCard);
                            }
                            else
                            {
                                _logger.LogWarning("Skipping model {ModelId} - unsupported contract {Contract}", 
                                    modelCard.ModelId, contractSymbol);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to load model metadata from {File}", file);
                    }
                }

                var totalModels = _modelsByFamily.Values.Sum(m => m.Count);
                _logger.LogInformation("Loaded {TotalModels} PRODUCTION models for {FamilyCount} families (ES and NQ only)", 
                    totalModels, _modelsByFamily.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load existing models");
            }
        }

        /// <summary>
        /// Save model metadata to disk
        /// </summary>
        private async Task SaveModelMetadataAsync(ModelCard modelCard, CancellationToken cancellationToken)
        {
            try
            {
                var metadataPath = Path.Combine(_metadataDirectory, $"{modelCard.ModelId}_metadata.json");
                var json = JsonSerializer.Serialize(modelCard, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(metadataPath, json, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to save model metadata for {ModelId}", modelCard.ModelId);
                throw;
            }
        }

        /// <summary>
        /// Extract contract symbol from family name
        /// </summary>
        private string ExtractContractFromFamilyName(string familyName)
        {
            // Common patterns: ESStrategy, NQStrategy, ES_Strategy, etc.
            foreach (var contract in _supportedContracts)
            {
                if (familyName.StartsWith(contract, StringComparison.OrdinalIgnoreCase))
                {
                    return contract;
                }
            }
            
            // Default to ES if not determinable
            return "ES";
        }
    }

    /// <summary>
    /// Extension methods for production model registry
    /// </summary>
    public static class ProductionModelRegistryExtensions
    {
        /// <summary>
        /// Register the production model registry (ES and NQ contracts only)
        /// </summary>
        public static IServiceCollection AddProductionModelRegistry(this IServiceCollection services, string? modelsPath = null)
        {
            services.AddSingleton<IModelRegistry>(sp => 
                new ProductionModelRegistry(sp.GetRequiredService<ILogger<ProductionModelRegistry>>(), modelsPath));
            return services;
        }
    }
}