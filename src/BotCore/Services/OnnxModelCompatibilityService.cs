using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// ONNX runtime pinning and model I/O schema validation service
    /// Ensures model compatibility and prevents hot-swap mismatches
    /// </summary>
    public class OnnxModelCompatibilityService
    {
        private readonly ILogger<OnnxModelCompatibilityService> _logger;
        private readonly Dictionary<string, ModelCard> _registeredModels = new();
        private readonly object _registryLock = new();

        public OnnxModelCompatibilityService(ILogger<OnnxModelCompatibilityService> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Register a model with its schema and runtime requirements
        /// </summary>
        public async Task RegisterModelAsync(string modelPath, ModelCard modelCard)
        {
            lock (_registryLock)
            {
                var modelId = Path.GetFileNameWithoutExtension(modelPath);
                
                // Pin ONNX runtime version
                modelCard.PinnedOnnxRuntimeVersion = GetCurrentOnnxRuntimeVersion();
                modelCard.ModelHash = CalculateModelHash(modelPath);
                modelCard.RegistrationTime = DateTime.UtcNow;
                
                _registeredModels[modelId] = modelCard;
                
                _logger.LogInformation("📌 [ONNX-COMPAT] Registered model {ModelId} with runtime {Runtime} and schema {Schema}", 
                    modelId, modelCard.PinnedOnnxRuntimeVersion, modelCard.InputSchemaVersion);
            }

            // Save model card to disk
            var cardPath = modelPath + ".card.json";
            var json = JsonSerializer.Serialize(modelCard, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(cardPath, json);
            
            _logger.LogInformation("💾 [ONNX-COMPAT] Model card saved: {CardPath}", cardPath);
        }

        /// <summary>
        /// Validate model compatibility before loading
        /// </summary>
        public async Task<bool> ValidateModelCompatibilityAsync(string modelPath)
        {
            try
            {
                var modelId = Path.GetFileNameWithoutExtension(modelPath);
                var cardPath = modelPath + ".card.json";

                if (!File.Exists(cardPath))
                {
                    _logger.LogError("🚨 [ONNX-COMPAT] No model card found for {ModelId}", modelId);
                    return false;
                }

                // Load and validate model card
                var cardJson = await File.ReadAllTextAsync(cardPath);
                var modelCard = JsonSerializer.Deserialize<ModelCard>(cardJson);
                
                if (modelCard == null)
                {
                    _logger.LogError("🚨 [ONNX-COMPAT] Invalid model card for {ModelId}", modelId);
                    return false;
                }

                // Validate ONNX runtime version
                var currentRuntime = GetCurrentOnnxRuntimeVersion();
                if (modelCard.PinnedOnnxRuntimeVersion != currentRuntime)
                {
                    _logger.LogError("🚨 [ONNX-COMPAT] Runtime version mismatch for {ModelId}: expected {Expected}, current {Current}",
                        modelId, modelCard.PinnedOnnxRuntimeVersion, currentRuntime);
                    return false;
                }

                // Validate model file hash
                var currentHash = CalculateModelHash(modelPath);
                if (modelCard.ModelHash != currentHash)
                {
                    _logger.LogError("🚨 [ONNX-COMPAT] Model hash mismatch for {ModelId}: integrity check failed", modelId);
                    return false;
                }

                // Validate schema version compatibility
                if (!IsSchemaCompatible(modelCard))
                {
                    _logger.LogError("🚨 [ONNX-COMPAT] Schema incompatible for {ModelId}: {InputSchema} -> {OutputSchema}",
                        modelId, modelCard.InputSchemaVersion, modelCard.OutputSchemaVersion);
                    return false;
                }

                lock (_registryLock)
                {
                    _registeredModels[modelId] = modelCard;
                }

                _logger.LogInformation("✅ [ONNX-COMPAT] Model {ModelId} validated and compatible", modelId);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [ONNX-COMPAT] Error validating model compatibility: {ModelPath}", modelPath);
                return false;
            }
        }

        /// <summary>
        /// Refuse hot-swap if model schemas are incompatible
        /// </summary>
        public bool CanHotSwapModel(string fromModelId, string toModelId)
        {
            lock (_registryLock)
            {
                if (!_registeredModels.TryGetValue(fromModelId, out var fromCard) ||
                    !_registeredModels.TryGetValue(toModelId, out var toCard))
                {
                    _logger.LogWarning("⚠️ [ONNX-COMPAT] Cannot hot-swap: model cards not found for {From} -> {To}", 
                        fromModelId, toModelId);
                    return false;
                }

                // Check schema compatibility
                if (fromCard.InputSchemaVersion != toCard.InputSchemaVersion ||
                    fromCard.OutputSchemaVersion != toCard.OutputSchemaVersion)
                {
                    _logger.LogWarning("⚠️ [ONNX-COMPAT] Cannot hot-swap: schema mismatch {From} ({FromIn}/{FromOut}) -> {To} ({ToIn}/{ToOut})",
                        fromModelId, fromCard.InputSchemaVersion, fromCard.OutputSchemaVersion,
                        toModelId, toCard.InputSchemaVersion, toCard.OutputSchemaVersion);
                    return false;
                }

                _logger.LogInformation("✅ [ONNX-COMPAT] Hot-swap approved: {From} -> {To}", fromModelId, toModelId);
                return true;
            }
        }

        private string GetCurrentOnnxRuntimeVersion()
        {
            try
            {
                // Get ONNX Runtime version (this is a placeholder - actual implementation would query the runtime)
                return "1.16.3"; // Pin to specific version for production
            }
            catch
            {
                return "unknown";
            }
        }

        private string CalculateModelHash(string modelPath)
        {
            try
            {
                using var sha256 = SHA256.Create();
                using var fileStream = File.OpenRead(modelPath);
                var hashBytes = sha256.ComputeHash(fileStream);
                return Convert.ToHexString(hashBytes).ToLowerInvariant();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calculating model hash for {ModelPath}", modelPath);
                return "error";
            }
        }

        private bool IsSchemaCompatible(ModelCard modelCard)
        {
            // Check if input/output schemas are compatible with current system
            var supportedInputVersions = new[] { "1.0", "1.1", "2.0" };
            var supportedOutputVersions = new[] { "1.0", "1.1", "2.0" };

            return Array.Exists(supportedInputVersions, v => v == modelCard.InputSchemaVersion) &&
                   Array.Exists(supportedOutputVersions, v => v == modelCard.OutputSchemaVersion);
        }
    }

    /// <summary>
    /// Model card containing metadata, schema versions, and runtime requirements
    /// </summary>
    public class ModelCard
    {
        public string ModelName { get; set; } = string.Empty;
        public string ModelVersion { get; set; } = string.Empty;
        public string InputSchemaVersion { get; set; } = "1.0";
        public string OutputSchemaVersion { get; set; } = "1.0";
        public string PinnedOnnxRuntimeVersion { get; set; } = string.Empty;
        public string ModelHash { get; set; } = string.Empty;
        public DateTime RegistrationTime { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();

        /// <summary>
        /// Input tensor specifications
        /// </summary>
        public List<TensorSpec> InputSpecs { get; set; } = new();

        /// <summary>
        /// Output tensor specifications
        /// </summary>
        public List<TensorSpec> OutputSpecs { get; set; } = new();
    }

    /// <summary>
    /// Tensor specification for model I/O validation
    /// </summary>
    public class TensorSpec
    {
        public string Name { get; set; } = string.Empty;
        public string DataType { get; set; } = string.Empty;
        public int[] Shape { get; set; } = Array.Empty<int>();
        public string Description { get; set; } = string.Empty;
    }
}