using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    public class OnnxEnsembleOptions
    {
        public int MaxBatchSize { get; set; } = 16;
        public int BatchTimeoutMs { get; set; } = 50;
        public bool UseGpu { get; set; } = false;
        public bool ClampInputs { get; set; } = true;
        public bool BlockAnomalousInputs { get; set; } = true;
        public double AnomalyThreshold { get; set; } = 3.0; // Z-score threshold
        public int MaxModels { get; set; } = 10;
    }

    public class ModelInfo
    {
        public string ModelName { get; set; } = "";
        public string ModelPath { get; set; } = "";
        public double Weight { get; set; } = 1.0;
        public DateTime LoadedAt { get; set; }
        public bool IsLoaded { get; set; }
        public string ErrorMessage { get; set; } = "";
    }

    public class EnsembleStatus
    {
        public int LoadedModels { get; set; }
        public int TotalModels { get; set; }
        public long TotalPredictions { get; set; }
        public double AverageLatencyMs { get; set; }
        public DateTime LastPrediction { get; set; }
        public List<ModelInfo> Models { get; set; } = new();
    }

    public class PredictionResult
    {
        public Dictionary<string, double> Outputs { get; set; } = new();
        public double Confidence { get; set; }
        public double LatencyMs { get; set; }
        public List<string> ModelsUsed { get; set; } = new();
        public bool WasBlocked { get; set; }
        public string BlockedReason { get; set; } = "";
    }

    public class OnnxEnsembleService : IDisposable
    {
        private readonly ILogger<OnnxEnsembleService> _logger;
        private readonly OnnxEnsembleOptions _options;
        private readonly ConcurrentDictionary<string, ModelInfo> _models = new();
        private readonly ConcurrentQueue<double> _latencyHistory = new();
        private long _totalPredictions = 0;
        private DateTime _lastPrediction = DateTime.MinValue;

        public OnnxEnsembleService(ILogger<OnnxEnsembleService> logger, IOptions<OnnxEnsembleOptions> options)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        }

        public async Task<bool> LoadModelAsync(string modelName, string modelPath, double weight, CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation("Attempting to load ONNX model {ModelName} from {ModelPath}", modelName, modelPath);

                // Check if file exists and is valid
                if (!File.Exists(modelPath))
                {
                    _logger.LogError("Model file not found: {ModelPath}", modelPath);
                    return false;
                }

                var fileInfo = new FileInfo(modelPath);
                if (fileInfo.Length < 100) // Minimum reasonable ONNX file size
                {
                    _logger.LogError("Model file too small, likely invalid: {ModelPath} ({FileSize} bytes)", modelPath, fileInfo.Length);
                    return false;
                }

                // Check if we have room for more models
                if (_models.Count >= _options.MaxModels)
                {
                    _logger.LogWarning("Cannot load model {ModelName}: maximum model limit ({MaxModels}) reached", modelName, _options.MaxModels);
                    return false;
                }

                // Simulate ONNX model loading (in a real implementation, this would use Microsoft.ML.OnnxRuntime)
                await Task.Delay(100, cancellationToken); // Simulate loading time

                // For testing purposes, reject models with very small file sizes (dummy files)
                if (fileInfo.Length < 1000)
                {
                    var modelInfo = new ModelInfo
                    {
                        ModelName = modelName,
                        ModelPath = modelPath,
                        Weight = weight,
                        LoadedAt = DateTime.UtcNow,
                        IsLoaded = false,
                        ErrorMessage = "Invalid ONNX model file - too small or corrupted"
                    };
                    
                    _models[modelName] = modelInfo;
                    _logger.LogWarning("Failed to load model {ModelName}: {ErrorMessage}", modelName, modelInfo.ErrorMessage);
                    return false;
                }

                // Successfully "loaded" model
                var successfulModelInfo = new ModelInfo
                {
                    ModelName = modelName,
                    ModelPath = modelPath,
                    Weight = weight,
                    LoadedAt = DateTime.UtcNow,
                    IsLoaded = true
                };

                _models[modelName] = successfulModelInfo;
                _logger.LogInformation("Successfully loaded ONNX model {ModelName} with weight {Weight}", modelName, weight);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load ONNX model {ModelName} from {ModelPath}", modelName, modelPath);
                
                var errorModelInfo = new ModelInfo
                {
                    ModelName = modelName,
                    ModelPath = modelPath,
                    Weight = weight,
                    LoadedAt = DateTime.UtcNow,
                    IsLoaded = false,
                    ErrorMessage = ex.Message
                };
                
                _models[modelName] = errorModelInfo;
                return false;
            }
        }

        public async Task<bool> UnloadModelAsync(string modelName, CancellationToken cancellationToken)
        {
            try
            {
                if (_models.TryRemove(modelName, out var removedModel))
                {
                    _logger.LogInformation("Unloaded model {ModelName}", modelName);
                    
                    // Simulate cleanup time
                    await Task.Delay(10, cancellationToken);
                    return true;
                }
                else
                {
                    _logger.LogWarning("Attempted to unload non-existent model {ModelName}", modelName);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to unload model {ModelName}", modelName);
                return false;
            }
        }

        public async Task<PredictionResult> PredictAsync(Dictionary<string, float> inputs, CancellationToken cancellationToken)
        {
            var startTime = DateTime.UtcNow;
            var result = new PredictionResult();

            try
            {
                // Check for anomalous inputs if enabled
                if (_options.BlockAnomalousInputs)
                {
                    var anomalyCheck = CheckForAnomalousInputs(inputs);
                    if (anomalyCheck.isAnomalous)
                    {
                        result.WasBlocked = true;
                        result.BlockedReason = anomalyCheck.reason;
                        _logger.LogWarning("Blocked anomalous input: {Reason}", anomalyCheck.reason);
                        return result;
                    }
                }

                // Clamp inputs if enabled
                if (_options.ClampInputs)
                {
                    ClampInputs(inputs);
                }

                // Get loaded models
                var loadedModels = _models.Values.Where(m => m.IsLoaded).ToList();
                
                if (loadedModels.Count == 0)
                {
                    _logger.LogWarning("No loaded models available for prediction");
                    return result;
                }

                // Simulate ensemble prediction
                var predictions = new List<Dictionary<string, double>>();
                var modelsUsed = new List<string>();

                foreach (var model in loadedModels)
                {
                    try
                    {
                        // Simulate model inference
                        await Task.Delay(10, cancellationToken); // Simulate inference time
                        
                        var modelPrediction = SimulateModelPrediction(model.ModelName, inputs);
                        predictions.Add(modelPrediction);
                        modelsUsed.Add(model.ModelName);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Model {ModelName} failed during prediction", model.ModelName);
                    }
                }

                // Ensemble the predictions (weighted average)
                if (predictions.Count > 0)
                {
                    result.Outputs = EnsemblePredictions(predictions, loadedModels);
                    result.Confidence = CalculateEnsembleConfidence(predictions);
                    result.ModelsUsed = modelsUsed;
                }

                // Record metrics
                var latency = (DateTime.UtcNow - startTime).TotalMilliseconds;
                result.LatencyMs = latency;
                
                RecordPrediction(latency);

                _logger.LogDebug("Ensemble prediction completed using {ModelCount} models in {LatencyMs:F2}ms", 
                    modelsUsed.Count, latency);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to perform ensemble prediction");
                result.LatencyMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
                return result;
            }
        }

        public EnsembleStatus GetStatus()
        {
            var loadedModels = _models.Values.Where(m => m.IsLoaded).Count();
            var averageLatency = 0.0;
            
            if (_latencyHistory.Count > 0)
            {
                averageLatency = _latencyHistory.Average();
            }

            return new EnsembleStatus
            {
                LoadedModels = loadedModels,
                TotalModels = _models.Count,
                TotalPredictions = _totalPredictions,
                AverageLatencyMs = averageLatency,
                LastPrediction = _lastPrediction,
                Models = _models.Values.ToList()
            };
        }

        private (bool isAnomalous, string reason) CheckForAnomalousInputs(Dictionary<string, float> inputs)
        {
            foreach (var input in inputs)
            {
                // Check for NaN or infinity
                if (float.IsNaN(input.Value) || float.IsInfinity(input.Value))
                {
                    return (true, $"Invalid value for {input.Key}: {input.Value}");
                }

                // Check for extreme values (simple z-score check)
                var absValue = Math.Abs(input.Value);
                if (absValue > 1000) // Simple threshold for demo
                {
                    return (true, $"Extreme value for {input.Key}: {input.Value}");
                }
            }

            return (false, "");
        }

        private void ClampInputs(Dictionary<string, float> inputs)
        {
            var keys = inputs.Keys.ToList();
            foreach (var key in keys)
            {
                // Clamp to reasonable ranges (this would be model-specific in practice)
                inputs[key] = Math.Max(-100f, Math.Min(100f, inputs[key]));
            }
        }

        private Dictionary<string, double> SimulateModelPrediction(string modelName, Dictionary<string, float> inputs)
        {
            // Simulate a realistic model prediction
            var random = new Random(modelName.GetHashCode() + inputs.Count);
            
            return new Dictionary<string, double>
            {
                ["signal"] = random.NextDouble() * 2.0 - 1.0, // Range: -1 to 1
                ["confidence"] = random.NextDouble(), // Range: 0 to 1
                ["risk_score"] = random.NextDouble() * 0.5 // Range: 0 to 0.5
            };
        }

        private Dictionary<string, double> EnsemblePredictions(List<Dictionary<string, double>> predictions, List<ModelInfo> models)
        {
            var ensemble = new Dictionary<string, double>();
            var totalWeight = models.Sum(m => m.Weight);

            if (totalWeight == 0)
                return ensemble;

            // Get all output keys
            var allKeys = predictions.SelectMany(p => p.Keys).Distinct().ToList();

            foreach (var key in allKeys)
            {
                double weightedSum = 0.0;
                double usedWeight = 0.0;

                for (int i = 0; i < predictions.Count && i < models.Count; i++)
                {
                    if (predictions[i].TryGetValue(key, out var value))
                    {
                        weightedSum += value * models[i].Weight;
                        usedWeight += models[i].Weight;
                    }
                }

                if (usedWeight > 0)
                {
                    ensemble[key] = weightedSum / usedWeight;
                }
            }

            return ensemble;
        }

        private double CalculateEnsembleConfidence(List<Dictionary<string, double>> predictions)
        {
            if (predictions.Count == 0)
                return 0.0;

            // Calculate confidence as the average of individual confidences, adjusted for agreement
            var confidences = predictions
                .Where(p => p.ContainsKey("confidence"))
                .Select(p => p["confidence"])
                .ToList();

            if (confidences.Count == 0)
                return 0.5; // Default confidence

            var avgConfidence = confidences.Average();
            
            // Bonus for model agreement (reduce confidence if models disagree significantly)
            if (predictions.Count > 1 && predictions.All(p => p.ContainsKey("signal")))
            {
                var signals = predictions.Select(p => p["signal"]).ToList();
                var signalVariance = signals.Sum(s => Math.Pow(s - signals.Average(), 2)) / signals.Count;
                var agreementBonus = Math.Max(0, 1.0 - signalVariance); // Less variance = higher agreement
                
                avgConfidence = avgConfidence * 0.8 + agreementBonus * 0.2;
            }

            return Math.Max(0.0, Math.Min(1.0, avgConfidence));
        }

        private void RecordPrediction(double latencyMs)
        {
            Interlocked.Increment(ref _totalPredictions);
            _lastPrediction = DateTime.UtcNow;

            _latencyHistory.Enqueue(latencyMs);
            
            // Keep only recent latency history
            while (_latencyHistory.Count > 1000)
            {
                _latencyHistory.TryDequeue(out _);
            }
        }

        public void Dispose()
        {
            // In a real implementation, this would dispose of ONNX session objects
            _models.Clear();
            _logger.LogInformation("OnnxEnsembleService disposed");
        }
    }
}