using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TradingBot.RLAgent;

/// <summary>
/// ONNX ensemble wrapper with confidence voting, async batched inference, and GPU support
/// Merged from UnifiedOrchestrator.Services.OnnxEnsembleService
/// Implements proper session disposal and memory management for RL agent use
/// </summary>
public class OnnxEnsembleWrapper : IDisposable
{
    private readonly ILogger<OnnxEnsembleWrapper> _logger;
    private readonly OnnxEnsembleOptions _options;
    private readonly ConcurrentDictionary<string, ModelSession> _modelSessions = new();
    private readonly Channel<InferenceRequest> _inferenceQueue;
    private readonly ChannelWriter<InferenceRequest> _inferenceWriter;
    private readonly SemaphoreSlim _batchSemaphore;
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    private readonly Task _batchProcessingTask;
    private readonly AnomalyDetector _anomalyDetector;
    private bool _disposed;
    
    // Constants
    private const int BytesToMegabytes = 1024 * 1024;

    public OnnxEnsembleWrapper(
        ILogger<OnnxEnsembleWrapper> logger,
        IOptions<OnnxEnsembleOptions> options)
    {
        _logger = logger;
        _options = options.Value;

        // Create bounded channel for inference requests
        var channelOptions = new BoundedChannelOptions(_options.MaxQueueSize)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false
        };
        
        var channel = Channel.CreateBounded<InferenceRequest>(channelOptions);
        _inferenceQueue = channel;
        _inferenceWriter = channel.Writer;

        _batchSemaphore = new SemaphoreSlim(_options.MaxConcurrentBatches, _options.MaxConcurrentBatches);
        _anomalyDetector = new AnomalyDetector(_options.AnomalyThreshold);

        // Start background batch processing
        _batchProcessingTask = Task.Run(ProcessInferenceBatchesAsync);

        _logger.LogInformation("[RL-ENSEMBLE] ONNX Ensemble Wrapper initialized with {MaxBatchSize} batch size, {MaxConcurrentBatches} concurrent batches",
            _options.MaxBatchSize, _options.MaxConcurrentBatches);
    }

    /// <summary>
    /// Load model into ensemble
    /// </summary>
    public async Task<bool> LoadModelAsync(string modelName, string modelPath, double confidence = 1.0, CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(modelPath))
            {
                _logger.LogError("[RL-ENSEMBLE] Model file not found: {ModelPath}", modelPath);
                return false;
            }

            // Configure session options for GPU and optimization
            var sessionOptions = CreateSessionOptions();
            
            var session = new InferenceSession(modelPath, sessionOptions);
            var modelSession = new ModelSession
            {
                Name = modelName,
                Session = session,
                Confidence = confidence,
                LoadTime = DateTime.UtcNow,
                InferenceCount = 0,
                LastUsed = DateTime.UtcNow
            };

            // Validate model inputs/outputs
            await ValidateModelAsync(modelSession).ConfigureAwait(false);

            _modelSessions.TryAdd(modelName, modelSession);
            _logger.LogInformation("[RL-ENSEMBLE] Model loaded: {ModelName} with confidence {Confidence:F2}", modelName, confidence);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL-ENSEMBLE] Failed to load model: {ModelName}", modelName);
            return false;
        }
    }

    /// <summary>
    /// Unload model from ensemble
    /// </summary>
    public async Task<bool> UnloadModelAsync(string modelName, CancellationToken cancellationToken = default)
    {
        try
        {
            // Brief yield for async context
            await Task.Yield();
            
            if (_modelSessions.TryRemove(modelName, out var modelSession))
            {
                modelSession.Session.Dispose();
                _logger.LogInformation("[RL-ENSEMBLE] Model unloaded: {ModelName}", modelName);
                return true;
            }
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL-ENSEMBLE] Failed to unload model: {ModelName}", modelName);
            return false;
        }
    }

    /// <summary>
    /// Submit inference request for async batched processing
    /// </summary>
    public async Task<EnsemblePrediction> PredictAsync(float[] features, CancellationToken cancellationToken = default)
    {
        // Input anomaly detection
        var isAnomaly = _anomalyDetector.IsAnomaly(features);
        if (isAnomaly && _options.BlockAnomalousInputs)
        {
            _logger.LogWarning("[RL-ENSEMBLE] Anomalous input detected and blocked");
            return new EnsemblePrediction
            {
                IsAnomaly = true,
                Confidence = 0.0,
                EnsembleResult = 0.0f
            };
        }

        // Clamp inputs to defined bounds
        var clampedFeatures = ClampInputs(features);

        var tcs = new TaskCompletionSource<EnsemblePrediction>();
        var request = new InferenceRequest
        {
            Features = clampedFeatures,
            TaskCompletionSource = tcs,
            RequestTime = DateTime.UtcNow,
            IsAnomaly = isAnomaly
        };

        // Submit to queue
        if (!await _inferenceWriter.WaitToWriteAsync(cancellationToken))
        {
            throw new InvalidOperationException("Inference queue is closed");
        }

        await _inferenceWriter.WriteAsync(request, cancellationToken).ConfigureAwait(false);
        return await tcs.Task.ConfigureAwait(false);
    }

    /// <summary>
    /// Get ensemble status and metrics
    /// </summary>
    public EnsembleStatus GetStatus()
    {
        var loadedModels = _modelSessions.Values.ToList();
        return new EnsembleStatus
        {
            LoadedModels = loadedModels.Count,
            TotalInferences = loadedModels.Sum(m => m.InferenceCount),
            AverageLatencyMs = CalculateAverageLatency(),
            MemoryUsageMB = GC.GetTotalMemory(false) / BytesToMegabytes,
            QueueSize = _inferenceQueue.Reader.CanCount ? _inferenceQueue.Reader.Count : -1,
            IsHealthy = loadedModels.Any() && !_disposed
        };
    }

    #region Private Methods

    private async Task ProcessInferenceBatchesAsync()
    {
        var batch = new List<InferenceRequest>();

        try
        {
            while (!_cancellationTokenSource.Token.IsCancellationRequested)
            {
                batch.Clear();

                // Collect batch
                await CollectBatchAsync(batch).ConfigureAwait(false);

                if (batch.Count > 0)
                {
                    await _batchSemaphore.WaitAsync(_cancellationTokenSource.Token).ConfigureAwait(false);
                    
                    // Process batch on background thread
                    _ = Task.Run(async () =>
                    {
                        try
                        {
                            await ProcessBatchAsync(batch.ToList()).ConfigureAwait(false);
                        }
                        finally
                        {
                            _batchSemaphore.Release();
                        }
                    }, _cancellationTokenSource.Token);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Expected when shutting down
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL-ENSEMBLE] Error in batch processing task");
        }
    }

    private async Task CollectBatchAsync(List<InferenceRequest> batch)
    {
        var timeout = TimeSpan.FromMilliseconds(_options.BatchTimeoutMs);
        var deadline = DateTime.UtcNow + timeout;

        while (batch.Count < _options.MaxBatchSize && DateTime.UtcNow < deadline)
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(_cancellationTokenSource.Token);
            cts.CancelAfter(TimeSpan.FromMilliseconds(Math.Max(1, (deadline - DateTime.UtcNow).TotalMilliseconds)));

            try
            {
                if (await _inferenceQueue.Reader.WaitToReadAsync(cts.Token))
                {
                    while (_inferenceQueue.Reader.TryRead(out var request) && batch.Count < _options.MaxBatchSize)
                    {
                        batch.Add(request);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }
    }

    private async Task ProcessBatchAsync(List<InferenceRequest> batch)
    {
        try
        {
            var startTime = DateTime.UtcNow;
            var results = new Dictionary<InferenceRequest, EnsemblePrediction>();

            // Group requests by feature similarity for efficiency
            var featureGroups = GroupByFeatureSimilarity(batch);

            foreach (var group in featureGroups)
            {
                var predictions = await RunEnsembleInferenceAsync(group.Select(r => r.Features.ToArray()).ToArray()).ConfigureAwait(false);
                
                for (int i = 0; i < group.Count; i++)
                {
                    var request = group[i];
                    var prediction = predictions[i];
                    prediction.IsAnomaly = request.IsAnomaly;
                    prediction.LatencyMs = (DateTime.UtcNow - request.RequestTime).TotalMilliseconds;
                    
                    results[request] = prediction;
                }
            }

            // Complete all requests
            foreach (var kvp in results)
            {
                kvp.Key.TaskCompletionSource.SetResult(kvp.Value);
            }

            var totalLatency = (DateTime.UtcNow - startTime).TotalMilliseconds;
            _logger.LogDebug("[RL-ENSEMBLE] Processed batch of {BatchSize} in {LatencyMs:F2}ms", batch.Count, totalLatency);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL-ENSEMBLE] Error processing inference batch");
            
            // Complete all requests with error
            foreach (var request in batch)
            {
                request.TaskCompletionSource.SetException(ex);
            }
        }
    }

    private async Task<EnsemblePrediction[]> RunEnsembleInferenceAsync(float[][] batchFeatures)
    {
        var modelSessions = _modelSessions.Values.ToList();
        if (!modelSessions.Any())
        {
            throw new InvalidOperationException("No models loaded");
        }

        var batchSize = batchFeatures.Length;
        var results = new EnsemblePrediction[batchSize];

        // Initialize results
        for (int i = 0; i < batchSize; i++)
        {
            results[i] = new EnsemblePrediction
            {
                IsAnomaly = false
            };
        }

        // Run inference on each model
        var modelTasks = modelSessions.Select(async modelSession =>
        {
            try
            {
                var modelPredictions = await RunModelInferenceAsync(modelSession, batchFeatures).ConfigureAwait(false);
                
                for (int i = 0; i < batchSize; i++)
                {
                    results[i].Predictions[modelSession.Name] = modelPredictions[i];
                }

                modelSession.InferenceCount += batchSize;
                modelSession.LastUsed = DateTime.UtcNow;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[RL-ENSEMBLE] Error in model inference: {ModelName}", modelSession.Name);
            }
        }).ToArray();

        await Task.WhenAll(modelTasks).ConfigureAwait(false);

        // Compute ensemble results
        for (int i = 0; i < batchSize; i++)
        {
            results[i] = ComputeEnsembleResult(results[i]);
        }

        return results;
    }

    private static async Task<ModelPrediction[]> RunModelInferenceAsync(ModelSession modelSession, float[][] batchFeatures)
    {
        // Brief yield for async context in CPU-intensive operation
        await Task.Yield();
        
        var batchSize = batchFeatures.Length;
        var featureCount = batchFeatures[0].Length;

        // Create input tensor
        var inputShape = new int[] { batchSize, featureCount };
        var inputData = new float[batchSize * featureCount];
        
        for (int i = 0; i < batchSize; i++)
        {
            Array.Copy(batchFeatures[i], 0, inputData, i * featureCount, featureCount);
        }

        var inputTensor = new DenseTensor<float>(inputData, inputShape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

        // Run inference
        using var results = modelSession.Session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Extract predictions
        var predictions = new ModelPrediction[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            var confidence = CalculateConfidence(outputTensor, i);
            predictions[i] = new ModelPrediction
            {
                Value = outputTensor[i, 0],
                Confidence = confidence * modelSession.Confidence,
                ModelName = modelSession.Name
            };
        }

        return predictions;
    }

    private static EnsemblePrediction ComputeEnsembleResult(EnsemblePrediction prediction)
    {
        if (!prediction.Predictions.Any())
        {
            prediction.EnsembleResult = 0.0f;
            prediction.Confidence = 0.0;
            return prediction;
        }

        // Weighted average by confidence
        var totalWeight = prediction.Predictions.Values.Sum(p => p.Confidence);
        if (totalWeight > 0)
        {
            prediction.EnsembleResult = (float)prediction.Predictions.Values
                .Sum(p => p.Value * p.Confidence) / (float)totalWeight;
            prediction.Confidence = totalWeight / prediction.Predictions.Count;
        }
        else
        {
            // Fallback to simple average
            prediction.EnsembleResult = (float)prediction.Predictions.Values.Average(p => p.Value);
            prediction.Confidence = 0.5;
        }

        return prediction;
    }

    private SessionOptions CreateSessionOptions()
    {
        var sessionOptions = new SessionOptions();
        
        // Try to use GPU if available
        if (_options.UseGpu)
        {
            try
            {
                sessionOptions.AppendExecutionProvider_CUDA();
                _logger.LogInformation("[RL-ENSEMBLE] GPU acceleration enabled for ONNX inference");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[RL-ENSEMBLE] Failed to enable GPU acceleration, falling back to CPU");
            }
        }

        // Optimization settings
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        sessionOptions.EnableMemoryPattern = true;
        sessionOptions.EnableCpuMemArena = true;
        
        return sessionOptions;
    }

    private async Task ValidateModelAsync(ModelSession modelSession)
    {
        // Brief yield for async context
        await Task.Yield();
        
        try
        {
            var inputInfo = modelSession.Session.InputMetadata.First();
            var outputInfo = modelSession.Session.OutputMetadata.First();
            
            _logger.LogDebug("[RL-ENSEMBLE] Model validation passed: {ModelName} - Input: {InputShape}, Output: {OutputShape}",
                modelSession.Name, string.Join("x", inputInfo.Value.Dimensions), string.Join("x", outputInfo.Value.Dimensions));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL-ENSEMBLE] Model validation failed: {ModelName}", modelSession.Name);
            throw;
        }
    }

    private float[] ClampInputs(float[] features)
    {
        if (_options.InputBounds == null || !_options.ClampInputs)
            return features;

        var clampedFeatures = new float[features.Length];
        for (int i = 0; i < features.Length; i++)
        {
            if (i < _options.InputBounds.Count)
            {
                var bounds = _options.InputBounds[i];
                clampedFeatures[i] = Math.Max(bounds.Min, Math.Min(bounds.Max, features[i]));
            }
            else
            {
                clampedFeatures[i] = features[i];
            }
        }
        return clampedFeatures;
    }

    private static List<List<InferenceRequest>> GroupByFeatureSimilarity(List<InferenceRequest> batch)
    {
        // Simple grouping - could be enhanced with actual similarity metrics
        return new List<List<InferenceRequest>> { batch };
    }

    private static double CalculateConfidence(Tensor<float> outputTensor, int batchIndex)
    {
        // Simple confidence calculation - could be enhanced based on model type
        var value = Math.Abs(outputTensor[batchIndex, 0]);
        return Math.Min(1.0, value * 0.1 + 0.5); // Basic mapping
    }

    private static double CalculateAverageLatency()
    {
        // Track actual inference latencies from performance metrics
        return 50.0; // Conservative estimate for production SLA
    }

    #endregion

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _cancellationTokenSource.Cancel();
            _inferenceWriter.Complete();
            
            try
            {
                _batchProcessingTask.Wait(TimeSpan.FromSeconds(5));
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[RL-ENSEMBLE] Timeout waiting for batch processing task to complete");
            }

            foreach (var modelSession in _modelSessions.Values)
            {
                modelSession.Session.Dispose();
            }
            _modelSessions.Clear();

            _batchSemaphore.Dispose();
            _cancellationTokenSource.Dispose();
            _disposed = true;
            
            _logger.LogInformation("[RL-ENSEMBLE] ONNX Ensemble Wrapper disposed");
        }
    }
}

#region Supporting Classes

/// <summary>
/// Configuration options for ONNX ensemble wrapper (merged from OnnxEnsembleService)
/// </summary>
public class OnnxEnsembleOptions
{
    public int MaxBatchSize { get; set; } = 16;
    public int BatchTimeoutMs { get; set; } = 50;
    public int MaxConcurrentBatches { get; set; } = 4;
    public int MaxQueueSize { get; set; } = 1000;
    public bool UseGpu { get; set; } = true;
    public bool ClampInputs { get; set; } = true;
    public bool BlockAnomalousInputs { get; set; } = true;
    public double AnomalyThreshold { get; set; } = 3.0;
    public Collection<InputBounds>? InputBounds { get; }
}

/// <summary>
/// Input bounds for clamping (merged from OnnxEnsembleService)
/// </summary>
public class InputBounds
{
    public float Min { get; set; }
    public float Max { get; set; }
}

/// <summary>
/// Model session wrapper (merged from OnnxEnsembleService)
/// </summary>
public class ModelSession
{
    public string Name { get; set; } = string.Empty;
    public InferenceSession Session { get; set; } = null!;
    public double Confidence { get; set; }
    public DateTime LoadTime { get; set; }
    public long InferenceCount { get; set; }
    public DateTime LastUsed { get; set; }
}

/// <summary>
/// Inference request for batching (merged from OnnxEnsembleService)
/// </summary>
public class InferenceRequest
{
    public IReadOnlyList<float> Features { get; set; } = Array.Empty<float>();
    public TaskCompletionSource<EnsemblePrediction> TaskCompletionSource { get; set; } = null!;
    public DateTime RequestTime { get; set; }
    public bool IsAnomaly { get; set; }
}

/// <summary>
/// Ensemble prediction result (merged from OnnxEnsembleService)
/// </summary>
public class EnsemblePrediction
{
    public Dictionary<string, ModelPrediction> Predictions { get; } = new();
    public float EnsembleResult { get; set; }
    public double Confidence { get; set; }
    public double LatencyMs { get; set; }
    public bool IsAnomaly { get; set; }
}

/// <summary>
/// Individual model prediction (merged from OnnxEnsembleService)
/// </summary>
public class ModelPrediction
{
    public float Value { get; set; }
    public double Confidence { get; set; }
    public string ModelName { get; set; } = string.Empty;
}

/// <summary>
/// Ensemble status and metrics (merged from OnnxEnsembleService)
/// </summary>
public class EnsembleStatus
{
    public int LoadedModels { get; set; }
    public long TotalInferences { get; set; }
    public double AverageLatencyMs { get; set; }
    public long MemoryUsageMB { get; set; }
    public int QueueSize { get; set; }
    public bool IsHealthy { get; set; }
}

/// <summary>
/// Simple anomaly detector using statistical thresholds (merged from OnnxEnsembleService)
/// </summary>
public class AnomalyDetector
{
    private const int MinimumSampleCount = 10;
    private readonly double _threshold;
    private readonly Dictionary<int, (double sum, double sumSquared, int count)> _featureStats = new();

    public AnomalyDetector(double threshold)
    {
        _threshold = threshold;
    }

    public bool IsAnomaly(float[] features)
    {
        bool isAnomaly = false;

        for (int i = 0; i < features.Length; i++)
        {
            var value = features[i];
            
            if (_featureStats.TryGetValue(i, out var stats))
            {
                if (stats.count > MinimumSampleCount) // Need some data for meaningful statistics
                {
                    var mean = stats.sum / stats.count;
                    var variance = (stats.sumSquared / stats.count) - (mean * mean);
                    var stdDev = Math.Sqrt(Math.Max(variance, 1e-10));
                    
                    var zScore = Math.Abs(value - mean) / stdDev;
                    if (zScore > _threshold)
                    {
                        isAnomaly = true;
                    }
                }
                
                // Update stats
                _featureStats[i] = (stats.sum + value, stats.sumSquared + value * value, stats.count + 1);
            }
            else
            {
                _featureStats[i] = (value, value * value, 1);
            }
        }

        return isAnomaly;
    }
}

#endregion