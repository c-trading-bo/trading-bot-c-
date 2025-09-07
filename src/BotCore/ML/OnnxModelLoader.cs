using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Concurrent;

namespace BotCore.ML;

/// <summary>
/// Professional ONNX model loader with proper error handling, logging, and inference validation
/// Replaces placeholder implementations with real Microsoft.ML.OnnxRuntime integration
/// </summary>
public sealed class OnnxModelLoader : IDisposable
{
    private readonly ILogger<OnnxModelLoader> _logger;
    private readonly ConcurrentDictionary<string, InferenceSession> _loadedSessions = new();
    private readonly SessionOptions _sessionOptions;
    private bool _disposed = false;

    public OnnxModelLoader(ILogger<OnnxModelLoader> logger)
    {
        _logger = logger;
        
        // Configure ONNX Runtime session options for optimal performance
        _sessionOptions = new SessionOptions
        {
            EnableCpuMemArena = true,
            EnableMemoryPattern = true,
            EnableProfiling = false, // Disable in production
            ExecutionMode = ExecutionMode.ORT_PARALLEL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
        };

        _logger.LogInformation("[ONNX-Loader] Initialized with optimized session options");
    }

    /// <summary>
    /// Load ONNX model with comprehensive error handling and validation
    /// </summary>
    public async Task<InferenceSession?> LoadModelAsync(string modelPath, bool validateInference = true)
    {
        if (string.IsNullOrEmpty(modelPath))
        {
            _logger.LogError("[ONNX-Loader] Model path is null or empty");
            return null;
        }

        if (!File.Exists(modelPath))
        {
            _logger.LogError("[ONNX-Loader] Model file not found: {ModelPath}", modelPath);
            return null;
        }

        var modelKey = Path.GetFullPath(modelPath);

        // Check if already loaded
        if (_loadedSessions.TryGetValue(modelKey, out var existingSession))
        {
            _logger.LogDebug("[ONNX-Loader] Reusing cached model: {ModelPath}", modelPath);
            return existingSession;
        }

        try
        {
            _logger.LogInformation("[ONNX-Loader] Loading ONNX model: {ModelPath}", modelPath);
            
            var startTime = DateTime.UtcNow;
            
            // Load the ONNX model
            var session = new InferenceSession(modelPath, _sessionOptions);
            
            var loadDuration = DateTime.UtcNow - startTime;
            
            // Validate model structure
            var inputInfo = session.InputMetadata;
            var outputInfo = session.OutputMetadata;
            
            _logger.LogInformation("[ONNX-Loader] Model loaded successfully in {Duration}ms", 
                loadDuration.TotalMilliseconds);
            _logger.LogInformation("[ONNX-Loader] Inputs: {InputCount}, Outputs: {OutputCount}", 
                inputInfo.Count, outputInfo.Count);
            
            // Log input/output details
            foreach (var input in inputInfo)
            {
                _logger.LogDebug("[ONNX-Loader] Input: {Name} - {Type} - {Shape}", 
                    input.Key, input.Value.ElementType, string.Join(",", input.Value.Dimensions));
            }
            
            foreach (var output in outputInfo)
            {
                _logger.LogDebug("[ONNX-Loader] Output: {Name} - {Type} - {Shape}", 
                    output.Key, output.Value.ElementType, string.Join(",", output.Value.Dimensions));
            }

            // Validate inference capability if requested
            if (validateInference)
            {
                var validationResult = await ValidateInferenceAsync(session);
                if (!validationResult)
                {
                    _logger.LogError("[ONNX-Loader] Model failed inference validation: {ModelPath}", modelPath);
                    session.Dispose();
                    return null;
                }
            }

            // Cache the loaded session
            _loadedSessions[modelKey] = session;
            
            _logger.LogInformation("[ONNX-Loader] Model successfully loaded and validated: {ModelPath}", modelPath);
            return session;
        }
        catch (OnnxRuntimeException ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] ONNX Runtime error loading model: {ModelPath} - {Error}", 
                modelPath, ex.Message);
            return null;
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] Invalid model format: {ModelPath} - {Error}", 
                modelPath, ex.Message);
            return null;
        }
        catch (FileNotFoundException ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] Model file access error: {ModelPath} - {Error}", 
                modelPath, ex.Message);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] Unexpected error loading model: {ModelPath} - {Error}", 
                modelPath, ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Validate that the model can perform inference with dummy data
    /// </summary>
    private async Task<bool> ValidateInferenceAsync(InferenceSession session)
    {
        try
        {
            _logger.LogDebug("[ONNX-Loader] Validating model inference capability...");
            
            var inputMetadata = session.InputMetadata;
            var inputs = new List<NamedOnnxValue>();

            // Create dummy inputs based on model metadata
            foreach (var input in inputMetadata)
            {
                var inputName = input.Key;
                var inputType = input.Value.ElementType;
                var dimensions = input.Value.Dimensions;

                // Create appropriately shaped dummy data
                var dummyInput = CreateDummyInput(inputName, inputType, dimensions);
                if (dummyInput != null)
                {
                    inputs.Add(dummyInput);
                }
            }

            if (inputs.Count == 0)
            {
                _logger.LogWarning("[ONNX-Loader] No valid inputs created for validation");
                return false;
            }

            // Run inference with dummy data
            var startTime = DateTime.UtcNow;
            using var results = session.Run(inputs);
            var inferenceDuration = DateTime.UtcNow - startTime;

            _logger.LogDebug("[ONNX-Loader] Inference validation successful in {Duration}ms", 
                inferenceDuration.TotalMilliseconds);

            // Validate outputs
            var outputCount = results.Count();
            if (outputCount == 0)
            {
                _logger.LogWarning("[ONNX-Loader] Model produced no outputs during validation");
                return false;
            }

            _logger.LogDebug("[ONNX-Loader] Model produced {OutputCount} outputs", outputCount);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] Model inference validation failed: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Create dummy input data for model validation
    /// </summary>
    private NamedOnnxValue? CreateDummyInput(string inputName, System.Type elementType, int[] dimensions)
    {
        try
        {
            // Handle dynamic dimensions (replace -1 with 1)
            var safeDimensions = dimensions.Select(d => d == -1 ? 1 : d).ToArray();
            
            if (elementType == typeof(float))
            {
                var shape = safeDimensions;
                var totalElements = shape.Aggregate(1, (a, b) => a * b);
                var data = new float[totalElements];
                
                // Fill with small random values
                var random = new Random(42); // Fixed seed for reproducibility
                for (int i = 0; i < totalElements; i++)
                {
                    data[i] = (float)(random.NextDouble() * 0.1 - 0.05); // Range: -0.05 to 0.05
                }
                
                var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(data, shape);
                return NamedOnnxValue.CreateFromTensor(inputName, tensor);
            }
            else if (elementType == typeof(long))
            {
                var shape = safeDimensions;
                var totalElements = shape.Aggregate(1, (a, b) => a * b);
                var data = new long[totalElements];
                
                // Fill with small integer values
                for (int i = 0; i < totalElements; i++)
                {
                    data[i] = i % 10; // Values 0-9
                }
                
                var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(data, shape);
                return NamedOnnxValue.CreateFromTensor(inputName, tensor);
            }
            else if (elementType == typeof(int))
            {
                var shape = safeDimensions;
                var totalElements = shape.Aggregate(1, (a, b) => a * b);
                var data = new int[totalElements];
                
                for (int i = 0; i < totalElements; i++)
                {
                    data[i] = i % 10;
                }
                
                var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<int>(data, shape);
                return NamedOnnxValue.CreateFromTensor(inputName, tensor);
            }
            else
            {
                _logger.LogWarning("[ONNX-Loader] Unsupported input type for validation: {Type}", elementType);
                return null;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] Error creating dummy input for {InputName}", inputName);
            return null;
        }
    }

    /// <summary>
    /// Get a loaded model session by path
    /// </summary>
    public InferenceSession? GetLoadedModel(string modelPath)
    {
        var modelKey = Path.GetFullPath(modelPath);
        return _loadedSessions.TryGetValue(modelKey, out var session) ? session : null;
    }

    /// <summary>
    /// Unload a specific model to free memory
    /// </summary>
    public bool UnloadModel(string modelPath)
    {
        var modelKey = Path.GetFullPath(modelPath);
        
        if (_loadedSessions.TryRemove(modelKey, out var session))
        {
            session.Dispose();
            _logger.LogInformation("[ONNX-Loader] Model unloaded: {ModelPath}", modelPath);
            return true;
        }
        
        return false;
    }

    /// <summary>
    /// Get count of currently loaded models
    /// </summary>
    public int LoadedModelCount => _loadedSessions.Count;

    /// <summary>
    /// Get list of loaded model paths
    /// </summary>
    public IEnumerable<string> LoadedModelPaths => _loadedSessions.Keys;

    public void Dispose()
    {
        if (!_disposed)
        {
            _logger.LogInformation("[ONNX-Loader] Disposing {ModelCount} loaded models", _loadedSessions.Count);
            
            foreach (var session in _loadedSessions.Values)
            {
                try
                {
                    session.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[ONNX-Loader] Error disposing model session");
                }
            }
            
            _loadedSessions.Clear();
            _sessionOptions.Dispose();
            
            _disposed = true;
            _logger.LogInformation("[ONNX-Loader] Disposed successfully");
        }
    }
}
