using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.ML;

/// <summary>
/// ONNX model loader and inference for multi-input multi-timeframe models.
/// Handles models with separate 5m and 1m input branches.
/// </summary>
public class MultiInputOnnxModelLoader
{
    private readonly ILogger<MultiInputOnnxModelLoader> _logger;
    private InferenceSession? _session;
    private string? _modelPath;
    
    public MultiInputOnnxModelLoader(ILogger<MultiInputOnnxModelLoader> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }
    
    /// <summary>
    /// Load multi-input ONNX model from file.
    /// </summary>
    public void LoadModel(string modelPath)
    {
        if (string.IsNullOrEmpty(modelPath))
        {
            throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));
        }
        
        if (!System.IO.File.Exists(modelPath))
        {
            throw new System.IO.FileNotFoundException($"Model file not found: {modelPath}");
        }
        
        _logger.LogInformation("[MULTI_INPUT_ONNX] Loading multi-input model from: {Path}", modelPath);
        
        var sessionOptions = new SessionOptions();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        
        _session = new InferenceSession(modelPath, sessionOptions);
        _modelPath = modelPath;
        
        // Log input/output metadata
        var inputMetadata = _session.InputMetadata;
        var outputMetadata = _session.OutputMetadata;
        
        _logger.LogInformation("[MULTI_INPUT_ONNX] Model loaded - Inputs: {InputCount}, Outputs: {OutputCount}",
            inputMetadata.Count, outputMetadata.Count);
        
        foreach (var input in inputMetadata)
        {
            _logger.LogDebug("[MULTI_INPUT_ONNX] Input: {Name}, Shape: {Shape}",
                input.Key, string.Join(",", input.Value.Dimensions));
        }
    }
    
    /// <summary>
    /// Run inference with multi-timeframe inputs.
    /// </summary>
    public float[] RunInference(double[] features5m, double[] features1m)
    {
        if (_session == null)
        {
            throw new InvalidOperationException("Model not loaded. Call LoadModel first.");
        }
        
        if (features5m == null || features1m == null)
        {
            throw new ArgumentNullException("Features cannot be null");
        }
        
        // Create input tensors
        var input5m = CreateTensor(features5m, "input_5m");
        var input1m = CreateTensor(features1m, "input_1m");
        
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_5m", input5m),
            NamedOnnxValue.CreateFromTensor("input_1m", input1m)
        };
        
        // Run inference
        using var results = _session.Run(inputs);
        var output = results.FirstOrDefault()?.AsEnumerable<float>().ToArray();
        
        if (output == null)
        {
            throw new InvalidOperationException("Model inference failed - no output");
        }
        
        return output;
    }
    
    /// <summary>
    /// Run batch inference with multiple samples.
    /// </summary>
    public List<float[]> RunBatchInference(double[,] features5mBatch, double[,] features1mBatch)
    {
        if (_session == null)
        {
            throw new InvalidOperationException("Model not loaded. Call LoadModel first.");
        }
        
        if (features5mBatch == null || features1mBatch == null)
        {
            throw new ArgumentNullException("Feature batches cannot be null");
        }
        
        int batchSize = features5mBatch.GetLength(0);
        if (batchSize != features1mBatch.GetLength(0))
        {
            throw new ArgumentException("Batch sizes must match");
        }
        
        var results = new List<float[]>();
        
        // Process each sample in batch
        for (int i = 0; i < batchSize; i++)
        {
            var features5m = ExtractRow(features5mBatch, i);
            var features1m = ExtractRow(features1mBatch, i);
            var output = RunInference(features5m, features1m);
            results.Add(output);
        }
        
        return results;
    }
    
    /// <summary>
    /// Get model input specifications.
    /// </summary>
    public (int features5mCount, int features1mCount) GetInputSpecs()
    {
        if (_session == null)
        {
            throw new InvalidOperationException("Model not loaded");
        }
        
        var inputMetadata = _session.InputMetadata;
        
        var input5mDims = inputMetadata.ContainsKey("input_5m") 
            ? inputMetadata["input_5m"].Dimensions 
            : new int[] { 1, 7 };
            
        var input1mDims = inputMetadata.ContainsKey("input_1m")
            ? inputMetadata["input_1m"].Dimensions
            : new int[] { 1, 7 };
        
        return (input5mDims[^1], input1mDims[^1]);
    }
    
    public void Dispose()
    {
        _session?.Dispose();
        _session = null;
    }
    
    private Tensor<float> CreateTensor(double[] features, string name)
    {
        var floatFeatures = features.Select(f => (float)f).ToArray();
        var dimensions = new[] { 1, floatFeatures.Length };
        var tensor = new DenseTensor<float>(dimensions);
        
        for (int i = 0; i < floatFeatures.Length; i++)
        {
            tensor[0, i] = floatFeatures[i];
        }
        
        return tensor;
    }
    
    private double[] ExtractRow(double[,] matrix, int row)
    {
        int cols = matrix.GetLength(1);
        var result = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            result[j] = matrix[row, j];
        }
        return result;
    }
}
