using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.ML;

/// <summary>
/// Multi-branch model architecture for handling multi-timeframe features.
/// Supports separate processing branches for 5m and 1m features with fusion layer.
/// </summary>
public class MultiBranchModelArchitecture
{
    private readonly ILogger<MultiBranchModelArchitecture> _logger;
    private readonly int _features5mCount;
    private readonly int _features1mCount;
    private readonly int _hiddenSize;
    private readonly int _fusionSize;
    
    public MultiBranchModelArchitecture(
        ILogger<MultiBranchModelArchitecture> logger,
        int features5mCount = 7,
        int features1mCount = 7,
        int hiddenSize = 64,
        int fusionSize = 128)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _features5mCount = features5mCount;
        _features1mCount = features1mCount;
        _hiddenSize = hiddenSize;
        _fusionSize = fusionSize;
        
        _logger.LogInformation(
            "[MULTI_BRANCH] Architecture initialized - 5m features: {F5m}, 1m features: {F1m}, hidden: {Hidden}, fusion: {Fusion}",
            _features5mCount, _features1mCount, _hiddenSize, _fusionSize);
    }
    
    /// <summary>
    /// Process 5-minute branch features through dedicated neural pathway.
    /// </summary>
    public double[] ProcessBranch5m(double[] features5m)
    {
        if (features5m == null || features5m.Length != _features5mCount)
        {
            throw new ArgumentException($"Expected {_features5mCount} 5m features, got {features5m?.Length ?? 0}");
        }
        
        // Process through 5m-specific branch
        var hidden = ApplyDenseLayer(features5m, _hiddenSize);
        var activated = ApplyReLU(hidden);
        
        return activated;
    }
    
    /// <summary>
    /// Process 1-minute branch features through dedicated neural pathway.
    /// </summary>
    public double[] ProcessBranch1m(double[] features1m)
    {
        if (features1m == null || features1m.Length != _features1mCount)
        {
            throw new ArgumentException($"Expected {_features1mCount} 1m features, got {features1m?.Length ?? 0}");
        }
        
        // Process through 1m-specific branch
        var hidden = ApplyDenseLayer(features1m, _hiddenSize);
        var activated = ApplyReLU(hidden);
        
        return activated;
    }
    
    /// <summary>
    /// Fuse outputs from both branches into unified representation.
    /// </summary>
    public double[] FuseBranches(double[] branch5mOutput, double[] branch1mOutput)
    {
        if (branch5mOutput == null || branch1mOutput == null)
        {
            throw new ArgumentNullException("Branch outputs cannot be null");
        }
        
        // Concatenate both branch outputs
        var concatenated = branch5mOutput.Concat(branch1mOutput).ToArray();
        
        // Process through fusion layer
        var fused = ApplyDenseLayer(concatenated, _fusionSize);
        var activated = ApplyReLU(fused);
        
        return activated;
    }
    
    /// <summary>
    /// Complete forward pass through multi-branch architecture.
    /// </summary>
    public double[] Forward(double[] features5m, double[] features1m)
    {
        var branch5m = ProcessBranch5m(features5m);
        var branch1m = ProcessBranch1m(features1m);
        var fused = FuseBranches(branch5m, branch1m);
        
        return fused;
    }
    
    /// <summary>
    /// Get model architecture specification for ONNX export.
    /// </summary>
    public ModelArchitectureSpec GetArchitectureSpec()
    {
        return new ModelArchitectureSpec
        {
            Input5mShape = new[] { _features5mCount },
            Input1mShape = new[] { _features1mCount },
            Branch5mHiddenSize = _hiddenSize,
            Branch1mHiddenSize = _hiddenSize,
            FusionSize = _fusionSize,
            OutputSize = _fusionSize
        };
    }
    
    private double[] ApplyDenseLayer(double[] input, int outputSize)
    {
        // Simplified dense layer - in production this would use actual weights
        var output = new double[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < input.Length; j++)
            {
                // Simple initialization - production would load trained weights
                sum += input[j] * (0.1 * ((i + j) % 3 - 1));
            }
            output[i] = sum;
        }
        return output;
    }
    
    private double[] ApplyReLU(double[] input)
    {
        return input.Select(x => Math.Max(0, x)).ToArray();
    }
}

/// <summary>
/// Specification for multi-branch model architecture.
/// </summary>
public class ModelArchitectureSpec
{
    public int[] Input5mShape { get; set; } = Array.Empty<int>();
    public int[] Input1mShape { get; set; } = Array.Empty<int>();
    public int Branch5mHiddenSize { get; set; }
    public int Branch1mHiddenSize { get; set; }
    public int FusionSize { get; set; }
    public int OutputSize { get; set; }
}
