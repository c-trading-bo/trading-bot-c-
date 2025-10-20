using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using TrainingMetadata = TradingBot.UnifiedOrchestrator.Interfaces.TrainingMetadata;

namespace TradingBot.UnifiedOrchestrator.Artifacts;

/// <summary>
/// Production-ready ONNX artifact builder for ML models
/// Handles building, validating, and extracting metadata from ONNX model files
/// </summary>
internal class OnnxArtifactBuilder : IArtifactBuilder
{
    private readonly ILogger<OnnxArtifactBuilder> _logger;

    public string SupportedModelType => "ONNX";

    public OnnxArtifactBuilder(ILogger<OnnxArtifactBuilder> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<string> BuildArtifactAsync(
        string modelPath,
        string outputPath,
        TrainingMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));
        }

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Output path cannot be null or empty", nameof(outputPath));
        }

        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        try
        {
            _logger.LogInformation("Building ONNX artifact from {ModelPath} to {OutputPath}", modelPath, outputPath);

            // Ensure source model exists
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Source model file not found: {modelPath}", modelPath);
            }

            // Ensure output directory exists
            var outputDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            // Copy ONNX model to output location
            await CopyFileAsync(modelPath, outputPath, cancellationToken).ConfigureAwait(false);

            // Create metadata sidecar file
            var metadataPath = outputPath + ".metadata.json";
            await WriteMetadataAsync(metadataPath, metadata, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("Successfully built ONNX artifact at {OutputPath}", outputPath);

            return outputPath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to build ONNX artifact from {ModelPath}", modelPath);
            throw;
        }
    }

    public Task<bool> ValidateArtifactAsync(string artifactPath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(artifactPath))
        {
            throw new ArgumentException("Artifact path cannot be null or empty", nameof(artifactPath));
        }

        try
        {
            _logger.LogDebug("Validating ONNX artifact at {ArtifactPath}", artifactPath);

            // Check if file exists
            if (!File.Exists(artifactPath))
            {
                _logger.LogError("ONNX artifact file not found: {ArtifactPath}", artifactPath);
                return Task.FromResult(false);
            }

            // Check file size (must be > 0)
            var fileInfo = new FileInfo(artifactPath);
            if (fileInfo.Length == 0)
            {
                _logger.LogError("ONNX artifact file is empty: {ArtifactPath}", artifactPath);
                return Task.FromResult(false);
            }

            // Try to load the ONNX model to validate it's well-formed
            try
            {
                using var session = new InferenceSession(artifactPath);
                var inputMetadata = session.InputMetadata;
                var outputMetadata = session.OutputMetadata;

                // Verify model has at least one input and one output
                if (!inputMetadata.Any())
                {
                    _logger.LogError("ONNX model has no inputs: {ArtifactPath}", artifactPath);
                    return Task.FromResult(false);
                }

                if (!outputMetadata.Any())
                {
                    _logger.LogError("ONNX model has no outputs: {ArtifactPath}", artifactPath);
                    return Task.FromResult(false);
                }

                _logger.LogInformation(
                    "ONNX artifact validated successfully: {ArtifactPath} ({FileSize} bytes, {InputCount} inputs, {OutputCount} outputs)",
                    artifactPath,
                    fileInfo.Length,
                    inputMetadata.Count,
                    outputMetadata.Count);

                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "ONNX model validation failed: {ArtifactPath}", artifactPath);
                return Task.FromResult(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating ONNX artifact: {ArtifactPath}", artifactPath);
            return Task.FromResult(false);
        }
    }

    public async Task<ArtifactMetadata> GetArtifactMetadataAsync(
        string artifactPath,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(artifactPath))
        {
            throw new ArgumentException("Artifact path cannot be null or empty", nameof(artifactPath));
        }

        try
        {
            _logger.LogDebug("Extracting metadata from ONNX artifact: {ArtifactPath}", artifactPath);

            var fileInfo = new FileInfo(artifactPath);
            if (!fileInfo.Exists)
            {
                throw new FileNotFoundException($"ONNX artifact not found: {artifactPath}", artifactPath);
            }

            // Compute file hash
            var hash = await ComputeFileHashAsync(artifactPath, cancellationToken).ConfigureAwait(false);

            // Extract model schema information
            string inputShape = string.Empty;
            string outputShape = string.Empty;
            var properties = new Dictionary<string, object>();

            try
            {
                using var session = new InferenceSession(artifactPath);
                
                // Get input shapes
                var inputs = session.InputMetadata.Select(kvp => 
                    $"{kvp.Key}:{FormatTensorShape(kvp.Value.Dimensions)}").ToList();
                inputShape = string.Join(", ", inputs);

                // Get output shapes
                var outputs = session.OutputMetadata.Select(kvp => 
                    $"{kvp.Key}:{FormatTensorShape(kvp.Value.Dimensions)}").ToList();
                outputShape = string.Join(", ", outputs);

                // Store additional properties
                properties["input_count"] = session.InputMetadata.Count;
                properties["output_count"] = session.OutputMetadata.Count;
                properties["input_names"] = session.InputMetadata.Keys.ToArray();
                properties["output_names"] = session.OutputMetadata.Keys.ToArray();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not extract schema from ONNX model: {ArtifactPath}", artifactPath);
                inputShape = "unknown";
                outputShape = "unknown";
            }

            var metadata = new ArtifactMetadata
            {
                ModelType = "ONNX",
                Version = "1.0",
                FileSizeBytes = fileInfo.Length,
                CreatedAt = fileInfo.CreationTimeUtc,
                Hash = hash,
                InputShape = inputShape,
                OutputShape = outputShape,
                Properties = properties
            };

            _logger.LogInformation("Successfully extracted metadata from ONNX artifact: {ArtifactPath}", artifactPath);

            return metadata;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to extract metadata from ONNX artifact: {ArtifactPath}", artifactPath);
            throw;
        }
    }

    private static async Task CopyFileAsync(string sourcePath, string destPath, CancellationToken cancellationToken)
    {
        const int bufferSize = 81920; // 80KB buffer
        using var sourceStream = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize, useAsync: true);
        using var destStream = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize, useAsync: true);
        await sourceStream.CopyToAsync(destStream, bufferSize, cancellationToken).ConfigureAwait(false);
    }

    private static async Task WriteMetadataAsync(
        string metadataPath,
        TrainingMetadata metadata,
        CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(metadata, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(metadataPath, json, cancellationToken).ConfigureAwait(false);
    }

    private static async Task<string> ComputeFileHashAsync(string filePath, CancellationToken cancellationToken)
    {
        using var sha256 = SHA256.Create();
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 8192, useAsync: true);
        var hashBytes = await Task.Run(() => sha256.ComputeHash(stream), cancellationToken).ConfigureAwait(false);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }

    private static string FormatTensorShape(int[] dimensions)
    {
        if (dimensions == null || dimensions.Length == 0)
        {
            return "scalar";
        }

        return "[" + string.Join(",", dimensions.Select(d => d < 0 ? "?" : d.ToString())) + "]";
    }
}
