using System;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;

namespace TradingBot.UnifiedOrchestrator.Artifacts;

/// <summary>
/// UCB (Upper Confidence Bound) bandit model serializer
/// Handles JSON-based UCB model artifacts with validation
/// Production-ready implementation with full error handling
/// </summary>
internal class UcbSerializer : IArtifactBuilder
{
    private readonly ILogger<UcbSerializer> _logger;

    public string SupportedModelType => "UCB";

    public UcbSerializer(ILogger<UcbSerializer> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Build UCB artifact from trained model
    /// Copies and validates the UCB model JSON file
    /// </summary>
    public async Task<string> BuildArtifactAsync(string modelPath, string outputPath, TradingBot.UnifiedOrchestrator.Interfaces.TrainingMetadata metadata, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelPath);
        ArgumentNullException.ThrowIfNull(outputPath);
        ArgumentNullException.ThrowIfNull(metadata);

        _logger.LogInformation("[UCB-ARTIFACT] Building UCB artifact from {ModelPath} to {OutputPath}", modelPath, outputPath);

        // Validate source model exists
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Source UCB model not found: {modelPath}");
        }

        // Validate it's a valid JSON file
        try
        {
            var jsonContent = await File.ReadAllTextAsync(modelPath, cancellationToken).ConfigureAwait(false);
            using var jsonDoc = JsonDocument.Parse(jsonContent);
            
            // Basic validation - ensure it has expected structure
            if (!jsonDoc.RootElement.TryGetProperty("model_type", out var modelType) ||
                modelType.GetString() != "UCB")
            {
                _logger.LogWarning("[UCB-ARTIFACT] Model file does not have UCB model_type, proceeding anyway");
            }
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Invalid JSON in UCB model file: {modelPath}", ex);
        }

        // Ensure output directory exists
        var outputDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDir))
        {
            Directory.CreateDirectory(outputDir);
        }

        // Copy UCB model to artifact location
        await CopyFileAsync(modelPath, outputPath, cancellationToken).ConfigureAwait(false);

        _logger.LogInformation("[UCB-ARTIFACT] UCB artifact built successfully: {OutputPath}", outputPath);

        return outputPath;
    }

    /// <summary>
    /// Validate UCB artifact file
    /// Checks file exists, has valid size, and valid JSON structure
    /// </summary>
    public async Task<bool> ValidateArtifactAsync(string artifactPath, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(artifactPath);

        try
        {
            // Check file exists
            if (!File.Exists(artifactPath))
            {
                _logger.LogWarning("[UCB-ARTIFACT] Validation failed - artifact not found: {ArtifactPath}", artifactPath);
                return false;
            }

            // Check file size is reasonable (> 0 bytes, < 100MB)
            var fileInfo = new FileInfo(artifactPath);
            if (fileInfo.Length == 0)
            {
                _logger.LogWarning("[UCB-ARTIFACT] Validation failed - artifact is empty: {ArtifactPath}", artifactPath);
                return false;
            }

            if (fileInfo.Length > 100L * 1024 * 1024) // 100MB limit for JSON
            {
                _logger.LogWarning("[UCB-ARTIFACT] Validation failed - artifact too large ({Size} bytes): {ArtifactPath}", 
                    fileInfo.Length, artifactPath);
                return false;
            }

            // Validate JSON structure
            var jsonContent = await File.ReadAllTextAsync(artifactPath, cancellationToken).ConfigureAwait(false);
            using var jsonDoc = JsonDocument.Parse(jsonContent);
            
            // Check for basic UCB model structure
            if (!jsonDoc.RootElement.TryGetProperty("arms", out _) &&
                !jsonDoc.RootElement.TryGetProperty("strategies", out _))
            {
                _logger.LogWarning("[UCB-ARTIFACT] Validation warning - no 'arms' or 'strategies' property found: {ArtifactPath}", 
                    artifactPath);
            }

            _logger.LogInformation("[UCB-ARTIFACT] Validation successful: {ArtifactPath} ({Size} bytes)", 
                artifactPath, fileInfo.Length);

            return true;
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "[UCB-ARTIFACT] Validation failed - invalid JSON: {ArtifactPath}", artifactPath);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[UCB-ARTIFACT] Validation error for {ArtifactPath}", artifactPath);
            return false;
        }
    }

    /// <summary>
    /// Get UCB artifact metadata
    /// Extracts file information and computes hash
    /// </summary>
    public async Task<ArtifactMetadata> GetArtifactMetadataAsync(string artifactPath, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(artifactPath);

        var fileInfo = new FileInfo(artifactPath);
        if (!fileInfo.Exists)
        {
            throw new FileNotFoundException($"Artifact not found: {artifactPath}");
        }

        var metadata = new ArtifactMetadata
        {
            ModelType = SupportedModelType,
            Version = "1.0",
            FileSizeBytes = fileInfo.Length,
            CreatedAt = fileInfo.CreationTimeUtc,
            Hash = await ComputeFileHashAsync(artifactPath, cancellationToken).ConfigureAwait(false),
            InputShape = "Dynamic",
            OutputShape = "Dynamic",
            Properties = new()
            {
                ["FileName"] = fileInfo.Name,
                ["Extension"] = fileInfo.Extension,
                ["LastModified"] = fileInfo.LastWriteTimeUtc,
                ["Format"] = "JSON"
            }
        };

        // Try to extract additional metadata from JSON
        try
        {
            var jsonContent = await File.ReadAllTextAsync(artifactPath, cancellationToken).ConfigureAwait(false);
            using var jsonDoc = JsonDocument.Parse(jsonContent);
            
            if (jsonDoc.RootElement.TryGetProperty("arms", out var arms))
            {
                metadata.Properties["ArmCount"] = arms.GetArrayLength();
            }
            
            if (jsonDoc.RootElement.TryGetProperty("exploration_rate", out var explorationRate))
            {
                metadata.Properties["ExplorationRate"] = explorationRate.GetDouble();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[UCB-ARTIFACT] Could not extract additional metadata from JSON: {ArtifactPath}", artifactPath);
        }

        return metadata;
    }

    /// <summary>
    /// Copy file asynchronously with cancellation support
    /// </summary>
    private static async Task CopyFileAsync(string sourcePath, string destinationPath, CancellationToken cancellationToken)
    {
        const int bufferSize = 81920; // 80KB buffer

        using var sourceStream = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize, useAsync: true);
        using var destinationStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize, useAsync: true);

        await sourceStream.CopyToAsync(destinationStream, bufferSize, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Compute SHA256 hash of file for integrity verification
    /// </summary>
    private static async Task<string> ComputeFileHashAsync(string filePath, CancellationToken cancellationToken)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);
        var hashBytes = await sha256.ComputeHashAsync(stream, cancellationToken).ConfigureAwait(false);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
