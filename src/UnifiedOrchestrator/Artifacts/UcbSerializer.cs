using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TrainingMetadata = TradingBot.UnifiedOrchestrator.Interfaces.TrainingMetadata;

namespace TradingBot.UnifiedOrchestrator.Artifacts;

/// <summary>
/// Production-ready UCB (Upper Confidence Bound) model serializer
/// Handles serialization, validation, and metadata extraction for UCB models
/// </summary>
internal class UcbSerializer : IArtifactBuilder
{
    private readonly ILogger<UcbSerializer> _logger;
    private const string ModelFileExtension = ".ucb.json";
    private const string SchemaVersion = "1.0";

    public string SupportedModelType => "UCB";

    public UcbSerializer(ILogger<UcbSerializer> logger)
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
            _logger.LogInformation("Building UCB artifact from {ModelPath} to {OutputPath}", modelPath, outputPath);

            // Ensure source model exists
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Source UCB model file not found: {modelPath}", modelPath);
            }

            // Ensure output directory exists
            var outputDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            // Load source UCB model
            var sourceModel = await LoadUcbModelAsync(modelPath, cancellationToken).ConfigureAwait(false);

            // Ensure output path has correct extension
            if (!outputPath.EndsWith(ModelFileExtension, StringComparison.OrdinalIgnoreCase))
            {
                outputPath = Path.ChangeExtension(outputPath, ModelFileExtension);
            }

            // Create enhanced artifact with metadata
            var artifact = new UcbArtifact
            {
                SchemaVersion = SchemaVersion,
                CreatedAt = DateTime.UtcNow,
                Model = sourceModel,
                TrainingMetadata = new UcbTrainingMetadata
                {
                    GitSha = metadata.GitSha,
                    CreatedBy = metadata.CreatedBy,
                    TrainingStartTime = metadata.TrainingStartTime,
                    TrainingEndTime = metadata.TrainingEndTime,
                    DataRangeStart = ParseDateTime(metadata.DataRangeStart),
                    DataRangeEnd = ParseDateTime(metadata.DataRangeEnd),
                    DataSamples = metadata.DataSamples,
                    Parameters = metadata.Parameters ?? new Dictionary<string, object>(),
                    PerformanceMetrics = metadata.PerformanceMetrics ?? new Dictionary<string, decimal>()
                }
            };

            // Serialize artifact to JSON
            var json = JsonSerializer.Serialize(artifact, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            await File.WriteAllTextAsync(outputPath, json, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("Successfully built UCB artifact at {OutputPath}", outputPath);

            return outputPath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to build UCB artifact from {ModelPath}", modelPath);
            throw;
        }
    }

    public async Task<bool> ValidateArtifactAsync(string artifactPath, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(artifactPath))
        {
            throw new ArgumentException("Artifact path cannot be null or empty", nameof(artifactPath));
        }

        try
        {
            _logger.LogDebug("Validating UCB artifact at {ArtifactPath}", artifactPath);

            // Check if file exists
            if (!File.Exists(artifactPath))
            {
                _logger.LogError("UCB artifact file not found: {ArtifactPath}", artifactPath);
                return false;
            }

            // Check file size (must be > 0)
            var fileInfo = new FileInfo(artifactPath);
            if (fileInfo.Length == 0)
            {
                _logger.LogError("UCB artifact file is empty: {ArtifactPath}", artifactPath);
                return false;
            }

            // Try to deserialize and validate structure
            try
            {
                var artifact = await LoadArtifactAsync(artifactPath, cancellationToken).ConfigureAwait(false);

                // Validate required fields
                if (string.IsNullOrEmpty(artifact.SchemaVersion))
                {
                    _logger.LogError("UCB artifact missing schema version: {ArtifactPath}", artifactPath);
                    return false;
                }

                if (artifact.Model == null)
                {
                    _logger.LogError("UCB artifact missing model data: {ArtifactPath}", artifactPath);
                    return false;
                }

                // Validate UCB model structure
                if (artifact.Model.Arms == null || artifact.Model.Arms.Count == 0)
                {
                    _logger.LogError("UCB model has no arms: {ArtifactPath}", artifactPath);
                    return false;
                }

                // Validate each arm has required fields
                foreach (var arm in artifact.Model.Arms)
                {
                    if (string.IsNullOrEmpty(arm.Name))
                    {
                        _logger.LogError("UCB model has arm with no name: {ArtifactPath}", artifactPath);
                        return false;
                    }

                    if (arm.TotalReward < 0)
                    {
                        _logger.LogError("UCB model arm '{ArmName}' has negative total reward: {ArtifactPath}", arm.Name, artifactPath);
                        return false;
                    }

                    if (arm.PullCount < 0)
                    {
                        _logger.LogError("UCB model arm '{ArmName}' has negative pull count: {ArtifactPath}", arm.Name, artifactPath);
                        return false;
                    }
                }

                _logger.LogInformation(
                    "UCB artifact validated successfully: {ArtifactPath} ({FileSize} bytes, {ArmCount} arms, {TotalPulls} total pulls)",
                    artifactPath,
                    fileInfo.Length,
                    artifact.Model.Arms.Count,
                    artifact.Model.TotalPulls);

                return true;
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "UCB artifact JSON is malformed: {ArtifactPath}", artifactPath);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating UCB artifact: {ArtifactPath}", artifactPath);
            return false;
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
            _logger.LogDebug("Extracting metadata from UCB artifact: {ArtifactPath}", artifactPath);

            var fileInfo = new FileInfo(artifactPath);
            if (!fileInfo.Exists)
            {
                throw new FileNotFoundException($"UCB artifact not found: {artifactPath}", artifactPath);
            }

            // Load artifact
            var artifact = await LoadArtifactAsync(artifactPath, cancellationToken).ConfigureAwait(false);

            // Compute file hash
            var hash = await ComputeFileHashAsync(artifactPath, cancellationToken).ConfigureAwait(false);

            // Extract model information
            var armCount = artifact.Model?.Arms?.Count ?? 0;
            var totalPulls = artifact.Model?.TotalPulls ?? 0;
            var explorationRate = artifact.Model?.ExplorationRate ?? 0.0m;

            var inputShape = $"arms:{armCount}";
            var outputShape = $"best_arm:1";

            var properties = new Dictionary<string, object>
            {
                ["arm_count"] = armCount,
                ["total_pulls"] = totalPulls,
                ["exploration_rate"] = explorationRate,
                ["schema_version"] = artifact.SchemaVersion ?? "unknown"
            };

            if (artifact.Model?.Arms != null)
            {
                properties["arm_names"] = artifact.Model.Arms.Select(a => a.Name).ToArray();
                properties["best_arm"] = artifact.Model.Arms.OrderByDescending(a => a.AverageReward).FirstOrDefault()?.Name ?? "none";
            }

            var metadata = new ArtifactMetadata
            {
                ModelType = "UCB",
                Version = artifact.SchemaVersion ?? "unknown",
                FileSizeBytes = fileInfo.Length,
                CreatedAt = artifact.CreatedAt,
                Hash = hash,
                InputShape = inputShape,
                OutputShape = outputShape,
                Properties = properties
            };

            _logger.LogInformation("Successfully extracted metadata from UCB artifact: {ArtifactPath}", artifactPath);

            return metadata;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to extract metadata from UCB artifact: {ArtifactPath}", artifactPath);
            throw;
        }
    }

    private static async Task<UcbModel> LoadUcbModelAsync(string modelPath, CancellationToken cancellationToken)
    {
        var json = await File.ReadAllTextAsync(modelPath, cancellationToken).ConfigureAwait(false);
        var model = JsonSerializer.Deserialize<UcbModel>(json, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });

        return model ?? throw new InvalidOperationException($"Failed to deserialize UCB model from {modelPath}");
    }

    private static async Task<UcbArtifact> LoadArtifactAsync(string artifactPath, CancellationToken cancellationToken)
    {
        var json = await File.ReadAllTextAsync(artifactPath, cancellationToken).ConfigureAwait(false);
        var artifact = JsonSerializer.Deserialize<UcbArtifact>(json, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });

        return artifact ?? throw new InvalidOperationException($"Failed to deserialize UCB artifact from {artifactPath}");
    }

    private static async Task<string> ComputeFileHashAsync(string filePath, CancellationToken cancellationToken)
    {
        using var sha256 = SHA256.Create();
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 8192, useAsync: true);
        var hashBytes = await Task.Run(() => sha256.ComputeHash(stream), cancellationToken).ConfigureAwait(false);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }

    private static DateTime ParseDateTime(string dateTimeString)
    {
        if (string.IsNullOrWhiteSpace(dateTimeString))
        {
            return DateTime.MinValue;
        }

        if (DateTime.TryParse(dateTimeString, out var result))
        {
            return result;
        }

        return DateTime.MinValue;
    }

    #region Model Classes

    private class UcbArtifact
    {
        public string SchemaVersion { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        public UcbModel Model { get; set; } = new();
        public UcbTrainingMetadata TrainingMetadata { get; set; } = new();
    }

    private class UcbModel
    {
        public List<UcbArm> Arms { get; set; } = new();
        public int TotalPulls { get; set; }
        public decimal ExplorationRate { get; set; }
        public decimal ConfidenceLevel { get; set; } = 1.96m; // Default 95% confidence
    }

    private class UcbArm
    {
        public string Name { get; set; } = string.Empty;
        public decimal TotalReward { get; set; }
        public int PullCount { get; set; }
        public decimal AverageReward => PullCount > 0 ? TotalReward / PullCount : 0;
    }

    private class UcbTrainingMetadata
    {
        public string GitSha { get; set; } = string.Empty;
        public string CreatedBy { get; set; } = string.Empty;
        public DateTime TrainingStartTime { get; set; }
        public DateTime TrainingEndTime { get; set; }
        public DateTime DataRangeStart { get; set; }
        public DateTime DataRangeEnd { get; set; }
        public long DataSamples { get; set; }
        public Dictionary<string, object> Parameters { get; set; } = new();
        public Dictionary<string, decimal> PerformanceMetrics { get; set; } = new();
    }

    #endregion
}
