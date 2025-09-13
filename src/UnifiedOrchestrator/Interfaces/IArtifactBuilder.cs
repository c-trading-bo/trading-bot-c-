using System;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for building model artifacts from trained models
/// </summary>
public interface IArtifactBuilder
{
    /// <summary>
    /// Build an artifact from a trained model
    /// </summary>
    Task<string> BuildArtifactAsync(string modelPath, string outputPath, TrainingMetadata metadata, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validate an artifact file
    /// </summary>
    Task<bool> ValidateArtifactAsync(string artifactPath, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get artifact metadata
    /// </summary>
    Task<ArtifactMetadata> GetArtifactMetadataAsync(string artifactPath, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Supported model type for this builder
    /// </summary>
    string SupportedModelType { get; }
}

/// <summary>
/// Artifact metadata
/// </summary>
public class ArtifactMetadata
{
    public string ModelType { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public long FileSizeBytes { get; set; }
    public DateTime CreatedAt { get; set; }
    public string Hash { get; set; } = string.Empty;
    public string InputShape { get; set; } = string.Empty;
    public string OutputShape { get; set; } = string.Empty;
    public Dictionary<string, object> Properties { get; set; } = new();
}