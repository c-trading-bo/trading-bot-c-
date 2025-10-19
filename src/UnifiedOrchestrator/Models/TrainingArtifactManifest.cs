using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Artifact manifest for training runs - tracks models, checksums, and metadata
/// Generated after each training session for reproducibility and validation
/// </summary>
public sealed class TrainingArtifactManifest
{
    /// <summary>
    /// Unique identifier for this training run
    /// </summary>
    [JsonPropertyName("runId")]
    public string RunId { get; set; } = string.Empty;

    /// <summary>
    /// Training session start timestamp (UTC)
    /// </summary>
    [JsonPropertyName("startTimestamp")]
    public DateTime StartTimestamp { get; set; }

    /// <summary>
    /// Training session completion timestamp (UTC)
    /// </summary>
    [JsonPropertyName("completionTimestamp")]
    public DateTime CompletionTimestamp { get; set; }

    /// <summary>
    /// Total training duration in minutes
    /// </summary>
    [JsonPropertyName("durationMinutes")]
    public double DurationMinutes { get; set; }

    /// <summary>
    /// Git commit hash for reproducibility
    /// </summary>
    [JsonPropertyName("gitCommitHash")]
    public string? GitCommitHash { get; set; }

    /// <summary>
    /// Data date range used for training
    /// </summary>
    [JsonPropertyName("dataDateRange")]
    public DateRange DataDateRange { get; set; } = new();

    /// <summary>
    /// Training parameters used
    /// </summary>
    [JsonPropertyName("trainingParameters")]
    public Dictionary<string, object> TrainingParameters { get; set; } = new();

    /// <summary>
    /// Model artifacts generated
    /// </summary>
    [JsonPropertyName("models")]
    public List<TrainingModelArtifact> Models { get; set; } = new();

    /// <summary>
    /// Performance metrics summary
    /// </summary>
    [JsonPropertyName("metrics")]
    public TrainingMetrics Metrics { get; set; } = new();

    /// <summary>
    /// Data integrity information
    /// </summary>
    [JsonPropertyName("dataIntegrity")]
    public DataIntegrityInfo DataIntegrity { get; set; } = new();
}

/// <summary>
/// Date range for training data
/// </summary>
public sealed class DateRange
{
    [JsonPropertyName("startDate")]
    public DateTime StartDate { get; set; }

    [JsonPropertyName("endDate")]
    public DateTime EndDate { get; set; }

    [JsonPropertyName("tradingDays")]
    public int TradingDays { get; set; }
}

/// <summary>
/// Training model artifact information (distinct from BotCore.Services.ModelArtifact)
/// </summary>
public sealed class TrainingModelArtifact
{
    /// <summary>
    /// Model name/identifier
    /// </summary>
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Model file path (relative)
    /// </summary>
    [JsonPropertyName("filePath")]
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// SHA256 checksum for integrity verification
    /// </summary>
    [JsonPropertyName("sha256")]
    public string Sha256 { get; set; } = string.Empty;

    /// <summary>
    /// File size in bytes
    /// </summary>
    [JsonPropertyName("sizeBytes")]
    public long SizeBytes { get; set; }

    /// <summary>
    /// Model version
    /// </summary>
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;

    /// <summary>
    /// Model type (e.g., CVaR-PPO, Neural-UCB, LSTM)
    /// </summary>
    [JsonPropertyName("modelType")]
    public string ModelType { get; set; } = string.Empty;
}

/// <summary>
/// Training metrics summary
/// </summary>
public sealed class TrainingMetrics
{
    /// <summary>
    /// Validation loss
    /// </summary>
    [JsonPropertyName("validationLoss")]
    public double? ValidationLoss { get; set; }

    /// <summary>
    /// Sharpe ratio
    /// </summary>
    [JsonPropertyName("sharpeRatio")]
    public double? SharpeRatio { get; set; }

    /// <summary>
    /// Average reward
    /// </summary>
    [JsonPropertyName("averageReward")]
    public double? AverageReward { get; set; }

    /// <summary>
    /// Win rate percentage
    /// </summary>
    [JsonPropertyName("winRatePercent")]
    public double? WinRatePercent { get; set; }

    /// <summary>
    /// Additional metrics
    /// </summary>
    [JsonPropertyName("additionalMetrics")]
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}

/// <summary>
/// Data integrity information
/// </summary>
public sealed class DataIntegrityInfo
{
    /// <summary>
    /// Total bars loaded
    /// </summary>
    [JsonPropertyName("totalBars")]
    public int TotalBars { get; set; }

    /// <summary>
    /// Total experiences loaded
    /// </summary>
    [JsonPropertyName("totalExperiences")]
    public int TotalExperiences { get; set; }

    /// <summary>
    /// Data hash for change detection
    /// </summary>
    [JsonPropertyName("dataHash")]
    public string? DataHash { get; set; }

    /// <summary>
    /// Missing trading days detected
    /// </summary>
    [JsonPropertyName("missingDays")]
    public List<DateTime> MissingDays { get; set; } = new();

    /// <summary>
    /// Data completeness percentage
    /// </summary>
    [JsonPropertyName("completenessPercent")]
    public double CompletenessPercent { get; set; }
}
