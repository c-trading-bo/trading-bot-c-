using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.LearningPipeline;

/// <summary>
/// Interface for dataset versioning and hash-based tracking
/// </summary>
public interface IDatasetVersionManager
{
    /// <summary>
    /// Register a new dataset version with hash and metadata
    /// </summary>
    Task<string> RegisterDatasetAsync(DatasetMetadata dataset, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Load dataset metadata by hash
    /// </summary>
    Task<DatasetMetadata?> GetDatasetAsync(string datasetHash, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validate dataset integrity using stored hash
    /// </summary>
    Task<bool> ValidateDatasetIntegrityAsync(string datasetHash, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// List all available dataset versions
    /// </summary>
    Task<List<DatasetMetadata>> ListDatasetsAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get dataset lineage and dependencies
    /// </summary>
    Task<DatasetLineage> GetDatasetLineageAsync(string datasetHash, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Create a new dataset version from transformations
    /// </summary>
    Task<string> CreateDerivedDatasetAsync(string sourceHash, DatasetTransformation transformation, CancellationToken cancellationToken = default);
}

/// <summary>
/// Dataset metadata with versioning and lineage information
/// </summary>
public record DatasetMetadata(
    string Hash,
    string Version,
    string Name,
    DateTime CreatedAt,
    long SizeBytes,
    int RecordCount,
    DatasetSchema Schema,
    string FilePath,
    string? SourceHash = null,
    DatasetTransformation? Transformation = null,
    Dictionary<string, object>? Statistics = null,
    string? Description = null
);

/// <summary>
/// Dataset schema definition
/// </summary>
public record DatasetSchema(
    string Version,
    Dictionary<string, ColumnDefinition> Columns,
    List<string> RequiredColumns,
    Dictionary<string, object>? Constraints = null
);

/// <summary>
/// Column definition in dataset schema
/// </summary>
public record ColumnDefinition(
    string Name,
    string DataType,
    bool Nullable = true,
    object? DefaultValue = null,
    Dictionary<string, object>? ValidationRules = null
);

/// <summary>
/// Dataset transformation record
/// </summary>
public record DatasetTransformation(
    string TransformationType,
    Dictionary<string, object> Parameters,
    string? CodeHash = null,
    List<string>? Dependencies = null
);

/// <summary>
/// Dataset lineage information
/// </summary>
public record DatasetLineage(
    string DatasetHash,
    List<string> ParentDatasets,
    List<string> ChildDatasets,
    List<DatasetTransformation> TransformationChain,
    DateTime CreatedAt
);