using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.ModelLifecycle;

/// <summary>
/// Centralized feature engineering and versioning interface
/// </summary>
public interface IFeatureStore
{
    /// <summary>
    /// Register a new feature set version with metadata
    /// </summary>
    Task<string> RegisterFeatureSetAsync(FeatureSetMetadata featureSet, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get features for a specific feature set version
    /// </summary>
    Task<Dictionary<string, object>?> GetFeaturesAsync(string featureSetHash, Dictionary<string, object> inputs, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// List all available feature set versions
    /// </summary>
    Task<List<FeatureSetMetadata>> ListFeatureSetsAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validate feature set integrity and schema compatibility
    /// </summary>
    Task<FeatureSetValidationResult> ValidateFeatureSetAsync(string featureSetHash, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get the currently active feature set
    /// </summary>
    Task<FeatureSetMetadata?> GetActiveFeatureSetAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Set the active feature set version
    /// </summary>
    Task SetActiveFeatureSetAsync(string featureSetHash, CancellationToken cancellationToken = default);
}

/// <summary>
/// Feature set metadata with versioning and schema information
/// </summary>
public record FeatureSetMetadata(
    string Hash,
    string Version,
    string Name,
    DateTime CreatedAt,
    Dictionary<string, FeatureDefinition> Features,
    FeatureSetSchema Schema,
    string? Description = null,
    string? Status = null,
    Dictionary<string, object>? Parameters = null
);

/// <summary>
/// Definition of a single feature
/// </summary>
public record FeatureDefinition(
    string Name,
    FeatureType Type,
    string Description,
    bool Required = true,
    object? DefaultValue = null,
    Dictionary<string, object>? ValidationRules = null
);

/// <summary>
/// Feature set schema for validation
/// </summary>
public record FeatureSetSchema(
    string Version,
    Dictionary<string, string> InputRequirements,
    Dictionary<string, string> OutputSpecification,
    List<string> RequiredInputs,
    List<string> OptionalInputs
);

/// <summary>
/// Feature set validation result
/// </summary>
public record FeatureSetValidationResult(
    bool IsValid,
    List<string> Errors,
    List<string> Warnings,
    DateTime ValidatedAt
);

/// <summary>
/// Feature data types
/// </summary>
public enum FeatureType
{
    Numeric,
    Categorical,
    Text,
    DateTime,
    Boolean,
    Array,
    Object
}