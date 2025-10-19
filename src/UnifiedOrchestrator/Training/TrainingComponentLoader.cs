using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Loads training-components.json and creates in-memory component registry
/// Provides component lookup, grouping, and sorting functionality
/// </summary>
public sealed class TrainingComponentLoader
{
    private readonly ILogger<TrainingComponentLoader> _logger;
    private readonly string _componentsFilePath;
    private TrainingComponentsInventory? _inventory;

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() }
    };

    public TrainingComponentLoader(
        ILogger<TrainingComponentLoader> logger,
        string? componentsFilePath = null)
    {
        _logger = logger;
        _componentsFilePath = componentsFilePath ??
            Path.Combine(Directory.GetCurrentDirectory(), "src", "UnifiedOrchestrator", "training-components.json");
    }

    /// <summary>
    /// Load and parse training-components.json
    /// </summary>
    public async Task<bool> LoadComponentsAsync()
    {
        try
        {
            if (!File.Exists(_componentsFilePath))
            {
                _logger.LogError("[COMPONENT-LOADER] Training components file not found: {Path}", _componentsFilePath);
                return false;
            }

            var json = await File.ReadAllTextAsync(_componentsFilePath).ConfigureAwait(false);
            _inventory = JsonSerializer.Deserialize<TrainingComponentsInventory>(json, _jsonOptions);

            if (_inventory == null)
            {
                _logger.LogError("[COMPONENT-LOADER] Failed to deserialize training components");
                return false;
            }

            // Validate inventory
            if (!ValidateInventory())
            {
                return false;
            }

            _logger.LogInformation(
                "[COMPONENT-LOADER] Loaded {Heavy} heavy, {Medium} medium, {Light} light components",
                _inventory.Components?.Heavy?.Count ?? 0,
                _inventory.Components?.Medium?.Count ?? 0,
                _inventory.Components?.Light?.Count ?? 0);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[COMPONENT-LOADER] Failed to load training components: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Get all heavy training components (sorted by estimated time, longest first)
    /// </summary>
    public List<TrainingComponent> GetHeavyComponents()
    {
        EnsureLoaded();
        return _inventory!.Components!.Heavy!
            .OrderByDescending(c => c.EstimatedTimeMinutes)
            .ToList();
    }

    /// <summary>
    /// Get all medium training components (sorted by estimated time, longest first)
    /// </summary>
    public List<TrainingComponent> GetMediumComponents()
    {
        EnsureLoaded();
        return _inventory!.Components!.Medium!
            .OrderByDescending(c => c.EstimatedTimeMinutes)
            .ToList();
    }

    /// <summary>
    /// Get all light training components (sorted by estimated time, longest first)
    /// </summary>
    public List<TrainingComponent> GetLightComponents()
    {
        EnsureLoaded();
        return _inventory!.Components!.Light!
            .OrderByDescending(c => c.EstimatedTimeMilliseconds ?? 0)
            .ToList();
    }

    /// <summary>
    /// Get component by name for error handling
    /// </summary>
    public TrainingComponent? GetComponentByName(string name)
    {
        EnsureLoaded();

        return _inventory!.Components!.Heavy!.FirstOrDefault(c => c.Name == name)
            ?? _inventory.Components.Medium!.FirstOrDefault(c => c.Name == name)
            ?? _inventory.Components.Light!.FirstOrDefault(c => c.Name == name);
    }

    /// <summary>
    /// Calculate total estimated time for a phase
    /// </summary>
    public TimeSpan GetPhaseEstimatedTime(TrainingPhase phase)
    {
        EnsureLoaded();

        var totalMinutes = phase switch
        {
            TrainingPhase.Heavy => _inventory!.Components!.Heavy!.Sum(c => c.EstimatedTimeMinutes),
            TrainingPhase.Medium => _inventory!.Components!.Medium!.Sum(c => c.EstimatedTimeMinutes),
            TrainingPhase.Light => (_inventory!.Components!.Light!.Sum(c => c.EstimatedTimeMilliseconds ?? 0)) / 1000.0 / 60.0,
            _ => 0
        };

        return TimeSpan.FromMinutes(totalMinutes);
    }

    /// <summary>
    /// Get total estimated time for all phases
    /// </summary>
    public TimeSpan GetTotalEstimatedTime()
    {
        return GetPhaseEstimatedTime(TrainingPhase.Heavy)
            + GetPhaseEstimatedTime(TrainingPhase.Medium)
            + GetPhaseEstimatedTime(TrainingPhase.Light);
    }

    /// <summary>
    /// Get total component count across all phases
    /// </summary>
    public int GetTotalComponentCount()
    {
        EnsureLoaded();
        return (_inventory!.Components!.Heavy?.Count ?? 0)
            + (_inventory.Components.Medium?.Count ?? 0)
            + (_inventory.Components.Light?.Count ?? 0);
    }

    private void EnsureLoaded()
    {
        if (_inventory == null)
        {
            throw new InvalidOperationException("Training components not loaded. Call LoadComponentsAsync first.");
        }
    }

    private bool ValidateInventory()
    {
        if (_inventory!.Components == null)
        {
            _logger.LogError("[COMPONENT-LOADER] Components section missing");
            return false;
        }

        var hasHeavy = _inventory.Components.Heavy != null && _inventory.Components.Heavy.Count > 0;
        var hasMedium = _inventory.Components.Medium != null && _inventory.Components.Medium.Count > 0;
        var hasLight = _inventory.Components.Light != null && _inventory.Components.Light.Count > 0;

        if (!hasHeavy && !hasMedium && !hasLight)
        {
            _logger.LogError("[COMPONENT-LOADER] No components found in any phase");
            return false;
        }

        // Validate each component has required fields
        var allComponents = new List<TrainingComponent>();
        if (hasHeavy) allComponents.AddRange(_inventory.Components.Heavy!);
        if (hasMedium) allComponents.AddRange(_inventory.Components.Medium!);
        if (hasLight) allComponents.AddRange(_inventory.Components.Light!);

        foreach (var component in allComponents)
        {
            if (string.IsNullOrWhiteSpace(component.Name))
            {
                _logger.LogError("[COMPONENT-LOADER] Component missing name");
                return false;
            }

            if (string.IsNullOrWhiteSpace(component.ClassName))
            {
                _logger.LogError("[COMPONENT-LOADER] Component {Name} missing className", component.Name);
                return false;
            }
        }

        return true;
    }
}

/// <summary>
/// Training components inventory structure (matches JSON schema)
/// </summary>
public sealed class TrainingComponentsInventory
{
    public string? Version { get; set; }
    public TrainingComponentsSummary? Summary { get; set; }
    public TrainingComponentsCollection? Components { get; set; }
    public Dictionary<string, object>? Metadata { get; set; }
}

public sealed class TrainingComponentsSummary
{
    public int Documented { get; set; }
    public int TargetTotal { get; set; }
}

public sealed class TrainingComponentsCollection
{
    public List<TrainingComponent>? Heavy { get; set; }
    public List<TrainingComponent>? Medium { get; set; }
    public List<TrainingComponent>? Light { get; set; }
}

/// <summary>
/// Individual training component definition
/// </summary>
public sealed class TrainingComponent
{
    public string Name { get; set; } = string.Empty;
    public string ClassName { get; set; } = string.Empty;
    public string? FilePath { get; set; }
    public string? Phase { get; set; }
    public string? Category { get; set; }
    public double EstimatedTimeMinutes { get; set; }
    public double? EstimatedTimeMilliseconds { get; set; }
    public string? Description { get; set; }
    public List<string>? Dependencies { get; set; }
    public Dictionary<string, object>? Configuration { get; set; }

    // Parsed dependencies
    public bool RequiresExperienceDb => Dependencies?.Contains("experience_database") ?? false;
    public bool RequiresHistoricalData => Dependencies?.Contains("historical_data") ?? false;
}
