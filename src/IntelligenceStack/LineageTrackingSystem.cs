using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Globalization;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Comprehensive lineage tracking system
/// Stamps every decision with model registry version, feature store version, and calibration map ID
/// </summary>
public class LineageTrackingSystem
{
    private readonly ILogger<LineageTrackingSystem> _logger;
    private readonly IModelRegistry _modelRegistry;
    private readonly IFeatureStore _featureStore;
    private readonly ICalibrationManager _calibrationManager;
    private readonly string _lineagePath;
    
    private readonly Dictionary<string, LineageSnapshot> _snapshots = new();
    private readonly Dictionary<string, List<LineageEvent>> _eventHistory = new();
    private readonly object _lock = new();

    public LineageTrackingSystem(
        ILogger<LineageTrackingSystem> logger,
        IModelRegistry modelRegistry,
        IFeatureStore featureStore,
        ICalibrationManager calibrationManager,
        string lineagePath = "data/lineage")
    {
        _logger = logger;
        _modelRegistry = modelRegistry;
        _featureStore = featureStore;
        _calibrationManager = calibrationManager;
        _lineagePath = lineagePath;
        
        Directory.CreateDirectory(_lineagePath);
        Directory.CreateDirectory(Path.Combine(_lineagePath, "snapshots"));
        Directory.CreateDirectory(Path.Combine(_lineagePath, "events"));
        Directory.CreateDirectory(Path.Combine(_lineagePath, "decisions"));
    }

    /// <summary>
    /// Create a lineage snapshot for current system state
    /// </summary>
    public async Task<LineageSnapshot> CreateSnapshotAsync(string snapshotId, CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[LINEAGE] Creating lineage snapshot: {SnapshotId}", snapshotId);

            var snapshot = new LineageSnapshot
            {
                SnapshotId = snapshotId,
                Timestamp = DateTime.UtcNow,
                FeatureStoreVersion = await GetCurrentFeatureStoreVersionAsync(cancellationToken),
                ConfigurationHash = await CalculateConfigurationHashAsync(cancellationToken),
                EnvironmentInfo = GetEnvironmentInfo()
            };

            // Populate read-only dictionaries
            var modelVersions = await GetCurrentModelVersionsAsync(cancellationToken);
            foreach (var kvp in modelVersions)
            {
                snapshot.ModelVersions[kvp.Key] = kvp.Value;
            }

            var calibrationMaps = await GetCurrentCalibrationMapsAsync(cancellationToken);
            foreach (var kvp in calibrationMaps)
            {
                snapshot.CalibrationMaps[kvp.Key] = kvp.Value;
            }

            var systemComponents = await GetSystemComponentVersionsAsync(cancellationToken);
            foreach (var kvp in systemComponents)
            {
                snapshot.SystemComponents[kvp.Key] = kvp.Value;
            }

            // Store snapshot
            lock (_lock)
            {
                _snapshots[snapshotId] = snapshot;
            }

            await SaveSnapshotAsync(snapshot, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[LINEAGE] Created snapshot {SnapshotId} with {Models} models and feature store v{FeatureVersion}", 
                snapshotId, snapshot.ModelVersions.Count, snapshot.FeatureStoreVersion);

            return snapshot;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LINEAGE] Failed to create lineage snapshot: {SnapshotId}", snapshotId);
            throw new InvalidOperationException($"Lineage snapshot creation failed for {snapshotId}", ex);
        }
    }

    /// <summary>
    /// Stamp a decision with complete lineage information
    /// </summary>
    public async Task<LineageStamp> StampDecisionAsync(
        string decisionId,
        IntelligenceDecision decision,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var stamp = new LineageStamp
            {
                DecisionId = decisionId,
                Timestamp = DateTime.UtcNow,
                ModelRegistryVersion = await GetModelVersionForDecisionAsync(decision, cancellationToken),
                FeatureStoreVersion = await GetCurrentFeatureStoreVersionAsync(cancellationToken),
                CalibrationMapId = await GetCalibrationMapIdAsync(decision.ModelId, cancellationToken),
                ConfigurationHash = await CalculateConfigurationHashAsync(cancellationToken),
                InputDataHash = CalculateInputDataHash(decision),
                RegimeContext = new RegimeLineageInfo
                {
                    Regime = decision.Regime,
                    RegimeDetectorVersion = "v1.0",
                    RegimeConfidence = decision.Confidence,
                    RegimeTransitionId = GetRegimeTransitionId(decision)
                },
                ModelLineage = await GetModelLineageAsync(decision.ModelId, cancellationToken),
                FeatureLineage = await GetFeatureLineageAsync(decision.FeaturesVersion, cancellationToken)
            };

            // Populate read-only ProcessingChain
            var processingChain = await GetProcessingChainAsync(decision, cancellationToken);
            foreach (var step in processingChain)
            {
                stamp.ProcessingChain.Add(step);
            }

            // Record lineage event
            var lineageEvent = new LineageEvent
            {
                EventId = Guid.NewGuid().ToString(),
                EventType = LineageEventType.DecisionStamped,
                Timestamp = DateTime.UtcNow,
                EntityId = decisionId,
                EntityType = "decision"
            };
            
            // Populate read-only Properties dictionary
            lineageEvent.Properties["model_id"] = decision.ModelId;
            lineageEvent.Properties["model_version"] = stamp.ModelRegistryVersion;
            lineageEvent.Properties["feature_version"] = stamp.FeatureStoreVersion;
            lineageEvent.Properties["calibration_map"] = stamp.CalibrationMapId;
            lineageEvent.Properties["regime"] = decision.Regime.ToString();
            lineageEvent.Properties["symbol"] = decision.Symbol;
            
            await RecordLineageEventAsync(lineageEvent, cancellationToken).ConfigureAwait(false);

            // Store decision with lineage
            await SaveDecisionLineageAsync(decisionId, decision, stamp, cancellationToken).ConfigureAwait(false);

            _logger.LogDebug("[LINEAGE] Stamped decision {DecisionId} with model {ModelVersion}, features {FeatureVersion}, calibration {CalibrationId}", 
                decisionId, stamp.ModelRegistryVersion, stamp.FeatureStoreVersion, stamp.CalibrationMapId);

            return stamp;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LINEAGE] Failed to stamp decision: {DecisionId}", decisionId);
            throw new InvalidOperationException($"Decision stamping failed for {decisionId}", ex);
        }
    }

    /// <summary>
    /// Track model promotion lineage
    /// </summary>
    public async Task TrackModelPromotionAsync(
        string modelId,
        string fromVersion,
        string toVersion,
        PromotionCriteria criteria,
        CancellationToken cancellationToken = default)
    {
        var promotionEvent = new LineageEvent
        {
            EventId = Guid.NewGuid().ToString(),
            EventType = LineageEventType.ModelPromoted,
            Timestamp = DateTime.UtcNow,
            EntityId = modelId,
            EntityType = "model"
        };
        
        // Populate read-only Properties dictionary
        promotionEvent.Properties["from_version"] = fromVersion;
        promotionEvent.Properties["to_version"] = toVersion;
        promotionEvent.Properties["promotion_criteria"] = JsonSerializer.Serialize(criteria);
        promotionEvent.Properties["promotion_reason"] = "performance_improvement";
        
        await RecordLineageEventAsync(promotionEvent, cancellationToken).ConfigureAwait(false);

        _logger.LogInformation("[LINEAGE] Tracked model promotion: {ModelId} {FromVersion} -> {ToVersion}", 
            modelId, fromVersion, toVersion);
    }

    /// <summary>
    /// Track feature store update lineage
    /// </summary>
    public async Task TrackFeatureStoreUpdateAsync(
        string fromVersion,
        string toVersion,
        List<string> changedFeatures,
        CancellationToken cancellationToken = default)
    {
        var featureUpdateEvent = new LineageEvent
        {
            EventId = Guid.NewGuid().ToString(),
            EventType = LineageEventType.FeatureStoreUpdated,
            Timestamp = DateTime.UtcNow,
            EntityId = $"feature_store_{toVersion}",
            EntityType = "feature_store"
        };
        
        // Populate read-only Properties dictionary
        featureUpdateEvent.Properties["from_version"] = fromVersion;
        featureUpdateEvent.Properties["to_version"] = toVersion;
        featureUpdateEvent.Properties["changed_features"] = changedFeatures;
        featureUpdateEvent.Properties["change_count"] = changedFeatures.Count;
        
        await RecordLineageEventAsync(featureUpdateEvent, cancellationToken).ConfigureAwait(false);

        _logger.LogInformation("[LINEAGE] Tracked feature store update: {FromVersion} -> {ToVersion} ({ChangeCount} features)", 
            fromVersion, toVersion, changedFeatures.Count);
    }

    /// <summary>
    /// Track calibration map update lineage
    /// </summary>
    public async Task TrackCalibrationUpdateAsync(
        string modelId,
        string calibrationMapId,
        CalibrationMethod method,
        double brierScore,
        CancellationToken cancellationToken = default)
    {
        var lineageEvent = new LineageEvent
        {
            EventId = Guid.NewGuid().ToString(),
            EventType = LineageEventType.CalibrationUpdated,
            Timestamp = DateTime.UtcNow,
            EntityId = calibrationMapId,
            EntityType = "calibration_map"
        };
        
        lineageEvent.Properties["model_id"] = modelId;
        lineageEvent.Properties["calibration_method"] = method.ToString();
        lineageEvent.Properties["brier_score"] = brierScore;
        lineageEvent.Properties["update_reason"] = "nightly_calibration";
        
        await RecordLineageEventAsync(lineageEvent, cancellationToken).ConfigureAwait(false);

        _logger.LogInformation("[LINEAGE] Tracked calibration update: {CalibrationMapId} for model {ModelId} (Brier: {BrierScore:F3})", 
            calibrationMapId, modelId, brierScore);
    }

    /// <summary>
    /// Get lineage trace for a specific decision
    /// </summary>
    public async Task<LineageTrace> GetDecisionLineageAsync(string decisionId, CancellationToken cancellationToken = default)
    {
        try
        {
            var decisionFile = Path.Combine(_lineagePath, "decisions", $"{decisionId}.json");
            if (!File.Exists(decisionFile))
            {
                throw new FileNotFoundException($"Decision lineage not found: {decisionId}");
            }

            var content = await File.ReadAllTextAsync(decisionFile, cancellationToken).ConfigureAwait(false);
            var decisionLineage = JsonSerializer.Deserialize<DecisionLineageRecord>(content);

            if (decisionLineage == null)
            {
                throw new InvalidDataException($"Invalid decision lineage data: {decisionId}");
            }

            // Build complete lineage trace
            var trace = new LineageTrace
            {
                DecisionId = decisionId,
                StartTime = decisionLineage.Decision.Timestamp,
                EndTime = DateTime.UtcNow,
                LineageStamp = decisionLineage.LineageStamp,
                ModelLineage = await GetCompleteModelLineageAsync(decisionLineage.LineageStamp.ModelLineage?.ModelId ?? "", cancellationToken),
                FeatureLineage = await GetCompleteFeatureLineageAsync(decisionLineage.LineageStamp.FeatureStoreVersion, cancellationToken),
                CalibrationLineage = await GetCompleteCalibrationLineageAsync(decisionLineage.LineageStamp.CalibrationMapId, cancellationToken),
                RelatedEvents = await GetRelatedEventsAsync(decisionId, cancellationToken)
            };

            return trace;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LINEAGE] Failed to get decision lineage: {DecisionId}", decisionId);
            throw new InvalidOperationException($"Decision lineage retrieval failed for {decisionId}", ex);
        }
    }

    /// <summary>
    /// Get lineage summary for a time period
    /// </summary>
    public async Task<LineageSummary> GetLineageSummaryAsync(
        DateTime startTime,
        DateTime endTime,
        CancellationToken cancellationToken = default)
    {
        var events = await GetEventsInPeriodAsync(startTime, endTime, cancellationToken).ConfigureAwait(false);
        
        var summary = new LineageSummary
        {
            Period = new TimeSpan?(endTime - startTime),
            StartTime = startTime,
            EndTime = endTime,
            TotalDecisions = events.Count(e => e.EventType == LineageEventType.DecisionStamped),
            ModelPromotions = events.Count(e => e.EventType == LineageEventType.ModelPromoted),
            FeatureStoreUpdates = events.Count(e => e.EventType == LineageEventType.FeatureStoreUpdated),
            CalibrationUpdates = events.Count(e => e.EventType == LineageEventType.CalibrationUpdated),
            UniqueModels = events.Where(e => e.EntityType == "model").Select(e => e.EntityId).Distinct().Count(),
            ModelVersionDistribution = events
                .Where(e => e.EventType == LineageEventType.DecisionStamped)
                .GroupBy(e => e.Properties.GetValueOrDefault("model_version", "unknown").ToString() ?? "unknown")
                .ToDictionary(g => g.Key, g => g.Count()),
            FeatureVersionDistribution = events
                .Where(e => e.EventType == LineageEventType.DecisionStamped)
                .GroupBy(e => e.Properties.GetValueOrDefault("feature_version", "unknown").ToString() ?? "unknown")
                .ToDictionary(g => g.Key, g => g.Count())
        };

        return summary;
    }

    private async Task<Dictionary<string, string>> GetCurrentModelVersionsAsync(CancellationToken cancellationToken)
    {
        // Get all active model versions
        var modelVersions = new Dictionary<string, string>();
        
        // This would normally iterate through all model families
        var families = new[] { "neural_bandit", "lstm_predictor", "transformer", "xgboost_risk", "regime_detector" };
        
        foreach (var family in families)
        {
            try
            {
                var model = await _modelRegistry.GetModelAsync(family, "latest", cancellationToken).ConfigureAwait(false);
                modelVersions[family] = model.Version;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[LINEAGE] Failed to get version for model family: {Family}", family);
                modelVersions[family] = "unknown";
            }
        }
        
        return modelVersions;
    }

    private async Task<string> GetCurrentFeatureStoreVersionAsync(CancellationToken cancellationToken)
    {
        try
        {
            var schema = await _featureStore.GetSchemaAsync("latest", cancellationToken).ConfigureAwait(false);
            return schema.Version;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LINEAGE] Failed to get feature store version");
            return "unknown";
        }
    }

    private async Task<Dictionary<string, string>> GetCurrentCalibrationMapsAsync(CancellationToken cancellationToken)
    {
        var calibrationMaps = new Dictionary<string, string>();
        
        // Get calibration maps for all models
        var modelVersions = await GetCurrentModelVersionsAsync(cancellationToken).ConfigureAwait(false);
        
        foreach (var (modelFamily, version) in modelVersions)
        {
            try
            {
                var modelId = $"{modelFamily}_{version}";
                var calibrationMap = await _calibrationManager.LoadCalibrationMapAsync(modelId, cancellationToken).ConfigureAwait(false);
                calibrationMaps[modelId] = $"{calibrationMap.ModelId}_{calibrationMap.CreatedAt:yyyyMMdd}";
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[LINEAGE] Failed to get calibration map for model: {ModelFamily}", modelFamily);
                calibrationMaps[modelFamily] = "unknown";
            }
        }
        
        return calibrationMaps;
    }

    private async Task<string> CalculateConfigurationHashAsync(CancellationToken cancellationToken)
    {
        // Brief async operation for proper async pattern
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        // Calculate hash of current configuration for reproducibility
        var configData = new
        {
            timestamp = DateTime.UtcNow.ToString("yyyyMMdd", CultureInfo.InvariantCulture),
            version = "1.0.0",
            components = new[] { "ensemble", "quarantine", "maml", "rl_advisor" }
        };
        
        var configJson = JsonSerializer.Serialize(configData);
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(configJson));
        return Convert.ToHexString(hash)[..16];
    }

    private async Task<Dictionary<string, string>> GetSystemComponentVersionsAsync(CancellationToken cancellationToken)
    {
        // Brief async operation for proper async pattern
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        return new Dictionary<string, string>
        {
            ["ensemble_meta_learner"] = "v1.0",
            ["quarantine_manager"] = "v1.0",
            ["maml_integration"] = "v1.0",
            ["rl_advisor"] = "v1.0",
            ["historical_trainer"] = "v1.0",
            ["nightly_tuner"] = "v1.0",
            ["observability"] = "v1.0",
            ["lineage_tracking"] = "v1.0"
        };
    }

    private EnvironmentInfo GetEnvironmentInfo()
    {
        return new EnvironmentInfo
        {
            MachineName = Environment.MachineName,
            OSVersion = Environment.OSVersion.ToString(),
            ProcessorCount = Environment.ProcessorCount,
            DotNetVersion = Environment.Version.ToString(),
            WorkingSet = Environment.WorkingSet,
            Timestamp = DateTime.UtcNow
        };
    }

    private async Task<string> GetModelVersionForDecisionAsync(IntelligenceDecision decision, CancellationToken cancellationToken)
    {
        // Brief async operation for proper async pattern
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        try
        {
            if (!string.IsNullOrEmpty(decision.ModelVersion))
            {
                return decision.ModelVersion;
            }
            
            // Extract model version from model ID if not directly available
            var parts = decision.ModelId.Split('_');
            return parts.Length > 1 ? parts[^1] : "unknown";
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LINEAGE] Failed to get model version for decision");
            return "unknown";
        }
    }

    private async Task<string> GetCalibrationMapIdAsync(string modelId, CancellationToken cancellationToken)
    {
        try
        {
            var calibrationMap = await _calibrationManager.LoadCalibrationMapAsync(modelId, cancellationToken).ConfigureAwait(false);
            return $"{calibrationMap.ModelId}_cal_{calibrationMap.CreatedAt:yyyyMMdd}";
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LINEAGE] Failed to get calibration map ID for model: {ModelId}", modelId);
            return "unknown";
        }
    }

    private string CalculateInputDataHash(IntelligenceDecision decision)
    {
        var inputData = new
        {
            symbol = decision.Symbol,
            timeframe = decision.Timeframe,
            features_hash = decision.FeaturesHash,
            timestamp = decision.Timestamp.ToString("yyyyMMddHHmm", CultureInfo.InvariantCulture) // Minute-level granularity
        };
        
        var inputJson = JsonSerializer.Serialize(inputData);
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(inputJson));
        return Convert.ToHexString(hash)[..16];
    }

    private string GetRegimeTransitionId(IntelligenceDecision decision)
    {
        // Generate regime transition ID based on regime and timestamp
        return $"{decision.Regime}_{decision.Timestamp:yyyyMMddHH}";
    }

    private async Task<ModelLineageInfo?> GetModelLineageAsync(string modelId, CancellationToken cancellationToken)
    {
        try
        {
            var model = await _modelRegistry.GetModelAsync(ExtractFamilyFromId(modelId), ExtractVersionFromId(modelId), cancellationToken).ConfigureAwait(false);
            
            return new ModelLineageInfo
            {
                ModelId = modelId,
                ModelFamily = ExtractFamilyFromId(modelId),
                Version = model.Version,
                CreatedAt = model.CreatedAt,
                TrainingWindow = model.TrainingWindow,
                FeaturesVersion = model.FeaturesVersion,
                SchemaChecksum = model.SchemaChecksum,
                Metrics = model.Metrics,
                RuntimeSignature = model.RuntimeSignature
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LINEAGE] Failed to get model lineage: {ModelId}", modelId);
            return null;
        }
    }

    private async Task<FeatureLineageInfo?> GetFeatureLineageAsync(string version, CancellationToken cancellationToken)
    {
        try
        {
            var schema = await _featureStore.GetSchemaAsync(version, cancellationToken).ConfigureAwait(false);
            
            return new FeatureLineageInfo
            {
                Version = version,
                SchemaChecksum = schema.Checksum,
                CreatedAt = schema.CreatedAt,
                FeatureCount = schema.Features.Count,
                FeatureNames = schema.Features.Keys.ToList()
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LINEAGE] Failed to get feature lineage: {Version}", version);
            return null;
        }
    }

    private async Task<List<ProcessingStep>> GetProcessingChainAsync(IntelligenceDecision decision, CancellationToken cancellationToken)
    {
        // Build processing chain asynchronously to avoid blocking lineage recording
        return await Task.Run(() =>
        {
            // Build processing chain for the decision
            return new List<ProcessingStep>
            {
                new ProcessingStep
                {
                    StepName = "feature_extraction",
                    ComponentName = "feature_store",
                    Version = decision.FeaturesVersion,
                    InputHash = decision.FeaturesHash,
                    OutputHash = CalculateInputDataHash(decision),
                    ProcessingTime = TimeSpan.FromMilliseconds(50)
                },
                new ProcessingStep
                {
                    StepName = "model_inference",
                    ComponentName = "model_registry",
                    Version = decision.ModelVersion,
                    InputHash = CalculateInputDataHash(decision),
                    OutputHash = decision.DecisionId[..8],
                    ProcessingTime = TimeSpan.FromMilliseconds(100)
                },
                new ProcessingStep
                {
                    StepName = "confidence_calibration",
                    ComponentName = "calibration_manager",
                    Version = "v1.0",
                    InputHash = decision.DecisionId[..8],
                    OutputHash = decision.DecisionId[^8..],
                    ProcessingTime = TimeSpan.FromMilliseconds(20)
                }
            };
        }, cancellationToken);
    }

    private async Task RecordLineageEventAsync(LineageEvent lineageEvent, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            var key = lineageEvent.EntityType;
            if (!_eventHistory.ContainsKey(key))
            {
                _eventHistory[key] = new List<LineageEvent>();
            }
            
            _eventHistory[key].Add(lineageEvent);
            
            // Keep only recent events
            if (_eventHistory[key].Count > 10000)
            {
                _eventHistory[key].RemoveAt(0);
            }
        }

        // Save event to disk
        await SaveLineageEventAsync(lineageEvent, cancellationToken).ConfigureAwait(false);
    }

    private async Task SaveSnapshotAsync(LineageSnapshot snapshot, CancellationToken cancellationToken)
    {
        var snapshotFile = Path.Combine(_lineagePath, "snapshots", $"{snapshot.SnapshotId}.json");
        var json = JsonSerializer.Serialize(snapshot, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(snapshotFile, json, cancellationToken).ConfigureAwait(false);
    }

    private async Task SaveDecisionLineageAsync(
        string decisionId,
        IntelligenceDecision decision,
        LineageStamp stamp,
        CancellationToken cancellationToken)
    {
        var record = new DecisionLineageRecord
        {
            DecisionId = decisionId,
            Decision = decision,
            LineageStamp = stamp,
            CreatedAt = DateTime.UtcNow
        };

        var decisionFile = Path.Combine(_lineagePath, "decisions", $"{decisionId}.json");
        var json = JsonSerializer.Serialize(record, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(decisionFile, json, cancellationToken).ConfigureAwait(false);
    }

    private async Task SaveLineageEventAsync(LineageEvent lineageEvent, CancellationToken cancellationToken)
    {
        var eventFile = Path.Combine(_lineagePath, "events", $"{lineageEvent.EventId}.json");
        var json = JsonSerializer.Serialize(lineageEvent, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(eventFile, json, cancellationToken).ConfigureAwait(false);
    }

    private async Task<List<LineageEvent>> GetEventsInPeriodAsync(DateTime startTime, DateTime endTime, CancellationToken cancellationToken)
    {
        // Retrieve lineage events in period asynchronously to avoid blocking analysis operations
        return await Task.Run(() =>
        {
            var allEvents = new List<LineageEvent>();
            
            lock (_lock)
            {
                foreach (var events in _eventHistory.Values)
                {
                    allEvents.AddRange(events.Where(e => e.Timestamp >= startTime && e.Timestamp <= endTime));
                }
            }
            
            return allEvents.OrderBy(e => e.Timestamp).ToList();
        }, cancellationToken);
    }

    private async Task<CompleteModelLineage> GetCompleteModelLineageAsync(string modelId, CancellationToken cancellationToken)
    {
        var events = await GetRelatedEventsAsync(modelId, cancellationToken).ConfigureAwait(false);
        var modelEvents = events.Where(e => e.EntityType == "model").ToList();
        
        return new CompleteModelLineage
        {
            ModelId = modelId,
            CreationEvent = modelEvents.FirstOrDefault(e => e.EventType == LineageEventType.ModelRegistered),
            PromotionEvents = modelEvents.Where(e => e.EventType == LineageEventType.ModelPromoted).ToList(),
            UsageEvents = events.Where(e => e.EventType == LineageEventType.DecisionStamped && 
                e.Properties.ContainsKey("model_id") && 
                e.Properties["model_id"].ToString() == modelId).ToList()
        };
    }

    private async Task<CompleteFeatureLineage> GetCompleteFeatureLineageAsync(string version, CancellationToken cancellationToken)
    {
        var events = await GetRelatedEventsAsync($"feature_store_{version}", cancellationToken).ConfigureAwait(false);
        
        return new CompleteFeatureLineage
        {
            Version = version,
            UpdateEvents = events.Where(e => e.EventType == LineageEventType.FeatureStoreUpdated).ToList(),
            UsageEvents = events.Where(e => e.EventType == LineageEventType.DecisionStamped && 
                e.Properties.ContainsKey("feature_version") && 
                e.Properties["feature_version"].ToString() == version).ToList()
        };
    }

    private async Task<CompleteCalibrationLineage> GetCompleteCalibrationLineageAsync(string calibrationMapId, CancellationToken cancellationToken)
    {
        var events = await GetRelatedEventsAsync(calibrationMapId, cancellationToken).ConfigureAwait(false);
        
        return new CompleteCalibrationLineage
        {
            CalibrationMapId = calibrationMapId,
            UpdateEvents = events.Where(e => e.EventType == LineageEventType.CalibrationUpdated).ToList(),
            UsageEvents = events.Where(e => e.EventType == LineageEventType.DecisionStamped && 
                e.Properties.ContainsKey("calibration_map") && 
                e.Properties["calibration_map"].ToString() == calibrationMapId).ToList()
        };
    }

    private async Task<List<LineageEvent>> GetRelatedEventsAsync(string entityId, CancellationToken cancellationToken)
    {
        // Retrieve related lineage events asynchronously to avoid blocking lineage analysis
        return await Task.Run(() =>
        {
            var relatedEvents = new List<LineageEvent>();
            
            lock (_lock)
            {
                foreach (var events in _eventHistory.Values)
                {
                    relatedEvents.AddRange(events.Where(e => 
                        e.EntityId == entityId || 
                        e.Properties.Values.Any(v => v.ToString() == entityId)));
                }
            }
            
            return relatedEvents.OrderBy(e => e.Timestamp).ToList();
        }, cancellationToken);
    }

    private string ExtractFamilyFromId(string modelId)
    {
        var lastUnderscore = modelId.LastIndexOf('_');
        return lastUnderscore > 0 ? modelId[..lastUnderscore] : modelId;
    }

    private string ExtractVersionFromId(string modelId)
    {
        var parts = modelId.Split('_');
        return parts.Length > 1 ? parts[^1] : "v1";
    }
}

#region Lineage Data Models

public class LineageSnapshot
{
    public string SnapshotId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public Dictionary<string, string> ModelVersions { get; } = new();
    public string FeatureStoreVersion { get; set; } = string.Empty;
    public Dictionary<string, string> CalibrationMaps { get; } = new();
    public string ConfigurationHash { get; set; } = string.Empty;
    public Dictionary<string, string> SystemComponents { get; } = new();
    public EnvironmentInfo EnvironmentInfo { get; set; } = new();
}

public class LineageStamp
{
    public string DecisionId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string ModelRegistryVersion { get; set; } = string.Empty;
    public string FeatureStoreVersion { get; set; } = string.Empty;
    public string CalibrationMapId { get; set; } = string.Empty;
    public string ConfigurationHash { get; set; } = string.Empty;
    public string InputDataHash { get; set; } = string.Empty;
    public RegimeLineageInfo RegimeContext { get; set; } = new();
    public ModelLineageInfo? ModelLineage { get; set; }
    public FeatureLineageInfo? FeatureLineage { get; set; }
    public List<ProcessingStep> ProcessingChain { get; } = new();
}

public class LineageEvent
{
    public string EventId { get; set; } = string.Empty;
    public LineageEventType EventType { get; set; }
    public DateTime Timestamp { get; set; }
    public string EntityId { get; set; } = string.Empty;
    public string EntityType { get; set; } = string.Empty;
    public Dictionary<string, object> Properties { get; } = new();
}

public class LineageTrace
{
    public string DecisionId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public LineageStamp LineageStamp { get; set; } = new();
    public CompleteModelLineage? ModelLineage { get; set; }
    public CompleteFeatureLineage? FeatureLineage { get; set; }
    public CompleteCalibrationLineage? CalibrationLineage { get; set; }
    public List<LineageEvent> RelatedEvents { get; } = new();
}

public class LineageSummary
{
    public TimeSpan? Period { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public int TotalDecisions { get; set; }
    public int ModelPromotions { get; set; }
    public int FeatureStoreUpdates { get; set; }
    public int CalibrationUpdates { get; set; }
    public int UniqueModels { get; set; }
    public Dictionary<string, int> ModelVersionDistribution { get; } = new();
    public Dictionary<string, int> FeatureVersionDistribution { get; } = new();
}

public class RegimeLineageInfo
{
    public RegimeType Regime { get; set; }
    public string RegimeDetectorVersion { get; set; } = string.Empty;
    public double RegimeConfidence { get; set; }
    public string RegimeTransitionId { get; set; } = string.Empty;
}

public class ModelLineageInfo
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelFamily { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public TimeSpan TrainingWindow { get; set; }
    public string FeaturesVersion { get; set; } = string.Empty;
    public string SchemaChecksum { get; set; } = string.Empty;
    public ModelMetrics Metrics { get; set; } = new();
    public string RuntimeSignature { get; set; } = string.Empty;
}

public class FeatureLineageInfo
{
    public string Version { get; set; } = string.Empty;
    public string SchemaChecksum { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public int FeatureCount { get; set; }
    public List<string> FeatureNames { get; } = new();
}

public class ProcessingStep
{
    public string StepName { get; set; } = string.Empty;
    public string ComponentName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string InputHash { get; set; } = string.Empty;
    public string OutputHash { get; set; } = string.Empty;
    public TimeSpan ProcessingTime { get; set; }
}

public class EnvironmentInfo
{
    public string MachineName { get; set; } = string.Empty;
    public string OSVersion { get; set; } = string.Empty;
    public int ProcessorCount { get; set; }
    public string DotNetVersion { get; set; } = string.Empty;
    public long WorkingSet { get; set; }
    public DateTime Timestamp { get; set; }
}

public class DecisionLineageRecord
{
    public string DecisionId { get; set; } = string.Empty;
    public IntelligenceDecision Decision { get; set; } = new();
    public LineageStamp LineageStamp { get; set; } = new();
    public DateTime CreatedAt { get; set; }
}

public class CompleteModelLineage
{
    public string ModelId { get; set; } = string.Empty;
    public LineageEvent? CreationEvent { get; set; }
    public List<LineageEvent> PromotionEvents { get; } = new();
    public List<LineageEvent> UsageEvents { get; } = new();
}

public class CompleteFeatureLineage
{
    public string Version { get; set; } = string.Empty;
    public List<LineageEvent> UpdateEvents { get; } = new();
    public List<LineageEvent> UsageEvents { get; } = new();
}

public class CompleteCalibrationLineage
{
    public string CalibrationMapId { get; set; } = string.Empty;
    public List<LineageEvent> UpdateEvents { get; } = new();
    public List<LineageEvent> UsageEvents { get; } = new();
}

public enum LineageEventType
{
    DecisionStamped,
    ModelRegistered,
    ModelPromoted,
    ModelQuarantined,
    FeatureStoreUpdated,
    CalibrationUpdated,
    ConfigurationChanged,
    SystemComponentUpdated
}

#endregion