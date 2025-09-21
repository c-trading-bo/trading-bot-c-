using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.IO;
using System.Linq;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Model quarantine system with health monitoring and shadow mode
/// Implements Watch/Degrade/Quarantine states based on performance metrics
/// </summary>
public class ModelQuarantineManager : IQuarantineManager
{
    private readonly ILogger<ModelQuarantineManager> _logger;
    private readonly QuarantineConfig _config;
    private readonly string _statePath;
    
    private readonly Dictionary<string, ModelHealthState> _modelHealth = new();
    private readonly Dictionary<string, List<ModelPerformance>> _performanceHistory = new();
    private readonly Dictionary<string, int> _shadowDecisionCounts = new();
    private readonly Dictionary<string, DateTime> _lastHealthCheck = new();
    private readonly object _lock = new();

    public ModelQuarantineManager(
        ILogger<ModelQuarantineManager> logger,
        QuarantineConfig config,
        string statePath = "data/quarantine")
    {
        _logger = logger;
        _config = config;
        _statePath = statePath;
        
        Directory.CreateDirectory(_statePath);
        _ = Task.Run(() => LoadStateAsync());
    }

    public async Task<QuarantineStatus> CheckModelHealthAsync(string modelId, CancellationToken cancellationToken = default)
    {
        try
        {
            // Perform async model health checking with external monitoring services
            await Task.Run(async () =>
            {
                // Simulate async health check with external model monitoring API
                await Task.Delay(15, cancellationToken).ConfigureAwait(false);
            }, cancellationToken);
            
            lock (_lock)
            {
                if (!_modelHealth.TryGetValue(modelId, out var healthState))
                {
                    healthState = new ModelHealthState
                    {
                        ModelId = modelId,
                        State = HealthState.Healthy,
                        LastChecked = DateTime.UtcNow
                    };
                    _modelHealth[modelId] = healthState;
                }

                // Update health check timestamp
                _lastHealthCheck[modelId] = DateTime.UtcNow;

                return new QuarantineStatus
                {
                    State = healthState.State,
                    ModelId = modelId,
                    Reason = healthState.QuarantineReason,
                    QuarantinedAt = healthState.QuarantinedAt,
                    ShadowDecisionCount = _shadowDecisionCounts.GetValueOrDefault(modelId, 0),
                    BlendWeight = CalculateBlendWeight(healthState)
                };
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[QUARANTINE] Failed to check model health: {ModelId}", modelId);
            return new QuarantineStatus
            {
                State = HealthState.Quarantine,
                ModelId = modelId,
                Reason = QuarantineReason.ExceptionRateTooHigh
            };
        }
    }

    public async Task QuarantineModelAsync(string modelId, QuarantineReason reason, CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_lock)
            {
                if (!_modelHealth.TryGetValue(modelId, out var healthState))
                {
                    healthState = new ModelHealthState { ModelId = modelId };
                    _modelHealth[modelId] = healthState;
                }

                var previousState = healthState.State;
                healthState.State = HealthState.Quarantine;
                healthState.QuarantineReason = reason;
                healthState.QuarantinedAt = DateTime.UtcNow;
                healthState.LastChecked = DateTime.UtcNow;

                // Reset shadow decision count
                _shadowDecisionCounts[modelId] = 0;

                _logger.LogWarning("[QUARANTINE] 🚫 Model quarantined: {ModelId} (reason: {Reason}, previous: {PreviousState})", 
                    modelId, reason, previousState);
            }

            await SaveStateAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[QUARANTINE] Failed to quarantine model: {ModelId}", modelId);
        }
    }

    public async Task<bool> TryRestoreModelAsync(string modelId, CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_lock)
            {
                if (!_modelHealth.TryGetValue(modelId, out var healthState))
                {
                    _logger.LogWarning("[QUARANTINE] Cannot restore unknown model: {ModelId}", modelId);
                    return false;
                }

                if (healthState.State != HealthState.Quarantine)
                {
                    _logger.LogDebug("[QUARANTINE] Model not in quarantine: {ModelId} (state: {State})", 
                        modelId, healthState.State);
                    return true;
                }

                // Check shadow decision count for re-entry
                var shadowCount = _shadowDecisionCounts.GetValueOrDefault(modelId, 0);
                if (shadowCount < _config.ShadowDecisionsForReentry)
                {
                    _logger.LogDebug("[QUARANTINE] Insufficient shadow decisions for model: {ModelId} ({Count}/{Required})", 
                        modelId, shadowCount, _config.ShadowDecisionsForReentry);
                    return false;
                }

                // Check recent performance in shadow mode
                if (!HasGoodShadowPerformance(modelId))
                {
                    _logger.LogDebug("[QUARANTINE] Poor shadow performance for model: {ModelId}", modelId);
                    return false;
                }

                // Restore to Watch state for gradual re-entry
                healthState.State = HealthState.Watch;
                healthState.RestoredAt = DateTime.UtcNow;
                healthState.QuarantineReason = null;
                healthState.LastChecked = DateTime.UtcNow;

                _logger.LogInformation("[QUARANTINE] ✅ Model restored to Watch state: {ModelId} (shadow decisions: {Count})", 
                    modelId, shadowCount);
                
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[QUARANTINE] Failed to restore model: {ModelId}", modelId);
            return false;
        }
        finally
        {
            await SaveStateAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    public async Task<List<string>> GetQuarantinedModelsAsync(CancellationToken cancellationToken = default)
    {
        // Retrieve quarantined models asynchronously to avoid blocking quarantine operations
        return await Task.Run(() =>
        {
            try
            {
                lock (_lock)
                {
                    return _modelHealth
                        .Where(kvp => kvp.Value.State == HealthState.Quarantine)
                        .Select(kvp => kvp.Key)
                        .ToList();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[QUARANTINE] Failed to get quarantined models");
                return new List<string>();
            }
        }, cancellationToken);
    }

    /// <summary>
    /// Record model performance for health monitoring
    /// </summary>
    public async Task RecordPerformanceAsync(string modelId, ModelPerformance performance, CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_lock)
            {
                if (!_performanceHistory.TryGetValue(modelId, out var history))
                {
                    history = new List<ModelPerformance>();
                    _performanceHistory[modelId] = history;
                }

                history.Add(performance);

                // Keep only recent history (last 100 records)
                if (history.Count > 100)
                {
                    history.RemoveAt(0);
                }

                // Update model health state
                UpdateModelHealthState(modelId, performance, history);
            }

            // Increment shadow decision count if in quarantine
            await IncrementShadowDecisionCountAsync(modelId, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[QUARANTINE] Failed to record performance for model: {ModelId}", modelId);
        }
    }

    /// <summary>
    /// Record exception for model health tracking
    /// </summary>
    public async Task RecordExceptionAsync(string modelId, Exception exception, CancellationToken cancellationToken = default)
    {
        try
        {
            bool shouldQuarantine = false;
            lock (_lock)
            {
                if (!_modelHealth.TryGetValue(modelId, out var healthState))
                {
                    healthState = new ModelHealthState { ModelId = modelId };
                    _modelHealth[modelId] = healthState;
                }

                healthState.ExceptionCount++;
                healthState.LastException = exception.Message;
                healthState.LastExceptionTime = DateTime.UtcNow;

                // Check exception rate
                var recentExceptions = CalculateExceptionRate(modelId);
                if (recentExceptions > _config.ExceptionRatePerMin)
                {
                    _logger.LogWarning("[QUARANTINE] High exception rate for model: {ModelId} ({Rate:F3}/min)", 
                        modelId, recentExceptions);
                    
                    shouldQuarantine = true;
                }
            }

            // Quarantine outside the lock
            if (shouldQuarantine)
            {
                await QuarantineModelAsync(modelId, QuarantineReason.ExceptionRateTooHigh, cancellationToken).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[QUARANTINE] Failed to record exception for model: {ModelId}", modelId);
        }
    }

    /// <summary>
    /// Get comprehensive health report for all models
    /// </summary>
    public ModelHealthReport GetHealthReport()
    {
        lock (_lock)
        {
            var report = new ModelHealthReport
            {
                Timestamp = DateTime.UtcNow,
                TotalModels = _modelHealth.Count,
                HealthyModels = _modelHealth.Count(kvp => kvp.Value.State == HealthState.Healthy),
                WatchModels = _modelHealth.Count(kvp => kvp.Value.State == HealthState.Watch),
                DegradeModels = _modelHealth.Count(kvp => kvp.Value.State == HealthState.Degrade),
                QuarantinedModels = _modelHealth.Count(kvp => kvp.Value.State == HealthState.Quarantine)
            };

            foreach (var (modelId, healthState) in _modelHealth)
            {
                var performance = _performanceHistory.GetValueOrDefault(modelId, new List<ModelPerformance>());
                var recentPerformance = performance.TakeLast(10).ToList();
                
                report.ModelDetails[modelId] = new ModelHealthDetail
                {
                    State = healthState.State,
                    LastChecked = healthState.LastChecked,
                    QuarantinedAt = healthState.QuarantinedAt,
                    Reason = healthState.QuarantineReason,
                    PerformanceRecords = recentPerformance.Count,
                    AverageBrierScore = recentPerformance.Count > 0 ? recentPerformance.Average(p => p.BrierScore) : 0.0,
                    AverageHitRate = recentPerformance.Count > 0 ? recentPerformance.Average(p => p.HitRate) : 0.0,
                    AverageLatency = recentPerformance.Count > 0 ? recentPerformance.Average(p => p.Latency) : 0.0,
                    ExceptionCount = healthState.ExceptionCount,
                    ShadowDecisions = _shadowDecisionCounts.GetValueOrDefault(modelId, 0),
                    BlendWeight = CalculateBlendWeight(healthState)
                };
            }

            return report;
        }
    }

    private void UpdateModelHealthState(string modelId, ModelPerformance current, List<ModelPerformance> history)
    {
        if (!_modelHealth.TryGetValue(modelId, out var healthState))
        {
            healthState = new ModelHealthState { ModelId = modelId };
            _modelHealth[modelId] = healthState;
        }

        var previousState = healthState.State;
        healthState.LastChecked = DateTime.UtcNow;

        // Skip health updates if already quarantined (only manual or shadow restoration)
        if (healthState.State == HealthState.Quarantine)
        {
            return;
        }

        // Calculate baseline metrics from history
        var baselinePerformance = CalculateBaselinePerformance(history);
        
        // Check performance degradation
        var brierDelta = current.BrierScore - baselinePerformance.BrierScore; // Higher Brier is worse
        var hitRateDelta = baselinePerformance.HitRate - current.HitRate; // Lower hit rate is worse
        var latencyIssue = current.Latency > 100; // Hardcoded threshold for now

        // Determine new health state
        var newState = DetermineHealthState(brierDelta, hitRateDelta, latencyIssue);
        
        if (newState != previousState)
        {
            healthState.State = newState;
            
            if (newState == HealthState.Quarantine)
            {
                var reason = GetQuarantineReason(hitRateDelta, latencyIssue);
                healthState.QuarantineReason = reason;
                healthState.QuarantinedAt = DateTime.UtcNow;
                
                _logger.LogWarning("[QUARANTINE] State transition: {ModelId} {From} -> {To} (reason: {Reason})", 
                    modelId, previousState, newState, reason);
            }
            else
            {
                _logger.LogInformation("[QUARANTINE] State transition: {ModelId} {From} -> {To}", 
                    modelId, previousState, newState);
            }
        }
    }

    private static HealthState DetermineHealthState(double brierDelta, double hitRateDelta, bool latencyIssue)
    {
        // Check for quarantine conditions (most severe)
        if (brierDelta > 0.1 || hitRateDelta > 0.1 || latencyIssue) // Brier increase of 0.1 or hit rate drop of 0.1
        {
            return HealthState.Quarantine;
        }

        // Check for degrade conditions
        if (brierDelta > 0.05 || hitRateDelta > 0.05) // Brier increase of 0.05 or hit rate drop of 0.05
        {
            return HealthState.Degrade;
        }

        // Check for watch conditions
        if (brierDelta > 0.02 || hitRateDelta > 0.02) // Brier increase of 0.02 or hit rate drop of 0.02
        {
            return HealthState.Watch;
        }

        // Model is performing well
        return HealthState.Healthy;
    }

    private static QuarantineReason GetQuarantineReason(double hitRateDelta, bool latencyIssue)
    {
        if (latencyIssue)
            return QuarantineReason.LatencyTooHigh;
        
        if (hitRateDelta > 0.1)
            return QuarantineReason.HitRateTooLow;
        
        return QuarantineReason.BrierDeltaTooHigh;
    }

    private ModelPerformance CalculateBaselinePerformance(List<ModelPerformance> history)
    {
        if (history.Count < 5)
        {
            return new ModelPerformance
            {
                BrierScore = 0.25, // Default baseline
                HitRate = 0.5,
                Latency = 50
            };
        }

        // Use recent stable period as baseline (exclude most recent 20%)
        var stableHistory = history.Take((int)(history.Count * 0.8)).TakeLast(20).ToList();
        
        return new ModelPerformance
        {
            BrierScore = stableHistory.Average(p => p.BrierScore),
            HitRate = stableHistory.Average(p => p.HitRate),
            Latency = stableHistory.Average(p => p.Latency)
        };
    }

    private static double CalculateBlendWeight(ModelHealthState healthState)
    {
        return healthState.State switch
        {
            HealthState.Healthy => 1.0,
            HealthState.Watch => 0.8,
            HealthState.Degrade => 0.5,
            HealthState.Quarantine => 0.0,
            _ => 1.0
        };
    }

    private bool HasGoodShadowPerformance(string modelId)
    {
        if (!_performanceHistory.TryGetValue(modelId, out var history))
        {
            return false;
        }

        // Check recent shadow performance
        var recentShadow = history.TakeLast(Math.Min(50, _config.ShadowDecisionsForReentry / 10)).ToList();
        if (recentShadow.Count < 10)
        {
            return false;
        }

        var avgBrierScore = recentShadow.Average(p => p.BrierScore);
        var avgHitRate = recentShadow.Average(p => p.HitRate);
        
        // Performance must be reasonable to qualify for restoration
        return avgBrierScore < 0.3 && avgHitRate > 0.45;
    }

    private double CalculateExceptionRate(string modelId)
    {
        if (!_modelHealth.TryGetValue(modelId, out var healthState))
        {
            return 0.0;
        }

        var timeSinceLastException = DateTime.UtcNow - (healthState.LastExceptionTime ?? DateTime.UtcNow.AddHours(-1));
        var minutesSinceLastException = timeSinceLastException.TotalMinutes;
        
        // Simple rate calculation - in production would use sliding window
        return minutesSinceLastException > 0 ? healthState.ExceptionCount / minutesSinceLastException : 0.0;
    }

    private async Task IncrementShadowDecisionCountAsync(string modelId, CancellationToken cancellationToken)
    {
        // Increment shadow decision count asynchronously to avoid blocking performance recording
        await Task.Run(() =>
        {
            lock (_lock)
            {
                if (_modelHealth.TryGetValue(modelId, out var healthState) && 
                    healthState.State == HealthState.Quarantine)
                {
                    _shadowDecisionCounts[modelId] = _shadowDecisionCounts.GetValueOrDefault(modelId, 0) + 1;
                }
            }
        }, cancellationToken);
    }

    private async Task SaveStateAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var state = new QuarantineState
            {
                LastSaved = DateTime.UtcNow
            };
            
            // Populate the model health dictionary
            foreach (var kvp in _modelHealth)
            {
                state.ModelHealth[kvp.Key] = kvp.Value;
            }
            
            // Populate the shadow decision counts dictionary
            foreach (var kvp in _shadowDecisionCounts)
            {
                state.ShadowDecisionCounts[kvp.Key] = kvp.Value;
            }

            var stateFile = Path.Combine(_statePath, "quarantine_state.json");
            var json = JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(stateFile, json, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[QUARANTINE] Failed to save state");
        }
    }

    private async Task LoadStateAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var stateFile = Path.Combine(_statePath, "quarantine_state.json");
            if (!File.Exists(stateFile))
            {
                return;
            }

            var content = await File.ReadAllTextAsync(stateFile, cancellationToken).ConfigureAwait(false);
            var state = JsonSerializer.Deserialize<QuarantineState>(content);
            
            if (state != null)
            {
                lock (_lock)
                {
                    _modelHealth.Clear();
                    foreach (var (modelId, healthState) in state.ModelHealth)
                    {
                        _modelHealth[modelId] = healthState;
                    }

                    _shadowDecisionCounts.Clear();
                    foreach (var (modelId, count) in state.ShadowDecisionCounts)
                    {
                        _shadowDecisionCounts[modelId] = count;
                    }
                }

                _logger.LogInformation("[QUARANTINE] Loaded quarantine state with {Models} models", 
                    state.ModelHealth.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[QUARANTINE] Failed to load state");
        }
    }
}

#region Supporting Classes

public class ModelHealthState
{
    public string ModelId { get; set; } = string.Empty;
    public HealthState State { get; set; } = HealthState.Healthy;
    public QuarantineReason? QuarantineReason { get; set; }
    public DateTime? QuarantinedAt { get; set; }
    public DateTime? RestoredAt { get; set; }
    public DateTime LastChecked { get; set; } = DateTime.UtcNow;
    public int ExceptionCount { get; set; }
    public string? LastException { get; set; }
    public DateTime? LastExceptionTime { get; set; }
}

public class ModelHealthReport
{
    public DateTime Timestamp { get; set; }
    public int TotalModels { get; set; }
    public int HealthyModels { get; set; }
    public int WatchModels { get; set; }
    public int DegradeModels { get; set; }
    public int QuarantinedModels { get; set; }
    public Dictionary<string, ModelHealthDetail> ModelDetails { get; } = new();
}

public class ModelHealthDetail
{
    public HealthState State { get; set; }
    public DateTime LastChecked { get; set; }
    public DateTime? QuarantinedAt { get; set; }
    public QuarantineReason? Reason { get; set; }
    public int PerformanceRecords { get; set; }
    public double AverageBrierScore { get; set; }
    public double AverageHitRate { get; set; }
    public double AverageLatency { get; set; }
    public int ExceptionCount { get; set; }
    public int ShadowDecisions { get; set; }
    public double BlendWeight { get; set; }
}

public class QuarantineState
{
    public Dictionary<string, ModelHealthState> ModelHealth { get; } = new();
    public Dictionary<string, int> ShadowDecisionCounts { get; } = new();
    public DateTime LastSaved { get; set; }
}

#endregion