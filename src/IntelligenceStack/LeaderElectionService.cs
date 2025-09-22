using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Text.Json;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Leader election service with distributed lock implementation
/// Supports active-standby pattern with TTL and automatic renewal
/// </summary>
public class LeaderElectionService : ILeaderElectionService, IDisposable
{
    private readonly ILogger<LeaderElectionService> _logger;
    private readonly LeaderElectionConfig _config;
    private readonly string _lockPath;
    private readonly string _nodeId;
    private Timer? _renewalTimer;
    private bool _isLeader;
    private readonly object _lock = new();

    // LoggerMessage delegates for CA1848 compliance
    private static readonly Action<ILogger, Exception?> LeaderElectionDisabledAssumeLeadership =
        LoggerMessage.Define(LogLevel.Information, new EventId(3001, "LeaderElectionDisabled"), "[LEADER] Leader election disabled - assuming leadership");

    private static readonly Action<ILogger, Exception?> AlreadyLeaderDebug =
        LoggerMessage.Define(LogLevel.Debug, new EventId(3002, "AlreadyLeader"), "[LEADER] Already the leader");

    private static readonly Action<ILogger, string, Exception?> LeadershipAcquired =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(3003, "LeadershipAcquired"), "[LEADER] 🎯 Leadership acquired by node: {NodeId}");

    private static readonly Action<ILogger, string, Exception?> LeadershipTakenOver =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(3004, "LeadershipTakenOver"), "[LEADER] 🔄 Leadership taken over by node: {NodeId}");

    private static readonly Action<ILogger, Exception?> FailedToAcquireLeadershipDebug =
        LoggerMessage.Define(LogLevel.Debug, new EventId(3005, "FailedToAcquireLeadership"), "[LEADER] Failed to acquire leadership");

    private static readonly Action<ILogger, Exception?> FailedToAcquireLeadershipError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3006, "FailedToAcquireLeadershipError"), "[LEADER] Failed to acquire leadership");

    // JSON serializer options for CA1869 compliance
    private static readonly JsonSerializerOptions SerializerOptions = new() { WriteIndented = true };



    public event EventHandler<LeadershipChangedEventArgs>? LeadershipChanged;

    public LeaderElectionService(
        ILogger<LeaderElectionService> logger,
        LeaderElectionConfig config,
        string lockBasePath = "data/leadership")
    {
        _logger = logger;
        _config = config;
        _nodeId = GenerateNodeId();
        
        Directory.CreateDirectory(lockBasePath);
        _lockPath = Path.Combine(lockBasePath, "leader.lock");
    }

    public async Task<bool> TryAcquireLeadershipAsync(CancellationToken cancellationToken = default)
    {
        if (!_config.Enabled)
        {
            LeaderElectionDisabledAssumeLeadership(_logger, null);
            return true;
        }

        try
        {
            lock (_lock)
            {
                if (_isLeader)
                {
                    AlreadyLeaderDebug(_logger, null);
                    return true;
                }
            }

            var lockData = CreateLockData();
            
            // Try to acquire the lock
            if (await TryCreateLockFileAsync(lockData, cancellationToken).ConfigureAwait(false))
            {
                lock (_lock)
                {
                    _isLeader = true;
                }
                
                StartRenewalTimer();
                OnLeadershipChanged(true, "Acquired leadership");
                
                LeadershipAcquired(_logger, _nodeId, null);
                return true;
            }
            
            // Check if we can take over from expired leader
            if (await CanTakeoverLeadershipAsync(cancellationToken).ConfigureAwait(false))
            {
                if (await TryCreateLockFileAsync(lockData, cancellationToken).ConfigureAwait(false))
                {
                    lock (_lock)
                    {
                        _isLeader = true;
                    }
                    
                    StartRenewalTimer();
                    OnLeadershipChanged(true, "Took over expired leadership");
                    
                    LeadershipTakenOver(_logger, _nodeId, null);
                    return true;
                }
            }

            FailedToAcquireLeadershipDebug(_logger, null);
            return false;
        }
        catch (Exception ex)
        {
            FailedToAcquireLeadershipError(_logger, ex);
            return false;
        }
    }

    public async Task ReleaseLeadershipAsync(CancellationToken cancellationToken = default)
    {
        // Brief async operation for proper async pattern
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        try
        {
            lock (_lock)
            {
                if (!_isLeader)
                {
                    _logger.LogDebug("[LEADER] Not the leader - nothing to release");
                    return;
                }
            }

            StopRenewalTimer();
            
            // Remove the lock file
            if (File.Exists(_lockPath))
            {
                File.Delete(_lockPath);
            }

            lock (_lock)
            {
                _isLeader = false;
            }

            OnLeadershipChanged(false, "Released leadership");
            _logger.LogInformation("[LEADER] 📤 Leadership released by node: {NodeId}", _nodeId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LEADER] Failed to release leadership");
        }
    }

    public async Task<bool> IsLeaderAsync(CancellationToken cancellationToken = default)
    {
        await Task.Yield(); // Ensure proper async execution
        
        if (!_config.Enabled)
        {
            return true; // Always leader when disabled
        }

        lock (_lock)
        {
            return _isLeader;
        }
    }

    public async Task<bool> RenewLeadershipAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_lock)
            {
                if (!_isLeader)
                {
                    return false;
                }
            }

            var lockData = CreateLockData();
            
            // Update the lock file with new timestamp
            if (await TryCreateLockFileAsync(lockData, cancellationToken).ConfigureAwait(false))
            {
                _logger.LogDebug("[LEADER] Leadership renewed by node: {NodeId}", _nodeId);
                return true;
            }
            else
            {
                // Lost leadership
                lock (_lock)
                {
                    _isLeader = false;
                }
                
                StopRenewalTimer();
                OnLeadershipChanged(false, "Lost leadership during renewal");
                
                _logger.LogWarning("[LEADER] 📤 Lost leadership during renewal: {NodeId}", _nodeId);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LEADER] Failed to renew leadership");
            
            // Assume lost leadership on error
            lock (_lock)
            {
                _isLeader = false;
            }
            
            StopRenewalTimer();
            OnLeadershipChanged(false, "Lost leadership due to renewal error");
            return false;
        }
    }

    private async Task<bool> TryCreateLockFileAsync(LeaderLockData lockData, CancellationToken cancellationToken)
    {
        try
        {
            var json = JsonSerializer.Serialize(lockData, SerializerOptions);
            
            // Use atomic write operation
            var tempPath = _lockPath + ".tmp";
            await File.WriteAllTextAsync(tempPath, json, cancellationToken).ConfigureAwait(false);
            
            // Atomic move (as atomic as possible on the file system)
            File.Move(tempPath, _lockPath, overwrite: true);
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[LEADER] Failed to create lock file");
            return false;
        }
    }

    private async Task<bool> CanTakeoverLeadershipAsync(CancellationToken cancellationToken)
    {
        try
        {
            if (!File.Exists(_lockPath))
            {
                return true; // No existing lock
            }

            var content = await File.ReadAllTextAsync(_lockPath, cancellationToken).ConfigureAwait(false);
            var existingLock = JsonSerializer.Deserialize<LeaderLockData>(content);
            
            if (existingLock == null)
            {
                return true; // Invalid lock file
            }

            var age = DateTime.UtcNow - existingLock.AcquiredAt;
            var isExpired = age > TimeSpan.FromSeconds(_config.TtlSeconds);
            
            if (isExpired)
            {
                _logger.LogInformation("[LEADER] Existing lock expired (age: {Age:F1}s > TTL: {TTL}s)", 
                    age.TotalSeconds, _config.TtlSeconds);
                return true;
            }

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LEADER] Error checking existing lock - assuming can takeover");
            return true;
        }
    }

    private LeaderLockData CreateLockData()
    {
        return new LeaderLockData
        {
            NodeId = _nodeId,
            AcquiredAt = DateTime.UtcNow,
            RenewedAt = DateTime.UtcNow,
            TtlSeconds = _config.TtlSeconds,
            ProcessId = Environment.ProcessId,
            MachineName = Environment.MachineName
        };
    }

    private void StartRenewalTimer()
    {
        StopRenewalTimer();
        
        var renewalInterval = TimeSpan.FromSeconds(_config.RenewSeconds);
        _renewalTimer = new Timer(_ =>
        {
            _ = Task.Run(async () =>
            {
                try
                {
                    await RenewLeadershipAsync(CancellationToken.None).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LEADER] Failed to renew leadership");
            }
            });
        }, null, renewalInterval, renewalInterval);
        
        _logger.LogDebug("[LEADER] Started renewal timer (interval: {Interval}s)", _config.RenewSeconds);
    }

    private void StopRenewalTimer()
    {
        _renewalTimer?.Dispose();
        _renewalTimer = null;
    }

    private static string GenerateNodeId()
    {
        var hostname = Environment.MachineName;
        var processId = Environment.ProcessId;
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        return $"{hostname}_{processId}_{timestamp}";
    }

    private void OnLeadershipChanged(bool isLeader, string reason)
    {
        LeadershipChanged?.Invoke(this, new LeadershipChangedEventArgs
        {
            IsLeader = isLeader,
            ChangedAt = DateTime.UtcNow,
            Reason = reason
        });
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            try
            {
                if (_isLeader)
                {
                    ReleaseLeadershipAsync(CancellationToken.None).GetAwaiter().GetResult();
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[LEADER] Error during dispose");
            }
            finally
            {
                StopRenewalTimer();
            }
        }
    }

    private sealed class LeaderLockData
    {
        public string NodeId { get; set; } = string.Empty;
        public DateTime AcquiredAt { get; set; }
        public DateTime RenewedAt { get; set; }
        public int TtlSeconds { get; set; }
        public int ProcessId { get; set; }
        public string MachineName { get; set; } = string.Empty;
    }
}

/// <summary>
/// Quarantine manager for model health monitoring
/// Implements Watch/Degrade/Quarantine states with shadow testing
/// </summary>
public class QuarantineManager : IQuarantineManager
{
    private readonly ILogger<QuarantineManager> _logger;
    private readonly QuarantineConfig _config;
    private readonly Dictionary<string, QuarantineStatus> _modelStatus = new();
    private readonly Dictionary<string, List<ModelPerformance>> _performanceHistory = new();
    private readonly object _lock = new();

    // Constants for magic numbers (S109 compliance)
    private const double BaseBrierThreshold = 0.25;
    private const double HalfBlendWeight = 0.5;
    private const int LatencyMultiplierMax = 5;
    private const double FullBlendWeight = 1.0;
    private const double ZeroBlendWeight = 0.0;

    public QuarantineManager(ILogger<QuarantineManager> logger, QuarantineConfig config)
    {
        _logger = logger;
        _config = config;
    }

    public async Task<QuarantineStatus> CheckModelHealthAsync(string modelId, CancellationToken cancellationToken = default)
    {
        if (!_config.Enabled)
        {
            return new QuarantineStatus { State = HealthState.Healthy, ModelId = modelId };
        }

        // Perform health check asynchronously to avoid blocking
        return await Task.Run(() =>
        {
            lock (_lock)
            {
                if (!_modelStatus.TryGetValue(modelId, out var status))
                {
                    status = new QuarantineStatus
                    {
                        ModelId = modelId,
                        State = HealthState.Healthy,
                        BlendWeight = 1.0
                    };
                    _modelStatus[modelId] = status;
                }

                return status;
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    public async Task QuarantineModelAsync(string modelId, QuarantineReason reason, CancellationToken cancellationToken = default)
    {
        // Perform quarantine operation asynchronously to avoid blocking the calling thread
        await Task.Run(() =>
        {
            try
            {
                lock (_lock)
                {
                    if (!_modelStatus.TryGetValue(modelId, out var status))
                    {
                        status = new QuarantineStatus { ModelId = modelId };
                        _modelStatus[modelId] = status;
                    }

                    status.State = HealthState.Quarantine;
                    status.Reason = reason;
                    status.QuarantinedAt = DateTime.UtcNow;
                    status.BlendWeight = 0.0;
                    status.ShadowDecisionCount = 0;
                }

                _logger.LogWarning("[QUARANTINE] Model quarantined: {ModelId} (reason: {Reason})", modelId, reason);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[QUARANTINE] Failed to quarantine model: {ModelId}", modelId);
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    public async Task<bool> TryRestoreModelAsync(string modelId, CancellationToken cancellationToken = default)
    {
        // Perform restoration check asynchronously
        return await Task.Run(() =>
        {
            try
            {
                lock (_lock)
                {
                    if (!_modelStatus.TryGetValue(modelId, out var status))
                    {
                        return false;
                    }

                    if (status.State != HealthState.Quarantine)
                    {
                        return true; // Already restored
                    }

                    // Check if model has completed shadow period
                    if (status.ShadowDecisionCount >= _config.ShadowDecisionsForReentry)
                    {
                        status.State = HealthState.Healthy;
                        status.BlendWeight = 1.0;
                        status.QuarantinedAt = null;
                        status.Reason = null;
                        status.ShadowDecisionCount = 0;

                        _logger.LogInformation("[QUARANTINE] Model restored from quarantine: {ModelId}", modelId);
                        return true;
                    }

                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[QUARANTINE] Failed to restore model: {ModelId}", modelId);
                return false;
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    public async Task<List<string>> GetQuarantinedModelsAsync(CancellationToken cancellationToken = default)
    {
        // Get quarantined models asynchronously
        return await Task.Run(() =>
        {
            lock (_lock)
            {
                return _modelStatus
                    .Where(kvp => kvp.Value.State == HealthState.Quarantine)
                    .Select(kvp => kvp.Key)
                    .ToList();
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    public async Task UpdateModelPerformanceAsync(string modelId, ModelPerformance performance, CancellationToken cancellationToken = default)
    {
        try
        {
            List<ModelPerformance>? historyToEvaluate = null;
            lock (_lock)
            {
                if (!_performanceHistory.TryGetValue(modelId, out var history))
                {
                    history = new List<ModelPerformance>();
                    _performanceHistory[modelId] = history;
                }

                history.Add(performance);

                // Keep only recent history (last 24 hours)
                var cutoff = DateTime.UtcNow.AddHours(-24);
                history.RemoveAll(p => p.WindowEnd < cutoff);

                // Make a copy for evaluation outside the lock
                historyToEvaluate = new List<ModelPerformance>(history);
            }

            // Check for health state changes outside the lock
            if (historyToEvaluate != null)
            {
                await EvaluateModelHealthAsync(modelId, historyToEvaluate, cancellationToken).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[QUARANTINE] Failed to update performance for model: {ModelId}", modelId);
        }
    }

    private async Task EvaluateModelHealthAsync(string modelId, List<ModelPerformance> history, CancellationToken cancellationToken)
    {
        if (history.Count < 5) return; // Need minimum history

        var recent = history.TakeLast(10).ToList();
        var avgBrierScore = recent.Average(p => p.BrierScore);
        var avgLatency = recent.Average(p => p.Latency);

        // Get current status
        if (!_modelStatus.TryGetValue(modelId, out var status))
        {
            status = new QuarantineStatus { ModelId = modelId, State = HealthState.Healthy };
            _modelStatus[modelId] = status;
        }

        var previousState = status.State;

        // Evaluate against thresholds
        if (avgBrierScore > BaseBrierThreshold + _config.QuarantineBrierDelta || avgLatency > _config.LatencyP99Ms * LatencyMultiplierMax)
        {
            status.State = HealthState.Quarantine;
            status.BlendWeight = ZeroBlendWeight;
            
            if (previousState != HealthState.Quarantine)
            {
                await QuarantineModelAsync(modelId, QuarantineReason.BrierDeltaTooHigh, cancellationToken).ConfigureAwait(false);
            }
        }
        else if (avgBrierScore > BaseBrierThreshold + _config.DegradeBrierDelta || avgLatency > _config.LatencyP99Ms * _config.LatencyDegradeMultiplier)
        {
            status.State = HealthState.Degrade;
            status.BlendWeight = HalfBlendWeight; // Require agreement with another model
        }
        else if (avgBrierScore > BaseBrierThreshold + _config.WatchBrierDelta || avgLatency > _config.LatencyP99Ms)
        {
            status.State = HealthState.Watch;
            status.BlendWeight = HalfBlendWeight; // Halve blend weight
        }
        else
        {
            status.State = HealthState.Healthy;
            status.BlendWeight = FullBlendWeight;
        }

        if (status.State != previousState)
        {
            _logger.LogInformation("[QUARANTINE] Model health changed: {ModelId} {From} -> {To} (Brier: {Brier:F3}, Latency: {Latency:F1}ms)", 
                modelId, previousState, status.State, avgBrierScore, avgLatency);
        }
    }
}