using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using System.Globalization;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Atomic model router - thread-safe, lock-free champion model access
/// </summary>
public class AtomicModelRouter<T> : IModelRouter<T> where T : class
{
    private readonly ILogger<AtomicModelRouter<T>> _logger;
    private readonly string _algorithm;
    private volatile T? _current;
    private volatile ModelVersion? _currentVersion;
    private volatile ModelRouterStats _stats;
    private readonly object _swapLock = new object();

    public AtomicModelRouter(ILogger<AtomicModelRouter<T>> logger, string algorithm)
    {
        _logger = logger;
        _algorithm = algorithm;
        _stats = new ModelRouterStats
        {
            Algorithm = algorithm,
            IsHealthy = false,
            LoadedAt = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Get the current champion model (read-only access)
    /// This is lock-free and safe for high-frequency live trading
    /// </summary>
    public T? Current => _current;

    /// <summary>
    /// Get the current champion version information
    /// </summary>
    public ModelVersion? CurrentVersion => _currentVersion;

    /// <summary>
    /// Atomically swap to a new champion model
    /// This is the ONLY way to update the model - no mutation allowed
    /// </summary>
    public Task<bool> SwapAsync(T newModel, ModelVersion newVersion, CancellationToken cancellationToken = default)
    {
        if (newModel == null)
        {
            _logger.LogError("[{Algorithm}] Cannot swap to null model", _algorithm);
            return Task.FromResult(false);
        }

        if (newVersion == null)
        {
            _logger.LogError("[{Algorithm}] Cannot swap with null version info", _algorithm);
            return Task.FromResult(false);
        }

        var swapStartTime = DateTime.UtcNow;
        
        try
        {
            // Use lock to ensure atomic swap
            lock (_swapLock)
            {
                var previousVersion = _currentVersion?.VersionId ?? "none";
                
                // Perform atomic swap
                _current = newModel;
                _currentVersion = newVersion;
                
                // Update stats
                _stats = new ModelRouterStats
                {
                    Algorithm = _algorithm,
                    CurrentVersionId = newVersion.VersionId,
                    LoadedAt = swapStartTime,
                    SwapCount = _stats.SwapCount + 1,
                    LastSwapAt = swapStartTime,
                    LastSwapDuration = DateTime.UtcNow - swapStartTime,
                    IsHealthy = true,
                    Metadata = new Dictionary<string, object>
                    {
                        ["PreviousVersionId"] = previousVersion,
                        ["SwapTimestamp"] = swapStartTime.ToString("O", CultureInfo.InvariantCulture),
                        ["ModelType"] = newVersion.ModelType,
                        ["SchemaVersion"] = newVersion.SchemaVersion
                    }
                };

                _logger.LogInformation("[{Algorithm}] Model swapped atomically: {PreviousVersion} → {NewVersion} in {Duration:F2}ms", 
                    _algorithm, previousVersion, newVersion.VersionId, _stats.LastSwapDuration.TotalMilliseconds);
                
                return Task.FromResult(true);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[{Algorithm}] Failed to swap model to version {VersionId}", _algorithm, newVersion.VersionId);
            
            // Update stats to reflect failure
            lock (_swapLock)
            {
                _stats = new ModelRouterStats
                { 
                    Algorithm = _stats.Algorithm,
                    CurrentVersionId = _stats.CurrentVersionId,
                    LoadedAt = _stats.LoadedAt,
                    SwapCount = _stats.SwapCount,
                    LastSwapAt = _stats.LastSwapAt,
                    IsHealthy = false,
                    LastSwapDuration = DateTime.UtcNow - swapStartTime,
                    Metadata = _stats.Metadata.ToDictionary(k => k.Key, v => v.Value)
                };
                _stats.Metadata["LastError"] = ex.Message;
                _stats.Metadata["LastErrorTime"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
            }
            
            return Task.FromResult(false);
        }
    }

    /// <summary>
    /// Get model loading statistics
    /// </summary>
    public Task<ModelRouterStats> GetStatsAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_stats);
    }

    /// <summary>
    /// Initialize with a champion model
    /// </summary>
    public Task<bool> InitializeAsync(T championModel, ModelVersion championVersion, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[{Algorithm}] Initializing with champion model version {VersionId}", _algorithm, championVersion.VersionId);
        return SwapAsync(championModel, championVersion, cancellationToken);
    }

    /// <summary>
    /// Check if router is ready for inference
    /// </summary>
    public bool IsReady => _current != null && _currentVersion != null && _stats.IsHealthy;
}

/// <summary>
/// Factory for creating and managing model routers
/// </summary>
public class ModelRouterFactory : IModelRouterFactory
{
    private readonly ILogger<ModelRouterFactory> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly ConcurrentDictionary<string, object> _routers = new();

    public ModelRouterFactory(ILogger<ModelRouterFactory> logger, IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    /// <summary>
    /// Create a model router for an algorithm
    /// </summary>
    public IModelRouter<T> CreateRouter<T>(string algorithm) where T : class
    {
        if (string.IsNullOrEmpty(algorithm))
        {
            throw new ArgumentException("Algorithm name cannot be null or empty", nameof(algorithm));
        }

        var key = $"{algorithm}_{typeof(T).Name}";
        
        return (IModelRouter<T>)_routers.GetOrAdd(key, _ =>
        {
            var logger = _serviceProvider.GetRequiredService<ILogger<AtomicModelRouter<T>>>();
            var router = new AtomicModelRouter<T>(logger, algorithm);
            
            _logger.LogInformation("Created model router for algorithm {Algorithm} with type {Type}", algorithm, typeof(T).Name);
            return router;
        });
    }

    /// <summary>
    /// Get an existing router
    /// </summary>
    public IModelRouter<T>? GetRouter<T>(string algorithm) where T : class
    {
        var key = $"{algorithm}_{typeof(T).Name}";
        return _routers.TryGetValue(key, out var router) ? (IModelRouter<T>)router : null;
    }

    /// <summary>
    /// Get all active routers
    /// </summary>
    public IReadOnlyDictionary<string, object> GetAllRouters()
    {
        return _routers.ToDictionary(k => k.Key, v => v.Value);
    }

    /// <summary>
    /// Get router health status for all algorithms
    /// </summary>
    public async Task<Dictionary<string, bool>> GetAllRouterHealthAsync()
    {
        var health = new Dictionary<string, bool>();
        
        foreach (var kvp in _routers)
        {
            try
            {
                if (kvp.Value is IModelRouter<object> router)
                {
                    var stats = await router.GetStatsAsync().ConfigureAwait(false);
                    health[kvp.Key] = stats.IsHealthy;
                }
                else
                {
                    health[kvp.Key];
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to get health for router {RouterKey}", kvp.Key);
                health[kvp.Key];
            }
        }
        
        return health;
    }
}