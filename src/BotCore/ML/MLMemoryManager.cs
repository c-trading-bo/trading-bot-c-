using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime;

namespace BotCore.ML;

/// <summary>
/// ML Memory Management System for preventing memory leaks in the ML pipeline
/// Manages ML model lifecycle, memory monitoring, and automatic cleanup
/// </summary>
public class MLMemoryManager : IMLMemoryManager
{
    private readonly ILogger<MLMemoryManager> _logger;
    private readonly ConcurrentDictionary<string, ModelVersion> _activeModels = new();
    private readonly Queue<ModelVersion> _modelHistory = new();
    private readonly Timer _garbageCollector;
    private readonly Timer _memoryMonitor;
    private readonly object _lockObject = new();
    
    private const long MAX_MEMORY_BYTES = 8L * 1024 * 1024 * 1024; // 8GB
    private const int MAX_MODEL_VERSIONS = 3;
    private bool _disposed = false;

    public class ModelVersion
    {
        public string ModelId { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public object? Model { get; set; }
        public long MemoryFootprint { get; set; }
        public DateTime LoadedAt { get; set; }
        public int UsageCount { get; set; }
        public DateTime LastUsed { get; set; }
        public WeakReference? WeakRef { get; set; }
    }

    public class MemorySnapshot
    {
        public long TotalMemory { get; set; }
        public long UsedMemory { get; set; }
        public long MLMemory { get; set; }
        public Dictionary<string, long> ModelMemory { get; set; } = new();
        public int LoadedModels { get; set; }
        public int CachedPredictions { get; set; }
        public List<string> MemoryLeaks { get; set; } = new();
    }

    public MLMemoryManager(ILogger<MLMemoryManager> logger)
    {
        _logger = logger;
        
        // Initialize timers
        _garbageCollector = new Timer(CollectGarbage, null, Timeout.Infinite, Timeout.Infinite);
        _memoryMonitor = new Timer(MonitorMemory, null, Timeout.Infinite, Timeout.Infinite);
        
        _logger.LogInformation("[ML-Memory] MLMemoryManager initialized");
    }

    /// <summary>
    /// Initialize memory management timers and monitoring
    /// </summary>
    public async Task InitializeMemoryManagementAsync()
    {
        _logger.LogInformation("[ML-Memory] Starting memory management services");
        
        // Start garbage collection timer (every 5 minutes)
        _garbageCollector.Change(TimeSpan.Zero, TimeSpan.FromMinutes(5));
        
        // Start memory monitoring (every 30 seconds)  
        _memoryMonitor.Change(TimeSpan.Zero, TimeSpan.FromSeconds(30));
        
        // Setup memory pressure notifications
        GC.RegisterForFullGCNotification(10, 10);
        _ = Task.Run(StartGCMonitoring);
        
        await Task.CompletedTask;
    }

    /// <summary>
    /// Load and manage ML model with memory tracking
    /// </summary>
    public async Task<T?> LoadModelAsync<T>(string modelPath, string version) where T : class
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));
            
        var modelId = Path.GetFileNameWithoutExtension(modelPath);
        var versionKey = $"{modelId}_{version}";
        
        // Check if model already loaded
        if (_activeModels.TryGetValue(versionKey, out var existing))
        {
            existing.UsageCount++;
            existing.LastUsed = DateTime.UtcNow;
            _logger.LogDebug("[ML-Memory] Reusing cached model: {ModelId}", modelId);
            return existing.Model as T;
        }
        
        // Check memory before loading
        await EnsureMemoryAvailableAsync();
        
        try
        {
            // Load model (placeholder - integrate with actual model loading)
            var model = await LoadModelFromDiskAsync<T>(modelPath);
            
            if (model == null)
            {
                _logger.LogWarning("[ML-Memory] Failed to load model from: {ModelPath}", modelPath);
                return null;
            }
            
            // Measure memory footprint
            var memoryBefore = GC.GetTotalMemory(false);
            var modelVersion = new ModelVersion
            {
                ModelId = modelId,
                Version = version,
                Model = model,
                LoadedAt = DateTime.UtcNow,
                UsageCount = 1,
                LastUsed = DateTime.UtcNow,
                WeakRef = new WeakReference(model)
            };
            
            GC.Collect(2, GCCollectionMode.Forced);
            var memoryAfter = GC.GetTotalMemory(false);
            modelVersion.MemoryFootprint = Math.Max(0, memoryAfter - memoryBefore);
            
            _activeModels[versionKey] = modelVersion;
            lock (_lockObject)
            {
                _modelHistory.Enqueue(modelVersion);
            }
            
            // Cleanup old versions
            await CleanupOldVersionsAsync(modelId);
            
            _logger.LogInformation("[ML-Memory] Loaded model: {ModelId} v{Version} ({MemoryMB:F1}MB)", 
                modelId, version, modelVersion.MemoryFootprint / 1024.0 / 1024.0);
            
            return model;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ML-Memory] Error loading model: {ModelPath}", modelPath);
            throw;
        }
    }

    /// <summary>
    /// Placeholder for actual model loading - integrate with existing ONNX loading
    /// </summary>
    private async Task<T?> LoadModelFromDiskAsync<T>(string modelPath) where T : class
    {
        // TODO: Integrate with existing ONNX model loading from StrategyMlModelManager
        await Task.Delay(100); // Simulate loading time
        
        if (!File.Exists(modelPath))
        {
            _logger.LogWarning("[ML-Memory] Model file not found: {ModelPath}", modelPath);
            return null;
        }
        
        // For now, return a placeholder - this should integrate with actual ONNX loading
        return Activator.CreateInstance<T>();
    }

    private async Task EnsureMemoryAvailableAsync()
    {
        var currentMemory = GC.GetTotalMemory(false);
        
        if (currentMemory > MAX_MEMORY_BYTES * 0.8)
        {
            _logger.LogWarning("[ML-Memory] Memory usage high ({MemoryMB:F1}MB), starting cleanup", 
                currentMemory / 1024.0 / 1024.0);
                
            // Aggressive cleanup
            await AggressiveCleanupAsync();
            
            // Force GC
            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true);
            
            // Recheck
            currentMemory = GC.GetTotalMemory(false);
            
            if (currentMemory > MAX_MEMORY_BYTES * 0.9)
            {
                var memoryMB = currentMemory / 1024 / 1024;
                throw new OutOfMemoryException($"ML memory limit reached: {memoryMB}MB");
            }
        }
    }

    private async Task CleanupOldVersionsAsync(string modelId)
    {
        var versions = _activeModels.Values
            .Where(m => m.ModelId == modelId)
            .OrderByDescending(m => m.Version)
            .ToList();
        
        if (versions.Count > MAX_MODEL_VERSIONS)
        {
            // Keep only recent versions
            var toRemove = versions.Skip(MAX_MODEL_VERSIONS);
            
            foreach (var version in toRemove)
            {
                var key = $"{version.ModelId}_{version.Version}";
                if (_activeModels.TryRemove(key, out var removed))
                {
                    // Dispose if IDisposable
                    if (removed.Model is IDisposable disposable)
                    {
                        disposable.Dispose();
                    }
                    
                    // Clear strong reference
                    removed.Model = null;
                    
                    _logger.LogInformation("[ML-Memory] Removed old model version: {Key}", key);
                }
            }
        }
        
        await Task.CompletedTask;
    }

    private void CollectGarbage(object? state)
    {
        try
        {
            var beforeMemory = GC.GetTotalMemory(false);
            
            // Remove unused models
            var unusedModels = _activeModels.Values
                .Where(m => DateTime.UtcNow - m.LastUsed > TimeSpan.FromMinutes(30))
                .ToList();
            
            foreach (var model in unusedModels)
            {
                var key = $"{model.ModelId}_{model.Version}";
                if (_activeModels.TryRemove(key, out var removed))
                {
                    if (removed.Model is IDisposable disposable)
                    {
                        disposable.Dispose();
                    }
                    removed.Model = null;
                }
            }
            
            // Compact large object heap
            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            
            // Collect garbage
            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true);
            
            var afterMemory = GC.GetTotalMemory(false);
            var freedMemory = (beforeMemory - afterMemory) / 1024 / 1024;
            
            if (freedMemory > 100) // More than 100MB freed
            {
                _logger.LogInformation("[ML-Memory] Garbage collection freed {FreedMB}MB", freedMemory);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ML-Memory] Garbage collection failed");
        }
    }

    private void MonitorMemory(object? state)
    {
        try
        {
            var snapshot = new MemorySnapshot
            {
                TotalMemory = GC.GetTotalMemory(false),
                UsedMemory = Process.GetCurrentProcess().WorkingSet64,
                MemoryLeaks = new List<string>()
            };
            
            // Calculate ML memory usage
            long mlMemory = 0;
            foreach (var model in _activeModels.Values)
            {
                snapshot.ModelMemory[model.ModelId] = model.MemoryFootprint;
                mlMemory += model.MemoryFootprint;
                
                // Check for memory leaks
                if (model.WeakRef?.IsAlive == true && model.UsageCount == 0 && 
                    DateTime.UtcNow - model.LastUsed > TimeSpan.FromHours(1))
                {
                    snapshot.MemoryLeaks.Add($"Potential leak: {model.ModelId} still in memory");
                }
            }
            
            snapshot.MLMemory = mlMemory;
            snapshot.LoadedModels = _activeModels.Count;
            
            // Alert if memory usage is high
            var memoryPercentage = (double)snapshot.UsedMemory / MAX_MEMORY_BYTES * 100;
            
            if (memoryPercentage > 90)
            {
                _logger.LogCritical("[ML-Memory] CRITICAL: Memory usage at {MemoryPercentage:F1}%", memoryPercentage);
                _ = Task.Run(AggressiveCleanupAsync);
            }
            else if (memoryPercentage > 75)
            {
                _logger.LogWarning("[ML-Memory] High memory usage: {MemoryPercentage:F1}%", memoryPercentage);
            }
            
            // Log detailed snapshot every 5 minutes
            if (DateTime.UtcNow.Minute % 5 == 0)
            {
                LogMemorySnapshot(snapshot);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ML-Memory] Memory monitoring failed");
        }
    }

    private async Task AggressiveCleanupAsync()
    {
        _logger.LogWarning("[ML-Memory] Starting aggressive memory cleanup");
        
        // Unload least recently used models
        var modelsToUnload = _activeModels.Values
            .OrderBy(m => m.LastUsed)
            .Take(_activeModels.Count / 2)
            .ToList();
        
        foreach (var model in modelsToUnload)
        {
            var key = $"{model.ModelId}_{model.Version}";
            if (_activeModels.TryRemove(key, out var removed))
            {
                if (removed.Model is IDisposable disposable)
                {
                    disposable.Dispose();
                }
                removed.Model = null;
            }
        }
        
        // Force immediate GC
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true);
        
        _logger.LogInformation("[ML-Memory] Aggressive cleanup completed");
        await Task.CompletedTask;
    }

    private void StartGCMonitoring()
    {
        _ = Task.Run(async () =>
        {
            try
            {
                while (!_disposed)
                {
                    GCNotificationStatus status = GC.WaitForFullGCApproach();
                    if (status == GCNotificationStatus.Succeeded)
                    {
                        _logger.LogDebug("[ML-Memory] Full GC approaching - preparing cleanup");
                    }
                    
                    status = GC.WaitForFullGCComplete();
                    if (status == GCNotificationStatus.Succeeded)
                    {
                        _logger.LogDebug("[ML-Memory] Full GC completed");
                    }
                    
                    await Task.Delay(1000); // Brief pause
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ML-Memory] GC monitoring failed");
            }
        });
    }

    private void LogMemorySnapshot(MemorySnapshot snapshot)
    {
        _logger.LogInformation("[ML-Memory] Memory snapshot - Total: {TotalMB:F1}MB, ML: {MLMB:F1}MB, Models: {ModelCount}, Leaks: {LeakCount}",
            snapshot.TotalMemory / 1024.0 / 1024.0,
            snapshot.MLMemory / 1024.0 / 1024.0,
            snapshot.LoadedModels,
            snapshot.MemoryLeaks.Count);
            
        if (snapshot.MemoryLeaks.Any())
        {
            foreach (var leak in snapshot.MemoryLeaks)
            {
                _logger.LogWarning("[ML-Memory] {Leak}", leak);
            }
        }
    }

    /// <summary>
    /// Get current memory usage statistics
    /// </summary>
    public MemorySnapshot GetMemorySnapshot()
    {
        var snapshot = new MemorySnapshot
        {
            TotalMemory = GC.GetTotalMemory(false),
            UsedMemory = Process.GetCurrentProcess().WorkingSet64,
            LoadedModels = _activeModels.Count,
            MemoryLeaks = new List<string>()
        };
        
        long mlMemory = 0;
        foreach (var model in _activeModels.Values)
        {
            snapshot.ModelMemory[model.ModelId] = model.MemoryFootprint;
            mlMemory += model.MemoryFootprint;
        }
        snapshot.MLMemory = mlMemory;
        
        return snapshot;
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _logger.LogInformation("[ML-Memory] Disposing MLMemoryManager");
        
        _disposed = true;
        
        _garbageCollector?.Dispose();
        _memoryMonitor?.Dispose();
        
        // Cleanup all models
        foreach (var model in _activeModels.Values)
        {
            if (model.Model is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
        
        _activeModels.Clear();
        
        GC.SuppressFinalize(this);
    }
}