using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Infra;

/// <summary>
/// COMPONENT 6: MEMORY LEAK PREVENTION IN ML PIPELINE
/// Manages ML model memory usage, prevents leaks, and provides automatic cleanup
/// </summary>
public class MLMemoryManager : IDisposable
{
    private readonly ILogger<MLMemoryManager> _logger;
    private readonly ConcurrentDictionary<string, ModelVersion> _activeModels = new();
    private readonly Queue<ModelVersion> _modelHistory = new();
    private Timer? _garbageCollector;
    private Timer? _memoryMonitor;
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

    // Parameterless constructor for dependency injection
    public MLMemoryManager() : this(null) { }

    public MLMemoryManager(ILogger<MLMemoryManager>? logger)
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<MLMemoryManager>.Instance;
    }

    public async Task InitializeMemoryManagement()
    {
        try
        {
            _logger.LogInformation("[MLMemory] Initializing ML Memory Management System");

            // Start garbage collection timer
            _garbageCollector = new Timer(CollectGarbage, null, TimeSpan.Zero, TimeSpan.FromMinutes(5));

            // Start memory monitoring
            _memoryMonitor = new Timer(MonitorMemory, null, TimeSpan.Zero, TimeSpan.FromSeconds(30));

            // Setup memory pressure notifications
            GC.RegisterForFullGCNotification(10, 10);
            StartGCMonitoring();

            _logger.LogInformation("[MLMemory] Memory management initialized successfully");
            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MLMemory] Failed to initialize memory management");
            throw;
        }
    }

    public async Task<T?> LoadModel<T>(string modelPath, string version) where T : class
    {
        var modelId = Path.GetFileNameWithoutExtension(modelPath);
        var versionKey = $"{modelId}_{version}";

        try
        {
            // Check if model already loaded
            if (_activeModels.TryGetValue(versionKey, out var existing))
            {
                existing.UsageCount++;
                existing.LastUsed = DateTime.UtcNow;
                _logger.LogDebug("[MLMemory] Reusing existing model: {ModelId}", modelId);
                return existing.Model as T;
            }

            // Check memory before loading
            await EnsureMemoryAvailable();

            // Load model (mock implementation for testing)
            var model = await LoadModelFromDisk<T>(modelPath);

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
            modelVersion.MemoryFootprint = Math.Max(0, GC.GetTotalMemory(false) - memoryBefore);

            _activeModels[versionKey] = modelVersion;
            _modelHistory.Enqueue(modelVersion);

            // Cleanup old versions
            await CleanupOldVersions(modelId);

            _logger.LogInformation("[MLMemory] Loaded model {ModelId} v{Version}, memory: {Memory}MB",
                modelId, version, modelVersion.MemoryFootprint / 1024 / 1024);

            return model;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MLMemory] Failed to load model {ModelPath}", modelPath);
            throw;
        }
    }

    private async Task<T?> LoadModelFromDisk<T>(string modelPath) where T : class
    {
        // Mock implementation - in real system would load actual ML model
        await Task.Delay(100); // Simulate loading time
        
        // For testing purposes, return a mock object
        if (typeof(T) == typeof(string))
            return $"MockModel_{Path.GetFileName(modelPath)}" as T;
        
        return Activator.CreateInstance<T>();
    }

    private async Task EnsureMemoryAvailable()
    {
        var currentMemory = GC.GetTotalMemory(false);

        if (currentMemory > MAX_MEMORY_BYTES * 0.8)
        {
            _logger.LogWarning("[MLMemory] High memory usage detected: {Memory}MB", currentMemory / 1024 / 1024);

            // Aggressive cleanup
            await AggressiveCleanup();

            // Force GC
            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true);

            // Recheck
            currentMemory = GC.GetTotalMemory(false);

            if (currentMemory > MAX_MEMORY_BYTES * 0.9)
            {
                var message = $"ML memory limit reached: {currentMemory / 1024 / 1024}MB";
                _logger.LogError("[MLMemory] {Message}", message);
                throw new OutOfMemoryException(message);
            }
        }
    }

    private async Task CleanupOldVersions(string modelId)
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

                    _logger.LogInformation("[MLMemory] Removed old model version: {Key}", key);
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

            // Clear training data caches (mock implementation)
            ClearTrainingDataCache();

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
                _logger.LogInformation("[MLMemory] Garbage collection freed {FreedMemory}MB", freedMemory);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MLMemory] Garbage collection failed");
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
                ModelMemory = new Dictionary<string, long>(),
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
                _logger.LogCritical("[MLMemory] CRITICAL: Memory usage at {Percentage:F1}%", memoryPercentage);
                Task.Run(async () => await AggressiveCleanup());
            }
            else if (memoryPercentage > 75)
            {
                _logger.LogWarning("[MLMemory] High memory usage: {Percentage:F1}%", memoryPercentage);
            }

            // Log snapshot periodically
            if (DateTime.UtcNow.Minute % 5 == 0)
            {
                _logger.LogInformation("[MLMemory] Memory snapshot - Models: {Count}, ML Memory: {MLMemory}MB, Total: {Total}MB",
                    snapshot.LoadedModels, snapshot.MLMemory / 1024 / 1024, snapshot.TotalMemory / 1024 / 1024);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MLMemory] Memory monitoring failed");
        }
    }

    private async Task AggressiveCleanup()
    {
        _logger.LogWarning("[MLMemory] Starting aggressive memory cleanup");

        try
        {
            // 1. Clear all prediction caches
            ClearAllCaches();

            // 2. Unload least recently used models
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

            // 3. Clear training queues
            ClearTrainingQueues();

            // 4. Force immediate GC
            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true);

            _logger.LogInformation("[MLMemory] Aggressive cleanup completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MLMemory] Aggressive cleanup failed");
        }

        await Task.CompletedTask;
    }

    private void StartGCMonitoring()
    {
        Task.Run(() =>
        {
            try
            {
                while (!_disposed)
                {
                    GCNotificationStatus status = GC.WaitForFullGCApproach();
                    if (status == GCNotificationStatus.Succeeded)
                    {
                        _logger.LogDebug("[MLMemory] Full GC approaching - preparing cleanup");
                        ClearNonEssentialData();
                    }

                    status = GC.WaitForFullGCComplete();
                    if (status == GCNotificationStatus.Succeeded)
                    {
                        _logger.LogDebug("[MLMemory] Full GC completed");
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MLMemory] GC monitoring failed");
            }
        });
    }

    public void CompressModel(object model)
    {
        // Mock implementation for model compression
        if (model is INeuralNetwork network)
        {
            // Quantization - reduce precision
            network.Quantize(8); // 8-bit quantization

            // Pruning - remove small weights
            network.Prune(0.01); // Remove weights < 0.01

            // Knowledge distillation
            network.Distill();

            _logger.LogInformation("[MLMemory] Model compressed successfully");
        }
    }

    // Mock methods for testing
    private void ClearTrainingDataCache() { /* Mock implementation */ }
    private void ClearAllCaches() { /* Mock implementation */ }
    private void ClearTrainingQueues() { /* Mock implementation */ }
    private void ClearNonEssentialData() { /* Mock implementation */ }

    public MemorySnapshot GetMemorySnapshot()
    {
        var snapshot = new MemorySnapshot
        {
            TotalMemory = GC.GetTotalMemory(false),
            UsedMemory = Process.GetCurrentProcess().WorkingSet64,
            LoadedModels = _activeModels.Count,
            ModelMemory = _activeModels.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.MemoryFootprint
            )
        };

        snapshot.MLMemory = snapshot.ModelMemory.Values.Sum();
        return snapshot;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            
            _garbageCollector?.Dispose();
            _memoryMonitor?.Dispose();

            // Cleanup all models
            foreach (var model in _activeModels.Values)
            {
                if (model.Model is IDisposable disposable)
                    disposable.Dispose();
            }
            
            _activeModels.Clear();
            
            _logger.LogInformation("[MLMemory] ML Memory Manager disposed");
        }
    }
}

// Mock interface for testing neural network compression
public interface INeuralNetwork
{
    void Quantize(int bits);
    void Prune(double threshold);
    void Distill();
}