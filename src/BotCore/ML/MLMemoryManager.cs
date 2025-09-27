using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Collections.ObjectModel;
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
    private readonly OnnxModelLoader _onnxLoader;
    private readonly ConcurrentDictionary<string, ModelVersion> _activeModels = new();
    private readonly Queue<ModelVersion> _modelHistory = new();
    private readonly Timer _garbageCollector;
    private readonly Timer _memoryMonitor;
    private readonly object _lockObject = new();
    
    private const long MAX_MEMORY_BYTES = 8L * 1024 * 1024 * 1024; // 8GB
    private const int MAX_MODEL_VERSIONS = 3;
    private bool _disposed;

    internal class ModelVersion
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
        public Dictionary<string, long> ModelMemory { get; } = new();
        public int LoadedModels { get; set; }
        public int CachedPredictions { get; set; }
        private readonly List<string> _memoryLeaks = new();
        public IReadOnlyList<string> MemoryLeaks => _memoryLeaks;
        
        internal void AddMemoryLeak(string leak) => _memoryLeaks.Add(leak);
    }

    public MLMemoryManager(ILogger<MLMemoryManager> logger, OnnxModelLoader onnxLoader)
    {
        _logger = logger;
        _onnxLoader = onnxLoader;
        
        // Initialize timers
        _garbageCollector = new Timer(CollectGarbage, null, Timeout.Infinite, Timeout.Infinite);
        _memoryMonitor = new Timer(MonitorMemory, null, Timeout.Infinite, Timeout.Infinite);
        
        _logger.LogInformation("[ML-Memory] MLMemoryManager initialized with real ONNX loader");
    }

    /// <summary>
    /// Initialize memory management timers and monitoring
    /// </summary>
    public Task InitializeMemoryManagementAsync()
    {
        _logger.LogInformation("[ML-Memory] Starting memory management services");
        
        // Start garbage collection timer (every 5 minutes)
        _garbageCollector.Change(TimeSpan.Zero, TimeSpan.FromMinutes(5));
        
        // Start memory monitoring (every 30 seconds)  
        _memoryMonitor.Change(TimeSpan.Zero, TimeSpan.FromSeconds(30));
        
        // Setup memory pressure notifications
        GC.RegisterForFullGCNotification(10, 10);
        _ = Task.Run(StartGCMonitoring);

        return Task.CompletedTask;
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
        await EnsureMemoryAvailableAsync().ConfigureAwait(false);
        
        try
        {
            // Load real ONNX model using OnnxModelLoader
            var model = await LoadModelFromDiskAsync<T>(modelPath).ConfigureAwait(false);
            
            if (model == null)
            {
                _logger.LogWarning("[ML-Memory] Failed to load model from: {ModelPath}", modelPath);
                return null;
            }
            
            // Measure memory footprint using memory pressure monitoring instead of forced GC
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
            
            // Monitor memory pressure instead of forcing collection
            var memoryAfter = GC.GetTotalMemory(false);
            modelVersion.MemoryFootprint = Math.Max(0, memoryAfter - memoryBefore);
            
            // Register for memory pressure notifications
            if (memoryAfter > MAX_MEMORY_BYTES * 0.7)
            {
                _logger.LogWarning("[ML-Memory] Memory pressure detected ({MemoryMB:F1}MB), will monitor for cleanup", 
                    memoryAfter / 1024.0 / 1024.0);
            }
            
            _activeModels[versionKey] = modelVersion;
            lock (_lockObject)
            {
                _modelHistory.Enqueue(modelVersion);
            }
            
            // Cleanup old versions
            await CleanupOldVersionsAsync(modelId).ConfigureAwait(false);
            
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
    /// Load ONNX models using professional Microsoft.ML.OnnxRuntime integration
    /// </summary>
    private async Task<T?> LoadModelFromDiskAsync<T>(string modelPath) where T : class
    {
        try
        {
            _logger.LogInformation("[ML-Memory] Loading ONNX model: {ModelPath}", modelPath);
            
            if (!File.Exists(modelPath))
            {
                _logger.LogWarning("[ML-Memory] Model file not found: {ModelPath}", modelPath);
                return null;
            }

            // Load ONNX model using professional OnnxModelLoader with validation
            var session = await _onnxLoader.LoadModelAsync(modelPath, validateInference: true).ConfigureAwait(false);
            
            if (session == null)
            {
                _logger.LogError("[ML-Memory] Failed to load ONNX model: {ModelPath}", modelPath);
                return null;
            }

            // Log model metadata for verification
            var inputCount = session.InputMetadata?.Count ?? 0;
            var outputCount = session.OutputMetadata?.Count ?? 0;
            _logger.LogInformation("[ML-Memory] ONNX model loaded successfully: {ModelPath}, Inputs: {InputCount}, Outputs: {OutputCount}", 
                modelPath, inputCount, outputCount);
            
            // Return the InferenceSession with proper type casting
            return session as T;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ML-Memory] Error loading ONNX model: {ModelPath}", modelPath);
            return null;
        }
    }

    private async Task EnsureMemoryAvailableAsync()
    {
        var currentMemory = GC.GetTotalMemory(false);
        
        if (currentMemory > MAX_MEMORY_BYTES * 0.8)
        {
            _logger.LogWarning("[ML-Memory] Memory usage high ({MemoryMB:F1}MB), starting intelligent cleanup", 
                currentMemory / 1024.0 / 1024.0);
                
            // Intelligent cleanup based on usage patterns
            await PerformIntelligentCleanupAsync().ConfigureAwait(false);
            
            // Wait for memory pressure to reduce naturally
            var maxWaitTime = TimeSpan.FromSeconds(10);
            var startTime = DateTime.UtcNow;
            
            while (DateTime.UtcNow - startTime < maxWaitTime)
            {
                currentMemory = GC.GetTotalMemory(false);
                if (currentMemory <= MAX_MEMORY_BYTES * 0.75)
                    break;
                    
                await Task.Delay(500).ConfigureAwait(false);
            }
            
            // Only as last resort, suggest collection to runtime
            if (currentMemory > MAX_MEMORY_BYTES * 0.9)
            {
                _logger.LogCritical("[ML-Memory] Memory critically high, suggesting collection to runtime");
                GC.Collect(0, GCCollectionMode.Optimized, false); // Gentle suggestion only
                await Task.Delay(1000).ConfigureAwait(false); // Give runtime time to respond
                
                currentMemory = GC.GetTotalMemory(false);
                if (currentMemory > MAX_MEMORY_BYTES * 0.95)
                {
                    var memoryMB = currentMemory / 1024 / 1024;
                    throw new InvalidOperationException($"ML memory limit reached: {memoryMB}MB, cannot continue safely");
                }
            }
        }
    }
    
    /// <summary>
    /// Perform intelligent cleanup based on usage patterns instead of forced GC
    /// </summary>
    private async Task PerformIntelligentCleanupAsync()
    {
        await Task.Yield();
        
        // Remove models that haven't been used recently and have low usage counts
        var candidatesForRemoval = _activeModels.Values
            .Where(m => DateTime.UtcNow - m.LastUsed > TimeSpan.FromMinutes(10) && m.UsageCount < 5)
            .OrderBy(m => m.UsageCount)
            .ThenBy(m => m.LastUsed)
            .Take(_activeModels.Count / 3) // Remove up to 1/3 of unused models
            .ToList();
        
        foreach (var model in candidatesForRemoval)
        {
            var key = $"{model.ModelId}_{model.Version}";
            if (_activeModels.TryRemove(key, out var removed))
            {
                if (removed.Model is IDisposable disposable)
                {
                    disposable.Dispose();
                }
                removed.Model = null;
                _logger.LogDebug("[ML-Memory] Removed unused model: {Key}", key);
            }
        }
        
        // Clear any weak references that are no longer alive
        var deadReferences = _activeModels.Values
            .Where(m => m.WeakRef?.IsAlive == false)
            .ToList();
            
        foreach (var deadRef in deadReferences)
        {
            var key = $"{deadRef.ModelId}_{deadRef.Version}";
            _activeModels.TryRemove(key, out _);
        }
        
        _logger.LogInformation("[ML-Memory] Intelligent cleanup completed - removed {RemovedCount} unused models, {DeadRefs} dead references", 
            candidatesForRemoval.Count, deadReferences.Count);
    }

    private Task CleanupOldVersionsAsync(string modelId)
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

        return Task.CompletedTask;
    }

    private void CollectGarbage(object? state)
    {
        try
        {
            var beforeMemory = GC.GetTotalMemory(false);
            
            // Remove unused models based on intelligent criteria
            var unusedModels = _activeModels.Values
                .Where(m => DateTime.UtcNow - m.LastUsed > TimeSpan.FromMinutes(30) && m.UsageCount == 0)
                .ToList();
            
            var longUnusedModels = _activeModels.Values
                .Where(m => DateTime.UtcNow - m.LastUsed > TimeSpan.FromHours(2))
                .ToList();
                
            var modelsToRemove = unusedModels.Concat(longUnusedModels).Distinct().ToList();
            
            foreach (var model in modelsToRemove)
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
            
            // Only suggest collection if memory pressure is actually high
            var currentMemory = GC.GetTotalMemory(false);
            if (currentMemory > MAX_MEMORY_BYTES * 0.8)
            {
                // Use memory pressure APIs instead of forcing collection
                if (GC.TryStartNoGCRegion(1024 * 1024)) // Try to prevent GC for 1MB allocations
                {
                    // Do critical work if needed
                    GC.EndNoGCRegion();
                }
                
                // Gentle suggestion to runtime - not forced
                GC.Collect(0, GCCollectionMode.Optimized, false);
            }
            
            var afterMemory = GC.GetTotalMemory(false);
            var freedMemory = (beforeMemory - afterMemory) / 1024 / 1024;
            
            if (freedMemory > 50 || modelsToRemove.Count > 0) // Log if meaningful cleanup occurred
            {
                _logger.LogInformation("[ML-Memory] Intelligent cleanup freed {FreedMB}MB, removed {ModelCount} models", 
                    freedMemory, modelsToRemove.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ML-Memory] Intelligent cleanup failed");
        }
    }

    private void MonitorMemory(object? state)
    {
        try
        {
            var snapshot = new MemorySnapshot
            {
                TotalMemory = GC.GetTotalMemory(false),
                UsedMemory = Process.GetCurrentProcess().WorkingSet64
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
                    snapshot.AddMemoryLeak($"Potential leak: {model.ModelId} still in memory");
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

    private Task AggressiveCleanupAsync()
    {
        _logger.LogWarning("[ML-Memory] Starting intelligent emergency cleanup");
        
        // Remove least recently used models with priority on memory footprint
        var modelsToUnload = _activeModels.Values
            .OrderBy(m => m.LastUsed)
            .ThenByDescending(m => m.MemoryFootprint) // Prioritize large models
            .Take(Math.Max(1, _activeModels.Count / 2)) // Remove up to half
            .ToList();
        
        var totalFreed = 0L;
        foreach (var model in modelsToUnload)
        {
            var key = $"{model.ModelId}_{model.Version}";
            if (_activeModels.TryRemove(key, out var removed))
            {
                totalFreed += removed.MemoryFootprint;
                if (removed.Model is IDisposable disposable)
                {
                    disposable.Dispose();
                }
                removed.Model = null;
            }
        }
        
        // Clean up model history queue
        lock (_lockObject)
        {
            while (_modelHistory.Count > 5) // Keep only 5 recent entries
            {
                _modelHistory.Dequeue();
            }
        }
        
        // Suggest LOH compaction only if we freed significant memory
        if (totalFreed > 100 * 1024 * 1024) // 100MB
        {
            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect(2, GCCollectionMode.Optimized, false); // Gentle suggestion
        }
        
        _logger.LogInformation("[ML-Memory] Emergency cleanup completed - freed ~{FreedMB}MB from {ModelCount} models", 
            totalFreed / 1024 / 1024, modelsToUnload.Count);
        return Task.CompletedTask;
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
                    
                    await Task.Delay(1000).ConfigureAwait(false); // Brief pause
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
            LoadedModels = _activeModels.Count
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
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _logger.LogInformation("[ML-Memory] Disposing MLMemoryManager");
                
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
            }
            _disposed = true;
        }
    }
}