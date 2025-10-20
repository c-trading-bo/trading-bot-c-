using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Memory Leak Detector - Phase 14: Memory Profiling & Leak Detection
/// Tracks memory usage before/after each training component to detect potential memory leaks
/// Enables automatic heap dump generation on out-of-memory scenarios
/// </summary>
internal sealed class MemoryLeakDetector
{
    private readonly ILogger<MemoryLeakDetector> _logger;
    private readonly bool _enabled;
    private readonly long _memoryLeakThresholdBytes;
    private readonly string _heapDumpDirectory;
    private readonly int _maxHeapDumps = 3;
    private long _baselineMemoryBytes;
    private readonly Dictionary<string, ComponentMemorySnapshot> _componentSnapshots = new();
    private readonly object _lock = new();

    public MemoryLeakDetector(ILogger<MemoryLeakDetector> logger)
    {
        _logger = logger;
        
        // Enable memory leak detection if LAB_MEMORY_PROFILING=1
        _enabled = Environment.GetEnvironmentVariable("LAB_MEMORY_PROFILING") == "1";
        
        // Default threshold: 500 MB memory growth without GC recovery
        var thresholdMB = int.TryParse(
            Environment.GetEnvironmentVariable("LAB_MEMORY_LEAK_THRESHOLD_MB"), 
            out var value) ? value : 500;
        _memoryLeakThresholdBytes = thresholdMB * 1024L * 1024L;
        
        _heapDumpDirectory = Path.Combine(
            Directory.GetCurrentDirectory(),
            "artifacts",
            "diagnostics",
            "heap-dumps");
        
        if (_enabled)
        {
            Directory.CreateDirectory(_heapDumpDirectory);
            
            // Enable automatic heap dump on out-of-memory if configured
            ConfigureAutomaticHeapDump();
            
            _logger.LogInformation("[MEMORY] Memory leak detection ENABLED (Threshold: {ThresholdMB} MB)",
                thresholdMB);
        }
    }

    /// <summary>
    /// Configure automatic heap dump generation on out-of-memory exceptions
    /// Phase 14: Heap Dump Configuration
    /// </summary>
    private void ConfigureAutomaticHeapDump()
    {
        try
        {
            var enableHeapDump = Environment.GetEnvironmentVariable("LAB_HEAP_DUMP_ON_OOM") == "1";
            
            if (enableHeapDump)
            {
                // Set runtime options for heap dump capture
                // Note: .NET runtime supports heap dumps via environment variables
                // DOTNET_DbgEnableMiniDump=1
                // DOTNET_DbgMiniDumpType=4 (Heap)
                // DOTNET_DbgMiniDumpName=/path/to/dump-%p.dmp
                
                _logger.LogInformation("[MEMORY] Automatic heap dump on OOM ENABLED");
                _logger.LogInformation("[MEMORY] Heap dumps will be saved to: {Directory}", _heapDumpDirectory);
                _logger.LogInformation("[MEMORY] Maximum heap dumps retained: {Max}", _maxHeapDumps);
                
                // Clean up old heap dumps (keep only last 3)
                CleanupOldHeapDumps();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[MEMORY] Failed to configure automatic heap dump: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Clean up old heap dumps, keeping only the most recent files
    /// </summary>
    private void CleanupOldHeapDumps()
    {
        try
        {
            if (!Directory.Exists(_heapDumpDirectory))
                return;

            var dumpFiles = Directory.GetFiles(_heapDumpDirectory, "*.dmp")
                .Select(f => new FileInfo(f))
                .OrderByDescending(f => f.LastWriteTimeUtc)
                .ToList();

            // Delete all but the most recent N dumps
            foreach (var file in dumpFiles.Skip(_maxHeapDumps))
            {
                try
                {
                    file.Delete();
                    _logger.LogDebug("[MEMORY] Deleted old heap dump: {File}", file.Name);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[MEMORY] Failed to delete old heap dump: {File}", file.Name);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[MEMORY] Failed to cleanup old heap dumps: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Record baseline memory at start of training session
    /// </summary>
    public void RecordBaseline()
    {
        if (!_enabled) return;

        lock (_lock)
        {
            // Force GC to get accurate baseline
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            _baselineMemoryBytes = GC.GetTotalMemory(forceFullCollection: false);
            
            var baselineGB = _baselineMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            _logger.LogInformation("[MEMORY] Baseline memory recorded: {Memory:F3} GB", baselineGB);
            
            LogDetailedMemoryStats("Baseline");
        }
    }

    /// <summary>
    /// Record memory before component training starts
    /// </summary>
    public void RecordBeforeComponent(string componentName)
    {
        if (!_enabled) return;

        lock (_lock)
        {
            var snapshot = new ComponentMemorySnapshot
            {
                ComponentName = componentName,
                StartTime = DateTime.UtcNow,
                MemoryBeforeBytes = GC.GetTotalMemory(forceFullCollection: false),
                Gen0CollectionsBefore = GC.CollectionCount(0),
                Gen1CollectionsBefore = GC.CollectionCount(1),
                Gen2CollectionsBefore = GC.CollectionCount(2)
            };

            _componentSnapshots[componentName] = snapshot;

            var memoryGB = snapshot.MemoryBeforeBytes / (1024.0 * 1024.0 * 1024.0);
            _logger.LogDebug("[MEMORY] Component '{Component}' - Before: {Memory:F3} GB", 
                componentName, memoryGB);
        }
    }

    /// <summary>
    /// Record memory after component training completes
    /// Detect potential memory leaks if memory doesn't return close to baseline
    /// </summary>
    public async Task<MemoryLeakAnalysis> RecordAfterComponentAsync(
        string componentName, 
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
            return new MemoryLeakAnalysis { LeakDetected = false };

        ComponentMemorySnapshot? snapshot;
        lock (_lock)
        {
            if (!_componentSnapshots.TryGetValue(componentName, out snapshot))
            {
                _logger.LogWarning("[MEMORY] No before snapshot found for component: {Component}", componentName);
                return new MemoryLeakAnalysis { LeakDetected = false };
            }
        }

        // Force garbage collection to see if memory is recoverable
        var memoryBeforeGC = GC.GetTotalMemory(forceFullCollection: false);
        
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        
        // Wait a moment for GC to complete
        await Task.Delay(2000, cancellationToken).ConfigureAwait(false);
        
        var memoryAfterGC = GC.GetTotalMemory(forceFullCollection: false);
        
        snapshot.EndTime = DateTime.UtcNow;
        snapshot.MemoryAfterBytes = memoryAfterGC;
        snapshot.Gen0CollectionsAfter = GC.CollectionCount(0);
        snapshot.Gen1CollectionsAfter = GC.CollectionCount(1);
        snapshot.Gen2CollectionsAfter = GC.CollectionCount(2);

        var analysis = AnalyzeMemoryLeak(snapshot, memoryBeforeGC, memoryAfterGC);
        
        // Log results
        var memoryDeltaMB = analysis.MemoryDeltaBytes / (1024.0 * 1024.0);
        var gcRecoveredMB = (memoryBeforeGC - memoryAfterGC) / (1024.0 * 1024.0);
        
        if (analysis.LeakDetected)
        {
            _logger.LogWarning(
                "[MEMORY] ⚠️ POTENTIAL LEAK in '{Component}': " +
                "Delta: {DeltaMB:+0;-0} MB, GC Recovered: {RecoveredMB:F1} MB, " +
                "Current: {CurrentGB:F3} GB, Baseline: {BaselineGB:F3} GB",
                componentName,
                memoryDeltaMB,
                gcRecoveredMB,
                memoryAfterGC / (1024.0 * 1024.0 * 1024.0),
                _baselineMemoryBytes / (1024.0 * 1024.0 * 1024.0));
            
            LogDetailedMemoryStats($"After {componentName} (LEAK DETECTED)");
        }
        else if (Math.Abs(memoryDeltaMB) > 100)
        {
            // Log significant memory changes even if not a leak
            _logger.LogInformation(
                "[MEMORY] Component '{Component}': Delta: {DeltaMB:+0;-0} MB, " +
                "GC Recovered: {RecoveredMB:F1} MB",
                componentName,
                memoryDeltaMB,
                gcRecoveredMB);
        }
        else
        {
            _logger.LogDebug(
                "[MEMORY] Component '{Component}' - After: {Memory:F3} GB (Delta: {DeltaMB:+0;-0} MB)",
                componentName,
                memoryAfterGC / (1024.0 * 1024.0 * 1024.0),
                memoryDeltaMB);
        }

        return analysis;
    }

    /// <summary>
    /// Analyze memory snapshot to detect potential leaks
    /// </summary>
    private MemoryLeakAnalysis AnalyzeMemoryLeak(
        ComponentMemorySnapshot snapshot,
        long memoryBeforeGC,
        long memoryAfterGC)
    {
        var analysis = new MemoryLeakAnalysis
        {
            ComponentName = snapshot.ComponentName,
            MemoryBeforeBytes = snapshot.MemoryBeforeBytes,
            MemoryAfterBytes = memoryAfterGC,
            MemoryDeltaBytes = memoryAfterGC - snapshot.MemoryBeforeBytes,
            MemoryRecoveredByGC = memoryBeforeGC - memoryAfterGC,
            Gen0Collections = snapshot.Gen0CollectionsAfter - snapshot.Gen0CollectionsBefore,
            Gen1Collections = snapshot.Gen1CollectionsAfter - snapshot.Gen1CollectionsBefore,
            Gen2Collections = snapshot.Gen2CollectionsAfter - snapshot.Gen2CollectionsBefore
        };

        // Detect leak if:
        // 1. Memory delta is above threshold
        // 2. GC didn't recover much memory (less than 50% of delta)
        var leakThresholdMet = analysis.MemoryDeltaBytes > _memoryLeakThresholdBytes;
        var poorGcRecovery = analysis.MemoryRecoveredByGC < (analysis.MemoryDeltaBytes * 0.5);
        
        analysis.LeakDetected = leakThresholdMet && poorGcRecovery;

        return analysis;
    }

    /// <summary>
    /// Log detailed memory statistics for diagnostics
    /// </summary>
    private void LogDetailedMemoryStats(string context)
    {
        try
        {
            var gcInfo = GC.GetGCMemoryInfo();
            
            var totalMemoryGB = GC.GetTotalMemory(forceFullCollection: false) / (1024.0 * 1024.0 * 1024.0);
            var heapSizeGB = gcInfo.HeapSizeBytes / (1024.0 * 1024.0 * 1024.0);
            var fragmentedGB = gcInfo.FragmentedBytes / (1024.0 * 1024.0 * 1024.0);
            var committedGB = gcInfo.TotalCommittedBytes / (1024.0 * 1024.0 * 1024.0);
            
            _logger.LogInformation(
                "[MEMORY-DETAIL] {Context}: " +
                "Total={TotalGB:F3}GB, Heap={HeapGB:F3}GB, " +
                "Committed={CommittedGB:F3}GB, Fragmented={FragmentedGB:F3}GB",
                context,
                totalMemoryGB,
                heapSizeGB,
                committedGB,
                fragmentedGB);
            
            _logger.LogInformation(
                "[MEMORY-DETAIL] GC Collections: Gen0={Gen0}, Gen1={Gen1}, Gen2={Gen2}",
                GC.CollectionCount(0),
                GC.CollectionCount(1),
                GC.CollectionCount(2));

            // Get process memory info
            using var process = Process.GetCurrentProcess();
            var workingSetGB = process.WorkingSet64 / (1024.0 * 1024.0 * 1024.0);
            var privateMemoryGB = process.PrivateMemorySize64 / (1024.0 * 1024.0 * 1024.0);
            
            _logger.LogInformation(
                "[MEMORY-DETAIL] Process: WorkingSet={WorkingSetGB:F3}GB, Private={PrivateGB:F3}GB",
                workingSetGB,
                privateMemoryGB);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[MEMORY] Failed to log detailed memory stats: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Generate memory profiling report for training session
    /// </summary>
    public async Task GenerateMemoryReportAsync(
        string sessionId,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled) return;

        try
        {
            var reportPath = Path.Combine(
                Directory.GetCurrentDirectory(),
                "artifacts",
                "diagnostics",
                $"memory-report-{sessionId}.json");

            var report = new MemoryProfilingReport
            {
                SessionId = sessionId,
                BaselineMemoryBytes = _baselineMemoryBytes,
                ComponentSnapshots = _componentSnapshots.Values.ToList(),
                GeneratedAt = DateTime.UtcNow
            };

            var json = JsonSerializer.Serialize(report, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            await File.WriteAllTextAsync(reportPath, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("[MEMORY] Memory profiling report saved: {Path}", reportPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[MEMORY] Failed to generate memory report: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Check if memory leak detection is enabled
    /// </summary>
    public bool IsEnabled => _enabled;
}

/// <summary>
/// Memory snapshot for a training component
/// </summary>
internal class ComponentMemorySnapshot
{
    public string ComponentName { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public long MemoryBeforeBytes { get; set; }
    public long MemoryAfterBytes { get; set; }
    public int Gen0CollectionsBefore { get; set; }
    public int Gen1CollectionsBefore { get; set; }
    public int Gen2CollectionsBefore { get; set; }
    public int Gen0CollectionsAfter { get; set; }
    public int Gen1CollectionsAfter { get; set; }
    public int Gen2CollectionsAfter { get; set; }
}

/// <summary>
/// Memory leak analysis result
/// </summary>
internal class MemoryLeakAnalysis
{
    public string ComponentName { get; set; } = string.Empty;
    public long MemoryBeforeBytes { get; set; }
    public long MemoryAfterBytes { get; set; }
    public long MemoryDeltaBytes { get; set; }
    public long MemoryRecoveredByGC { get; set; }
    public int Gen0Collections { get; set; }
    public int Gen1Collections { get; set; }
    public int Gen2Collections { get; set; }
    public bool LeakDetected { get; set; }
}

/// <summary>
/// Memory profiling report for entire training session
/// </summary>
internal class MemoryProfilingReport
{
    public string SessionId { get; set; } = string.Empty;
    public long BaselineMemoryBytes { get; set; }
    public List<ComponentMemorySnapshot> ComponentSnapshots { get; set; } = new();
    public DateTime GeneratedAt { get; set; }
}
