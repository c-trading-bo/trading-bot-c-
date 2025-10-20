using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Training Debug Logger - Phase 14: Debugging & Diagnostics Tools
/// Provides verbose logging and detailed metrics when debug mode is enabled
/// Enabled via LAB_DEBUG_MODE=1 environment variable
/// </summary>
internal sealed class TrainingDebugLogger
{
    private readonly ILogger<TrainingDebugLogger> _logger;
    private readonly bool _debugMode;
    private readonly bool _traceDataMode;
    private readonly string _debugOutputDir;

    public TrainingDebugLogger(ILogger<TrainingDebugLogger> logger)
    {
        _logger = logger;
        
        // Check environment variables
        _debugMode = Environment.GetEnvironmentVariable("LAB_DEBUG_MODE") == "1";
        _traceDataMode = Environment.GetEnvironmentVariable("LAB_TRACE_DATA") == "1" || _debugMode;

        _debugOutputDir = Path.Combine(
            Directory.GetCurrentDirectory(),
            "artifacts",
            "debug");

        if (_debugMode || _traceDataMode)
        {
            Directory.CreateDirectory(_debugOutputDir);
            _logger.LogInformation("[DEBUG] Debug logging ENABLED (DebugMode: {Debug}, TraceData: {Trace})",
                _debugMode, _traceDataMode);
        }
    }

    /// <summary>
    /// Log verbose details before component training
    /// Phase 14.2: Verbose Training Logging
    /// </summary>
    public void LogBeforeComponent(
        string componentName,
        string phase,
        int componentIndex,
        int totalComponents)
    {
        if (!_debugMode) return;

        _logger.LogInformation("[DEBUG] ═══════════════════════════════════════════════════════");
        _logger.LogInformation("[DEBUG] Starting Component: {Component}", componentName);
        _logger.LogInformation("[DEBUG] Phase: {Phase}, Index: {Index}/{Total}", phase, componentIndex, totalComponents);
        
        // Log current memory state with enhanced profiling
        LogMemoryStatistics("Before " + componentName);

        // Log disk space
        var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
        if (Directory.Exists(dataPath))
        {
            var drive = new System.IO.DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
            var freeSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
            _logger.LogInformation("[DEBUG] Disk Space: {Free:F1} GB available", freeSpaceGB);
        }

        // Log GC collections
        _logger.LogInformation("[DEBUG] GC Collections: Gen0={Gen0}, Gen1={Gen1}, Gen2={Gen2}",
            GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2));
    }

    /// <summary>
    /// Log verbose details during component training
    /// Phase 14.2: Verbose Training Logging
    /// </summary>
    public void LogDuringTraining(
        string componentName,
        int epoch,
        int totalEpochs,
        double loss,
        double learningRate)
    {
        if (!_debugMode) return;

        _logger.LogInformation("[DEBUG] {Component} - Epoch {Epoch}/{Total}: Loss={Loss:F6}, LR={LR:F8}",
            componentName, epoch, totalEpochs, loss, learningRate);
    }

    /// <summary>
    /// Log verbose details after component training
    /// Phase 14.2: Verbose Training Logging
    /// </summary>
    public void LogAfterComponent(
        string componentName,
        bool success,
        TimeSpan duration,
        ComponentDebugMetrics? metrics = null)
    {
        if (!_debugMode) return;

        var status = success ? "✓ SUCCESS" : "✗ FAILED";
        _logger.LogInformation("[DEBUG] Completed Component: {Component} - {Status}", componentName, status);
        _logger.LogInformation("[DEBUG] Duration: {Duration:F2}s", duration.TotalSeconds);

        if (metrics != null)
        {
            _logger.LogInformation("[DEBUG] Final Loss: {Loss:F6}", metrics.FinalLoss);
            _logger.LogInformation("[DEBUG] Best Loss: {Loss:F6} (Epoch {Epoch})", 
                metrics.BestLoss, metrics.BestEpoch);
            _logger.LogInformation("[DEBUG] Total Epochs: {Epochs}", metrics.TotalEpochs);
            _logger.LogInformation("[DEBUG] Model Size: {Size:F2} MB", metrics.ModelSizeMB);
        }

        // Log final memory state with enhanced profiling
        LogMemoryStatistics("After " + componentName);

        _logger.LogInformation("[DEBUG] ═══════════════════════════════════════════════════════");
    }

    /// <summary>
    /// Log data pipeline trace information
    /// Phase 14.2: Verbose Training Logging
    /// </summary>
    public void LogDataTrace(string operation, Dictionary<string, object> details)
    {
        if (!_traceDataMode) return;

        _logger.LogDebug("[DATA-TRACE] {Operation}: {Details}",
            operation, JsonSerializer.Serialize(details));
    }

    /// <summary>
    /// Log training metrics to JSON file
    /// Phase 14.4: Model Training Metrics Logger
    /// </summary>
    public async Task LogTrainingMetricsAsync(
        string sessionId,
        string componentName,
        TrainingMetrics metrics,
        CancellationToken cancellationToken = default)
    {
        if (!_debugMode) return;

        try
        {
            var metricsFile = Path.Combine(
                _debugOutputDir,
                $"training-metrics-{sessionId}-{componentName}.json");

            var json = JsonSerializer.Serialize(metrics, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            await File.WriteAllTextAsync(metricsFile, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogDebug("[DEBUG] Metrics saved: {File}", Path.GetFileName(metricsFile));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DEBUG] Failed to save training metrics: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Check if debug mode is enabled
    /// </summary>
    public bool IsDebugEnabled => _debugMode;

    /// <summary>
    /// Check if data tracing is enabled
    /// </summary>
    public bool IsDataTraceEnabled => _traceDataMode;
    
    /// <summary>
    /// Log detailed memory statistics for profiling
    /// Phase 14: Memory Profiling Enhancement
    /// </summary>
    private void LogMemoryStatistics(string context)
    {
        if (!_debugMode) return;

        try
        {
            var gcInfo = GC.GetGCMemoryInfo();
            
            // Managed memory
            var managedMemoryGB = GC.GetTotalMemory(forceFullCollection: false) / (1024.0 * 1024.0 * 1024.0);
            var heapSizeGB = gcInfo.HeapSizeBytes / (1024.0 * 1024.0 * 1024.0);
            var fragmentedBytes = gcInfo.FragmentedBytes;
            var fragmentedMB = fragmentedBytes / (1024.0 * 1024.0);
            
            // Total available memory
            var memoryLoadGB = gcInfo.MemoryLoadBytes / (1024.0 * 1024.0 * 1024.0);
            var totalMemoryGB = gcInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            var memoryPressure = (memoryLoadGB / totalMemoryGB) * 100;
            
            _logger.LogInformation(
                "[DEBUG-MEMORY] {Context}: Managed={ManagedGB:F3}GB, Heap={HeapGB:F3}GB, " +
                "Fragmented={FragmentedMB:F1}MB, Pressure={Pressure:F1}%",
                context,
                managedMemoryGB,
                heapSizeGB,
                fragmentedMB,
                memoryPressure);
            
            // GC pressure indicators
            var gen0Count = GC.CollectionCount(0);
            var gen1Count = GC.CollectionCount(1);
            var gen2Count = GC.CollectionCount(2);
            
            // Calculate GC pressure (high Gen2 collections indicate memory pressure)
            if (gen2Count > 100)
            {
                _logger.LogWarning(
                    "[DEBUG-MEMORY] High GC pressure detected: Gen2 collections = {Gen2Count}",
                    gen2Count);
            }
            
            // Check for memory fragmentation
            if (fragmentedMB > 100)
            {
                _logger.LogWarning(
                    "[DEBUG-MEMORY] Significant memory fragmentation: {FragmentedMB:F1} MB",
                    fragmentedMB);
            }
            
            // Process memory breakdown
            using var process = System.Diagnostics.Process.GetCurrentProcess();
            var workingSetGB = process.WorkingSet64 / (1024.0 * 1024.0 * 1024.0);
            var privateMemoryGB = process.PrivateMemorySize64 / (1024.0 * 1024.0 * 1024.0);
            var unmanagedGB = privateMemoryGB - managedMemoryGB;
            
            _logger.LogInformation(
                "[DEBUG-MEMORY] Process: WorkingSet={WorkingSetGB:F3}GB, " +
                "Private={PrivateGB:F3}GB, Unmanaged~{UnmanagedGB:F3}GB",
                workingSetGB,
                privateMemoryGB,
                Math.Max(0, unmanagedGB));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DEBUG-MEMORY] Failed to log memory statistics: {Error}", ex.Message);
        }
    }
}

/// <summary>
/// Component debug metrics for logging
/// </summary>
internal class ComponentDebugMetrics
{
    public double FinalLoss { get; set; }
    public double BestLoss { get; set; }
    public int BestEpoch { get; set; }
    public int TotalEpochs { get; set; }
    public double ModelSizeMB { get; set; }
}

/// <summary>
/// Training metrics for detailed logging
/// Phase 14.4: Model Training Metrics Logger
/// </summary>
internal class TrainingMetrics
{
    public string ComponentName { get; set; } = string.Empty;
    public string SessionId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public double DurationSeconds { get; set; }
    
    // Training metrics
    public int TotalEpochs { get; set; }
    public double InitialLoss { get; set; }
    public double FinalLoss { get; set; }
    public double BestLoss { get; set; }
    public int BestEpoch { get; set; }
    public List<double> LossHistory { get; set; } = new();
    public double AverageLoss { get; set; }
    
    // Learning rate
    public double InitialLearningRate { get; set; }
    public double FinalLearningRate { get; set; }
    
    // Model info
    public long ModelParameterCount { get; set; }
    public double ModelSizeMB { get; set; }
    public string ModelArchitecture { get; set; } = string.Empty;
    
    // Data info
    public int TrainingSamples { get; set; }
    public int ValidationSamples { get; set; }
    public int BatchSize { get; set; }
    
    // Resource usage
    public double PeakMemoryGB { get; set; }
    public double AverageCpuPercent { get; set; }
    public bool UsedGpu { get; set; }
    public string GpuType { get; set; } = "None";
    
    // Convergence metrics
    public bool Converged { get; set; }
    public int EpochsToConverge { get; set; }
    public double ConvergenceThreshold { get; set; }
    
    // Performance metrics
    public double SamplesPerSecond { get; set; }
    public double TimePerEpochSeconds { get; set; }
    public double TimePerBatchMs { get; set; }
    
    // Issues encountered
    public List<string> Warnings { get; set; } = new();
    public List<string> Errors { get; set; } = new();
}
