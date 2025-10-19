using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Collects and exports training metrics for observability
/// Tracks resource usage, timing, and outcomes per training run
/// </summary>
internal sealed class TrainingMetricsCollector
{
    private readonly ILogger<TrainingMetricsCollector> _logger;
    private readonly string _metricsDirectory;
    private readonly Dictionary<string, Stopwatch> _activeTimers = new();
    private readonly Dictionary<string, object> _metrics = new();

    public TrainingMetricsCollector(ILogger<TrainingMetricsCollector> logger)
    {
        _logger = logger;
        _metricsDirectory = Path.Combine(Directory.GetCurrentDirectory(), "state", "metrics");
        Directory.CreateDirectory(_metricsDirectory);
    }

    /// <summary>
    /// Start tracking a training run
    /// </summary>
    public void StartRun(string runId)
    {
        _metrics.Clear();
        _metrics["RunId"] = runId;
        _metrics["StartTime"] = DateTime.UtcNow;
        StartTimer("TotalDuration");
        
        _logger.LogInformation("[METRICS] Started tracking run: {RunId}", runId);
    }

    /// <summary>
    /// End tracking a training run
    /// </summary>
    public void EndRun(bool success, string? errorMessage = null)
    {
        StopTimer("TotalDuration");
        _metrics["EndTime"] = DateTime.UtcNow;
        _metrics["Success"] = success;
        
        if (!success && errorMessage != null)
        {
            _metrics["ErrorMessage"] = errorMessage;
        }

        _logger.LogInformation("[METRICS] Ended tracking - Success: {Success}", success);
    }

    /// <summary>
    /// Start a timer for a specific metric
    /// </summary>
    public void StartTimer(string metricName)
    {
        if (_activeTimers.ContainsKey(metricName))
        {
            _logger.LogWarning("[METRICS] Timer {Metric} already running", metricName);
            return;
        }

        _activeTimers[metricName] = Stopwatch.StartNew();
    }

    /// <summary>
    /// Stop a timer and record the duration
    /// </summary>
    public void StopTimer(string metricName)
    {
        if (!_activeTimers.TryGetValue(metricName, out var stopwatch))
        {
            _logger.LogWarning("[METRICS] Timer {Metric} not found", metricName);
            return;
        }

        stopwatch.Stop();
        _metrics[$"{metricName}Seconds"] = stopwatch.Elapsed.TotalSeconds;
        _activeTimers.Remove(metricName);
    }

    /// <summary>
    /// Record a metric value
    /// </summary>
    public void RecordMetric(string name, object value)
    {
        _metrics[name] = value;
    }

    /// <summary>
    /// Record multiple metrics at once
    /// </summary>
    public void RecordMetrics(Dictionary<string, object> metrics)
    {
        foreach (var kvp in metrics)
        {
            _metrics[kvp.Key] = kvp.Value;
        }
    }

    /// <summary>
    /// Capture system resource metrics
    /// </summary>
    public void CaptureResourceMetrics()
    {
        try
        {
            var process = Process.GetCurrentProcess();
            
            // CPU and memory
            _metrics["WorkingSetMB"] = process.WorkingSet64 / (1024.0 * 1024.0);
            _metrics["PrivateMemoryMB"] = process.PrivateMemorySize64 / (1024.0 * 1024.0);
            _metrics["TotalProcessorTimeSeconds"] = process.TotalProcessorTime.TotalSeconds;
            
            // GC stats
            var gcInfo = GC.GetGCMemoryInfo();
            _metrics["GCTotalMemoryMB"] = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
            _metrics["GCGen0Collections"] = GC.CollectionCount(0);
            _metrics["GCGen1Collections"] = GC.CollectionCount(1);
            _metrics["GCGen2Collections"] = GC.CollectionCount(2);
            
            // Disk usage
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
            if (Directory.Exists(dataPath))
            {
                var drive = new DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
                _metrics["DiskFreeSpaceGB"] = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
                _metrics["DiskTotalSpaceGB"] = drive.TotalSize / (1024.0 * 1024.0 * 1024.0);
            }
            
            _logger.LogInformation("[METRICS] Captured resource metrics");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[METRICS] Failed to capture resource metrics: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Export metrics to JSON file
    /// </summary>
    public async Task ExportMetricsAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var runId = _metrics.ContainsKey("RunId") ? _metrics["RunId"].ToString() : "unknown";
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            var fileName = $"training_metrics_{runId}_{timestamp}.json";
            var filePath = Path.Combine(_metricsDirectory, fileName);

            var json = JsonSerializer.Serialize(_metrics, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(filePath, json, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[METRICS] Exported metrics to: {Path}", filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[METRICS] Failed to export metrics: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Get current metrics snapshot
    /// </summary>
    public Dictionary<string, object> GetMetrics()
    {
        return new Dictionary<string, object>(_metrics);
    }

    /// <summary>
    /// Calculate and record CPU utilization percentage
    /// </summary>
    public async Task<double> MeasureCpuUtilizationAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var process = Process.GetCurrentProcess();
            var startCpuUsage = process.TotalProcessorTime;
            var startTime = DateTime.UtcNow;

            await Task.Delay(1000, cancellationToken).ConfigureAwait(false);

            var endCpuUsage = process.TotalProcessorTime;
            var endTime = DateTime.UtcNow;

            var cpuUsedMs = (endCpuUsage - startCpuUsage).TotalMilliseconds;
            var totalMsPassed = (endTime - startTime).TotalMilliseconds;
            var cpuUsageTotal = cpuUsedMs / (Environment.ProcessorCount * totalMsPassed);
            var cpuPercent = cpuUsageTotal * 100;

            _metrics["CpuUtilizationPercent"] = cpuPercent;
            return cpuPercent;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[METRICS] Failed to measure CPU: {Error}", ex.Message);
            return 0;
        }
    }

    /// <summary>
    /// Record training phase completion
    /// </summary>
    public void RecordPhaseCompletion(string phaseName, bool success, TimeSpan duration)
    {
        var phaseKey = $"Phase_{phaseName}";
        _metrics[$"{phaseKey}_Success"] = success;
        _metrics[$"{phaseKey}_DurationSeconds"] = duration.TotalSeconds;

        _logger.LogInformation("[METRICS] Recorded phase: {Phase} - Success: {Success}, Duration: {Duration:F1}s",
            phaseName, success, duration.TotalSeconds);
    }

    /// <summary>
    /// Record model training results
    /// </summary>
    public void RecordModelTraining(string modelName, bool success, Dictionary<string, double> metrics)
    {
        var modelKey = $"Model_{modelName}";
        _metrics[$"{modelKey}_Success"] = success;
        
        foreach (var kvp in metrics)
        {
            _metrics[$"{modelKey}_{kvp.Key}"] = kvp.Value;
        }

        _logger.LogInformation("[METRICS] Recorded model training: {Model} - Success: {Success}",
            modelName, success);
    }

    /// <summary>
    /// Get summary statistics for logging
    /// </summary>
    public string GetSummary()
    {
        var summary = new System.Text.StringBuilder();
        summary.AppendLine("=== Training Metrics Summary ===");
        
        foreach (var kvp in _metrics)
        {
            summary.AppendLine($"  {kvp.Key}: {kvp.Value}");
        }
        
        return summary.ToString();
    }
}
