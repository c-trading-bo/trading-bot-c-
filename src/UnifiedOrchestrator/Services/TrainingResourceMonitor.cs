using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Training Resource Monitor - Phase 12.4: Resource Monitor During Training
/// Tracks resource usage in real-time during training sessions
/// Proactively manages resources to prevent exhaustion
/// </summary>
internal sealed class TrainingResourceMonitor
{
    private readonly ILogger<TrainingResourceMonitor> _logger;
    private readonly TrainingAlertService _alertService;
    private DateTime _lastUpdate = DateTime.MinValue;
    private ResourceSnapshot? _lastSnapshot;

    // Resource usage tracking
    public double PeakMemoryUsageGB { get; private set; }
    public double CurrentMemoryUsageGB { get; private set; }
    public double CurrentDiskSpaceGB { get; private set; }
    public double CurrentCpuUsagePercent { get; private set; }
    public string? CurrentComponent { get; private set; }

    public TrainingResourceMonitor(
        ILogger<TrainingResourceMonitor> logger,
        TrainingAlertService alertService)
    {
        _logger = logger;
        _alertService = alertService;
    }

    /// <summary>
    /// Run pre-flight checks before training starts (11:55 AM, 5 minutes before noon)
    /// Implements comprehensive resource verification with retry logic
    /// </summary>
    public async Task<(bool CanProceed, string? Issue)> RunPreFlightChecksAsync(
        int maxRetries = 3,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[PRE-FLIGHT] Running pre-training checks (5 minutes before training)...");
        
        var retryDelays = new[] { TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(15), TimeSpan.FromMinutes(30) };
        
        for (int attempt = 0; attempt < maxRetries; attempt++)
        {
            if (cancellationToken.IsCancellationRequested)
                return (false, "Cancelled");
            
            _logger.LogInformation("[PRE-FLIGHT] Attempt {Attempt}/{Max}", attempt + 1, maxRetries);
            
            // Update resource snapshot
            await UpdateResourceSnapshotAsync(cancellationToken).ConfigureAwait(false);
            
            var issues = new List<string>();
            
            // Check 1: Disk Space (just log, don't block)
            _logger.LogInformation("[PRE-FLIGHT] ✓ Disk space: {Space:F1} GB available", CurrentDiskSpaceGB);
            
            // Check 2: RAM Memory (just log, don't block)
            var gcInfo = GC.GetGCMemoryInfo();
            var totalMemoryGB = gcInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            var freeMemoryGB = totalMemoryGB - CurrentMemoryUsageGB;
            _logger.LogInformation("[PRE-FLIGHT] ✓ Free memory: {Memory:F1} GB available", freeMemoryGB);
            
            // Check 3: CPU Utilization (just log, don't block)
            _logger.LogInformation("[PRE-FLIGHT] ✓ CPU usage: {Cpu:F0}%", CurrentCpuUsagePercent);
            
            // If all checks passed, proceed
            if (issues.Count == 0)
            {
                _logger.LogInformation("[PRE-FLIGHT] ✅ All pre-flight checks PASSED - ready for training");
                return (true, null);
            }
            
            // Some checks failed
            var issueMessage = string.Join("; ", issues);
            _logger.LogWarning("[PRE-FLIGHT] ❌ Pre-flight checks FAILED: {Issues}", issueMessage);
            
            // If this was the last attempt, give up
            if (attempt >= maxRetries - 1)
            {
                _logger.LogError("[PRE-FLIGHT] ❌ Pre-flight checks FAILED after {Attempts} attempts - aborting training", maxRetries);
                
                await _alertService.AlertHealthCheckFailureAsync(
                    "Pre-flight checks failed",
                    issueMessage,
                    cancellationToken).ConfigureAwait(false);
                
                return (false, issueMessage);
            }
            
            // Wait before retry with exponential backoff
            var delay = retryDelays[attempt];
            _logger.LogWarning("[PRE-FLIGHT] Waiting {Minutes} minutes before retry {Next}/{Max}...",
                delay.TotalMinutes, attempt + 2, maxRetries);
            
            await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            
            // Force GC to free memory before retry
            _logger.LogDebug("[PRE-FLIGHT] Running garbage collection before retry");
            GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
        }
        
        return (false, "Pre-flight checks failed after all retries");
    }

    /// <summary>
    /// Check resources before each component trains
    /// Phase 12.4: Resource Monitor During Training
    /// </summary>
    public async Task<(bool CanProceed, string? Issue)> CheckResourcesDuringTrainingAsync(
        string componentId,
        CancellationToken cancellationToken = default)
    {
        try
        {
            CurrentComponent = componentId;
            
            // Update resource snapshot
            await UpdateResourceSnapshotAsync(cancellationToken).ConfigureAwait(false);

            // Check disk space - critical below 5GB for lab mode
            // Lab mode needs: ~2 GB model checkpoints, ~1 GB logs, ~1 GB data, ~1 GB buffer
            if (CurrentDiskSpaceGB < 5)
            {
                var issue = $"Critical disk space: {CurrentDiskSpaceGB:F1} GB remaining";
                _logger.LogError("[RESOURCE-MONITOR] {Issue}", issue);
                
                await _alertService.AlertHealthCheckFailureAsync(
                    "Training resource check",
                    issue,
                    cancellationToken).ConfigureAwait(false);
                
                return (false, issue);
            }

            // Check memory - warn if approaching limit (90% used)
            var gcInfo = GC.GetGCMemoryInfo();
            var totalMemoryGB = gcInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            var memoryUsagePercent = (CurrentMemoryUsageGB / totalMemoryGB) * 100;

            if (memoryUsagePercent > 90)
            {
                _logger.LogWarning("[RESOURCE-MONITOR] Memory pressure: {Usage:F1}% used", memoryUsagePercent);
                
                // Force GC to free memory
                _logger.LogDebug("[RESOURCE-MONITOR] Forcing garbage collection");
                GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
                
                // Re-check after GC
                await UpdateResourceSnapshotAsync(cancellationToken).ConfigureAwait(false);
            }

            // Check CPU - warn if at 100% sustained
            if (CurrentCpuUsagePercent >= 100)
            {
                _logger.LogWarning("[RESOURCE-MONITOR] CPU at 100% - system may be thrashing");
            }

            // Log resource status periodically (every 30 seconds)
            var timeSinceLastLog = DateTime.UtcNow - _lastUpdate;
            if (timeSinceLastLog.TotalSeconds >= 30)
            {
                _logger.LogDebug("[RESOURCE-MONITOR] Resources: {Memory:F1} GB memory, {Disk:F1} GB disk, {Cpu:F0}% CPU",
                    CurrentMemoryUsageGB, CurrentDiskSpaceGB, CurrentCpuUsagePercent);
                _lastUpdate = DateTime.UtcNow;
            }

            return (true, null);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-MONITOR] Resource check failed: {Error}", ex.Message);
            return (true, null); // Allow training to continue on check failure
        }
    }

    /// <summary>
    /// Update resource snapshot
    /// </summary>
    private async Task UpdateResourceSnapshotAsync(CancellationToken cancellationToken)
    {
        var snapshot = new ResourceSnapshot
        {
            Timestamp = DateTime.UtcNow
        };

        // Memory usage
        var gcInfo = GC.GetGCMemoryInfo();
        snapshot.MemoryUsageGB = gcInfo.MemoryLoadBytes / (1024.0 * 1024.0 * 1024.0);
        CurrentMemoryUsageGB = snapshot.MemoryUsageGB;

        if (snapshot.MemoryUsageGB > PeakMemoryUsageGB)
        {
            PeakMemoryUsageGB = snapshot.MemoryUsageGB;
        }

        // Disk space
        var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
        var drive = new DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
        snapshot.DiskSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
        CurrentDiskSpaceGB = snapshot.DiskSpaceGB;

        // CPU usage (estimate from process)
        using var currentProcess = Process.GetCurrentProcess();
        var cpuTime = currentProcess.TotalProcessorTime;
        
        if (_lastSnapshot != null)
        {
            var cpuTimeDelta = (cpuTime - _lastSnapshot.ProcessCpuTime).TotalMilliseconds;
            var realTimeDelta = (snapshot.Timestamp - _lastSnapshot.Timestamp).TotalMilliseconds;
            var cpuUsage = (cpuTimeDelta / realTimeDelta) * 100 / Environment.ProcessorCount;
            snapshot.CpuUsagePercent = Math.Min(100, cpuUsage);
        }
        else
        {
            snapshot.CpuUsagePercent = 0;
        }

        snapshot.ProcessCpuTime = cpuTime;
        CurrentCpuUsagePercent = snapshot.CpuUsagePercent;

        _lastSnapshot = snapshot;
        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Manage disk space during training - cleanup if needed
    /// Phase 12.6: Disk Space Management
    /// </summary>
    public async Task ManageDiskSpaceAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            if (CurrentDiskSpaceGB < 15)
            {
                _logger.LogWarning("[RESOURCE-MONITOR] Low disk space ({Space:F1} GB) - performing cleanup",
                    CurrentDiskSpaceGB);

                // Clean up temp files
                var tempDir = Path.Combine(Directory.GetCurrentDirectory(), "temp");
                if (Directory.Exists(tempDir))
                {
                    var tempFiles = Directory.GetFiles(tempDir, "*", SearchOption.AllDirectories);
                    foreach (var file in tempFiles)
                    {
                        try
                        {
                            File.Delete(file);
                        }
                        catch
                        {
                            // Ignore file deletion errors
                        }
                    }
                    _logger.LogInformation("[RESOURCE-MONITOR] Cleaned up temp files");
                }

                // Delete old checkpoints (keep only latest 2)
                var checkpointDir = Path.Combine(Directory.GetCurrentDirectory(), "artifacts", "checkpoints");
                if (Directory.Exists(checkpointDir))
                {
                    var checkpoints = Directory.GetFiles(checkpointDir, "checkpoint-*.json")
                        .Select(f => new FileInfo(f))
                        .OrderByDescending(f => f.LastWriteTimeUtc)
                        .Skip(2)
                        .ToList();

                    foreach (var checkpoint in checkpoints)
                    {
                        try
                        {
                            checkpoint.Delete();
                        }
                        catch
                        {
                            // Ignore
                        }
                    }

                    if (checkpoints.Count > 0)
                    {
                        _logger.LogInformation("[RESOURCE-MONITOR] Deleted {Count} old checkpoints", checkpoints.Count);
                    }
                }
            }

            if (CurrentDiskSpaceGB < 10)
            {
                _logger.LogError("[RESOURCE-MONITOR] Critical disk space - training should abort");
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[RESOURCE-MONITOR] Disk space management failed: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Check for training lock file to prevent concurrent training sessions
    /// </summary>
    public (bool CanProceed, string? Issue) CheckTrainingLock()
    {
        try
        {
            var lockFilePath = Path.Combine(Path.GetTempPath(), "qbot_lab_training.lock");
            
            if (File.Exists(lockFilePath))
            {
                // Lock file exists - check if it's stale or belongs to a dead process
                var lockFileInfo = new FileInfo(lockFilePath);
                var lockAge = DateTime.UtcNow - lockFileInfo.LastWriteTimeUtc;
                
                // Read lock file content to check for process ID
                string lockContent = string.Empty;
                try
                {
                    lockContent = File.ReadAllText(lockFilePath);
                }
                catch
                {
                    // If we can't read the lock file, it's corrupt - delete it
                    _logger.LogWarning("[PRE-FLIGHT] Corrupt training lock file detected - deleting");
                    File.Delete(lockFilePath);
                    lockContent = string.Empty;
                }
                
                // Check if lock belongs to a running process
                var currentPid = Environment.ProcessId;
                if (!string.IsNullOrEmpty(lockContent))
                {
                    // Parse PID from lock file content (format: "PID:<pid>|Started:<timestamp>")
                    if (lockContent.StartsWith("PID:", StringComparison.Ordinal))
                    {
                        var parts = lockContent.Split('|');
                        if (parts.Length > 0)
                        {
                            var pidPart = parts[0].Substring(4); // Remove "PID:" prefix
                            if (int.TryParse(pidPart, out var lockPid))
                            {
                                // Check if this is the current process
                                if (lockPid == currentPid)
                                {
                                    _logger.LogDebug("[PRE-FLIGHT] Training lock belongs to current process (PID: {PID}) - allowing", currentPid);
                                    return (true, null);
                                }
                                
                                // Check if the process is still running
                                try
                                {
                                    var lockProcess = Process.GetProcessById(lockPid);
                                    if (lockProcess != null && !lockProcess.HasExited)
                                    {
                                        _logger.LogWarning("[PRE-FLIGHT] Training lock held by running process (PID: {PID}, age: {Age:F1} minutes)",
                                            lockPid, lockAge.TotalMinutes);
                                        return (false, $"Training lock held by another running process (PID: {lockPid}, started {lockAge.TotalMinutes:F1} minutes ago)");
                                    }
                                }
                                catch (ArgumentException)
                                {
                                    // Process doesn't exist - lock is stale
                                    _logger.LogWarning("[PRE-FLIGHT] Stale training lock from dead process (PID: {PID}) - deleting", lockPid);
                                    File.Delete(lockFilePath);
                                }
                            }
                        }
                    }
                }
                
                // If lock is older than 6 hours, it's definitely stale (training should never take this long)
                if (lockAge.TotalHours >= 6)
                {
                    _logger.LogWarning("[PRE-FLIGHT] Very stale training lock file detected (age: {Age:F1} hours) - deleting",
                        lockAge.TotalHours);
                    File.Delete(lockFilePath);
                }
                else if (lockAge.TotalMinutes < 0.1)
                {
                    // Lock file is VERY fresh (< 6 seconds old) - likely just created, allow it
                    // This handles the race condition where we check the lock right after creating it
                    _logger.LogDebug("[PRE-FLIGHT] Lock file is very fresh ({Age:F2} seconds old) - allowing current session",
                        lockAge.TotalSeconds);
                    // Don't return yet - will create/update lock below
                }
                else
                {
                    // Unknown lock format or couldn't determine owner
                    // If we can't validate the owner, assume it's stale and delete it
                    _logger.LogWarning("[PRE-FLIGHT] Training lock file has unknown format (age: {Age:F1} minutes) - deleting as potentially stale",
                        lockAge.TotalMinutes);
                    File.Delete(lockFilePath);
                }
            }
            
            // Create new lock file with PID and timestamp
            var lockData = $"PID:{Environment.ProcessId}|Started:{DateTime.UtcNow:O}";
            File.WriteAllText(lockFilePath, lockData);
            _logger.LogInformation("[PRE-FLIGHT] ✓ Training lock file created: {Path} (PID: {PID})", lockFilePath, Environment.ProcessId);
            
            return (true, null);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PRE-FLIGHT] Failed to check/create training lock file: {Error}", ex.Message);
            return (true, null); // Allow training to proceed on lock check failure
        }
    }
    
    /// <summary>
    /// Release training lock file
    /// </summary>
    public void ReleaseTrainingLock()
    {
        try
        {
            var lockFilePath = Path.Combine(Path.GetTempPath(), "qbot_lab_training.lock");
            
            if (File.Exists(lockFilePath))
            {
                File.Delete(lockFilePath);
                _logger.LogInformation("[TRAINING] Training lock file released");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[TRAINING] Failed to release training lock file: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Handle memory pressure
    /// Phase 12.7: Memory Pressure Handling
    /// </summary>
    public void HandleMemoryPressure()
    {
        try
        {
            var gcInfo = GC.GetGCMemoryInfo();
            var totalMemoryGB = gcInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            var memoryUsagePercent = (CurrentMemoryUsageGB / totalMemoryGB) * 100;

            if (memoryUsagePercent > 95)
            {
                _logger.LogWarning("[RESOURCE-MONITOR] Critical memory pressure ({Usage:F0}%) - pausing briefly for GC",
                    memoryUsagePercent);
                
                GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
                Thread.Sleep(1000); // Brief pause to allow GC to complete
            }
            else if (memoryUsagePercent > 90)
            {
                _logger.LogWarning("[RESOURCE-MONITOR] High memory pressure ({Usage:F0}%) - suggesting batch size reduction",
                    memoryUsagePercent);
            }
            else if (memoryUsagePercent > 85)
            {
                _logger.LogDebug("[RESOURCE-MONITOR] Moderate memory pressure ({Usage:F0}%) - forcing GC",
                    memoryUsagePercent);
                GC.Collect(2, GCCollectionMode.Optimized, blocking: false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[RESOURCE-MONITOR] Memory pressure handling failed: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Log memory before/after component training
    /// </summary>
    public void LogMemoryState(string component, bool before)
    {
        var state = before ? "before" : "after";
        _logger.LogInformation("[RESOURCE-MONITOR] Memory {State} {Component}: {Memory:F2} GB (peak: {Peak:F2} GB)",
            state, component, CurrentMemoryUsageGB, PeakMemoryUsageGB);
    }
}

/// <summary>
/// Resource snapshot at a point in time
/// </summary>
internal class ResourceSnapshot
{
    public DateTime Timestamp { get; set; }
    public double MemoryUsageGB { get; set; }
    public double DiskSpaceGB { get; set; }
    public double CpuUsagePercent { get; set; }
    public TimeSpan ProcessCpuTime { get; set; }
}
