using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Pre-training resource checks - verify system has sufficient resources
/// Prevents training from starting if resources are insufficient
/// 
/// Thresholds are now configurable via appsettings.json:
/// - MinDiskSpaceGB: Lowered from 50GB to 20GB (configurable)
/// - MinRamGB: Lowered from 8GB to 4GB (configurable)
/// - WarningOnly: Can emit warnings instead of hard failures
/// </summary>
internal sealed class ResourcePreCheckService
{
    private readonly ILogger<ResourcePreCheckService> _logger;
    private readonly ResourcePreCheckOptions _options;

    public ResourcePreCheckService(
        ILogger<ResourcePreCheckService> logger,
        IOptions<ResourcePreCheckOptions> options)
    {
        _logger = logger;
        _options = options?.Value ?? new ResourcePreCheckOptions();
    }

    /// <summary>
    /// Run all pre-training resource checks
    /// Returns true if all checks pass, false otherwise
    /// </summary>
    public async Task<(bool Success, List<string> FailedChecks)> RunAllChecksAsync(
        CancellationToken cancellationToken = default)
    {
        var failedChecks = new List<string>();

        _logger.LogInformation("[RESOURCE-CHECK] Starting pre-training resource checks...");

        // Check 1: Disk space
        if (!await CheckDiskSpaceAsync(cancellationToken).ConfigureAwait(false))
        {
            failedChecks.Add("Disk space");
        }

        // Check 2: Available RAM
        if (!CheckAvailableMemory())
        {
            failedChecks.Add("Available memory");
        }

        // Check 3: CPU utilization (ensure not already overloaded)
        if (!await CheckCpuUtilizationAsync(cancellationToken).ConfigureAwait(false))
        {
            failedChecks.Add("CPU utilization");
        }

        // Check 4: GPU availability (optional, don't fail if not available)
        if (_options.EnableGpuCheck)
        {
            await CheckGpuAvailabilityAsync(cancellationToken).ConfigureAwait(false);
        }

        if (failedChecks.Any())
        {
            _logger.LogError("[RESOURCE-CHECK] ❌ Failed checks: {Checks}", string.Join(", ", failedChecks));
            return (false, failedChecks);
        }

        _logger.LogInformation("[RESOURCE-CHECK] ✅ All resource checks passed");
        return (true, failedChecks);
    }

    /// <summary>
    /// Check available disk space
    /// </summary>
    private Task<bool> CheckDiskSpaceAsync(CancellationToken cancellationToken)
    {
        try
        {
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
            Directory.CreateDirectory(dataPath); // Ensure it exists

            var drive = new DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
            var freeSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);

            _logger.LogInformation("[RESOURCE-CHECK] Disk space: {Free:F1} GB free (required: {Required} GB)",
                freeSpaceGB, _options.MinDiskSpaceGB);

            if (freeSpaceGB < _options.MinDiskSpaceGB)
            {
                if (_options.WarningOnly)
                {
                    _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Low disk space: {Free:F1} GB < {Required} GB (warning only)",
                        freeSpaceGB, _options.MinDiskSpaceGB);
                    return Task.FromResult(true); // Pass check but log warning
                }
                
                _logger.LogError("[RESOURCE-CHECK] ❌ Insufficient disk space: {Free:F1} GB < {Required} GB",
                    freeSpaceGB, _options.MinDiskSpaceGB);
                return Task.FromResult(false);
            }

            _logger.LogInformation("[RESOURCE-CHECK] ✓ Sufficient disk space available");
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check disk space: {Error}", ex.Message);
            return Task.FromResult(false);
        }
    }

    /// <summary>
    /// Check available RAM
    /// </summary>
    private bool CheckAvailableMemory()
    {
        try
        {
            var totalMemoryGB = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            
            _logger.LogInformation("[RESOURCE-CHECK] Available memory: {Memory:F1} GB (required: {Required} GB)",
                totalMemoryGB, _options.MinRamGB);

            if (totalMemoryGB < _options.MinRamGB)
            {
                if (_options.WarningOnly)
                {
                    _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Low memory: {Memory:F1} GB < {Required} GB (warning only)",
                        totalMemoryGB, _options.MinRamGB);
                    return true; // Pass check but log warning
                }
                
                _logger.LogError("[RESOURCE-CHECK] ❌ Insufficient memory: {Memory:F1} GB < {Required} GB",
                    totalMemoryGB, _options.MinRamGB);
                return false;
            }

            _logger.LogInformation("[RESOURCE-CHECK] ✓ Sufficient memory available");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check memory: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Check CPU utilization (don't start training if system is already overloaded)
    /// </summary>
    private async Task<bool> CheckCpuUtilizationAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Take a snapshot of CPU usage
            var process = Process.GetCurrentProcess();
            var startTime = process.TotalProcessorTime;
            var startCpuUsage = DateTime.UtcNow;

            await Task.Delay(1000, cancellationToken).ConfigureAwait(false);

            var endTime = process.TotalProcessorTime;
            var endCpuUsage = DateTime.UtcNow;

            var cpuUsedMs = (endTime - startTime).TotalMilliseconds;
            var totalMsPassed = (endCpuUsage - startCpuUsage).TotalMilliseconds;
            var cpuUsageTotal = cpuUsedMs / (Environment.ProcessorCount * totalMsPassed);

            var cpuPercent = cpuUsageTotal * 100;

            _logger.LogInformation("[RESOURCE-CHECK] Current CPU usage: {Cpu:F1}% (threshold: {Threshold}%)",
                cpuPercent, _options.MaxCpuThreshold);

            if (cpuPercent > _options.MaxCpuThreshold)
            {
                if (_options.WarningOnly)
                {
                    _logger.LogWarning("[RESOURCE-CHECK] ⚠️ High CPU usage: {Cpu:F1}% > {Threshold}% (warning only)",
                        cpuPercent, _options.MaxCpuThreshold);
                    return true; // Pass check but log warning
                }
                
                _logger.LogWarning("[RESOURCE-CHECK] ⚠️ High CPU usage detected: {Cpu:F1}% > {Threshold}%",
                    cpuPercent, _options.MaxCpuThreshold);
                return false;
            }

            _logger.LogInformation("[RESOURCE-CHECK] ✓ CPU utilization acceptable");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[RESOURCE-CHECK] Failed to check CPU utilization: {Error}", ex.Message);
            // Don't fail the check if we can't measure CPU (allow training to proceed)
            return true;
        }
    }

    /// <summary>
    /// Check GPU availability (informational only, not required)
    /// </summary>
    private async Task<bool> CheckGpuAvailabilityAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Try to detect NVIDIA GPU using nvidia-smi
            var startInfo = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=name,memory.free --format=csv,noheader",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process != null)
            {
                var output = await process.StandardOutput.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
                await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);

                if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
                {
                    _logger.LogInformation("[RESOURCE-CHECK] ✓ GPU detected: {Gpu}", output.Trim());
                    return true;
                }
            }

            _logger.LogInformation("[RESOURCE-CHECK] ℹ️ No GPU detected (will use CPU for training)");
            return false;
        }
        catch
        {
            _logger.LogInformation("[RESOURCE-CHECK] ℹ️ No GPU detected (will use CPU for training)");
            return false;
        }
    }

    /// <summary>
    /// Check for resource-intensive processes that might interfere with training
    /// </summary>
    public List<string> CheckForResourceIntensiveProcesses()
    {
        var intensiveProcesses = new List<string>();

        try
        {
            var allProcesses = Process.GetProcesses();
            var cpuThreshold = 50.0; // Consider processes using > 50% CPU as intensive

            foreach (var proc in allProcesses)
            {
                try
                {
                    var startTime = proc.TotalProcessorTime;
                    var startCheck = DateTime.UtcNow;

                    System.Threading.Thread.Sleep(100);

                    var endTime = proc.TotalProcessorTime;
                    var endCheck = DateTime.UtcNow;

                    var cpuUsedMs = (endTime - startTime).TotalMilliseconds;
                    var totalMsPassed = (endCheck - startCheck).TotalMilliseconds;
                    var cpuUsage = (cpuUsedMs / totalMsPassed) * 100;

                    if (cpuUsage > cpuThreshold && proc.ProcessName != Process.GetCurrentProcess().ProcessName)
                    {
                        intensiveProcesses.Add($"{proc.ProcessName} ({cpuUsage:F0}% CPU)");
                    }
                }
                catch
                {
                    // Process may have exited, skip
                }
            }

            if (intensiveProcesses.Any())
            {
                _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Resource-intensive processes detected: {Processes}",
                    string.Join(", ", intensiveProcesses));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[RESOURCE-CHECK] Failed to check for intensive processes: {Error}", ex.Message);
        }

        return intensiveProcesses;
    }
}

/// <summary>
/// Configuration options for resource pre-check service
/// </summary>
public sealed class ResourcePreCheckOptions
{
    /// <summary>
    /// Minimum disk space required in GB (default: 20GB, previously 50GB)
    /// </summary>
    public long MinDiskSpaceGB { get; set; } = 20;
    
    /// <summary>
    /// Minimum RAM required in GB (default: 4GB, previously 8GB)
    /// </summary>
    public long MinRamGB { get; set; } = 4;
    
    /// <summary>
    /// Maximum CPU threshold percentage (default: 90%)
    /// </summary>
    public double MaxCpuThreshold { get; set; } = 90.0;
    
    /// <summary>
    /// If true, emit warnings instead of hard failures for marginal resources
    /// Allows training to proceed but logs concerns (default: false)
    /// </summary>
    public bool WarningOnly { get; set; } = false;
    
    /// <summary>
    /// Enable GPU availability check (default: true)
    /// GPU is optional - check is informational only
    /// </summary>
    public bool EnableGpuCheck { get; set; } = true;
}
