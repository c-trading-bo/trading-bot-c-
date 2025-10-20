using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// System Capability Profiler - Phase 12: Resource Optimization
/// Profiles system capabilities at startup to determine optimal training configuration
/// Replaces hardcoded thresholds with intelligent, adaptive resource management
/// </summary>
internal sealed class SystemCapabilityProfiler
{
    private readonly ILogger<SystemCapabilityProfiler> _logger;
    private SystemProfile? _cachedProfile;

    public SystemCapabilityProfiler(ILogger<SystemCapabilityProfiler> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Profile system capabilities - called once at Lab Mode startup
    /// Phase 12.1: System Capability Profiler
    /// </summary>
    public async Task<SystemProfile> ProfileSystemCapabilitiesAsync(CancellationToken cancellationToken = default)
    {
        if (_cachedProfile != null)
        {
            _logger.LogDebug("[PROFILER] Using cached system profile");
            return _cachedProfile;
        }

        _logger.LogInformation("[PROFILER] Profiling system capabilities...");
        var profile = new SystemProfile
        {
            ProfiledAt = DateTime.UtcNow
        };

        try
        {
            // Measure total and available disk space
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
            Directory.CreateDirectory(dataPath);
            var drive = new DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
            
            profile.TotalDiskSpaceGB = drive.TotalSize / (1024.0 * 1024.0 * 1024.0);
            profile.AvailableDiskSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
            
            _logger.LogInformation("[PROFILER] Disk: {Available:F1} GB available / {Total:F1} GB total",
                profile.AvailableDiskSpaceGB, profile.TotalDiskSpaceGB);

            // Measure total and available RAM
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var gcMemoryInfo = GC.GetGCMemoryInfo();
                profile.TotalRamGB = gcMemoryInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
                profile.AvailableRamGB = (gcMemoryInfo.TotalAvailableMemoryBytes - gcMemoryInfo.MemoryLoadBytes) / (1024.0 * 1024.0 * 1024.0);
            }
            else
            {
                // Linux/macOS - read from /proc/meminfo or use GC info
                try
                {
                    if (File.Exists("/proc/meminfo"))
                    {
                        var memInfo = await File.ReadAllLinesAsync("/proc/meminfo", cancellationToken).ConfigureAwait(false);
                        var memTotal = ParseMemInfoLine(memInfo.FirstOrDefault(l => l.StartsWith("MemTotal:")));
                        var memAvailable = ParseMemInfoLine(memInfo.FirstOrDefault(l => l.StartsWith("MemAvailable:")));
                        
                        profile.TotalRamGB = memTotal / (1024.0 * 1024.0);
                        profile.AvailableRamGB = memAvailable / (1024.0 * 1024.0);
                    }
                    else
                    {
                        // Fallback to GC info
                        var gcMemoryInfo = GC.GetGCMemoryInfo();
                        profile.TotalRamGB = gcMemoryInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
                        profile.AvailableRamGB = (gcMemoryInfo.TotalAvailableMemoryBytes - gcMemoryInfo.MemoryLoadBytes) / (1024.0 * 1024.0 * 1024.0);
                    }
                }
                catch
                {
                    var gcMemoryInfo = GC.GetGCMemoryInfo();
                    profile.TotalRamGB = gcMemoryInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
                    profile.AvailableRamGB = (gcMemoryInfo.TotalAvailableMemoryBytes - gcMemoryInfo.MemoryLoadBytes) / (1024.0 * 1024.0 * 1024.0);
                }
            }
            
            _logger.LogInformation("[PROFILER] Memory: {Available:F1} GB available / {Total:F1} GB total",
                profile.AvailableRamGB, profile.TotalRamGB);

            // CPU core count and max threads
            profile.CpuCoreCount = Environment.ProcessorCount;
            profile.MaxThreads = Environment.ProcessorCount * 2; // Typical hyperthreading
            
            _logger.LogInformation("[PROFILER] CPU: {Cores} cores, {Threads} max threads",
                profile.CpuCoreCount, profile.MaxThreads);

            // Detect GPU capabilities
            await DetectGpuCapabilitiesAsync(profile, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("[PROFILER] GPU: {Type} ({Available})",
                profile.GpuType, profile.HasGpu ? "Available" : "Not detected");

            // Detect SSD vs HDD (measure small file write speed)
            await MeasureStorageTypeAsync(profile, dataPath, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("[PROFILER] Storage: {Type} (write speed: {Speed:F1} MB/s)",
                profile.StorageType, profile.StorageWriteSpeedMBps);

            // Cache the profile
            _cachedProfile = profile;

            _logger.LogInformation("[PROFILER] System profile complete - {Category} system detected",
                DetermineSystemCategory(profile));

            return profile;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PROFILER] Failed to profile system capabilities: {Error}", ex.Message);
            
            // Return conservative defaults
            return new SystemProfile
            {
                ProfiledAt = DateTime.UtcNow,
                TotalDiskSpaceGB = 100,
                AvailableDiskSpaceGB = 30,
                TotalRamGB = 8,
                AvailableRamGB = 4,
                CpuCoreCount = 4,
                MaxThreads = 8,
                HasGpu = false,
                GpuType = "None",
                StorageType = "HDD",
                StorageWriteSpeedMBps = 50
            };
        }
    }

    /// <summary>
    /// Detect GPU capabilities - CUDA, DirectML, or CPU-only
    /// Phase 12.5: GPU Detection and Utilization
    /// </summary>
    private async Task DetectGpuCapabilitiesAsync(SystemProfile profile, CancellationToken cancellationToken)
    {
        try
        {
            // Try loading ONNX Runtime with CUDA provider
            // This is a best-effort detection - actual ONNX loading would require the runtime
            // For now, check environment variables and process list
            
            var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPath) && Directory.Exists(cudaPath))
            {
                profile.HasGpu = true;
                profile.GpuType = "CUDA";
                _logger.LogDebug("[PROFILER] CUDA installation detected at {Path}", cudaPath);
                return;
            }

            // Check for DirectML on Windows
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // DirectML is available on Windows 10/11 with compatible GPU
                // Simple heuristic: check if we're on Windows
                profile.HasGpu = true;
                profile.GpuType = "DirectML";
                _logger.LogDebug("[PROFILER] DirectML may be available (Windows detected)");
                return;
            }

            // No GPU detected
            profile.HasGpu = false;
            profile.GpuType = "None";
            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[PROFILER] GPU detection failed: {Error}", ex.Message);
            profile.HasGpu = false;
            profile.GpuType = "None";
        }
    }

    /// <summary>
    /// Measure storage type (SSD vs HDD) by testing write speed
    /// Phase 12.1: System Capability Profiler
    /// </summary>
    private async Task MeasureStorageTypeAsync(SystemProfile profile, string testPath, CancellationToken cancellationToken)
    {
        try
        {
            var testFile = Path.Combine(testPath, ".storage_speed_test");
            var testData = new byte[10 * 1024 * 1024]; // 10 MB test file
            new Random().NextBytes(testData);

            var stopwatch = Stopwatch.StartNew();
            await File.WriteAllBytesAsync(testFile, testData, cancellationToken).ConfigureAwait(false);
            stopwatch.Stop();

            var writeMBps = 10.0 / stopwatch.Elapsed.TotalSeconds;
            profile.StorageWriteSpeedMBps = writeMBps;

            // SSD typically > 200 MB/s, HDD typically < 150 MB/s
            profile.StorageType = writeMBps > 200 ? "SSD" : "HDD";

            // Clean up test file
            try
            {
                File.Delete(testFile);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[PROFILER] Storage speed test failed: {Error}", ex.Message);
            profile.StorageType = "Unknown";
            profile.StorageWriteSpeedMBps = 0;
        }
    }

    /// <summary>
    /// Parse memory info line from /proc/meminfo (Linux)
    /// </summary>
    private static long ParseMemInfoLine(string? line)
    {
        if (string.IsNullOrEmpty(line))
            return 0;

        var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 2)
            return 0;

        if (long.TryParse(parts[1], out var value))
            return value; // Value in KB

        return 0;
    }

    /// <summary>
    /// Determine system category for display purposes
    /// </summary>
    private string DetermineSystemCategory(SystemProfile profile)
    {
        // High-end: 32GB+ RAM, GPU, SSD, 50+ GB disk
        if (profile.TotalRamGB >= 32 && profile.HasGpu && profile.StorageType == "SSD" && profile.AvailableDiskSpaceGB >= 50)
            return "HIGH-END";

        // Mid-range: 8-16GB RAM, SSD, 30-50GB disk
        if (profile.TotalRamGB >= 8 && profile.StorageType == "SSD" && profile.AvailableDiskSpaceGB >= 30)
            return "MID-RANGE";

        // Low-end: 4-8GB RAM, HDD, 20-30GB disk
        if (profile.TotalRamGB >= 4 && profile.AvailableDiskSpaceGB >= 20)
            return "LOW-END";

        // Very constrained: <4GB RAM, <20GB disk
        return "CONSTRAINED";
    }

    /// <summary>
    /// Get cached profile (avoid re-profiling)
    /// </summary>
    public SystemProfile? GetCachedProfile() => _cachedProfile;
}

/// <summary>
/// System profile data structure
/// </summary>
internal class SystemProfile
{
    public DateTime ProfiledAt { get; set; }
    
    // Disk
    public double TotalDiskSpaceGB { get; set; }
    public double AvailableDiskSpaceGB { get; set; }
    
    // Memory
    public double TotalRamGB { get; set; }
    public double AvailableRamGB { get; set; }
    
    // CPU
    public int CpuCoreCount { get; set; }
    public int MaxThreads { get; set; }
    
    // GPU
    public bool HasGpu { get; set; }
    public string GpuType { get; set; } = "None";
    
    // Storage
    public string StorageType { get; set; } = "Unknown";
    public double StorageWriteSpeedMBps { get; set; }
}
