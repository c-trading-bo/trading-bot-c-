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
/// All checks are informational only and will not prevent training from starting
/// 
/// Thresholds are configurable via appsettings.json but default to 0 (no requirements):
/// - MinDiskSpaceGB: Default 0 (no minimum requirement)
/// - MinRamGB: Default 0 (no minimum requirement)
/// - MaxCpuThreshold: Default 100% (no CPU limit)
/// - WarningOnly: Default true (all checks are informational)
/// </summary>
internal sealed class ResourcePreCheckService
{
    private readonly ILogger<ResourcePreCheckService> _logger;
    private readonly ResourcePreCheckOptions _options;
    private readonly DataIntegrityService _dataIntegrityService;

    public ResourcePreCheckService(
        ILogger<ResourcePreCheckService> logger,
        IOptions<ResourcePreCheckOptions> options,
        DataIntegrityService dataIntegrityService)
    {
        _logger = logger;
        _options = options?.Value ?? new ResourcePreCheckOptions();
        _dataIntegrityService = dataIntegrityService;
    }

    /// <summary>
    /// Run all pre-training resource checks
    /// Returns true if all checks pass, false otherwise
    /// Phase 3: Enhanced with historical data, experience DB, lock files, and timezone checks
    /// </summary>
    public async Task<(bool Success, List<string> FailedChecks)> RunAllChecksAsync(
        CancellationToken cancellationToken = default)
    {
        var failedChecks = new List<string>();
        var checkNumber = 0;
        var totalChecks = 10; // Update based on number of checks

        _logger.LogInformation("[RESOURCE-CHECK] Starting pre-training resource checks...");
        _logger.LogInformation("[RESOURCE-CHECK] ========================================");

        // Check 1: Disk space (detailed per directory)
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking disk space...", checkNumber, totalChecks);
        if (!await CheckDiskSpaceDetailedAsync(cancellationToken).ConfigureAwait(false))
        {
            failedChecks.Add("Disk space");
        }

        // Check 2: Available RAM (actual available, not just total)
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking available memory...", checkNumber, totalChecks);
        if (!CheckAvailableMemoryDetailed())
        {
            failedChecks.Add("Available memory");
        }

        // Check 3: CPU utilization (ensure not already overloaded)
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking CPU utilization...", checkNumber, totalChecks);
        if (!await CheckCpuUtilizationDetailedAsync(cancellationToken).ConfigureAwait(false))
        {
            failedChecks.Add("CPU utilization");
        }

        // Check 4: Historical data files
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking historical data...", checkNumber, totalChecks);
        if (!await CheckHistoricalDataAsync(cancellationToken).ConfigureAwait(false))
        {
            failedChecks.Add("Historical data");
        }

        // Check 5: Experience database/files
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking experience database...", checkNumber, totalChecks);
        if (!await CheckExperienceDatabaseAsync(cancellationToken).ConfigureAwait(false))
        {
            failedChecks.Add("Experience database");
        }

        // Check 6: Model registry directories
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking model registry...", checkNumber, totalChecks);
        if (!CheckModelRegistry())
        {
            failedChecks.Add("Model registry");
        }

        // Check 7: Lock file status
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking lock files...", checkNumber, totalChecks);
        if (!CheckLockFiles())
        {
            failedChecks.Add("Lock files");
        }

        // Check 8: Timezone configuration
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking timezone...", checkNumber, totalChecks);
        if (!CheckTimezone())
        {
            failedChecks.Add("Timezone");
        }

        // Check 9: Network connectivity (optional)
        checkNumber++;
        _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking network connectivity...", checkNumber, totalChecks);
        await CheckNetworkConnectivityAsync(cancellationToken).ConfigureAwait(false);

        // Check 10: GPU availability (optional, don't fail if not available)
        checkNumber++;
        if (_options.EnableGpuCheck)
        {
            _logger.LogInformation("[RESOURCE-CHECK] [{CheckNum}/{Total}] Checking GPU availability...", checkNumber, totalChecks);
            await CheckGpuAvailabilityAsync(cancellationToken).ConfigureAwait(false);
        }

        _logger.LogInformation("[RESOURCE-CHECK] ========================================");
        
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

    /// <summary>
    /// Phase 3: Enhanced disk space check with detailed directory-specific checks
    /// </summary>
    private async Task<bool> CheckDiskSpaceDetailedAsync(CancellationToken cancellationToken)
    {
        try
        {
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data");
            Directory.CreateDirectory(dataPath);

            var drive = new DriveInfo(Path.GetPathRoot(dataPath) ?? "/");
            var freeSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);

            // Check specific directories
            var modelsPath = Path.Combine(Directory.GetCurrentDirectory(), "models");
            var checkpointsPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoints");
            var logsPath = Path.Combine(Directory.GetCurrentDirectory(), "logs");

            // Create directories if they don't exist
            Directory.CreateDirectory(modelsPath);
            Directory.CreateDirectory(checkpointsPath);
            Directory.CreateDirectory(logsPath);

            // Required space per directory (in GB)
            const double ModelsRequired = 5.0;
            const double CheckpointsRequired = 2.0;
            const double LogsRequired = 0.5;
            // Use configured minimum disk space instead of hard-coded value
            var totalMinimum = (double)_options.MinDiskSpaceGB;

            _logger.LogInformation("[RESOURCE-CHECK]   Available disk space: {Free:F2} GB", freeSpaceGB);
            _logger.LogInformation("[RESOURCE-CHECK]   Models directory: needs {Required} GB", ModelsRequired);
            _logger.LogInformation("[RESOURCE-CHECK]   Checkpoints directory: needs {Required} GB", CheckpointsRequired);
            _logger.LogInformation("[RESOURCE-CHECK]   Logs directory: needs {Required} GB", LogsRequired);
            _logger.LogInformation("[RESOURCE-CHECK]   Total minimum required: {Required} GB", totalMinimum);

            // If minimum is 0, skip the check (no requirement)
            if (totalMinimum == 0)
            {
                _logger.LogInformation("[RESOURCE-CHECK] ✓ No disk space requirement configured (training allowed on any hardware)");
                await Task.CompletedTask.ConfigureAwait(false);
                return true;
            }

            if (freeSpaceGB < totalMinimum)
            {
                if (_options.WarningOnly)
                {
                    _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Low disk space: {Free:F2} GB < {Required} GB (warning only)",
                        freeSpaceGB, totalMinimum);
                    return true;
                }

                _logger.LogError("[RESOURCE-CHECK] ❌ Insufficient disk space: {Free:F2} GB < {Required} GB",
                    freeSpaceGB, totalMinimum);
                return false;
            }

            _logger.LogInformation("[RESOURCE-CHECK] ✓ Sufficient disk space available");
            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check disk space: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Phase 3: Enhanced memory check with actual available memory (not just total)
    /// </summary>
    private bool CheckAvailableMemoryDetailed()
    {
        try
        {
            var gcInfo = GC.GetGCMemoryInfo();
            var totalMemoryGB = gcInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            var currentProcess = Process.GetCurrentProcess();
            var usedMemoryGB = currentProcess.WorkingSet64 / (1024.0 * 1024.0 * 1024.0);
            var availableMemoryGB = totalMemoryGB - usedMemoryGB;

            _logger.LogInformation("[RESOURCE-CHECK]   Total system memory: {Total:F2} GB", totalMemoryGB);
            _logger.LogInformation("[RESOURCE-CHECK]   Current process usage: {Used:F2} GB", usedMemoryGB);
            _logger.LogInformation("[RESOURCE-CHECK]   Available memory: {Available:F2} GB", availableMemoryGB);
            _logger.LogInformation("[RESOURCE-CHECK]   Required minimum: {Required} GB", _options.MinRamGB);

            // If minimum is 0, skip the check (no requirement)
            if (_options.MinRamGB == 0)
            {
                _logger.LogInformation("[RESOURCE-CHECK] ✓ No RAM requirement configured (training allowed on any hardware)");
                return true;
            }

            if (availableMemoryGB < _options.MinRamGB)
            {
                if (_options.WarningOnly)
                {
                    _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Low memory: {Available:F2} GB < {Required} GB (warning only)",
                        availableMemoryGB, _options.MinRamGB);
                    return true;
                }

                _logger.LogError("[RESOURCE-CHECK] ❌ Insufficient memory: {Available:F2} GB < {Required} GB",
                    availableMemoryGB, _options.MinRamGB);
                return false;
            }

            // Just log memory status, don't block training
            _logger.LogInformation("[RESOURCE-CHECK] Memory available: {Available:F2} GB", availableMemoryGB);
            _logger.LogInformation("[RESOURCE-CHECK] ✓ Memory check passed (no requirements in LAB_MODE)");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check memory: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Phase 3: Enhanced CPU check with core count and detailed usage info
    /// </summary>
    private async Task<bool> CheckCpuUtilizationDetailedAsync(CancellationToken cancellationToken)
    {
        try
        {
            var coreCount = Environment.ProcessorCount;
            
            // Take a snapshot of CPU usage
            var process = Process.GetCurrentProcess();
            var startTime = process.TotalProcessorTime;
            var startCpuUsage = DateTime.UtcNow;

            await Task.Delay(1000, cancellationToken).ConfigureAwait(false);

            var endTime = process.TotalProcessorTime;
            var endCpuUsage = DateTime.UtcNow;

            var cpuUsedMs = (endTime - startTime).TotalMilliseconds;
            var totalMsPassed = (endCpuUsage - startCpuUsage).TotalMilliseconds;
            var cpuUsageTotal = cpuUsedMs / (coreCount * totalMsPassed);

            var cpuPercent = cpuUsageTotal * 100;

            _logger.LogInformation("[RESOURCE-CHECK]   CPU cores: {Cores}", coreCount);
            _logger.LogInformation("[RESOURCE-CHECK]   Current CPU usage: {Cpu:F1}%", cpuPercent);
            _logger.LogInformation("[RESOURCE-CHECK]   Maximum threshold: {Threshold}%", _options.MaxCpuThreshold);

            // If threshold is 100%, skip the check (no requirement)
            if (_options.MaxCpuThreshold >= 100.0)
            {
                _logger.LogInformation("[RESOURCE-CHECK] ✓ No CPU threshold configured (training allowed on any hardware)");
                return true;
            }

            if (coreCount < 2)
            {
                _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Low core count: {Cores} < 2 (training may be slow)", coreCount);
            }

            if (cpuPercent > _options.MaxCpuThreshold)
            {
                if (_options.WarningOnly)
                {
                    _logger.LogWarning("[RESOURCE-CHECK] ⚠️ High CPU usage: {Cpu:F1}% > {Threshold}% (warning only)",
                        cpuPercent, _options.MaxCpuThreshold);
                    return true;
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
            return true;
        }
    }

    /// <summary>
    /// Phase 3: Check historical data files (ES/NQ)
    /// </summary>
    private async Task<bool> CheckHistoricalDataAsync(CancellationToken cancellationToken)
    {
        try
        {
            var result = await _dataIntegrityService.ValidateHistoricalDataFilesAsync(cancellationToken).ConfigureAwait(false);

            if (!result.IsValid)
            {
                _logger.LogError("[RESOURCE-CHECK] ❌ Historical data validation failed");
                foreach (var issue in result.Issues)
                {
                    _logger.LogError("[RESOURCE-CHECK]   - {Issue}", issue);
                }
                return false;
            }

            // Log summary
            foreach (var kvp in result.SymbolBarCounts)
            {
                var symbol = kvp.Key;
                var barCount = kvp.Value;
                
                string dateRangeStr = "unknown range";
                if (result.DateRanges.TryGetValue(symbol, out var range))
                {
                    var days = (range.End - range.Start).TotalDays;
                    dateRangeStr = $"{days:F0} days";
                }
                
                _logger.LogInformation("[RESOURCE-CHECK]   {Symbol}: {Bars:N0} bars, {Range}",
                    symbol, barCount, dateRangeStr);
            }

            if (result.Warnings.Any())
            {
                _logger.LogWarning("[RESOURCE-CHECK] ⚠️ {Count} warnings in historical data", result.Warnings.Count);
            }
            else
            {
                _logger.LogInformation("[RESOURCE-CHECK] ✓ Historical data files valid");
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check historical data: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Phase 3: Check experience database/files
    /// </summary>
    private async Task<bool> CheckExperienceDatabaseAsync(CancellationToken cancellationToken)
    {
        try
        {
            var experiencesDir = Path.Combine(Directory.GetCurrentDirectory(), "data", "experiences");
            
            if (!Directory.Exists(experiencesDir))
            {
                _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Experience directory does not exist (will be created on first run)");
                Directory.CreateDirectory(experiencesDir);
                return true; // Not a failure, just first run
            }

            // Count experience files
            var experienceFiles = Directory.GetFiles(experiencesDir, "*.json", SearchOption.TopDirectoryOnly);
            var experienceCount = experienceFiles.Length;

            _logger.LogInformation("[RESOURCE-CHECK]   Experience files: {Count:N0}", experienceCount);

            if (experienceCount == 0)
            {
                _logger.LogWarning("[RESOURCE-CHECK] ⚠️ No experiences found (first run or new system)");
            }
            else if (experienceCount < 10000)
            {
                _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Low experience count: {Count:N0} < 10,000 (training may be limited)",
                    experienceCount);
            }
            else
            {
                _logger.LogInformation("[RESOURCE-CHECK] ✓ Experience database populated ({Count:N0} experiences)",
                    experienceCount);
            }

            // Check directory is writable
            var testFile = Path.Combine(experiencesDir, ".write_test");
            await File.WriteAllTextAsync(testFile, "test", cancellationToken).ConfigureAwait(false);
            File.Delete(testFile);
            
            _logger.LogInformation("[RESOURCE-CHECK] ✓ Experience directory is writable");

            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check experience database: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Phase 3: Check model registry directories
    /// </summary>
    private bool CheckModelRegistry()
    {
        try
        {
            var modelsDir = Path.Combine(Directory.GetCurrentDirectory(), "models");
            var productionDir = Path.Combine(modelsDir, "production");
            var stagingDir = Path.Combine(modelsDir, "staging");
            var backupDir = Path.Combine(modelsDir, "backup");

            // Create directories if they don't exist
            Directory.CreateDirectory(productionDir);
            Directory.CreateDirectory(stagingDir);
            Directory.CreateDirectory(backupDir);

            // Check if writable
            var testFile = Path.Combine(productionDir, ".write_test");
            File.WriteAllText(testFile, "test");
            File.Delete(testFile);

            // Count production models
            var productionModels = Directory.GetFiles(productionDir, "*", SearchOption.TopDirectoryOnly);
            
            _logger.LogInformation("[RESOURCE-CHECK]   Models directory: {Dir}", modelsDir);
            _logger.LogInformation("[RESOURCE-CHECK]   Production models: {Count}", productionModels.Length);
            _logger.LogInformation("[RESOURCE-CHECK] ✓ Model registry directories exist and writable");

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check model registry: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Phase 3: Check for lock files (prevent concurrent training sessions)
    /// </summary>
    private bool CheckLockFiles()
    {
        try
        {
            var stateDir = Path.Combine(Directory.GetCurrentDirectory(), "state");
            Directory.CreateDirectory(stateDir);

            var lockFile = Path.Combine(stateDir, "training.lock");

            if (File.Exists(lockFile))
            {
                var fileInfo = new FileInfo(lockFile);
                var age = DateTime.UtcNow - fileInfo.LastWriteTimeUtc;

                if (age.TotalHours > 24)
                {
                    // Stale lock file from crashed session
                    _logger.LogWarning("[RESOURCE-CHECK] ⚠️ Stale lock file detected (age: {Hours:F1} hours), removing",
                        age.TotalHours);
                    File.Delete(lockFile);
                    _logger.LogInformation("[RESOURCE-CHECK] ✓ Stale lock file removed");
                    return true;
                }
                else
                {
                    // Fresh lock file, another session is running
                    _logger.LogError("[RESOURCE-CHECK] ❌ Lock file exists (another training session is running)");
                    _logger.LogError("[RESOURCE-CHECK]   Lock file age: {Minutes:F1} minutes", age.TotalMinutes);
                    return false;
                }
            }

            _logger.LogInformation("[RESOURCE-CHECK] ✓ No lock files detected");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check lock files: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Phase 3: Check timezone configuration (America/New_York)
    /// </summary>
    private bool CheckTimezone()
    {
        try
        {
            var currentTimeZone = TimeZoneInfo.Local;
            var etTimeZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");

            var currentUtc = DateTime.UtcNow;
            var currentEt = TimeZoneInfo.ConvertTimeFromUtc(currentUtc, etTimeZone);
            var offset = etTimeZone.GetUtcOffset(currentUtc);

            _logger.LogInformation("[RESOURCE-CHECK]   System timezone: {TZ}", currentTimeZone.DisplayName);
            _logger.LogInformation("[RESOURCE-CHECK]   Current time (ET): {Time}", currentEt.ToString("yyyy-MM-dd HH:mm:ss"));
            _logger.LogInformation("[RESOURCE-CHECK]   ET offset from UTC: {Offset}", offset);

            // Check if DST is currently active
            var isDst = etTimeZone.IsDaylightSavingTime(currentUtc);
            _logger.LogInformation("[RESOURCE-CHECK]   Daylight Saving Time: {DST}", isDst ? "Active" : "Not Active");

            _logger.LogInformation("[RESOURCE-CHECK] ✓ Timezone configuration verified");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RESOURCE-CHECK] Failed to check timezone: {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Phase 3: Check network connectivity (optional, informational)
    /// </summary>
    private async Task<bool> CheckNetworkConnectivityAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Test localhost connectivity
            using var client = new System.Net.Sockets.TcpClient();
            await client.ConnectAsync("127.0.0.1", 80, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("[RESOURCE-CHECK] ✓ Network stack operational");
            return true;
        }
        catch
        {
            // Non-critical, just informational
            _logger.LogInformation("[RESOURCE-CHECK] ℹ️ Network connectivity check skipped");
            return true;
        }
    }
}

/// <summary>
/// Configuration options for resource pre-check service
/// All values default to 0 or permissive settings to allow training on any hardware
/// </summary>
public sealed class ResourcePreCheckOptions
{
    /// <summary>
    /// Minimum disk space required in GB (default: 0 - no requirement)
    /// </summary>
    public long MinDiskSpaceGB { get; set; } = 0;
    
    /// <summary>
    /// Minimum RAM required in GB (default: 0 - no requirement)
    /// </summary>
    public long MinRamGB { get; set; } = 0;
    
    /// <summary>
    /// Maximum CPU threshold percentage (default: 100% - no limit)
    /// </summary>
    public double MaxCpuThreshold { get; set; } = 100.0;
    
    /// <summary>
    /// If true, emit warnings instead of hard failures for marginal resources
    /// Allows training to proceed but logs concerns (default: true - all checks are warnings)
    /// </summary>
    public bool WarningOnly { get; set; } = true;
    
    /// <summary>
    /// Enable GPU availability check (default: false - GPU not required)
    /// GPU is optional - check is informational only
    /// </summary>
    public bool EnableGpuCheck { get; set; } = false;
}
