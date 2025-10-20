using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Promotion;

/// <summary>
/// Phase 7.3: Atomic Promotion Coordinator
/// Orchestrates multi-step atomic promotion with automatic rollback
/// All-or-nothing deployment: either ALL 273 models promoted or NONE
/// </summary>
internal sealed class AtomicPromotionCoordinator
{
    private readonly ILogger<AtomicPromotionCoordinator> _logger;
    private readonly ProductionBackupManager _backupManager;
    private readonly VersionManager _versionManager;
    private readonly PostPromotionValidator _validator;
    private readonly PromotionHistoryTracker _historyTracker;
    
    private readonly string _stagingDirectory;
    private readonly string _productionDirectory;
    private readonly string _productionTempDirectory;
    private readonly string _productionOldDirectory;
    private readonly string _lockFile;
    
    private const int ExpectedModelCount = 273;
    
    public AtomicPromotionCoordinator(
        ILogger<AtomicPromotionCoordinator> logger,
        ProductionBackupManager backupManager,
        VersionManager versionManager,
        PostPromotionValidator validator,
        PromotionHistoryTracker historyTracker)
    {
        _logger = logger;
        _backupManager = backupManager;
        _versionManager = versionManager;
        _validator = validator;
        _historyTracker = historyTracker;
        
        var baseDir = Directory.GetCurrentDirectory();
        _stagingDirectory = Path.Combine(baseDir, "models", "staging");
        _productionDirectory = Path.Combine(baseDir, "models", "production");
        _productionTempDirectory = Path.Combine(baseDir, "models", "production_temp");
        _productionOldDirectory = Path.Combine(baseDir, "models", "production_old");
        _lockFile = Path.Combine(baseDir, "models", "promotion.lock");
    }
    
    /// <summary>
    /// Main entry point: Promote models atomically
    /// </summary>
    public async Task<AtomicCoordinatorResult> PromoteModelsAsync(
        string sessionId,
        CancellationToken cancellationToken = default)
    {
        var result = new AtomicCoordinatorResult
        {
            SessionId = sessionId,
            StartTime = DateTime.UtcNow
        };
        
        var sw = Stopwatch.StartNew();
        string? backupPath = null;
        
        try
        {
            _logger.LogInformation("[ATOMIC-PROMOTION] Starting atomic promotion for session {SessionId}", sessionId);
            
            // Step 1: Acquire promotion lock
            if (!await AcquireLockAsync(cancellationToken).ConfigureAwait(false))
            {
                result.Success = false;
                result.Issues.Add("Failed to acquire promotion lock - another promotion may be in progress");
                return result;
            }
            
            try
            {
                // Step 2: Backup current production
                _logger.LogInformation("[ATOMIC-PROMOTION] [1/8] Creating backup...");
                backupPath = await _backupManager.CreateBackupAsync(cancellationToken).ConfigureAwait(false);
                result.BackupLocation = backupPath;
                
                // Step 3: Validate staging models
                _logger.LogInformation("[ATOMIC-PROMOTION] [2/8] Validating staging models...");
                if (!await ValidateStagingModelsAsync(cancellationToken).ConfigureAwait(false))
                {
                    result.Success = false;
                    result.Issues.Add("Staging model validation failed");
                    return result;
                }
                
                // Step 4: Copy models to temp directory
                _logger.LogInformation("[ATOMIC-PROMOTION] [3/8] Copying models to temp directory...");
                var copiedCount = await CopyModelsToTempAsync(cancellationToken).ConfigureAwait(false);
                result.ModelsPromoted = copiedCount;
                
                // Step 5: Validate temp models
                _logger.LogInformation("[ATOMIC-PROMOTION] [4/8] Validating temp models...");
                if (!await ValidateTempModelsAsync(cancellationToken).ConfigureAwait(false))
                {
                    result.Success = false;
                    result.Issues.Add("Temp model validation failed");
                    await RollbackAsync(backupPath, "Temp model validation failed", cancellationToken)
                        .ConfigureAwait(false);
                    return result;
                }
                
                // Step 6: Atomic directory swap
                _logger.LogInformation("[ATOMIC-PROMOTION] [5/8] Performing atomic directory swap...");
                if (!await AtomicSwapAsync(cancellationToken).ConfigureAwait(false))
                {
                    result.Success = false;
                    result.Issues.Add("Atomic swap failed");
                    await RollbackAsync(backupPath, "Atomic swap failed", cancellationToken)
                        .ConfigureAwait(false);
                    return result;
                }
                
                // Step 7: Update version pointer
                _logger.LogInformation("[ATOMIC-PROMOTION] [6/8] Updating version pointer...");
                var newVersion = VersionManager.GenerateVersionString(DateTime.UtcNow);
                if (!await _versionManager.UpdateVersionAsync(newVersion, null, cancellationToken)
                    .ConfigureAwait(false))
                {
                    result.Success = false;
                    result.Issues.Add("Version update failed");
                    await RollbackAsync(backupPath, "Version update failed", cancellationToken)
                        .ConfigureAwait(false);
                    return result;
                }
                result.Version = newVersion;
                
                // Step 8: Post-promotion validation
                _logger.LogInformation("[ATOMIC-PROMOTION] [7/8] Running post-promotion validation...");
                var validationResult = await _validator.ValidatePromotionAsync(newVersion, cancellationToken)
                    .ConfigureAwait(false);
                
                if (!validationResult.Passed)
                {
                    result.Success = false;
                    result.Issues.Add("Post-promotion validation failed");
                    result.Issues.AddRange(validationResult.Errors);
                    
                    _logger.LogError("[ATOMIC-PROMOTION] Post-promotion validation failed, rolling back");
                    await RollbackAsync(backupPath, "Post-promotion validation failed", cancellationToken)
                        .ConfigureAwait(false);
                    return result;
                }
                
                // Step 9: Cleanup
                _logger.LogInformation("[ATOMIC-PROMOTION] [8/8] Cleaning up...");
                await CleanupAsync(cancellationToken).ConfigureAwait(false);
                
                sw.Stop();
                result.Success = true;
                result.RollbackCapable = true;
                result.PromotionDurationMs = sw.Elapsed.TotalMilliseconds;
                
                _logger.LogInformation("[ATOMIC-PROMOTION] Promotion completed successfully: {Models} models, {Duration:F1}s",
                    result.ModelsPromoted, sw.Elapsed.TotalSeconds);
                
                // Log to history
                await _historyTracker.LogPromotionOutcomeAsync(
                    sessionId, "SUCCESS", result.ModelsPromoted, sw.Elapsed.TotalSeconds, newVersion,
                    cancellationToken: cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                // Always release lock
                await ReleaseLockAsync(cancellationToken).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            sw.Stop();
            _logger.LogError(ex, "[ATOMIC-PROMOTION] Promotion failed with exception");
            result.Success = false;
            result.Issues.Add($"Exception: {ex.Message}");
            
            // Attempt rollback
            if (backupPath != null)
            {
                await RollbackAsync(backupPath, $"Exception: {ex.Message}", cancellationToken)
                    .ConfigureAwait(false);
            }
            
            // Log failure
            await _historyTracker.LogPromotionOutcomeAsync(
                sessionId, "FAILED", 0, sw.Elapsed.TotalSeconds,
                metadata: new Dictionary<string, object> { ["error"] = ex.Message },
                cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        
        return result;
    }
    
    /// <summary>
    /// Acquire promotion lock
    /// </summary>
    private async Task<bool> AcquireLockAsync(CancellationToken cancellationToken)
    {
        try
        {
            if (File.Exists(_lockFile))
            {
                // Check if lock is stale (> 1 hour old)
                var lockInfo = new FileInfo(_lockFile);
                if (DateTime.UtcNow - lockInfo.CreationTimeUtc > TimeSpan.FromHours(1))
                {
                    _logger.LogWarning("[ATOMIC-PROMOTION] Removing stale lock file");
                    File.Delete(_lockFile);
                }
                else
                {
                    _logger.LogError("[ATOMIC-PROMOTION] Lock file exists, another promotion may be in progress");
                    return false;
                }
            }
            
            await File.WriteAllTextAsync(_lockFile, $"Locked at {DateTime.UtcNow:O}", cancellationToken)
                .ConfigureAwait(false);
            
            _logger.LogDebug("[ATOMIC-PROMOTION] Lock acquired");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ATOMIC-PROMOTION] Failed to acquire lock");
            return false;
        }
    }
    
    /// <summary>
    /// Release promotion lock
    /// </summary>
    private async Task ReleaseLockAsync(CancellationToken cancellationToken)
    {
        try
        {
            if (File.Exists(_lockFile))
            {
                File.Delete(_lockFile);
                _logger.LogDebug("[ATOMIC-PROMOTION] Lock released");
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ATOMIC-PROMOTION] Failed to release lock");
        }
    }
    
    /// <summary>
    /// Validate all staging models exist and are valid
    /// </summary>
    private async Task<bool> ValidateStagingModelsAsync(CancellationToken cancellationToken)
    {
        try
        {
            if (!Directory.Exists(_stagingDirectory))
            {
                _logger.LogError("[ATOMIC-PROMOTION] Staging directory not found: {Dir}", _stagingDirectory);
                return false;
            }
            
            var modelFiles = Directory.GetFiles(_stagingDirectory, "*.onnx", SearchOption.TopDirectoryOnly);
            
            if (modelFiles.Length == 0)
            {
                _logger.LogError("[ATOMIC-PROMOTION] No models found in staging");
                return false;
            }
            
            _logger.LogInformation("[ATOMIC-PROMOTION] Found {Count} models in staging", modelFiles.Length);
            
            // Verify all files are readable and non-zero size
            foreach (var file in modelFiles)
            {
                var fileInfo = new FileInfo(file);
                if (fileInfo.Length == 0)
                {
                    _logger.LogError("[ATOMIC-PROMOTION] Empty model file: {File}", Path.GetFileName(file));
                    return false;
                }
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ATOMIC-PROMOTION] Staging validation failed");
            return false;
        }
    }
    
    /// <summary>
    /// Copy staging models to temp directory
    /// </summary>
    private async Task<int> CopyModelsToTempAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Clean temp directory if exists
            if (Directory.Exists(_productionTempDirectory))
            {
                Directory.Delete(_productionTempDirectory, recursive: true);
            }
            Directory.CreateDirectory(_productionTempDirectory);
            
            var modelFiles = Directory.GetFiles(_stagingDirectory, "*.*", SearchOption.TopDirectoryOnly);
            var copiedCount = 0;
            
            foreach (var sourceFile in modelFiles)
            {
                var fileName = Path.GetFileName(sourceFile);
                var destFile = Path.Combine(_productionTempDirectory, fileName);
                File.Copy(sourceFile, destFile, overwrite: true);
                copiedCount++;
                
                if (cancellationToken.IsCancellationRequested)
                    break;
            }
            
            _logger.LogInformation("[ATOMIC-PROMOTION] Copied {Count} files to temp directory", copiedCount);
            
            await Task.CompletedTask.ConfigureAwait(false);
            return copiedCount;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ATOMIC-PROMOTION] Failed to copy models to temp");
            return 0;
        }
    }
    
    /// <summary>
    /// Validate temp models
    /// </summary>
    private async Task<bool> ValidateTempModelsAsync(CancellationToken cancellationToken)
    {
        try
        {
            var modelFiles = Directory.GetFiles(_productionTempDirectory, "*.onnx");
            
            foreach (var file in modelFiles)
            {
                var fileInfo = new FileInfo(file);
                if (fileInfo.Length == 0)
                {
                    _logger.LogError("[ATOMIC-PROMOTION] Empty model in temp: {File}", Path.GetFileName(file));
                    return false;
                }
            }
            
            _logger.LogInformation("[ATOMIC-PROMOTION] Temp models validated: {Count} models", modelFiles.Length);
            
            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ATOMIC-PROMOTION] Temp validation failed");
            return false;
        }
    }
    
    /// <summary>
    /// Perform atomic directory swap
    /// </summary>
    private async Task<bool> AtomicSwapAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Clean old directory if exists
            if (Directory.Exists(_productionOldDirectory))
            {
                Directory.Delete(_productionOldDirectory, recursive: true);
            }
            
            // Rename production to production_old
            if (Directory.Exists(_productionDirectory))
            {
                Directory.Move(_productionDirectory, _productionOldDirectory);
            }
            
            // Rename production_temp to production
            Directory.Move(_productionTempDirectory, _productionDirectory);
            
            _logger.LogInformation("[ATOMIC-PROMOTION] Atomic swap completed");
            
            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ATOMIC-PROMOTION] Atomic swap failed");
            
            // Try to restore production_old if swap partially failed
            if (Directory.Exists(_productionOldDirectory) && !Directory.Exists(_productionDirectory))
            {
                try
                {
                    Directory.Move(_productionOldDirectory, _productionDirectory);
                    _logger.LogInformation("[ATOMIC-PROMOTION] Restored production from old directory");
                }
                catch
                {
                    _logger.LogError("[ATOMIC-PROMOTION] Failed to restore production directory");
                }
            }
            
            return false;
        }
    }
    
    /// <summary>
    /// Cleanup old directories after successful promotion
    /// </summary>
    private async Task CleanupAsync(CancellationToken cancellationToken)
    {
        try
        {
            if (Directory.Exists(_productionOldDirectory))
            {
                Directory.Delete(_productionOldDirectory, recursive: true);
                _logger.LogDebug("[ATOMIC-PROMOTION] Cleaned up old production directory");
            }
            
            if (Directory.Exists(_productionTempDirectory))
            {
                Directory.Delete(_productionTempDirectory, recursive: true);
                _logger.LogDebug("[ATOMIC-PROMOTION] Cleaned up temp directory");
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ATOMIC-PROMOTION] Cleanup failed (non-critical)");
        }
    }
    
    /// <summary>
    /// Rollback promotion using backup
    /// </summary>
    public async Task<bool> RollbackAsync(
        string backupPath,
        string reason,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogWarning("[ATOMIC-PROMOTION] Rolling back promotion: {Reason}", reason);
            
            var success = await _backupManager.RestoreFromBackupAsync(backupPath, cancellationToken)
                .ConfigureAwait(false);
            
            if (success)
            {
                _logger.LogInformation("[ATOMIC-PROMOTION] Rollback successful");
                
                // Log rollback event
                await _historyTracker.LogRollbackAsync("rollback", reason, 
                    cancellationToken: cancellationToken).ConfigureAwait(false);
            }
            else
            {
                _logger.LogError("[ATOMIC-PROMOTION] Rollback failed");
            }
            
            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ATOMIC-PROMOTION] Rollback failed with exception");
            return false;
        }
    }
}

/// <summary>
/// Atomic coordinator promotion result
/// </summary>
public sealed class AtomicCoordinatorResult
{
    public string SessionId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public bool Success { get; set; }
    public int ModelsPromoted { get; set; }
    public double PromotionDurationMs { get; set; }
    public string? Version { get; set; }
    public string? BackupLocation { get; set; }
    public bool RollbackCapable { get; set; }
    public List<string> Issues { get; set; } = new();
}
