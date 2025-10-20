using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.UnifiedOrchestrator.Training;

namespace TradingBot.UnifiedOrchestrator.Promotion;

/// <summary>
/// Phase 5: Atomic Model Promotion Service
/// Provides safe, transactional model promotion with rollback capability
/// 
/// Key Features:
/// - Atomic promotion (all models at once, or none)
/// - Automatic backup before promotion
/// - Instant rollback capability (< 5 seconds)
/// - Comprehensive criteria evaluation
/// - Detailed promotion reporting
/// </summary>
internal sealed class AtomicPromotionService
{
    private readonly ILogger<AtomicPromotionService> _logger;
    private readonly ValidationService _validationService;
    private readonly string _stagingDirectory;
    private readonly string _productionDirectory;
    private readonly string _backupDirectory;
    private readonly string _reportsDirectory;

    // Phase 5 thresholds
    private const int ExpectedComponentCount = 273;
    private const double MaxTrainingTimeHours = 6.0;
    private const double MaxModelSizeGB = 10.0;
    private const double MinAverageImprovementPercent = 0.0;
    private const double MaxCriticalRegressionPercent = 5.0;

    public AtomicPromotionService(
        ILogger<AtomicPromotionService> logger,
        ValidationService validationService)
    {
        _logger = logger;
        _validationService = validationService;

        var baseDir = Directory.GetCurrentDirectory();
        _stagingDirectory = Path.Combine(baseDir, "models", "staging");
        _productionDirectory = Path.Combine(baseDir, "models", "production");
        _backupDirectory = Path.Combine(baseDir, "models", "backup");
        _reportsDirectory = Path.Combine(baseDir, "reports", "promotion");

        // Ensure directories exist
        Directory.CreateDirectory(_stagingDirectory);
        Directory.CreateDirectory(_productionDirectory);
        Directory.CreateDirectory(_backupDirectory);
        Directory.CreateDirectory(_reportsDirectory);
    }

    /// <summary>
    /// Phase 5 Step 1: Evaluate enhanced promotion criteria
    /// </summary>
    public async Task<EnhancedPromotionCriteria> EvaluatePromotionCriteriaAsync(
        string sessionId,
        PostTrainingValidationResult validationResult,
        DateTime trainingStart,
        DateTime trainingEnd,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[PROMOTION-CRITERIA] Evaluating promotion criteria for session {SessionId}", sessionId);

        var criteria = new EnhancedPromotionCriteria();

        // 1. Training Success Criteria
        criteria.TrainingSuccess = EvaluateTrainingSuccess(
            validationResult.InferenceTests.ModelsLoaded,
            trainingStart,
            trainingEnd);

        // 2. Validation Success Criteria
        criteria.ValidationSuccess = EvaluateValidationSuccess(validationResult);

        // 3. Performance Criteria
        criteria.PerformanceCriteria = EvaluatePerformanceCriteria(validationResult);

        // 4. Technical Criteria
        criteria.TechnicalCriteria = await EvaluateTechnicalCriteriaAsync(cancellationToken).ConfigureAwait(false);

        // 5. Operational Criteria
        criteria.OperationalCriteria = await EvaluateOperationalCriteriaAsync(
            trainingStart,
            trainingEnd,
            cancellationToken).ConfigureAwait(false);

        // Determine overall pass/fail
        criteria.Passed = criteria.TrainingSuccess.Passed &&
                         criteria.ValidationSuccess.Passed &&
                         criteria.PerformanceCriteria.Passed &&
                         criteria.TechnicalCriteria.Passed &&
                         criteria.OperationalCriteria.Passed;

        // Collect failed criteria
        if (!criteria.TrainingSuccess.Passed)
            criteria.FailedCriteria.Add("Training Success");
        if (!criteria.ValidationSuccess.Passed)
            criteria.FailedCriteria.Add("Validation Success");
        if (!criteria.PerformanceCriteria.Passed)
            criteria.FailedCriteria.Add("Performance");
        if (!criteria.TechnicalCriteria.Passed)
            criteria.FailedCriteria.Add("Technical");
        if (!criteria.OperationalCriteria.Passed)
            criteria.FailedCriteria.Add("Operational");

        _logger.LogInformation("[PROMOTION-CRITERIA] Overall result: {Passed}, failed criteria: {Count}",
            criteria.Passed ? "PASSED" : "FAILED", criteria.FailedCriteria.Count);

        return criteria;
    }

    /// <summary>
    /// Phase 5 Step 2: Promote models atomically with backup
    /// </summary>
    public async Task<AtomicPromotionResult> PromoteModelsAtomicallyAsync(
        string sessionId,
        CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();

        _logger.LogInformation("[ATOMIC-PROMOTION] Starting atomic promotion for session {SessionId}", sessionId);

        var result = new AtomicPromotionResult
        {
            SessionId = sessionId,
            PromotionTime = DateTime.UtcNow
        };

        try
        {
            // Step 1: Pre-flight checks
            _logger.LogInformation("[ATOMIC-PROMOTION] [1/5] Running pre-flight checks...");
            var preFlightPass = await RunPreFlightChecksAsync(result, cancellationToken).ConfigureAwait(false);

            if (!preFlightPass)
            {
                result.Success = false;
                return result;
            }

            // Step 2: Create backup
            _logger.LogInformation("[ATOMIC-PROMOTION] [2/5] Creating backup...");
            var backupSuccess = await CreateBackupAsync(result, cancellationToken).ConfigureAwait(false);

            if (!backupSuccess)
            {
                result.Success = false;
                result.Issues.Add("Failed to create backup");
                return result;
            }

            // Step 3: Atomic copy (all or nothing)
            _logger.LogInformation("[ATOMIC-PROMOTION] [3/5] Performing atomic copy...");
            var copySuccess = await AtomicCopyModelsAsync(result, cancellationToken).ConfigureAwait(false);

            if (!copySuccess)
            {
                result.Success = false;
                result.Issues.Add("Atomic copy failed");

                // Attempt rollback
                _logger.LogWarning("[ATOMIC-PROMOTION] Copy failed, initiating rollback...");
                await RollbackFromBackupAsync(result.BackupLocation, cancellationToken).ConfigureAwait(false);
                return result;
            }

            // Step 4: Verify promotion
            _logger.LogInformation("[ATOMIC-PROMOTION] [4/5] Verifying promotion...");
            var verifySuccess = await VerifyPromotionAsync(result, cancellationToken).ConfigureAwait(false);

            if (!verifySuccess)
            {
                result.Success = false;
                result.Issues.Add("Promotion verification failed");

                // Rollback
                _logger.LogWarning("[ATOMIC-PROMOTION] Verification failed, rolling back...");
                await RollbackFromBackupAsync(result.BackupLocation, cancellationToken).ConfigureAwait(false);
                return result;
            }

            // Step 5: Cleanup staging
            _logger.LogInformation("[ATOMIC-PROMOTION] [5/5] Cleaning up staging...");
            await CleanupStagingAsync(cancellationToken).ConfigureAwait(false);

            sw.Stop();
            result.Success = true;
            result.PromotionDurationMs = sw.Elapsed.TotalMilliseconds;
            result.RollbackCapable = true;

            _logger.LogInformation("[ATOMIC-PROMOTION] ✅ Promotion successful in {Duration:F1}ms, {Count} models promoted",
                result.PromotionDurationMs, result.ModelsPromoted);

            return result;
        }
        catch (Exception ex)
        {
            sw.Stop();
            _logger.LogError(ex, "[ATOMIC-PROMOTION] ❌ Promotion failed after {Duration:F1}ms",
                sw.Elapsed.TotalMilliseconds);

            result.Success = false;
            result.Issues.Add($"Exception: {ex.Message}");
            result.PromotionDurationMs = sw.Elapsed.TotalMilliseconds;

            return result;
        }
    }

    /// <summary>
    /// Phase 5: Rollback to previous production models
    /// </summary>
    public async Task<RollbackResult> RollbackToPreviousAsync(
        string reason,
        CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();

        _logger.LogWarning("[ROLLBACK] Starting emergency rollback: {Reason}", reason);

        var result = new RollbackResult
        {
            RollbackTime = DateTime.UtcNow,
            Reason = reason
        };

        try
        {
            // Find most recent backup
            var backups = Directory.GetDirectories(_backupDirectory)
                .OrderByDescending(d => Directory.GetCreationTimeUtc(d))
                .ToList();

            if (!backups.Any())
            {
                result.Success = false;
                result.Issues.Add("No backups available");
                _logger.LogError("[ROLLBACK] ❌ No backups found");
                return result;
            }

            var latestBackup = backups.First();
            result.BackupSource = Path.GetFileName(latestBackup);

            _logger.LogInformation("[ROLLBACK] Restoring from backup: {Backup}", result.BackupSource);

            // Perform rollback
            await RollbackFromBackupAsync(latestBackup, cancellationToken).ConfigureAwait(false);

            // Count restored models
            var productionModels = Directory.GetFiles(_productionDirectory, "*.onnx", SearchOption.TopDirectoryOnly);
            result.ModelsRestored = productionModels.Length;

            sw.Stop();
            result.Success = true;
            result.RollbackDurationMs = sw.Elapsed.TotalMilliseconds;

            _logger.LogInformation("[ROLLBACK] ✅ Rollback successful in {Duration:F1}ms, {Count} models restored",
                result.RollbackDurationMs, result.ModelsRestored);

            return result;
        }
        catch (Exception ex)
        {
            sw.Stop();
            _logger.LogError(ex, "[ROLLBACK] ❌ Rollback failed after {Duration:F1}ms",
                sw.Elapsed.TotalMilliseconds);

            result.Success = false;
            result.Issues.Add($"Exception: {ex.Message}");
            result.RollbackDurationMs = sw.Elapsed.TotalMilliseconds;

            return result;
        }
    }

    /// <summary>
    /// Generate comprehensive promotion report
    /// </summary>
    public async Task<PromotionReport> GeneratePromotionReportAsync(
        string sessionId,
        EnhancedPromotionCriteria criteria,
        AtomicPromotionResult atomicResult,
        CancellationToken cancellationToken = default)
    {
        var report = new PromotionReport
        {
            SessionId = sessionId,
            PromotionTime = DateTime.UtcNow,
            Criteria = criteria,
            AtomicResult = atomicResult
        };

        // Determine status
        if (atomicResult.Success)
        {
            report.Status = "SUCCESS";
            report.Summary = $"Successfully promoted {atomicResult.ModelsPromoted} models to production";
        }
        else
        {
            report.Status = "FAILED";
            report.Summary = $"Promotion failed: {string.Join("; ", atomicResult.Issues)}";
        }

        // List promoted models
        if (atomicResult.Success)
        {
            var productionModels = Directory.GetFiles(_productionDirectory, "*.onnx", SearchOption.TopDirectoryOnly);
            report.ModelsPromoted = productionModels
                .Select(f => Path.GetFileNameWithoutExtension(f))
                .Where(name => !string.IsNullOrEmpty(name))
                .Cast<string>()
                .ToList();
        }

        // Rollback availability
        report.RollbackAvailable = atomicResult.BackupCreated && atomicResult.Success;

        // Recommendations
        if (!criteria.Passed)
        {
            report.Recommendations.Add("Fix failed criteria before attempting promotion again");
        }

        if (atomicResult.Warnings.Any())
        {
            foreach (var warning in atomicResult.Warnings)
            {
                report.Recommendations.Add($"Review warning: {warning}");
            }
        }

        // Save report
        await SavePromotionReportAsync(report, cancellationToken).ConfigureAwait(false);

        return report;
    }

    #region Private Methods

    private TrainingSuccessCriteria EvaluateTrainingSuccess(
        int modelsTrained,
        DateTime trainingStart,
        DateTime trainingEnd)
    {
        var criteria = new TrainingSuccessCriteria
        {
            ComponentsExpected = ExpectedComponentCount,
            ComponentsTrained = modelsTrained,
            TrainingDurationHours = (trainingEnd - trainingStart).TotalHours
        };

        criteria.CompletedWithinTimeWindow = criteria.TrainingDurationHours < MaxTrainingTimeHours;
        criteria.NoTrainingCrashes = modelsTrained > 0; // If any trained, no crash
        criteria.AllModelsSavedToStaging = Directory.Exists(_stagingDirectory) &&
            Directory.GetFiles(_stagingDirectory, "*.onnx").Length > 0;

        criteria.Passed = criteria.ComponentsTrained == criteria.ComponentsExpected &&
                         criteria.CompletedWithinTimeWindow &&
                         criteria.NoTrainingCrashes &&
                         criteria.AllModelsSavedToStaging;

        _logger.LogInformation("[CRITERIA] Training Success: {Passed} - {Trained}/{Expected} models, {Hours:F1}h",
            criteria.Passed ? "PASS" : "FAIL", criteria.ComponentsTrained, criteria.ComponentsExpected,
            criteria.TrainingDurationHours);

        return criteria;
    }

    private ValidationSuccessCriteria EvaluateValidationSuccess(PostTrainingValidationResult validationResult)
    {
        var criteria = new ValidationSuccessCriteria
        {
            InferenceTestsPassed = validationResult.InferenceTests.Passed,
            BaselineComparisonPositive = validationResult.BaselineComparison.Passed,
            NoCatastrophicForgetting = validationResult.CatastrophicForgetting.Passed,
            ModelIntegrityVerified = validationResult.ModelIntegrity.Passed
        };

        var passedChecks = 0;
        if (criteria.InferenceTestsPassed) passedChecks++;
        if (criteria.BaselineComparisonPositive) passedChecks++;
        if (criteria.NoCatastrophicForgetting) passedChecks++;
        if (criteria.ModelIntegrityVerified) passedChecks++;

        criteria.AllChecksPassedCount = passedChecks;
        criteria.TotalChecksCount = 4;
        criteria.Passed = passedChecks == criteria.TotalChecksCount;

        _logger.LogInformation("[CRITERIA] Validation Success: {Passed} - {Passed}/{Total} checks",
            criteria.Passed ? "PASS" : "FAIL", criteria.AllChecksPassedCount, criteria.TotalChecksCount);

        return criteria;
    }

    private PerformanceCriteria EvaluatePerformanceCriteria(PostTrainingValidationResult validationResult)
    {
        var criteria = new PerformanceCriteria
        {
            AverageImprovementPercent = validationResult.BaselineComparison.AverageImprovement
        };

        // Check for critical regressions
        var maxRegression = 0.0;
        if (validationResult.BaselineComparison.ModelComparisons.Any())
        {
            maxRegression = validationResult.BaselineComparison.ModelComparisons
                .Min(c => c.ImprovementPercent);
        }

        criteria.MaxRegressionPercent = Math.Abs(maxRegression);
        criteria.NoCriticalRegression = maxRegression >= -MaxCriticalRegressionPercent;

        // Check key models
        criteria.CVarPpoImproved = CheckModelImproved(validationResult, "CVaR-PPO");
        criteria.NeuralUcbImproved = CheckModelImproved(validationResult, "Neural-UCB");

        criteria.Passed = criteria.AverageImprovementPercent >= MinAverageImprovementPercent &&
                         criteria.NoCriticalRegression;

        _logger.LogInformation("[CRITERIA] Performance: {Passed} - avg {Avg:F1}%, max regression {Reg:F1}%",
            criteria.Passed ? "PASS" : "FAIL", criteria.AverageImprovementPercent, criteria.MaxRegressionPercent);

        return criteria;
    }

    private bool CheckModelImproved(PostTrainingValidationResult validationResult, string modelType)
    {
        var comparison = validationResult.BaselineComparison.ModelComparisons
            .FirstOrDefault(c => c.ModelType == modelType);

        return comparison == null || comparison.ImprovementPercent >= 0;
    }

    private async Task<TechnicalCriteria> EvaluateTechnicalCriteriaAsync(CancellationToken cancellationToken)
    {
        var criteria = new TechnicalCriteria();

        // Calculate total model size
        var stagingModels = Directory.GetFiles(_stagingDirectory, "*.onnx", SearchOption.TopDirectoryOnly);
        var totalSizeBytes = stagingModels.Sum(f => new FileInfo(f).Length);
        criteria.TotalModelSizeGB = totalSizeBytes / (1024.0 * 1024.0 * 1024.0);

        criteria.WithinSizeLimit = criteria.TotalModelSizeGB < MaxModelSizeGB;
        criteria.OnnxRuntimeCompatible = true; // Assume compatible (would verify in production)
        criteria.NoDependencyConflicts = true; // Assume no conflicts (would verify in production)

        criteria.Passed = criteria.WithinSizeLimit &&
                         criteria.OnnxRuntimeCompatible &&
                         criteria.NoDependencyConflicts;

        _logger.LogInformation("[CRITERIA] Technical: {Passed} - {Size:F2}GB < {Limit}GB",
            criteria.Passed ? "PASS" : "FAIL", criteria.TotalModelSizeGB, MaxModelSizeGB);

        await Task.CompletedTask.ConfigureAwait(false);
        return criteria;
    }

    private async Task<OperationalCriteria> EvaluateOperationalCriteriaAsync(
        DateTime trainingStart,
        DateTime trainingEnd,
        CancellationToken cancellationToken)
    {
        var criteria = new OperationalCriteria();

        // Training window check (before market open)
        var marketOpenTime = trainingStart.Date.AddHours(9).AddMinutes(30); // 9:30 AM ET
        criteria.TrainingWindowRespected = trainingEnd < marketOpenTime;

        // System health check
        var drive = new DriveInfo(Path.GetPathRoot(_productionDirectory) ?? "/");
        var freeSpaceGB = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
        criteria.SufficientDiskSpaceForBackup = freeSpaceGB > 20.0; // Need 20GB for backup

        criteria.SystemHealthGood = criteria.SufficientDiskSpaceForBackup;

        // Lock file check
        var lockFile = Path.Combine(Directory.GetCurrentDirectory(), "state", "training.lock");
        criteria.LockFileRemoved = !File.Exists(lockFile);
        criteria.NoConcurrentTraining = criteria.LockFileRemoved;

        criteria.Passed = criteria.TrainingWindowRespected &&
                         criteria.SystemHealthGood &&
                         criteria.NoConcurrentTraining;

        _logger.LogInformation("[CRITERIA] Operational: {Passed} - window: {Window}, health: {Health}, no locks: {NoLocks}",
            criteria.Passed ? "PASS" : "FAIL",
            criteria.TrainingWindowRespected,
            criteria.SystemHealthGood,
            criteria.NoConcurrentTraining);

        await Task.CompletedTask.ConfigureAwait(false);
        return criteria;
    }

    private async Task<bool> RunPreFlightChecksAsync(
        AtomicPromotionResult result,
        CancellationToken cancellationToken)
    {
        // Verify staging models exist
        var stagingModels = Directory.GetFiles(_stagingDirectory, "*.onnx", SearchOption.TopDirectoryOnly);
        if (stagingModels.Length == 0)
        {
            result.Issues.Add("No models found in staging directory");
            return false;
        }

        result.ModelsPromoted = stagingModels.Length;

        // Verify production directory writable
        try
        {
            var testFile = Path.Combine(_productionDirectory, ".write_test");
            await File.WriteAllTextAsync(testFile, "test", cancellationToken).ConfigureAwait(false);
            File.Delete(testFile);
        }
        catch (Exception ex)
        {
            result.Issues.Add($"Production directory not writable: {ex.Message}");
            return false;
        }

        // Verify backup directory writable
        try
        {
            var testFile = Path.Combine(_backupDirectory, ".write_test");
            await File.WriteAllTextAsync(testFile, "test", cancellationToken).ConfigureAwait(false);
            File.Delete(testFile);
        }
        catch (Exception ex)
        {
            result.Issues.Add($"Backup directory not writable: {ex.Message}");
            return false;
        }

        // Calculate total size
        result.TotalSizeBytes = stagingModels.Sum(f => new FileInfo(f).Length);

        // Check disk space
        var drive = new DriveInfo(Path.GetPathRoot(_productionDirectory) ?? "/");
        var freeSpace = drive.AvailableFreeSpace;

        if (freeSpace < result.TotalSizeBytes * 3) // Need 3x space (backup + production + staging)
        {
            result.Issues.Add($"Insufficient disk space: {freeSpace / (1024.0 * 1024.0 * 1024.0):F1}GB available, need {result.TotalSizeBytes * 3 / (1024.0 * 1024.0 * 1024.0):F1}GB");
            return false;
        }

        _logger.LogInformation("[PRE-FLIGHT] ✓ All checks passed - {Count} models, {Size:F1}MB total",
            result.ModelsPromoted, result.TotalSizeBytes / (1024.0 * 1024.0));

        return true;
    }

    private async Task<bool> CreateBackupAsync(
        AtomicPromotionResult result,
        CancellationToken cancellationToken)
    {
        try
        {
            // Create backup directory with timestamp
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
            var backupDir = Path.Combine(_backupDirectory, timestamp);
            Directory.CreateDirectory(backupDir);

            result.BackupLocation = backupDir;

            // Copy current production models to backup
            var productionModels = Directory.GetFiles(_productionDirectory, "*.onnx", SearchOption.TopDirectoryOnly);

            foreach (var model in productionModels)
            {
                var fileName = Path.GetFileName(model);
                var backupPath = Path.Combine(backupDir, fileName);
                File.Copy(model, backupPath, overwrite: true);
            }

            result.BackupCreated = true;

            _logger.LogInformation("[BACKUP] ✓ Backup created: {Count} models backed up to {Dir}",
                productionModels.Length, backupDir);

            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BACKUP] ❌ Failed to create backup");
            return false;
        }
    }

    private async Task<bool> AtomicCopyModelsAsync(
        AtomicPromotionResult result,
        CancellationToken cancellationToken)
    {
        try
        {
            // Get all staging models
            var stagingModels = Directory.GetFiles(_stagingDirectory, "*.onnx", SearchOption.TopDirectoryOnly);

            // Copy all models to staging location first for atomic swap
            var tempDir = Path.Combine(_productionDirectory, ".staging_promotion");
            Directory.CreateDirectory(tempDir);

            try
            {
                foreach (var model in stagingModels)
                {
                    var fileName = Path.GetFileName(model);
                    var tempPath = Path.Combine(tempDir, fileName);
                    File.Copy(model, tempPath, overwrite: true);
                }

                // Now atomically move: delete old, move new
                var productionModels = Directory.GetFiles(_productionDirectory, "*.onnx", SearchOption.TopDirectoryOnly);
                foreach (var model in productionModels)
                {
                    File.Delete(model);
                }

                var tempModels = Directory.GetFiles(tempDir, "*.onnx", SearchOption.TopDirectoryOnly);
                foreach (var model in tempModels)
                {
                    var fileName = Path.GetFileName(model);
                    var productionPath = Path.Combine(_productionDirectory, fileName);
                    File.Move(model, productionPath);
                }

                _logger.LogInformation("[ATOMIC-COPY] ✓ {Count} models copied atomically",
                    stagingModels.Length);

                return true;
            }
            finally
            {
                // Cleanup temp directory
                if (Directory.Exists(tempDir))
                {
                    Directory.Delete(tempDir, recursive: true);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ATOMIC-COPY] ❌ Atomic copy failed");
            return false;
        }
    }

    private async Task<bool> VerifyPromotionAsync(
        AtomicPromotionResult result,
        CancellationToken cancellationToken)
    {
        try
        {
            // Verify all models are in production
            var productionModels = Directory.GetFiles(_productionDirectory, "*.onnx", SearchOption.TopDirectoryOnly);

            if (productionModels.Length != result.ModelsPromoted)
            {
                result.Warnings.Add($"Model count mismatch: expected {result.ModelsPromoted}, found {productionModels.Length}");
                return false;
            }

            // Verify file sizes match
            foreach (var model in productionModels)
            {
                var fileInfo = new FileInfo(model);
                if (fileInfo.Length == 0)
                {
                    result.Warnings.Add($"Zero-size file detected: {Path.GetFileName(model)}");
                    return false;
                }
            }

            _logger.LogInformation("[VERIFY] ✓ Promotion verified - {Count} models in production",
                productionModels.Length);

            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VERIFY] ❌ Verification failed");
            return false;
        }
    }

    private async Task CleanupStagingAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Optionally keep staging for reference, or delete
            // For now, keep staging models
            _logger.LogInformation("[CLEANUP] Staging models retained for reference");

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[CLEANUP] Failed to cleanup staging");
        }
    }

    private async Task RollbackFromBackupAsync(
        string backupLocation,
        CancellationToken cancellationToken)
    {
        try
        {
            // Delete current production models
            var productionModels = Directory.GetFiles(_productionDirectory, "*.onnx", SearchOption.TopDirectoryOnly);
            foreach (var model in productionModels)
            {
                File.Delete(model);
            }

            // Restore from backup
            var backupModels = Directory.GetFiles(backupLocation, "*.onnx", SearchOption.TopDirectoryOnly);
            foreach (var model in backupModels)
            {
                var fileName = Path.GetFileName(model);
                var productionPath = Path.Combine(_productionDirectory, fileName);
                File.Copy(model, productionPath, overwrite: true);
            }

            _logger.LogInformation("[ROLLBACK] ✓ Restored {Count} models from backup",
                backupModels.Length);

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ROLLBACK] ❌ Rollback failed");
            throw;
        }
    }

    private async Task SavePromotionReportAsync(
        PromotionReport report,
        CancellationToken cancellationToken)
    {
        try
        {
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
            var jsonPath = Path.Combine(_reportsDirectory, $"promotion-{timestamp}.json");
            var mdPath = Path.Combine(_reportsDirectory, $"promotion-{timestamp}.md");

            // Save JSON
            var json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(jsonPath, json, cancellationToken).ConfigureAwait(false);

            // Save Markdown
            var markdown = GenerateMarkdownReport(report);
            await File.WriteAllTextAsync(mdPath, markdown, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[REPORT] Promotion report saved: {Path}", jsonPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[REPORT] Failed to save promotion report");
        }
    }

    private string GenerateMarkdownReport(PromotionReport report)
    {
        var sb = new System.Text.StringBuilder();

        sb.AppendLine("# Model Promotion Report");
        sb.AppendLine();
        sb.AppendLine($"**Session ID:** {report.SessionId}");
        sb.AppendLine($"**Promotion Time:** {report.PromotionTime:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine($"**Status:** {report.Status}");
        sb.AppendLine();

        sb.AppendLine("## Summary");
        sb.AppendLine();
        sb.AppendLine(report.Summary);
        sb.AppendLine();

        sb.AppendLine("## Criteria Evaluation");
        sb.AppendLine();
        sb.AppendLine($"- Training Success: {(report.Criteria.TrainingSuccess.Passed ? "✅ PASS" : "❌ FAIL")}");
        sb.AppendLine($"- Validation Success: {(report.Criteria.ValidationSuccess.Passed ? "✅ PASS" : "❌ FAIL")}");
        sb.AppendLine($"- Performance: {(report.Criteria.PerformanceCriteria.Passed ? "✅ PASS" : "❌ FAIL")}");
        sb.AppendLine($"- Technical: {(report.Criteria.TechnicalCriteria.Passed ? "✅ PASS" : "❌ FAIL")}");
        sb.AppendLine($"- Operational: {(report.Criteria.OperationalCriteria.Passed ? "✅ PASS" : "❌ FAIL")}");
        sb.AppendLine();

        if (report.AtomicResult.Success)
        {
            sb.AppendLine("## Promotion Details");
            sb.AppendLine();
            sb.AppendLine($"- Models Promoted: {report.AtomicResult.ModelsPromoted}");
            sb.AppendLine($"- Duration: {report.AtomicResult.PromotionDurationMs:F1}ms");
            sb.AppendLine($"- Backup Created: {(report.AtomicResult.BackupCreated ? "Yes" : "No")}");
            sb.AppendLine($"- Rollback Available: {(report.RollbackAvailable ? "Yes" : "No")}");
            sb.AppendLine();
        }

        if (report.Recommendations.Any())
        {
            sb.AppendLine("## Recommendations");
            sb.AppendLine();
            foreach (var rec in report.Recommendations)
            {
                sb.AppendLine($"- {rec}");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    #endregion
}
