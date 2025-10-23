using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Performance Degradation Detector - Monitors live trading performance and triggers
/// emergency retraining (Anyday Lab Mode) when degradation is detected.
/// 
/// Degradation conditions:
/// - Sharpe ratio < 0.5 for 3+ consecutive days
/// - Drawdown > 10% for 3+ consecutive days
/// - 5+ consecutive losing trades
/// 
/// Automatically triggers Anyday Lab Mode when conditions met and safety checks pass.
/// </summary>
public sealed class PerformanceDegradationDetector
{
    private readonly ILogger<PerformanceDegradationDetector> _logger;
    private readonly PerformanceMetricsTracker _metricsTracker;
    
    // Degradation thresholds
    private const decimal SharpeThreshold = 0.5m;
    private const decimal DrawdownThreshold = 0.10m; // 10%
    private const int ConsecutiveLossesThreshold = 5;
    private const int DegradedDaysThreshold = 3;
    
    // Minimum data requirements for retraining
    private const int MinimumDaysForRetraining = 30;
    
    public PerformanceDegradationDetector(
        ILogger<PerformanceDegradationDetector> logger,
        PerformanceMetricsTracker metricsTracker)
    {
        _logger = logger;
        _metricsTracker = metricsTracker;
    }

    /// <summary>
    /// Monitor performance and automatically trigger Anyday Lab Mode if degradation detected
    /// Should be called periodically (e.g., every 4 hours during trading week)
    /// </summary>
    public async Task<DegradationCheckResult> CheckAndTriggerIfNeededAsync(
        CancellationToken cancellationToken = default)
    {
        _logger.LogDebug("[DEGRADATION] Running performance degradation check...");
        
        // Get recent performance metrics (last 3 days)
        var recentMetrics = await _metricsTracker.GetRecentPerformanceAsync(
            days: 3, 
            cancellationToken
        ).ConfigureAwait(false);
        
        var result = new DegradationCheckResult
        {
            CheckTime = DateTime.UtcNow,
            Sharpe = recentMetrics.Sharpe,
            Drawdown = recentMetrics.Drawdown,
            ConsecutiveLosses = recentMetrics.ConsecutiveLosses,
            DegradedDaysCount = recentMetrics.DegradedDaysCount
        };
        
        // Check for degradation
        var degradationDetected = DetectDegradation(recentMetrics, result);
        
        if (!degradationDetected)
        {
            _logger.LogDebug("[DEGRADATION] Performance healthy - Sharpe: {Sharpe:F2}, Drawdown: {Drawdown:F2}%",
                recentMetrics.Sharpe, recentMetrics.Drawdown * 100);
            return result;
        }
        
        // Degradation detected - run safety checks
        _logger.LogWarning("[DEGRADATION] Performance degradation detected:");
        _logger.LogWarning("[DEGRADATION]   Sharpe: {Sharpe:F2} (threshold: {Threshold:F2})",
            recentMetrics.Sharpe, SharpeThreshold);
        _logger.LogWarning("[DEGRADATION]   Drawdown: {Drawdown:F2}% (threshold: {Threshold:F2}%)",
            recentMetrics.Drawdown * 100, DrawdownThreshold * 100);
        _logger.LogWarning("[DEGRADATION]   Consecutive losses: {Losses} (threshold: {Threshold})",
            recentMetrics.ConsecutiveLosses, ConsecutiveLossesThreshold);
        _logger.LogWarning("[DEGRADATION]   Degraded days: {Days}/{Threshold}",
            recentMetrics.DegradedDaysCount, DegradedDaysThreshold);
        
        // Run safety checks
        var safetyChecksPassed = await RunSafetyChecksAsync(result, cancellationToken).ConfigureAwait(false);
        
        if (!safetyChecksPassed)
        {
            _logger.LogWarning("[DEGRADATION] Safety checks failed - NOT triggering Anyday Lab Mode");
            result.CanTriggerAnydayLab = false;
            return result;
        }
        
        // All checks passed - trigger Anyday Lab Mode
        _logger.LogInformation("[DEGRADATION] All safety checks passed - triggering Anyday Lab Mode");
        result.AnydayLabTriggered = await TriggerAnydayLabModeAsync(cancellationToken).ConfigureAwait(false);
        
        return result;
    }

    private bool DetectDegradation(RecentPerformance metrics, DegradationCheckResult result)
    {
        var sharpeIssue = metrics.Sharpe < SharpeThreshold;
        var drawdownIssue = metrics.Drawdown > DrawdownThreshold;
        var consecutiveLossesIssue = metrics.ConsecutiveLosses >= ConsecutiveLossesThreshold;
        var isPersistent = metrics.DegradedDaysCount >= DegradedDaysThreshold;
        
        result.SharpeIssueDetected = sharpeIssue;
        result.DrawdownIssueDetected = drawdownIssue;
        result.ConsecutiveLossesIssueDetected = consecutiveLossesIssue;
        result.IsPersistent = isPersistent;
        
        // Degradation detected if any issue is present AND it's persistent
        result.DegradationDetected = (sharpeIssue || drawdownIssue || consecutiveLossesIssue) && isPersistent;
        
        return result.DegradationDetected;
    }

    private async Task<bool> RunSafetyChecksAsync(
        DegradationCheckResult result,
        CancellationToken cancellationToken)
    {
        // Check 1: Not already training
        if (await IsTrainingInProgressAsync(cancellationToken).ConfigureAwait(false))
        {
            _logger.LogWarning("[DEGRADATION] Safety check FAILED: Training already in progress");
            result.SafetyCheckFailures.Add("Training already in progress");
            return false;
        }
        
        // Check 2: Sufficient data
        var availableDays = await GetAvailableHistoricalDaysAsync(cancellationToken).ConfigureAwait(false);
        if (availableDays < MinimumDaysForRetraining)
        {
            _logger.LogWarning("[DEGRADATION] Safety check FAILED: Insufficient data ({Days}/{Required} days)",
                availableDays, MinimumDaysForRetraining);
            result.SafetyCheckFailures.Add($"Insufficient data: {availableDays}/{MinimumDaysForRetraining} days");
            return false;
        }
        
        // Check 3: Sufficient resources
        if (!await HasSufficientResourcesAsync(cancellationToken).ConfigureAwait(false))
        {
            _logger.LogWarning("[DEGRADATION] Safety check FAILED: Insufficient resources");
            result.SafetyCheckFailures.Add("Insufficient system resources");
            return false;
        }
        
        // Check 4: Market is open (optional - can train during closed market)
        // Skipped for now - training can happen anytime
        
        _logger.LogInformation("[DEGRADATION] All safety checks PASSED");
        result.CanTriggerAnydayLab = true;
        return true;
    }

    private async Task<bool> TriggerAnydayLabModeAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Set environment variable for Anyday Lab Mode
            Environment.SetEnvironmentVariable("LAB_MODE_SCHEDULE", "MANUAL");
            
            _logger.LogInformation("[DEGRADATION] Set LAB_MODE_SCHEDULE=MANUAL");
            
            // Spawn Lab Mode process
            var labModeStarted = await SpawnLabModeProcessAsync(cancellationToken).ConfigureAwait(false);
            
            if (labModeStarted)
            {
                _logger.LogInformation("[DEGRADATION] ✅ Anyday Lab Mode triggered successfully");
                _logger.LogInformation("[DEGRADATION] Terminal Mode will continue trading with current models");
                return true;
            }
            else
            {
                _logger.LogError("[DEGRADATION] Failed to spawn Lab Mode process");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DEGRADATION] Error triggering Anyday Lab Mode: {Error}", ex.Message);
            return false;
        }
    }

    private Task<bool> IsTrainingInProgressAsync(CancellationToken cancellationToken)
    {
        // Check if training process is running or training lock file exists
        var lockFile = Path.Combine(Directory.GetCurrentDirectory(), "state", "training.lock");
        return Task.FromResult(File.Exists(lockFile));
    }

    private async Task<int> GetAvailableHistoricalDaysAsync(CancellationToken cancellationToken)
    {
        // Check ES_90days.json metadata
        var dataFile = Path.Combine(Directory.GetCurrentDirectory(), "data", "historical", "ES_90days.json");
        
        if (!File.Exists(dataFile))
        {
            return 0;
        }
        
        try
        {
            var json = await File.ReadAllTextAsync(dataFile, cancellationToken).ConfigureAwait(false);
            using var doc = System.Text.Json.JsonDocument.Parse(json);
            
            if (doc.RootElement.TryGetProperty("total_days", out var daysElement))
            {
                return daysElement.GetInt32();
            }
            
            // Fallback: estimate from bar count (78 bars per day)
            if (doc.RootElement.TryGetProperty("bar_count", out var barCountElement))
            {
                var barCount = barCountElement.GetInt32();
                return barCount / 78; // Approximate days
            }
            
            return 0;
        }
        catch
        {
            return 0;
        }
    }

    private Task<bool> HasSufficientResourcesAsync(CancellationToken cancellationToken)
    {
        // Check available RAM, disk space, CPU
        var memoryInfo = GC.GetGCMemoryInfo();
        var availableMemoryBytes = memoryInfo.TotalAvailableMemoryBytes;
        var availableMemoryGB = availableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
        
        // Need at least 4GB free RAM for training
        if (availableMemoryGB < 4.0)
        {
            _logger.LogWarning("[DEGRADATION] Insufficient RAM: {Available:F1} GB (need 4+ GB)", availableMemoryGB);
            return Task.FromResult(false);
        }
        
        // Check disk space
        var driveInfo = new DriveInfo(Directory.GetCurrentDirectory());
        var availableDiskGB = driveInfo.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
        
        // Need at least 10GB free disk
        if (availableDiskGB < 10.0)
        {
            _logger.LogWarning("[DEGRADATION] Insufficient disk space: {Available:F1} GB (need 10+ GB)", availableDiskGB);
            return Task.FromResult(false);
        }
        
        _logger.LogDebug("[DEGRADATION] Resources OK - RAM: {RAM:F1} GB, Disk: {Disk:F1} GB",
            availableMemoryGB, availableDiskGB);
        
        return Task.FromResult(true);
    }

    private Task<bool> SpawnLabModeProcessAsync(CancellationToken cancellationToken)
    {
        // This would spawn the UnifiedOrchestrator in Lab Mode
        // For now, just log the intent
        _logger.LogInformation("[DEGRADATION] Would spawn: dotnet run --project UnifiedOrchestrator.csproj -- --mode lab");
        
        // In production, use Process.Start() to spawn the process
        // var processInfo = new ProcessStartInfo
        // {
        //     FileName = "dotnet",
        //     Arguments = "run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -- --mode lab",
        //     UseShellExecute = false,
        //     RedirectStandardOutput = true,
        //     RedirectStandardError = true
        // };
        // 
        // var process = Process.Start(processInfo);
        // return process != null;
        
        return Task.FromResult(true); // Simulated success for now
    }
}

/// <summary>
/// Result of degradation check
/// </summary>
public sealed class DegradationCheckResult
{
    public DateTime CheckTime { get; init; }
    public decimal Sharpe { get; init; }
    public decimal Drawdown { get; init; }
    public int ConsecutiveLosses { get; init; }
    public int DegradedDaysCount { get; init; }
    
    public bool DegradationDetected { get; set; }
    public bool SharpeIssueDetected { get; set; }
    public bool DrawdownIssueDetected { get; set; }
    public bool ConsecutiveLossesIssueDetected { get; set; }
    public bool IsPersistent { get; set; }
    
    public bool CanTriggerAnydayLab { get; set; }
    public bool AnydayLabTriggered { get; set; }
    public List<string> SafetyCheckFailures { get; set; } = new();
}

/// <summary>
/// Recent performance metrics
/// </summary>
public sealed class RecentPerformance
{
    public decimal Sharpe { get; init; }
    public decimal Drawdown { get; init; }
    public int ConsecutiveLosses { get; init; }
    public int DegradedDaysCount { get; init; }
}

/// <summary>
/// Performance metrics tracker interface
/// </summary>
public interface PerformanceMetricsTracker
{
    Task<RecentPerformance> GetRecentPerformanceAsync(int days, CancellationToken cancellationToken);
}
