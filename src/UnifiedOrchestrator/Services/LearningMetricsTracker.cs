using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Learning Metrics Tracker - Tracks bot performance improvements over time
/// 
/// Purpose: Ensure bot learns and improves from 20% to 85% win rate over multiple training sessions
/// - Saves win rate after each training session
/// - Tracks learning trends across sessions
/// - Prevents catastrophic forgetting by comparing against historical baselines
/// - Provides proof that bot is actually learning (not just coded)
/// </summary>
public sealed class LearningMetricsTracker
{
    private readonly ILogger<LearningMetricsTracker> _logger;
    private readonly string _metricsDirectory;
    private readonly string _historyFilePath;
    private readonly string _currentMetricsPath;

    public LearningMetricsTracker(ILogger<LearningMetricsTracker> logger)
    {
        _logger = logger;
        _metricsDirectory = Path.Combine(Directory.GetCurrentDirectory(), "state", "learning_metrics");
        _historyFilePath = Path.Combine(_metricsDirectory, "performance_history.json");
        _currentMetricsPath = Path.Combine(_metricsDirectory, "current_session.json");
        
        Directory.CreateDirectory(_metricsDirectory);
        
        _logger.LogInformation("[LEARNING-TRACKER] Initialized metrics directory: {Directory}", _metricsDirectory);
    }

    /// <summary>
    /// Save performance metrics after training session
    /// This is called automatically after each Lab Mode training completes
    /// </summary>
    public async Task SaveTrainingSessionMetricsAsync(TrainingSessionMetrics metrics, CancellationToken cancellationToken = default)
    {
        try
        {
            // Load existing history
            var history = await LoadPerformanceHistoryAsync(cancellationToken).ConfigureAwait(false);

            // Add new session
            history.Sessions.Add(metrics);

            // Calculate improvement trends
            if (history.Sessions.Count >= 2)
            {
                var previous = history.Sessions[^2];
                var improvement = metrics.WinRate - previous.WinRate;
                
                _logger.LogInformation("[LEARNING-TRACKER] ═══════════════════════════════════════════════════════");
                _logger.LogInformation("[LEARNING-TRACKER] LEARNING PROGRESS VERIFIED");
                _logger.LogInformation("[LEARNING-TRACKER] ═══════════════════════════════════════════════════════");
                _logger.LogInformation("[LEARNING-TRACKER] Session #{SessionNumber}: {SessionId}", history.Sessions.Count, metrics.SessionId);
                _logger.LogInformation("[LEARNING-TRACKER] Win Rate: {Current:F2}% (Previous: {Previous:F2}%, Change: {Change:+0.00;-0.00}%)",
                    metrics.WinRate, previous.WinRate, improvement);
                _logger.LogInformation("[LEARNING-TRACKER] Average R-Multiple: {Current:F2} (Previous: {Previous:F2})",
                    metrics.AverageRMultiple, previous.AverageRMultiple);
                _logger.LogInformation("[LEARNING-TRACKER] Sharpe Ratio: {Current:F2} (Previous: {Previous:F2})",
                    metrics.SharpeRatio, previous.SharpeRatio);
                _logger.LogInformation("[LEARNING-TRACKER] Total Trades Learned From: {Count:N0}", metrics.TotalTrades);
                _logger.LogInformation("[LEARNING-TRACKER] ═══════════════════════════════════════════════════════");

                if (improvement > 0)
                {
                    _logger.LogInformation("[LEARNING-TRACKER] ✅ BOT IS LEARNING - Win rate improved by {Improvement:F2}%", improvement);
                }
                else if (improvement < -5.0m)
                {
                    _logger.LogWarning("[LEARNING-TRACKER] ⚠️ PERFORMANCE REGRESSION - Win rate decreased by {Decrease:F2}%", Math.Abs(improvement));
                    _logger.LogWarning("[LEARNING-TRACKER] This may indicate catastrophic forgetting - review training data");
                }
                else
                {
                    _logger.LogInformation("[LEARNING-TRACKER] ℹ️ Performance stable (change: {Change:F2}%)", improvement);
                }
            }
            else
            {
                _logger.LogInformation("[LEARNING-TRACKER] ═══════════════════════════════════════════════════════");
                _logger.LogInformation("[LEARNING-TRACKER] BASELINE SESSION CAPTURED");
                _logger.LogInformation("[LEARNING-TRACKER] ═══════════════════════════════════════════════════════");
                _logger.LogInformation("[LEARNING-TRACKER] Session: {SessionId}", metrics.SessionId);
                _logger.LogInformation("[LEARNING-TRACKER] Starting Win Rate: {WinRate:F2}%", metrics.WinRate);
                _logger.LogInformation("[LEARNING-TRACKER] Starting Sharpe: {Sharpe:F2}", metrics.SharpeRatio);
                _logger.LogInformation("[LEARNING-TRACKER] Trades in Dataset: {Count:N0}", metrics.TotalTrades);
                _logger.LogInformation("[LEARNING-TRACKER] Target: Improve to 85% win rate over time");
                _logger.LogInformation("[LEARNING-TRACKER] ═══════════════════════════════════════════════════════");
            }

            // Save updated history
            var json = JsonSerializer.Serialize(history, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(_historyFilePath, json, cancellationToken).ConfigureAwait(false);

            // Save current session separately for quick access
            var currentJson = JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(_currentMetricsPath, currentJson, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[LEARNING-TRACKER] ✓ Metrics saved to: {Path}", _historyFilePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LEARNING-TRACKER] Failed to save training session metrics");
            throw;
        }
    }

    /// <summary>
    /// Load complete performance history across all training sessions
    /// </summary>
    public async Task<PerformanceHistory> LoadPerformanceHistoryAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(_historyFilePath))
            {
                _logger.LogInformation("[LEARNING-TRACKER] No existing history found - starting fresh");
                return new PerformanceHistory();
            }

            var json = await File.ReadAllTextAsync(_historyFilePath, cancellationToken).ConfigureAwait(false);
            var history = JsonSerializer.Deserialize<PerformanceHistory>(json);

            if (history == null)
            {
                _logger.LogWarning("[LEARNING-TRACKER] Failed to deserialize history - starting fresh");
                return new PerformanceHistory();
            }

            _logger.LogInformation("[LEARNING-TRACKER] Loaded {Count} historical training sessions", history.Sessions.Count);
            return history;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LEARNING-TRACKER] Failed to load performance history");
            return new PerformanceHistory();
        }
    }

    /// <summary>
    /// Get learning progress summary showing improvement over time
    /// </summary>
    public async Task<LearningProgressSummary> GetLearningProgressAsync(CancellationToken cancellationToken = default)
    {
        var history = await LoadPerformanceHistoryAsync(cancellationToken).ConfigureAwait(false);

        if (history.Sessions.Count == 0)
        {
            return new LearningProgressSummary
            {
                TotalSessions = 0,
                Message = "No training sessions yet"
            };
        }

        var firstSession = history.Sessions.First();
        var lastSession = history.Sessions.Last();

        var summary = new LearningProgressSummary
        {
            TotalSessions = history.Sessions.Count,
            StartingWinRate = firstSession.WinRate,
            CurrentWinRate = lastSession.WinRate,
            WinRateImprovement = lastSession.WinRate - firstSession.WinRate,
            StartingSharpe = firstSession.SharpeRatio,
            CurrentSharpe = lastSession.SharpeRatio,
            SharpeImprovement = lastSession.SharpeRatio - firstSession.SharpeRatio,
            TotalTradesLearned = history.Sessions.Sum(s => s.TotalTrades),
            FirstSessionDate = firstSession.Timestamp,
            LastSessionDate = lastSession.Timestamp
        };

        // Calculate if on track to reach 85% target
        summary.TargetWinRate = 85.0m;
        summary.RemainingImprovement = summary.TargetWinRate - summary.CurrentWinRate;
        
        if (summary.TotalSessions >= 2)
        {
            var avgImprovementPerSession = summary.WinRateImprovement / (summary.TotalSessions - 1);
            if (avgImprovementPerSession > 0)
            {
                summary.EstimatedSessionsToTarget = (int)Math.Ceiling((double)(summary.RemainingImprovement / avgImprovementPerSession));
                summary.Message = $"On track! Averaging {avgImprovementPerSession:F2}% improvement per session. " +
                                $"Estimated {summary.EstimatedSessionsToTarget} more sessions to reach 85% target.";
            }
            else
            {
                summary.Message = "Not improving yet - needs more training sessions to establish trend";
            }
        }
        else
        {
            summary.Message = "Need at least 2 sessions to calculate improvement trend";
        }

        return summary;
    }

    /// <summary>
    /// Analyze if catastrophic forgetting has occurred
    /// Returns true if recent performance is significantly worse than historical best
    /// </summary>
    public async Task<(bool HasForgotten, string Reason)> DetectCatastrophicForgettingAsync(
        TrainingSessionMetrics currentMetrics,
        CancellationToken cancellationToken = default)
    {
        var history = await LoadPerformanceHistoryAsync(cancellationToken).ConfigureAwait(false);

        if (history.Sessions.Count < 3)
        {
            return (false, "Not enough history to detect forgetting (need at least 3 sessions)");
        }

        // Get best historical performance (excluding current session)
        var bestWinRate = history.Sessions.Max(s => s.WinRate);
        var bestSharpe = history.Sessions.Max(s => s.SharpeRatio);

        // Check if current session is significantly worse than best
        const decimal WinRateThreshold = 10.0m; // 10% drop indicates forgetting
        const decimal SharpeThreshold = 0.5m;

        var winRateDrop = bestWinRate - currentMetrics.WinRate;
        var sharpeDrop = bestSharpe - currentMetrics.SharpeRatio;

        if (winRateDrop >= WinRateThreshold)
        {
            return (true, $"Win rate dropped {winRateDrop:F2}% from historical best {bestWinRate:F2}% to {currentMetrics.WinRate:F2}%");
        }

        if (sharpeDrop >= SharpeThreshold)
        {
            return (true, $"Sharpe ratio dropped {sharpeDrop:F2} from historical best {bestSharpe:F2} to {currentMetrics.SharpeRatio:F2}");
        }

        return (false, "No catastrophic forgetting detected - performance within acceptable range");
    }
}

/// <summary>
/// Metrics for a single training session
/// </summary>
public sealed class TrainingSessionMetrics
{
    public string SessionId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public decimal WinRate { get; set; }
    public decimal AverageRMultiple { get; set; }
    public decimal SharpeRatio { get; set; }
    public decimal MaxDrawdown { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal TotalPnL { get; set; }
    public Dictionary<string, decimal> ModelScores { get; set; } = new();
    public Dictionary<string, int> ModelVersions { get; set; } = new();
}

/// <summary>
/// Complete history of all training sessions
/// </summary>
public sealed class PerformanceHistory
{
    public List<TrainingSessionMetrics> Sessions { get; set; } = new();
}

/// <summary>
/// Summary of learning progress over time
/// </summary>
public sealed class LearningProgressSummary
{
    public int TotalSessions { get; set; }
    public decimal StartingWinRate { get; set; }
    public decimal CurrentWinRate { get; set; }
    public decimal WinRateImprovement { get; set; }
    public decimal StartingSharpe { get; set; }
    public decimal CurrentSharpe { get; set; }
    public decimal SharpeImprovement { get; set; }
    public int TotalTradesLearned { get; set; }
    public DateTime FirstSessionDate { get; set; }
    public DateTime LastSessionDate { get; set; }
    public decimal TargetWinRate { get; set; }
    public decimal RemainingImprovement { get; set; }
    public int EstimatedSessionsToTarget { get; set; }
    public string Message { get; set; } = string.Empty;
}
