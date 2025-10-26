using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Services;

/// <summary>
/// Gate 3: S15 Shadow Learning Promotion Validation
/// Monitors S15 RL strategy performance in shadow mode and promotes to live when validated.
/// Converted to event-driven IHostedService with Timer (Phase 4 refactoring)
/// </summary>
public class S15ShadowLearningService : IHostedService, IDisposable
{
    private static readonly JsonSerializerOptions s_jsonOptions = new() { WriteIndented = true };

    private readonly ILogger<S15ShadowLearningService> _logger;
    private readonly ConcurrentQueue<ShadowDecision> _shadowDecisions = new();
    private Timer? _evaluationTimer;
    private bool _disposed;
    
    // Gate 3 validation thresholds
    private const int MIN_SHADOW_DECISIONS = 1000;
    private const double MIN_SHARPE_RATIO = 2.0;
    private const double MIN_WIN_RATE = 0.50;  // 50%
    private const double MIN_SHARPE_MULTIPLIER = 1.20;  // 20% better than baseline
    private const double MIN_SHARPE_MULTIPLIER_THIN = 1.30;  // 30% for thin markets
    private const double BOOTSTRAP_P_VALUE_THRESHOLD = 0.05;
    private const double CANARY_TRAFFIC_PERCENTAGE = 0.05;  // 5% traffic for canary
    private const int MaxRecentDecisionsToKeep = 100; // Keep last 100 decisions when resetting
    private const int EvaluationIntervalMinutes = 5; // Check every 5 minutes
    
    private int _totalShadowDecisions;
    private bool _isPromotedToCanary;

    public S15ShadowLearningService(ILogger<S15ShadowLearningService> logger)
    {
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔬 S15 Shadow Learning Service started (event-driven mode)");
        
        // Start timer for periodic evaluation
        var interval = TimeSpan.FromMinutes(EvaluationIntervalMinutes);
        _evaluationTimer = new Timer(EvaluationCallback, null, interval, interval);
        
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔬 S15 Shadow Learning Service stopped");
        _evaluationTimer?.Change(Timeout.Infinite, 0);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Timer callback for shadow decision evaluation - uses fire-and-forget pattern
    /// </summary>
    private void EvaluationCallback(object? state)
    {
        if (_disposed) return;

        // Fire-and-forget pattern for async operation in timer callback
        _ = Task.Run(async () =>
        {
            try
            {
                // Check if we have enough shadow decisions to evaluate
                if (_totalShadowDecisions >= MIN_SHADOW_DECISIONS && !_isPromotedToCanary)
                {
                    await EvaluatePromotionAsync(CancellationToken.None).ConfigureAwait(false);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in S15 shadow learning evaluation");
            }
        });
    }

    /// <summary>
    /// Record a shadow decision where S15 observed but did not trade.
    /// </summary>
    public void RecordShadowDecision(ShadowDecision decision)
    {
        ArgumentNullException.ThrowIfNull(decision);
        
        _shadowDecisions.Enqueue(decision);
        Interlocked.Increment(ref _totalShadowDecisions);
        
        _logger.LogDebug("Shadow decision recorded: S15 recommended {Action}, actual was {Actual}", 
            decision.S15Recommendation, decision.ActualDecision);
    }

    private async Task EvaluatePromotionAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("=== GATE 3: S15 SHADOW LEARNING PROMOTION VALIDATION ===");
        
        // Collect all shadow decisions
        var decisions = new List<ShadowDecision>();
        while (_shadowDecisions.TryDequeue(out var decision))
        {
            decisions.Add(decision);
        }
        
        if (decisions.Count < MIN_SHADOW_DECISIONS)
        {
            _logger.LogInformation("Insufficient shadow decisions: {Count} < {Min}", 
                decisions.Count, MIN_SHADOW_DECISIONS);
            
            // Re-queue decisions
            foreach (var d in decisions) _shadowDecisions.Enqueue(d);
            return;
        }
        
        _logger.LogInformation("[1/4] Calculating S15 shadow metrics...");
        var s15Metrics = CalculateMetrics(decisions.Where(d => d.S15Recommendation != "HOLD").ToList());
        
        _logger.LogInformation("[2/4] Calculating baseline metrics...");
        var baselineMetrics = CalculateBaselineMetrics(decisions);
        
        _logger.LogInformation("  S15 Shadow: WR={WinRate:P1}, Sharpe={Sharpe:F2}, Trades={Trades}",
            s15Metrics.WinRate, s15Metrics.Sharpe, s15Metrics.NumTrades);
        _logger.LogInformation("  Baseline:   WR={WinRate:P1}, Sharpe={Sharpe:F2}",
            baselineMetrics.WinRate, baselineMetrics.Sharpe);
        
        // Check 1: Minimum Sharpe ratio
        _logger.LogInformation("[3/4] Validating performance thresholds...");
        if (s15Metrics.Sharpe < MIN_SHARPE_RATIO)
        {
            _logger.LogWarning("✗ S15 Sharpe {Sharpe:F2} below minimum {Min:F2}", 
                s15Metrics.Sharpe, MIN_SHARPE_RATIO);
            ResetAndContinueLearning(decisions);
            return;
        }
        _logger.LogInformation("  ✓ Sharpe threshold met");
        
        // Check 2: Minimum win rate
        if (s15Metrics.WinRate < MIN_WIN_RATE)
        {
            _logger.LogWarning("✗ S15 win rate {WinRate:P1} below minimum {Min:P1}", 
                s15Metrics.WinRate, MIN_WIN_RATE);
            ResetAndContinueLearning(decisions);
            return;
        }
        _logger.LogInformation("  ✓ Win rate threshold met");
        
        // Check 3: Sharpe multiplier vs baseline
        var sharpeMultiplier = s15Metrics.Sharpe / baselineMetrics.Sharpe;
        var requiredMultiplier = decisions.Count < MIN_SHADOW_DECISIONS 
            ? MIN_SHARPE_MULTIPLIER_THIN 
            : MIN_SHARPE_MULTIPLIER;
            
        if (sharpeMultiplier < requiredMultiplier)
        {
            _logger.LogWarning("✗ S15 Sharpe multiplier {Mult:F2}x below required {Req:F2}x", 
                sharpeMultiplier, requiredMultiplier);
            ResetAndContinueLearning(decisions);
            return;
        }
        _logger.LogInformation("  ✓ Sharpe multiplier {Mult:F2}x meets requirement", sharpeMultiplier);
        
        // Check 4: Statistical significance
        _logger.LogInformation("[4/4] Running bootstrap statistical test...");
        var pValue = BootstrapTest(s15Metrics.PnLs, baselineMetrics.PnLs);
        _logger.LogInformation("  Bootstrap p-value: {PValue:F4}", pValue);
        
        if (pValue >= BOOTSTRAP_P_VALUE_THRESHOLD)
        {
            _logger.LogWarning("✗ Not statistically significant: p={PValue:F4} >= {Threshold:F4}", 
                pValue, BOOTSTRAP_P_VALUE_THRESHOLD);
            ResetAndContinueLearning(decisions);
            return;
        }
        _logger.LogInformation("  ✓ Statistically significant");
        
        // All checks passed - promote to canary
        _logger.LogInformation("=== GATE 3 PASSED - Promoting S15 to canary mode ===");
        await PromoteToCanaryAsync(s15Metrics, baselineMetrics, pValue).ConfigureAwait(false);
    }

    private static PerformanceMetrics CalculateMetrics(List<ShadowDecision> decisions)
    {
        if (decisions.Count == 0)
        {
            return new PerformanceMetrics { NumTrades = 0, WinRate = 0, Sharpe = 0, PnLs = new List<double>() };
        }
        
        var wins = decisions.Count(d => d.S15HypotheticalPnL > 0);
        var winRate = (double)wins / decisions.Count;
        
        var pnls = decisions.Select(d => d.S15HypotheticalPnL).ToList();
        var meanPnL = pnls.Average();
        var stdPnL = Math.Sqrt(pnls.Select(p => Math.Pow(p - meanPnL, 2)).Average());
        var sharpe = stdPnL > 0 ? (meanPnL / stdPnL) * Math.Sqrt(252) : 0;
        
        return new PerformanceMetrics
        {
            NumTrades = decisions.Count,
            WinRate = winRate,
            Sharpe = sharpe,
            MeanPnL = meanPnL,
            PnLs = pnls
        };
    }

    private static PerformanceMetrics CalculateBaselineMetrics(List<ShadowDecision> decisions)
    {
        var tradedDecisions = decisions.Where(d => d.ActualDecision != "HOLD").ToList();
        if (tradedDecisions.Count == 0)
        {
            return new PerformanceMetrics { NumTrades = 0, WinRate = 0, Sharpe = 0, PnLs = new List<double>() };
        }
        
        var wins = tradedDecisions.Count(d => d.ActualPnL > 0);
        var winRate = (double)wins / tradedDecisions.Count;
        
        var pnls = tradedDecisions.Select(d => d.ActualPnL).ToList();
        var meanPnL = pnls.Average();
        var stdPnL = Math.Sqrt(pnls.Select(p => Math.Pow(p - meanPnL, 2)).Average());
        var sharpe = stdPnL > 0 ? (meanPnL / stdPnL) * Math.Sqrt(252) : 0;
        
        return new PerformanceMetrics
        {
            NumTrades = tradedDecisions.Count,
            WinRate = winRate,
            Sharpe = sharpe,
            MeanPnL = meanPnL,
            PnLs = pnls
        };
    }

    private static double BootstrapTest(IReadOnlyList<double> sample1, IReadOnlyList<double> sample2, int iterations = 10000)
    {
        if (sample1.Count == 0 || sample2.Count == 0) return 1.0;
        
        _ = sample1.Average() - sample2.Average();
        var combined = sample1.Concat(sample2).ToArray();
        var random = new Random(42);
        
        int countGreater = 0;
        for (int i = 0; i < iterations; i++)
        {
            var shuffled = combined.OrderBy(_ => random.Next()).ToArray();
            var group1 = shuffled.Take(sample1.Count).Average();
            var group2 = shuffled.Skip(sample1.Count).Average();
            
            if (group1 - group2 <= 0) countGreater++;
        }
        
        return (double)countGreater / iterations;
    }

    private void ResetAndContinueLearning(List<ShadowDecision> decisions)
    {
        _logger.LogInformation("Resetting counter and continuing shadow learning...");
        _totalShadowDecisions = 0;
        
        // Keep recent decisions for next evaluation
        foreach (var d in decisions.TakeLast(MaxRecentDecisionsToKeep))
        {
            _shadowDecisions.Enqueue(d);
        }
    }

    private async Task PromoteToCanaryAsync(PerformanceMetrics s15, PerformanceMetrics baseline, double pValue)
    {
        _isPromotedToCanary = true;
        
        // Write promotion configuration
        var promotionConfig = new
        {
            strategy = "S15_RL",
            status = "canary",
            traffic_percentage = CANARY_TRAFFIC_PERCENTAGE,
            promoted_at = DateTime.UtcNow,
            validation_metrics = new
            {
                s15_sharpe = s15.Sharpe,
                s15_win_rate = s15.WinRate,
                baseline_sharpe = baseline.Sharpe,
                sharpe_multiplier = s15.Sharpe / baseline.Sharpe,
                p_value = pValue
            }
        };
        
        var configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "config", "s15_promotion.json");
        Directory.CreateDirectory(Path.GetDirectoryName(configPath) ?? ".");
        
        await File.WriteAllTextAsync(configPath, JsonSerializer.Serialize(promotionConfig, s_jsonOptions)).ConfigureAwait(false);
        
        _logger.LogInformation("✓ S15 promoted to canary mode with {Pct:P1} traffic", CANARY_TRAFFIC_PERCENTAGE);
        _logger.LogInformation("  Config written to: {Path}", configPath);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _evaluationTimer?.Dispose();
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}

public class ShadowDecision
{
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = "";
    public string S15Recommendation { get; set; } = "";
    public string ActualDecision { get; set; } = "";
    public double S15HypotheticalPnL { get; set; }
    public double ActualPnL { get; set; }
    public bool S7Allowed { get; set; }
}

public class PerformanceMetrics
{
    public int NumTrades { get; set; }
    public double WinRate { get; set; }
    public double Sharpe { get; set; }
    public double MeanPnL { get; set; }
    public IReadOnlyList<double> PnLs { get; init; } = new List<double>();
}
