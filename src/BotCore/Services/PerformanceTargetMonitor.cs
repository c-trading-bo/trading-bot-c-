using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Services;

/// <summary>
/// Performance Target Monitor
/// Tracks and enforces Terminal Mode performance targets from Owner's Manual:
/// - Decision latency: Sub-22 milliseconds
/// - Uptime: 99.9% during market hours
/// - Fill quality: Average slippage within 0.5 ticks
/// </summary>
public class PerformanceTargetMonitor : BackgroundService
{
    private readonly ILogger<PerformanceTargetMonitor> _logger;
    
    // Performance targets from Owner's Manual
    private const double MaxDecisionLatencyMs = 22.0;
    private const double MinUptimePercentage = 99.9;
    private const double MaxAverageSlippageTicks = 0.5;
    
    // Tracking state
    private readonly object _statsLock = new();
    private long _totalDecisions;
    private long _decisionsOverLatencyTarget;
    private double _totalDecisionLatencyMs;
    private DateTime _startTime;
    private TimeSpan _totalDowntime;
    private DateTime? _lastDowntimeStart;
    
    // Slippage tracking
    private long _totalFills;
    private double _totalSlippageTicks;
    
    // Alert thresholds
    private const int AlertAfterDecisions = 100; // Alert after 100 decisions if issues
    private const double LatencyViolationThreshold = 0.05; // 5% of decisions over target
    
    public PerformanceTargetMonitor(ILogger<PerformanceTargetMonitor> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _startTime = DateTime.UtcNow;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Only run in Terminal Mode
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE");
        
        if (labMode == "1" || historicalMode == "1")
        {
            _logger.LogInformation("[PERF-MONITOR] Skipping performance monitoring (not in Terminal Mode)");
            // FIXED: Wait indefinitely instead of returning to prevent shutdown signal
            await Task.Delay(Timeout.Infinite, stoppingToken).ConfigureAwait(false);
            return;
        }

        _logger.LogInformation("═══════════════════════════════════════════════════════════════");
        _logger.LogInformation("⚡ PERFORMANCE TARGET MONITOR - Owner's Manual Enforcement");
        _logger.LogInformation("═══════════════════════════════════════════════════════════════");
        _logger.LogInformation("  📊 Decision Latency Target: <{Target}ms", MaxDecisionLatencyMs);
        _logger.LogInformation("  ⏰ Uptime Target: {Target}% during market hours", MinUptimePercentage);
        _logger.LogInformation("  📈 Slippage Target: ≤{Target} ticks average", MaxAverageSlippageTicks);
        _logger.LogInformation("═══════════════════════════════════════════════════════════════");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);
                
                // Report metrics every 5 minutes
                ReportPerformanceMetrics();
            }
            catch (OperationCanceledException)
            {
                // Normal shutdown
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[PERF-MONITOR] Error in performance monitoring loop");
            }
        }
    }

    /// <summary>
    /// Record decision latency (call this after each trading decision)
    /// </summary>
    public void RecordDecisionLatency(double latencyMs)
    {
        lock (_statsLock)
        {
            _totalDecisions++;
            _totalDecisionLatencyMs += latencyMs;
            
            if (latencyMs > MaxDecisionLatencyMs)
            {
                _decisionsOverLatencyTarget++;
                
                // Log individual violations
                _logger.LogWarning(
                    "[PERF-MONITOR] ⚠️  Decision latency EXCEEDED target: {Latency:F2}ms (target: <{Target}ms)",
                    latencyMs, MaxDecisionLatencyMs);
            }
            
            // Alert if threshold exceeded
            if (_totalDecisions >= AlertAfterDecisions)
            {
                var violationRate = (double)_decisionsOverLatencyTarget / _totalDecisions;
                if (violationRate > LatencyViolationThreshold)
                {
                    _logger.LogWarning(
                        "[PERF-MONITOR] 🚨 LATENCY TARGET VIOLATION: {Rate:F2}% of decisions exceed {Target}ms (threshold: {Threshold:F2}%)",
                        violationRate * 100, MaxDecisionLatencyMs, LatencyViolationThreshold * 100);
                }
            }
        }
    }

    /// <summary>
    /// Record fill slippage (call this after each order fill)
    /// </summary>
    public void RecordSlippage(double slippageTicks)
    {
        lock (_statsLock)
        {
            _totalFills++;
            _totalSlippageTicks += Math.Abs(slippageTicks);
            
            var avgSlippage = _totalSlippageTicks / _totalFills;
            
            if (avgSlippage > MaxAverageSlippageTicks)
            {
                _logger.LogWarning(
                    "[PERF-MONITOR] ⚠️  Average slippage EXCEEDED target: {Avg:F3} ticks (target: ≤{Target} ticks)",
                    avgSlippage, MaxAverageSlippageTicks);
            }
        }
    }

    /// <summary>
    /// Record downtime start (call when system goes down)
    /// </summary>
    public void RecordDowntimeStart()
    {
        lock (_statsLock)
        {
            if (_lastDowntimeStart == null)
            {
                _lastDowntimeStart = DateTime.UtcNow;
                _logger.LogWarning("[PERF-MONITOR] ⚠️  System downtime started at {Time}", _lastDowntimeStart);
            }
        }
    }

    /// <summary>
    /// Record downtime end (call when system comes back up)
    /// </summary>
    public void RecordDowntimeEnd()
    {
        lock (_statsLock)
        {
            if (_lastDowntimeStart != null)
            {
                var downtime = DateTime.UtcNow - _lastDowntimeStart.Value;
                _totalDowntime += downtime;
                
                _logger.LogWarning(
                    "[PERF-MONITOR] ⚠️  System downtime ended. Duration: {Duration:F2} minutes",
                    downtime.TotalMinutes);
                
                _lastDowntimeStart = null;
                
                // Check uptime
                CheckUptimeTarget();
            }
        }
    }

    private void CheckUptimeTarget()
    {
        var elapsed = DateTime.UtcNow - _startTime;
        var uptimePercentage = 100.0 * (1.0 - (_totalDowntime.TotalSeconds / elapsed.TotalSeconds));
        
        if (uptimePercentage < MinUptimePercentage)
        {
            _logger.LogWarning(
                "[PERF-MONITOR] 🚨 UPTIME TARGET VIOLATION: {Uptime:F3}% (target: ≥{Target}%)",
                uptimePercentage, MinUptimePercentage);
        }
        else
        {
            _logger.LogInformation(
                "[PERF-MONITOR] ✅ Uptime: {Uptime:F3}% (target: ≥{Target}%)",
                uptimePercentage, MinUptimePercentage);
        }
    }

    private void ReportPerformanceMetrics()
    {
        lock (_statsLock)
        {
            if (_totalDecisions == 0)
            {
                _logger.LogDebug("[PERF-MONITOR] No decisions recorded yet");
                return;
            }

            var avgLatency = _totalDecisionLatencyMs / _totalDecisions;
            var latencyViolationRate = 100.0 * _decisionsOverLatencyTarget / _totalDecisions;
            
            var elapsed = DateTime.UtcNow - _startTime;
            var uptimePercentage = 100.0 * (1.0 - (_totalDowntime.TotalSeconds / elapsed.TotalSeconds));
            
            _logger.LogInformation("");
            _logger.LogInformation("═══════════════════════════════════════════════════════════════");
            _logger.LogInformation("⚡ PERFORMANCE METRICS REPORT");
            _logger.LogInformation("═══════════════════════════════════════════════════════════════");
            _logger.LogInformation("📊 Decision Latency:");
            _logger.LogInformation("   • Average: {Avg:F2}ms (target: <{Target}ms) {Status}",
                avgLatency, MaxDecisionLatencyMs,
                avgLatency < MaxDecisionLatencyMs ? "✅" : "❌");
            _logger.LogInformation("   • Violations: {Count}/{Total} ({Rate:F2}%)",
                _decisionsOverLatencyTarget, _totalDecisions, latencyViolationRate);
            
            _logger.LogInformation("⏰ Uptime:");
            _logger.LogInformation("   • Current: {Uptime:F3}% (target: ≥{Target}%) {Status}",
                uptimePercentage, MinUptimePercentage,
                uptimePercentage >= MinUptimePercentage ? "✅" : "❌");
            _logger.LogInformation("   • Total downtime: {Downtime:F2} minutes",
                _totalDowntime.TotalMinutes);
            
            if (_totalFills > 0)
            {
                var avgSlippage = _totalSlippageTicks / _totalFills;
                _logger.LogInformation("📈 Fill Quality:");
                _logger.LogInformation("   • Average slippage: {Avg:F3} ticks (target: ≤{Target} ticks) {Status}",
                    avgSlippage, MaxAverageSlippageTicks,
                    avgSlippage <= MaxAverageSlippageTicks ? "✅" : "❌");
                _logger.LogInformation("   • Total fills: {Count}", _totalFills);
            }
            else
            {
                _logger.LogInformation("📈 Fill Quality: No fills recorded yet");
            }
            
            _logger.LogInformation("═══════════════════════════════════════════════════════════════");
            _logger.LogInformation("");
        }
    }

    /// <summary>
    /// Get current performance metrics (for external monitoring/dashboards)
    /// </summary>
    public TerminalPerformanceMetrics GetCurrentMetrics()
    {
        lock (_statsLock)
        {
            var elapsed = DateTime.UtcNow - _startTime;
            var uptimePercentage = 100.0 * (1.0 - (_totalDowntime.TotalSeconds / elapsed.TotalSeconds));
            
            return new TerminalPerformanceMetrics
            {
                AverageDecisionLatencyMs = _totalDecisions > 0 ? _totalDecisionLatencyMs / _totalDecisions : 0,
                LatencyViolationRate = _totalDecisions > 0 ? (double)_decisionsOverLatencyTarget / _totalDecisions : 0,
                UptimePercentage = uptimePercentage,
                AverageSlippageTicks = _totalFills > 0 ? _totalSlippageTicks / _totalFills : 0,
                TotalDecisions = _totalDecisions,
                TotalFills = _totalFills,
                TotalDowntime = _totalDowntime,
                MeetsLatencyTarget = _totalDecisions == 0 || (_totalDecisionLatencyMs / _totalDecisions) < MaxDecisionLatencyMs,
                MeetsUptimeTarget = uptimePercentage >= MinUptimePercentage,
                MeetsSlippageTarget = _totalFills == 0 || (_totalSlippageTicks / _totalFills) <= MaxAverageSlippageTicks
            };
        }
    }
}

/// <summary>
/// Terminal Mode performance metrics snapshot
/// </summary>
public class TerminalPerformanceMetrics
{
    public double AverageDecisionLatencyMs { get; set; }
    public double LatencyViolationRate { get; set; }
    public double UptimePercentage { get; set; }
    public double AverageSlippageTicks { get; set; }
    public long TotalDecisions { get; set; }
    public long TotalFills { get; set; }
    public TimeSpan TotalDowntime { get; set; }
    public bool MeetsLatencyTarget { get; set; }
    public bool MeetsUptimeTarget { get; set; }
    public bool MeetsSlippageTarget { get; set; }
}
