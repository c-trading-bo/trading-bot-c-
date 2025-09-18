using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Infrastructure.Alerts;

namespace TradingBot.Monitoring
{
    /// <summary>
    /// Monitors decision latency and order latency
    /// Triggers alerts when latency exceeds configured thresholds for N consecutive trades
    /// </summary>
    public class LatencyMonitor : ILatencyMonitor
    {
        // Monitoring constants
        private const int MaxMeasurementHistory = 100;
        private const double MedianPercentile = 0.5;
        private const double P95Percentile = 0.95;
        private const double P99Percentile = 0.99;
        
        private readonly ILogger<LatencyMonitor> _logger;
        private readonly IAlertService _alertService;
        private readonly double _decisionLatencyThreshold;
        private readonly double _orderLatencyThreshold;
        private readonly int _consecutiveThresholdCount;
        
        private readonly Queue<LatencyMeasurement> _decisionLatencies = new();
        private readonly Queue<LatencyMeasurement> _orderLatencies = new();
        private int _consecutiveDecisionViolations;
        private int _consecutiveOrderViolations;
        private readonly object _lockObject = new();

        public LatencyMonitor(ILogger<LatencyMonitor> logger, IAlertService alertService)
        {
            _logger = logger;
            _alertService = alertService;
            
            // Load thresholds from configuration
            _decisionLatencyThreshold = double.Parse(Environment.GetEnvironmentVariable("ALERT_DECISION_LATENCY_THRESHOLD_MS") ?? "5000");
            _orderLatencyThreshold = double.Parse(Environment.GetEnvironmentVariable("ALERT_ORDER_LATENCY_THRESHOLD_MS") ?? "2000");
            _consecutiveThresholdCount = int.Parse(Environment.GetEnvironmentVariable("ALERT_CONSECUTIVE_THRESHOLD_COUNT") ?? "3");
            
            _logger.LogInformation("[LATENCY] Monitor started - Decision threshold: {DecThreshold}ms, Order threshold: {OrderThreshold}ms, Consecutive: {Consecutive}",
                _decisionLatencyThreshold, _orderLatencyThreshold, _consecutiveThresholdCount);
        }

        public void RecordDecisionLatency(double latencyMs, string? context = null)
        {
            lock (_lockObject)
            {
                var measurement = new LatencyMeasurement
                {
                    Timestamp = DateTime.UtcNow,
                    LatencyMs = latencyMs,
                    Context = context ?? string.Empty
                };

                _decisionLatencies.Enqueue(measurement);
                if (_decisionLatencies.Count > MaxMeasurementHistory) // Keep last 100 measurements
                    _decisionLatencies.Dequeue();

                // Check for threshold violation
                if (latencyMs > _decisionLatencyThreshold)
                {
                    _consecutiveDecisionViolations++;
                    _logger.LogWarning("[LATENCY] Decision latency violation #{Count}: {Latency:F2}ms (threshold: {Threshold}ms) - {Context}",
                        _consecutiveDecisionViolations, latencyMs, _decisionLatencyThreshold, context ?? "N/A");

                    if (_consecutiveDecisionViolations >= _consecutiveThresholdCount)
                    {
                        _ = Task.Run(async () => await _alertService.SendLatencyAlertAsync(
                            "Decision Processing", latencyMs, _decisionLatencyThreshold)).ConfigureAwait(false);
                        _consecutiveDecisionViolations; // Reset to avoid spam
                    }
                }
                else
                {
                    _consecutiveDecisionViolations; // Reset on good latency
                }
            }
        }

        public void RecordOrderLatency(double latencyMs, string? context = null)
        {
            lock (_lockObject)
            {
                var measurement = new LatencyMeasurement
                {
                    Timestamp = DateTime.UtcNow,
                    LatencyMs = latencyMs,
                    Context = context ?? string.Empty
                };

                _orderLatencies.Enqueue(measurement);
                if (_orderLatencies.Count > MaxMeasurementHistory) // Keep last 100 measurements
                    _orderLatencies.Dequeue();

                // Check for threshold violation
                if (latencyMs > _orderLatencyThreshold)
                {
                    _consecutiveOrderViolations++;
                    _logger.LogWarning("[LATENCY] Order latency violation #{Count}: {Latency:F2}ms (threshold: {Threshold}ms) - {Context}",
                        _consecutiveOrderViolations, latencyMs, _orderLatencyThreshold, context ?? "N/A");

                    if (_consecutiveOrderViolations >= _consecutiveThresholdCount)
                    {
                        _ = Task.Run(async () => await _alertService.SendLatencyAlertAsync(
                            "Order Processing", latencyMs, _orderLatencyThreshold)).ConfigureAwait(false);
                        _consecutiveOrderViolations; // Reset to avoid spam
                    }
                }
                else
                {
                    _consecutiveOrderViolations; // Reset on good latency
                }
            }
        }

        public ILatencyTracker StartDecisionTracking(string? context = null)
        {
            return new LatencyTracker(this, LatencyType.Decision, context);
        }

        public ILatencyTracker StartOrderTracking(string? context = null)
        {
            return new LatencyTracker(this, LatencyType.Order, context);
        }

        public LatencyStats GetDecisionStats()
        {
            lock (_lockObject)
            {
                return CalculateStats(_decisionLatencies, "Decision");
            }
        }

        public LatencyStats GetOrderStats()
        {
            lock (_lockObject)
            {
                return CalculateStats(_orderLatencies, "Order");
            }
        }

        public LatencyHealthStatus GetLatencyHealth()
        {
            lock (_lockObject)
            {
                var decisionStats = GetDecisionStats();
                var orderStats = GetOrderStats();

                return new LatencyHealthStatus
                {
                    Timestamp = DateTime.UtcNow,
                    DecisionStats = decisionStats,
                    OrderStats = orderStats,
                    IsHealthy = decisionStats.P99 < _decisionLatencyThreshold && orderStats.P99 < _orderLatencyThreshold,
                    ConsecutiveDecisionViolations = _consecutiveDecisionViolations,
                    ConsecutiveOrderViolations = _consecutiveOrderViolations
                };
            }
        }

        private LatencyStats CalculateStats(Queue<LatencyMeasurement> measurements, string component)
        {
            if (!measurements.Any())
            {
                return new LatencyStats
                {
                    Component = component,
                    Count = 0,
                    Average = 0,
                    P50 = 0,
                    P95 = 0,
                    P99 = 0,
                    Max = 0,
                    Min = 0
                };
            }

            var latencies = measurements.Select(m => m.LatencyMs).OrderBy(x => x).ToArray();
            
            return new LatencyStats
            {
                Component = component,
                Count = latencies.Length,
                Average = latencies.Average(),
                P50 = GetPercentile(latencies, MedianPercentile),
                P95 = GetPercentile(latencies, P95Percentile),
                P99 = GetPercentile(latencies, P99Percentile),
                Max = latencies.Max(),
                Min = latencies.Min()
            };
        }

        private static double GetPercentile(double[] sortedValues, double percentile)
        {
            if (sortedValues.Length == 0) return 0;
            if (sortedValues.Length == 1) return sortedValues[0];

            var index = percentile * (sortedValues.Length - 1);
            var lower = (int)Math.Floor(index);
            var upper = (int)Math.Ceiling(index);

            if (lower == upper) return sortedValues[lower];

            var weight = index - lower;
            return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
        }
    }

    public interface ILatencyMonitor
    {
        void RecordDecisionLatency(double latencyMs, string? context = null);
        void RecordOrderLatency(double latencyMs, string? context = null);
        ILatencyTracker StartDecisionTracking(string? context = null);
        ILatencyTracker StartOrderTracking(string? context = null);
        LatencyStats GetDecisionStats();
        LatencyStats GetOrderStats();
        LatencyHealthStatus GetLatencyHealth();
    }

    public interface ILatencyTracker : IDisposable
    {
        void Stop();
    }

    public class LatencyTracker : ILatencyTracker
    {
        private readonly LatencyMonitor _monitor;
        private readonly LatencyType _type;
        private readonly string? _context;
        private readonly Stopwatch _stopwatch;
        private bool _stopped;

        public LatencyTracker(LatencyMonitor monitor, LatencyType type, string? context)
        {
            _monitor = monitor;
            _type = type;
            _context = context;
            _stopwatch = Stopwatch.StartNew();
        }

        public void Stop()
        {
            if (_stopped) return;
            
            _stopwatch.Stop();
            var latencyMs = _stopwatch.Elapsed.TotalMilliseconds;

            switch (_type)
            {
                case LatencyType.Decision:
                    _monitor.RecordDecisionLatency(latencyMs, _context);
                    break;
                case LatencyType.Order:
                    _monitor.RecordOrderLatency(latencyMs, _context);
                    break;
            }

            _stopped = true;
        }

        private bool _disposed;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    Stop();
                }
                _disposed = true;
            }
        }
    }

    public enum LatencyType
    {
        Decision,
        Order
    }

    public class LatencyMeasurement
    {
        public DateTime Timestamp { get; set; }
        public double LatencyMs { get; set; }
        public string Context { get; set; } = string.Empty;
    }

    public class LatencyStats
    {
        public string Component { get; set; } = string.Empty;
        public int Count { get; set; }
        public double Average { get; set; }
        public double P50 { get; set; }
        public double P95 { get; set; }
        public double P99 { get; set; }
        public double Max { get; set; }
        public double Min { get; set; }
    }

    public class LatencyHealthStatus
    {
        public DateTime Timestamp { get; set; }
        public LatencyStats DecisionStats { get; set; } = new();
        public LatencyStats OrderStats { get; set; } = new();
        public bool IsHealthy { get; set; }
        public int ConsecutiveDecisionViolations { get; set; }
        public int ConsecutiveOrderViolations { get; set; }
    }
}