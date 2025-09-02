using System;
using System.Collections.Generic;
using System.Linq;

namespace OrchestratorAgent.Execution
{
    /// <summary>
    /// Enhanced drift detection system with automatic safe mode switching.
    /// Uses Page-Hinkley test to detect regime changes and automatically
    /// switches to conservative "safe policy" for session remainder.
    /// </summary>
    public class DriftDetectionSafeMode
    {
        private readonly Dictionary<string, PageHinkleyDetector> _detectors = new();
        private readonly Dictionary<string, SafeModeState> _safeModeStates = new();
        private readonly List<DriftEvent> _driftHistory = new();

        // Configuration
        public double DriftThreshold { get; set; } = 3.0; // Page-Hinkley threshold
        public double SafeModeDuration { get; set; } = 4.0; // Hours to stay in safe mode
        public double SafeModePositionMultiplier { get; set; } = 0.5; // Reduce position size by 50%
        public string[] SafeModeStrategies { get; set; } = { "S2a" }; // Only most conservative strategies
        public double MinDriftSignificance { get; set; } = 2.5; // Minimum significance to trigger safe mode

        /// <summary>
        /// Updates drift detection with new performance metric
        /// </summary>
        public void UpdateMetric(string context, double value, DateTime timestamp)
        {
            try
            {
                var detector = GetOrCreateDetector(context);
                var driftDetected = detector.Update(value, timestamp);

                if (driftDetected && detector.CurrentSignificance >= MinDriftSignificance)
                {
                    TriggerSafeMode(context, detector.CurrentSignificance, timestamp);
                }

                // Log significant changes
                if (Math.Abs(detector.CurrentCUSUM) > DriftThreshold * 0.7)
                {
                    Console.WriteLine($"[DRIFT-DETECT] {context}: CUSUM={detector.CurrentCUSUM:F2}, trend={'↑' + (detector.CurrentCUSUM > 0 ? "UP" : "DOWN")}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DRIFT-DETECT] Update error for {context}: {ex.Message}");
            }
        }

        /// <summary>
        /// Checks if a strategy should be filtered due to safe mode
        /// </summary>
        public bool IsStrategyAllowed(string strategy, string symbol, string regime)
        {
            var context = GetContext(symbol, regime);
            var safeState = GetSafeModeState(context);

            if (!safeState.IsInSafeMode)
                return true;

            // Only allow safe mode strategies
            var isAllowed = SafeModeStrategies.Contains(strategy);

            if (!isAllowed)
            {
                Console.WriteLine($"[SAFE-MODE] Blocked {strategy} for {context} (safe mode active until {safeState.ExitTime:HH:mm:ss})");
            }

            return isAllowed;
        }

        /// <summary>
        /// Gets position size multiplier considering safe mode
        /// </summary>
        public double GetPositionSizeMultiplier(string symbol, string regime)
        {
            var context = GetContext(symbol, regime);
            var safeState = GetSafeModeState(context);

            if (safeState.IsInSafeMode)
            {
                Console.WriteLine($"[SAFE-MODE] Reducing position size: {SafeModePositionMultiplier:F1}x for {context}");
                return SafeModePositionMultiplier;
            }

            return 1.0; // Normal sizing
        }

        /// <summary>
        /// Gets current drift status for all contexts
        /// </summary>
        public DriftStatusReport GetDriftStatus()
        {
            var report = new DriftStatusReport
            {
                GeneratedAt = DateTime.Now,
                ContextStatuses = new List<ContextDriftStatus>()
            };

            foreach (var kvp in _detectors)
            {
                var detector = kvp.Value;
                var safeState = GetSafeModeState(kvp.Key);

                report.ContextStatuses.Add(new ContextDriftStatus
                {
                    Context = kvp.Key,
                    CurrentCUSUM = detector.CurrentCUSUM,
                    Significance = detector.CurrentSignificance,
                    IsInSafeMode = safeState.IsInSafeMode,
                    SafeModeExitTime = safeState.ExitTime,
                    TrendDirection = detector.CurrentCUSUM > 0 ? "UP" : "DOWN",
                    RiskLevel = GetRiskLevel(detector.CurrentSignificance)
                });
            }

            // Calculate overall system status
            var activeSafeModes = report.ContextStatuses.Count(s => s.IsInSafeMode);
            var highRiskContexts = report.ContextStatuses.Count(s => s.RiskLevel == "HIGH");

            if (highRiskContexts > 0)
                report.OverallRiskLevel = "HIGH";
            else if (activeSafeModes > 0)
                report.OverallRiskLevel = "MEDIUM";
            else
                report.OverallRiskLevel = "LOW";

            report.ActiveSafeModes = activeSafeModes;

            return report;
        }

        /// <summary>
        /// Forces exit from safe mode for a specific context (manual override)
        /// </summary>
        public void ForceExitSafeMode(string context, string reason = "Manual override")
        {
            if (_safeModeStates.ContainsKey(context))
            {
                _safeModeStates[context].IsInSafeMode = false;
                _safeModeStates[context].ExitTime = DateTime.Now;
                Console.WriteLine($"[SAFE-MODE] Forced exit for {context}: {reason}");
            }
        }

        /// <summary>
        /// Performs nightly maintenance - reset detectors, clean old data
        /// </summary>
        public void PerformNightlyReset()
        {
            try
            {
                Console.WriteLine("[DRIFT-DETECT] Performing nightly reset...");

                // Reset all detectors for new session
                foreach (var detector in _detectors.Values)
                {
                    detector.Reset();
                }

                // Clear expired safe mode states
                var expiredStates = _safeModeStates.Where(kvp =>
                    kvp.Value.IsInSafeMode && DateTime.Now > kvp.Value.ExitTime).ToList();

                foreach (var expired in expiredStates)
                {
                    _safeModeStates[expired.Key].IsInSafeMode = false;
                    Console.WriteLine($"[SAFE-MODE] Expired safe mode for {expired.Key}");
                }

                // Clean old drift history (keep last 30 days)
                var cutoff = DateTime.Now.AddDays(-30);
                _driftHistory.RemoveAll(d => d.Timestamp < cutoff);

                Console.WriteLine($"[DRIFT-DETECT] Reset complete. Drift history: {_driftHistory.Count} events");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DRIFT-DETECT] Reset error: {ex.Message}");
            }
        }

        private void TriggerSafeMode(string context, double significance, DateTime timestamp)
        {
            var exitTime = timestamp.AddHours(SafeModeDuration);

            _safeModeStates[context] = new SafeModeState
            {
                IsInSafeMode = true,
                EntryTime = timestamp,
                ExitTime = exitTime,
                TriggerSignificance = significance
            };

            // Record drift event
            _driftHistory.Add(new DriftEvent
            {
                Context = context,
                Timestamp = timestamp,
                Significance = significance,
                SafeModeTriggered = true
            });

            Console.WriteLine($"[SAFE-MODE] 🚨 TRIGGERED for {context}: significance={significance:F2}, exit at {exitTime:HH:mm:ss}");
            Console.WriteLine($"[SAFE-MODE] Active restrictions: strategies={string.Join(",", SafeModeStrategies)}, positionMult={SafeModePositionMultiplier:F1}x");
        }

        private PageHinkleyDetector GetOrCreateDetector(string context)
        {
            if (!_detectors.ContainsKey(context))
            {
                _detectors[context] = new PageHinkleyDetector(DriftThreshold);
            }
            return _detectors[context];
        }

        private SafeModeState GetSafeModeState(string context)
        {
            if (!_safeModeStates.ContainsKey(context))
            {
                _safeModeStates[context] = new SafeModeState();
            }

            var state = _safeModeStates[context];

            // Auto-exit if time expired
            if (state.IsInSafeMode && DateTime.Now > state.ExitTime)
            {
                state.IsInSafeMode = false;
                Console.WriteLine($"[SAFE-MODE] Auto-exit for {context} (time expired)");
            }

            return state;
        }

        private static string GetContext(string symbol, string regime) => $"{symbol}_{regime}";

        private string GetRiskLevel(double significance)
        {
            if (significance >= MinDriftSignificance) return "HIGH";
            if (significance >= MinDriftSignificance * 0.7) return "MEDIUM";
            return "LOW";
        }
    }

    /// <summary>
    /// Page-Hinkley test for detecting changes in time series mean
    /// </summary>
    public class PageHinkleyDetector
    {
        private readonly double _threshold;
        private readonly Queue<double> _values = new();
        private readonly int _windowSize = 50;

        public double CurrentCUSUM { get; private set; }
        public double CurrentSignificance { get; private set; }
        public double BaseMean { get; private set; }
        public double BaseStdDev { get; private set; }
        public bool IsInitialized { get; private set; }

        public PageHinkleyDetector(double threshold)
        {
            _threshold = threshold;
        }

        public bool Update(double value, DateTime timestamp)
        {
            _values.Enqueue(value);

            // Maintain sliding window
            while (_values.Count > _windowSize)
            {
                _values.Dequeue();
            }

            // Initialize baseline if we have enough data
            if (!IsInitialized && _values.Count >= 20)
            {
                var baselineValues = _values.Take(15).ToArray(); // Use first 15 values as baseline
                BaseMean = baselineValues.Average();
                BaseStdDev = CalculateStdDev(baselineValues, BaseMean);
                IsInitialized = true;

                Console.WriteLine($"[PAGE-HINKLEY] Initialized: mean={BaseMean:F3}, stddev={BaseStdDev:F3}");
            }

            if (!IsInitialized || Math.Abs(BaseStdDev) < 1e-10)
                return false;

            // Calculate standardized deviation from baseline
            var deviation = (value - BaseMean) / BaseStdDev;

            // Update CUSUM (Cumulative Sum)
            CurrentCUSUM = Math.Max(0, CurrentCUSUM + deviation - 0.5); // drift magnitude parameter = 0.5

            // Calculate significance level
            CurrentSignificance = CurrentCUSUM;

            // Detect drift
            bool driftDetected = CurrentCUSUM > _threshold;

            if (driftDetected)
            {
                Console.WriteLine($"[PAGE-HINKLEY] 🚨 DRIFT DETECTED: CUSUM={CurrentCUSUM:F2} > {_threshold:F1}");
                Reset(); // Reset after detection
            }

            return driftDetected;
        }

        public void Reset()
        {
            CurrentCUSUM = 0.0;
            CurrentSignificance = 0.0;
            // Keep baseline mean/stddev for comparison
        }

        private double CalculateStdDev(double[] values, double mean)
        {
            if (values.Length < 2) return 1.0;

            var variance = values.Sum(v => Math.Pow(v - mean, 2)) / (values.Length - 1);
            return Math.Sqrt(variance);
        }
    }

    public class SafeModeState
    {
        public bool IsInSafeMode { get; set; } = false;
        public DateTime EntryTime { get; set; }
        public DateTime ExitTime { get; set; }
        public double TriggerSignificance { get; set; }
    }

    public class DriftEvent
    {
        public string Context { get; set; } = "";
        public DateTime Timestamp { get; set; }
        public double Significance { get; set; }
        public bool SafeModeTriggered { get; set; }
    }

    public class DriftStatusReport
    {
        public DateTime GeneratedAt { get; set; }
        public List<ContextDriftStatus> ContextStatuses { get; set; } = new();
        public string OverallRiskLevel { get; set; } = "";
        public int ActiveSafeModes { get; set; }
    }

    public class ContextDriftStatus
    {
        public string Context { get; set; } = "";
        public double CurrentCUSUM { get; set; }
        public double Significance { get; set; }
        public bool IsInSafeMode { get; set; }
        public DateTime SafeModeExitTime { get; set; }
        public string TrendDirection { get; set; } = "";
        public string RiskLevel { get; set; } = "";
    }
}
