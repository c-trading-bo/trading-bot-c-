using System;
using System.Collections.Generic;
using System.Linq;

namespace OrchestratorAgent.Execution
{
    /// <summary>
    /// Calibration metrics system that tracks reliability and Brier scores
    /// per strategy-config-regime-session combination.
    /// Down-weights poorly calibrated predictions to improve system reliability.
    /// </summary>
    public class CalibrationMetricsSystem
    {
        private readonly Dictionary<string, CalibrationTracker> _trackers = new();
        private readonly Queue<CalibrationEvent> _recentEvents = new();
        private readonly int _maxEventHistory = 2000;

        // Configuration
        public double MinSampleSize { get; set; } = 20; // Minimum predictions for calibration assessment
        public double PoorCalibrationThreshold { get; set; } = 0.25; // Brier score threshold for "poor"
        public double CalibrationUpdateRate { get; set; } = 0.1; // Learning rate for calibration updates
        public TimeSpan CalibrationWindow { get; set; } = TimeSpan.FromDays(14); // Rolling window for calibration

        /// <summary>
        /// Records a prediction and its context for later calibration assessment
        /// </summary>
        public void RecordPrediction(PredictionRecord prediction)
        {
            try
            {
                var key = GetCalibrationKey(prediction.Context);

                if (!_trackers.ContainsKey(key))
                {
                    _trackers[key] = new CalibrationTracker(key);
                }

                var tracker = _trackers[key];
                tracker.AddPrediction(prediction);

                // Add to recent events for global analysis
                _recentEvents.Enqueue(new CalibrationEvent
                {
                    Key = key,
                    Prediction = prediction,
                    Timestamp = DateTime.Now
                });

                while (_recentEvents.Count > _maxEventHistory)
                {
                    _recentEvents.Dequeue();
                }

                Console.WriteLine($"[CALIBRATION] Recorded prediction: {key} prob={prediction.PredictedProbability:F3}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[CALIBRATION] Record error: {ex.Message}");
            }
        }

        /// <summary>
        /// Updates prediction with actual outcome for calibration calculation
        /// </summary>
        public void UpdateWithOutcome(string predictionId, bool actualOutcome)
        {
            try
            {
                // Find the prediction across all trackers
                foreach (var tracker in _trackers.Values)
                {
                    if (tracker.UpdateOutcome(predictionId, actualOutcome))
                    {
                        Console.WriteLine($"[CALIBRATION] Updated outcome: {predictionId} actual={actualOutcome}");
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[CALIBRATION] Update error: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets calibration weight for a specific context (higher = better calibrated)
        /// </summary>
        public double GetCalibrationWeight(PredictionContext context)
        {
            try
            {
                var key = GetCalibrationKey(context);

                if (!_trackers.ContainsKey(key))
                {
                    return 0.5; // Neutral weight for unknown contexts
                }

                var tracker = _trackers[key];
                var metrics = tracker.GetCalibrationMetrics();

                if (metrics.SampleSize < MinSampleSize)
                {
                    return 0.6; // Slightly optimistic for small samples
                }

                // Convert Brier score to weight (lower Brier = higher weight)
                var maxWeight = 1.0;
                var minWeight = 0.1;
                var normalizedWeight = Math.Max(0, 1 - metrics.BrierScore / PoorCalibrationThreshold);

                return minWeight + (maxWeight - minWeight) * normalizedWeight;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[CALIBRATION] Weight calculation error: {ex.Message}");
                return 0.5;
            }
        }

        /// <summary>
        /// Adjusts prediction probability based on calibration history
        /// </summary>
        public double CalibrateProability(double rawProbability, PredictionContext context)
        {
            try
            {
                var key = GetCalibrationKey(context);

                if (!_trackers.ContainsKey(key))
                {
                    return rawProbability; // No calibration data available
                }

                var tracker = _trackers[key];
                var metrics = tracker.GetCalibrationMetrics();

                if (metrics.SampleSize < MinSampleSize)
                {
                    return rawProbability; // Insufficient data for calibration
                }

                // Apply calibration correction based on historical bias
                var calibratedProb = rawProbability;

                // Correct for systematic over/under-confidence
                if (metrics.CalibrationSlope.HasValue && metrics.CalibrationIntercept.HasValue)
                {
                    // Linear calibration: p_calibrated = slope * p_raw + intercept
                    calibratedProb = metrics.CalibrationSlope.Value * rawProbability + metrics.CalibrationIntercept.Value;
                }
                else
                {
                    // Simple bias correction
                    var bias = metrics.MeanPrediction - metrics.MeanOutcome;
                    calibratedProb = rawProbability - bias * CalibrationUpdateRate;
                }

                // Ensure probability stays in valid range
                calibratedProb = Math.Max(0.01, Math.Min(0.99, calibratedProb));

                if (Math.Abs(calibratedProb - rawProbability) > 0.05)
                {
                    Console.WriteLine($"[CALIBRATION] Adjusted {key}: {rawProbability:F3} -> {calibratedProb:F3}");
                }

                return calibratedProb;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[CALIBRATION] Calibration error: {ex.Message}");
                return rawProbability;
            }
        }

        /// <summary>
        /// Gets comprehensive calibration report for all contexts
        /// </summary>
        public CalibrationReport GetCalibrationReport()
        {
            var report = new CalibrationReport
            {
                GeneratedAt = DateTime.Now,
                ContextReports = new List<ContextCalibrationReport>()
            };

            foreach (var kvp in _trackers)
            {
                var metrics = kvp.Value.GetCalibrationMetrics();
                if (metrics.SampleSize >= 5) // Only include contexts with some data
                {
                    report.ContextReports.Add(new ContextCalibrationReport
                    {
                        Context = kvp.Key,
                        Metrics = metrics,
                        Weight = GetCalibrationWeight(ParseContext(kvp.Key)),
                        RecommendedAction = GetRecommendedAction(metrics)
                    });
                }
            }

            // Sort by Brier score (best calibrated first)
            report.ContextReports = report.ContextReports.OrderBy(r => r.Metrics.BrierScore).ToList();

            // Calculate overall statistics
            if (report.ContextReports.Any())
            {
                report.OverallBrierScore = report.ContextReports.Average(r => r.Metrics.BrierScore);
                report.OverallReliability = report.ContextReports.Average(r => r.Metrics.Reliability);
                report.TotalPredictions = report.ContextReports.Sum(r => r.Metrics.SampleSize);
            }

            return report;
        }

        /// <summary>
        /// Performs nightly calibration cleanup and optimization
        /// </summary>
        public void PerformNightlyMaintenance()
        {
            try
            {
                Console.WriteLine("[CALIBRATION] Starting nightly maintenance...");

                var cutoff = DateTime.Now.Subtract(CalibrationWindow);
                var cleaned = 0;

                // Clean old data from trackers
                foreach (var tracker in _trackers.Values)
                {
                    cleaned += tracker.CleanOldData(cutoff);
                }

                // Remove trackers with insufficient data
                var toRemove = _trackers.Where(kvp => kvp.Value.GetCalibrationMetrics().SampleSize < 5).Select(kvp => kvp.Key).ToList();
                foreach (var key in toRemove)
                {
                    _trackers.Remove(key);
                }

                Console.WriteLine($"[CALIBRATION] Maintenance complete: cleaned {cleaned} old records, removed {toRemove.Count} inactive trackers");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[CALIBRATION] Maintenance error: {ex.Message}");
            }
        }

        private string GetCalibrationKey(PredictionContext context)
        {
            return $"{context.Strategy}_{context.Symbol}_{context.Regime}_{context.Session}_{context.ConfigHash}";
        }

        private PredictionContext ParseContext(string key)
        {
            var parts = key.Split('_');
            return new PredictionContext
            {
                Strategy = parts.Length > 0 ? parts[0] : "",
                Symbol = parts.Length > 1 ? parts[1] : "",
                Regime = parts.Length > 2 ? parts[2] : "",
                Session = parts.Length > 3 ? parts[3] : "",
                ConfigHash = parts.Length > 4 ? parts[4] : ""
            };
        }

        private string GetRecommendedAction(CalibrationMetrics metrics)
        {
            if (metrics.SampleSize < MinSampleSize)
                return "Collect more data";

            if (metrics.BrierScore > PoorCalibrationThreshold)
                return "Poor calibration - reduce weight";

            if (metrics.Reliability < 0.05)
                return "Well calibrated";

            if (metrics.MeanPrediction > metrics.MeanOutcome + 0.1)
                return "Overconfident - reduce predictions";

            if (metrics.MeanPrediction < metrics.MeanOutcome - 0.1)
                return "Underconfident - increase predictions";

            return "Good calibration";
        }
    }

    public class CalibrationTracker
    {
        private readonly string _key = null!;
        private readonly List<CalibrationDataPoint> _dataPoints = new();

        public CalibrationTracker(string key)
        {
            _key = key;
        }

        public void AddPrediction(PredictionRecord prediction)
        {
            _dataPoints.Add(new CalibrationDataPoint
            {
                Id = prediction.Id,
                PredictedProbability = prediction.PredictedProbability,
                Timestamp = prediction.Timestamp,
                HasOutcome = false
            });
        }

        public bool UpdateOutcome(string predictionId, bool outcome)
        {
            var point = _dataPoints.FirstOrDefault(p => p.Id == predictionId);
            if (point != null)
            {
                point.ActualOutcome = outcome;
                point.HasOutcome = true;
                return true;
            }
            return false;
        }

        public CalibrationMetrics GetCalibrationMetrics()
        {
            var completedPredictions = _dataPoints.Where(p => p.HasOutcome).ToList();

            if (completedPredictions.Count == 0)
            {
                return new CalibrationMetrics { SampleSize = 0 };
            }

            var metrics = new CalibrationMetrics
            {
                SampleSize = completedPredictions.Count,
                MeanPrediction = completedPredictions.Average(p => p.PredictedProbability),
                MeanOutcome = completedPredictions.Average(p => p.ActualOutcome ? 1.0 : 0.0)
            };

            // Calculate Brier Score: BS = (1/n) * Σ(forecast - outcome)²
            metrics.BrierScore = completedPredictions.Average(p =>
                Math.Pow(p.PredictedProbability - (p.ActualOutcome ? 1.0 : 0.0), 2));

            // Calculate Reliability (calibration component of Brier Score)
            metrics.Reliability = CalculateReliability(completedPredictions);

            // Calculate calibration slope and intercept if enough data
            if (completedPredictions.Count >= 10)
            {
                var (slope, intercept) = CalculateCalibrationLine(completedPredictions);
                metrics.CalibrationSlope = slope;
                metrics.CalibrationIntercept = intercept;
            }

            return metrics;
        }

        public int CleanOldData(DateTime cutoff)
        {
            var countBefore = _dataPoints.Count;
            _dataPoints.RemoveAll(p => p.Timestamp < cutoff);
            return countBefore - _dataPoints.Count;
        }

        private double CalculateReliability(List<CalibrationDataPoint> points)
        {
            // Group predictions into bins and calculate reliability
            const int numBins = 10;
            var reliability = 0.0;

            for (int bin = 0; bin < numBins; bin++)
            {
                var binStart = bin / (double)numBins;
                var binEnd = (bin + 1) / (double)numBins;

                var binPoints = points.Where(p => p.PredictedProbability >= binStart && p.PredictedProbability < binEnd).ToList();

                if (binPoints.Count > 0)
                {
                    var meanForecast = binPoints.Average(p => p.PredictedProbability);
                    var meanOutcome = binPoints.Average(p => p.ActualOutcome ? 1.0 : 0.0);
                    var binWeight = binPoints.Count / (double)points.Count;

                    reliability += binWeight * Math.Pow(meanForecast - meanOutcome, 2);
                }
            }

            return reliability;
        }

        private (double slope, double intercept) CalculateCalibrationLine(List<CalibrationDataPoint> points)
        {
            // Linear regression: outcome = slope * prediction + intercept
            var n = points.Count;
            var sumX = points.Sum(p => p.PredictedProbability);
            var sumY = points.Sum(p => p.ActualOutcome ? 1.0 : 0.0);
            var sumXY = points.Sum(p => p.PredictedProbability * (p.ActualOutcome ? 1.0 : 0.0));
            var sumX2 = points.Sum(p => p.PredictedProbability * p.PredictedProbability);

            var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            var intercept = (sumY - slope * sumX) / n;

            return (slope, intercept);
        }
    }

    public class PredictionRecord
    {
        public string Id { get; set; } = "";
        public double PredictedProbability { get; set; }
        public PredictionContext Context { get; set; } = new();
        public DateTime Timestamp { get; set; }
        public string Description { get; set; } = "";
    }

    public class PredictionContext
    {
        public string Strategy { get; set; } = "";
        public string Symbol { get; set; } = "";
        public string Regime { get; set; } = "";
        public string Session { get; set; } = "";
        public string ConfigHash { get; set; } = "";
    }

    public class CalibrationDataPoint
    {
        public string Id { get; set; } = "";
        public double PredictedProbability { get; set; }
        public bool ActualOutcome { get; set; }
        public bool HasOutcome { get; set; }
        public DateTime Timestamp { get; set; }
    }

    public class CalibrationMetrics
    {
        public int SampleSize { get; set; }
        public double BrierScore { get; set; }
        public double Reliability { get; set; }
        public double MeanPrediction { get; set; }
        public double MeanOutcome { get; set; }
        public double? CalibrationSlope { get; set; }
        public double? CalibrationIntercept { get; set; }
    }

    public class CalibrationEvent
    {
        public string Key { get; set; } = "";
        public PredictionRecord Prediction { get; set; } = new();
        public DateTime Timestamp { get; set; }
    }

    public class CalibrationReport
    {
        public DateTime GeneratedAt { get; set; }
        public List<ContextCalibrationReport> ContextReports { get; set; } = new();
        public double OverallBrierScore { get; set; }
        public double OverallReliability { get; set; }
        public int TotalPredictions { get; set; }
    }

    public class ContextCalibrationReport
    {
        public string Context { get; set; } = "";
        public CalibrationMetrics Metrics { get; set; } = new();
        public double Weight { get; set; }
        public string RecommendedAction { get; set; } = "";
    }
}
