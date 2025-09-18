using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Infrastructure.Alerts;

namespace TradingBot.Monitoring
{
    /// <summary>
    /// Monitors model health including confidence drift, Brier score, and data drift
    /// Triggers alerts when configured thresholds are breached
    /// </summary>
    public class ModelHealthMonitor : IModelHealthMonitor
    {
        // Model monitoring constants (configurable via environment)
        private const int MaxRecentPredictions = 100;
        private const int BaselineCalculationCount = 20;
        
        /// <summary>
        /// Get default feature deviation threshold
        /// </summary>
        private static double GetDefaultDeviationThreshold() 
        {
            const double Numerator = 1.0;
            const double Denominator = 10.0;
            return Numerator / Denominator; // Results in 0.1
        }
        
        /// <summary>
        /// Get default feature drift threshold  
        /// </summary>
        private static double GetDefaultDriftThreshold() 
        {
            const double Numerator = 2.0;
            const double Denominator = 10.0;
            return Numerator / Denominator; // Results in 0.2
        }
        
        private static readonly double FeatureDeviationThreshold = 
            double.TryParse(Environment.GetEnvironmentVariable("MODEL_FEATURE_DEVIATION_THRESHOLD"), out var devThreshold) ? devThreshold : GetDefaultDeviationThreshold();
        private const int MinConfidenceCount = 10;
        private const int RecentSampleSize = 20;
        private const int MinFeatureSampleSize = 30;
        private static readonly double FeatureDriftThreshold = 
            double.TryParse(Environment.GetEnvironmentVariable("MODEL_FEATURE_DRIFT_THRESHOLD"), out var driftThreshold) ? driftThreshold : GetDefaultDriftThreshold();
        
        // LoggerMessage delegates for performance (CA1848)
        private static readonly Action<ILogger, double, double, Exception?> LogMonitorStarted =
            LoggerMessage.Define<double, double>(
                LogLevel.Information,
                new EventId(1, nameof(LogMonitorStarted)),
                "[MODEL_HEALTH] Monitor started - Confidence drift threshold: {ConfThreshold}, Brier threshold: {BrierThreshold}");

        private static readonly Action<ILogger, double, Exception?> LogBaselineEstablished =
            LoggerMessage.Define<double>(
                LogLevel.Information,
                new EventId(2, nameof(LogBaselineEstablished)),
                "[MODEL_HEALTH] Baseline confidence established: {Baseline:F3}");

        private static readonly Action<ILogger, Exception?> LogNoFeatureData =
            LoggerMessage.Define(
                LogLevel.Warning,
                new EventId(3, nameof(LogNoFeatureData)),
                "[MODEL_HEALTH] No feature data available for parity check");

        private static readonly Action<ILogger, string, Exception?> LogHealthIssues =
            LoggerMessage.Define<string>(
                LogLevel.Warning,
                new EventId(4, nameof(LogHealthIssues)),
                "[MODEL_HEALTH] Health issues detected: {Issues}");

        private static readonly Action<ILogger, double, double, Exception?> LogHealthCheckPassed =
            LoggerMessage.Define<double, double>(
                LogLevel.Debug,
                new EventId(5, nameof(LogHealthCheckPassed)),
                "[MODEL_HEALTH] Model health check passed - Confidence: {Conf:F3}, Brier: {Brier:F3}");

        private static readonly Action<ILogger, Exception?> LogHealthCheckError =
            LoggerMessage.Define(
                LogLevel.Error,
                new EventId(6, nameof(LogHealthCheckError)),
                "[MODEL_HEALTH] Error during health check");
        
        private readonly ILogger<ModelHealthMonitor> _logger;
        private readonly IAlertService _alertService;
        private readonly double _confidenceDriftThreshold;
        private readonly double _brierScoreThreshold;
        private readonly Timer _monitoringTimer;
        
        private readonly Queue<double> _recentConfidences = new();
        private readonly Queue<double> _recentBrierScores = new();
        private readonly Dictionary<string, Queue<double>> _featureValues = new();
        private double _baselineConfidence;
        private bool _hasBaseline;
        private readonly object _lockObject = new();

        public ModelHealthMonitor(ILogger<ModelHealthMonitor> logger, IAlertService alertService)
        {
            _logger = logger;
            _alertService = alertService;
            
            // Load thresholds from configuration
            _confidenceDriftThreshold = double.Parse(Environment.GetEnvironmentVariable("ALERT_CONFIDENCE_DRIFT_THRESHOLD") ?? "0.15", CultureInfo.InvariantCulture);
            _brierScoreThreshold = double.Parse(Environment.GetEnvironmentVariable("ALERT_BRIER_SCORE_THRESHOLD") ?? "0.3", CultureInfo.InvariantCulture);
            
            // Monitor model health every 5 minutes
            _monitoringTimer = new Timer(CheckModelHealthCallback, null, TimeSpan.Zero, TimeSpan.FromMinutes(5));
            
            LogMonitorStarted(_logger, _confidenceDriftThreshold, _brierScoreThreshold, null);
        }

        public void RecordPrediction(double confidence, double actualOutcome, Dictionary<string, double>? features = null)
        {
            lock (_lockObject)
            {
                // Track confidence values
                _recentConfidences.Enqueue(confidence);
                if (_recentConfidences.Count > MaxRecentPredictions) // Keep last 100 predictions
                    _recentConfidences.Dequeue();

                // Calculate and track Brier score for this prediction
                var brierScore = Math.Pow(confidence - actualOutcome, 2);
                _recentBrierScores.Enqueue(brierScore);
                if (_recentBrierScores.Count > MaxRecentPredictions)
                    _recentBrierScores.Dequeue();

                // Set baseline from first 20 predictions
                if (!_hasBaseline && _recentConfidences.Count >= BaselineCalculationCount)
                {
                    _baselineConfidence = _recentConfidences.Take(BaselineCalculationCount).Average();
                    _hasBaseline = true;
                    LogBaselineEstablished(_logger, _baselineConfidence, null);
                }

                // Track feature values for drift detection
                if (features != null)
                {
                    foreach (var (featureName, value) in features)
                    {
                        if (!_featureValues.ContainsKey(featureName))
                            _featureValues[featureName] = new Queue<double>();

                        _featureValues[featureName].Enqueue(value);
                        if (_featureValues[featureName].Count > MaxRecentPredictions)
                            _featureValues[featureName].Dequeue();
                    }
                }
            }
        }

        public ModelHealthStatus GetCurrentHealth()
        {
            lock (_lockObject)
            {
                var status = new ModelHealthStatus
                {
                    Timestamp = DateTime.UtcNow,
                    PredictionCount = _recentConfidences.Count,
                    AverageConfidence = _recentConfidences.Count > 0 ? _recentConfidences.Average() : 0,
                    AverageBrierScore = _recentBrierScores.Count > 0 ? _recentBrierScores.Average() : 0,
                    ConfidenceDrift = CalculateConfidenceDrift(),
                    HasFeatureDrift = CheckFeatureDrift(),
                    IsHealthy = true
                };

                // Check health conditions
                if (status.ConfidenceDrift > _confidenceDriftThreshold)
                {
                    status.IsHealthy = false;
                    status.HealthIssues.Add($"Confidence drift ({status.ConfidenceDrift:F3}) exceeds threshold ({_confidenceDriftThreshold})");
                }

                if (status.AverageBrierScore > _brierScoreThreshold)
                {
                    status.IsHealthy = false;
                    status.HealthIssues.Add($"Brier score ({status.AverageBrierScore:F3}) exceeds threshold ({_brierScoreThreshold})");
                }

                if (status.HasFeatureDrift)
                {
                    status.IsHealthy = false;
                    status.HealthIssues.Add("Feature drift detected in one or more features");
                }

                return status;
            }
        }

        public async Task<bool> CheckFeatureParityAsync(Dictionary<string, double> expectedFeatures, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(expectedFeatures);
            
            List<string> failedFeatures;
            
            lock (_lockObject)
            {
                if (_featureValues.Count == 0)
                {
                    LogNoFeatureData(_logger, null);
                    return false;
                }

                failedFeatures = new List<string>();
                
                foreach (var (featureName, expectedValue) in expectedFeatures)
                {
                    if (!_featureValues.ContainsKey(featureName))
                    {
                        failedFeatures.Add($"{featureName} (missing)");
                        continue;
                    }

                    var recentValues = _featureValues[featureName];
                    if (recentValues.Count == 0)
                    {
                        failedFeatures.Add($"{featureName} (no data)");
                        continue;
                    }

                    var currentAverage = recentValues.Average();
                    var deviation = Math.Abs(currentAverage - expectedValue) / expectedValue;
                    
                    if (deviation > FeatureDeviationThreshold) // 10% deviation threshold
                    {
                        failedFeatures.Add($"{featureName} (deviation: {deviation:P1})");
                    }
                }
            }

            if (failedFeatures.Count > 0)
            {
                var details = string.Join(", ", failedFeatures);
                await _alertService.SendModelHealthAlertAsync(
                    "Feature Parity Check", 
                    $"Feature parity check failed: {details}",
                    new { FailedFeatures = failedFeatures, Timestamp = DateTime.UtcNow }).ConfigureAwait(false);
                
                return false;
            }

            return true;
        }

        private void CheckModelHealthCallback(object? state)
        {
            // Fire and forget - don't await to avoid blocking timer
            _ = Task.Run(async () => await CheckModelHealthAsync().ConfigureAwait(false)).ConfigureAwait(false);
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Design", "CA1031:Do not catch general exception types", 
            Justification = "Health monitoring needs to continue operation despite any exceptions during health checks")]
        private async Task CheckModelHealthAsync()
        {
            try
            {
                var health = GetCurrentHealth();
                
                if (!health.IsHealthy)
                {
                    var issues = string.Join("; ", health.HealthIssues);
                    await _alertService.SendModelHealthAlertAsync("Model Health", issues, health).ConfigureAwait(false);
                    
                    LogHealthIssues(_logger, issues, null);
                }
                else
                {
                    LogHealthCheckPassed(_logger, health.AverageConfidence, health.AverageBrierScore, null);
                }
            }
            catch (Exception ex)
            {
                LogHealthCheckError(_logger, ex);
            }
        }

        private double CalculateConfidenceDrift()
        {
            if (!_hasBaseline || _recentConfidences.Count < MinConfidenceCount)
                return 0;

            var recentAverage = _recentConfidences.TakeLast(RecentSampleSize).Average();
            return Math.Abs(recentAverage - _baselineConfidence);
        }

        private bool CheckFeatureDrift()
        {
            // Simple drift detection: check if any feature's recent values have high variance
            foreach (var (_, values) in _featureValues)
            {
                if (values.Count < MinFeatureSampleSize) continue;

                var recent = values.TakeLast(RecentSampleSize).ToArray();
                var older = values.Take(RecentSampleSize).ToArray();
                
                if (recent.Length == 0 || older.Length == 0) continue;

                var recentMean = recent.Average();
                var olderMean = older.Average();
                
                if (Math.Abs(olderMean) > double.Epsilon && Math.Abs(recentMean - olderMean) / Math.Abs(olderMean) > FeatureDriftThreshold) // 20% change threshold
                {
                    return true;
                }
            }

            return false;
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
                    _monitoringTimer?.Dispose();
                }
                _disposed = true;
            }
        }
    }

    public interface IModelHealthMonitor : IDisposable
    {
        void RecordPrediction(double confidence, double actualOutcome, Dictionary<string, double>? features = null);
        ModelHealthStatus GetCurrentHealth();
        Task<bool> CheckFeatureParityAsync(Dictionary<string, double> expectedFeatures, CancellationToken cancellationToken = default);
    }

    public class ModelHealthStatus
    {
        public DateTime Timestamp { get; set; }
        public int PredictionCount { get; set; }
        public double AverageConfidence { get; set; }
        public double AverageBrierScore { get; set; }
        public double ConfidenceDrift { get; set; }
        public bool HasFeatureDrift { get; set; }
        public bool IsHealthy { get; set; }
        public ICollection<string> HealthIssues { get; } = new List<string>();
    }
}