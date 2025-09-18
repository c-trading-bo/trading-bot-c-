using System;
using System.Collections.Generic;
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
        // Model monitoring constants
        private const int MaxRecentPredictions = 100;
        private const int BaselineCalculationCount = 20;
        private const double FeatureDeviationThreshold = 0.1; // 10% deviation threshold
        private const int MinConfidenceCount = 10;
        private const int RecentSampleSize = 20;
        private const int MinFeatureSampleSize = 30;
        private const double FeatureDriftThreshold = 0.2; // 20% change threshold
        
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
            _confidenceDriftThreshold = double.Parse(Environment.GetEnvironmentVariable("ALERT_CONFIDENCE_DRIFT_THRESHOLD") ?? "0.15");
            _brierScoreThreshold = double.Parse(Environment.GetEnvironmentVariable("ALERT_BRIER_SCORE_THRESHOLD") ?? "0.3");
            
            // Monitor model health every 5 minutes
            _monitoringTimer = new Timer(CheckModelHealthCallback, null, TimeSpan.Zero, TimeSpan.FromMinutes(5));
            
            _logger.LogInformation("[MODEL_HEALTH] Monitor started - Confidence drift threshold: {ConfThreshold}, Brier threshold: {BrierThreshold}",
                _confidenceDriftThreshold, _brierScoreThreshold);
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
                    _logger.LogInformation("[MODEL_HEALTH] Baseline confidence established: {Baseline:F3}", _baselineConfidence);
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
                    AverageConfidence = _recentConfidences.Any() ? _recentConfidences.Average() : 0,
                    AverageBrierScore = _recentBrierScores.Any() ? _recentBrierScores.Average() : 0,
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
            List<string> failedFeatures;
            
            lock (_lockObject)
            {
                if (!_featureValues.Any())
                {
                    _logger.LogWarning("[MODEL_HEALTH] No feature data available for parity check");
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
                    if (!recentValues.Any())
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

            if (failedFeatures.Any())
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
            _ = Task.Run(async () => await CheckModelHealthAsync()).ConfigureAwait(false);
        }

        private async Task CheckModelHealthAsync()
        {
            try
            {
                var health = GetCurrentHealth();
                
                if (!health.IsHealthy)
                {
                    var issues = string.Join("; ", health.HealthIssues);
                    await _alertService.SendModelHealthAlertAsync("Model Health", issues, health).ConfigureAwait(false);
                    
                    _logger.LogWarning("[MODEL_HEALTH] Health issues detected: {Issues}", issues);
                }
                else
                {
                    _logger.LogDebug("[MODEL_HEALTH] Model health check passed - Confidence: {Conf:F3}, Brier: {Brier:F3}",
                        health.AverageConfidence, health.AverageBrierScore);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL_HEALTH] Error during health check");
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

        private bool _disposed = false;

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
        public List<string> HealthIssues { get; set; } = new();
    }
}