using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Prometheus;

namespace UnifiedOrchestrator.Services
{
    /// <summary>
    /// Comprehensive ML/RL metrics instrumentation with Prometheus integration
    /// Tracks prediction accuracy, feature drift, inference latency, episode rewards, exploration rates, and policy norms
    /// </summary>
    public class MLRLMetricsService : BackgroundService
    {
        private readonly ILogger<MLRLMetricsService> _logger;
        private readonly MLRLMetricsOptions _options;
        private readonly ConcurrentDictionary<string, ModelMetrics> _modelMetrics = new();
        
        // Prometheus metrics
        private readonly Counter _predictionsTotal;
        private readonly Histogram _inferenceLatency;
        private readonly Gauge _predictionAccuracy;
        private readonly Gauge _featureDrift;
        private readonly Gauge _episodeReward;
        private readonly Gauge _explorationRate;
        private readonly Gauge _policyNorm;
        private readonly Gauge _memoryUsage;
        private readonly Counter _anomaliesDetected;
        private readonly Histogram _batchSize;
        private readonly Gauge _modelHealth;

        public MLRLMetricsService(
            ILogger<MLRLMetricsService> logger,
            IOptions<MLRLMetricsOptions> options)
        {
            _logger = logger;
            _options = options.Value;

            // Initialize Prometheus metrics
            _predictionsTotal = Metrics.CreateCounter(
                "ml_predictions_total", 
                "Total number of ML predictions made",
                new[] { "model_name", "prediction_type" });

            _inferenceLatency = Metrics.CreateHistogram(
                "ml_inference_latency_seconds",
                "ML inference latency in seconds",
                new[] { "model_name" },
                new HistogramConfiguration
                {
                    Buckets = new[] { 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0 }
                });

            _predictionAccuracy = Metrics.CreateGauge(
                "ml_prediction_accuracy",
                "ML prediction accuracy (0-1)",
                new[] { "model_name", "time_window" });

            _featureDrift = Metrics.CreateGauge(
                "ml_feature_drift",
                "Feature drift score",
                new[] { "feature_set", "drift_type" });

            _episodeReward = Metrics.CreateGauge(
                "rl_episode_reward",
                "Reinforcement learning episode reward",
                new[] { "agent_name", "episode_type" });

            _explorationRate = Metrics.CreateGauge(
                "rl_exploration_rate",
                "RL exploration rate (epsilon)",
                new[] { "agent_name" });

            _policyNorm = Metrics.CreateGauge(
                "rl_policy_norm",
                "RL policy network parameter norm",
                new[] { "agent_name", "layer" });

            _memoryUsage = Metrics.CreateGauge(
                "ml_memory_usage_bytes",
                "ML service memory usage in bytes",
                new[] { "component" });

            _anomaliesDetected = Metrics.CreateCounter(
                "ml_anomalies_detected_total",
                "Total number of anomalies detected",
                new[] { "anomaly_type", "severity" });

            _batchSize = Metrics.CreateHistogram(
                "ml_batch_size",
                "ML inference batch sizes",
                new[] { "model_name" },
                new HistogramConfiguration
                {
                    Buckets = new double[] { 1, 2, 4, 8, 16, 32, 64, 128 }
                });

            _modelHealth = Metrics.CreateGauge(
                "ml_model_health",
                "ML model health status (0=unhealthy, 1=healthy)",
                new[] { "model_name", "health_check" });

            _logger.LogInformation("📊 ML/RL Metrics Service initialized with Prometheus integration");
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🚀 ML/RL Metrics Service started");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await UpdateAggregatedMetricsAsync(stoppingToken);
                    await Task.Delay(TimeSpan.FromSeconds(_options.UpdateIntervalSeconds), stoppingToken);
                }
                catch (OperationCanceledException)
                {
                    // Expected when cancelling
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ Error updating aggregated metrics");
                    await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
                }
            }

            _logger.LogInformation("🛑 ML/RL Metrics Service stopped");
        }

        /// <summary>
        /// Record prediction made by a model
        /// </summary>
        public void RecordPrediction(string modelName, string predictionType, double latencySeconds, double confidence = 0.0)
        {
            try
            {
                _predictionsTotal.WithLabels(modelName, predictionType).Inc();
                _inferenceLatency.WithLabels(modelName).Observe(latencySeconds);

                var metrics = _modelMetrics.GetOrAdd(modelName, _ => new ModelMetrics(modelName));
                metrics.AddPrediction(latencySeconds, confidence);

                _logger.LogTrace("📈 Recorded prediction: {ModelName}, Type: {Type}, Latency: {Latency}ms", 
                    modelName, predictionType, latencySeconds * 1000);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record prediction metric");
            }
        }

        /// <summary>
        /// Record prediction accuracy
        /// </summary>
        public void RecordAccuracy(string modelName, string timeWindow, double accuracy)
        {
            try
            {
                _predictionAccuracy.WithLabels(modelName, timeWindow).Set(accuracy);

                var metrics = _modelMetrics.GetOrAdd(modelName, _ => new ModelMetrics(modelName));
                metrics.AddAccuracy(timeWindow, accuracy);

                _logger.LogDebug("🎯 Recorded accuracy: {ModelName}, Window: {Window}, Accuracy: {Accuracy:F3}", 
                    modelName, timeWindow, accuracy);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record accuracy metric");
            }
        }

        /// <summary>
        /// Record feature drift
        /// </summary>
        public void RecordFeatureDrift(string featureSet, string driftType, double driftScore)
        {
            try
            {
                _featureDrift.WithLabels(featureSet, driftType).Set(driftScore);
                
                if (driftScore > _options.DriftAlertThreshold)
                {
                    _anomaliesDetected.WithLabels("feature_drift", "high").Inc();
                    _logger.LogWarning("🚨 High feature drift detected: {FeatureSet}, Score: {Score:F4}", 
                        featureSet, driftScore);
                }

                _logger.LogTrace("📊 Recorded feature drift: {FeatureSet}, Type: {Type}, Score: {Score:F4}", 
                    featureSet, driftType, driftScore);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record feature drift metric");
            }
        }

        /// <summary>
        /// Record RL episode reward
        /// </summary>
        public void RecordEpisodeReward(string agentName, string episodeType, double reward)
        {
            try
            {
                _episodeReward.WithLabels(agentName, episodeType).Set(reward);

                var metrics = _modelMetrics.GetOrAdd(agentName, _ => new ModelMetrics(agentName));
                metrics.AddEpisodeReward(reward);

                _logger.LogDebug("🏆 Recorded episode reward: {AgentName}, Type: {Type}, Reward: {Reward:F2}", 
                    agentName, episodeType, reward);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record episode reward metric");
            }
        }

        /// <summary>
        /// Record RL exploration rate
        /// </summary>
        public void RecordExplorationRate(string agentName, double explorationRate)
        {
            try
            {
                _explorationRate.WithLabels(agentName).Set(explorationRate);

                _logger.LogTrace("🔍 Recorded exploration rate: {AgentName}, Rate: {Rate:F3}", 
                    agentName, explorationRate);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record exploration rate metric");
            }
        }

        /// <summary>
        /// Record RL policy norm
        /// </summary>
        public void RecordPolicyNorm(string agentName, string layer, double norm)
        {
            try
            {
                _policyNorm.WithLabels(agentName, layer).Set(norm);

                _logger.LogTrace("📏 Recorded policy norm: {AgentName}, Layer: {Layer}, Norm: {Norm:F3}", 
                    agentName, layer, norm);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record policy norm metric");
            }
        }

        /// <summary>
        /// Record batch size
        /// </summary>
        public void RecordBatchSize(string modelName, int batchSize)
        {
            try
            {
                _batchSize.WithLabels(modelName).Observe(batchSize);

                _logger.LogTrace("📦 Recorded batch size: {ModelName}, Size: {Size}", modelName, batchSize);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record batch size metric");
            }
        }

        /// <summary>
        /// Record memory usage
        /// </summary>
        public void RecordMemoryUsage(string component, long memoryBytes)
        {
            try
            {
                _memoryUsage.WithLabels(component).Set(memoryBytes);

                if (memoryBytes > _options.MemoryAlertThresholdMB * 1024 * 1024)
                {
                    _anomaliesDetected.WithLabels("high_memory", "warning").Inc();
                    _logger.LogWarning("⚠️ High memory usage: {Component}, Usage: {UsageMB}MB", 
                        component, memoryBytes / (1024 * 1024));
                }

                _logger.LogTrace("💾 Recorded memory usage: {Component}, Usage: {UsageMB}MB", 
                    component, memoryBytes / (1024 * 1024));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record memory usage metric");
            }
        }

        /// <summary>
        /// Record model health status
        /// </summary>
        public void RecordModelHealth(string modelName, string healthCheck, bool isHealthy)
        {
            try
            {
                _modelHealth.WithLabels(modelName, healthCheck).Set(isHealthy ? 1 : 0);

                if (!isHealthy)
                {
                    _anomaliesDetected.WithLabels("model_unhealthy", "critical").Inc();
                    _logger.LogError("❌ Model health check failed: {ModelName}, Check: {HealthCheck}", 
                        modelName, healthCheck);
                }

                _logger.LogTrace("🏥 Recorded model health: {ModelName}, Check: {Check}, Healthy: {Healthy}", 
                    modelName, healthCheck, isHealthy);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to record model health metric");
            }
        }

        /// <summary>
        /// Get comprehensive metrics summary
        /// </summary>
        public MLRLMetricsSummary GetMetricsSummary()
        {
            try
            {
                var summary = new MLRLMetricsSummary
                {
                    Timestamp = DateTime.UtcNow,
                    ModelMetrics = _modelMetrics.Values.Select(m => m.GetSummary()).ToList(),
                    SystemMetrics = GetSystemMetrics()
                };

                return summary;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to get metrics summary");
                return new MLRLMetricsSummary { Timestamp = DateTime.UtcNow };
            }
        }

        /// <summary>
        /// Check if any critical alerts should be triggered
        /// </summary>
        public List<MLRLAlert> CheckAlerts()
        {
            var alerts = new List<MLRLAlert>();

            try
            {
                foreach (var modelMetrics in _modelMetrics.Values)
                {
                    var summary = modelMetrics.GetSummary();

                    // Check accuracy degradation
                    if (summary.CurrentAccuracy < _options.AccuracyAlertThreshold)
                    {
                        alerts.Add(new MLRLAlert
                        {
                            Type = "accuracy_degradation",
                            Severity = "critical",
                            ModelName = summary.ModelName,
                            Message = $"Model accuracy dropped to {summary.CurrentAccuracy:F3}",
                            Timestamp = DateTime.UtcNow
                        });
                    }

                    // Check latency spikes
                    if (summary.AverageLatencyMs > _options.LatencyAlertThresholdMs)
                    {
                        alerts.Add(new MLRLAlert
                        {
                            Type = "high_latency",
                            Severity = "warning",
                            ModelName = summary.ModelName,
                            Message = $"High inference latency: {summary.AverageLatencyMs:F2}ms",
                            Timestamp = DateTime.UtcNow
                        });
                    }

                    // Check low exploration in RL
                    if (summary.ExplorationRate < _options.MinExplorationRate && summary.ExplorationRate > 0)
                    {
                        alerts.Add(new MLRLAlert
                        {
                            Type = "low_exploration",
                            Severity = "warning",
                            ModelName = summary.ModelName,
                            Message = $"Low exploration rate: {summary.ExplorationRate:F3}",
                            Timestamp = DateTime.UtcNow
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to check alerts");
            }

            return alerts;
        }

        private async Task UpdateAggregatedMetricsAsync(CancellationToken cancellationToken)
        {
            try
            {
                // Update system memory usage
                var totalMemory = GC.GetTotalMemory(false);
                RecordMemoryUsage("system_total", totalMemory);

                // Update model health checks
                foreach (var modelMetrics in _modelMetrics.Values)
                {
                    var summary = modelMetrics.GetSummary();
                    
                    // Health based on recent performance
                    var isHealthy = summary.CurrentAccuracy > _options.AccuracyAlertThreshold &&
                                   summary.AverageLatencyMs < _options.LatencyAlertThresholdMs;
                    
                    RecordModelHealth(summary.ModelName, "performance", isHealthy);
                }

                // Check and log alerts
                var alerts = CheckAlerts();
                if (alerts.Any())
                {
                    _logger.LogWarning("🚨 Active ML/RL alerts: {AlertCount}", alerts.Count);
                    foreach (var alert in alerts.Take(5)) // Log first 5 alerts
                    {
                        _logger.LogWarning("Alert: {Type} - {Message}", alert.Type, alert.Message);
                    }
                }

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to update aggregated metrics");
            }
        }

        private SystemMetrics GetSystemMetrics()
        {
            return new SystemMetrics
            {
                TotalMemoryMB = GC.GetTotalMemory(false) / (1024 * 1024),
                ActiveModels = _modelMetrics.Count,
                TotalPredictions = _modelMetrics.Values.Sum(m => m.GetSummary().TotalPredictions),
                AverageLatencyMs = _modelMetrics.Values
                    .Where(m => m.GetSummary().AverageLatencyMs > 0)
                    .DefaultIfEmpty()
                    .Average(m => m?.GetSummary().AverageLatencyMs ?? 0)
            };
        }

        public override void Dispose()
        {
            _logger.LogInformation("🛑 ML/RL Metrics Service disposing");
            base.Dispose();
        }
    }

    /// <summary>
    /// Configuration options for ML/RL metrics service
    /// </summary>
    public class MLRLMetricsOptions
    {
        public int UpdateIntervalSeconds { get; set; } = 30;
        public double DriftAlertThreshold { get; set; } = 0.05;
        public double AccuracyAlertThreshold { get; set; } = 0.6;
        public double LatencyAlertThresholdMs { get; set; } = 500;
        public double MinExplorationRate { get; set; } = 0.01;
        public long MemoryAlertThresholdMB { get; set; } = 1024;
    }

    /// <summary>
    /// Individual model metrics tracker
    /// </summary>
    public class ModelMetrics
    {
        private readonly string _modelName;
        private readonly object _lock = new();
        private readonly Queue<double> _recentLatencies = new();
        private readonly Queue<double> _recentConfidences = new();
        private readonly Queue<double> _recentRewards = new();
        private readonly Dictionary<string, Queue<double>> _accuracyByWindow = new();
        private long _totalPredictions = 0;
        private double _explorationRate = 0.1;

        public ModelMetrics(string modelName)
        {
            _modelName = modelName;
        }

        public void AddPrediction(double latencySeconds, double confidence)
        {
            lock (_lock)
            {
                _totalPredictions++;
                
                _recentLatencies.Enqueue(latencySeconds * 1000); // Convert to ms
                if (_recentLatencies.Count > 1000)
                    _recentLatencies.Dequeue();

                if (confidence > 0)
                {
                    _recentConfidences.Enqueue(confidence);
                    if (_recentConfidences.Count > 1000)
                        _recentConfidences.Dequeue();
                }
            }
        }

        public void AddAccuracy(string timeWindow, double accuracy)
        {
            lock (_lock)
            {
                if (!_accuracyByWindow.TryGetValue(timeWindow, out var accuracies))
                {
                    accuracies = new Queue<double>();
                    _accuracyByWindow[timeWindow] = accuracies;
                }

                accuracies.Enqueue(accuracy);
                if (accuracies.Count > 100)
                    accuracies.Dequeue();
            }
        }

        public void AddEpisodeReward(double reward)
        {
            lock (_lock)
            {
                _recentRewards.Enqueue(reward);
                if (_recentRewards.Count > 1000)
                    _recentRewards.Dequeue();
            }
        }

        public void SetExplorationRate(double rate)
        {
            lock (_lock)
            {
                _explorationRate = rate;
            }
        }

        public ModelMetricsSummary GetSummary()
        {
            lock (_lock)
            {
                return new ModelMetricsSummary
                {
                    ModelName = _modelName,
                    TotalPredictions = _totalPredictions,
                    AverageLatencyMs = _recentLatencies.Any() ? _recentLatencies.Average() : 0,
                    CurrentAccuracy = GetLatestAccuracy(),
                    AverageConfidence = _recentConfidences.Any() ? _recentConfidences.Average() : 0,
                    AverageReward = _recentRewards.Any() ? _recentRewards.Average() : 0,
                    ExplorationRate = _explorationRate,
                    LastUpdated = DateTime.UtcNow
                };
            }
        }

        private double GetLatestAccuracy()
        {
            var allAccuracies = _accuracyByWindow.Values.SelectMany(q => q).ToList();
            return allAccuracies.Any() ? allAccuracies.TakeLast(10).Average() : 0;
        }
    }

    /// <summary>
    /// ML/RL metrics summary
    /// </summary>
    public class MLRLMetricsSummary
    {
        public DateTime Timestamp { get; set; }
        public List<ModelMetricsSummary> ModelMetrics { get; set; } = new();
        public SystemMetrics SystemMetrics { get; set; } = new();
    }

    /// <summary>
    /// Individual model metrics summary
    /// </summary>
    public class ModelMetricsSummary
    {
        public string ModelName { get; set; } = string.Empty;
        public long TotalPredictions { get; set; }
        public double AverageLatencyMs { get; set; }
        public double CurrentAccuracy { get; set; }
        public double AverageConfidence { get; set; }
        public double AverageReward { get; set; }
        public double ExplorationRate { get; set; }
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// System-wide metrics
    /// </summary>
    public class SystemMetrics
    {
        public long TotalMemoryMB { get; set; }
        public int ActiveModels { get; set; }
        public long TotalPredictions { get; set; }
        public double AverageLatencyMs { get; set; }
    }

    /// <summary>
    /// ML/RL alert
    /// </summary>
    public class MLRLAlert
    {
        public string Type { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
        public string ModelName { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
    }
}