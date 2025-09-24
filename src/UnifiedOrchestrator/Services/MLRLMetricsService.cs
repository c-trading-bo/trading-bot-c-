using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    internal class MLRLMetricsOptions
    {
        public int UpdateIntervalSeconds { get; set; } = 60;
        public double DriftAlertThreshold { get; set; } = 0.05;
        public double AccuracyAlertThreshold { get; set; } = 0.7;
        public double LatencyAlertThresholdMs { get; set; } = 500;
        public int MaxMetricHistory { get; set; } = 1000;
    }

    internal class ModelMetrics
    {
        public string ModelName { get; set; } = "";
        public double AverageLatency { get; set; }
        public double Accuracy { get; set; }
        public double F1Score { get; set; }
        public int PredictionCount { get; set; }
        public DateTime LastUpdated { get; set; }
    }

    internal class MetricsSummary
    {
        public DateTime GeneratedAt { get; set; }
        public List<ModelMetrics> ModelMetrics { get; } = new();
        public Dictionary<string, double> FeatureDriftScores { get; } = new();
        public Dictionary<string, double> AgentPerformance { get; } = new();
    }

    internal class MetricAlert
    {
        public string Type { get; set; } = "";
        public string Message { get; set; } = "";
        public string Component { get; set; } = "";
        public double Value { get; set; }
        public double Threshold { get; set; }
        public DateTime Timestamp { get; set; }
    }

    internal class MLRLMetricsService : IDisposable
    {
        private readonly ILogger<MLRLMetricsService> _logger;
        private readonly MLRLMetricsOptions _options;
        
        private readonly ConcurrentDictionary<string, List<double>> _modelLatencies = new();
        private readonly ConcurrentDictionary<string, List<double>> _modelAccuracies = new();
        private readonly ConcurrentDictionary<string, int> _predictionCounts = new();
        private readonly ConcurrentDictionary<string, double> _featureDriftScores = new();
        private readonly ConcurrentDictionary<string, List<double>> _episodeRewards = new();
        private readonly ConcurrentDictionary<string, double> _explorationRates = new();
        private readonly ConcurrentDictionary<string, Dictionary<string, double>> _policyNorms = new();
        private readonly ConcurrentDictionary<string, long> _memoryUsage = new();
        private readonly ConcurrentDictionary<string, bool> _modelHealth = new();

        public MLRLMetricsService(ILogger<MLRLMetricsService> logger, IOptions<MLRLMetricsOptions> options)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        }

        public void RecordPrediction(string modelName, string predictionType, double latencySeconds, double confidence)
        {
            try
            {
                var latencyMs = latencySeconds * 1000;

                var latencies = _modelLatencies.GetOrAdd(modelName, _ => new List<double>());
                lock (latencies)
                {
                    latencies.Add(latencyMs);
                    if (latencies.Count > _options.MaxMetricHistory)
                    {
                        latencies.RemoveAt(0);
                    }
                }

                _predictionCounts.AddOrUpdate(modelName, 1, (k, v) => v + 1);

                _logger.LogDebug("Recorded prediction for {ModelName}: {LatencyMs:F2}ms, confidence: {Confidence:F3}", 
                    modelName, latencyMs, confidence);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to record prediction metrics for {ModelName}", modelName);
            }
        }

        public void RecordAccuracy(string modelName, string timeframe, double accuracy)
        {
            try
            {
                var accuracies = _modelAccuracies.GetOrAdd(modelName, _ => new List<double>());
                lock (accuracies)
                {
                    accuracies.Add(accuracy);
                    if (accuracies.Count > _options.MaxMetricHistory)
                    {
                        accuracies.RemoveAt(0);
                    }
                }

                _logger.LogDebug("Recorded accuracy for {ModelName} ({Timeframe}): {Accuracy:F3}", 
                    modelName, timeframe, accuracy);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to record accuracy for {ModelName}", modelName);
            }
        }

        public void RecordFeatureDrift(string featureName, string driftType, double driftScore)
        {
            try
            {
                _featureDriftScores[featureName] = driftScore;

                _logger.LogDebug("Recorded feature drift for {FeatureName} ({DriftType}): {DriftScore:F4}", 
                    featureName, driftType, driftScore);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to record feature drift for {FeatureName}", featureName);
            }
        }

        public void RecordEpisodeReward(string agentName, string phase, double reward)
        {
            try
            {
                var rewards = _episodeRewards.GetOrAdd(agentName, _ => new List<double>());
                lock (rewards)
                {
                    rewards.Add(reward);
                    if (rewards.Count > _options.MaxMetricHistory)
                    {
                        rewards.RemoveAt(0);
                    }
                }

                _logger.LogDebug("Recorded episode reward for {AgentName} ({Phase}): {Reward:F2}", 
                    agentName, phase, reward);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to record episode reward for {AgentName}", agentName);
            }
        }

        public void RecordExplorationRate(string agentName, double explorationRate)
        {
            try
            {
                _explorationRates[agentName] = explorationRate;

                _logger.LogDebug("Recorded exploration rate for {AgentName}: {ExplorationRate:F3}", 
                    agentName, explorationRate);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to record exploration rate for {AgentName}", agentName);
            }
        }

        public void RecordPolicyNorm(string agentName, string layer, double norm)
        {
            try
            {
                var norms = _policyNorms.GetOrAdd(agentName, _ => new Dictionary<string, double>());
                lock (norms)
                {
                    norms[layer] = norm;
                }

                _logger.LogDebug("Recorded policy norm for {AgentName}.{Layer}: {Norm:F3}", 
                    agentName, layer, norm);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to record policy norm for {AgentName}.{Layer}", agentName, layer);
            }
        }

        public void RecordMemoryUsage(string component, long bytesUsed)
        {
            try
            {
                _memoryUsage[component] = bytesUsed;

                _logger.LogDebug("Recorded memory usage for {Component}: {MemoryMB:F1} MB", 
                    component, bytesUsed / (1024.0 * 1024.0));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to record memory usage for {Component}", component);
            }
        }

        public void RecordModelHealth(string modelName, string healthType, bool isHealthy)
        {
            try
            {
                _modelHealth[modelName] = isHealthy;

                _logger.LogDebug("Recorded model health for {ModelName} ({HealthType}): {IsHealthy}", 
                    modelName, healthType, isHealthy);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to record model health for {ModelName}", modelName);
            }
        }

        public MetricsSummary GetMetricsSummary()
        {
            try
            {
                var summary = new MetricsSummary
                {
                    GeneratedAt = DateTime.UtcNow
                };

                // Aggregate model metrics
                var allModels = _modelLatencies.Keys.Union(_modelAccuracies.Keys).Union(_predictionCounts.Keys).Distinct();
                
                foreach (var modelName in allModels)
                {
                    var metrics = new ModelMetrics
                    {
                        ModelName = modelName,
                        LastUpdated = DateTime.UtcNow
                    };

                    // Calculate average latency
                    if (_modelLatencies.TryGetValue(modelName, out var latencies) && latencies.Count > 0)
                    {
                        lock (latencies)
                        {
                            metrics.AverageLatency = latencies.Average();
                        }
                    }

                    // Calculate average accuracy
                    if (_modelAccuracies.TryGetValue(modelName, out var accuracies) && accuracies.Count > 0)
                    {
                        lock (accuracies)
                        {
                            metrics.Accuracy = accuracies.Average();
                        }
                    }

                    // Get prediction count
                    if (_predictionCounts.TryGetValue(modelName, out var count))
                    {
                        metrics.PredictionCount = count;
                    }

                    // Set F1Score (simplified - could be calculated from precision/recall if available)
                    metrics.F1Score = metrics.Accuracy * 0.9; // Approximate F1 from accuracy

                    summary.ModelMetrics.Add(metrics);
                }

                // Add feature drift scores
                foreach (var kvp in _featureDriftScores)
                {
                    summary.FeatureDriftScores[kvp.Key] = kvp.Value;
                }

                // Add agent performance (average episode rewards)
                foreach (var kvp in _episodeRewards)
                {
                    lock (kvp.Value)
                    {
                        if (kvp.Value.Count > 0)
                        {
                            summary.AgentPerformance[kvp.Key] = kvp.Value.Average();
                        }
                    }
                }

                return summary;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate metrics summary");
                return new MetricsSummary { GeneratedAt = DateTime.UtcNow };
            }
        }

        public List<MetricAlert> CheckAlerts()
        {
            var alerts = new List<MetricAlert>();

            try
            {
                var now = DateTime.UtcNow;

                // Check accuracy alerts
                foreach (var kvp in _modelAccuracies)
                {
                    lock (kvp.Value)
                    {
                        if (kvp.Value.Count > 0)
                        {
                            var recentAccuracy = kvp.Value.Average();
                            if (recentAccuracy < _options.AccuracyAlertThreshold)
                            {
                                alerts.Add(new MetricAlert
                                {
                                    Type = "accuracy_degradation",
                                    Message = $"Model {kvp.Key} accuracy {recentAccuracy:F3} below threshold {_options.AccuracyAlertThreshold:F3}",
                                    Component = kvp.Key,
                                    Value = recentAccuracy,
                                    Threshold = _options.AccuracyAlertThreshold,
                                    Timestamp = now
                                });
                            }
                        }
                    }
                }

                // Check latency alerts
                foreach (var kvp in _modelLatencies)
                {
                    lock (kvp.Value)
                    {
                        if (kvp.Value.Count > 0)
                        {
                            var recentLatency = kvp.Value.Average();
                            if (recentLatency > _options.LatencyAlertThresholdMs)
                            {
                                alerts.Add(new MetricAlert
                                {
                                    Type = "high_latency",
                                    Message = $"Model {kvp.Key} latency {recentLatency:F2}ms exceeds threshold {_options.LatencyAlertThresholdMs:F2}ms",
                                    Component = kvp.Key,
                                    Value = recentLatency,
                                    Threshold = _options.LatencyAlertThresholdMs,
                                    Timestamp = now
                                });
                            }
                        }
                    }
                }

                // Check feature drift alerts
                foreach (var kvp in _featureDriftScores)
                {
                    if (kvp.Value > _options.DriftAlertThreshold)
                    {
                        alerts.Add(new MetricAlert
                        {
                            Type = "feature_drift",
                            Message = $"Feature {kvp.Key} drift {kvp.Value:F4} exceeds threshold {_options.DriftAlertThreshold:F4}",
                            Component = kvp.Key,
                            Value = kvp.Value,
                            Threshold = _options.DriftAlertThreshold,
                            Timestamp = now
                        });
                    }
                }

                // Check model health alerts
                foreach (var kvp in _modelHealth)
                {
                    if (!kvp.Value)
                    {
                        alerts.Add(new MetricAlert
                        {
                            Type = "model_unhealthy",
                            Message = $"Model {kvp.Key} is in unhealthy state",
                            Component = kvp.Key,
                            Value = 0,
                            Threshold = 1,
                            Timestamp = now
                        });
                    }
                }

                _logger.LogDebug("Generated {AlertCount} alerts", alerts.Count);
                return alerts;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to check alerts");
                return alerts;
            }
        }

        public void Dispose()
        {
            // No resources to dispose currently
        }
    }
}