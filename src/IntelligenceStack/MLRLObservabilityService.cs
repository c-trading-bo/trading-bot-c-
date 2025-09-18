using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Full ML/RL observability system with Prometheus/Grafana export
/// Implements requirement: Full ML/RL observability: prediction accuracy, drift, latency, RL rewards, exploration rates, policy norms
/// </summary>
public class MlrlObservabilityService : IDisposable
{
    private readonly ILogger<MlrlObservabilityService> _logger;
    private readonly HttpClient _httpClient;
    private readonly Timer _exportTimer;
    
    // Metrics collection
    private readonly Meter _meter;
    private readonly Counter<int> _predictionCounter;
    private readonly Histogram<double> _predictionAccuracy;
    private readonly Histogram<double> _predictionLatency;
    private readonly Histogram<double> _driftScore;
    private readonly Histogram<double> _rlRewards;
    private readonly Gauge<double> _explorationRate;
    private readonly Histogram<double> _policyNorms;
    private readonly Counter<int> _modelInferences;
    private readonly Histogram<double> _ensembleVariance;

    // In-memory metrics storage for Prometheus export
    private readonly ConcurrentDictionary<string, MetricValue> _metricsStorage = new();
    private readonly object _lock = new();

    public MlrlObservabilityService(
        ILogger<MlrlObservabilityService> logger,
        HttpClient httpClient)
    {
        _logger = logger;
        _httpClient = httpClient;

        // Initialize metrics
        _meter = new Meter("TradingBot.ML.RL", "1.0.0");
        
        _predictionCounter = _meter.CreateCounter<int>("ml_predictions_total", "count", "Total number of ML predictions made");
        _predictionAccuracy = _meter.CreateHistogram<double>("ml_prediction_accuracy", "ratio", "Accuracy of ML predictions");
        _predictionLatency = _meter.CreateHistogram<double>("ml_prediction_latency_ms", "milliseconds", "Latency of ML predictions");
        _driftScore = _meter.CreateHistogram<double>("ml_drift_score", "score", "Model drift detection score");
        _rlRewards = _meter.CreateHistogram<double>("rl_rewards", "reward", "RL agent rewards");
        _explorationRate = _meter.CreateGauge<double>("rl_exploration_rate", "ratio", "RL exploration rate");
        _policyNorms = _meter.CreateHistogram<double>("rl_policy_norms", "norm", "RL policy gradient norms");
        _modelInferences = _meter.CreateCounter<int>("model_inferences_total", "count", "Total model inferences");
        _ensembleVariance = _meter.CreateHistogram<double>("ensemble_prediction_variance", "variance", "Variance in ensemble predictions");

        // Start export timer (every 15 seconds)
        _exportTimer = new Timer(ExportMetricsAsync, null, TimeSpan.FromSeconds(15), TimeSpan.FromSeconds(15));

        _logger.LogInformation("ML/RL Observability service initialized with Prometheus export");
    }

    #region Prediction Metrics

    public void RecordPrediction(string modelId, double prediction, double confidence, TimeSpan latency)
    {
        var tags = new TagList { { "model_id", modelId } };
        _predictionCounter.Add(1, tags);
        _predictionLatency.Record(latency.TotalMilliseconds, tags);
        _modelInferences.Add(1, tags);

        // Store for Prometheus export
        UpdateMetricValue($"ml_predictions_total{{model_id=\"{modelId}\"}}", 1, MetricType.Counter);
        UpdateMetricValue($"ml_prediction_latency_ms{{model_id=\"{modelId}\"}}", latency.TotalMilliseconds, MetricType.Histogram);
        
        _logger.LogDebug("Recorded prediction metric for {ModelId}: prediction={Prediction:F3}, confidence={Confidence:F3}, latency={Latency}ms",
            modelId, prediction, confidence, latency.TotalMilliseconds);
    }

    public void RecordPredictionAccuracy(string modelId, double actualValue, double predictedValue)
    {
        var accuracy = 1.0 - Math.Abs(actualValue - predictedValue) / Math.Max(Math.Abs(actualValue), 1.0);
        accuracy = Math.Max(0.0, Math.Min(1.0, accuracy)); // Clamp to [0,1]

        _predictionAccuracy.Record(accuracy, new TagList { { "model_id", modelId } });
        UpdateMetricValue($"ml_prediction_accuracy{{model_id=\"{modelId}\"}}", accuracy, MetricType.Histogram);

        _logger.LogDebug("Recorded prediction accuracy for {ModelId}: {Accuracy:F3}", modelId, accuracy);
    }

    public void RecordDriftScore(string modelId, double driftScore)
    {
        _driftScore.Record(driftScore, new TagList { { "model_id", modelId } });
        UpdateMetricValue($"ml_drift_score{{model_id=\"{modelId}\"}}", driftScore, MetricType.Histogram);

        if (driftScore > 0.7) // Configurable threshold
        {
            _logger.LogWarning("High drift score detected for {ModelId}: {DriftScore:F3}", modelId, driftScore);
        }
    }

    #endregion

    #region RL Metrics

    public void RecordRLReward(string agentId, double reward, int episode)
    {
        _rlRewards.Record(reward, new TagList { { "agent_id", agentId } });
        UpdateMetricValue($"rl_rewards{{agent_id=\"{agentId}\"}}", reward, MetricType.Histogram);

        _logger.LogDebug("Recorded RL reward for {AgentId}: episode={Episode}, reward={Reward:F3}", 
            agentId, episode, reward);
    }

    public void RecordExplorationRate(string agentId, double explorationRate)
    {
        _explorationRate.Record(explorationRate, new TagList { { "agent_id", agentId } });
        UpdateMetricValue($"rl_exploration_rate{{agent_id=\"{agentId}\"}}", explorationRate, MetricType.Gauge);
    }

    public void RecordPolicyNorm(string agentId, double policyNorm)
    {
        _policyNorms.Record(policyNorm, new TagList { { "agent_id", agentId } });
        UpdateMetricValue($"rl_policy_norms{{agent_id=\"{agentId}\"}}", policyNorm, MetricType.Histogram);

        if (policyNorm > 10.0) // Configurable threshold
        {
            _logger.LogWarning("High policy norm detected for {AgentId}: {PolicyNorm:F3}", agentId, policyNorm);
        }
    }

    #endregion

    #region Ensemble Metrics

    public void RecordEnsembleVariance(double variance, int modelCount)
    {
        _ensembleVariance.Record(variance, new TagList { { "model_count", modelCount.ToString() } });
        UpdateMetricValue($"ensemble_prediction_variance{{model_count=\"{modelCount}\"}}", variance, MetricType.Histogram);

        if (variance > 0.1) // Configurable threshold
        {
            _logger.LogWarning("High ensemble variance detected: {Variance:F3} with {ModelCount} models", 
                variance, modelCount);
        }
    }

    #endregion

    #region Prometheus Export

    private void UpdateMetricValue(string metricName, double value, MetricType type)
    {
        lock (_lock)
        {
            if (_metricsStorage.TryGetValue(metricName, out var existing))
            {
                switch (type)
                {
                    case MetricType.Counter:
                        existing.Value += value;
                        break;
                    case MetricType.Gauge:
                        existing.Value = value;
                        break;
                    case MetricType.Histogram:
                        existing.Samples.Add(value);
                        existing.Value = existing.Samples.Count > 0 ? existing.Samples.Average() : value;
                        // Keep only recent samples to prevent memory growth
                        if (existing.Samples.Count > 1000)
                        {
                            existing.Samples.RemoveRange(0, 500);
                        }
                        break;
                }
                existing.LastUpdated = DateTime.UtcNow;
            }
            else
            {
                _metricsStorage[metricName] = new MetricValue
                {
                    Name = metricName,
                    Value = value,
                    Type = type,
                    LastUpdated = DateTime.UtcNow,
                    Samples = type == MetricType.Histogram ? new List<double> { value } : new List<double>()
                };
            }
        }
    }

    private async void ExportMetricsAsync(object? state)
    {
        try
        {
            // Export to Prometheus gateway if configured
            var prometheusGateway = Environment.GetEnvironmentVariable("PROMETHEUS_GATEWAY_URL");
            if (!string.IsNullOrEmpty(prometheusGateway))
            {
                await ExportToPrometheusAsync(prometheusGateway).ConfigureAwait(false);
            }

            // Export to Grafana Cloud if configured
            var grafanaUrl = Environment.GetEnvironmentVariable("GRAFANA_CLOUD_URL");
            var grafanaApiKey = Environment.GetEnvironmentVariable("GRAFANA_API_KEY");
            if (!string.IsNullOrEmpty(grafanaUrl) && !string.IsNullOrEmpty(grafanaApiKey))
            {
                await ExportToGrafanaAsync(grafanaUrl, grafanaApiKey).ConfigureAwait(false);
            }

            // Export to local file for development
            await ExportToFileAsync().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to export metrics");
        }
    }

    private async Task ExportToPrometheusAsync(string gatewayUrl)
    {
        try
        {
            var prometheusFormat = GeneratePrometheusFormat();
            var content = new StringContent(prometheusFormat, Encoding.UTF8, "text/plain");
            
            var response = await _httpClient.PostAsync($"{gatewayUrl}/metrics/job/trading_bot", content).ConfigureAwait(false).ConfigureAwait(false);
            
            if (response.IsSuccessStatusCode)
            {
                _logger.LogDebug("Successfully exported metrics to Prometheus gateway");
            }
            else
            {
                _logger.LogWarning("Failed to export metrics to Prometheus: {StatusCode}", response.StatusCode);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error exporting to Prometheus");
        }
    }

    private async Task ExportToGrafanaAsync(string grafanaUrl, string apiKey)
    {
        try
        {
            var metrics = GenerateGrafanaMetrics();
            var json = JsonSerializer.Serialize(metrics);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            
            _httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey);
            
            var response = await _httpClient.PostAsync($"{grafanaUrl}/api/push", content).ConfigureAwait(false).ConfigureAwait(false);
            
            if (response.IsSuccessStatusCode)
            {
                _logger.LogDebug("Successfully exported metrics to Grafana Cloud");
            }
            else
            {
                _logger.LogWarning("Failed to export metrics to Grafana: {StatusCode}", response.StatusCode);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error exporting to Grafana");
        }
    }

    private async Task ExportToFileAsync()
    {
        try
        {
            var metricsDir = "data/metrics";
            System.IO.Directory.CreateDirectory(metricsDir);
            
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            var filePath = System.IO.Path.Combine(metricsDir, $"ml_rl_metrics_{timestamp}.json");
            
            var allMetrics = new Dictionary<string, object>();
            lock (_lock)
            {
                foreach (var (name, metric) in _metricsStorage)
                {
                    allMetrics[name] = new
                    {
                        value = metric.Value,
                        type = metric.Type.ToString(),
                        last_updated = metric.LastUpdated,
                        sample_count = metric.Samples.Count,
                        samples = metric.Type == MetricType.Histogram ? metric.Samples.TakeLast(10).ToArray() : null
                    };
                }
            }
            
            var json = JsonSerializer.Serialize(allMetrics, new JsonSerializerOptions { WriteIndented = true });
            await System.IO.File.WriteAllTextAsync(filePath, json).ConfigureAwait(false);
            
            _logger.LogDebug("Exported {MetricCount} metrics to {FilePath}", allMetrics.Count, filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error exporting metrics to file");
        }
    }

    private string GeneratePrometheusFormat()
    {
        var sb = new StringBuilder();
        
        lock (_lock)
        {
            foreach (var (name, metric) in _metricsStorage)
            {
                sb.AppendLine($"# TYPE {metric.Name.Split('{')[0]} {metric.Type.ToString().ToLower()}");
                sb.AppendLine($"{name} {metric.Value}");
            }
        }
        
        return sb.ToString();
    }

    private object GenerateGrafanaMetrics()
    {
        var series = new List<object>();
        
        lock (_lock)
        {
            foreach (var (name, metric) in _metricsStorage)
            {
                series.Add(new
                {
                    name = name,
                    value = metric.Value,
                    timestamp = ((DateTimeOffset)metric.LastUpdated).ToUnixTimeMilliseconds(),
                    tags = new { job = "trading_bot", type = metric.Type.ToString() }
                });
            }
        }
        
        return new { series };
    }

    #endregion

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _exportTimer?.Dispose();
            _meter?.Dispose();
            _httpClient?.Dispose();
        }
    }
}

/// <summary>
/// Metric value storage
/// </summary>
internal class MetricValue
{
    public string Name { get; set; } = string.Empty;
    public double Value { get; set; }
    public MetricType Type { get; set; }
    public DateTime LastUpdated { get; set; }
    public List<double> Samples { get; set; } = new();
}

/// <summary>
/// Metric types for export
/// </summary>
internal enum MetricType
{
    Counter,
    Gauge,
    Histogram
}