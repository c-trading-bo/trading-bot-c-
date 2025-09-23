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
using System.Globalization;

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
    
    // LoggerMessage delegates for CA1848 performance compliance
    private static readonly Action<ILogger, string, double, Exception?> LogHighPolicyNormDetected =
        LoggerMessage.Define<string, double>(LogLevel.Warning, new EventId(3001, nameof(LogHighPolicyNormDetected)),
            "High policy norm detected for {AgentId}: {PolicyNorm:F3}");
    
    private static readonly Action<ILogger, double, int, Exception?> LogHighEnsembleVarianceDetected =
        LoggerMessage.Define<double, int>(LogLevel.Warning, new EventId(3002, nameof(LogHighEnsembleVarianceDetected)),
            "High ensemble variance detected: {Variance:F3} with {ModelCount} models");
    
    private static readonly Action<ILogger, Exception?> LogMetricsExportFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(3003, nameof(LogMetricsExportFailed)),
            "Failed to export metrics to Prometheus endpoint");
    
    private static readonly Action<ILogger, string, Exception?> LogMetricUpdateDebug =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(3004, nameof(LogMetricUpdateDebug)),
            "Updated metric: {MetricName}");
    
    private static readonly Action<ILogger, Exception?> LogFailedToResetMetrics =
        LoggerMessage.Define(LogLevel.Warning, new EventId(3005, nameof(LogFailedToResetMetrics)),
            "Failed to reset metrics storage");
    
    private static readonly Action<ILogger, string, Exception?> LogFailedToUpdateRLMetric =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(3006, nameof(LogFailedToUpdateRLMetric)),
            "Failed to update RL metric: {MetricName}");
    
    private static readonly Action<ILogger, Exception?> LogFailedToRecordReward =
        LoggerMessage.Define(LogLevel.Warning, new EventId(3007, nameof(LogFailedToRecordReward)),
            "Failed to record RL reward due to validation error");
    
    private static readonly Action<ILogger, Exception?> LogFailedToExportMetrics =
        LoggerMessage.Define(LogLevel.Error, new EventId(3008, nameof(LogFailedToExportMetrics)),
            "Failed to export metrics");
    
    private static readonly Action<ILogger, Exception?> LogSuccessfulPrometheusExport =
        LoggerMessage.Define(LogLevel.Debug, new EventId(3009, nameof(LogSuccessfulPrometheusExport)),
            "Successfully exported metrics to Prometheus gateway");
    
    private static readonly Action<ILogger, int, Exception?> LogFailedPrometheusExport =
        LoggerMessage.Define<int>(LogLevel.Warning, new EventId(3010, nameof(LogFailedPrometheusExport)),
            "Failed to export metrics to Prometheus: {StatusCode}");
    
    private static readonly Action<ILogger, Exception?> LogPrometheusExportError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3011, nameof(LogPrometheusExportError)),
            "Error exporting to Prometheus");
    
    private static readonly Action<ILogger, Exception?> LogSuccessfulGrafanaExport =
        LoggerMessage.Define(LogLevel.Debug, new EventId(3012, nameof(LogSuccessfulGrafanaExport)),
            "Successfully exported metrics to Grafana Cloud");
    
    private static readonly Action<ILogger, int, Exception?> LogFailedGrafanaExport =
        LoggerMessage.Define<int>(LogLevel.Warning, new EventId(3013, nameof(LogFailedGrafanaExport)),
            "Failed to export metrics to Grafana: {StatusCode}");
    
    private static readonly Action<ILogger, Exception?> LogGrafanaExportError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3014, nameof(LogGrafanaExportError)),
            "Error exporting to Grafana");
    
    private static readonly Action<ILogger, int, string, Exception?> LogFileExportSuccess =
        LoggerMessage.Define<int, string>(LogLevel.Debug, new EventId(3015, nameof(LogFileExportSuccess)),
            "Exported {MetricCount} metrics to {FilePath}");
    
    private static readonly Action<ILogger, Exception?> LogFileExportError =
        LoggerMessage.Define(LogLevel.Error, new EventId(3016, nameof(LogFileExportError)),
            "Error exporting metrics to file");
    
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

    // S109 Magic Number Constants for Observability
    private const int HistogramSampleLimit = 10;

    // In-memory metrics storage for Prometheus export
    private readonly ConcurrentDictionary<string, MetricValue> _metricsStorage = new();
    private readonly object _lock = new();

    // S109 Magic Number Constants - ML/RL Thresholds and Configuration
    private const double HighDriftScoreThreshold = 0.7;
    private const double HighPolicyNormThreshold = 10.0;
    private const double HighEnsembleVarianceThreshold = 0.1;
    private const int MaxHistogramSamples = 1000;
    private const int HistogramPruneSize = 500;

    // LoggerMessage delegates for CA1848 compliance - MLRLObservabilityService
    private static readonly Action<ILogger, Exception?> ObservabilityServiceInitialized =
        LoggerMessage.Define(LogLevel.Information, new EventId(5001, "ObservabilityServiceInitialized"), 
            "ML/RL Observability service initialized with Prometheus export");

    private static readonly Action<ILogger, string, double, double, int, Exception?> RecordedPredictionMetric =
        LoggerMessage.Define<string, double, double, int>(LogLevel.Debug, new EventId(5002, "RecordedPredictionMetric"), 
            "Recorded prediction metric for {ModelId}: prediction={Prediction:F3}, confidence={Confidence:F3}, latency={Latency}ms");

    private static readonly Action<ILogger, string, double, Exception?> RecordedPredictionAccuracy =
        LoggerMessage.Define<string, double>(LogLevel.Debug, new EventId(5003, "RecordedPredictionAccuracy"), 
            "Recorded prediction accuracy for {ModelId}: {Accuracy:F3}");

    private static readonly Action<ILogger, string, double, Exception?> HighDriftScoreDetected =
        LoggerMessage.Define<string, double>(LogLevel.Warning, new EventId(5004, "HighDriftScoreDetected"), 
            "High drift score detected for {ModelId}: {DriftScore:F3}");

    private static readonly Action<ILogger, string, int, double, Exception?> RecordedRLReward =
        LoggerMessage.Define<string, int, double>(LogLevel.Debug, new EventId(5005, "RecordedRLReward"), 
            "Recorded RL reward for {AgentId}: episode={Episode}, reward={Reward:F3}");










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

        ObservabilityServiceInitialized(_logger, null);
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
        
        RecordedPredictionMetric(_logger, modelId, prediction, confidence, (int)latency.TotalMilliseconds, null);
    }

    public void RecordPredictionAccuracy(string modelId, double actualValue, double predictedValue)
    {
        var accuracy = 1.0 - Math.Abs(actualValue - predictedValue) / Math.Max(Math.Abs(actualValue), 1.0);
        accuracy = Math.Max(0.0, Math.Min(1.0, accuracy)); // Clamp to [0,1]

        _predictionAccuracy.Record(accuracy, new TagList { { "model_id", modelId } });
        UpdateMetricValue($"ml_prediction_accuracy{{model_id=\"{modelId}\"}}", accuracy, MetricType.Histogram);

        RecordedPredictionAccuracy(_logger, modelId, accuracy, null);
    }

    public void RecordDriftScore(string modelId, double driftScore)
    {
        _driftScore.Record(driftScore, new TagList { { "model_id", modelId } });
        UpdateMetricValue($"ml_drift_score{{model_id=\"{modelId}\"}}", driftScore, MetricType.Histogram);

        if (driftScore > HighDriftScoreThreshold) // Configurable threshold
        {
            HighDriftScoreDetected(_logger, modelId, driftScore, null);
        }
    }

    #endregion

    #region RL Metrics

    public void RecordRLReward(string agentId, double reward, int episode)
    {
        _rlRewards.Record(reward, new TagList { { "agent_id", agentId } });
        UpdateMetricValue($"rl_rewards{{agent_id=\"{agentId}\"}}", reward, MetricType.Histogram);

        RecordedRLReward(_logger, agentId, episode, reward, null);
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

        if (policyNorm > HighPolicyNormThreshold) // Configurable threshold
        {
            LogHighPolicyNormDetected(_logger, agentId, policyNorm, null);
        }
    }

    #endregion

    #region Ensemble Metrics

    public void RecordEnsembleVariance(double variance, int modelCount)
    {
        _ensembleVariance.Record(variance, new TagList { { "model_count", modelCount.ToString(CultureInfo.InvariantCulture) } });
        UpdateMetricValue($"ensemble_prediction_variance{{model_count=\"{modelCount}\"}}", variance, MetricType.Histogram);

        if (variance > HighEnsembleVarianceThreshold) // Configurable threshold
        {
            LogHighEnsembleVarianceDetected(_logger, variance, modelCount, null);
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
                        if (existing.Samples.Count > MaxHistogramSamples)
                        {
                            existing.Samples.RemoveRange(0, HistogramPruneSize);
                        }
                        break;
                }
                existing.LastUpdated = DateTime.UtcNow;
            }
            else
            {
                var metricValue = new MetricValue
                {
                    Name = metricName,
                    Value = value,
                    Type = type,
                    LastUpdated = DateTime.UtcNow
                };
                
                if (type == MetricType.Histogram)
                {
                    metricValue.Samples.Add(value);
                }
                
                _metricsStorage[metricName] = metricValue;
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
        catch (HttpRequestException ex)
        {
            LogFailedToExportMetrics(_logger, ex);
        }
        catch (TimeoutException ex)
        {
            LogFailedToExportMetrics(_logger, ex);
        }
        catch (IOException ex)
        {
            LogFailedToExportMetrics(_logger, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            LogFailedToExportMetrics(_logger, ex);
        }
        catch (InvalidOperationException ex)
        {
            LogFailedToExportMetrics(_logger, ex);
        }
    }

    private async Task ExportToPrometheusAsync(string gatewayUrl)
    {
        try
        {
            var prometheusFormat = GeneratePrometheusFormat();
            var content = new StringContent(prometheusFormat, Encoding.UTF8, "text/plain");
            
            var response = await _httpClient.PostAsync($"{gatewayUrl}/metrics/job/trading_bot", content).ConfigureAwait(false);
            
            if (response.IsSuccessStatusCode)
            {
                LogSuccessfulPrometheusExport(_logger, null);
            }
            else
            {
                LogFailedPrometheusExport(_logger, (int)response.StatusCode, null);
            }
        }
        catch (HttpRequestException ex)
        {
            LogPrometheusExportError(_logger, ex);
        }
        catch (TimeoutException ex)
        {
            LogPrometheusExportError(_logger, ex);
        }
        catch (TaskCanceledException ex)
        {
            LogPrometheusExportError(_logger, ex);
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
            
            var response = await _httpClient.PostAsync($"{grafanaUrl}/api/push", content).ConfigureAwait(false);
            
            if (response.IsSuccessStatusCode)
            {
                LogSuccessfulGrafanaExport(_logger, null);
            }
            else
            {
                LogFailedGrafanaExport(_logger, (int)response.StatusCode, null);
            }
        }
        catch (HttpRequestException ex)
        {
            LogGrafanaExportError(_logger, ex);
        }
        catch (TimeoutException ex)
        {
            LogGrafanaExportError(_logger, ex);
        }
        catch (TaskCanceledException ex)
        {
            LogGrafanaExportError(_logger, ex);
        }
    }

    private async Task ExportToFileAsync()
    {
        try
        {
            var metricsDir = "data/metrics";
            System.IO.Directory.CreateDirectory(metricsDir);
            
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
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
                        samples = metric.Type == MetricType.Histogram ? metric.Samples.TakeLast(HistogramSampleLimit).ToArray() : null
                    };
                }
            }
            
            var json = JsonSerializer.Serialize(allMetrics, new JsonSerializerOptions { WriteIndented = true });
            await System.IO.File.WriteAllTextAsync(filePath, json).ConfigureAwait(false);
            
            LogFileExportSuccess(_logger, allMetrics.Count, filePath, null);
        }
        catch (IOException ex)
        {
            LogFileExportError(_logger, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            LogFileExportError(_logger, ex);
        }
        catch (JsonException ex)
        {
            LogFileExportError(_logger, ex);
        }
    }

    private string GeneratePrometheusFormat()
    {
        var sb = new StringBuilder();
        
        lock (_lock)
        {
            foreach (var (name, metric) in _metricsStorage)
            {
                sb.AppendLine(CultureInfo.InvariantCulture, $"# TYPE {metric.Name.Split('{')[0]} {metric.Type.ToString().ToLower(CultureInfo.InvariantCulture)}");
                sb.AppendLine(CultureInfo.InvariantCulture, $"{name} {metric.Value}");
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
internal sealed class MetricValue
{
    public string Name { get; set; } = string.Empty;
    public double Value { get; set; }
    public MetricType Type { get; set; }
    public DateTime LastUpdated { get; set; }
    public List<double> Samples { get; } = new();
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