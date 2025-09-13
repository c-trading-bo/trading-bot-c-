using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Diagnostics;
using System.Text.Json;

namespace BotCore.Services;

/// <summary>
/// Production-grade monitoring and health check service for ML/RL/Cloud trading system
/// Tracks model performance, API health, and system metrics
/// </summary>
public class ProductionMonitoringService : IHealthCheck
{
    private readonly ILogger<ProductionMonitoringService> _logger;
    private readonly ProductionTradingConfig _config;
    private readonly Dictionary<string, HealthMetric> _healthMetrics = new();
    private readonly Dictionary<string, PerformanceMetric> _performanceMetrics = new();
    private readonly object _metricsLock = new();

    public ProductionMonitoringService(ILogger<ProductionMonitoringService> logger, IOptions<ProductionTradingConfig> config)
    {
        _logger = logger;
        _config = config.Value;
        InitializeMetrics();
    }

    public async Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var healthChecks = new List<(string Name, bool IsHealthy, string Message)>();

            // Check model health
            var modelHealth = await CheckModelHealthAsync(cancellationToken);
            healthChecks.Add(("Models", modelHealth.IsHealthy, modelHealth.Message));

            // Check GitHub API connectivity
            var githubHealth = await CheckGitHubConnectivityAsync(cancellationToken);
            healthChecks.Add(("GitHub", githubHealth.IsHealthy, githubHealth.Message));

            // Check system resources
            var systemHealth = CheckSystemResourcesHealth();
            healthChecks.Add(("System", systemHealth.IsHealthy, systemHealth.Message));

            // Check trading performance
            var performanceHealth = CheckTradingPerformanceHealth();
            healthChecks.Add(("Performance", performanceHealth.IsHealthy, performanceHealth.Message));

            var overallHealthy = healthChecks.All(h => h.IsHealthy);
            var status = overallHealthy ? HealthStatus.Healthy : HealthStatus.Degraded;
            
            var data = healthChecks.ToDictionary(h => h.Name, h => (object)new { IsHealthy = h.IsHealthy, Message = h.Message });
            
            _logger.LogInformation("🏥 [HEALTH] System health check: {Status} - {HealthyCount}/{TotalCount} systems healthy", 
                status, healthChecks.Count(h => h.IsHealthy), healthChecks.Count);

            return new HealthCheckResult(status, description: "ML/RL Trading System Health", data: data);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HEALTH] Health check failed: {Error}", ex.Message);
            return HealthCheckResult.Unhealthy("Health check failed", ex);
        }
    }

    /// <summary>
    /// Track model prediction performance
    /// </summary>
    public void TrackModelPrediction(string modelName, double confidence, bool wasCorrect, TimeSpan predictionTime)
    {
        lock (_metricsLock)
        {
            var key = $"model_{modelName}";
            if (!_performanceMetrics.TryGetValue(key, out var metric))
            {
                metric = new PerformanceMetric(modelName);
                _performanceMetrics[key] = metric;
            }

            metric.TotalPredictions++;
            metric.TotalConfidence += confidence;
            metric.TotalPredictionTime += predictionTime;
            
            if (wasCorrect)
            {
                metric.CorrectPredictions++;
            }

            metric.LastUpdated = DateTime.UtcNow;

            // Log performance statistics periodically
            if (metric.TotalPredictions % 10 == 0)
            {
                var accuracy = (double)metric.CorrectPredictions / metric.TotalPredictions;
                var avgConfidence = metric.TotalConfidence / metric.TotalPredictions;
                var avgTime = metric.TotalPredictionTime.TotalMilliseconds / metric.TotalPredictions;

                _logger.LogInformation("📊 [MONITORING] Model {ModelName} performance: {Accuracy:P1} accuracy, {AvgConfidence:F2} avg confidence, {AvgTime:F1}ms avg time",
                    modelName, accuracy, avgConfidence, avgTime);
            }
        }
    }

    /// <summary>
    /// Track API operation metrics
    /// </summary>
    public void TrackApiOperation(string operationName, TimeSpan duration, bool successful, string? errorMessage = null)
    {
        lock (_metricsLock)
        {
            var key = $"api_{operationName}";
            if (!_healthMetrics.TryGetValue(key, out var metric))
            {
                metric = new HealthMetric(operationName);
                _healthMetrics[key] = metric;
            }

            metric.TotalCalls++;
            metric.TotalDuration += duration;
            
            if (successful)
            {
                metric.SuccessfulCalls++;
            }
            else
            {
                metric.FailedCalls++;
                if (!string.IsNullOrEmpty(errorMessage))
                {
                    metric.LastError = errorMessage;
                }
            }

            metric.LastUpdated = DateTime.UtcNow;

            // Alert on high failure rate
            var failureRate = (double)metric.FailedCalls / metric.TotalCalls;
            if (failureRate > 0.1 && metric.TotalCalls >= 10)
            {
                _logger.LogWarning("⚠️ [MONITORING] High failure rate for {Operation}: {FailureRate:P1} ({Failed}/{Total})",
                    operationName, failureRate, metric.FailedCalls, metric.TotalCalls);
            }
        }
    }

    /// <summary>
    /// Track trading decision metrics
    /// </summary>
    public void TrackTradingDecision(string strategy, decimal confidence, decimal positionSize, bool enhanced)
    {
        lock (_metricsLock)
        {
            var key = $"trading_{strategy}";
            if (!_performanceMetrics.TryGetValue(key, out var metric))
            {
                metric = new PerformanceMetric(strategy);
                _performanceMetrics[key] = metric;
            }

            metric.TotalPredictions++;
            metric.TotalConfidence += (double)confidence;
            metric.LastUpdated = DateTime.UtcNow;

            if (enhanced)
            {
                metric.CorrectPredictions++; // Use as "enhanced decisions" counter
            }

            _logger.LogDebug("📈 [MONITORING] Trading decision: {Strategy} confidence={Confidence:P1}, size={Size}, enhanced={Enhanced}",
                strategy, confidence, positionSize, enhanced);
        }
    }

    /// <summary>
    /// Get comprehensive system metrics
    /// </summary>
    public SystemMetrics GetSystemMetrics()
    {
        lock (_metricsLock)
        {
            var process = Process.GetCurrentProcess();
            
            return new SystemMetrics
            {
                Timestamp = DateTime.UtcNow,
                MemoryUsageMB = process.WorkingSet64 / 1024 / 1024,
                CpuTimeMs = process.TotalProcessorTime.TotalMilliseconds,
                UptimeHours = (DateTime.UtcNow - process.StartTime).TotalHours,
                HealthMetrics = _healthMetrics.Values.ToList(),
                PerformanceMetrics = _performanceMetrics.Values.ToList(),
                ThreadCount = process.Threads.Count,
                GCCollections = new Dictionary<int, long>
                {
                    { 0, GC.CollectionCount(0) },
                    { 1, GC.CollectionCount(1) },
                    { 2, GC.CollectionCount(2) }
                }
            };
        }
    }

    /// <summary>
    /// Export metrics in JSON format for external monitoring systems
    /// </summary>
    public string ExportMetricsAsJson()
    {
        var metrics = GetSystemMetrics();
        return JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true });
    }

    private void InitializeMetrics()
    {
        _logger.LogInformation("📊 [MONITORING] Production monitoring service initialized");
        
        // Initialize basic health metrics
        var systemMetric = new HealthMetric("system")
        {
            LastUpdated = DateTime.UtcNow
        };
        _healthMetrics["system"] = systemMetric;
    }

    private Task<(bool IsHealthy, string Message)> CheckModelHealthAsync(CancellationToken cancellationToken)
    {
        try
        {
            lock (_metricsLock)
            {
                var modelMetrics = _performanceMetrics.Where(kvp => kvp.Key.StartsWith("model_")).ToList();
                
                if (!modelMetrics.Any())
                {
                    return Task.FromResult((false, "No model metrics available"));
                }

                var unhealthyModels = modelMetrics.Where(kvp =>
                {
                    var metric = kvp.Value;
                    var accuracy = metric.TotalPredictions > 0 ? (double)metric.CorrectPredictions / metric.TotalPredictions : 0;
                    return accuracy < _config.Performance.AccuracyThreshold && metric.TotalPredictions >= _config.Performance.MinTradesForEvaluation;
                }).ToList();

                if (unhealthyModels.Any())
                {
                    var unhealthyNames = string.Join(", ", unhealthyModels.Select(kvp => kvp.Value.Name));
                    return Task.FromResult((false, $"Models below threshold: {unhealthyNames}"));
                }

                return Task.FromResult((true, $"{modelMetrics.Count} models healthy"));
            }
        }
        catch (Exception ex)
        {
            return Task.FromResult((false, $"Model health check failed: {ex.Message}"));
        }
    }

    private async Task<(bool IsHealthy, string Message)> CheckGitHubConnectivityAsync(CancellationToken cancellationToken)
    {
        try
        {
            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromSeconds(10);
            httpClient.DefaultRequestHeaders.Add("User-Agent", "TradingBot-HealthCheck");
            
            var response = await httpClient.GetAsync("https://api.github.com/", cancellationToken);
            return response.IsSuccessStatusCode 
                ? (true, "GitHub API accessible") 
                : (false, $"GitHub API returned {response.StatusCode}");
        }
        catch (Exception ex)
        {
            return (false, $"GitHub connectivity failed: {ex.Message}");
        }
    }

    private (bool IsHealthy, string Message) CheckSystemResourcesHealth()
    {
        try
        {
            var process = Process.GetCurrentProcess();
            var memoryMB = process.WorkingSet64 / 1024 / 1024;
            
            // Alert if memory usage is excessive
            if (memoryMB > 2048) // 2GB threshold
            {
                return (false, $"High memory usage: {memoryMB}MB");
            }

            // Check if process is responsive
            if (process.Responding)
            {
                return (true, $"System healthy - {memoryMB}MB memory");
            }
            else
            {
                return (false, "Process not responding");
            }
        }
        catch (Exception ex)
        {
            return (false, $"System health check failed: {ex.Message}");
        }
    }

    private (bool IsHealthy, string Message) CheckTradingPerformanceHealth()
    {
        try
        {
            lock (_metricsLock)
            {
                var tradingMetrics = _performanceMetrics.Where(kvp => kvp.Key.StartsWith("trading_")).ToList();
                
                if (!tradingMetrics.Any())
                {
                    return (true, "No trading activity yet");
                }

                var recentMetrics = tradingMetrics.Where(kvp => 
                    DateTime.UtcNow - kvp.Value.LastUpdated < _config.Performance.EvaluationWindow).ToList();

                if (!recentMetrics.Any())
                {
                    return (false, "No recent trading activity");
                }

                var totalDecisions = recentMetrics.Sum(kvp => kvp.Value.TotalPredictions);
                var enhancedDecisions = recentMetrics.Sum(kvp => kvp.Value.CorrectPredictions);
                var enhancementRate = totalDecisions > 0 ? (double)enhancedDecisions / totalDecisions : 0;

                return (true, $"{totalDecisions} decisions, {enhancementRate:P1} enhanced");
            }
        }
        catch (Exception ex)
        {
            return (false, $"Trading performance check failed: {ex.Message}");
        }
    }
}

#region Data Models

public class HealthMetric
{
    public string Name { get; set; }
    public int TotalCalls { get; set; }
    public int SuccessfulCalls { get; set; }
    public int FailedCalls { get; set; }
    public TimeSpan TotalDuration { get; set; }
    public DateTime LastUpdated { get; set; }
    public string? LastError { get; set; }

    public HealthMetric(string name)
    {
        Name = name;
        LastUpdated = DateTime.UtcNow;
    }

    public double SuccessRate => TotalCalls > 0 ? (double)SuccessfulCalls / TotalCalls : 0;
    public double AvgDurationMs => TotalCalls > 0 ? TotalDuration.TotalMilliseconds / TotalCalls : 0;
}

public class PerformanceMetric
{
    public string Name { get; set; }
    public int TotalPredictions { get; set; }
    public int CorrectPredictions { get; set; }
    public double TotalConfidence { get; set; }
    public TimeSpan TotalPredictionTime { get; set; }
    public DateTime LastUpdated { get; set; }

    public PerformanceMetric(string name)
    {
        Name = name;
        LastUpdated = DateTime.UtcNow;
    }

    public double Accuracy => TotalPredictions > 0 ? (double)CorrectPredictions / TotalPredictions : 0;
    public double AvgConfidence => TotalPredictions > 0 ? TotalConfidence / TotalPredictions : 0;
    public double AvgPredictionTimeMs => TotalPredictions > 0 ? TotalPredictionTime.TotalMilliseconds / TotalPredictions : 0;
}

public class SystemMetrics
{
    public DateTime Timestamp { get; set; }
    public long MemoryUsageMB { get; set; }
    public double CpuTimeMs { get; set; }
    public double UptimeHours { get; set; }
    public int ThreadCount { get; set; }
    public Dictionary<int, long> GCCollections { get; set; } = new();
    public List<HealthMetric> HealthMetrics { get; set; } = new();
    public List<PerformanceMetric> PerformanceMetrics { get; set; } = new();
}

#endregion