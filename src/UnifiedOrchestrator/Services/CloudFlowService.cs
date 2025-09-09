using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    /// <summary>
    /// Two-way cloud flow service for pushing trade records and service metrics
    /// Handles after /v1/close push to cloud endpoint with retry logic
    /// </summary>
    public class CloudFlowService : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<CloudFlowService> _logger;
        private readonly CloudFlowOptions _options;
        private readonly JsonSerializerOptions _jsonOptions;
        private bool _disposed = false;

        public CloudFlowService(
            HttpClient httpClient,
            ILogger<CloudFlowService> logger,
            IOptions<CloudFlowOptions> options)
        {
            _httpClient = httpClient;
            _logger = logger;
            _options = options.Value;
            
            // Configure HTTP client for cloud endpoints
            _httpClient.Timeout = TimeSpan.FromSeconds(30);
            
            // Configure JSON serialization
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = false
            };
            
            _logger.LogInformation("☁️ Cloud Flow Service initialized: {CloudEndpoint}", _options.CloudEndpoint);
        }

        /// <summary>
        /// Push trade record to cloud after /v1/close
        /// </summary>
        public async Task PushTradeRecordAsync(TradeRecord tradeRecord, CancellationToken cancellationToken = default)
        {
            if (!_options.Enabled)
            {
                _logger.LogDebug("📤 Cloud flow disabled, skipping trade record push");
                return;
            }

            try
            {
                var payload = new
                {
                    type = "trade_record",
                    timestamp = DateTime.UtcNow,
                    trade = tradeRecord,
                    instanceId = _options.InstanceId
                };

                await PushToCloudWithRetryAsync("trades", payload, cancellationToken);
                _logger.LogInformation("📤 Trade record pushed to cloud: {TradeId}", tradeRecord.TradeId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to push trade record to cloud: {TradeId}", tradeRecord.TradeId);
                throw;
            }
        }

        /// <summary>
        /// Push service metrics to cloud
        /// </summary>
        public async Task PushServiceMetricsAsync(ServiceMetrics metrics, CancellationToken cancellationToken = default)
        {
            if (!_options.Enabled)
            {
                _logger.LogDebug("📤 Cloud flow disabled, skipping metrics push");
                return;
            }

            try
            {
                var payload = new
                {
                    type = "service_metrics",
                    timestamp = DateTime.UtcNow,
                    metrics = metrics,
                    instanceId = _options.InstanceId
                };

                await PushToCloudWithRetryAsync("metrics", payload, cancellationToken);
                _logger.LogDebug("📤 Service metrics pushed to cloud");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to push service metrics to cloud");
                // Don't throw - metrics push failures shouldn't stop trading
            }
        }

        /// <summary>
        /// Push to cloud with exponential backoff retry logic
        /// </summary>
        private async Task PushToCloudWithRetryAsync(string endpoint, object payload, CancellationToken cancellationToken)
        {
            const int maxRetries = 3;
            var baseDelay = TimeSpan.FromSeconds(1);

            for (int attempt = 0; attempt < maxRetries; attempt++)
            {
                try
                {
                    var json = JsonSerializer.Serialize(payload, _jsonOptions);
                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    
                    var url = $"{_options.CloudEndpoint}/{endpoint}";
                    var response = await _httpClient.PostAsync(url, content, cancellationToken);

                    if (response.IsSuccessStatusCode)
                    {
                        return; // Success
                    }

                    // Log non-success response
                    var responseContent = await response.Content.ReadAsStringAsync(cancellationToken);
                    _logger.LogWarning("⚠️ Cloud push failed with status {StatusCode}: {Response}", 
                        response.StatusCode, responseContent);

                    // Don't retry on client errors (4xx)
                    if ((int)response.StatusCode >= 400 && (int)response.StatusCode < 500)
                    {
                        throw new InvalidOperationException($"Client error from cloud endpoint: {response.StatusCode}");
                    }
                }
                catch (TaskCanceledException)
                {
                    _logger.LogWarning("⏱️ Cloud push timeout on attempt {Attempt}", attempt + 1);
                }
                catch (HttpRequestException ex)
                {
                    _logger.LogWarning(ex, "🌐 Network error on cloud push attempt {Attempt}", attempt + 1);
                }

                // Wait before retry (exponential backoff)
                if (attempt < maxRetries - 1)
                {
                    var delay = TimeSpan.FromMilliseconds(baseDelay.TotalMilliseconds * Math.Pow(2, attempt));
                    await Task.Delay(delay, cancellationToken);
                }
            }

            throw new InvalidOperationException($"Failed to push to cloud after {maxRetries} attempts");
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _httpClient?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Configuration options for cloud flow service
    /// </summary>
    public class CloudFlowOptions
    {
        public bool Enabled { get; set; } = true;
        public string CloudEndpoint { get; set; } = string.Empty;
        public string InstanceId { get; set; } = Environment.MachineName;
        public int TimeoutSeconds { get; set; } = 30;
    }

    /// <summary>
    /// Trade record for cloud push
    /// </summary>
    public class TradeRecord
    {
        public string TradeId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string Side { get; set; } = string.Empty;
        public decimal Quantity { get; set; }
        public decimal EntryPrice { get; set; }
        public decimal ExitPrice { get; set; }
        public decimal PnL { get; set; }
        public DateTime EntryTime { get; set; }
        public DateTime ExitTime { get; set; }
        public string Strategy { get; set; } = string.Empty;
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Service metrics for cloud push
    /// </summary>
    public class ServiceMetrics
    {
        public double InferenceLatencyMs { get; set; }
        public double PredictionAccuracy { get; set; }
        public double FeatureDrift { get; set; }
        public int ActiveModels { get; set; }
        public long MemoryUsageMB { get; set; }
        public Dictionary<string, double> CustomMetrics { get; set; } = new();
    }
}