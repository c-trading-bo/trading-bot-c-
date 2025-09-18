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
    public class CloudFlowOptions
    {
        public bool Enabled { get; set; } = false;
        public string CloudEndpoint { get; set; } = "";
        public string InstanceId { get; set; } = "";
        public int TimeoutSeconds { get; set; } = 30;
    }

    public class TradeRecord
    {
        public string TradeId { get; set; } = "";
        public string Symbol { get; set; } = "";
        public string Side { get; set; } = "";
        public int Quantity { get; set; }
        public decimal EntryPrice { get; set; }
        public decimal ExitPrice { get; set; }
        public decimal PnL { get; set; }
        public DateTime EntryTime { get; set; }
        public DateTime ExitTime { get; set; }
        public string Strategy { get; set; } = "";
    }

    public class CloudFlowService : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<CloudFlowService> _logger;
        private readonly CloudFlowOptions _options;

        public CloudFlowService(HttpClient httpClient, ILogger<CloudFlowService> logger, IOptions<CloudFlowOptions> options)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            
            _httpClient.Timeout = TimeSpan.FromSeconds(_options.TimeoutSeconds);
        }

        public async Task PushTradeRecordAsync(TradeRecord tradeRecord, CancellationToken cancellationToken)
        {
            if (!_options.Enabled)
            {
                _logger.LogDebug("CloudFlow disabled, skipping trade record push for {TradeId}", tradeRecord.TradeId);
                return;
            }

            try
            {
                var payload = new
                {
                    instanceId = _options.InstanceId,
                    timestamp = DateTime.UtcNow,
                    tradeRecord = tradeRecord
                };

                var json = JsonSerializer.Serialize(payload);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                _logger.LogDebug("Pushing trade record {TradeId} to cloud endpoint", tradeRecord.TradeId);
                
                var response = await _httpClient.PostAsync(_options.CloudEndpoint, content, cancellationToken).ConfigureAwait(false);
                
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation("Successfully pushed trade record {TradeId} to cloud", tradeRecord.TradeId);
                }
                else
                {
                    _logger.LogWarning("Failed to push trade record {TradeId}, status: {StatusCode}", 
                        tradeRecord.TradeId, response.StatusCode);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error pushing trade record {TradeId} to cloud", tradeRecord.TradeId);
                throw;
            }
        }

        public void Dispose()
        {
            // HttpClient is managed externally, don't dispose it
        }
    }
}