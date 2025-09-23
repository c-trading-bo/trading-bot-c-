using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Cloud flow service for pushing trading data to cloud endpoints
/// Extracted from IntelligenceOrchestrator to reduce file size while maintaining full functionality
/// </summary>
public partial class CloudFlowService
{
    private const int HttpClientErrorStart = 400;
    private const int HttpServerErrorStart = 500;

    private readonly ILogger<CloudFlowService> _logger;
    private readonly HttpClient _httpClient;
    private readonly CloudFlowOptions _cloudFlowOptions;
    private readonly JsonSerializerOptions _jsonOptions;

    public CloudFlowService(
        ILogger<CloudFlowService> logger,
        HttpClient httpClient,
        IOptions<CloudFlowOptions> cloudFlowOptions)
    {
        _logger = logger;
        _httpClient = httpClient;
        _cloudFlowOptions = cloudFlowOptions.Value;
        
        // Configure HTTP client for cloud endpoints
        _httpClient.Timeout = TimeSpan.FromSeconds(_cloudFlowOptions.TimeoutSeconds);
        
        // Configure JSON serialization
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = false
        };
    }

    /// <summary>
    /// Push trade record to cloud after decision execution
    /// </summary>
    public async Task PushTradeRecordAsync(CloudTradeRecord tradeRecord, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(tradeRecord);
        if (!_cloudFlowOptions.Enabled)
        {
            CloudFlowDisabledDebug(_logger, null);
            return;
        }

        try
        {
            var payload = new
            {
                type = "trade_record",
                timestamp = DateTime.UtcNow,
                trade = tradeRecord,
                instanceId = _cloudFlowOptions.InstanceId
            };

            await PushToCloudWithRetryAsync("trades", payload, cancellationToken).ConfigureAwait(false);
            TradeRecordPushedInfo(_logger, tradeRecord.TradeId, null);
        }
        catch (HttpRequestException ex)
        {
            TradeRecordPushFailed(_logger, tradeRecord.TradeId, ex);
            // Don't throw - cloud push failures shouldn't stop trading
        }
        catch (TaskCanceledException ex)
        {
            TradeRecordPushFailed(_logger, tradeRecord.TradeId, ex);
            // Don't throw - cloud push failures shouldn't stop trading
        }
        catch (JsonException ex)
        {
            TradeRecordPushFailed(_logger, tradeRecord.TradeId, ex);
            // Don't throw - cloud push failures shouldn't stop trading
        }
    }

    /// <summary>
    /// Push service metrics to cloud
    /// </summary>
    public async Task PushServiceMetricsAsync(CloudServiceMetrics metrics, CancellationToken cancellationToken = default)
    {
        if (!_cloudFlowOptions.Enabled)
        {
            CloudFlowDisabledMetricsDebug(_logger, null);
            return;
        }

        try
        {
            var payload = new
            {
                type = "service_metrics",
                timestamp = DateTime.UtcNow,
                metrics = metrics,
                instanceId = _cloudFlowOptions.InstanceId
            };

            await PushToCloudWithRetryAsync("metrics", payload, cancellationToken).ConfigureAwait(false);
            MetricsPushedDebug(_logger, null);
        }
        catch (HttpRequestException ex)
        {
            MetricsPushFailed(_logger, ex);
            // Don't throw - metrics push failures shouldn't stop trading
        }
        catch (TaskCanceledException ex)
        {
            MetricsPushFailed(_logger, ex);
            // Don't throw - metrics push failures shouldn't stop trading
        }
        catch (ArgumentException ex)
        {
            MetricsPushFailed(_logger, ex);
            // Don't throw - metrics push failures shouldn't stop trading
        }
    }

    /// <summary>
    /// Push decision intelligence data to cloud
    /// </summary>
    public async Task PushDecisionIntelligenceAsync(TradingDecision decision, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(decision);
        if (!_cloudFlowOptions.Enabled)
        {
            return;
        }

        try
        {
            var intelligenceData = new
            {
                type = "decision_intelligence",
                timestamp = DateTime.UtcNow,
                decisionId = decision.DecisionId,
                symbol = decision.Signal?.Symbol,
                action = decision.Action.ToString(),
                confidence = decision.Confidence,
                mlStrategy = decision.MLStrategy,
                marketRegime = decision.MarketRegime,
                reasoning = decision.Reasoning,
                instanceId = _cloudFlowOptions.InstanceId
            };

            await PushToCloudWithRetryAsync("intelligence", intelligenceData, cancellationToken).ConfigureAwait(false);
            DecisionIntelligencePushedDebug(_logger, decision.DecisionId, null);
        }
        catch (HttpRequestException ex)
        {
            DecisionIntelligencePushFailed(_logger, decision.DecisionId, ex);
        }
        catch (TaskCanceledException ex)
        {
            DecisionIntelligencePushFailed(_logger, decision.DecisionId, ex);
        }
        catch (ArgumentException ex)
        {
            DecisionIntelligencePushFailed(_logger, decision.DecisionId, ex);
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
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var url = $"{_cloudFlowOptions.CloudEndpoint}/{endpoint}";
                using var response = await _httpClient.PostAsync(url, content, cancellationToken).ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    return; // Success
                }

                // Log non-success response
                var responseContent = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                CloudPushFailedWarning(_logger, (int)response.StatusCode, responseContent, null);

                // Don't retry on client errors (4xx)
                if ((int)response.StatusCode >= HttpClientErrorStart && (int)response.StatusCode < HttpServerErrorStart)
                {
                    throw new InvalidOperationException($"Client error from cloud endpoint: {response.StatusCode}");
                }
            }
            catch (TaskCanceledException ex)
            {
                CloudPushTimeoutWarning(_logger, attempt + 1, ex);
            }
            catch (HttpRequestException ex)
            {
                CloudPushNetworkErrorWarning(_logger, attempt + 1, ex);
            }

            // Wait before retry (exponential backoff)
            if (attempt < maxRetries - 1)
            {
                var delay = TimeSpan.FromMilliseconds(baseDelay.TotalMilliseconds * Math.Pow(2, attempt));
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }

        throw new InvalidOperationException($"Failed to push to cloud after {maxRetries} attempts");
    }

    #region LoggerMessage Delegates

    [LoggerMessage(EventId = 4030, Level = LogLevel.Debug, Message = "[CLOUDFLOW] Cloud flow disabled, skipping push")]
    private static partial void CloudFlowDisabledDebug(ILogger logger, Exception? ex);

    [LoggerMessage(EventId = 4031, Level = LogLevel.Information, Message = "[CLOUDFLOW] Trade record pushed to cloud: {TradeId}")]
    private static partial void TradeRecordPushedInfo(ILogger logger, string tradeId, Exception? ex);

    [LoggerMessage(EventId = 4032, Level = LogLevel.Error, Message = "[CLOUDFLOW] Failed to push trade record to cloud: {TradeId}")]
    private static partial void TradeRecordPushFailed(ILogger logger, string tradeId, Exception ex);

    [LoggerMessage(EventId = 4033, Level = LogLevel.Debug, Message = "[CLOUDFLOW] Cloud flow disabled, skipping metrics push")]
    private static partial void CloudFlowDisabledMetricsDebug(ILogger logger, Exception? ex);

    [LoggerMessage(EventId = 4034, Level = LogLevel.Debug, Message = "[CLOUDFLOW] Service metrics pushed to cloud")]
    private static partial void MetricsPushedDebug(ILogger logger, Exception? ex);

    [LoggerMessage(EventId = 4035, Level = LogLevel.Error, Message = "[CLOUDFLOW] Failed to push service metrics to cloud")]
    private static partial void MetricsPushFailed(ILogger logger, Exception ex);

    [LoggerMessage(EventId = 4036, Level = LogLevel.Debug, Message = "[CLOUDFLOW] Decision intelligence pushed to cloud: {DecisionId}")]
    private static partial void DecisionIntelligencePushedDebug(ILogger logger, string decisionId, Exception? ex);

    [LoggerMessage(EventId = 4037, Level = LogLevel.Error, Message = "[CLOUDFLOW] Failed to push decision intelligence to cloud: {DecisionId}")]
    private static partial void DecisionIntelligencePushFailed(ILogger logger, string decisionId, Exception ex);

    [LoggerMessage(EventId = 4038, Level = LogLevel.Warning, Message = "[CLOUDFLOW] Cloud push failed with status {StatusCode}: {Response}")]
    private static partial void CloudPushFailedWarning(ILogger logger, int statusCode, string response, Exception? ex);

    [LoggerMessage(EventId = 4039, Level = LogLevel.Warning, Message = "[CLOUDFLOW] Cloud push timeout on attempt {Attempt}")]
    private static partial void CloudPushTimeoutWarning(ILogger logger, int attempt, Exception ex);

    [LoggerMessage(EventId = 4040, Level = LogLevel.Warning, Message = "[CLOUDFLOW] Network error on cloud push attempt {Attempt}")]
    private static partial void CloudPushNetworkErrorWarning(ILogger logger, int attempt, Exception ex);

    #endregion
}