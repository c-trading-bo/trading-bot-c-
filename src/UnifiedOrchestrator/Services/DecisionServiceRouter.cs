using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Text.Json;
using System.Net.Http;
using BotCore.Services;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// 🔄 DECISION SERVICE ROUTER - PYTHON/C# INTEGRATION LAYER 🔄
/// 
/// Integrates with Python decision services and falls back to C# brains when needed.
/// Ensures seamless integration between Python UCB services and C# trading logic.
/// 
/// Integration Flow:
/// 1. Python Decision Service (if available and healthy)
/// 2. C# UnifiedDecisionRouter (primary C# logic)
/// 3. Direct strategy execution (ultimate fallback)
/// </summary>
public class DecisionServiceRouter
{
    private readonly ILogger<DecisionServiceRouter> _logger;
    private readonly HttpClient _httpClient;
    private readonly DecisionServiceOptions _options;
    private readonly UnifiedDecisionRouter _unifiedRouter;
    private readonly IServiceProvider _serviceProvider;
    
    // Service health tracking
    private bool _pythonServiceHealthy = false;
    private DateTime _lastHealthCheck = DateTime.MinValue;
    private readonly TimeSpan _healthCheckInterval = TimeSpan.FromSeconds(30);
    
    public DecisionServiceRouter(
        ILogger<DecisionServiceRouter> logger,
        HttpClient httpClient,
        IOptions<DecisionServiceOptions> options,
        UnifiedDecisionRouter unifiedRouter,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _httpClient = httpClient;
        _options = options.Value;
        _unifiedRouter = unifiedRouter;
        _serviceProvider = serviceProvider;
        
        _httpClient.Timeout = TimeSpan.FromMilliseconds(_options.TimeoutMs);
        
        _logger.LogInformation("🔄 [DECISION-SERVICE-ROUTER] Initialized with Python endpoint: {Endpoint}", 
            _options.BaseUrl);
    }
    
    /// <summary>
    /// Route decision through integrated Python/C# system
    /// </summary>
    public async Task<UnifiedTradingDecision> RouteIntegratedDecisionAsync(
        string symbol,
        TradingBot.Abstractions.MarketContext marketContext,
        CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;
        
        try
        {
            _logger.LogDebug("🔄 [DECISION-SERVICE-ROUTER] Routing integrated decision for {Symbol}", symbol);
            
            // Step 1: Check Python decision service health
            await CheckPythonServiceHealthAsync(cancellationToken).ConfigureAwait(false);
            
            // Step 2: Try Python decision service if healthy
            if (_pythonServiceHealthy && _options.Enabled)
            {
                var pythonDecision = await TryPythonDecisionServiceAsync(symbol, marketContext, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                if (pythonDecision != null)
                {
                    pythonDecision.DecisionSource = "PythonDecisionService";
                    pythonDecision.ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
                    
                    _logger.LogInformation("🐍 [PYTHON-DECISION] Decision: {Action} {Symbol} confidence={Confidence:P1}",
                        pythonDecision.Action, symbol, pythonDecision.Confidence);
                    return pythonDecision;
                }
            }
            
            // Step 3: Fall back to C# UnifiedDecisionRouter
            _logger.LogDebug("🔄 [DECISION-SERVICE-ROUTER] Python service unavailable, using C# unified router");
            
            // No conversion needed - both are BotCore.Services.MarketContext
            var csharpDecision = await _unifiedRouter.RouteDecisionAsync(symbol, marketContext, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            csharpDecision.DecisionSource = $"CSharp_{csharpDecision.DecisionSource}";
            
            // Convert from BotCore.Services.UnifiedTradingDecision to local UnifiedTradingDecision
            return new UnifiedTradingDecision
            {
                DecisionId = csharpDecision.DecisionId,
                Symbol = csharpDecision.Symbol,
                Action = csharpDecision.Action,
                Confidence = csharpDecision.Confidence,
                Quantity = csharpDecision.Quantity,
                Strategy = csharpDecision.Strategy,
                DecisionSource = csharpDecision.DecisionSource,
                Reasoning = csharpDecision.Reasoning,
                Timestamp = csharpDecision.Timestamp,
                ProcessingTimeMs = csharpDecision.ProcessingTimeMs
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [DECISION-SERVICE-ROUTER] Error in integrated routing for {Symbol}", symbol);
            
            // Ultimate fallback to C# system
            var fallbackDecision = await _unifiedRouter.RouteDecisionAsync(symbol, marketContext, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            // Convert from BotCore.Services.UnifiedTradingDecision to local UnifiedTradingDecision  
            return new UnifiedTradingDecision
            {
                DecisionId = fallbackDecision.DecisionId,
                Symbol = fallbackDecision.Symbol,
                Action = fallbackDecision.Action,
                Confidence = fallbackDecision.Confidence,
                Quantity = fallbackDecision.Quantity,
                Strategy = fallbackDecision.Strategy,
                DecisionSource = fallbackDecision.DecisionSource,
                Reasoning = fallbackDecision.Reasoning,
                Timestamp = fallbackDecision.Timestamp,
                ProcessingTimeMs = fallbackDecision.ProcessingTimeMs
            };
        }
    }
    
    /// <summary>
    /// Check Python decision service health
    /// </summary>
    private async Task CheckPythonServiceHealthAsync(CancellationToken cancellationToken)
    {
        if (DateTime.UtcNow - _lastHealthCheck < _healthCheckInterval)
        {
            return; // Skip if recently checked
        }
        
        try
        {
            if (!_options.Enabled)
            {
                _pythonServiceHealthy = false;
                return;
            }
            
            var healthUrl = $"{_options.BaseUrl}/health";
            var response = await _httpClient.GetAsync(healthUrl, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            _pythonServiceHealthy = response.IsSuccessStatusCode;
            _lastHealthCheck = DateTime.UtcNow;
            
            if (_pythonServiceHealthy)
            {
                _logger.LogDebug("✅ [PYTHON-SERVICE-HEALTH] Python decision service is healthy");
            }
            else
            {
                _logger.LogWarning("⚠️ [PYTHON-SERVICE-HEALTH] Python decision service returned {StatusCode}", 
                    response.StatusCode);
            }
        }
        catch (Exception ex)
        {
            _pythonServiceHealthy = false;
            _lastHealthCheck = DateTime.UtcNow;
            _logger.LogWarning(ex, "⚠️ [PYTHON-SERVICE-HEALTH] Python decision service health check failed");
        }
    }
    
    /// <summary>
    /// Try to get decision from Python decision service
    /// </summary>
    private async Task<UnifiedTradingDecision?> TryPythonDecisionServiceAsync(
        string symbol,
        TradingBot.Abstractions.MarketContext marketContext,
        CancellationToken cancellationToken)
    {
        try
        {
            var requestPayload = new
            {
                symbol = symbol,
                price = marketContext.Price,
                volume = marketContext.Volume,
                timestamp = marketContext.Timestamp.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                technical_indicators = marketContext.TechnicalIndicators,
                request_id = Guid.NewGuid().ToString()
            };
            
            var json = JsonSerializer.Serialize(requestPayload);
            var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            
            var decisionUrl = $"{_options.BaseUrl}/decision";
            var response = await _httpClient.PostAsync(decisionUrl, content, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("⚠️ [PYTHON-DECISION] Service returned {StatusCode}", response.StatusCode);
                return null;
            }
            
            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            var pythonResponse = JsonSerializer.Deserialize<PythonDecisionResponse>(responseJson);
            
            if (pythonResponse == null)
            {
                _logger.LogWarning("⚠️ [PYTHON-DECISION] Failed to parse response");
                return null;
            }
            
            return ConvertPythonDecision(pythonResponse, symbol);
        }
        catch (TaskCanceledException)
        {
            _logger.LogWarning("⏰ [PYTHON-DECISION] Request timed out after {TimeoutMs}ms", _options.TimeoutMs);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ [PYTHON-DECISION] Failed to get decision from Python service");
            return null;
        }
    }
    
    /// <summary>
    /// Convert Python decision response to unified format
    /// </summary>
    private UnifiedTradingDecision ConvertPythonDecision(PythonDecisionResponse pythonResponse, string symbol)
    {
        // Parse Python decision action
        var action = pythonResponse.Decision?.Action?.ToUpper() switch
        {
            "BUY" or "LONG" => TradingAction.Buy,
            "SELL" or "SHORT" => TradingAction.Sell,
            _ => pythonResponse.Confidence > 0.5 ? TradingAction.Buy : TradingAction.Sell // Force decision if unclear
        };
        
        // Ensure minimum viable confidence
        var confidence = Math.Max(0.51m, (decimal)pythonResponse.Confidence);
        
        // Conservative position sizing from Python service
        var quantity = Math.Max(1m, Math.Min(3m, (decimal)(pythonResponse.Decision?.PositionSize ?? 1.0)));
        
        return new UnifiedTradingDecision
        {
            Symbol = symbol,
            Action = action,
            Confidence = confidence,
            Quantity = quantity,
            Strategy = pythonResponse.Decision?.Strategy ?? "PYTHON_UCB",
            Reasoning = new Dictionary<string, object>
            {
                ["python_response"] = pythonResponse,
                ["original_action"] = pythonResponse.Decision?.Action ?? "unknown",
                ["original_confidence"] = pythonResponse.Confidence,
                ["model_used"] = pythonResponse.ModelInfo?.ModelName ?? "unknown",
                ["features_count"] = pythonResponse.Features?.Count ?? 0,
                ["processing_time_python"] = pythonResponse.ProcessingTime,
                ["confidence_boost_applied"] = pythonResponse.Confidence < 0.51
            },
            Timestamp = DateTime.UtcNow
        };
    }
    
    /// <summary>
    /// Get Python service status for monitoring
    /// </summary>
    public DecisionServiceStatus GetServiceStatus()
    {
        return new DecisionServiceStatus
        {
            PythonServiceHealthy = _pythonServiceHealthy,
            LastHealthCheck = _lastHealthCheck,
            ServiceEndpoint = _options.BaseUrl,
            Enabled = _options.Enabled,
            TimeoutMs = _options.TimeoutMs,
            MaxRetries = _options.MaxRetries
        };
    }
    
    /// <summary>
    /// Submit trading outcome to both Python service and C# brains
    /// </summary>
    public async Task SubmitIntegratedOutcomeAsync(
        string decisionId,
        decimal realizedPnL,
        bool wasCorrect,
        TimeSpan holdTime,
        string decisionSource,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Always submit to C# system
            await _unifiedRouter.SubmitTradingOutcomeAsync(decisionId, realizedPnL, wasCorrect, holdTime, cancellationToken).ConfigureAwait(false);
            
            // Submit to Python service if it was the decision source
            if (decisionSource.StartsWith("Python") && _pythonServiceHealthy && _options.Enabled)
            {
                await SubmitOutcomeToPythonServiceAsync(decisionId, realizedPnL, wasCorrect, holdTime, cancellationToken).ConfigureAwait(false);
            }
            
            _logger.LogInformation("📈 [INTEGRATED-FEEDBACK] Outcome submitted to both systems: {DecisionId}", decisionId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [INTEGRATED-FEEDBACK] Failed to submit integrated outcome");
        }
    }
    
    /// <summary>
    /// Submit outcome to Python decision service
    /// </summary>
    private async Task SubmitOutcomeToPythonServiceAsync(
        string decisionId,
        decimal realizedPnL,
        bool wasCorrect,
        TimeSpan holdTime,
        CancellationToken cancellationToken)
    {
        try
        {
            var outcomePayload = new
            {
                decision_id = decisionId,
                realized_pnl = (double)realizedPnL,
                was_correct = wasCorrect,
                hold_time_minutes = holdTime.TotalMinutes,
                timestamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            };
            
            var json = JsonSerializer.Serialize(outcomePayload);
            var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            
            var outcomeUrl = $"{_options.BaseUrl}/outcome";
            var response = await _httpClient.PostAsync(outcomeUrl, content, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            if (response.IsSuccessStatusCode)
            {
                _logger.LogDebug("✅ [PYTHON-FEEDBACK] Outcome submitted to Python service successfully");
            }
            else
            {
                _logger.LogWarning("⚠️ [PYTHON-FEEDBACK] Python service returned {StatusCode} for outcome submission", 
                    response.StatusCode);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [PYTHON-FEEDBACK] Failed to submit outcome to Python service");
        }
    }
}

#region Python Service Models

public class PythonDecisionResponse
{
    public DecisionDetails? Decision { get; set; }
    public double Confidence { get; set; }
    public List<FeatureInfo>? Features { get; set; }
    public ModelInfo? ModelInfo { get; set; }
    public double ProcessingTime { get; set; }
    public string? RequestId { get; set; }
}

public class DecisionDetails
{
    public string? Action { get; set; }
    public string? Strategy { get; set; }
    public double PositionSize { get; set; }
    public Dictionary<string, object>? Metadata { get; set; }
}

public class FeatureInfo
{
    public string? Name { get; set; }
    public double Value { get; set; }
    public double Weight { get; set; }
}

public class ModelInfo
{
    public string? ModelName { get; set; }
    public string? Version { get; set; }
    public double Accuracy { get; set; }
    public DateTime LastTrained { get; set; }
}

public class DecisionServiceStatus
{
    public bool PythonServiceHealthy { get; set; }
    public DateTime LastHealthCheck { get; set; }
    public string ServiceEndpoint { get; set; } = string.Empty;
    public bool Enabled { get; set; }
    public int TimeoutMs { get; set; }
    public int MaxRetries { get; set; }
}

public class DecisionServiceOptions
{
    public string BaseUrl { get; set; } = string.Empty;
    public int TimeoutMs { get; set; } = 5000;
    public bool Enabled { get; set; } = true;
    public int MaxRetries { get; set; } = 3;
}

#endregion