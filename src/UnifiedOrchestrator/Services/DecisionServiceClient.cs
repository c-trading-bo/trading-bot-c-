using System;
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
    /// Client for ML/RL Decision Service
    /// Provides the four required endpoints: on_new_bar, on_signal, on_order_fill, on_trade_close
    /// </summary>
    public class DecisionServiceClient : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<DecisionServiceClient> _logger;
        private readonly DecisionServiceOptions _options;
        private readonly JsonSerializerOptions _jsonOptions;
        private bool _disposed = false;

        public DecisionServiceClient(
            HttpClient httpClient,
            ILogger<DecisionServiceClient> logger,
            IOptions<DecisionServiceOptions> options)
        {
            _httpClient = httpClient;
            _logger = logger;
            _options = options.Value;
            
            // Configure HTTP client
            _httpClient.BaseAddress = new Uri(_options.BaseUrl);
            _httpClient.Timeout = TimeSpan.FromMilliseconds(_options.TimeoutMs);
            
            // Configure JSON serialization
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = false
            };
            
            _logger.LogInformation("🧠 Decision Service Client initialized: {BaseUrl}", _options.BaseUrl);
        }

        /// <summary>
        /// Send new bar data for regime detection and feature updates
        /// </summary>
        public async Task<TickResponse> OnNewBarAsync(TickRequest request, CancellationToken cancellationToken = default)
        {
            try
            {
                var json = JsonSerializer.Serialize(request, _jsonOptions);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                _logger.LogDebug("📊 Sending tick data for {Symbol}", request.Symbol);
                
                var response = await _httpClient.PostAsync("/v1/tick", content, cancellationToken);
                response.EnsureSuccessStatusCode();
                
                var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
                var result = JsonSerializer.Deserialize<TickResponse>(responseJson, _jsonOptions);
                
                return result ?? new TickResponse { Status = "error", Message = "Empty response" };
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "❌ HTTP error in OnNewBarAsync for {Symbol}", request.Symbol);
                return new TickResponse { Status = "error", Message = $"HTTP error: {ex.Message}" };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error in OnNewBarAsync for {Symbol}", request.Symbol);
                return new TickResponse { Status = "error", Message = ex.Message };
            }
        }

        /// <summary>
        /// Main decision logic: regime → ML blend → UCB → SAC sizing → risk caps
        /// </summary>
        public async Task<SignalResponse> OnSignalAsync(SignalRequest request, CancellationToken cancellationToken = default)
        {
            try
            {
                var json = JsonSerializer.Serialize(request, _jsonOptions);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                _logger.LogDebug("🎯 Processing signal: {Strategy} {Side} {Symbol}", 
                    request.StrategyId, request.Side, request.Symbol);
                
                var response = await _httpClient.PostAsync("/v1/signal", content, cancellationToken);
                response.EnsureSuccessStatusCode();
                
                var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
                var result = JsonSerializer.Deserialize<SignalResponse>(responseJson, _jsonOptions);
                
                if (result != null)
                {
                    _logger.LogInformation("🧠 [DECISION] {DecisionId}: Gate={Gate}, Regime={Regime}, Size={Size}, P={PFinal:F3}", 
                        result.DecisionId, result.Gate, result.Regime, result.FinalContracts, result.PFinal);
                }
                
                return result ?? new SignalResponse { Gate = false, Reason = "Empty response", DecisionId = Guid.NewGuid().ToString() };
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "❌ HTTP error in OnSignalAsync for {Symbol} {Strategy}", request.Symbol, request.StrategyId);
                return new SignalResponse { Gate = false, Reason = $"HTTP error: {ex.Message}", DecisionId = Guid.NewGuid().ToString() };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error in OnSignalAsync for {Symbol} {Strategy}", request.Symbol, request.StrategyId);
                return new SignalResponse { Gate = false, Reason = ex.Message, DecisionId = Guid.NewGuid().ToString() };
            }
        }

        /// <summary>
        /// Handle order fill notification
        /// </summary>
        public async Task<FillResponse> OnOrderFillAsync(FillRequest request, CancellationToken cancellationToken = default)
        {
            try
            {
                var json = JsonSerializer.Serialize(request, _jsonOptions);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                _logger.LogDebug("📈 Notifying fill: {DecisionId} {Contracts} {Symbol} @ {Price}", 
                    request.DecisionId, request.Contracts, request.Symbol, request.EntryPrice);
                
                var response = await _httpClient.PostAsync("/v1/fill", content, cancellationToken);
                response.EnsureSuccessStatusCode();
                
                var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
                var result = JsonSerializer.Deserialize<FillResponse>(responseJson, _jsonOptions);
                
                return result ?? new FillResponse { Status = "error", Message = "Empty response" };
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "❌ HTTP error in OnOrderFillAsync for {DecisionId}", request.DecisionId);
                return new FillResponse { Status = "error", Message = $"HTTP error: {ex.Message}" };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error in OnOrderFillAsync for {DecisionId}", request.DecisionId);
                return new FillResponse { Status = "error", Message = ex.Message };
            }
        }

        /// <summary>
        /// Handle trade close and update online learning
        /// </summary>
        public async Task<CloseResponse> OnTradeCloseAsync(CloseRequest request, CancellationToken cancellationToken = default)
        {
            try
            {
                var json = JsonSerializer.Serialize(request, _jsonOptions);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                _logger.LogDebug("💰 Notifying close: {DecisionId} @ {Price}", 
                    request.DecisionId, request.ExitPrice);
                
                var response = await _httpClient.PostAsync("/v1/close", content, cancellationToken);
                response.EnsureSuccessStatusCode();
                
                var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
                var result = JsonSerializer.Deserialize<CloseResponse>(responseJson, _jsonOptions);
                
                if (result != null && result.Status == "ok")
                {
                    _logger.LogInformation("💰 [CLOSE] {DecisionId}: PnL=${Pnl:F2}, Daily=${DailyPnl:F2}", 
                        request.DecisionId, result.Pnl, result.DailyPnl);
                }
                
                return result ?? new CloseResponse { Status = "error", Message = "Empty response" };
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "❌ HTTP error in OnTradeCloseAsync for {DecisionId}", request.DecisionId);
                return new CloseResponse { Status = "error", Message = $"HTTP error: {ex.Message}" };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error in OnTradeCloseAsync for {DecisionId}", request.DecisionId);
                return new CloseResponse { Status = "error", Message = ex.Message };
            }
        }

        /// <summary>
        /// Get service health status
        /// </summary>
        public async Task<HealthResponse> GetHealthAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                var response = await _httpClient.GetAsync("/health", cancellationToken);
                response.EnsureSuccessStatusCode();
                
                var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
                var result = JsonSerializer.Deserialize<HealthResponse>(responseJson, _jsonOptions);
                
                return result ?? new HealthResponse { Status = "UNKNOWN" };
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ Health check failed for Decision Service");
                return new HealthResponse { Status = "ERROR", Message = ex.Message };
            }
        }

        /// <summary>
        /// Get service statistics
        /// </summary>
        public async Task<StatsResponse> GetStatsAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                var response = await _httpClient.GetAsync("/v1/stats", cancellationToken);
                response.EnsureSuccessStatusCode();
                
                var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
                var result = JsonSerializer.Deserialize<StatsResponse>(responseJson, _jsonOptions);
                
                return result ?? new StatsResponse();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ Stats request failed for Decision Service");
                return new StatsResponse();
            }
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
    /// Configuration options for Decision Service client
    /// </summary>
    public class DecisionServiceOptions
    {
        public string BaseUrl { get; set; } = "http://127.0.0.1:7080";
        public int TimeoutMs { get; set; } = 5000;
        public bool Enabled { get; set; } = true;
        public int MaxRetries { get; set; } = 3;
        public int RetryDelayMs { get; set; } = 100;
    }

    // Request/Response DTOs
    public class TickRequest
    {
        public string Ts { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public decimal O { get; set; }
        public decimal H { get; set; }
        public decimal L { get; set; }
        public decimal C { get; set; }
        public long V { get; set; }
        public int BidSize { get; set; }
        public int AskSize { get; set; }
        public int LastTradeDir { get; set; }
        public string Session { get; set; } = "RTH";
    }

    public class TickResponse
    {
        public string Status { get; set; } = string.Empty;
        public string Regime { get; set; } = string.Empty;
        public string FeatureSnapshotId { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
    }

    public class SignalRequest
    {
        public string Symbol { get; set; } = string.Empty;
        public string StrategyId { get; set; } = string.Empty;
        public string Side { get; set; } = string.Empty;
        public string SignalId { get; set; } = string.Empty;
        public Dictionary<string, object> Hints { get; set; } = new();
        public CloudData Cloud { get; set; } = new();
    }

    public class CloudData
    {
        public decimal P { get; set; }
        public string SourceModelId { get; set; } = string.Empty;
        public int LatencyMs { get; set; }
    }

    public class SignalResponse
    {
        public bool Gate { get; set; }
        public string Reason { get; set; } = string.Empty;
        public string Regime { get; set; } = string.Empty;
        public decimal PCloud { get; set; }
        public decimal POnline { get; set; }
        public decimal PFinal { get; set; }
        public decimal Ucb { get; set; }
        public int ProposedContracts { get; set; }
        public int FinalContracts { get; set; }
        public RiskInfo Risk { get; set; } = new();
        public ManagementPlan ManagementPlan { get; set; } = new();
        public string DecisionId { get; set; } = string.Empty;
        public double LatencyMs { get; set; }
        public bool DegradedMode { get; set; }
    }

    public class RiskInfo
    {
        public decimal StopPoints { get; set; }
        public int PointValue { get; set; }
        public decimal DailySoftLossRemaining { get; set; }
        public bool MlHeadroomOk { get; set; }
    }

    public class ManagementPlan
    {
        public decimal Tp1AtR { get; set; }
        public decimal Tp1Pct { get; set; }
        public bool MoveStopToBEOnTp1 { get; set; }
        public List<string> AllowedActions { get; set; } = new();
        public decimal TrailATRMultiplier { get; set; }
        public decimal MaxTrailATRMultiplier { get; set; }
        public decimal StopPoints { get; set; }
    }

    public class FillRequest
    {
        public string DecisionId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string StrategyId { get; set; } = string.Empty;
        public string Side { get; set; } = string.Empty;
        public string EntryTs { get; set; } = string.Empty;
        public decimal EntryPrice { get; set; }
        public int Contracts { get; set; }
    }

    public class FillResponse
    {
        public string Status { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
    }

    public class CloseRequest
    {
        public string DecisionId { get; set; } = string.Empty;
        public string ExitTs { get; set; } = string.Empty;
        public decimal ExitPrice { get; set; }
        public int FinalContracts { get; set; }
    }

    public class CloseResponse
    {
        public string Status { get; set; } = string.Empty;
        public decimal Pnl { get; set; }
        public decimal DailyPnl { get; set; }
        public string Message { get; set; } = string.Empty;
    }

    public class HealthResponse
    {
        public string Status { get; set; } = string.Empty;
        public string Regime { get; set; } = string.Empty;
        public decimal DailyPnl { get; set; }
        public int TotalContracts { get; set; }
        public int ActivePositions { get; set; }
        public bool DegradedMode { get; set; }
        public double AvgLatencyMs { get; set; }
        public string Message { get; set; } = string.Empty;
    }

    public class StatsResponse
    {
        public string Regime { get; set; } = string.Empty;
        public decimal DailyPnl { get; set; }
        public int TotalContracts { get; set; }
        public int ActivePositions { get; set; }
        public bool DegradedMode { get; set; }
        public int DecisionCount { get; set; }
        public double AvgLatencyMs { get; set; }
        public List<PositionInfo> Positions { get; set; } = new();
    }

    public class PositionInfo
    {
        public string DecisionId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string Side { get; set; } = string.Empty;
        public int Contracts { get; set; }
        public string Status { get; set; } = string.Empty;
        public decimal Pnl { get; set; }
    }
}