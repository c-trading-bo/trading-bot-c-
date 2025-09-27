using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Services
{
    /// <summary>
    /// Integrates C# Trading Bot with GitHub Actions Workflows
    /// Reads workflow outputs and triggers workflow execution
    /// </summary>
    internal class WorkflowIntegrationService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<WorkflowIntegrationService> _logger;
        private readonly string _intelligenceDataPath = null!;
        private readonly string? _githubToken;
        private readonly string _repoOwner = "c-trading-bo";
        private readonly string _repoName = "trading-bot-c-";

        public WorkflowIntegrationService(
            HttpClient httpClient,
            ILogger<WorkflowIntegrationService> logger,
            string intelligenceDataPath = "data/integrated",
            string? githubToken = null)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _intelligenceDataPath = intelligenceDataPath;
            _githubToken = githubToken ?? Environment.GetEnvironmentVariable("GITHUB_TOKEN");

            if (!string.IsNullOrEmpty(_githubToken))
            {
                _httpClient.DefaultRequestHeaders.Authorization = 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _githubToken);
            }
        }

        /// <summary>
        /// Get latest market intelligence from ML/RL workflows
        /// </summary>
        public async Task<MarketIntelligence> GetLatestMarketIntelligenceAsync()
        {
            try
            {
                var mlRlPath = Path.Combine(_intelligenceDataPath, "ultimate_ml_rl_intel_system_integrated.json");
                var regimePath = Path.Combine(_intelligenceDataPath, "ultimate_regime_detection_pipeline_integrated.json");
                var sentimentPath = Path.Combine(_intelligenceDataPath, "ultimate_news_sentiment_pipeline_integrated.json");

                var intelligence = new MarketIntelligence();

                // Read ML/RL features
                if (File.Exists(mlRlPath))
                {
                    var mlRlData = await File.ReadAllTextAsync(mlRlPath).ConfigureAwait(false);
                    var mlRlJson = JsonDocument.Parse(mlRlData);
                    
                    intelligence.MlConfidence = mlRlJson.RootElement.GetProperty("model_confidence").GetDecimal();
                    intelligence.MlPredictions = JsonSerializer.Deserialize<Dictionary<string, decimal>>(
                        mlRlJson.RootElement.GetProperty("predictions").GetRawText()) ?? new Dictionary<string, decimal>();
                    
                    _logger.LogInformation("[WorkflowIntegration] Loaded ML/RL intelligence: confidence={Confidence:P0}", 
                        intelligence.MlConfidence);
                }

                // Read regime detection
                if (File.Exists(regimePath))
                {
                    var regimeData = await File.ReadAllTextAsync(regimePath).ConfigureAwait(false);
                    var regimeJson = JsonDocument.Parse(regimeData);
                    
                    intelligence.CurrentRegime = regimeJson.RootElement.GetProperty("current_regime").GetString() ?? "unknown";
                    intelligence.RegimeConfidence = regimeJson.RootElement.GetProperty("confidence").GetDecimal();
                    
                    _logger.LogInformation("[WorkflowIntegration] Loaded regime: {Regime} ({Confidence:P0})", 
                        intelligence.CurrentRegime, intelligence.RegimeConfidence);
                }

                // Read news sentiment
                if (File.Exists(sentimentPath))
                {
                    var sentimentData = await File.ReadAllTextAsync(sentimentPath).ConfigureAwait(false);
                    var sentimentJson = JsonDocument.Parse(sentimentData);
                    
                    intelligence.NewsSentiment = sentimentJson.RootElement.GetProperty("overall_sentiment").GetDecimal();
                    intelligence.NewsIntensity = sentimentJson.RootElement.GetProperty("intensity").GetDecimal();
                    
                    _logger.LogInformation("[WorkflowIntegration] Loaded sentiment: {Sentiment:F2} intensity={Intensity:F2}", 
                        intelligence.NewsSentiment, intelligence.NewsIntensity);
                }

                intelligence.LastUpdated = DateTime.UtcNow;
                return intelligence;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[WorkflowIntegration] Failed to load market intelligence");
                return new MarketIntelligence { LastUpdated = DateTime.UtcNow };
            }
        }

        /// <summary>
        /// Get latest supply/demand zones from zone analysis workflow
        /// </summary>
        public async Task<ZoneAnalysis> GetLatestZoneAnalysisAsync(string symbol)
        {
            try
            {
                var zonePath = Path.Combine(_intelligenceDataPath, "zones_identifier_integrated.json");
                
                if (!File.Exists(zonePath))
                {
                    _logger.LogDebug("[WorkflowIntegration] Zone data not found: {Path}", zonePath);
                    return new ZoneAnalysis { Symbol = symbol };
                }

                var zoneData = await File.ReadAllTextAsync(zonePath).ConfigureAwait(false);
                var zoneJson = JsonDocument.Parse(zoneData);

                var zones = new ZoneAnalysis { Symbol = symbol };

                if (zoneJson.RootElement.TryGetProperty(symbol, out var symbolData))
                {
                    if (symbolData.TryGetProperty("supply_zones", out var supplyZones))
                    {
                        zones.SupplyZones = JsonSerializer.Deserialize<List<decimal>>(supplyZones.GetRawText()) ?? new List<decimal>();
                    }

                    if (symbolData.TryGetProperty("demand_zones", out var demandZones))
                    {
                        zones.DemandZones = JsonSerializer.Deserialize<List<decimal>>(demandZones.GetRawText()) ?? new List<decimal>();
                    }

                    if (symbolData.TryGetProperty("poc", out var poc))
                    {
                        zones.PointOfControl = poc.GetDecimal();
                    }

                    _logger.LogInformation("[WorkflowIntegration] Loaded zones for {Symbol}: {Supply} supply, {Demand} demand", 
                        symbol, zones.SupplyZones?.Count ?? 0, zones.DemandZones?.Count ?? 0);
                }

                zones.LastUpdated = DateTime.UtcNow;
                return zones;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[WorkflowIntegration] Failed to load zone analysis for {Symbol}", symbol);
                return new ZoneAnalysis { Symbol = symbol, LastUpdated = DateTime.UtcNow };
            }
        }

        /// <summary>
        /// Trigger a specific GitHub Actions workflow
        /// </summary>
        public async Task<bool> TriggerWorkflowAsync(string workflowFileName, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(_githubToken))
            {
                _logger.LogWarning("[WorkflowIntegration] Cannot trigger workflow - no GitHub token");
                return false;
            }

            try
            {
                var url = $"https://api.github.com/repos/{_repoOwner}/{_repoName}/actions/workflows/{workflowFileName}/dispatches";
                var payload = new { @ref = "main" };
                var json = JsonSerializer.Serialize(payload);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync(url, content, cancellationToken).ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation("[WorkflowIntegration] Successfully triggered workflow: {Workflow}", workflowFileName);
                    return true;
                }
                else
                {
                    _logger.LogWarning("[WorkflowIntegration] Failed to trigger workflow {Workflow}: {Status}", 
                        workflowFileName, response.StatusCode);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[WorkflowIntegration] Error triggering workflow {Workflow}", workflowFileName);
                return false;
            }
        }

        /// <summary>
        /// Get latest correlation matrix from intermarket workflow
        /// </summary>
        public async Task<Dictionary<string, Dictionary<string, decimal>>> GetCorrelationMatrixAsync()
        {
            try
            {
                var correlationPath = Path.Combine(_intelligenceDataPath, "intermarket_integrated.json");
                
                if (!File.Exists(correlationPath))
                {
                    return new Dictionary<string, Dictionary<string, decimal>>();
                }

                var correlationData = await File.ReadAllTextAsync(correlationPath).ConfigureAwait(false);
                var correlationJson = JsonDocument.Parse(correlationData);

                var matrix = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, decimal>>>(
                    correlationJson.RootElement.GetProperty("correlation_matrix").GetRawText());

                _logger.LogInformation("[WorkflowIntegration] Loaded correlation matrix with {Count} assets", 
                    matrix?.Count ?? 0);

                return matrix ?? new Dictionary<string, Dictionary<string, decimal>>();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[WorkflowIntegration] Failed to load correlation matrix");
                return new Dictionary<string, Dictionary<string, decimal>>();
            }
        }

        /// <summary>
        /// Get latest microstructure analysis
        /// </summary>
        public async Task<MicrostructureData> GetMicrostructureAnalysisAsync(string symbol)
        {
            try
            {
                var microPath = Path.Combine(_intelligenceDataPath, "microstructure_integrated.json");
                
                if (!File.Exists(microPath))
                {
                    return new MicrostructureData { Symbol = symbol };
                }

                var microData = await File.ReadAllTextAsync(microPath).ConfigureAwait(false);
                var microJson = JsonDocument.Parse(microData);

                var data = new MicrostructureData { Symbol = symbol };

                if (microJson.RootElement.TryGetProperty(symbol, out var symbolData))
                {
                    if (symbolData.TryGetProperty("order_flow_imbalance", out var imbalance))
                    {
                        data.OrderFlowImbalance = imbalance.GetDecimal();
                    }

                    if (symbolData.TryGetProperty("bid_ask_spread", out var spread))
                    {
                        data.BidAskSpread = spread.GetDecimal();
                    }

                    if (symbolData.TryGetProperty("market_depth", out var depth))
                    {
                        data.MarketDepth = depth.GetInt32();
                    }

                    _logger.LogInformation("[WorkflowIntegration] Loaded microstructure for {Symbol}: imbalance={Imbalance:F3}", 
                        symbol, data.OrderFlowImbalance);
                }

                data.LastUpdated = DateTime.UtcNow;
                return data;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[WorkflowIntegration] Failed to load microstructure for {Symbol}", symbol);
                return new MicrostructureData { Symbol = symbol, LastUpdated = DateTime.UtcNow };
            }
        }
    }

    // Data models for workflow integration
    internal class MarketIntelligence
    {
        public decimal MlConfidence { get; set; }
        public Dictionary<string, decimal> MlPredictions { get; } = new();
        public string CurrentRegime { get; set; } = "Unknown";
        public decimal RegimeConfidence { get; set; }
        public decimal NewsSentiment { get; set; }
        public decimal NewsIntensity { get; set; }
        public DateTime LastUpdated { get; set; }
    }

    internal class ZoneAnalysis
    {
        public string Symbol { get; set; } = "";
        public List<decimal> SupplyZones { get; } = new();
        public List<decimal> DemandZones { get; } = new();
        public decimal PointOfControl { get; set; }
        public DateTime LastUpdated { get; set; }
    }

    internal class MicrostructureData
    {
        public string Symbol { get; set; } = "";
        public decimal OrderFlowImbalance { get; set; }
        public decimal BidAskSpread { get; set; }
        public int MarketDepth { get; set; }
        public DateTime LastUpdated { get; set; }
    }
}
