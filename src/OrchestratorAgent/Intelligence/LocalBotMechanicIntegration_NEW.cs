using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Intelligence
{
    /// <summary>
    /// Background service that integrates local C# bot with GitHub Actions workflows
    /// Continuously reads workflow outputs and applies intelligence to trading decisions
    /// </summary>
    public class LocalBotMechanicIntegration : BackgroundService
    {
        private readonly WorkflowIntegrationService _workflowService;
        private readonly ILogger<LocalBotMechanicIntegration> _logger;
        private readonly TimeSpan _refreshInterval = TimeSpan.FromMinutes(2); // Read workflow data every 2 minutes

        public LocalBotMechanicIntegration(
            WorkflowIntegrationService workflowService,
            ILogger<LocalBotMechanicIntegration> logger)
        {
            _workflowService = workflowService ?? throw new ArgumentNullException(nameof(workflowService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("[LocalBotMechanic] Starting workflow integration service...");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await RefreshWorkflowIntelligenceAsync(stoppingToken);
                    await Task.Delay(_refreshInterval, stoppingToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[LocalBotMechanic] Error in workflow integration loop");
                    await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken); // Wait before retry
                }
            }

            _logger.LogInformation("[LocalBotMechanic] Workflow integration service stopped");
        }

        /// <summary>
        /// Reads latest workflow outputs and applies intelligence to trading logic
        /// </summary>
        private async Task RefreshWorkflowIntelligenceAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogDebug("[LocalBotMechanic] Refreshing workflow intelligence...");

                // Get latest market intelligence from workflow outputs
                var marketIntel = await _workflowService.GetLatestMarketIntelligenceAsync();
                if (marketIntel != null)
                {
                    _logger.LogInformation("[LocalBotMechanic] Applied market intelligence: regime={Regime} confidence={Confidence:P0} bias={Bias}",
                        marketIntel.Regime, marketIntel.Confidence, marketIntel.PrimaryBias);
                    
                    await ApplyMarketIntelligenceAsync(marketIntel);
                }

                // Get zone analysis for ES and NQ
                foreach (var symbol in new[] { "ES", "NQ" })
                {
                    var zones = await _workflowService.GetLatestZoneAnalysisAsync(symbol);
                    if (zones != null)
                    {
                        _logger.LogInformation("[LocalBotMechanic] Applied zone analysis for {Symbol}: {Supply} supply, {Demand} demand zones",
                            symbol, zones.SupplyZones.Count, zones.DemandZones.Count);
                        
                        await ApplyZoneAnalysisAsync(symbol, zones);
                    }
                }

                // Get correlation data
                var correlations = await _workflowService.GetLatestCorrelationDataAsync();
                if (correlations != null)
                {
                    _logger.LogInformation("[LocalBotMechanic] Applied correlation analysis: {Count} instruments tracked",
                        correlations.Correlations.Count);
                    
                    await ApplyCorrelationAnalysisAsync(correlations);
                }

                // Get sentiment data
                var sentiment = await _workflowService.GetLatestSentimentAsync();
                if (sentiment != null)
                {
                    _logger.LogInformation("[LocalBotMechanic] Applied sentiment analysis: score={Score:F2} from {Sources} sources",
                        sentiment.OverallScore, sentiment.Sources.Count);
                    
                    await ApplySentimentAnalysisAsync(sentiment);
                }

                _logger.LogDebug("[LocalBotMechanic] Workflow intelligence refresh completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Failed to refresh workflow intelligence");
            }
        }

        /// <summary>
        /// Applies market regime intelligence to strategy selection and risk management
        /// </summary>
        private async Task ApplyMarketIntelligenceAsync(MarketIntelligence intel)
        {
            try
            {
                // Apply regime-based strategy preferences
                if (intel.Regime == "Trending" && intel.Confidence > 0.7m)
                {
                    _logger.LogInformation("[LocalBotMechanic] High-confidence trending regime: favoring breakout strategies");
                    // Set environment variables or status flags for strategy engine to read
                    Environment.SetEnvironmentVariable("REGIME_STRATEGY_PREFERENCE", "S6,S2"); // Favor S6 breakout, S2 momentum
                    Environment.SetEnvironmentVariable("REGIME_CONFIDENCE", intel.Confidence.ToString("F2"));
                }
                else if (intel.Regime == "Ranging" && intel.Confidence > 0.7m)
                {
                    _logger.LogInformation("[LocalBotMechanic] High-confidence ranging regime: favoring mean reversion");
                    Environment.SetEnvironmentVariable("REGIME_STRATEGY_PREFERENCE", "S3,S11"); // Favor S3 pullback, S11 reversal
                    Environment.SetEnvironmentVariable("REGIME_CONFIDENCE", intel.Confidence.ToString("F2"));
                }

                // Apply news-based position sizing adjustments
                if (intel.IsFomcDay || intel.IsCpiDay)
                {
                    _logger.LogInformation("[LocalBotMechanic] High-impact news day detected: reducing position sizes");
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_SCALE", "0.5"); // Halve position sizes
                }
                else if (intel.NewsIntensity > 70m)
                {
                    _logger.LogInformation("[LocalBotMechanic] High news intensity: moderate position reduction");
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_SCALE", "0.75"); // 25% reduction
                }
                else
                {
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_SCALE", "1.0"); // Normal sizing
                }

                // Apply bias-based entry filtering
                Environment.SetEnvironmentVariable("REGIME_PRIMARY_BIAS", intel.PrimaryBias);
                
                await Task.CompletedTask; // Async for consistency
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying market intelligence");
            }
        }

        /// <summary>
        /// Applies zone analysis to stop/target placement and entry timing
        /// </summary>
        private async Task ApplyZoneAnalysisAsync(string symbol, ZoneAnalysis zones)
        {
            try
            {
                // Find strongest supply/demand zones for optimal stop placement
                var strongestSupply = zones.SupplyZones
                    .OrderByDescending(z => z.Strength)
                    .FirstOrDefault();
                
                var strongestDemand = zones.DemandZones
                    .OrderByDescending(z => z.Strength)
                    .FirstOrDefault();

                if (strongestSupply != null)
                {
                    Environment.SetEnvironmentVariable($"ZONE_STRONGEST_SUPPLY_{symbol}", 
                        $"{strongestSupply.Top:F2},{strongestSupply.Bottom:F2},{strongestSupply.Strength:F2}");
                }

                if (strongestDemand != null)
                {
                    Environment.SetEnvironmentVariable($"ZONE_STRONGEST_DEMAND_{symbol}", 
                        $"{strongestDemand.Top:F2},{strongestDemand.Bottom:F2},{strongestDemand.Strength:F2}");
                }

                // Check if current price is near a significant zone
                var nearZone = zones.SupplyZones.Concat(zones.DemandZones)
                    .Any(z => Math.Abs(zones.CurrentPrice - (z.Top + z.Bottom) / 2) < (z.Top - z.Bottom) * 1.5m);

                Environment.SetEnvironmentVariable($"ZONE_NEAR_SIGNIFICANT_{symbol}", nearZone.ToString());
                Environment.SetEnvironmentVariable($"ZONE_POC_{symbol}", zones.POC.ToString("F2"));

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying zone analysis for {Symbol}", symbol);
            }
        }

        /// <summary>
        /// Applies correlation analysis to risk management and position sizing
        /// </summary>
        private async Task ApplyCorrelationAnalysisAsync(CorrelationData correlations)
        {
            try
            {
                // Apply correlation-based risk adjustments
                foreach (var correlation in correlations.Correlations)
                {
                    var instrument = correlation.Key;
                    var corrValue = correlation.Value;

                    // High correlation (>0.8) means increased risk when holding both ES and NQ
                    if (instrument == "NQ" && Math.Abs(corrValue) > 0.8m)
                    {
                        Environment.SetEnvironmentVariable("ES_NQ_CORRELATION", corrValue.ToString("F3"));
                        Environment.SetEnvironmentVariable("HIGH_CORRELATION_RISK", "true");
                        
                        _logger.LogInformation("[LocalBotMechanic] High ES/NQ correlation detected: {Correlation:F3}", corrValue);
                    }
                }

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying correlation analysis");
            }
        }

        /// <summary>
        /// Applies sentiment analysis to bias and position sizing
        /// </summary>
        private async Task ApplySentimentAnalysisAsync(SentimentData sentiment)
        {
            try
            {
                // Apply sentiment-based bias
                if (Math.Abs(sentiment.OverallScore) > 0.6m)
                {
                    var bias = sentiment.OverallScore > 0 ? "BULLISH" : "BEARISH";
                    Environment.SetEnvironmentVariable("SENTIMENT_BIAS", bias);
                    Environment.SetEnvironmentVariable("SENTIMENT_STRENGTH", Math.Abs(sentiment.OverallScore).ToString("F2"));
                    
                    _logger.LogInformation("[LocalBotMechanic] Strong sentiment bias: {Bias} (strength: {Strength:F2})", 
                        bias, Math.Abs(sentiment.OverallScore));
                }
                else
                {
                    Environment.SetEnvironmentVariable("SENTIMENT_BIAS", "NEUTRAL");
                    Environment.SetEnvironmentVariable("SENTIMENT_STRENGTH", "0");
                }

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying sentiment analysis");
            }
        }
    }
}
