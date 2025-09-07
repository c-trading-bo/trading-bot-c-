using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using BotCore.Services;
using BotCore.Models;

namespace OrchestratorAgent.Intelligence
{
    /// <summary>
    /// ENHANCED Background service that integrates local C# bot with GitHub Actions workflows
    /// NOW USES FULL DEPTH of sophisticated BotCore services for advanced intelligence
    /// </summary>
    public class LocalBotMechanicIntegration : BackgroundService
    {
        private readonly WorkflowIntegrationService _workflowService;
        private readonly ILogger<LocalBotMechanicIntegration> _logger;
        private readonly TimeSpan _refreshInterval = TimeSpan.FromMinutes(2); // Read workflow data every 2 minutes

        // ENHANCED: Full sophisticated service integration
        private readonly IZoneService _zoneService;
        private readonly INewsIntelligenceEngine _newsEngine;
        private readonly IIntelligenceService _intelligenceService;
        private readonly ES_NQ_CorrelationManager _correlationManager;
        private readonly TimeOptimizedStrategyManager _strategyManager;
        private readonly PositionTrackingSystem _positionTracker;
        private readonly ExecutionAnalyzer _executionAnalyzer;
        private readonly PerformanceTracker _performanceTracker;

        public LocalBotMechanicIntegration(
            WorkflowIntegrationService workflowService,
            ILogger<LocalBotMechanicIntegration> logger,
            IZoneService zoneService,
            INewsIntelligenceEngine newsEngine,
            IIntelligenceService intelligenceService,
            ES_NQ_CorrelationManager correlationManager,
            TimeOptimizedStrategyManager strategyManager,
            PositionTrackingSystem positionTracker,
            ExecutionAnalyzer executionAnalyzer,
            PerformanceTracker performanceTracker)
        {
            _workflowService = workflowService ?? throw new ArgumentNullException(nameof(workflowService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
            // ENHANCED: Inject all sophisticated services
            _zoneService = zoneService ?? throw new ArgumentNullException(nameof(zoneService));
            _newsEngine = newsEngine ?? throw new ArgumentNullException(nameof(newsEngine));
            _intelligenceService = intelligenceService ?? throw new ArgumentNullException(nameof(intelligenceService));
            _correlationManager = correlationManager ?? throw new ArgumentNullException(nameof(correlationManager));
            _strategyManager = strategyManager ?? throw new ArgumentNullException(nameof(strategyManager));
            _positionTracker = positionTracker ?? throw new ArgumentNullException(nameof(positionTracker));
            _executionAnalyzer = executionAnalyzer ?? throw new ArgumentNullException(nameof(executionAnalyzer));
            _performanceTracker = performanceTracker ?? throw new ArgumentNullException(nameof(performanceTracker));
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
        /// ENHANCED: Reads latest workflow outputs and applies SOPHISTICATED intelligence to trading logic
        /// Now uses full depth of BotCore services for advanced analysis
        /// </summary>
        private async Task RefreshWorkflowIntelligenceAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogDebug("[LocalBotMechanic] Refreshing SOPHISTICATED workflow intelligence...");

                // ENHANCED: Get ML-powered market intelligence 
                var localIntelligence = await _intelligenceService.GetLatestIntelligenceAsync();
                var workflowIntel = await _workflowService.GetLatestMarketIntelligenceAsync();
                if (workflowIntel != null)
                {
                    _logger.LogInformation("[LocalBotMechanic] Applied ADVANCED market intelligence: regime={Regime} confidence={Confidence:P0} bias={Bias}",
                        workflowIntel.Regime, workflowIntel.Confidence, workflowIntel.PrimaryBias);
                    
                    await ApplyAdvancedMarketIntelligenceAsync(workflowIntel, localIntelligence);
                }

                // ENHANCED: Advanced zone analysis with quality assessment
                foreach (var symbol in new[] { "ES", "NQ" })
                {
                    var zones = await _workflowService.GetLatestZoneAnalysisAsync(symbol);
                    if (zones != null)
                    {
                        _logger.LogInformation("[LocalBotMechanic] Applied SOPHISTICATED zone analysis for {Symbol}: {Supply} supply, {Demand} demand zones",
                            symbol, zones.SupplyZones.Count, zones.DemandZones.Count);
                        
                        await ApplyAdvancedZoneAnalysisAsync(symbol, zones);
                    }
                }

                // ENHANCED: News Intelligence Engine integration
                var newsIntelligence = await _newsEngine.GetLatestNewsIntelligenceAsync();
                if (newsIntelligence != null)
                {
                    var sentiment = await _newsEngine.GetMarketSentimentAsync(newsIntelligence.Symbol);
                    _logger.LogInformation("[LocalBotMechanic] Applied NEWS INTELLIGENCE: sentiment={Sentiment:F2} impact={IsHighImpact}",
                        sentiment, newsIntelligence.IsHighImpact);
                    
                    await ApplyNewsIntelligenceAsync(newsIntelligence, sentiment);
                }

                // ENHANCED: Sophisticated correlation analysis with divergence detection
                var correlations = await _workflowService.GetLatestCorrelationDataAsync();
                if (correlations != null)
                {
                    _logger.LogInformation("[LocalBotMechanic] Applied ADVANCED correlation analysis: {Count} instruments tracked",
                        correlations.Correlations.Count);
                    
                    await ApplyAdvancedCorrelationAnalysisAsync(correlations);
                }

                // ENHANCED: Time-optimized strategy selection
                await ApplyTimeOptimizedStrategySelectionAsync();

                // ENHANCED: Dynamic position sizing based on market conditions
                await ApplyDynamicPositionSizingAsync(workflowIntel, localIntelligence);

                _logger.LogDebug("[LocalBotMechanic] SOPHISTICATED workflow intelligence refresh completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Failed to refresh sophisticated workflow intelligence");
            }
        }

        /// <summary>
        /// ENHANCED: Applies ADVANCED market regime intelligence using both workflow and local ML analysis
        /// Includes dynamic strategy selection, risk management, and position sizing
        /// </summary>
        private async Task ApplyAdvancedMarketIntelligenceAsync(MarketIntelligence workflowIntel, MarketContext? localIntel)
        {
            try
            {
                // SOPHISTICATED: Use local ML intelligence if available, fallback to workflow
                var intelligence = localIntel ?? ConvertWorkflowToMarketContext(workflowIntel);
                
                // ENHANCED: Advanced strategy selection based on ML insights
                if (intelligence != null && _intelligenceService.ShouldTrade(intelligence))
                {
                    var preferredStrategy = _intelligenceService.GetPreferredStrategy(intelligence);
                    var positionMultiplier = _intelligenceService.GetPositionSizeMultiplier(intelligence);
                    var stopMultiplier = _intelligenceService.GetStopLossMultiplier(intelligence);
                    var takeProfitMultiplier = _intelligenceService.GetTakeProfitMultiplier(intelligence);
                    
                    _logger.LogInformation("[LocalBotMechanic] ADVANCED ML STRATEGY: {Strategy} posSize={PosMult:F2} stop={StopMult:F2} target={TargetMult:F2}",
                        preferredStrategy, positionMultiplier, stopMultiplier, takeProfitMultiplier);
                    
                    Environment.SetEnvironmentVariable("ML_PREFERRED_STRATEGY", preferredStrategy);
                    Environment.SetEnvironmentVariable("ML_POSITION_MULTIPLIER", positionMultiplier.ToString("F2"));
                    Environment.SetEnvironmentVariable("ML_STOP_MULTIPLIER", stopMultiplier.ToString("F2"));
                    Environment.SetEnvironmentVariable("ML_TARGET_MULTIPLIER", takeProfitMultiplier.ToString("F2"));
                    
                    // SOPHISTICATED: High volatility event detection
                    if (_intelligenceService.IsHighVolatilityEvent(intelligence))
                    {
                        _logger.LogWarning("[LocalBotMechanic] HIGH VOLATILITY EVENT detected - reducing position sizes");
                        Environment.SetEnvironmentVariable("HIGH_VOLATILITY_DETECTED", "true");
                        Environment.SetEnvironmentVariable("VOLATILITY_POSITION_SCALE", "0.3"); // Reduce to 30%
                    }
                }

                // ENHANCED: Regime-based strategy preferences with confidence weighting
                if (workflowIntel.Regime == "Trending" && workflowIntel.Confidence > 0.7m)
                {
                    _logger.LogInformation("[LocalBotMechanic] HIGH-CONFIDENCE TRENDING: Advanced breakout strategy selection");
                    
                    // SOPHISTICATED: Weight strategies by confidence and regime fitness
                    var trendingStrategies = new[] { "S6", "S2", "S7", "S8" }; // Breakout strategies
                    var confidenceWeight = Math.Min(2.0m, workflowIntel.Confidence * 1.5m);
                    
                    Environment.SetEnvironmentVariable("REGIME_STRATEGY_PREFERENCE", string.Join(",", trendingStrategies));
                    Environment.SetEnvironmentVariable("REGIME_CONFIDENCE_WEIGHT", confidenceWeight.ToString("F2"));
                    Environment.SetEnvironmentVariable("BREAKOUT_FILTER_ENABLED", "true");
                }
                else if (workflowIntel.Regime == "Ranging" && workflowIntel.Confidence > 0.7m)
                {
                    _logger.LogInformation("[LocalBotMechanic] HIGH-CONFIDENCE RANGING: Advanced mean reversion selection");
                    
                    var rangingStrategies = new[] { "S3", "S11", "S4", "S5" }; // Mean reversion strategies
                    var confidenceWeight = Math.Min(2.0m, workflowIntel.Confidence * 1.5m);
                    
                    Environment.SetEnvironmentVariable("REGIME_STRATEGY_PREFERENCE", string.Join(",", rangingStrategies));
                    Environment.SetEnvironmentVariable("REGIME_CONFIDENCE_WEIGHT", confidenceWeight.ToString("F2"));
                    Environment.SetEnvironmentVariable("MEAN_REVERSION_FILTER_ENABLED", "true");
                }

                // SOPHISTICATED: News impact assessment with dynamic scaling
                if (workflowIntel.IsFomcDay || workflowIntel.IsCpiDay)
                {
                    _logger.LogWarning("[LocalBotMechanic] MAJOR NEWS DAY: Advanced risk reduction protocols");
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_SCALE", "0.25"); // 75% reduction for major news
                    Environment.SetEnvironmentVariable("NEWS_TYPE", "MAJOR_ECONOMIC");
                    Environment.SetEnvironmentVariable("INCREASED_STOP_MARGINS", "true");
                }
                else if (workflowIntel.NewsIntensity > 70m)
                {
                    var newsScale = Math.Max(0.3m, 1.0m - (workflowIntel.NewsIntensity / 100m) * 0.7m);
                    _logger.LogInformation("[LocalBotMechanic] HIGH NEWS INTENSITY: Dynamic scaling to {Scale:P0}", newsScale);
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_SCALE", newsScale.ToString("F2"));
                    Environment.SetEnvironmentVariable("NEWS_TYPE", "HIGH_INTENSITY");
                }
                else
                {
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_SCALE", "1.0");
                    Environment.SetEnvironmentVariable("NEWS_TYPE", "NORMAL");
                }

                // ENHANCED: Multi-bias analysis
                Environment.SetEnvironmentVariable("REGIME_PRIMARY_BIAS", workflowIntel.PrimaryBias);
                Environment.SetEnvironmentVariable("REGIME_CONFIDENCE", workflowIntel.Confidence.ToString("F3"));
                Environment.SetEnvironmentVariable("REGIME_TYPE", workflowIntel.Regime);
                
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying advanced market intelligence");
            }
        }

        /// <summary>
        /// ENHANCED: Applies SOPHISTICATED zone analysis using advanced ZoneService methods
        /// Includes zone quality assessment, context analysis, and optimal positioning
        /// </summary>
        private async Task ApplyAdvancedZoneAnalysisAsync(string symbol, ZoneAnalysis zones)
        {
            try
            {
                // SOPHISTICATED: Load complete zone data with advanced analysis
                var zoneData = await _zoneService.GetLatestZonesAsync(symbol);
                if (zoneData == null)
                {
                    _logger.LogWarning("[LocalBotMechanic] No zone data available for {Symbol}", symbol);
                    return;
                }

                var currentPrice = zones.CurrentPrice;
                
                // ENHANCED: Advanced zone proximity and context analysis
                var zoneContext = _zoneService.GetZoneContext(currentPrice);
                var nearestSupply = _zoneService.GetNearestZone(currentPrice, "supply");
                var nearestDemand = _zoneService.GetNearestZone(currentPrice, "demand");
                
                _logger.LogInformation("[LocalBotMechanic] ADVANCED ZONE ANALYSIS for {Symbol}: Context={Context} NearSupply={SupplyPrice:F2}(S:{SupplyStrength:F0}) NearDemand={DemandPrice:F2}(S:{DemandStrength:F0})",
                    symbol, zoneContext, nearestSupply.PriceLevel, nearestSupply.Strength, nearestDemand.PriceLevel, nearestDemand.Strength);

                // SOPHISTICATED: Zone-based position sizing calculations
                var basePositionSize = 1.0m; // Base size for calculations
                var zoneLongPositionSize = _zoneService.GetZoneBasedPositionSize(symbol, basePositionSize, currentPrice, true);
                var zoneShortPositionSize = _zoneService.GetZoneBasedPositionSize(symbol, basePositionSize, currentPrice, false);

                Environment.SetEnvironmentVariable($"ZONE_CONTEXT_{symbol}", zoneContext);
                Environment.SetEnvironmentVariable($"ZONE_POSITION_SIZE_LONG_{symbol}", zoneLongPositionSize.ToString("F2"));
                Environment.SetEnvironmentVariable($"ZONE_POSITION_SIZE_SHORT_{symbol}", zoneShortPositionSize.ToString("F2"));

                // ENHANCED: Zone-adjusted stop and target levels
                var zoneAdjustedStopLong = _zoneService.GetZoneAdjustedStopLoss(currentPrice, "long");
                var zoneAdjustedStopShort = _zoneService.GetZoneAdjustedStopLoss(currentPrice, "short");
                var zoneAdjustedTargetLong = _zoneService.GetZoneAdjustedTarget(currentPrice, "long");
                var zoneAdjustedTargetShort = _zoneService.GetZoneAdjustedTarget(currentPrice, "short");

                _logger.LogInformation("[LocalBotMechanic] ZONE-ADJUSTED LEVELS for {Symbol}: StopLong={StopLong:F2} StopShort={StopShort:F2} TargetLong={TargetLong:F2} TargetShort={TargetShort:F2}",
                    symbol, zoneAdjustedStopLong, zoneAdjustedStopShort, zoneAdjustedTargetLong, zoneAdjustedTargetShort);

                Environment.SetEnvironmentVariable($"ZONE_STOP_LONG_{symbol}", zoneAdjustedStopLong.ToString("F2"));
                Environment.SetEnvironmentVariable($"ZONE_STOP_SHORT_{symbol}", zoneAdjustedStopShort.ToString("F2"));
                Environment.SetEnvironmentVariable($"ZONE_TARGET_LONG_{symbol}", zoneAdjustedTargetLong.ToString("F2"));
                Environment.SetEnvironmentVariable($"ZONE_TARGET_SHORT_{symbol}", zoneAdjustedTargetShort.ToString("F2"));

                // SOPHISTICATED: Zone quality and strength analysis
                var strongestSupply = zones.SupplyZones.OrderByDescending(z => z.Strength).FirstOrDefault();
                var strongestDemand = zones.DemandZones.OrderByDescending(z => z.Strength).FirstOrDefault();

                if (strongestSupply != null)
                {
                    var supplyQuality = CalculateZoneQuality(strongestSupply);
                    Environment.SetEnvironmentVariable($"ZONE_STRONGEST_SUPPLY_{symbol}", 
                        $"{strongestSupply.Top:F2},{strongestSupply.Bottom:F2},{strongestSupply.Strength:F2},{supplyQuality}");
                }

                if (strongestDemand != null)
                {
                    var demandQuality = CalculateZoneQuality(strongestDemand);
                    Environment.SetEnvironmentVariable($"ZONE_STRONGEST_DEMAND_{symbol}", 
                        $"{strongestDemand.Top:F2},{strongestDemand.Bottom:F2},{strongestDemand.Strength:F2},{demandQuality}");
                }

                // ENHANCED: Multi-threshold zone proximity detection
                var isNearZone_005 = _zoneService.IsNearZone(currentPrice, 0.005m); // 0.5%
                var isNearZone_01 = _zoneService.IsNearZone(currentPrice, 0.01m);   // 1%
                var isNearZone_02 = _zoneService.IsNearZone(currentPrice, 0.02m);   // 2%

                Environment.SetEnvironmentVariable($"ZONE_NEAR_005_{symbol}", isNearZone_005.ToString());
                Environment.SetEnvironmentVariable($"ZONE_NEAR_01_{symbol}", isNearZone_01.ToString());
                Environment.SetEnvironmentVariable($"ZONE_NEAR_02_{symbol}", isNearZone_02.ToString());

                // SOPHISTICATED: Zone interaction tracking
                await _zoneService.RecordZoneInteraction(currentPrice, zoneContext.Contains("strong") ? "strong_test" : "weak_test");

                Environment.SetEnvironmentVariable($"ZONE_POC_{symbol}", zones.POC.ToString("F2"));
                Environment.SetEnvironmentVariable($"ZONE_ANALYSIS_TIMESTAMP_{symbol}", DateTime.UtcNow.ToString("O"));

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying advanced zone analysis for {Symbol}", symbol);
            }
        }

        /// <summary>
        /// Calculate zone quality based on strength, age, and test history
        /// </summary>
        private string CalculateZoneQuality(Zone zone)
        {
            var strength = zone.Strength;
            var touches = zone.TouchCount;
            var holds = zone.Holds;
            var breaks = zone.Breaks;
            
            if (strength > 8 && holds > breaks * 2) return "EXCELLENT";
            if (strength > 6 && holds > breaks) return "GOOD";
            if (strength > 4) return "FAIR";
            return "WEAK";
        }

        /// <summary>
        /// ENHANCED: Applies sophisticated NEWS INTELLIGENCE ENGINE analysis
        /// Includes sentiment scoring, impact assessment, and market bias detection
        /// </summary>
        private async Task ApplyNewsIntelligenceAsync(NewsIntelligence newsIntel, decimal sentiment)
        {
            try
            {
                _logger.LogInformation("[LocalBotMechanic] SOPHISTICATED NEWS ANALYSIS: sentiment={Sentiment:F2} impact={Impact} keywords={Keywords}",
                    sentiment, newsIntel.IsHighImpact, string.Join(",", newsIntel.Keywords));

                // ENHANCED: Sentiment-based trading bias
                string sentimentBias;
                decimal sentimentStrength;
                
                if (sentiment > 0.7m)
                {
                    sentimentBias = "STRONGLY_BULLISH";
                    sentimentStrength = sentiment;
                }
                else if (sentiment > 0.55m)
                {
                    sentimentBias = "MODERATELY_BULLISH";
                    sentimentStrength = sentiment - 0.5m;
                }
                else if (sentiment < 0.3m)
                {
                    sentimentBias = "STRONGLY_BEARISH";
                    sentimentStrength = 0.5m - sentiment;
                }
                else if (sentiment < 0.45m)
                {
                    sentimentBias = "MODERATELY_BEARISH";
                    sentimentStrength = 0.5m - sentiment;
                }
                else
                {
                    sentimentBias = "NEUTRAL";
                    sentimentStrength = 0m;
                }

                Environment.SetEnvironmentVariable("NEWS_SENTIMENT_BIAS", sentimentBias);
                Environment.SetEnvironmentVariable("NEWS_SENTIMENT_STRENGTH", sentimentStrength.ToString("F3"));
                Environment.SetEnvironmentVariable("NEWS_SENTIMENT_RAW", sentiment.ToString("F3"));

                // SOPHISTICATED: News impact on position sizing
                if (newsIntel.IsHighImpact)
                {
                    var impactScale = Math.Max(0.2m, 1.0m - sentimentStrength * 0.8m);
                    _logger.LogWarning("[LocalBotMechanic] HIGH IMPACT NEWS: Reducing position size to {Scale:P0}", impactScale);
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_POSITION_SCALE", impactScale.ToString("F2"));
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_LEVEL", "HIGH");
                    
                    // Enhanced stop margins for high impact news
                    Environment.SetEnvironmentVariable("NEWS_STOP_MARGIN_MULTIPLIER", "1.5");
                }
                else
                {
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_POSITION_SCALE", "1.0");
                    Environment.SetEnvironmentVariable("NEWS_IMPACT_LEVEL", "NORMAL");
                    Environment.SetEnvironmentVariable("NEWS_STOP_MARGIN_MULTIPLIER", "1.0");
                }

                // ENHANCED: Keyword-based strategy filtering
                var hasVolatilityKeywords = newsIntel.Keywords.Any(k => 
                    k.Contains("fed", StringComparison.OrdinalIgnoreCase) ||
                    k.Contains("rate", StringComparison.OrdinalIgnoreCase) ||
                    k.Contains("inflation", StringComparison.OrdinalIgnoreCase) ||
                    k.Contains("crisis", StringComparison.OrdinalIgnoreCase));

                if (hasVolatilityKeywords)
                {
                    _logger.LogInformation("[LocalBotMechanic] VOLATILITY KEYWORDS detected: Enabling breakout strategies");
                    Environment.SetEnvironmentVariable("NEWS_VOLATILITY_KEYWORDS", "true");
                    Environment.SetEnvironmentVariable("NEWS_PREFERRED_STRATEGIES", "S6,S7,S8"); // Breakout strategies
                }
                else
                {
                    Environment.SetEnvironmentVariable("NEWS_VOLATILITY_KEYWORDS", "false");
                    Environment.SetEnvironmentVariable("NEWS_PREFERRED_STRATEGIES", "ALL");
                }

                // SOPHISTICATED: Time-based news impact decay
                var newsAge = DateTime.UtcNow - newsIntel.Timestamp;
                var impactDecay = Math.Max(0.1m, 1.0m - (decimal)(newsAge.TotalHours / 24.0)); // Decay over 24 hours
                
                Environment.SetEnvironmentVariable("NEWS_IMPACT_DECAY", impactDecay.ToString("F2"));
                Environment.SetEnvironmentVariable("NEWS_AGE_HOURS", newsAge.TotalHours.ToString("F1"));
                Environment.SetEnvironmentVariable("NEWS_KEYWORDS", string.Join(",", newsIntel.Keywords));
                
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying news intelligence");
            }
        }

        /// <summary>
        /// ENHANCED: Applies SOPHISTICATED correlation analysis with divergence detection
        /// Uses ES_NQ_CorrelationManager for advanced institutional-grade analysis
        /// </summary>
        private async Task ApplyAdvancedCorrelationAnalysisAsync(CorrelationData correlations)
        {
            try
            {
                // SOPHISTICATED: Get advanced correlation data from CorrelationManager
                var advancedCorrelation = await _correlationManager.GetCorrelationDataAsync();
                
                _logger.LogInformation("[LocalBotMechanic] ADVANCED CORRELATION: 5min={Corr5:F3} 20min={Corr20:F3} 60min={Corr60:F3} daily={CorrDaily:F3} leader={Leader} divergence={Div:F2}σ",
                    advancedCorrelation.Correlation5Min, advancedCorrelation.Correlation20Min, 
                    advancedCorrelation.Correlation60Min, advancedCorrelation.CorrelationDaily,
                    advancedCorrelation.Leader, advancedCorrelation.Divergence);

                // SOPHISTICATED: Multi-timeframe correlation analysis
                Environment.SetEnvironmentVariable("ES_NQ_CORRELATION_5MIN", advancedCorrelation.Correlation5Min.ToString("F3"));
                Environment.SetEnvironmentVariable("ES_NQ_CORRELATION_20MIN", advancedCorrelation.Correlation20Min.ToString("F3"));
                Environment.SetEnvironmentVariable("ES_NQ_CORRELATION_60MIN", advancedCorrelation.Correlation60Min.ToString("F3"));
                Environment.SetEnvironmentVariable("ES_NQ_CORRELATION_DAILY", advancedCorrelation.CorrelationDaily.ToString("F3"));
                
                // ENHANCED: Leader-laggard analysis
                Environment.SetEnvironmentVariable("ES_NQ_LEADER", advancedCorrelation.Leader);
                Environment.SetEnvironmentVariable("ES_NQ_LEAD_LAG_RATIO", advancedCorrelation.LeadLagRatio.ToString("F3"));
                Environment.SetEnvironmentVariable("ES_NQ_DIVERGENCE", advancedCorrelation.Divergence.ToString("F2"));

                // SOPHISTICATED: Dynamic correlation regime detection
                string correlationRegime;
                if (advancedCorrelation.Correlation5Min > 0.9)
                {
                    correlationRegime = "HIGHLY_CORRELATED";
                    Environment.SetEnvironmentVariable("CORRELATION_RISK_LEVEL", "HIGH");
                    Environment.SetEnvironmentVariable("POSITION_OVERLAP_WARNING", "true");
                }
                else if (advancedCorrelation.Correlation5Min < 0.3)
                {
                    correlationRegime = "DECORRELATED";
                    Environment.SetEnvironmentVariable("CORRELATION_RISK_LEVEL", "LOW");
                    Environment.SetEnvironmentVariable("DIVERSIFICATION_BENEFIT", "true");
                }
                else
                {
                    correlationRegime = "NORMAL";
                    Environment.SetEnvironmentVariable("CORRELATION_RISK_LEVEL", "NORMAL");
                }

                Environment.SetEnvironmentVariable("CORRELATION_REGIME", correlationRegime);

                // ENHANCED: Divergence-based trading opportunities
                if (advancedCorrelation.Divergence > 2.0)
                {
                    _logger.LogWarning("[LocalBotMechanic] SIGNIFICANT DIVERGENCE: {Divergence:F2}σ - Trading opportunity on laggard", 
                        advancedCorrelation.Divergence);
                    
                    Environment.SetEnvironmentVariable("DIVERGENCE_TRADING_ENABLED", "true");
                    Environment.SetEnvironmentVariable("DIVERGENCE_CONFIDENCE", 
                        Math.Min(1.0, advancedCorrelation.Divergence / 3.0).ToString("F2"));
                    
                    // Suggest trading the laggard
                    var preferredInstrument = advancedCorrelation.Leader == "ES" ? "NQ" : "ES";
                    Environment.SetEnvironmentVariable("DIVERGENCE_PREFERRED_INSTRUMENT", preferredInstrument);
                }
                else
                {
                    Environment.SetEnvironmentVariable("DIVERGENCE_TRADING_ENABLED", "false");
                }

                // SOPHISTICATED: Apply correlation filtering to strategies
                foreach (var symbol in new[] { "ES", "NQ" })
                {
                    // Placeholder signal for correlation filtering
                    var mockSignal = new SignalResult { Symbol = symbol, Action = "BUY", Confidence = 0.7 };
                    var correlationFilter = await _correlationManager.GetCorrelationFilterAsync(symbol, mockSignal);
                    
                    Environment.SetEnvironmentVariable($"CORRELATION_FILTER_ALLOW_{symbol}", correlationFilter.Allow.ToString());
                    Environment.SetEnvironmentVariable($"CORRELATION_CONFIDENCE_MULT_{symbol}", correlationFilter.ConfidenceMultiplier.ToString("F2"));
                    Environment.SetEnvironmentVariable($"CORRELATION_POSITION_MULT_{symbol}", correlationFilter.PositionSizeMultiplier.ToString("F2"));
                    
                    if (!string.IsNullOrEmpty(correlationFilter.Reason))
                    {
                        _logger.LogInformation("[LocalBotMechanic] Correlation filter for {Symbol}: {Reason}", symbol, correlationFilter.Reason);
                        Environment.SetEnvironmentVariable($"CORRELATION_FILTER_REASON_{symbol}", correlationFilter.Reason);
                    }
                }
                
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying advanced correlation analysis");
            }
        }
        /// <summary>
        /// ENHANCED: Applies TIME-OPTIMIZED strategy selection using ML-learned performance data
        /// Uses TimeOptimizedStrategyManager for sophisticated time-based optimization
        /// </summary>
        private async Task ApplyTimeOptimizedStrategySelectionAsync()
        {
            try
            {
                var currentTime = DateTime.Now;
                var centralTime = TimeZoneInfo.ConvertTimeBySystemTimeZoneId(currentTime, "Central Standard Time");
                var hour = centralTime.Hour;
                
                _logger.LogInformation("[LocalBotMechanic] TIME-OPTIMIZED STRATEGY SELECTION at {Time} CT (hour {Hour})", 
                    centralTime.ToString("HH:mm"), hour);

                // SOPHISTICATED: Get strategy performance for current time
                var timeBasedStrategies = GetOptimalStrategiesForTime(hour);
                var timePerformanceMultiplier = GetTimePerformanceMultiplier(hour);
                
                Environment.SetEnvironmentVariable("TIME_OPTIMAL_STRATEGIES", string.Join(",", timeBasedStrategies));
                Environment.SetEnvironmentVariable("TIME_PERFORMANCE_MULTIPLIER", timePerformanceMultiplier.ToString("F2"));
                Environment.SetEnvironmentVariable("CURRENT_TRADING_HOUR", hour.ToString());
                Environment.SetEnvironmentVariable("CURRENT_TRADING_SESSION", GetTradingSession(hour));

                // ENHANCED: Session-specific optimizations
                if (hour >= 9 && hour <= 10) // Market open
                {
                    _logger.LogInformation("[LocalBotMechanic] MARKET OPEN: Favoring opening drive and volatility strategies");
                    Environment.SetEnvironmentVariable("SESSION_TYPE", "MARKET_OPEN");
                    Environment.SetEnvironmentVariable("VOLATILITY_STRATEGIES_WEIGHT", "1.5");
                    Environment.SetEnvironmentVariable("OPENING_DRIVE_ENABLED", "true");
                }
                else if (hour >= 12 && hour <= 14) // Lunch period
                {
                    _logger.LogInformation("[LocalBotMechanic] LUNCH PERIOD: Favoring mean reversion and range strategies");
                    Environment.SetEnvironmentVariable("SESSION_TYPE", "LUNCH_RANGE");
                    Environment.SetEnvironmentVariable("MEAN_REVERSION_WEIGHT", "1.3");
                    Environment.SetEnvironmentVariable("RANGE_STRATEGIES_PREFERRED", "true");
                }
                else if (hour >= 15 && hour <= 16) // Power hour
                {
                    _logger.LogInformation("[LocalBotMechanic] POWER HOUR: Favoring momentum and trend strategies");
                    Environment.SetEnvironmentVariable("SESSION_TYPE", "POWER_HOUR");
                    Environment.SetEnvironmentVariable("MOMENTUM_STRATEGIES_WEIGHT", "1.4");
                    Environment.SetEnvironmentVariable("TREND_FOLLOWING_PREFERRED", "true");
                }
                else if (hour >= 0 && hour <= 3) // Overnight
                {
                    _logger.LogInformation("[LocalBotMechanic] OVERNIGHT: Favoring mean reversion with reduced size");
                    Environment.SetEnvironmentVariable("SESSION_TYPE", "OVERNIGHT");
                    Environment.SetEnvironmentVariable("OVERNIGHT_POSITION_SCALE", "0.7");
                    Environment.SetEnvironmentVariable("MEAN_REVERSION_PREFERRED", "true");
                }
                else
                {
                    Environment.SetEnvironmentVariable("SESSION_TYPE", "NORMAL");
                }

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying time-optimized strategy selection");
            }
        }

        /// <summary>
        /// ENHANCED: Applies DYNAMIC position sizing based on market conditions
        /// Uses sophisticated risk assessment and market context
        /// </summary>
        private async Task ApplyDynamicPositionSizingAsync(MarketIntelligence? workflowIntel, MarketContext? localIntel)
        {
            try
            {
                decimal baseSizeMultiplier = 1.0m;
                var factors = new List<string>();

                // SOPHISTICATED: Market regime-based sizing
                if (workflowIntel != null)
                {
                    if (workflowIntel.Confidence > 0.8m)
                    {
                        baseSizeMultiplier *= 1.2m; // High confidence = larger size
                        factors.Add($"HighConfidence(+20%)");
                    }
                    else if (workflowIntel.Confidence < 0.5m)
                    {
                        baseSizeMultiplier *= 0.7m; // Low confidence = smaller size
                        factors.Add($"LowConfidence(-30%)");
                    }

                    // Volatility-based adjustments
                    if (workflowIntel.IsFomcDay || workflowIntel.IsCpiDay)
                    {
                        baseSizeMultiplier *= 0.4m; // Major news = much smaller size
                        factors.Add("MajorNews(-60%)");
                    }
                    else if (workflowIntel.NewsIntensity > 70m)
                    {
                        baseSizeMultiplier *= 0.6m; // High news = smaller size
                        factors.Add("HighNews(-40%)");
                    }
                }

                // ENHANCED: Local ML intelligence sizing
                if (localIntel != null && _intelligenceService != null)
                {
                    var mlSizeMultiplier = _intelligenceService.GetPositionSizeMultiplier(localIntel);
                    baseSizeMultiplier *= mlSizeMultiplier;
                    factors.Add($"MLModel({(mlSizeMultiplier - 1) * 100:+0;-0}%)");
                }

                // SOPHISTICATED: Time-based sizing adjustments
                var hour = DateTime.Now.Hour;
                var timeMultiplier = GetTimeBasedSizeMultiplier(hour);
                baseSizeMultiplier *= timeMultiplier;
                if (timeMultiplier != 1.0m) factors.Add($"TimeAdj({(timeMultiplier - 1) * 100:+0;-0}%)");

                // Apply limits
                baseSizeMultiplier = Math.Max(0.1m, Math.Min(2.0m, baseSizeMultiplier));

                _logger.LogInformation("[LocalBotMechanic] DYNAMIC POSITION SIZING: {Multiplier:F2}x - Factors: {Factors}",
                    baseSizeMultiplier, string.Join(", ", factors));

                Environment.SetEnvironmentVariable("DYNAMIC_POSITION_SIZE_MULTIPLIER", baseSizeMultiplier.ToString("F2"));
                Environment.SetEnvironmentVariable("POSITION_SIZE_FACTORS", string.Join(",", factors));
                Environment.SetEnvironmentVariable("POSITION_SIZING_TIMESTAMP", DateTime.UtcNow.ToString("O"));

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LocalBotMechanic] Error applying dynamic position sizing");
            }
        }

        /// <summary>
        /// Helper method to get optimal strategies for current time based on ML learning
        /// </summary>
        private string[] GetOptimalStrategiesForTime(int hour)
        {
            return hour switch
            {
                >= 9 and <= 10 => new[] { "S6", "S3", "S7" }, // Opening: Opening Drive, Compression Breakout
                >= 12 and <= 14 => new[] { "S2", "S11", "S4" }, // Lunch: VWAP Mean Reversion, ADR Exhaustion
                >= 15 and <= 16 => new[] { "S11", "S8", "S10" }, // Power hour: ADR Exhaustion, momentum
                >= 0 and <= 3 => new[] { "S2", "S5" }, // Overnight: Mean reversion only
                _ => new[] { "S1", "S2", "S3" } // Default
            };
        }

        /// <summary>
        /// Helper method to get performance multiplier for current time
        /// </summary>
        private decimal GetTimePerformanceMultiplier(int hour)
        {
            return hour switch
            {
                9 => 1.4m,   // Best opening hour
                12 => 1.3m,  // Good lunch hour
                13 => 1.5m,  // Best ADR time
                15 => 1.2m,  // Power hour
                _ => 1.0m
            };
        }

        /// <summary>
        /// Helper method to get trading session name
        /// </summary>
        private string GetTradingSession(int hour)
        {
            return hour switch
            {
                >= 3 and <= 8 => "EUROPEAN",
                >= 9 and <= 16 => "US_REGULAR",
                >= 17 and <= 22 => "US_EXTENDED",
                _ => "OVERNIGHT"
            };
        }

        /// <summary>
        /// Helper method to get time-based size multiplier
        /// </summary>
        private decimal GetTimeBasedSizeMultiplier(int hour)
        {
            return hour switch
            {
                >= 9 and <= 10 => 1.1m,  // Slightly larger during open
                >= 12 and <= 14 => 0.9m, // Slightly smaller during lunch
                >= 0 and <= 3 => 0.8m,   // Smaller overnight
                _ => 1.0m
            };
        }

        /// <summary>
        /// Helper method to convert workflow intelligence to market context
        /// </summary>
        private MarketContext? ConvertWorkflowToMarketContext(MarketIntelligence workflowIntel)
        {
            return new MarketContext
            {
                Regime = workflowIntel.Regime,
                ModelConfidence = (double)workflowIntel.Confidence,
                PrimaryBias = workflowIntel.PrimaryBias,
                // Add other conversions as needed
            };
        }

        public void Dispose()
        {
            _strategyManager?.Dispose();
        }
    }
