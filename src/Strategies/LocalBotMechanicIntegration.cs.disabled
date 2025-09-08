using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Services;
using BotCore.Models;

namespace Intelligence
{
    /// <summary>
    /// Complete Bot-Intelligence Integration Bridge
    /// Connects GitHub Actions workflows → Intelligence data → Strategy-specific routing → Environment variables → AllStrategies.cs execution
    /// Preserves existing trading logic while enhancing with intelligence
    /// </summary>
    public class LocalBotMechanicIntegration
    {
        private readonly ILogger<LocalBotMechanicIntegration> _logger;
        private readonly IIntelligenceService _intelligenceService;
        private readonly StrategyClassifier _strategyClassifier;
        private readonly Dictionary<string, StrategyCategory> _strategyMapping;
        private MarketContext? _currentIntelligence;
        private readonly object _lock = new();

        public LocalBotMechanicIntegration(
            ILogger<LocalBotMechanicIntegration> logger,
            IIntelligenceService intelligenceService,
            StrategyClassifier strategyClassifier)
        {
            _logger = logger;
            _intelligenceService = intelligenceService;
            _strategyClassifier = strategyClassifier;
            
            // Initialize strategy mapping from StrategyClassifier
            _strategyMapping = new Dictionary<string, StrategyCategory>(StringComparer.OrdinalIgnoreCase);
            InitializeStrategyMapping();
        }

        private void InitializeStrategyMapping()
        {
            // Map all 14 strategies to their categories for intelligence routing
            var strategies = new[] { "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14" };
            
            foreach (var strategy in strategies)
            {
                _strategyMapping[strategy] = _strategyClassifier.GetCategory(strategy);
            }
            
            _logger.LogInformation("[INTEGRATION] Initialized strategy mapping for {Count} strategies", strategies.Length);
        }

        /// <summary>
        /// Main integration method - Call this BEFORE strategy execution in OrchestratorAgent
        /// Loads intelligence and populates environment variables for strategy consumption
        /// </summary>
        public async Task<bool> UpdateIntelligenceEnvironmentAsync()
        {
            try
            {
                // Load latest intelligence from GitHub Actions workflows
                var intelligence = await _intelligenceService.GetLatestIntelligenceAsync();
                
                lock (_lock)
                {
                    _currentIntelligence = intelligence;
                }

                if (intelligence == null)
                {
                    _logger.LogDebug("[INTEGRATION] No intelligence available - strategies will use defaults");
                    // Clear intelligence environment variables but preserve existing ones
                    ClearIntelligenceEnvironmentVariables();
                    return false;
                }

                // Log intelligence context for debugging
                _logger.LogInformation("[INTEGRATION] Intelligence loaded: Regime={Regime}, Confidence={Confidence:P0}, Bias={Bias}, NewsIntensity={News:F1}",
                    intelligence.Regime, intelligence.ModelConfidence, intelligence.PrimaryBias, intelligence.NewsIntensity);

                // Route intelligence to strategy-specific environment variables
                await RouteIntelligenceToStrategiesAsync(intelligence);

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[INTEGRATION] Failed to update intelligence environment");
                return false;
            }
        }

        /// <summary>
        /// Routes intelligence data to strategy-specific environment variables
        /// Maps intelligence → strategy categories → specific environment variables → AllStrategies.cs consumption
        /// </summary>
        private async Task RouteIntelligenceToStrategiesAsync(MarketContext intelligence)
        {
            try
            {
                // Global intelligence environment variables (used by AllStrategies.cs)
                SetEnvironmentVariable("INTELLIGENCE_REGIME", intelligence.Regime);
                SetEnvironmentVariable("INTELLIGENCE_CONFIDENCE", intelligence.ModelConfidence.ToString("F2"));
                SetEnvironmentVariable("INTELLIGENCE_BIAS", intelligence.PrimaryBias);
                SetEnvironmentVariable("INTELLIGENCE_NEWS_INTENSITY", intelligence.NewsIntensity.ToString("F1"));
                SetEnvironmentVariable("INTELLIGENCE_AVAILABLE", "true");
                SetEnvironmentVariable("INTELLIGENCE_TIMESTAMP", DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"));

                // Major event flags
                SetEnvironmentVariable("FOMC_DAY", intelligence.IsFomcDay.ToString().ToLowerInvariant());
                SetEnvironmentVariable("CPI_DAY", intelligence.IsCpiDay.ToString().ToLowerInvariant());
                SetEnvironmentVariable("HIGH_VOLATILITY_EVENT", _intelligenceService.IsHighVolatilityEvent(intelligence).ToString().ToLowerInvariant());

                // Position sizing multipliers (used by risk management)
                var positionMultiplier = _intelligenceService.GetPositionSizeMultiplier(intelligence);
                var stopMultiplier = _intelligenceService.GetStopLossMultiplier(intelligence);
                var targetMultiplier = _intelligenceService.GetTakeProfitMultiplier(intelligence);

                SetEnvironmentVariable("POSITION_SIZE_MULTIPLIER", positionMultiplier.ToString("F2"));
                SetEnvironmentVariable("STOP_LOSS_MULTIPLIER", stopMultiplier.ToString("F2"));
                SetEnvironmentVariable("TAKE_PROFIT_MULTIPLIER", targetMultiplier.ToString("F2"));

                // Preferred strategy from intelligence
                var preferredStrategy = _intelligenceService.GetPreferredStrategy(intelligence);
                SetEnvironmentVariable("PREFERRED_STRATEGY", preferredStrategy);

                // Strategy-category-specific routing
                await RouteToBreakoutStrategiesAsync(intelligence);
                await RouteToMeanReversionStrategiesAsync(intelligence);
                await RouteToMomentumStrategiesAsync(intelligence);
                await RouteToScalpingStrategiesAsync(intelligence);

                // Quality score thresholds based on intelligence
                SetQualityThresholdsFromIntelligence(intelligence);

                _logger.LogDebug("[INTEGRATION] Routed intelligence to {Count} strategy categories", 4);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[INTEGRATION] Failed to route intelligence to strategies");
            }
        }

        /// <summary>
        /// Routes intelligence to Breakout strategies: S6, S8, S2
        /// </summary>
        private async Task RouteToBreakoutStrategiesAsync(MarketContext intelligence)
        {
            var breakoutMultiplier = 1.0m;
            
            // Breakout strategies perform better in trending regimes
            if (intelligence.Regime.Equals("Trending", StringComparison.OrdinalIgnoreCase))
            {
                breakoutMultiplier = 1.3m;
            }
            else if (intelligence.Regime.Equals("Ranging", StringComparison.OrdinalIgnoreCase))
            {
                breakoutMultiplier = 0.7m; // Reduce breakout aggression in ranging markets
            }

            // High confidence breakouts
            if (intelligence.ModelConfidence >= 0.8m && intelligence.Regime.Equals("Trending", StringComparison.OrdinalIgnoreCase))
            {
                breakoutMultiplier *= 1.5m;
            }

            SetEnvironmentVariable("BREAKOUT_MULTIPLIER", breakoutMultiplier.ToString("F2"));
            SetEnvironmentVariable("S6_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.7m).ToString().ToLowerInvariant());
            SetEnvironmentVariable("S8_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.7m).ToString().ToLowerInvariant());
            
            // S2-specific adjustments (mean reversion with breakout elements)
            var s2Sigma = intelligence.Regime.Equals("Volatile", StringComparison.OrdinalIgnoreCase) ? 2.5m : 2.0m;
            SetEnvironmentVariable("S2_SIGMA_ADJUSTMENT", s2Sigma.ToString("F1"));
        }

        /// <summary>
        /// Routes intelligence to Mean Reversion strategies: S3, S11, S5, S7
        /// </summary>
        private async Task RouteToMeanReversionStrategiesAsync(MarketContext intelligence)
        {
            var meanReversionMultiplier = 1.0m;
            
            // Mean reversion works best in ranging markets
            if (intelligence.Regime.Equals("Ranging", StringComparison.OrdinalIgnoreCase))
            {
                meanReversionMultiplier = 1.4m;
            }
            else if (intelligence.Regime.Equals("Trending", StringComparison.OrdinalIgnoreCase))
            {
                meanReversionMultiplier = 0.6m; // Reduce mean reversion in trending markets
            }

            // Volatile regimes can have quick mean reversions
            if (intelligence.Regime.Equals("Volatile", StringComparison.OrdinalIgnoreCase))
            {
                meanReversionMultiplier = 1.2m;
            }

            SetEnvironmentVariable("MEAN_REVERSION_MULTIPLIER", meanReversionMultiplier.ToString("F2"));
            SetEnvironmentVariable("S3_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.6m).ToString().ToLowerInvariant());
            SetEnvironmentVariable("S11_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.6m).ToString().ToLowerInvariant());
            
            // News-based adjustments for mean reversion
            if (intelligence.NewsIntensity >= 70m)
            {
                SetEnvironmentVariable("NEWS_FADE_OPPORTUNITY", "true");
            }
        }

        /// <summary>
        /// Routes intelligence to Momentum strategies: S1, S9, S10, S4
        /// </summary>
        private async Task RouteToMomentumStrategiesAsync(MarketContext intelligence)
        {
            var momentumMultiplier = 1.0m;
            
            // Momentum strategies excel in trending regimes
            if (intelligence.Regime.Equals("Trending", StringComparison.OrdinalIgnoreCase))
            {
                momentumMultiplier = 1.6m;
            }
            else if (intelligence.Regime.Equals("Ranging", StringComparison.OrdinalIgnoreCase))
            {
                momentumMultiplier = 0.5m; // Momentum struggles in ranges
            }

            // High confidence momentum
            if (intelligence.ModelConfidence >= 0.8m && intelligence.Regime.Equals("Trending", StringComparison.OrdinalIgnoreCase))
            {
                momentumMultiplier *= 1.8m;
            }

            SetEnvironmentVariable("MOMENTUM_MULTIPLIER", momentumMultiplier.ToString("F2"));
            SetEnvironmentVariable("S1_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.7m).ToString().ToLowerInvariant());
            SetEnvironmentVariable("S9_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.7m).ToString().ToLowerInvariant());
            SetEnvironmentVariable("S10_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.7m).ToString().ToLowerInvariant());
            SetEnvironmentVariable("S4_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.7m).ToString().ToLowerInvariant());
            
            // Trend strength indicator
            var trendStrength = intelligence.ModelConfidence * (intelligence.Regime.Equals("Trending", StringComparison.OrdinalIgnoreCase) ? 1.5m : 1.0m);
            SetEnvironmentVariable("TREND_STRENGTH", trendStrength.ToString("F2"));
        }

        /// <summary>
        /// Routes intelligence to Scalping strategies: S12, S13, S14
        /// </summary>
        private async Task RouteToScalpingStrategiesAsync(MarketContext intelligence)
        {
            var scalpingMultiplier = 1.0m;
            
            // Scalping works well in all regimes but needs adjustment
            if (intelligence.Regime.Equals("Volatile", StringComparison.OrdinalIgnoreCase))
            {
                scalpingMultiplier = 1.3m; // More opportunities in volatile markets
            }
            else if (intelligence.NewsIntensity >= 80m)
            {
                scalpingMultiplier = 0.4m; // Reduce scalping during high news
            }

            SetEnvironmentVariable("SCALPING_MULTIPLIER", scalpingMultiplier.ToString("F2"));
            SetEnvironmentVariable("S12_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.5m).ToString().ToLowerInvariant());
            SetEnvironmentVariable("S13_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.5m).ToString().ToLowerInvariant());
            SetEnvironmentVariable("S14_INTELLIGENCE_BOOST", (intelligence.ModelConfidence >= 0.5m).ToString().ToLowerInvariant());
            
            // Scalping-specific adjustments
            var scalpingAtrMultiplier = intelligence.Regime.Equals("Volatile", StringComparison.OrdinalIgnoreCase) ? 0.8m : 1.0m;
            SetEnvironmentVariable("SCALPING_ATR_MULTIPLIER", scalpingAtrMultiplier.ToString("F2"));
        }

        /// <summary>
        /// Sets quality score thresholds based on intelligence confidence
        /// These are consumed by AllStrategies.cs generate_candidates methods
        /// </summary>
        private void SetQualityThresholdsFromIntelligence(MarketContext intelligence)
        {
            // Base thresholds adjusted by intelligence confidence
            var baseNight = 0.80m;
            var baseOpen = 0.85m;
            var baseRth = 0.75m;

            // Higher confidence = lower thresholds (more trades)
            var confidenceAdjustment = (intelligence.ModelConfidence - 0.5m) * 0.3m;
            
            var qthNight = Math.Max(0.6m, baseNight - confidenceAdjustment);
            var qthOpen = Math.Max(0.65m, baseOpen - confidenceAdjustment);
            var qthRth = Math.Max(0.55m, baseRth - confidenceAdjustment);

            // Major events require higher quality
            if (intelligence.IsFomcDay || intelligence.IsCpiDay)
            {
                qthNight += 0.15m;
                qthOpen += 0.15m;
                qthRth += 0.15m;
            }

            SetEnvironmentVariable("QTH_NIGHT", qthNight.ToString("F2"));
            SetEnvironmentVariable("QTH_OPEN", qthOpen.ToString("F2"));
            SetEnvironmentVariable("QTH_RTH", qthRth.ToString("F2"));
        }

        /// <summary>
        /// Gets current intelligence for OrchestratorAgent consumption
        /// </summary>
        public MarketContext? GetCurrentIntelligence()
        {
            lock (_lock)
            {
                return _currentIntelligence;
            }
        }

        /// <summary>
        /// Gets strategy-specific position size multiplier
        /// Called by OrchestratorAgent for position sizing decisions
        /// </summary>
        public decimal GetStrategyPositionMultiplier(string strategyId)
        {
            if (!_strategyMapping.TryGetValue(strategyId, out var category))
                return 1.0m;

            var intelligence = GetCurrentIntelligence();
            if (intelligence == null)
                return 1.0m;

            var baseMultiplier = _intelligenceService.GetPositionSizeMultiplier(intelligence);

            // Apply strategy-category-specific adjustments
            var categoryMultiplier = category switch
            {
                StrategyCategory.Breakout => GetEnvironmentDecimal("BREAKOUT_MULTIPLIER", 1.0m),
                StrategyCategory.MeanReversion => GetEnvironmentDecimal("MEAN_REVERSION_MULTIPLIER", 1.0m),
                StrategyCategory.Momentum => GetEnvironmentDecimal("MOMENTUM_MULTIPLIER", 1.0m),
                StrategyCategory.Scalping => GetEnvironmentDecimal("SCALPING_MULTIPLIER", 1.0m),
                _ => 1.0m
            };

            return baseMultiplier * categoryMultiplier;
        }

        /// <summary>
        /// Logs trade result for intelligence feedback loop
        /// Called by OrchestratorAgent after trade execution
        /// </summary>
        public async Task LogTradeResultAsync(string symbol, string strategyId, decimal entryPrice, decimal exitPrice, decimal pnl)
        {
            try
            {
                var intelligence = GetCurrentIntelligence();
                await _intelligenceService.LogTradeResultAsync(symbol, entryPrice, exitPrice, pnl, intelligence);
                
                _logger.LogInformation("[INTEGRATION] Logged trade result: {Strategy} {Symbol} PnL={PnL:F2} with intelligence={HasIntel}",
                    strategyId, symbol, pnl, intelligence != null);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[INTEGRATION] Failed to log trade result");
            }
        }

        /// <summary>
        /// Clears intelligence-specific environment variables when intelligence unavailable
        /// </summary>
        private void ClearIntelligenceEnvironmentVariables()
        {
            SetEnvironmentVariable("INTELLIGENCE_AVAILABLE", "false");
            SetEnvironmentVariable("INTELLIGENCE_REGIME", "Unknown");
            SetEnvironmentVariable("INTELLIGENCE_CONFIDENCE", "0.5");
            SetEnvironmentVariable("INTELLIGENCE_BIAS", "Neutral");
            SetEnvironmentVariable("POSITION_SIZE_MULTIPLIER", "1.0");
            SetEnvironmentVariable("STOP_LOSS_MULTIPLIER", "1.0");
            SetEnvironmentVariable("TAKE_PROFIT_MULTIPLIER", "1.0");
            SetEnvironmentVariable("FOMC_DAY", "false");
            SetEnvironmentVariable("CPI_DAY", "false");
            SetEnvironmentVariable("HIGH_VOLATILITY_EVENT", "false");
        }

        private void SetEnvironmentVariable(string key, string value)
        {
            Environment.SetEnvironmentVariable(key, value);
        }

        private decimal GetEnvironmentDecimal(string key, decimal defaultValue)
        {
            var value = Environment.GetEnvironmentVariable(key);
            return decimal.TryParse(value, out var result) ? result : defaultValue;
        }
    }

    /// <summary>
    /// Extension methods for easy integration with OrchestratorAgent
    /// </summary>
    public static class OrchestratorAgentIntegrationExtensions
    {
        /// <summary>
        /// Adds LocalBotMechanicIntegration to DI container
        /// Call this in Program.cs service registration
        /// </summary>
        public static IServiceCollection AddLocalBotMechanicIntegration(this IServiceCollection services)
        {
            services.AddSingleton<StrategyClassifier>();
            services.AddScoped<LocalBotMechanicIntegration>();
            return services;
        }
    }
}
