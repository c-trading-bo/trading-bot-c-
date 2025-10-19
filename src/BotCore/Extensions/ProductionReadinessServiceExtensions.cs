using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using BotCore.Services;

namespace BotCore.Extensions
{
    /// <summary>
    /// Production readiness service registration extensions
    /// Registers the new trading readiness components for production-ready trading
    /// </summary>
    public static class ProductionReadinessServiceExtensions
    {
        // Production readiness default configuration constants
        private const int ProductionMinBarsSeen = 10;
        private const int ProductionMinSeededBars = 8;
        private const int ProductionMinLiveTicks = 2;
        private const int MaxHistoricalDataAgeHours = 24;
        private const int MarketDataTimeoutSeconds = 300;
        private const int DevMinBarsSeen = 5;
        private const int DevMinSeededBars = 3;
        private const int DevMinLiveTicks = 1;
        
        // Default seeding contracts for production
        private static readonly string[] DefaultSeedingContracts = new[] { "CON.F.US.EP.Z25", "CON.F.US.ENQ.Z25" };

        /// <summary>
        /// Register all production readiness services
        /// Adds configuration, historical data bridge, and enhanced market data flow services
        /// Only registers HistoricalDataBridgeService when in HISTORICAL_MODE
        /// </summary>
        public static IServiceCollection AddProductionReadinessServices(
            this IServiceCollection services, 
            IConfiguration configuration)
        {
            ArgumentNullException.ThrowIfNull(configuration);
            
            // Register trading readiness configuration
            services.Configure<TradingReadinessConfiguration>(
                configuration.GetSection("TradingReadiness"));

            // Register trading readiness tracker
            services.AddSingleton<ITradingReadinessTracker, TradingReadinessTracker>();

            // Only register HistoricalDataBridgeService in HISTORICAL_MODE
            // In LIVE/DRY-RUN modes, we don't need historical data seeding
            if (global::BotCore.Services.ProductionKillSwitchService.IsHistoricalMode())
            {
                // Register historical data bridge service - Singleton for consumption by singleton services
                services.AddSingleton<IHistoricalDataBridgeService, HistoricalDataBridgeService>();

                // Register bar consumer for historical data integration
                services.AddSingleton<IHistoricalBarConsumer, TradingSystemBarConsumer>();
            }

            // Register enhanced market data flow service (needed in all modes)
            services.AddSingleton<IEnhancedMarketDataFlowService, EnhancedMarketDataFlowService>();

            return services;
        }

        /// <summary>
        /// Add default trading readiness configuration if not present in appsettings
        /// </summary>
        public static IServiceCollection AddDefaultTradingReadinessConfiguration(
            this IServiceCollection services)
        {
            services.Configure<TradingReadinessConfiguration>(config =>
            {
                // Production defaults
                config.MinBarsSeen = ProductionMinBarsSeen;
                config.MinSeededBars = ProductionMinSeededBars;
                config.MinLiveTicks = ProductionMinLiveTicks;
                config.MaxHistoricalDataAgeHours = MaxHistoricalDataAgeHours;
                config.MarketDataTimeoutSeconds = MarketDataTimeoutSeconds;
                config.EnableHistoricalSeeding = true;
                config.EnableProgressiveReadiness = true;
                config.SeedingContracts = DefaultSeedingContracts;

                // Environment-specific settings
                config.Environment = new EnvironmentSettings
                {
                    Name = "production",
                    Dev = new DevEnvironmentSettings
                    {
                        MinBarsSeen = DevMinBarsSeen,
                        MinSeededBars = DevMinSeededBars,
                        MinLiveTicks = DevMinLiveTicks
                    },
                    Production = new ProductionEnvironmentSettings
                    {
                        MinBarsSeen = ProductionMinBarsSeen,
                        MinSeededBars = ProductionMinSeededBars,
                        MinLiveTicks = ProductionMinLiveTicks
                    }
                };
            });

            return services;
        }
    }
}