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
        /// <summary>
        /// Register all production readiness services
        /// Adds configuration, historical data bridge, and enhanced market data flow services
        /// </summary>
        public static IServiceCollection AddProductionReadinessServices(
            this IServiceCollection services, 
            IConfiguration configuration)
        {
            // Register trading readiness configuration
            services.Configure<TradingReadinessConfiguration>(
                configuration.GetSection("TradingReadiness"));

            // Register historical data bridge service
            services.AddScoped<IHistoricalDataBridgeService, HistoricalDataBridgeService>();

            // Register bar consumer for historical data integration
            services.AddScoped<IHistoricalBarConsumer, TradingSystemBarConsumer>();

            // Register enhanced market data flow service
            services.AddScoped<IEnhancedMarketDataFlowService, EnhancedMarketDataFlowService>();

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
                config.MinBarsSeen = 10;
                config.MinSeededBars = 8;
                config.MinLiveTicks = 2;
                config.MaxHistoricalDataAgeHours = 24;
                config.MarketDataTimeoutSeconds = 300;
                config.EnableHistoricalSeeding = true;
                config.EnableProgressiveReadiness = true;
                config.SeedingContracts = new[] { "CON.F.US.EP.Z25", "CON.F.US.ENQ.Z25" };

                // Environment-specific settings
                config.Environment = new EnvironmentSettings
                {
                    Name = "production",
                    Dev = new DevEnvironmentSettings
                    {
                        MinBarsSeen = 5,
                        MinSeededBars = 3,
                        MinLiveTicks = 1,
                        AllowMockData = true
                    },
                    Production = new ProductionEnvironmentSettings
                    {
                        MinBarsSeen = 10,
                        MinSeededBars = 8,
                        MinLiveTicks = 2,
                        AllowMockData = false
                    }
                };
            });

            return services;
        }
    }
}