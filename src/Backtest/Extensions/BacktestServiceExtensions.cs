using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TradingBot.Backtest;
using TradingBot.Backtest.Adapters;
using TradingBot.Backtest.ExecutionSimulators;
using TradingBot.Backtest.Metrics;

namespace TradingBot.Backtest.Extensions
{
    /// <summary>
    /// Dependency injection extensions for backtest services
    /// Wire the new production backtest system into the existing DI container
    /// </summary>
    public static class BacktestServiceExtensions
    {
        /// <summary>
        /// Add comprehensive backtest services to DI container
        /// This replaces fake simulation methods with real historical data processing
        /// PRODUCTION: ES and NQ contracts only, no mock implementations
        /// </summary>
        public static IServiceCollection AddProductionBacktestServices(this IServiceCollection services, string metricsPath = "reports/bt")
        {
            // Core backtest services
            services.AddSingleton<TradingBot.Backtest.BacktestHarnessService>();
            services.AddSingleton<WalkForwardValidationService>();

            // Execution simulation
            services.AddSingleton<IExecutionSimulator, SimpleExecutionSimulator>();

            // Metric collection
            services.AddSingleton<IMetricSink>(sp =>
                new JsonMetricSink(metricsPath, sp.GetRequiredService<ILogger<JsonMetricSink>>()));

            // 🔴 REAL TopstepX data provider (ES and NQ only)
            services.AddTopstepXHistoricalDataProvider();

            // 🟢 PRODUCTION Model Registry (ES and NQ only) - NO MOCK
            services.AddProductionModelRegistry();

            // Configuration
            services.Configure<BacktestOptions>(options =>
            {
                options.InitialCapital = 100000m;
                options.CommissionPerContract = 2.50m;
                options.BaseSlippagePercent = 0.5m;
                options.MaxPositionSizePercent = 0.02m;
            });

            services.Configure<WfvOptions>(options =>
            {
                options.TrainingWindowDays = 30;
                options.ValidationWindowDays = 7;
                options.StepSizeDays = 7;
                options.PurgeDays = 1;
                options.EmbargoDays = 1;
                options.MinSharpeThreshold = 0.5m;
                options.MaxDrawdownLimit = 0.15m;
                options.MinTradesPerFold = 10;
            });

            // Integration service that bridges old and new systems
            services.AddSingleton<TradingBot.UnifiedOrchestrator.Services.BacktestIntegrationService>();

            return services;
        }

        // REMOVED: AddMockBacktestServices() - No mock implementations allowed in production

        /// <summary>
        /// Add real production adapters for live TopstepX integration
        /// PRODUCTION: ES and NQ contracts only, no mock implementations
        /// </summary>
        public static IServiceCollection AddRealProductionAdapters(this IServiceCollection services)
        {
            // 🟢 PRODUCTION: Real TopstepX historical data provider (ES and NQ only)
            services.AddTopstepXHistoricalDataProvider();
            
            // 🟢 PRODUCTION: Real model registry (ES and NQ only) - NO MOCK
            services.AddProductionModelRegistry();

            return services;
        }
    }

    /// <summary>
    /// Extension methods for easy configuration
    /// </summary>
    public static class BacktestConfigurationExtensions
    {
        /// <summary>
        /// Configure backtest options from configuration section
        /// </summary>
        public static IServiceCollection ConfigureBacktestOptions(
            this IServiceCollection services,
            Action<BacktestOptions> configure)
        {
            services.Configure(configure);
            return services;
        }

        /// <summary>
        /// Configure walk-forward validation options
        /// </summary>
        public static IServiceCollection ConfigureWfvOptions(
            this IServiceCollection services,
            Action<WfvOptions> configure)
        {
            services.Configure(configure);
            return services;
        }
    }
}