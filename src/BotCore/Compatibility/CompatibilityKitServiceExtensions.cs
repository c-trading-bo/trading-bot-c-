using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System;
using System.IO;

namespace BotCore.Compatibility
{
    /// <summary>
    /// Extension methods for registering Compatibility Kit services with dependency injection
    /// Ensures proper lifetimes and container verification for production readiness
    /// </summary>
    public static class CompatibilityKitServiceExtensions
    {
        /// <summary>
        /// Registers all Compatibility Kit services with proper dependency injection lifetimes
        /// </summary>
        public static IServiceCollection AddCompatibilityKit(this IServiceCollection services, IConfiguration configuration)
        {
            // Core configuration services (Singleton - shared state)
            services.AddSingleton<StructuredConfigurationManager>();
            services.AddSingleton<FileStateStore>();
            
            // Market data and coordination services (Singleton - shared subscriptions)
            services.AddSingleton<MarketDataBridge>();
            services.AddSingleton<RewardSystemConnector>();
            
            // Decision and control services (Scoped - per trading session)
            services.AddScoped<BanditController>();
            services.AddScoped<PolicyGuard>();
            services.AddScoped<RiskManagementCoordinator>();
            
            // Main compatibility kit (Scoped - per trading session)
            services.AddScoped<CompatibilityKit>();
            
            // Register auditing and validation systems
            services.AddSingleton<CompatibilityKitAuditor>();
            services.AddScoped<CompatibilityKitHardeningValidator>();
            
            return services;
        }

        /// <summary>
        /// Validates that all Compatibility Kit services are properly registered
        /// Fails fast if any dependency is missing or mis-scoped
        /// </summary>
        public static void VerifyCompatibilityKitRegistration(this IServiceProvider serviceProvider, ILogger logger)
        {
            logger.LogInformation("🔍 Verifying Compatibility Kit service registration...");
            
            try
            {
                // Verify singleton services
                VerifyService<StructuredConfigurationManager>(serviceProvider, logger, "StructuredConfigurationManager");
                VerifyService<FileStateStore>(serviceProvider, logger, "FileStateStore");
                VerifyService<MarketDataBridge>(serviceProvider, logger, "MarketDataBridge");
                VerifyService<RewardSystemConnector>(serviceProvider, logger, "RewardSystemConnector");
                VerifyService<CompatibilityKitAuditor>(serviceProvider, logger, "CompatibilityKitAuditor");
                
                // Verify scoped services
                using var scope = serviceProvider.CreateScope();
                VerifyService<BanditController>(scope.ServiceProvider, logger, "BanditController");
                VerifyService<PolicyGuard>(scope.ServiceProvider, logger, "PolicyGuard");
                VerifyService<RiskManagementCoordinator>(scope.ServiceProvider, logger, "RiskManagementCoordinator");
                VerifyService<CompatibilityKit>(scope.ServiceProvider, logger, "CompatibilityKit");
                VerifyService<CompatibilityKitHardeningValidator>(scope.ServiceProvider, logger, "CompatibilityKitHardeningValidator");
                
                logger.LogInformation("✅ All Compatibility Kit services verified successfully");
            }
            catch (Exception ex)
            {
                logger.LogCritical(ex, "❌ FATAL: Compatibility Kit service registration verification failed");
                throw new InvalidOperationException("Compatibility Kit service registration verification failed", ex);
            }
        }

        /// <summary>
        /// Run comprehensive hardening validation for production readiness
        /// </summary>
        public static async Task<HardeningValidationReport> RunHardeningValidationAsync(this IServiceProvider serviceProvider, ILogger logger)
        {
            logger.LogInformation("🛡️ Running Compatibility Kit hardening validation...");
            
            try
            {
                var validator = serviceProvider.GetRequiredService<CompatibilityKitHardeningValidator>();
                var report = await validator.ValidateProductionReadinessAsync();
                
                if (report.OverallValidationSuccess)
                {
                    logger.LogInformation("✅ Compatibility Kit hardening validation PASSED");
                }
                else
                {
                    logger.LogError("❌ Compatibility Kit hardening validation FAILED");
                    foreach (var error in report.CriticalErrors)
                    {
                        logger.LogError("🚨 Critical Error: {Error}", error);
                    }
                }
                
                return report;
            }
            catch (Exception ex)
            {
                logger.LogCritical(ex, "🚨 FATAL: Hardening validation failed with exception");
                throw;
            }
        }
        
        /// <summary>
        /// Validates configuration files are present and valid
        /// </summary>
        public static void ValidateCompatibilityKitConfiguration(this IServiceProvider serviceProvider, ILogger logger)
        {
            logger.LogInformation("🔍 Validating Compatibility Kit configuration...");
            
            var configManager = serviceProvider.GetRequiredService<StructuredConfigurationManager>();
            
            // Validate main configuration
            if (!File.Exists("config/compatibility-kit.json"))
            {
                throw new FileNotFoundException("Compatibility Kit main configuration file not found: config/compatibility-kit.json");
            }
            
            // Validate parameter bundle configuration
            if (!File.Exists("config/bundles.stage.json"))
            {
                throw new FileNotFoundException("Parameter bundles configuration file not found: config/bundles.stage.json");
            }
            
            // Validate parameter bounds configuration
            if (!File.Exists("config/bounds.json"))
            {
                throw new FileNotFoundException("Parameter bounds configuration file not found: config/bounds.json");
            }
            
            // Validate strategy configurations
            var strategyConfigPaths = new[] { "config/strategies/S2.json", "config/strategies/S3.json" };
            foreach (var path in strategyConfigPaths)
            {
                if (!File.Exists(path))
                {
                    throw new FileNotFoundException($"Strategy configuration file not found: {path}");
                }
            }
            
            // Load and validate configurations
            var mainConfig = configManager.LoadMainConfiguration();
            if (mainConfig == null)
            {
                throw new InvalidOperationException("Failed to load main compatibility kit configuration");
            }
            
            // Load and validate parameter bundles
            var bundlesConfig = configManager.LoadParameterBundles();
            if (bundlesConfig == null || bundlesConfig.Bundles.Count == 0)
            {
                throw new InvalidOperationException("Failed to load parameter bundles configuration or no bundles defined");
            }
            
            // Load and validate parameter bounds
            var boundsConfig = configManager.LoadParameterBounds();
            if (boundsConfig == null || boundsConfig.ParameterBounds.Count == 0)
            {
                throw new InvalidOperationException("Failed to load parameter bounds configuration or no bounds defined");
            }
            
            logger.LogInformation("✅ Compatibility Kit configuration validation completed successfully");
        }

        private static void VerifyService<T>(IServiceProvider serviceProvider, ILogger logger, string serviceName) 
            where T : class
        {
            try
            {
                var service = serviceProvider.GetRequiredService<T>();
                if (service == null)
                {
                    throw new InvalidOperationException($"Service {serviceName} resolved to null");
                }
                logger.LogDebug("✅ {ServiceName} verified", serviceName);
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "❌ Failed to verify service: {ServiceName}", serviceName);
                throw;
            }
        }
    }
}