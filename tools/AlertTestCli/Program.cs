using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.Infrastructure.Alerts;
using TradingBot.Monitoring;
using TradingBot.Monitoring;

namespace AlertTestCli
{
    /// <summary>
    /// CLI application for testing alert functionality
    /// Can be used to verify email and Slack alerts are working
    /// Usage: dotnet run -- [test-type]
    /// </summary>
    class Program
    {
        static async Task Main(string[] args)
        {
            // Setup DI container
            var services = new ServiceCollection();
            services.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information);
            });
            services.AddHttpClient();
            services.AddSingleton<IAlertService, AlertService>();
            services.AddSingleton<IModelHealthMonitor, ModelHealthMonitor>();
            services.AddSingleton<ILatencyMonitor, LatencyMonitor>();
            services.AddSingleton<IModelDeploymentManager, ModelDeploymentManager>();

            var serviceProvider = services.BuildServiceProvider();
            var logger = serviceProvider.GetRequiredService<ILogger<Program>>();

            logger.LogInformation("=== Trading Bot Alert Test CLI ===");
            logger.LogInformation("Testing alert system functionality...");

            try
            {
                var testType = args.Length > 0 ? args[0].ToLowerInvariant() : "all";
                
                var alertService = serviceProvider.GetRequiredService<IAlertService>();
                var modelHealthMonitor = serviceProvider.GetRequiredService<IModelHealthMonitor>();
                var latencyMonitor = serviceProvider.GetRequiredService<ILatencyMonitor>();
                var deploymentManager = serviceProvider.GetRequiredService<IModelDeploymentManager>();

                switch (testType)
                {
                    case "email":
                        await TestEmailAlert(alertService, logger);
                        break;
                    case "slack":
                        await TestSlackAlert(alertService, logger);
                        break;
                    case "model-health":
                        await TestModelHealthAlert(modelHealthMonitor, logger);
                        break;
                    case "latency":
                        await TestLatencyAlert(latencyMonitor, logger);
                        break;
                    case "deployment":
                        await TestDeploymentAlert(deploymentManager, logger);
                        break;
                    case "critical":
                        await TestCriticalAlert(alertService, logger);
                        break;
                    case "all":
                    default:
                        await TestAllAlerts(alertService, modelHealthMonitor, latencyMonitor, deploymentManager, logger);
                        break;
                }

                logger.LogInformation("Alert testing completed successfully!");
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Alert testing failed");
            }
            finally
            {
                serviceProvider.Dispose();
            }
        }

        static async Task TestEmailAlert(IAlertService alertService, ILogger logger)
        {
            logger.LogInformation("Testing email alert...");
            
            await alertService.SendEmailAsync(
                "Test Email Alert",
                "This is a test email alert from the Alert Test CLI.\n\nIf you receive this, email alerts are working correctly!",
                AlertSeverity.Info);
                
            logger.LogInformation("Email alert test completed");
        }

        static async Task TestSlackAlert(IAlertService alertService, ILogger logger)
        {
            logger.LogInformation("Testing Slack alert...");
            
            await alertService.SendSlackAsync(
                "Test Slack alert from Alert Test CLI. If you see this message, Slack alerts are working correctly! 🎉",
                AlertSeverity.Info);
                
            logger.LogInformation("Slack alert test completed");
        }

        static async Task TestModelHealthAlert(IModelHealthMonitor modelHealthMonitor, ILogger logger)
        {
            logger.LogInformation("Testing model health alert...");
            
            // Simulate poor model performance to trigger alert
            for (int i = 0; i < 30; i++)
            {
                // Very confident predictions that are wrong (high Brier score)
                modelHealthMonitor.RecordPrediction(0.9, 0.0);
            }
            
            var health = modelHealthMonitor.GetCurrentHealth();
            logger.LogInformation("Model health status: {IsHealthy}, Issues: {Issues}", 
                health.IsHealthy, string.Join("; ", health.HealthIssues));
                
            logger.LogInformation("Model health alert test completed");
        }

        static async Task TestLatencyAlert(ILatencyMonitor latencyMonitor, ILogger logger)
        {
            logger.LogInformation("Testing latency alert...");
            
            // Simulate high latency to trigger alert
            for (int i = 0; i < 5; i++)
            {
                latencyMonitor.RecordDecisionLatency(7000.0, $"Test high latency #{i}");
                latencyMonitor.RecordOrderLatency(3000.0, $"Test high order latency #{i}");
            }
            
            var health = latencyMonitor.GetLatencyHealth();
            logger.LogInformation("Latency health status: {IsHealthy}, Decision P99: {DecisionP99}ms, Order P99: {OrderP99}ms",
                health.IsHealthy, health.DecisionStats.P99, health.OrderStats.P99);
                
            logger.LogInformation("Latency alert test completed");
        }

        static async Task TestDeploymentAlert(IModelDeploymentManager deploymentManager, ILogger logger)
        {
            logger.LogInformation("Testing deployment alert...");
            
            var modelName = $"TestModel_{DateTime.UtcNow:yyyyMMdd_HHmmss}";
            
            // Test successful promotion
            await deploymentManager.PromoteModelToProductionAsync(modelName, "v1.0.0");
            
            // Test canary rollout
            await deploymentManager.StartCanaryRolloutAsync(modelName, "v1.1.0", 0.1);
            
            // Test canary failure and rollback
            await deploymentManager.FailCanaryRolloutAsync(modelName, "Test failure from CLI");
            
            var health = await deploymentManager.GetDeploymentHealthAsync();
            logger.LogInformation("Deployment health status: {IsHealthy}, Total: {Total}, Active: {Active}, Failed: {Failed}",
                health.IsHealthy, health.TotalDeployments, health.ActiveDeployments, health.FailedDeployments);
                
            logger.LogInformation("Deployment alert test completed");
        }

        static async Task TestCriticalAlert(IAlertService alertService, ILogger logger)
        {
            logger.LogInformation("Testing critical alert...");
            
            await alertService.SendCriticalAlertAsync(
                "Test Critical Alert from CLI",
                "This is a test critical alert that should trigger both email and Slack notifications. " +
                "If you receive this on both channels, the critical alert system is working correctly!");
                
            logger.LogInformation("Critical alert test completed");
        }

        static async Task TestAllAlerts(IAlertService alertService, IModelHealthMonitor modelHealthMonitor, 
            ILatencyMonitor latencyMonitor, IModelDeploymentManager deploymentManager, ILogger logger)
        {
            logger.LogInformation("Testing all alert types...");
            
            await TestEmailAlert(alertService, logger);
            await Task.Delay(1000);
            
            await TestSlackAlert(alertService, logger);
            await Task.Delay(1000);
            
            await TestModelHealthAlert(modelHealthMonitor, logger);
            await Task.Delay(1000);
            
            await TestLatencyAlert(latencyMonitor, logger);
            await Task.Delay(1000);
            
            await TestDeploymentAlert(deploymentManager, logger);
            await Task.Delay(1000);
            
            await TestCriticalAlert(alertService, logger);
            
            logger.LogInformation("All alert types tested");
        }
    }
}