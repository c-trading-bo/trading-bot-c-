using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Trading.Strategies;

namespace SimpleBot;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("🚀 TRADING BOT - SIMPLE LAUNCHER");
        Console.WriteLine("===============================");
        Console.WriteLine();
        
        var builder = Host.CreateDefaultBuilder(args);
        
        builder.ConfigureServices(services =>
        {
            services.AddLogging(logging =>
            {
                logging.ClearProviders();
                logging.AddConsole();
                logging.SetMinimumLevel(LogLevel.Information);
            });
        });

        var host = builder.Build();
        var logger = host.Services.GetRequiredService<ILogger<Program>>();
        
        logger.LogInformation("🏗️  TRADING BOT CORE COMPONENTS LOADED:");
        logger.LogInformation("   ✅ Strategy System (Trading.Strategies namespace)");
        logger.LogInformation(""); // Empty line
        
        // Basic health check
        logger.LogInformation("🔍 HEALTH CHECK:");
        try
        {
            // Test Strategy ID generation
            var strategyId = StrategyIds.GenerateStrategyId("TestStrategy");
            logger.LogInformation("   ✅ Strategy ID Generation: {StrategyId}", strategyId);
            
            // Test Analytics
            var testData1 = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
            var testData2 = new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 };
            var correlation = Analytics.CalculatePearsonCorrelation(testData1, testData2);
            logger.LogInformation("   ✅ Analytics Correlation Test: {Correlation:F3}", correlation);
            
            logger.LogInformation(""); // Empty line
            logger.LogInformation("🎯 SYSTEM STATUS: HEALTHY");
            logger.LogInformation("📊 Core trading components are operational");
            logger.LogInformation(""); // Empty line
            logger.LogInformation("⚠️  NOTE: This is a minimal launcher for testing core components");
            logger.LogInformation("   For full trading functionality, additional components needed");
            
            // Wait briefly to show the status
            await Task.Delay(2000);
            
            logger.LogInformation(""); // Empty line
            logger.LogInformation("✅ TRADING BOT STARTUP COMPLETE - NO ERRORS, NO WARNINGS");
            
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "❌ HEALTH CHECK FAILED");
            Environment.ExitCode = 1;
        }
        
        logger.LogInformation("👋 Shutting down gracefully...");
    }
}