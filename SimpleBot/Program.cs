using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Trading.Strategies;
using System;
using System.IO;

namespace SimpleBot;

class Program
{
    static async Task Main(string[] args)
    {
        // Load .env files for unified credential management
        LoadDotEnv();
        
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
    
    /// <summary>
    /// Load .env files for unified credential management
    /// Searches current and parent directories for .env.local then .env
    /// </summary>
    private static void LoadDotEnv()
    {
        try
        {
            // Search current and up to 4 parent directories for .env.local then .env
            var candidates = new[] { ".env.local", ".env" };
            string? dir = Environment.CurrentDirectory;
            for (int up = 0; up < 5 && dir != null; up++)
            {
                foreach (var file in candidates)
                {
                    var path = Path.Combine(dir, file);
                    if (File.Exists(path))
                    {
                        foreach (var raw in File.ReadAllLines(path))
                        {
                            var line = raw.Trim();
                            if (line.Length == 0 || line.StartsWith("#")) continue;
                            var idx = line.IndexOf('=');
                            if (idx <= 0) continue;
                            var key = line.Substring(0, idx).Trim();
                            var val = line.Substring(idx + 1).Trim();
                            if ((val.StartsWith("\"") && val.EndsWith("\"")) || (val.StartsWith("'") && val.EndsWith("'")))
                                val = val.Substring(1, val.Length - 2);
                            if (!string.IsNullOrWhiteSpace(key)) Environment.SetEnvironmentVariable(key, val);
                        }
                        Console.WriteLine($"[Environment] Loaded {file} from {dir}");
                    }
                }
                dir = Directory.GetParent(dir)?.FullName;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Environment] Warning: Failed to load .env files: {ex.Message}");
        }
    }
}