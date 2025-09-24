using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using BotCore.Services;
using BotCore.Extensions;
using BotCore.Testing;
using System;
using System.Threading.Tasks;

namespace BotCore.TestApp;

/// <summary>
/// Simple console app to test production guardrails
/// </summary>
internal sealed class Program
{
    static async Task<int> Main(string[] args)
    {
        Console.WriteLine("🛡️ Production Guardrail Test App");
        Console.WriteLine("================================");

        try
        {
            // Setup services
            var services = new ServiceCollection()
                .AddProductionTradingServices()
                .AddConsoleLogger()
                .BuildServiceProvider();

            // Validate setup
            var logger = services.GetRequiredService<ILogger<Program>>();
            services.ValidateProductionGuardrails(logger);

            // Run tests
            logger.LogInformation("🧪 Running production guardrail tests...");
            var tester = ActivatorUtilities.CreateInstance<ProductionGuardrailTester>(services);
            var allPassed = await tester.RunAllTestsAsync();

            if (allPassed)
            {
                logger.LogInformation("✅ All tests PASSED - Production guardrails are working correctly");
                return 0;
            }
            else
            {
                logger.LogCritical("🔴 Some tests FAILED - Production guardrails need attention");
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Test app failed: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            return 2;
        }
    }
}

/// <summary>
/// Extension methods for console logging
/// </summary>
public static class ConsoleLoggingExtensions
{
    public static IServiceCollection AddConsoleLogger(this IServiceCollection services)
    {
        return services.AddLogging(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Information);
        });
    }
}