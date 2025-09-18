using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using Infrastructure.TopstepX;
using BotCore.Infrastructure;
using BotCore.Testing;
using BotCore.Reporting;
using BotCore.AutoRemediation;
using BotCore.ProductionGate;
using BotCore.Orchestration;

namespace BotCore.DeploymentPipeline;

/// <summary>
/// Main application entry point for the comprehensive deployment pipeline
/// </summary>
public static class Program
{
    #region Exit Codes Constants
    
    private const int SUCCESS_EXIT_CODE = 0;
    private const int GENERAL_ERROR_EXIT_CODE = 1;
    private const int CREDENTIAL_ERROR_EXIT_CODE = 2;
    private const int STAGING_DEPLOYMENT_ERROR_EXIT_CODE = 3;
    private const int TEST_FAILURE_EXIT_CODE = 4;
    private const int PERFORMANCE_ERROR_EXIT_CODE = 5;
    private const int SECURITY_ERROR_EXIT_CODE = 6;
    private const int AUTO_REMEDIATION_ERROR_EXIT_CODE = 7;
    private const int PRODUCTION_GATE_ERROR_EXIT_CODE = 8;
    private const int CRITICAL_SYSTEM_ERROR_EXIT_CODE = 9;
    private const int TIMEOUT_EXIT_CODE = 124;
    
    #endregion
    static async Task<int> Main(string[] args)
    {
        Console.WriteLine("🚀 TopStep Trading Bot - Comprehensive Deployment Pipeline");
        Console.WriteLine("=========================================================");
        Console.WriteLine($"Started at: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        Console.WriteLine();

        try
        {
            // Build the application host with dependency injection
            var host = CreateHostBuilder(args).Build();

            // Get the pipeline orchestrator
            var orchestrator = host.Services.GetRequiredService<DeploymentPipelineOrchestrator>();

            // Execute the complete deployment pipeline
            using var cancellationTokenSource = new CancellationTokenSource(TimeSpan.FromMinutes(30)); // 30-minute timeout
            var result = await orchestrator.ExecuteFullPipelineAsync(cancellationTokenSource.Token);

            // Determine exit code based on result
            var exitCode = DetermineExitCode(result);
            
            Console.WriteLine();
            Console.WriteLine($"🏁 Pipeline completed with exit code: {exitCode}");
            Console.WriteLine($"Finished at: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
            
            return exitCode;
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("⏰ Pipeline execution timed out");
            return TIMEOUT_EXIT_CODE; // Timeout exit code
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Critical pipeline error: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            return GENERAL_ERROR_EXIT_CODE; // General error
        }
    }

    private static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureLogging(logging =>
            {
                logging.ClearProviders();
                logging.AddSimpleConsole(options =>
                {
                    options.IncludeScopes = false;
                    options.SingleLine = true;
                    options.TimestampFormat = "HH:mm:ss ";
                    options.UseUtcTimestamp = true;
                });
                logging.SetMinimumLevel(LogLevel.Information);
            })
            .ConfigureServices((context, services) =>
            {
                // Register core services
                services.AddLogging();
                
                // Register credential management
                services.AddSingleton<TopstepXCredentialManager>();
                
                // Register infrastructure services
                services.AddSingleton<StagingEnvironmentManager>();
                
                // Register testing services
                services.AddSingleton<ComprehensiveSmokeTestSuite>();
                
                // Register reporting services
                services.AddSingleton<ComprehensiveReportingSystem>();
                
                // Register auto-remediation services
                services.AddSingleton<AutoRemediationSystem>();
                
                // Register production gate services
                services.AddSingleton<ProductionGateSystem>();
                
                // Register main orchestrator
                services.AddSingleton<DeploymentPipelineOrchestrator>();
            });

    private static int DetermineExitCode(PipelineExecutionResult result)
    {
        // Exit codes following standard conventions:
        // 0 = Success (production ready)
        // 1 = General error
        // 2 = Credential issues
        // 3 = Staging deployment issues
        // 4 = Test failures
        // 5 = Performance issues
        // 6 = Security issues
        // 7 = Auto-remediation issues
        // 8 = Production gate failure
        // 9 = Critical system error

        if (!string.IsNullOrEmpty(result.CriticalError))
        {
            return CRITICAL_SYSTEM_ERROR_EXIT_CODE; // Critical system error
        }

        if (result.IsProductionReady)
        {
            return SUCCESS_EXIT_CODE; // Complete success - ready for production
        }

        // Determine specific failure reason
        if (!result.CredentialDetection.IsSuccessful)
        {
            return CREDENTIAL_ERROR_EXIT_CODE; // Credential issues
        }

        if (!result.StagingDeployment.IsSuccessful)
        {
            return STAGING_DEPLOYMENT_ERROR_EXIT_CODE; // Staging deployment issues
        }

        if (!result.TestSuiteExecution.IsOverallSuccess)
        {
            return TEST_FAILURE_EXIT_CODE; // Test failures
        }

        if (!result.ProductionGate.PerformanceValidation.IsSuccessful)
        {
            return PERFORMANCE_ERROR_EXIT_CODE; // Performance issues
        }

        if (!result.ProductionGate.SecurityValidation.IsSuccessful)
        {
            return SECURITY_ERROR_EXIT_CODE; // Security issues
        }

        if (!result.AutoRemediation.OverallSuccess)
        {
            return AUTO_REMEDIATION_ERROR_EXIT_CODE; // Auto-remediation issues
        }

        if (!result.ProductionGate.IsProductionReady)
        {
            return PRODUCTION_GATE_ERROR_EXIT_CODE; // Production gate failure
        }

        return GENERAL_ERROR_EXIT_CODE; // General error (fallback)
    }
}