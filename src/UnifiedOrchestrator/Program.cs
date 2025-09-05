using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Services;
using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.UnifiedOrchestrator.Infrastructure;
using BotCore.Infra;

namespace TradingBot.UnifiedOrchestrator;

/// <summary>
/// 🚀 UNIFIED TRADING ORCHESTRATOR SYSTEM 🚀
/// 
/// This is the ONE MASTER ORCHESTRATOR that replaces all 4+ separate orchestrators:
/// - Enhanced/TradingOrchestrator.cs
/// - Core/Intelligence/TradingIntelligenceOrchestrator.cs  
/// - src/OrchestratorAgent/Program.cs
/// - workflow-orchestrator.js
/// 
/// ALL FUNCTIONALITY IS NOW UNIFIED INTO ONE SYSTEM THAT WORKS TOGETHER
/// </summary>
public class Program
{
    public static async Task Main(string[] args)
    {
        Console.WriteLine(@"
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                          🚀 UNIFIED TRADING ORCHESTRATOR SYSTEM 🚀                    ║
║                                                                                       ║
║  🧠 ONE BRAIN - Consolidates all trading bot functionality into one unified system   ║
║  ⚡ ONE SYSTEM - Replaces 4+ separate orchestrators with clean, integrated solution  ║
║  🔄 ONE WORKFLOW ENGINE - All workflows managed by single scheduler                  ║
║  🌐 ONE TOPSTEPX CONNECTION - Unified API and SignalR hub management                ║
║  📊 ONE INTELLIGENCE SYSTEM - ML/RL models and predictions unified                  ║
║  📈 ONE TRADING ENGINE - All trading logic consolidated                             ║
║  📁 ONE DATA SYSTEM - Centralized data collection and reporting                     ║
║                                                                                       ║
║  ✅ Clean Build - No duplicated logic or conflicts                                  ║
║  🔧 Wired Together - All 1000+ features work in unison                             ║
║  🎯 Single Purpose - Connect to TopstepX and trade effectively                     ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
        ");

        try
        {
            // Build the unified host with all services
            var host = CreateHostBuilder(args).Build();
            
            // Display startup information
            DisplayStartupInfo();
            
            // Run the unified orchestrator
            await host.RunAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ CRITICAL ERROR: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            Environment.Exit(1);
        }
    }

    private static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureLogging(logging =>
            {
                logging.ClearProviders();
                logging.AddConsole();
                logging.SetMinimumLevel(LogLevel.Information);
            })
            .ConfigureServices((context, services) =>
            {
                // Configure unified orchestrator services
                ConfigureUnifiedServices(services);
                
                // Register initialization as hosted service
                services.AddHostedService<AdvancedSystemInitializationService>();
            });

    private static void ConfigureUnifiedServices(IServiceCollection services)
    {
        Console.WriteLine("🔧 Configuring Unified Orchestrator Services...");

        // Core HTTP client for TopstepX API
        services.AddHttpClient<TopstepAuthAgent>(client =>
        {
            client.BaseAddress = new Uri("https://api.topstepx.com");
            client.DefaultRequestHeaders.Add("User-Agent", "UnifiedTradingOrchestrator/1.0");
        });

        // Register the CENTRAL MESSAGE BUS - The "ONE BRAIN" communication system
        services.AddSingleton<ICentralMessageBus, CentralMessageBus>();
        Console.WriteLine("🧠 Central Message Bus registered - ONE BRAIN communication enabled");

        // ================================================================================
        // ADVANCED SYSTEM COMPONENTS - ALL FEATURES INTEGRATED INTO ONE BRAIN
        // ================================================================================
        
        // Register ML Memory Management - Prevents memory leaks and optimizes model lifecycle
        services.AddMLMemoryManagement();
        Console.WriteLine("🧠 ML Memory Management integrated - Memory leak prevention enabled");
        
        // Register Enhanced ML Model Manager with memory management integration
        services.AddEnhancedMLModelManager();
        Console.WriteLine("🤖 Enhanced ML Model Manager integrated - Model lifecycle optimized");
        
        // Register Economic Event Management - Trading restrictions during high-impact events
        services.AddEconomicEventManagement();
        Console.WriteLine("📈 Economic Event Management integrated - Trading protection enabled");
        
        // Register Workflow Orchestration - Prevents workflow collisions and manages priorities
        services.AddWorkflowOrchestration();
        Console.WriteLine("⚡ Workflow Orchestration integrated - Collision prevention enabled");
        
        // Register Advanced System Integration Service - The UNIFIED COORDINATOR
        services.AddSingleton<AdvancedSystemIntegrationService>();
        Console.WriteLine("🌟 Advanced System Integration Service registered - UNIFIED BRAIN COORDINATOR");

        // Register TopstepX authentication agent
        services.AddSingleton<TopstepAuthAgent>();

        // Register orchestrator components (use demo mode if no credentials)
        var hasCredentials = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TOPSTEPX_JWT")) ||
                           (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME")) &&
                            !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY")));

        if (hasCredentials)
        {
            services.AddSingleton<ITradingOrchestrator, TradingOrchestratorService>();
            Console.WriteLine("✅ Live TopstepX mode enabled");
        }
        else
        {
            services.AddSingleton<ITradingOrchestrator, DemoTradingOrchestratorService>();
            Console.WriteLine("🎭 Demo mode enabled (no TopstepX credentials)");
        }
        
        // Register Intelligence and Data orchestrators with full AI integration
        services.AddSingleton<IIntelligenceOrchestrator, IntelligenceOrchestratorService>();
        services.AddSingleton<IDataOrchestrator, DataOrchestratorService>();
        services.AddSingleton<IWorkflowScheduler, WorkflowSchedulerService>();
        
        // Register Cloud Data Integration - Links 27 GitHub workflows to trading decisions
        services.AddSingleton<ICloudDataIntegration, CloudDataIntegrationService>();
        Console.WriteLine("🌐 Cloud Data Integration enabled - GitHub workflows linked to trading");

        // Register the main unified orchestrator as both interface and hosted service
        services.AddSingleton<UnifiedOrchestratorService>();
        services.AddSingleton<IUnifiedOrchestrator>(provider => provider.GetRequiredService<UnifiedOrchestratorService>());
        services.AddHostedService<UnifiedOrchestratorService>(provider => provider.GetRequiredService<UnifiedOrchestratorService>());

        Console.WriteLine("✅ UNIFIED ORCHESTRATOR SERVICES CONFIGURED - ALL FEATURES INTEGRATED INTO ONE BRAIN");
    }

    private static void DisplayStartupInfo()
    {
        Console.WriteLine();
        Console.WriteLine("🏗️  UNIFIED ORCHESTRATOR SYSTEM STARTUP");
        Console.WriteLine("═══════════════════════════════════════");
        Console.WriteLine();
        
        Console.WriteLine("📋 CONSOLIDATION SUMMARY:");
        Console.WriteLine("  • Enhanced/TradingOrchestrator.cs           ❌ REPLACED");
        Console.WriteLine("  • Core/Intelligence/TradingIntelligenceOrchestrator.cs ❌ REPLACED");
        Console.WriteLine("  • src/OrchestratorAgent/Program.cs         ❌ REPLACED");
        Console.WriteLine("  • workflow-orchestrator.js                 ❌ REPLACED");
        Console.WriteLine("  • UnifiedOrchestrator                      ✅ ACTIVE");
        Console.WriteLine();
        
        Console.WriteLine("🔧 UNIFIED COMPONENTS:");
        Console.WriteLine("  • TradingOrchestratorService       - TopstepX connectivity & trading");
        Console.WriteLine("  • IntelligenceOrchestratorService  - ML/RL models & predictions");
        Console.WriteLine("  • DataOrchestratorService          - Data collection & reporting");
        Console.WriteLine("  • WorkflowSchedulerService         - Unified workflow scheduling");
        Console.WriteLine("  • UnifiedOrchestratorService       - Master coordinator");
        Console.WriteLine();
        
        Console.WriteLine("🌟 ADVANCED SYSTEM COMPONENTS - ALL INTEGRATED:");
        Console.WriteLine("  • MLMemoryManager                  - Memory leak prevention & model lifecycle");
        Console.WriteLine("  • WorkflowOrchestrationManager     - Collision prevention & priority management");
        Console.WriteLine("  • RedundantDataFeedManager         - High availability data feeds");
        Console.WriteLine("  • EconomicEventManager             - Trading restrictions during events");
        Console.WriteLine("  • AdvancedSystemIntegrationService - UNIFIED BRAIN COORDINATOR");
        Console.WriteLine("  • StrategyMlModelManager           - Enhanced ML model management");
        Console.WriteLine();
        
        Console.WriteLine("🌐 TOPSTEPX INTEGRATION:");
        Console.WriteLine("  • REST API:      https://api.topstepx.com");
        Console.WriteLine("  • User Hub:      https://rtc.topstepx.com/hubs/user");
        Console.WriteLine("  • Market Hub:    https://rtc.topstepx.com/hubs/market");
        Console.WriteLine("  • Authentication: JWT token or username/API key");
        Console.WriteLine();
        
        Console.WriteLine("📊 WORKFLOW OVERVIEW:");
        Console.WriteLine("  • ES/NQ Critical Trading        (Every 5-30 min)");
        Console.WriteLine("  • Portfolio Heat Management     (Every 10-30 min)");
        Console.WriteLine("  • ML/RL Intelligence System     (Every 10-60 min)");
        Console.WriteLine("  • Microstructure Analysis       (Every 5-15 min)");
        Console.WriteLine("  • Options Flow Analysis         (Every 5-10 min)");
        Console.WriteLine("  • Intermarket Correlations      (Every 15-30 min)");
        Console.WriteLine("  • Daily Data Collection         (3x daily)");
        Console.WriteLine("  • Daily Reporting System        (5 PM ET)");
        Console.WriteLine();
        
        Console.WriteLine("🔐 ENVIRONMENT VARIABLES:");
        Console.WriteLine("  • TOPSTEPX_JWT           - Direct JWT token");
        Console.WriteLine("  • TOPSTEPX_USERNAME      - TopstepX username");
        Console.WriteLine("  • TOPSTEPX_API_KEY       - TopstepX API key");
        Console.WriteLine("  • TOPSTEPX_API_BASE      - API base URL (optional)");
        Console.WriteLine();
        
        Console.WriteLine("🚀 Starting Unified Orchestrator...");
        Console.WriteLine();
    }
}

/// <summary>
/// Extension methods for the unified orchestrator
/// </summary>
public static class UnifiedOrchestratorExtensions
{
    /// <summary>
    /// Get status information for the unified orchestrator
    /// </summary>
    public static async Task<string> GetFormattedStatusAsync(this IUnifiedOrchestrator orchestrator)
    {
        var status = await orchestrator.GetStatusAsync();
        
        return $@"
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                        UNIFIED ORCHESTRATOR STATUS                                ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║ Running:          {(status.IsRunning ? "✅ YES" : "❌ NO"),-60} ║
║ TopstepX:         {(status.IsConnectedToTopstep ? "✅ CONNECTED" : "❌ DISCONNECTED"),-60} ║
║ Active Workflows: {status.ActiveWorkflows,-60} ║
║ Total Workflows:  {status.TotalWorkflows,-60} ║
║ Uptime:           {status.Uptime:dd\\.hh\\:mm\\:ss,-60} ║
║ Started:          {status.StartTime:yyyy-MM-dd HH:mm:ss} UTC{"",-36} ║
╚═══════════════════════════════════════════════════════════════════════════════════╝";
    }
    
    /// <summary>
    /// Get workflow summary for the unified orchestrator
    /// </summary>
    public static string GetWorkflowSummary(this IUnifiedOrchestrator orchestrator)
    {
        var workflows = orchestrator.GetWorkflows();
        
        var summary = @"
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                             WORKFLOW SUMMARY                                      ║
╠═══════════════════════════════════════════════════════════════════════════════════╣";

        foreach (var workflow in workflows.OrderBy(w => w.Priority).ThenBy(w => w.Name))
        {
            var status = workflow.Enabled ? "✅" : "❌";
            var tier = workflow.Priority == 1 ? "CRITICAL" : workflow.Priority == 2 ? "HIGH" : "NORMAL";
            summary += $@"
║ {status} [{tier}] {workflow.Name,-50} ║";
        }
        
        summary += @"
╚═══════════════════════════════════════════════════════════════════════════════════╝";
        
        return summary;
    }
}

/// <summary>
/// Hosted service that initializes all advanced system components during startup
/// This ensures everything is properly integrated into the unified orchestrator brain
/// </summary>
public class AdvancedSystemInitializationService : IHostedService
{
    private readonly ILogger<AdvancedSystemInitializationService> _logger;
    private readonly IServiceProvider _serviceProvider;

    public AdvancedSystemInitializationService(
        ILogger<AdvancedSystemInitializationService> logger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🚀 Initializing ALL Advanced System Components for Unified Orchestrator Brain");

        try
        {
            // Initialize BotCore advanced system components
            await AdvancedSystemConfiguration.InitializeAdvancedSystemAsync(_serviceProvider);
            _logger.LogInformation("✅ BotCore advanced components initialized");

            // Initialize workflow orchestration
            await WorkflowOrchestrationConfiguration.InitializeWorkflowOrchestrationAsync(_serviceProvider);
            _logger.LogInformation("✅ Workflow orchestration initialized");

            // Wire workflow orchestration with existing services
            WorkflowOrchestrationConfiguration.WireWorkflowOrchestration(_serviceProvider);
            _logger.LogInformation("✅ Workflow orchestration wired with existing services");

            // Initialize the unified advanced system integration service
            var integrationService = _serviceProvider.GetService<AdvancedSystemIntegrationService>();
            if (integrationService != null)
            {
                await integrationService.InitializeAsync();
                _logger.LogInformation("✅ Advanced System Integration Service initialized - UNIFIED BRAIN ACTIVE");
            }

            _logger.LogInformation("🌟 ALL ADVANCED SYSTEM COMPONENTS SUCCESSFULLY INTEGRATED INTO UNIFIED ORCHESTRATOR BRAIN");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to initialize advanced system components");
            throw;
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🛑 Advanced System Initialization Service stopping");
        return Task.CompletedTask;
    }
}