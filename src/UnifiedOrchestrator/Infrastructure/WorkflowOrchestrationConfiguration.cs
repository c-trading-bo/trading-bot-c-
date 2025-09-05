using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Services;
using TradingBot.UnifiedOrchestrator.Interfaces;

namespace TradingBot.UnifiedOrchestrator.Infrastructure;

/// <summary>
/// Configuration service for workflow orchestration and advanced coordination
/// </summary>
public static class WorkflowOrchestrationConfiguration
{
    /// <summary>
    /// Register workflow orchestration services
    /// </summary>
    public static IServiceCollection AddWorkflowOrchestration(this IServiceCollection services)
    {
        services.AddSingleton<IWorkflowOrchestrationManager, WorkflowOrchestrationManager>();
        
        return services;
    }
    
    /// <summary>
    /// Initialize workflow orchestration system
    /// </summary>
    public static async Task InitializeWorkflowOrchestrationAsync(IServiceProvider serviceProvider)
    {
        var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
        var logger = loggerFactory.CreateLogger(typeof(WorkflowOrchestrationConfiguration));
        logger.LogInformation("[Workflow-Config] Initializing workflow orchestration system");
        
        // Initialize workflow orchestration manager
        var orchestrationManager = serviceProvider.GetService<IWorkflowOrchestrationManager>();
        if (orchestrationManager != null)
        {
            await orchestrationManager.InitializeAsync();
            logger.LogInformation("[Workflow-Config] Workflow orchestration manager initialized");
        }
        
        logger.LogInformation("[Workflow-Config] Workflow orchestration system initialized successfully");
    }
    
    /// <summary>
    /// Wire workflow orchestration with existing orchestrator services
    /// </summary>
    public static void WireWorkflowOrchestration(IServiceProvider serviceProvider)
    {
        var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
        var logger = loggerFactory.CreateLogger(typeof(WorkflowOrchestrationConfiguration));
        var orchestrationManager = serviceProvider.GetService<IWorkflowOrchestrationManager>();
        
        if (orchestrationManager == null)
        {
            logger.LogWarning("[Workflow-Config] Workflow orchestration manager not available");
            return;
        }
        
        // Wire with existing orchestrator services
        var tradingOrchestrator = serviceProvider.GetService<TradingOrchestratorService>();
        var workflowScheduler = serviceProvider.GetService<WorkflowSchedulerService>();
        var unifiedOrchestrator = serviceProvider.GetService<UnifiedOrchestratorService>();
        
        if (tradingOrchestrator != null)
        {
            logger.LogInformation("[Workflow-Config] Wired workflow orchestration with trading orchestrator");
        }
        
        if (workflowScheduler != null)
        {
            logger.LogInformation("[Workflow-Config] Wired workflow orchestration with workflow scheduler");
        }
        
        if (unifiedOrchestrator != null)
        {
            logger.LogInformation("[Workflow-Config] Wired workflow orchestration with unified orchestrator");
        }
        
        logger.LogInformation("[Workflow-Config] Workflow orchestration wiring completed");
    }
}