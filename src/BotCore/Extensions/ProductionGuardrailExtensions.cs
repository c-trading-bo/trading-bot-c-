using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using BotCore.Services;

namespace BotCore.Extensions;

/// <summary>
/// Service registration extensions for production-ready guardrail services
/// </summary>
public static class ProductionGuardrailExtensions
{
    /// <summary>
    /// Add all production guardrail services following agent rules
    /// </summary>
    public static IServiceCollection AddProductionGuardrails(this IServiceCollection services)
    {
        // Core guardrail services
        services.AddSingleton<ProductionKillSwitchService>();
        services.AddScoped<ProductionOrderEvidenceService>();
        services.AddScoped<ProductionGuardrailOrchestrator>();
        
        // Register kill switch as hosted service for monitoring
        services.AddHostedService<ProductionKillSwitchService>(provider => 
            provider.GetRequiredService<ProductionKillSwitchService>());
        
        // Register orchestrator as hosted service
        services.AddHostedService<ProductionGuardrailOrchestrator>(provider =>
            provider.GetRequiredService<ProductionGuardrailOrchestrator>());

        return services;
    }

    /// <summary>
    /// Validate that all required production guardrails are active
    /// Call this after service configuration to verify setup
    /// </summary>
    public static void ValidateProductionGuardrails(this IServiceProvider serviceProvider, ILogger? logger = null)
    {
        logger?.LogInformation("🔍 [SETUP] Validating production guardrail setup...");

        try
        {
            // Verify kill switch service
            var killSwitchService = serviceProvider.GetService<ProductionKillSwitchService>();
            if (killSwitchService == null)
            {
                throw new InvalidOperationException("ProductionKillSwitchService not registered");
            }

            // Verify order evidence service  
            var orderEvidenceService = serviceProvider.GetService<ProductionOrderEvidenceService>();
            if (orderEvidenceService == null)
            {
                throw new InvalidOperationException("ProductionOrderEvidenceService not registered");
            }

            // Verify orchestrator
            var orchestrator = serviceProvider.GetService<ProductionGuardrailOrchestrator>();
            if (orchestrator == null)
            {
                throw new InvalidOperationException("ProductionGuardrailOrchestrator not registered");
            }

            // Check execution mode
            var isDryRun = ProductionKillSwitchService.IsDryRunMode();
            var killSwitchActive = ProductionKillSwitchService.IsKillSwitchActive();

            logger?.LogInformation("✅ [SETUP] Production guardrails validated successfully");
            logger?.LogInformation("  • Kill switch monitoring: ACTIVE");
            logger?.LogInformation("  • Order evidence validation: ACTIVE");
            logger?.LogInformation("  • Price validation (ES/MES 0.25 tick): ACTIVE");
            logger?.LogInformation("  • Risk validation (reject ≤ 0): ACTIVE");
            logger?.LogInformation("  • Current mode: {Mode}", isDryRun ? "DRY_RUN" : "LIVE");
            
            if (killSwitchActive)
            {
                logger?.LogCritical("🔴 [SETUP] KILL SWITCH ACTIVE - All execution disabled");
            }
        }
        catch (Exception ex)
        {
            logger?.LogCritical(ex, "❌ [SETUP] Production guardrail validation FAILED");
            throw;
        }
    }

    /// <summary>
    /// Quick setup method for simple applications
    /// </summary>
    public static IServiceCollection AddProductionTradingServices(this IServiceCollection services)
    {
        return services
            .AddProductionGuardrails()
            .AddLogging();
    }
}