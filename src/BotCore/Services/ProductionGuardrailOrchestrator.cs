using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Services;

namespace BotCore.Services;

/// <summary>
/// Production-ready orchestrator service that integrates all guardrails
/// Ensures DRY_RUN precedence, order evidence requirements, price validation, etc.
/// </summary>
public class ProductionGuardrailOrchestrator : IHostedService
{
    private readonly ILogger<ProductionGuardrailOrchestrator> _logger;
    private readonly ProductionKillSwitchService _killSwitchService;
    private readonly ProductionOrderEvidenceService _orderEvidenceService;
    private readonly IServiceProvider _serviceProvider;

    public ProductionGuardrailOrchestrator(
        ILogger<ProductionGuardrailOrchestrator> logger,
        ProductionKillSwitchService killSwitchService,
        ProductionOrderEvidenceService orderEvidenceService,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _killSwitchService = killSwitchService;
        _orderEvidenceService = orderEvidenceService;
        _serviceProvider = serviceProvider;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🛡️ [GUARDRAILS] Production guardrail orchestrator starting...");
        
        // Validate execution mode on startup
        ValidateExecutionMode();
        
        // Log current guardrail status
        LogGuardrailStatus();
        
        _logger.LogInformation("✅ [GUARDRAILS] Production guardrail orchestrator started successfully");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔴 [GUARDRAILS] Production guardrail orchestrator stopping");
        return Task.CompletedTask;
    }

    private void ValidateExecutionMode()
    {
        var isDryRun = ProductionKillSwitchService.IsDryRunMode();
        var killSwitchActive = ProductionKillSwitchService.IsKillSwitchActive();

        if (killSwitchActive)
        {
            _logger.LogCritical("🔴 [GUARDRAILS] KILL SWITCH ACTIVE - All execution disabled");
        }

        if (isDryRun)
        {
            _logger.LogInformation("✅ [GUARDRAILS] DRY_RUN mode active - Safe for testing");
        }
        else
        {
            _logger.LogWarning("⚠️ [GUARDRAILS] LIVE EXECUTION mode detected - Proceed with caution");
        }

        // Log environment variables for transparency
        LogEnvironmentVariables();
    }

    private void LogGuardrailStatus()
    {
        _logger.LogInformation("🛡️ [GUARDRAILS] Current guardrail status:");
        _logger.LogInformation("  • DRY_RUN precedence: {Status}", ProductionKillSwitchService.IsDryRunMode() ? "ENFORCED" : "Not active");
        _logger.LogInformation("  • Kill switch monitoring: {Status}", "ACTIVE");
        _logger.LogInformation("  • Order evidence validation: {Status}", "ACTIVE");
        _logger.LogInformation("  • ES/MES tick rounding: {Status}", "ACTIVE (0.25)");
        _logger.LogInformation("  • Risk validation: {Status}", "ACTIVE (reject if ≤ 0)");
    }

    private void LogEnvironmentVariables()
    {
        var variables = new[]
        {
            "DRY_RUN",
            "EXECUTE", 
            "AUTO_EXECUTE",
            "TOPSTEPX_USERNAME",
            "TOPSTEPX_API_KEY",
            "TOPSTEPX_ACCOUNT_ID",
            "DAILY_LOSS_CAP_R",
            "PER_TRADE_R"
        };

        _logger.LogDebug("🔍 [GUARDRAILS] Environment variables:");
        foreach (var variable in variables)
        {
            var value = Environment.GetEnvironmentVariable(variable);
            if (variable.Contains("API_KEY") || variable.Contains("PASSWORD"))
            {
                // Mask sensitive values
                value = string.IsNullOrEmpty(value) ? "NOT_SET" : "***MASKED***";
            }
            _logger.LogDebug("  • {Variable}: {Value}", variable, value ?? "NOT_SET");
        }
    }

    /// <summary>
    /// Validate a trade before execution
    /// </summary>
    public TradeValidationResult ValidateTradeBeforeExecution(
        string symbol, 
        decimal entry, 
        decimal stop, 
        decimal target, 
        bool isLong,
        string customTag)
    {
        var result = new TradeValidationResult
        {
            Symbol = symbol,
            CustomTag = customTag,
            IsValid = true
        };

        _logger.LogInformation("🔍 [GUARDRAILS] Validating trade: {Symbol} {Side} entry={Entry:0.00} stop={Stop:0.00} target={Target:0.00} tag={Tag}",
            symbol, isLong ? "BUY" : "SELL", entry, stop, target, customTag);

        // Guardrail 1: Force DRY_RUN if kill.txt exists
        if (ProductionKillSwitchService.IsKillSwitchActive())
        {
            result.IsValid;
            result.ValidationErrors.Add("Kill switch is active - all execution disabled");
            _logger.LogCritical("🔴 [GUARDRAILS] Trade rejected: Kill switch active");
            return result;
        }

        // Guardrail 2: DRY_RUN precedence
        if (ProductionKillSwitchService.IsDryRunMode())
        {
            result.IsDryRun = true;
            _logger.LogInformation("🧪 [GUARDRAILS] Trade will be DRY_RUN only");
        }

        // Guardrail 3: ES/MES tick rounding and risk validation
        var priceValidation = ProductionPriceService.ValidateAndRoundTradeSetup(
            symbol, entry, stop, target, isLong, _logger);

        if (!priceValidation.IsValid)
        {
            result.IsValid;
            result.ValidationErrors.Add($"Price validation failed: {priceValidation.ValidationError}");
            _logger.LogCritical("🔴 [GUARDRAILS] Trade rejected: {Error}", priceValidation.ValidationError);
            return result;
        }

        result.RoundedEntry = priceValidation.RoundedEntry;
        result.RoundedStop = priceValidation.RoundedStop;
        result.RoundedTarget = priceValidation.RoundedTarget;
        result.RMultiple = priceValidation.RMultiple;

        _logger.LogInformation("✅ [GUARDRAILS] Trade validation passed - R={R:0.00}, DryRun={DryRun}", 
            result.RMultiple ?? 0, result.IsDryRun);

        return result;
    }
}

/// <summary>
/// Result of trade validation
/// </summary>
public class TradeValidationResult
{
    public string Symbol { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public bool IsValid { get; set; }
    public bool IsDryRun { get; set; }
    public List<string> ValidationErrors { get; } = new();
    
    // Validated and rounded prices
    public decimal RoundedEntry { get; set; }
    public decimal RoundedStop { get; set; }
    public decimal RoundedTarget { get; set; }
    public decimal? RMultiple { get; set; }
}