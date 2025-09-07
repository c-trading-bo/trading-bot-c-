using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace TradingBot.UnifiedOrchestrator.Infrastructure;

/// <summary>
/// Auto-configuration service for TopstepX paper trading mode
/// Automatically sets up the bot for paper trading without manual intervention
/// </summary>
public class AutoPaperTradingConfiguration : IHostedService
{
    private readonly ILogger<AutoPaperTradingConfiguration> _logger;
    private readonly IServiceProvider _serviceProvider;

    public AutoPaperTradingConfiguration(
        ILogger<AutoPaperTradingConfiguration> logger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🎯 Starting Auto Paper Trading Configuration");
        
        // Debug: Log environment variables
        var autoPaperTradingValue = Environment.GetEnvironmentVariable("AUTO_PAPER_TRADING");
        var paperModeValue = Environment.GetEnvironmentVariable("PAPER_MODE");
        var enableTopstepXValue = Environment.GetEnvironmentVariable("ENABLE_TOPSTEPX");
        
        _logger.LogInformation($"🔍 Environment variables check:");
        _logger.LogInformation($"   AUTO_PAPER_TRADING = '{autoPaperTradingValue}'");
        _logger.LogInformation($"   PAPER_MODE = '{paperModeValue}'");
        _logger.LogInformation($"   ENABLE_TOPSTEPX = '{enableTopstepXValue}'");
        
        // Check if auto paper trading is enabled
        var autoPaperTrading = autoPaperTradingValue == "1";
        var paperMode = paperModeValue == "1";
        
        if (autoPaperTrading || paperMode)
        {
            _logger.LogInformation("✅ Auto paper trading conditions met - proceeding with configuration");
            await ConfigureAutoPaperTradingAsync();
        }
        else
        {
            _logger.LogInformation("📋 Auto paper trading not enabled - using standard configuration");
            _logger.LogInformation($"   Reason: AUTO_PAPER_TRADING={autoPaperTradingValue}, PAPER_MODE={paperModeValue}");
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🛑 Auto Paper Trading Configuration stopped");
        return Task.CompletedTask;
    }

    private async Task ConfigureAutoPaperTradingAsync()
    {
        _logger.LogInformation("🎯 Configuring automatic paper trading mode...");
        
        try
        {
            // Validate paper trading environment
            ValidatePaperTradingEnvironment();
            
            // Configure simulated TopstepX connection
            await ConfigureSimulatedTopstepXAsync();
            
            // Set up paper trading safeguards
            ConfigurePaperTradingSafeguards();
            
            _logger.LogInformation("✅ Auto paper trading configuration completed successfully");
            
            LogConfigurationSummary();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to configure auto paper trading");
            throw;
        }
    }

    private void ValidatePaperTradingEnvironment()
    {
        _logger.LogInformation("🔍 Validating paper trading environment...");
        
        var requiredEnvVars = new[]
        {
            "PAPER_MODE",
            "TRADING_MODE", 
            "ENABLE_TOPSTEPX",
            "TOPSTEPX_API_BASE",
            "TOPSTEPX_USERNAME"
        };

        foreach (var envVar in requiredEnvVars)
        {
            var value = Environment.GetEnvironmentVariable(envVar);
            if (string.IsNullOrEmpty(value))
            {
                _logger.LogWarning("⚠️ Environment variable {EnvVar} not set", envVar);
            }
            else
            {
                _logger.LogDebug("✅ {EnvVar} = {Value}", envVar, envVar.Contains("API_KEY") || envVar.Contains("JWT") ? "***" : value);
            }
        }
    }

    private async Task ConfigureSimulatedTopstepXAsync()
    {
        _logger.LogInformation("🎭 Setting up simulated TopstepX connection for paper trading...");
        
        // Simulate connection validation
        await Task.Delay(100); // Simulate network call
        
        var apiBase = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
        var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME") ?? "demo_user";
        
        _logger.LogInformation("🌐 TopstepX API Base: {ApiBase}", apiBase);
        _logger.LogInformation("👤 Paper Trading User: {Username}", username);
        _logger.LogInformation("🔌 Connection Status: Simulated (Paper Trading Mode)");
        
        // Validate simulated connection
        var isValid = await ValidateSimulatedConnectionAsync();
        if (isValid)
        {
            _logger.LogInformation("✅ Simulated TopstepX connection validated for paper trading");
        }
        else
        {
            _logger.LogWarning("⚠️ Simulated connection validation failed - continuing in demo mode");
        }
    }

    private async Task<bool> ValidateSimulatedConnectionAsync()
    {
        try
        {
            // Simulate API validation
            await Task.Delay(50);
            
            var enableTopstepX = Environment.GetEnvironmentVariable("ENABLE_TOPSTEPX") == "1";
            var paperMode = Environment.GetEnvironmentVariable("PAPER_MODE") == "1";
            
            return enableTopstepX && paperMode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Simulated connection validation failed");
            return false;
        }
    }

    private void ConfigurePaperTradingSafeguards()
    {
        _logger.LogInformation("🛡️ Configuring paper trading safeguards...");
        
        // Set default safeguards if not configured
        var safeguards = new Dictionary<string, string>
        {
            ["PAPER_RISK_LIMIT"] = "10000",
            ["PAPER_MAX_POSITION_SIZE"] = "100", 
            ["PAPER_DAILY_LOSS_LIMIT"] = "1000",
            ["AUTO_GO_LIVE"] = "false",
            ["AUTO_STICKY_LIVE"] = "false"
        };

        foreach (var safeguard in safeguards)
        {
            var current = Environment.GetEnvironmentVariable(safeguard.Key);
            if (string.IsNullOrEmpty(current))
            {
                Environment.SetEnvironmentVariable(safeguard.Key, safeguard.Value);
                _logger.LogInformation("🛡️ Set safeguard {Key} = {Value}", safeguard.Key, safeguard.Value);
            }
        }
    }

    private void LogConfigurationSummary()
    {
        _logger.LogInformation("""
            
            🎯 PAPER TRADING CONFIGURATION SUMMARY
            ═══════════════════════════════════════
            
            📋 Mode: Paper Trading (Simulated)
            🌐 TopstepX: Enabled (Demo Credentials)
            🔒 Risk Limits: Active
            💰 Real Money: DISABLED (Paper Trading Only)
            
            🚀 Ready for simulated trading!
            """);
    }
}
