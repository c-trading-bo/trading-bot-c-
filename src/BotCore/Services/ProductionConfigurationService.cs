using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.ComponentModel.DataAnnotations;

namespace BotCore.Services;

/// <summary>
/// Production-grade configuration management for ML/RL/Cloud trading system
/// Handles environment-specific settings, validation, and secure credential management
/// </summary>
public class ProductionConfigurationService : IValidateOptions<ProductionTradingConfig>
{
    private readonly IConfiguration _configuration;
    private readonly ILogger<ProductionConfigurationService> _logger;

    public ProductionConfigurationService(IConfiguration configuration, ILogger<ProductionConfigurationService> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    public ValidateOptionsResult Validate(string? name, ProductionTradingConfig options)
    {
        var errors = new List<string>();

        // Validate GitHub configuration
        if (string.IsNullOrWhiteSpace(options.GitHub.Token))
            errors.Add("GitHub.Token is required for model synchronization");
        
        if (string.IsNullOrWhiteSpace(options.GitHub.Repository))
            errors.Add("GitHub.Repository is required");
        
        if (string.IsNullOrWhiteSpace(options.GitHub.Owner))
            errors.Add("GitHub.Owner is required");

        // Validate ensemble configuration
        if (options.Ensemble.CloudWeight < 0 || options.Ensemble.CloudWeight > 1)
            errors.Add("Ensemble.CloudWeight must be between 0 and 1");
        
        if (options.Ensemble.LocalWeight < 0 || options.Ensemble.LocalWeight > 1)
            errors.Add("Ensemble.LocalWeight must be between 0 and 1");
        
        if (Math.Abs(options.Ensemble.CloudWeight + options.Ensemble.LocalWeight - 1.0) > 0.001)
            errors.Add("Ensemble.CloudWeight + LocalWeight must equal 1.0");

        // Validate model lifecycle
        if (options.ModelLifecycle.SyncIntervalMinutes < 1)
            errors.Add("ModelLifecycle.SyncIntervalMinutes must be at least 1");
        
        if (options.ModelLifecycle.ModelMaxAge.TotalHours < 1)
            errors.Add("ModelLifecycle.ModelMaxAge must be at least 1 hour");

        // Validate performance thresholds
        if (options.Performance.AccuracyThreshold < 0.1 || options.Performance.AccuracyThreshold > 1.0)
            errors.Add("Performance.AccuracyThreshold must be between 0.1 and 1.0");

        // Validate security settings
        if (options.Security.EnableEncryption && string.IsNullOrWhiteSpace(options.Security.EncryptionKey))
            errors.Add("Security.EncryptionKey is required when EnableEncryption is true");

        if (errors.Any())
        {
            var errorMessage = string.Join(", ", errors);
            _logger.LogError("❌ [CONFIG] Configuration validation failed: {Errors}", errorMessage);
            return ValidateOptionsResult.Fail(errors);
        }

        _logger.LogInformation("✅ [CONFIG] Production configuration validated successfully");
        return ValidateOptionsResult.Success;
    }

    /// <summary>
    /// Get environment-specific configuration with secure credential handling
    /// </summary>
    public ProductionTradingConfig GetValidatedConfiguration()
    {
        var config = new ProductionTradingConfig();
        _configuration.Bind("TradingBot", config);

        // Override with environment variables for sensitive data
        OverrideWithEnvironmentVariables(config);
        
        // Validate configuration
        var validationResult = Validate(null, config);
        if (validationResult.Failed)
        {
            throw new InvalidOperationException($"Configuration validation failed: {string.Join(", ", validationResult.Failures)}");
        }

        LogConfigurationSummary(config);
        return config;
    }

    private void OverrideWithEnvironmentVariables(ProductionTradingConfig config)
    {
        // GitHub token from environment (never store in appsettings.json)
        var githubToken = Environment.GetEnvironmentVariable("GITHUB_TOKEN") ?? 
                         Environment.GetEnvironmentVariable("TRADING_BOT_GITHUB_TOKEN");
        if (!string.IsNullOrWhiteSpace(githubToken))
        {
            config.GitHub.Token = githubToken;
            _logger.LogDebug("🔐 [CONFIG] GitHub token loaded from environment variable");
        }

        // TopstepX API credentials
        var topstepToken = Environment.GetEnvironmentVariable("TOPSTEP_API_TOKEN");
        if (!string.IsNullOrWhiteSpace(topstepToken))
        {
            config.TopstepX.ApiToken = topstepToken;
            _logger.LogDebug("🔐 [CONFIG] TopstepX API token loaded from environment variable");
        }

        // Encryption key
        var encryptionKey = Environment.GetEnvironmentVariable("TRADING_BOT_ENCRYPTION_KEY");
        if (!string.IsNullOrWhiteSpace(encryptionKey))
        {
            config.Security.EncryptionKey = encryptionKey;
            _logger.LogDebug("🔐 [CONFIG] Encryption key loaded from environment variable");
        }

        // Environment-specific overrides
        var environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production";
        config.Environment = environment;
        
        if (environment == "Development")
        {
            config.Ensemble.EnableFallback = true;
            config.Security.EnableEncryption = true;
            config.ModelLifecycle.SyncIntervalMinutes = 1; // Faster sync in dev
        }
    }

    private void LogConfigurationSummary(ProductionTradingConfig config)
    {
        _logger.LogInformation("🔧 [CONFIG] Production configuration summary:");
        _logger.LogInformation("  Environment: {Environment}", config.Environment);
        _logger.LogInformation("  GitHub Repository: {Owner}/{Repo}", config.GitHub.Owner, config.GitHub.Repository);
        _logger.LogInformation("  Ensemble Weights: Cloud={CloudWeight:P0}, Local={LocalWeight:P0}", 
            config.Ensemble.CloudWeight, config.Ensemble.LocalWeight);
        _logger.LogInformation("  Model Sync Interval: {Interval} minutes", config.ModelLifecycle.SyncIntervalMinutes);
        _logger.LogInformation("  Performance Threshold: {Threshold:P0}", config.Performance.AccuracyThreshold);
        _logger.LogInformation("  Security Enabled: {SecurityEnabled}", config.Security.EnableEncryption);
        _logger.LogInformation("  Fallback Enabled: {FallbackEnabled}", config.Ensemble.EnableFallback);
    }
}

#region Configuration Models

[Serializable]
public class ProductionTradingConfig
{
    public string Environment { get; set; } = "Production";
    public GitHubConfig GitHub { get; set; } = new();
    public TopstepXConfig TopstepX { get; set; } = new();
    public EnsembleConfig Ensemble { get; set; } = new();
    public ModelLifecycleConfig ModelLifecycle { get; set; } = new();
    public PerformanceConfig Performance { get; set; } = new();
    public SecurityConfig Security { get; set; } = new();
}

public class GitHubConfig
{
    [Required]
    public string Token { get; set; } = string.Empty;
    
    [Required]
    public string Owner { get; set; } = "trading-bot";
    
    [Required]
    public string Repository { get; set; } = "ml-models";
    
    public string Branch { get; set; } = "main";
    public string WorkflowsPath { get; set; } = ".github/workflows";
    public int MaxConcurrentDownloads { get; set; } = 3;
}

public class TopstepXConfig
{
    public string ApiToken { get; set; } = string.Empty;
    public string BaseUrl { get; set; } = "https://api.topstepx.com";
    public string UserHubUrl { get; set; } = "https://rtc.topstepx.com/hubs/user";
    public string MarketHubUrl { get; set; } = "https://rtc.topstepx.com/hubs/market";
    public TimeSpan RequestTimeout { get; set; } = TimeSpan.FromSeconds(30);
}

public class EnsembleConfig
{
    [Range(0.0, 1.0)]
    public double CloudWeight { get; set; } = 0.7;
    
    [Range(0.0, 1.0)]
    public double LocalWeight { get; set; } = 0.3;
    
    public bool EnableFallback { get; set; } = true;
    public string FallbackStrategy { get; set; } = "conservative";
    public int MinModelsRequired { get; set; } = 2;
}

public class ModelLifecycleConfig
{
    public int SyncIntervalMinutes { get; set; } = 15;
    public TimeSpan ModelMaxAge { get; set; } = TimeSpan.FromHours(24);
    public int MaxModelsToKeep { get; set; } = 10;
    public bool AutoRetrain { get; set; } = true;
    public string ModelStoragePath { get; set; } = "models/production";
}

public class PerformanceConfig
{
    [Range(0.1, 1.0)]
    public double AccuracyThreshold { get; set; } = 0.6;
    
    public TimeSpan EvaluationWindow { get; set; } = TimeSpan.FromHours(4);
    public int MinTradesForEvaluation { get; set; } = 10;
    public bool EnablePerformanceLogging { get; set; } = true;
}

public class SecurityConfig
{
    public bool EnableEncryption { get; set; } = true;
    public string EncryptionKey { get; set; } = string.Empty;
    public bool EnableAuditLogging { get; set; } = true;
    public bool ValidateCertificates { get; set; } = true;
    public TimeSpan TokenRefreshInterval { get; set; } = TimeSpan.FromHours(1);
}

#endregion