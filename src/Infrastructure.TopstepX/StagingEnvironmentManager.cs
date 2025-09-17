using Microsoft.Extensions.Logging;
using System.Text.Json;
using TradingBot.Abstractions;
using Infrastructure.TopstepX;

namespace BotCore.Infrastructure;

/// <summary>
/// Manages staging environment setup and deployment configuration for TopstepX integration
/// </summary>
public class StagingEnvironmentManager
{
    private readonly ILogger<StagingEnvironmentManager> _logger;
    private readonly TopstepXCredentialManager _credentialManager;

    public StagingEnvironmentManager(
        ILogger<StagingEnvironmentManager> logger,
        TopstepXCredentialManager credentialManager)
    {
        _logger = logger;
        _credentialManager = credentialManager;
    }

    /// <summary>
    /// Configure staging environment to match TopstepX production setup
    /// </summary>
    public async Task<StagingEnvironmentResult> ConfigureStagingEnvironmentAsync()
    {
        var result = new StagingEnvironmentResult();
        
        try
        {
            _logger.LogInformation("🏗️ Configuring staging environment for TopstepX...");

            // Step 1: Validate credentials are available
            var credentialDiscovery = _credentialManager.DiscoverAllCredentialSources();
            if (!credentialDiscovery.HasAnyCredentials)
            {
                result.AddError("No TopstepX credentials available for staging environment");
                return result;
            }

            result.CredentialsSource = credentialDiscovery.RecommendedSource;
            result.HasValidCredentials = true;

            // Step 2: Configure environment variables for staging
            ConfigureEnvironmentVariables(credentialDiscovery.RecommendedCredentials!);
            result.EnvironmentConfigured = true;

            // Step 3: Setup TopstepX API endpoints for staging
            ConfigureTopstepXEndpoints();
            result.EndpointsConfigured = true;

            // Step 4: Initialize safety and risk management for staging
            ConfigureStagingRiskManagement();
            result.RiskManagementConfigured = true;

            // Step 5: Setup monitoring and logging for staging
            ConfigureStagingMonitoring();
            result.MonitoringConfigured = true;

            // Step 6: Validate connectivity to TopstepX services
            var connectivityResult = await ValidateTopstepXConnectivity();
            result.ConnectivityValidated = connectivityResult.IsSuccessful;
            result.ApiResponseTime = connectivityResult.ResponseTime;

            result.IsSuccessful = result.HasValidCredentials && 
                                result.EnvironmentConfigured && 
                                result.EndpointsConfigured && 
                                result.RiskManagementConfigured;

            _logger.LogInformation("✅ Staging environment configuration complete - Success: {Success}", result.IsSuccessful);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error configuring staging environment");
            result.AddError($"Configuration error: {ex.Message}");
        }

        return result;
    }

    private void ConfigureEnvironmentVariables(TopstepXCredentials credentials)
    {
        var stagingVars = new Dictionary<string, string>
        {
            // Core credentials
            ["TOPSTEPX_USERNAME"] = credentials.Username,
            ["TOPSTEPX_API_KEY"] = credentials.ApiKey,
            ["TOPSTEPX_JWT"] = credentials.JwtToken ?? "",
            ["TOPSTEPX_ACCOUNT_ID"] = credentials.AccountId ?? "",

            // Staging-specific settings
            ["BOT_MODE"] = "staging",
            ["AUTO_GO_LIVE"] = "false",
            ["EXECUTION_MODE"] = "DRY_RUN",
            ["ENVIRONMENT"] = "staging",
            
            // TopstepX endpoints
            ["TOPSTEPX_API_BASE"] = "https://api.topstepx.com",
            ["TOPSTEPX_RTC_BASE"] = "https://rtc.topstepx.com",
            ["RTC_USER_HUB"] = "https://rtc.topstepx.com/hubs/user",
            ["RTC_MARKET_HUB"] = "https://rtc.topstepx.com/hubs/market",

            // Enhanced safety for staging
            ["DAILY_LOSS_CAP_R"] = "1.0",
            ["PER_TRADE_R"] = "0.5",
            ["MAX_CONCURRENT"] = "1",
            ["CRITICAL_SYSTEM_ENABLE"] = "1",
            ["EXECUTION_VERIFICATION_ENABLE"] = "1",
            ["DISASTER_RECOVERY_ENABLE"] = "1",

            // Staging-specific monitoring
            ["ASPNETCORE_URLS"] = "https://localhost:5051",
            ["LOGGING_LEVEL"] = "Information",
            ["STAGING_MODE"] = "true"
        };

        foreach (var (key, value) in stagingVars)
        {
            Environment.SetEnvironmentVariable(key, value);
        }

        _logger.LogInformation("🔧 Configured {Count} environment variables for staging", stagingVars.Count);
    }

    private void ConfigureTopstepXEndpoints()
    {
        // Ensure all TopstepX endpoints are properly configured for staging
        var endpoints = new Dictionary<string, string>
        {
            ["TOPSTEPX_API_BASE"] = "https://api.topstepx.com",
            ["TOPSTEPX_RTC_BASE"] = "https://rtc.topstepx.com",
            ["RTC_USER_HUB"] = "https://rtc.topstepx.com/hubs/user",
            ["RTC_MARKET_HUB"] = "https://rtc.topstepx.com/hubs/market"
        };

        foreach (var (key, value) in endpoints)
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(key)))
            {
                Environment.SetEnvironmentVariable(key, value);
            }
        }

        _logger.LogInformation("🌐 TopstepX endpoints configured for staging");
    }

    private void ConfigureStagingRiskManagement()
    {
        // Configure conservative risk management for staging
        var riskSettings = new Dictionary<string, string>
        {
            ["DAILY_LOSS_CAP_R"] = "1.0",           // Lower loss cap for staging
            ["PER_TRADE_R"] = "0.5",                // Smaller position size
            ["MAX_CONCURRENT"] = "1",               // Single position only
            ["SLIPPAGE_TICKS_WARN"] = "1",          // More sensitive slippage warning
            ["MAX_CORRELATION_EXPOSURE"] = "0.5",   // Lower correlation exposure
            ["ES_NQ_MAX_COMBINED_EXPOSURE"] = "2500", // Reduced exposure
            ["EXECUTION_TIMEOUT_SECONDS"] = "5",    // Faster timeout
            ["MAX_SLIPPAGE_TICKS"] = "1"            // Stricter slippage control
        };

        foreach (var (key, value) in riskSettings)
        {
            Environment.SetEnvironmentVariable(key, value);
        }

        _logger.LogInformation("🛡️ Conservative risk management configured for staging");
    }

    private void ConfigureStagingMonitoring()
    {
        // Enhanced monitoring and logging for staging
        var monitoringSettings = new Dictionary<string, string>
        {
            ["ASPNETCORE_URLS"] = "https://localhost:5051",
            ["LOGGING_LEVEL"] = "Information",
            ["STRUCTURED_LOGGING"] = "true",
            ["HEALTH_CHECK_INTERVAL"] = "10",       // More frequent health checks
            ["PERFORMANCE_MONITORING"] = "true",
            ["STAGING_TELEMETRY"] = "true"
        };

        foreach (var (key, value) in monitoringSettings)
        {
            Environment.SetEnvironmentVariable(key, value);
        }

        _logger.LogInformation("📊 Enhanced monitoring configured for staging");
    }

    private async Task<ConnectivityResult> ValidateTopstepXConnectivity()
    {
        var result = new ConnectivityResult();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromSeconds(10);

            // Test API endpoint connectivity
            var apiBase = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
            var response = await httpClient.GetAsync($"{apiBase}/api/health");
            
            stopwatch.Stop();
            result.ResponseTime = stopwatch.ElapsedMilliseconds;
            result.IsSuccessful = response.IsSuccessStatusCode || response.StatusCode == System.Net.HttpStatusCode.Unauthorized; // 401 is expected without auth
            result.StatusCode = (int)response.StatusCode;

            _logger.LogInformation("🌐 TopstepX connectivity validated - Response time: {ResponseTime}ms, Status: {Status}", 
                result.ResponseTime, response.StatusCode);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            result.ResponseTime = stopwatch.ElapsedMilliseconds;
            result.ErrorMessage = ex.Message;
            _logger.LogWarning(ex, "⚠️ TopstepX connectivity check failed: {Error}", ex.Message);
        }

        return result;
    }

    /// <summary>
    /// Generate staging environment status report
    /// </summary>
    public async Task<StagingStatusReport> GenerateStatusReportAsync()
    {
        var report = new StagingStatusReport
        {
            GeneratedAt = DateTime.UtcNow,
            Environment = Environment.GetEnvironmentVariable("ENVIRONMENT") ?? "unknown"
        };

        // Check credential status
        var credDiscovery = _credentialManager.DiscoverAllCredentialSources();
        report.CredentialStatus = new CredentialStatusInfo
        {
            HasCredentials = credDiscovery.HasAnyCredentials,
            Source = credDiscovery.RecommendedSource ?? "none",
            SourceCount = credDiscovery.TotalSourcesFound
        };

        // Check environment configuration
        var requiredVars = new[] { "TOPSTEPX_API_BASE", "BOT_MODE", "ENVIRONMENT" };
        report.EnvironmentVariables = requiredVars.ToDictionary(
            var => var, 
            var => Environment.GetEnvironmentVariable(var) ?? "NOT_SET"
        );

        // Check connectivity
        var connectivity = await ValidateTopstepXConnectivity();
        report.ConnectivityStatus = connectivity;

        report.IsHealthy = report.CredentialStatus.HasCredentials && 
                          report.EnvironmentVariables.All(kv => kv.Value != "NOT_SET") &&
                          connectivity.IsSuccessful;

        return report;
    }
}

public class StagingEnvironmentResult
{
    public bool IsSuccessful { get; set; }
    public bool HasValidCredentials { get; set; }
    public bool EnvironmentConfigured { get; set; }
    public bool EndpointsConfigured { get; set; }
    public bool RiskManagementConfigured { get; set; }
    public bool MonitoringConfigured { get; set; }
    public bool ConnectivityValidated { get; set; }
    public string? CredentialsSource { get; set; }
    public long ApiResponseTime { get; set; }
    public List<string> Errors { get; private set; } = new();

    public void AddError(string error) => Errors.Add(error);
}

public class ConnectivityResult
{
    public bool IsSuccessful { get; set; }
    public long ResponseTime { get; set; }
    public int StatusCode { get; set; }
    public string? ErrorMessage { get; set; }
}

public class StagingStatusReport
{
    public DateTime GeneratedAt { get; set; }
    public string Environment { get; set; } = "";
    public bool IsHealthy { get; set; }
    public CredentialStatusInfo CredentialStatus { get; set; } = new();
    public Dictionary<string, string> EnvironmentVariables { get; set; } = new();
    public ConnectivityResult ConnectivityStatus { get; set; } = new();
}

public class CredentialStatusInfo
{
    public bool HasCredentials { get; set; }
    public string Source { get; set; } = "";
    public int SourceCount { get; set; }
}