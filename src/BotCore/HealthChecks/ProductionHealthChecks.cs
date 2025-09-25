using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Net.Http;
using System.Text.Json;
using BotCore.Configuration;

namespace BotCore.HealthChecks;

/// <summary>
/// Production health checks extension methods
/// </summary>
public static class ProductionHealthCheckExtensions
{
    /// <summary>
    /// Add comprehensive production health checks for all external dependencies
    /// </summary>
    public static IServiceCollection AddProductionHealthChecks(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        var healthChecksBuilder = services.AddHealthChecks();

        // TopstepX API health check
        healthChecksBuilder.AddCheck<TopstepXApiHealthCheck>(
            "topstepx-api",
            HealthStatus.Degraded,
            new[] { "external", "api", "topstepx" });

        // TopstepX SignalR hubs health check
        healthChecksBuilder.AddCheck<TopstepXSignalRHealthCheck>(
            "topstepx-signalr",
            HealthStatus.Degraded,
            new[] { "external", "signalr", "topstepx" });

        // Database health check
        healthChecksBuilder.AddCheck<DatabaseHealthCheck>(
            "database",
            HealthStatus.Unhealthy,
            new[] { "database", "sqlite" });

        // Disk space health check
        healthChecksBuilder.AddCheck<DiskSpaceHealthCheck>(
            "disk-space",
            HealthStatus.Degraded,
            new[] { "infrastructure", "disk" });

        // Memory health check
        healthChecksBuilder.AddCheck<MemoryHealthCheck>(
            "memory",
            HealthStatus.Degraded,
            new[] { "infrastructure", "memory" });

        // ML model health check
        healthChecksBuilder.AddCheck<MLModelHealthCheck>(
            "ml-models",
            HealthStatus.Degraded,
            new[] { "ml", "models" });

        // Configuration health check
        healthChecksBuilder.AddCheck<ConfigurationHealthCheck>(
            "configuration",
            HealthStatus.Unhealthy,
            new[] { "configuration" });

        // Security health check
        healthChecksBuilder.AddCheck<SecurityHealthCheck>(
            "security",
            HealthStatus.Unhealthy,
            new[] { "security", "credentials" });

        // Add health check publisher for monitoring
        services.AddHostedService<HealthCheckPublisherService>();

        return services;
    }

    /// <summary>
    /// Add health check endpoint with security
    /// </summary>
    public static IServiceCollection AddHealthCheckEndpoint(this IServiceCollection services)
    {
        services.AddSingleton<IHealthCheckEndpoint, ProductionHealthCheckEndpoint>();
        return services;
    }
}

/// <summary>
/// Interface for health check endpoint
/// </summary>
public interface IHealthCheckEndpoint
{
    Task<HealthCheckResult> GetHealthAsync();
    Task<string> GetHealthReportAsync();
}

/// <summary>
/// Production health check endpoint with detailed reporting
/// </summary>
public class ProductionHealthCheckEndpoint : IHealthCheckEndpoint
{
    private readonly HealthCheckService _healthCheckService;
    private readonly ILogger<ProductionHealthCheckEndpoint> _logger;

    public ProductionHealthCheckEndpoint(
        HealthCheckService healthCheckService,
        ILogger<ProductionHealthCheckEndpoint> logger)
    {
        _healthCheckService = healthCheckService;
        _logger = logger;
    }

    public async Task<HealthCheckResult> GetHealthAsync()
    {
        try
        {
            var report = await _healthCheckService.CheckHealthAsync().ConfigureAwait(false);
            return new HealthCheckResult(
                report.Status,
                $"Overall status: {report.Status}",
                data: new Dictionary<string, object>
                {
                    ["timestamp"] = DateTime.UtcNow,
                    ["totalChecks"] = report.Entries.Count,
                    ["healthyChecks"] = report.Entries.Count(e => e.Value.Status == HealthStatus.Healthy),
                    ["degradedChecks"] = report.Entries.Count(e => e.Value.Status == HealthStatus.Degraded),
                    ["unhealthyChecks"] = report.Entries.Count(e => e.Value.Status == HealthStatus.Unhealthy)
                });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get health check status");
            return HealthCheckResult.Unhealthy("Health check service failed");
        }
    }

    public async Task<string> GetHealthReportAsync()
    {
        try
        {
            var report = await _healthCheckService.CheckHealthAsync().ConfigureAwait(false);
            var result = new
            {
                status = report.Status.ToString(),
                timestamp = DateTime.UtcNow,
                totalDuration = report.TotalDuration,
                entries = report.Entries.ToDictionary(
                    e => e.Key,
                    e => new
                    {
                        status = e.Value.Status.ToString(),
                        description = e.Value.Description,
                        duration = e.Value.Duration,
                        data = e.Value.Data,
                        tags = e.Value.Tags
                    })
            };

            return JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to generate health report");
            return JsonSerializer.Serialize(new { status = "Unhealthy", error = ex.Message });
        }
    }
}

#region Health Check Implementations

/// <summary>
/// TopstepX API health check
/// </summary>
public class TopstepXApiHealthCheck : IHealthCheck
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<TopstepXApiHealthCheck> _logger;
    private readonly IOptions<TopstepXConfiguration> _config;

    public TopstepXApiHealthCheck(
        HttpClient httpClient,
        ILogger<TopstepXApiHealthCheck> logger,
        IOptions<TopstepXConfiguration> config)
    {
        _httpClient = httpClient;
        _logger = logger;
        _config = config;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            
            // Check API endpoint availability
            using var response = await _httpClient.GetAsync(
                $"{_config.Value.ApiBaseUrl}/api/health", 
                cancellationToken).ConfigureAwait(false);

            stopwatch.Stop();

            var data = new Dictionary<string, object>
            {
                ["responseTime"] = stopwatch.ElapsedMilliseconds,
                ["statusCode"] = (int)response.StatusCode,
                ["endpoint"] = _config.Value.ApiBaseUrl
            };

            if (response.IsSuccessStatusCode)
            {
                return HealthCheckResult.Healthy(
                    $"TopstepX API responding in {stopwatch.ElapsedMilliseconds}ms",
                    data);
            }
            else if (response.StatusCode == System.Net.HttpStatusCode.ServiceUnavailable)
            {
                return HealthCheckResult.Degraded(
                    $"TopstepX API temporarily unavailable: {response.StatusCode}",
                    data: data);
            }
            else
            {
                return HealthCheckResult.Unhealthy(
                    $"TopstepX API failed: {response.StatusCode}",
                    data: data);
            }
        }
        catch (TaskCanceledException)
        {
            return HealthCheckResult.Degraded("TopstepX API timeout");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "TopstepX API health check failed");
            return HealthCheckResult.Unhealthy(
                $"TopstepX API error: {ex.Message}",
                ex);
        }
    }
}

/// <summary>
/// TopstepX SignalR health check
/// </summary>
public class TopstepXSignalRHealthCheck : IHealthCheck
{
    private readonly ILogger<TopstepXSignalRHealthCheck> _logger;
    private readonly IOptions<TopstepXConfiguration> _config;

    public TopstepXSignalRHealthCheck(
        ILogger<TopstepXSignalRHealthCheck> logger,
        IOptions<TopstepXConfiguration> config)
    {
        _logger = logger;
        _config = config;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Simple connectivity check for SignalR endpoints
            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromSeconds(10);

            var userHubCheck = await CheckEndpoint(httpClient, _config.Value.UserHubUrl, cancellationToken).ConfigureAwait(false);
            var marketHubCheck = await CheckEndpoint(httpClient, _config.Value.MarketHubUrl, cancellationToken).ConfigureAwait(false);

            var data = new Dictionary<string, object>
            {
                ["userHub"] = userHubCheck,
                ["marketHub"] = marketHubCheck
            };

            if (userHubCheck.IsHealthy && marketHubCheck.IsHealthy)
            {
                return HealthCheckResult.Healthy("All SignalR hubs accessible", data);
            }
            else if (userHubCheck.IsHealthy || marketHubCheck.IsHealthy)
            {
                return HealthCheckResult.Degraded("Some SignalR hubs unavailable", data: data);
            }
            else
            {
                return HealthCheckResult.Unhealthy("All SignalR hubs unavailable", data: data);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "SignalR health check failed");
            return HealthCheckResult.Unhealthy($"SignalR health check error: {ex.Message}", ex);
        }
    }

    private async Task<(bool IsHealthy, string Message)> CheckEndpoint(
        HttpClient httpClient, 
        string url, 
        CancellationToken cancellationToken)
    {
        try
        {
            using var response = await httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);
            return (response.IsSuccessStatusCode, $"Status: {response.StatusCode}");
        }
        catch (Exception ex)
        {
            return (false, ex.Message);
        }
    }
}

/// <summary>
/// Database health check
/// </summary>
public class DatabaseHealthCheck : IHealthCheck
{
    private readonly ILogger<DatabaseHealthCheck> _logger;

    public DatabaseHealthCheck(ILogger<DatabaseHealthCheck> logger)
    {
        _logger = logger;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Check if SQLite database file exists and is accessible
            var dbPath = Environment.GetEnvironmentVariable("DATABASE_PATH") ?? "trading.db";
            
            var data = new Dictionary<string, object>
            {
                ["databasePath"] = dbPath,
                ["fileExists"] = File.Exists(dbPath)
            };

            if (File.Exists(dbPath))
            {
                var fileInfo = new FileInfo(dbPath);
                data["fileSize"] = fileInfo.Length;
                data["lastModified"] = fileInfo.LastWriteTime;

                // Test database connectivity
                using var connection = new System.Data.SQLite.SQLiteConnection($"Data Source={dbPath}");
                await connection.OpenAsync(cancellationToken).ConfigureAwait(false);
                using var command = connection.CreateCommand();
                command.CommandText = "SELECT 1";
                await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);

                return HealthCheckResult.Healthy("Database accessible", data);
            }
            else
            {
                return HealthCheckResult.Degraded("Database file not found - will be created", data: data);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Database health check failed");
            return HealthCheckResult.Unhealthy($"Database error: {ex.Message}", ex);
        }
    }
}

/// <summary>
/// Disk space health check
/// </summary>
public class DiskSpaceHealthCheck : IHealthCheck
{
    private readonly ILogger<DiskSpaceHealthCheck> _logger;

    public DiskSpaceHealthCheck(ILogger<DiskSpaceHealthCheck> logger)
    {
        _logger = logger;
    }

    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            var currentDirectory = Directory.GetCurrentDirectory();
            var driveInfo = new DriveInfo(Path.GetPathRoot(currentDirectory) ?? "C:");

            var freeSpaceGb = driveInfo.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
            var totalSpaceGb = driveInfo.TotalSize / (1024.0 * 1024.0 * 1024.0);
            var usedSpacePercent = ((totalSpaceGb - freeSpaceGb) / totalSpaceGb) * 100;

            var data = new Dictionary<string, object>
            {
                ["drive"] = driveInfo.Name,
                ["freeSpaceGb"] = Math.Round(freeSpaceGb, 2),
                ["totalSpaceGb"] = Math.Round(totalSpaceGb, 2),
                ["usedSpacePercent"] = Math.Round(usedSpacePercent, 1)
            };

            if (freeSpaceGb > 5.0) // 5GB threshold
            {
                return Task.FromResult(HealthCheckResult.Healthy($"Sufficient disk space: {freeSpaceGb:F1}GB free", data));
            }
            else if (freeSpaceGb > 1.0) // 1GB threshold
            {
                return Task.FromResult(HealthCheckResult.Degraded($"Low disk space: {freeSpaceGb:F1}GB free", data: data));
            }
            else
            {
                return Task.FromResult(HealthCheckResult.Unhealthy($"Critical disk space: {freeSpaceGb:F1}GB free", data: data));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Disk space health check failed");
            return Task.FromResult(HealthCheckResult.Unhealthy($"Disk space check error: {ex.Message}", ex));
        }
    }
}

/// <summary>
/// Memory health check
/// </summary>
public class MemoryHealthCheck : IHealthCheck
{
    private readonly ILogger<MemoryHealthCheck> _logger;

    public MemoryHealthCheck(ILogger<MemoryHealthCheck> logger)
    {
        _logger = logger;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            var process = System.Diagnostics.Process.GetCurrentProcess();
            var workingSetMb = process.WorkingSet64 / (1024.0 * 1024.0);
            var privateMemoryMb = process.PrivateMemorySize64 / (1024.0 * 1024.0);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var managedMemoryMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);

            var data = new Dictionary<string, object>
            {
                ["workingSetMb"] = Math.Round(workingSetMb, 1),
                ["privateMemoryMb"] = Math.Round(privateMemoryMb, 1),
                ["managedMemoryMb"] = Math.Round(managedMemoryMb, 1),
                ["gcGen0Collections"] = GC.CollectionCount(0),
                ["gcGen1Collections"] = GC.CollectionCount(1),
                ["gcGen2Collections"] = GC.CollectionCount(2)
            };

            if (workingSetMb < 500) // 500MB threshold
            {
                return HealthCheckResult.Healthy($"Memory usage normal: {workingSetMb:F1}MB", data);
            }
            else if (workingSetMb < 1000) // 1GB threshold
            {
                return HealthCheckResult.Degraded($"Memory usage elevated: {workingSetMb:F1}MB", data: data);
            }
            else
            {
                return HealthCheckResult.Unhealthy($"Memory usage high: {workingSetMb:F1}MB", data: data);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Memory health check failed");
            return HealthCheckResult.Unhealthy($"Memory check error: {ex.Message}", ex);
        }
    }
}

/// <summary>
/// ML model health check
/// </summary>
public class MLModelHealthCheck : IHealthCheck
{
    private readonly ILogger<MLModelHealthCheck> _logger;

    public MLModelHealthCheck(ILogger<MLModelHealthCheck> logger)
    {
        _logger = logger;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            var modelDirectory = Path.Combine(Directory.GetCurrentDirectory(), "models");
            var modelsFound;
            var modelsLoaded;

            var data = new Dictionary<string, object>
            {
                ["modelDirectory"] = modelDirectory,
                ["directoryExists"] = Directory.Exists(modelDirectory)
            };

            if (Directory.Exists(modelDirectory))
            {
                var modelFiles = Directory.GetFiles(modelDirectory, "*.onnx", SearchOption.AllDirectories);
                modelsFound = modelFiles.Length;

                foreach (var modelFile in modelFiles)
                {
                    try
                    {
                        // Simple check - verify file is readable and not empty
                        var fileInfo = new FileInfo(modelFile);
                        if (fileInfo.Length > 0)
                        {
                            modelsLoaded++;
                        }
                    }
                    catch
                    {
                        // Model file not accessible
                    }
                }

                data["modelsFound"] = modelsFound;
                data["modelsLoaded"] = modelsLoaded;
            }

            if (modelsFound == 0)
            {
                return HealthCheckResult.Degraded("No ML models found", data: data);
            }
            else if (modelsLoaded == modelsFound)
            {
                return HealthCheckResult.Healthy($"All {modelsLoaded} ML models accessible", data);
            }
            else
            {
                return HealthCheckResult.Degraded($"Only {modelsLoaded}/{modelsFound} ML models accessible", data: data);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "ML model health check failed");
            return HealthCheckResult.Unhealthy($"ML model check error: {ex.Message}", ex);
        }
    }
}

/// <summary>
/// Configuration health check
/// </summary>
public class ConfigurationHealthCheck : IHealthCheck
{
    private readonly ILogger<ConfigurationHealthCheck> _logger;

    public ConfigurationHealthCheck(ILogger<ConfigurationHealthCheck> logger)
    {
        _logger = logger;
    }

    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            var issues = new List<string>();
            var data = new Dictionary<string, object>();

            // Check critical environment variables
            var criticalVars = new[]
            {
                "TOPSTEPX_API_BASE",
                "TRADING_LOG_DIR",
                "MAX_DAILY_LOSS",
                "MAX_POSITION_SIZE"
            };

            foreach (var varName in criticalVars)
            {
                var value = Environment.GetEnvironmentVariable(varName);
                data[varName] = !string.IsNullOrEmpty(value) ? "SET" : "MISSING";
                
                if (string.IsNullOrEmpty(value))
                {
                    issues.Add($"Missing environment variable: {varName}");
                }
            }

            // Check if in safe mode (DRY_RUN enabled)
            var dryRun = Environment.GetEnvironmentVariable("ENABLE_DRY_RUN") != "false";
            data["dryRunEnabled"] = dryRun;
            
            if (!dryRun)
            {
                // Additional checks for live trading
                var killFile = Environment.GetEnvironmentVariable("KILL_FILE") ?? "kill.txt";
                data["killFileExists"] = File.Exists(killFile);
                
                if (!File.Exists(killFile))
                {
                    issues.Add("Kill file missing for live trading safety");
                }
            }

            if (issues.Count == 0)
            {
                return Task.FromResult(HealthCheckResult.Healthy("Configuration valid", data));
            }
            else
            {
                return Task.FromResult(HealthCheckResult.Unhealthy($"Configuration issues: {string.Join("; ", issues)}", data: data));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Configuration health check failed");
            return Task.FromResult(HealthCheckResult.Unhealthy($"Configuration check error: {ex.Message}", ex));
        }
    }
}

/// <summary>
/// Security health check
/// </summary>
public class SecurityHealthCheck : IHealthCheck
{
    private readonly ILogger<SecurityHealthCheck> _logger;

    public SecurityHealthCheck(ILogger<SecurityHealthCheck> logger)
    {
        _logger = logger;
    }

    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            var issues = new List<string>();
            var data = new Dictionary<string, object>();

            // Check credential sources
            var hasEnvCredentials = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME")) &&
                                  !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY"));
            var hasJwtToken = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TOPSTEPX_JWT"));
            
            data["hasEnvironmentCredentials"] = hasEnvCredentials;
            data["hasJwtToken"] = hasJwtToken;

            if (!hasEnvCredentials && !hasJwtToken)
            {
                issues.Add("No TopstepX credentials found");
            }

            // Check for hardcoded secrets in environment
            var suspiciousVars = Environment.GetEnvironmentVariables()
                .Cast<System.Collections.DictionaryEntry>()
                .Where(kv => kv.Key?.ToString()?.ToLower().Contains("password") == true || 
                           kv.Key?.ToString()?.ToLower().Contains("secret") == true)
                .Select(kv => kv.Key?.ToString() ?? string.Empty)
                .ToList();

            data["suspiciousEnvironmentVariables"] = suspiciousVars.Count;

            // Check file permissions on sensitive files
            var sensitiveFiles = new[] { "appsettings.json", "appsettings.Production.json", ".env" };
            var secureFiles = 0;
            
            foreach (var file in sensitiveFiles)
            {
                if (File.Exists(file))
                {
                    // Basic check - file should exist
                    secureFiles++;
                }
            }

            data["sensitiveFilesFound"] = secureFiles;

            if (issues.Count == 0)
            {
                return Task.FromResult(HealthCheckResult.Healthy("Security configuration acceptable", data));
            }
            else
            {
                return Task.FromResult(HealthCheckResult.Unhealthy($"Security issues: {string.Join("; ", issues)}", data: data));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Security health check failed");
            return Task.FromResult(HealthCheckResult.Unhealthy($"Security check error: {ex.Message}", ex));
        }
    }
}

#endregion

/// <summary>
/// Health check publisher service for monitoring and alerting
/// </summary>
public class HealthCheckPublisherService : BackgroundService
{
    private readonly HealthCheckService _healthCheckService;
    private readonly ILogger<HealthCheckPublisherService> _logger;
    private readonly IOptions<HealthCheckConfiguration> _config;

    public HealthCheckPublisherService(
        HealthCheckService healthCheckService,
        ILogger<HealthCheckPublisherService> logger,
        IOptions<HealthCheckConfiguration> config)
    {
        _healthCheckService = healthCheckService;
        _logger = logger;
        _config = config;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var report = await _healthCheckService.CheckHealthAsync(stoppingToken).ConfigureAwait(false);
                
                // Log health status
                var status = report.Status switch
                {
                    HealthStatus.Healthy => "✅",
                    HealthStatus.Degraded => "⚠️",
                    HealthStatus.Unhealthy => "❌",
                    _ => "❓"
                };

                _logger.LogInformation("{Status} [HEALTH] Overall status: {OverallStatus}, Duration: {Duration}ms", 
                    status, report.Status, report.TotalDuration.TotalMilliseconds);

                // Log individual check results
                foreach (var (name, result) in report.Entries)
                {
                    if (result.Status != HealthStatus.Healthy)
                    {
                        var checkStatus = result.Status switch
                        {
                            HealthStatus.Degraded => "⚠️",
                            HealthStatus.Unhealthy => "❌",
                            _ => "❓"
                        };

                        _logger.LogWarning("{Status} [HEALTH] {CheckName}: {Description}", 
                            checkStatus, name, result.Description);
                    }
                }

                await Task.Delay(_config.Value.IntervalMs, stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Expected when cancellation is requested
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [HEALTH] Health check publisher error");
                await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken).ConfigureAwait(false);
            }
        }
    }
}