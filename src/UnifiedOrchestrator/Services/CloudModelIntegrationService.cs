using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using CloudTrainer;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Wrapper service that connects CloudRlTrainerV2 to the unified model registry
/// Provides proper integration between Cloud layer and UnifiedOrchestrator layer
/// </summary>
internal sealed class CloudModelIntegrationService : BackgroundService
{
    private readonly ILogger<CloudModelIntegrationService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private CloudRlTrainerV2? _cloudTrainer;

    public CloudModelIntegrationService(
        ILogger<CloudModelIntegrationService> logger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("🔗 Cloud model integration service starting...");

        // Wait a moment for other services to initialize
        await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken).ConfigureAwait(false);

        try
        {
            // Get CloudRlTrainerV2 instance
            _cloudTrainer = _serviceProvider.GetService<CloudRlTrainerV2>();
            if (_cloudTrainer == null)
            {
                _logger.LogWarning("⚠️ CloudRlTrainerV2 not available - integration disabled");
                return;
            }

            // Monitor for model promotion events by watching the artifacts directory
            await MonitorModelPromotionsAsync(stoppingToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error in cloud model integration service");
        }

        _logger.LogInformation("Cloud model integration service stopped");
    }

    private async Task MonitorModelPromotionsAsync(CancellationToken stoppingToken)
    {
        var lastCurrentUpdate = DateTime.MinValue;
        
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Check if artifacts/current has been updated
                var currentDir = "artifacts/current";
                if (System.IO.Directory.Exists(currentDir))
                {
                    var currentUpdate = System.IO.Directory.GetLastWriteTimeUtc(currentDir);
                    
                    if (currentUpdate > lastCurrentUpdate && lastCurrentUpdate != DateTime.MinValue)
                    {
                        _logger.LogInformation("🔄 Model promotion detected in artifacts/current");
                        
                        // Calculate SHA from directory modification time and contents
                        var sha = CalculateDirectorySha(currentDir);
                        
                        // Notify model registry
                        await NotifyModelRegistryAsync(sha).ConfigureAwait(false);
                        
                        // Start canary monitoring
                        StartCanaryMonitoring(sha);
                    }
                    
                    lastCurrentUpdate = currentUpdate;
                }

                // Check every 30 seconds for changes
                await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error monitoring model promotions");
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken).ConfigureAwait(false);
            }
        }
    }

    private async Task NotifyModelRegistryAsync(string sha)
    {
        try
        {
            var modelRegistry = _serviceProvider.GetService<IModelRegistry>();
            if (modelRegistry != null)
            {
                modelRegistry.NotifyUpdated(sha);
                _logger.LogInformation("🔔 Notified model registry for hot-reload: {Sha}", sha);
            }
            else
            {
                _logger.LogWarning("⚠️ Model registry not available for notification");
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to notify model registry for SHA: {Sha}", sha);
        }
    }

    private void StartCanaryMonitoring(string sha)
    {
        try
        {
            var canaryWatchdog = _serviceProvider.GetService<CanaryWatchdog>();
            if (canaryWatchdog != null)
            {
                canaryWatchdog.StartCanary(sha);
                _logger.LogInformation("🕊️ Started canary monitoring for SHA: {Sha}", sha);
            }
            else
            {
                _logger.LogWarning("⚠️ Canary watchdog not available");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to start canary monitoring for SHA: {Sha}", sha);
        }
    }

    private string CalculateDirectorySha(string directory)
    {
        try
        {
            // Simple hash based on directory modification time and file count
            var lastWrite = System.IO.Directory.GetLastWriteTimeUtc(directory);
            var fileCount = System.IO.Directory.GetFiles(directory, "*", System.IO.SearchOption.AllDirectories).Length;
            var input = $"{lastWrite:O}-{fileCount}";
            
            using var sha256 = System.Security.Cryptography.SHA256.Create();
            var hash = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(input));
            return Convert.ToHexString(hash)[..8];
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to calculate directory SHA, using timestamp");
            return DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString("x8");
        }
    }

    public override void Dispose()
    {
        _logger.LogDebug("Disposing cloud model integration service");
        base.Dispose();
    }
}