using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using TradingBot.Abstractions;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// System health monitoring service with comprehensive logging
/// Tracks service lifecycle, memory usage, thread pool stats, GC metrics
/// </summary>
public class SystemHealthMonitoringService : IHostedService
{
    private readonly ILogger<SystemHealthMonitoringService> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly IServiceProvider _serviceProvider;
    private readonly Timer _healthCheckTimer;
    private readonly Timer _performanceTimer;
    private readonly Process _currentProcess;

    private long _lastGcCollectionCount = 0;
    private long _lastWorkingSet = 0;

    public SystemHealthMonitoringService(
        ILogger<SystemHealthMonitoringService> logger,
        ITradingLogger tradingLogger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _serviceProvider = serviceProvider;
        _currentProcess = Process.GetCurrentProcess();
        
        // Health checks every 60 seconds
        _healthCheckTimer = new Timer(PerformHealthCheck, null, 
            TimeSpan.FromSeconds(60), TimeSpan.FromSeconds(60));
            
        // Performance metrics every 5 minutes
        _performanceTimer = new Timer(CollectPerformanceMetrics, null,
            TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(5));
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "HealthMonitor", 
            "System health monitoring started");

        // Log initial system state
        await LogSystemInitialization();
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "HealthMonitor", 
            "System health monitoring stopped");
            
        _healthCheckTimer?.Dispose();
        _performanceTimer?.Dispose();
    }

    private async Task LogSystemInitialization()
    {
        try
        {
            var memoryInfo = GC.GetTotalMemory(false);
            var systemInfo = new
            {
                machineName = Environment.MachineName,
                osVersion = Environment.OSVersion.ToString(),
                processorCount = Environment.ProcessorCount,
                workingSet = _currentProcess.WorkingSet64,
                initialMemory = memoryInfo,
                startTime = _currentProcess.StartTime.ToString("yyyy-MM-dd HH:mm:ss UTC"),
                processId = _currentProcess.Id
            };

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "HealthMonitor", 
                "System initialization completed", systemInfo);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to log system initialization info");
        }
    }

    private async void PerformHealthCheck(object? state)
    {
        try
        {
            var healthData = new
            {
                timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC"),
                services = await CheckServiceHealth(),
                connections = await CheckConnectionHealth(),
                memory = await CheckMemoryHealth(),
                threadPool = await GetThreadPoolStatus()
            };

            await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, "HealthMonitor", 
                "Health check completed", healthData);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error during health check");
        }
    }

    private async void CollectPerformanceMetrics(object? state)
    {
        try
        {
            var currentWorkingSet = _currentProcess.WorkingSet64;
            var workingSetDelta = currentWorkingSet - _lastWorkingSet;
            _lastWorkingSet = currentWorkingSet;

            var gcCollectionCount = GC.CollectionCount(0) + GC.CollectionCount(1) + GC.CollectionCount(2);
            var gcDelta = gcCollectionCount - _lastGcCollectionCount;
            _lastGcCollectionCount = gcCollectionCount;

            var performanceData = new
            {
                memoryUsageMB = Math.Round(currentWorkingSet / 1024.0 / 1024.0, 2),
                memoryDeltaMB = Math.Round(workingSetDelta / 1024.0 / 1024.0, 2),
                gcCollections = gcCollectionCount,
                gcDelta = gcDelta,
                totalProcessorTime = _currentProcess.TotalProcessorTime.TotalMilliseconds,
                threadCount = _currentProcess.Threads.Count,
                handleCount = _currentProcess.HandleCount
            };

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "HealthMonitor", 
                "Performance metrics collected", performanceData);

            // Alert on high memory usage
            if (currentWorkingSet > 1024 * 1024 * 1024) // 1GB
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "HealthMonitor", 
                    "High memory usage detected", new { memoryUsageGB = Math.Round(currentWorkingSet / 1024.0 / 1024.0 / 1024.0, 2) });
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error collecting performance metrics");
        }
    }

    private async Task<object> CheckServiceHealth()
    {
        var tokenProvider = _serviceProvider.GetService<ITokenProvider>();
        
        var serviceChecks = new
        {
            tokenProvider = tokenProvider?.IsTokenValid ?? false,
            topstepXAdapter = CheckTopstepXAdapterHealth(),
            authenticationService = await CheckAuthenticationHealthAsync()
        };

        return serviceChecks;
    }

    private bool CheckTopstepXAdapterHealth()
    {
        try
        {
            var topstepXAdapter = _serviceProvider.GetService<ITopstepXAdapterService>();
            return topstepXAdapter?.IsConnected == true && topstepXAdapter?.ConnectionHealth >= 80.0;
        }
        catch
        {
            return false;
        }
    }

    private async Task<bool> CheckAuthenticationHealthAsync()
    {
        try
        {
            var tokenProvider = _serviceProvider.GetService<ITokenProvider>();
            if (tokenProvider == null) return false;
            
            var token = await tokenProvider.GetTokenAsync();
            return !string.IsNullOrEmpty(token);
        }
        catch
        {
            return false;
        }
    }

    private async Task<object> CheckConnectionHealth()
    {
        var connectionHealth = new
        {
            userHub = await CheckHubConnection("User"),
            marketHub = await CheckHubConnection("Market"),
            httpClient = await CheckHttpClientHealth()
        };

        return connectionHealth;
    }

    private async Task<object> CheckHubConnection(string hubType)
    {
        await Task.Delay(1); // Make method actually async
        
        try
        {
            var topstepXAdapter = _serviceProvider.GetService<ITopstepXAdapterService>();
            if (topstepXAdapter == null)
            {
                return new { connected = false, reason = "TopstepX adapter not available" };
            }

            var isConnected = topstepXAdapter.IsConnected;
            var health = topstepXAdapter.ConnectionHealth;

            return new { connected = isConnected, hubType, health };
        }
        catch (Exception ex)
        {
            return new { connected = false, error = ex.Message };
        }
    }

    private async Task<object> CheckHttpClientHealth()
    {
        try
        {
            // Quick health check to TopstepX API
            using var httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var response = await httpClient.GetAsync("https://api.topstepx.com/health");
            
            return new 
            { 
                apiAvailable = response.IsSuccessStatusCode,
                statusCode = (int)response.StatusCode,
                responseTimeMs = response.Headers.Date.HasValue 
                    ? (DateTime.UtcNow - response.Headers.Date.Value.DateTime).TotalMilliseconds 
                    : 0
            };
        }
        catch (Exception ex)
        {
            return new { apiAvailable = false, error = ex.Message };
        }
    }

    private async Task<object> CheckMemoryHealth()
    {
        await Task.Delay(1); // Make method actually async
        
        var memoryData = new
        {
            workingSetMB = Math.Round(_currentProcess.WorkingSet64 / 1024.0 / 1024.0, 2),
            privateMemoryMB = Math.Round(_currentProcess.PrivateMemorySize64 / 1024.0 / 1024.0, 2),
            gcTotalMemoryMB = Math.Round(GC.GetTotalMemory(false) / 1024.0 / 1024.0, 2),
            gen0Collections = GC.CollectionCount(0),
            gen1Collections = GC.CollectionCount(1),
            gen2Collections = GC.CollectionCount(2)
        };

        return memoryData;
    }

    private static async Task<object> GetThreadPoolStatus()
    {
        await Task.Delay(1); // Make method actually async
        
        ThreadPool.GetAvailableThreads(out var availableWorkerThreads, out var availableCompletionPortThreads);
        ThreadPool.GetMaxThreads(out var maxWorkerThreads, out var maxCompletionPortThreads);
        
        return new
        {
            availableWorkerThreads,
            availableCompletionPortThreads,
            maxWorkerThreads,
            maxCompletionPortThreads,
            usedWorkerThreads = maxWorkerThreads - availableWorkerThreads,
            usedCompletionPortThreads = maxCompletionPortThreads - availableCompletionPortThreads
        };
    }
}