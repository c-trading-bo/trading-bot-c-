using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// File cleanup service implementing log retention policies
/// Automatically removes old log files based on configured retention periods
/// </summary>
public class LogRetentionService : IHostedService
{
    private readonly ILogger<LogRetentionService> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly TradingLoggerOptions _options;
    private readonly Timer _cleanupTimer;

    public LogRetentionService(
        ILogger<LogRetentionService> logger,
        ITradingLogger tradingLogger,
        IOptions<TradingLoggerOptions> options)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _options = options.Value;
        
        // Run cleanup daily at 2 AM
        var now = DateTime.Now;
        var next2AM = now.Date.AddDays(1).AddHours(2);
        var timeUntil2AM = next2AM - now;
        
        _cleanupTimer = new Timer(PerformCleanup, null, timeUntil2AM, TimeSpan.FromDays(1));
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
            "Log retention service started");
            
        // Perform initial cleanup
        await PerformCleanupAsync();
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
            "Log retention service stopped");
            
        _cleanupTimer?.Dispose();
    }

    private async void PerformCleanup(object? state)
    {
        await PerformCleanupAsync();
    }

    private async Task PerformCleanupAsync()
    {
        try
        {
            if (!Directory.Exists(_options.LogDirectory))
            {
                return;
            }

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
                "Starting log cleanup process");

            var totalFilesRemoved = 0;
            var totalSizeFreed = 0L;

            // Clean up trading logs (30 days retention)
            totalFilesRemoved += await CleanupDirectory(
                Path.Combine(_options.LogDirectory, "trading"), 
                _options.LogRetentionDays);

            totalFilesRemoved += await CleanupDirectory(
                Path.Combine(_options.LogDirectory, "system"), 
                _options.LogRetentionDays);

            totalFilesRemoved += await CleanupDirectory(
                Path.Combine(_options.LogDirectory, "ml"), 
                _options.LogRetentionDays);

            // Clean up debug logs (7 days retention)
            totalFilesRemoved += await CleanupDirectory(
                Path.Combine(_options.LogDirectory, "market"), 
                _options.DebugLogRetentionDays);

            // Clean up critical alerts older than 90 days
            var criticalAlertsPath = Path.Combine(_options.LogDirectory, "critical_alerts.txt");
            if (File.Exists(criticalAlertsPath))
            {
                await CleanupCriticalAlerts(criticalAlertsPath);
            }

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
                "Log cleanup completed", new
                {
                    filesRemoved = totalFilesRemoved,
                    sizeFreeBytes = totalSizeFreed
                });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during log cleanup");
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "LogRetention", 
                $"Log cleanup failed: {ex.Message}");
        }
    }

    private async Task<int> CleanupDirectory(string directoryPath, int retentionDays)
    {
        if (!Directory.Exists(directoryPath))
        {
            return 0;
        }

        var cutoffDate = DateTime.UtcNow.AddDays(-retentionDays);
        var files = Directory.GetFiles(directoryPath, "*", SearchOption.AllDirectories)
            .Where(f => File.GetCreationTimeUtc(f) < cutoffDate)
            .ToArray();

        var removedCount = 0;
        foreach (var file in files)
        {
            try
            {
                File.Delete(file);
                removedCount++;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to delete old log file: {FilePath}", file);
            }
        }

        if (removedCount > 0)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
                $"Cleaned up {removedCount} files from {directoryPath}");
        }

        return removedCount;
    }

    private async Task CleanupCriticalAlerts(string alertsPath)
    {
        try
        {
            var lines = await File.ReadAllLinesAsync(alertsPath);
            var cutoffDate = DateTime.UtcNow.AddDays(-90);
            
            var filteredLines = lines.Where(line =>
            {
                // Keep header lines and recent alerts
                if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line))
                    return true;
                
                // Try to parse timestamp from alert line
                if (line.StartsWith("[") && line.Contains("]"))
                {
                    var timestampEnd = line.IndexOf("]");
                    var timestampStr = line[1..timestampEnd];
                    
                    if (DateTime.TryParse(timestampStr, out var alertTime))
                    {
                        return alertTime > cutoffDate;
                    }
                }
                
                return true; // Keep if we can't parse the date
            }).ToArray();

            if (filteredLines.Length < lines.Length)
            {
                await File.WriteAllLinesAsync(alertsPath, filteredLines);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
                    $"Cleaned up {lines.Length - filteredLines.Length} old critical alerts");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to cleanup critical alerts file");
        }
    }
}