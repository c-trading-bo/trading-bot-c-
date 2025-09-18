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
            "Log retention service started").ConfigureAwait(false);
            
        // Perform initial cleanup
        await PerformCleanupAsync().ConfigureAwait(false);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
            "Log retention service stopped").ConfigureAwait(false);
            
        _cleanupTimer?.Dispose();
    }

    private async void PerformCleanup(object? state)
    {
        await PerformCleanupAsync().ConfigureAwait(false);
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
                "Starting log cleanup process").ConfigureAwait(false);

            var totalFilesRemoved;
            var totalSizeFreed = 0L;

            // Clean up trading logs (30 days retention)
            totalFilesRemoved += await CleanupDirectory(
                Path.Combine(_options.LogDirectory, "trading"), 
                _options.LogRetentionDays).ConfigureAwait(false);

            totalFilesRemoved += await CleanupDirectory(
                Path.Combine(_options.LogDirectory, "system"), 
                _options.LogRetentionDays).ConfigureAwait(false);

            totalFilesRemoved += await CleanupDirectory(
                Path.Combine(_options.LogDirectory, "ml"), 
                _options.LogRetentionDays).ConfigureAwait(false);

            // Clean up debug logs (7 days retention)
            totalFilesRemoved += await CleanupDirectory(
                Path.Combine(_options.LogDirectory, "market"), 
                _options.DebugLogRetentionDays).ConfigureAwait(false);

            // Clean up critical alerts older than 90 days
            var criticalAlertsPath = Path.Combine(_options.LogDirectory, "critical_alerts.txt");
            if (File.Exists(criticalAlertsPath))
            {
                await CleanupCriticalAlerts(criticalAlertsPath).ConfigureAwait(false);
            }

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
                "Log cleanup completed", new
                {
                    filesRemoved = totalFilesRemoved,
                    sizeFreeBytes = totalSizeFreed
                }).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during log cleanup");
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "LogRetention", 
                $"Log cleanup failed: {ex.Message}").ConfigureAwait(false);
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

        var removedCount;
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
                $"Cleaned up {removedCount} files from {directoryPath}").ConfigureAwait(false);
        }

        return removedCount;
    }

    private async Task CleanupCriticalAlerts(string alertsPath)
    {
        try
        {
            var lines = await File.ReadAllLinesAsync(alertsPath).ConfigureAwait(false);
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
                await File.WriteAllLinesAsync(alertsPath, filteredLines).ConfigureAwait(false);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "LogRetention", 
                    $"Cleaned up {lines.Length - filteredLines.Length} old critical alerts").ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to cleanup critical alerts file");
        }
    }
}