using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Monitoring integration service providing REST API endpoints for log queries and metrics
/// Exposes Prometheus-compatible metrics and real-time log streaming
/// </summary>
public class MonitoringIntegrationService : IHostedService
{
    private readonly ILogger<MonitoringIntegrationService> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly IServiceProvider _serviceProvider;
    private WebApplication? _app;
    private readonly int _port;

    public MonitoringIntegrationService(
        ILogger<MonitoringIntegrationService> logger,
        ITradingLogger tradingLogger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _serviceProvider = serviceProvider;
        _port = int.Parse(Environment.GetEnvironmentVariable("MONITORING_PORT") ?? "8080");
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "MonitoringService", 
            $"Starting monitoring endpoints on port {_port}").ConfigureAwait(false);

        var builder = WebApplication.CreateBuilder();
        builder.Services.AddSingleton(_tradingLogger);
        builder.WebHost.UseUrls($"http://localhost:{_port}");
        
        // Disable request logging to reduce noise
        builder.Logging.AddFilter("Microsoft.AspNetCore", LogLevel.Warning);

        _app = builder.Build();

        // Configure endpoints
        ConfigureEndpoints();

        await _app.StartAsync(cancellationToken).ConfigureAwait(false);
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "MonitoringService", 
            $"Monitoring endpoints available at http://localhost:{_port}").ConfigureAwait(false);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "MonitoringService", 
            "Monitoring service stopping").ConfigureAwait(false);
            
        if (_app != null)
        {
            await _app.StopAsync(cancellationToken).ConfigureAwait(false);
            await _app.DisposeAsync().ConfigureAwait(false);
        }
    }

    private void ConfigureEndpoints()
    {
        if (_app == null) return;

        // Health check endpoint
        _app.MapGet("/health", async () =>
        {
            var metrics = await ((TradingLogger)_tradingLogger).GetPerformanceMetricsAsync().ConfigureAwait(false);
            return Results.Ok(new
            {
                status = "healthy",
                timestamp = DateTime.UtcNow,
                metrics = metrics.GetMetrics()
            });
        });

        // Prometheus metrics endpoint
        _app.MapGet("/metrics", async () =>
        {
            var metrics = await ((TradingLogger)_tradingLogger).GetPerformanceMetricsAsync().ConfigureAwait(false);
            var metricsData = metrics.GetMetrics();
            
            // Convert to Prometheus format
            var prometheusMetrics = "# HELP trading_log_entries_total Total number of log entries\n" +
                                   "# TYPE trading_log_entries_total counter\n" +
                                   $"trading_log_entries_total {metricsData.GetType().GetProperty("totalEntries")?.GetValue(metricsData)}\n" +
                                   "\n" +
                                   "# HELP trading_log_entries_per_second Log entries per second\n" +
                                   "# TYPE trading_log_entries_per_second gauge\n" +
                                   $"trading_log_entries_per_second {metricsData.GetType().GetProperty("entriesPerSecond")?.GetValue(metricsData)}";
            
            return Results.Text(prometheusMetrics, "text/plain");
        });

        // Recent logs endpoint
        _app.MapGet("/logs/recent", async (HttpContext context) =>
        {
            var count = int.Parse(context.Request.Query["count"].FirstOrDefault() ?? "100");
            var categoryStr = context.Request.Query["category"].FirstOrDefault();
            
            TradingLogCategory? category = null;
            if (!string.IsNullOrEmpty(categoryStr) && Enum.TryParse<TradingLogCategory>(categoryStr, true, out var parsedCategory))
            {
                category = parsedCategory;
            }

            var entries = await _tradingLogger.GetRecentEntriesAsync(count, category).ConfigureAwait(false);
            return Results.Ok(entries);
        });

        // Performance metrics endpoint
        _app.MapGet("/metrics/performance", async () =>
        {
            var metrics = await ((TradingLogger)_tradingLogger).GetPerformanceMetricsAsync().ConfigureAwait(false);
            return Results.Ok(metrics.GetMetrics());
        });

        // Force flush endpoint (for debugging)
        _app.MapPost("/logs/flush", async () =>
        {
            await _tradingLogger.FlushAsync().ConfigureAwait(false);
            return Results.Ok(new { status = "flushed", timestamp = DateTime.UtcNow });
        });
    }
}