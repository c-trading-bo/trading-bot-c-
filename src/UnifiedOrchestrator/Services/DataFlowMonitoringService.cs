using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using TradingBot.Abstractions;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Comprehensive data flow monitoring service that tracks live data reception, 
/// historical data processing, and identifies connection issues
/// </summary>
public class DataFlowMonitoringService : BackgroundService
{
    private readonly ILogger<DataFlowMonitoringService> _logger;
    private readonly IServiceProvider _serviceProvider;
    
    // Monitoring state
    private readonly ConcurrentQueue<DataFlowMetric> _liveDataMetrics = new();
    private readonly ConcurrentQueue<DataFlowMetric> _historicalDataMetrics = new();
    private readonly ConcurrentDictionary<string, ConnectionHealth> _connectionHealth = new();
    
    private DateTime _lastLiveDataReceived = DateTime.MinValue;
    private DateTime _lastHistoricalDataProcessed = DateTime.MinValue;
    private int _totalLiveDataEvents = 0;
    private int _totalHistoricalDataEvents = 0;

    public DataFlowMonitoringService(
        ILogger<DataFlowMonitoringService> logger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[DATA-MONITOR] Starting comprehensive data flow monitoring");
        
        // Wait for system initialization
        await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken).ConfigureAwait(false);
        
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await MonitorDataFlows(stoppingToken).ConfigureAwait(false);
                await CheckConnectionHealth(stoppingToken).ConfigureAwait(false);
                await ReportDataFlowStatus(stoppingToken).ConfigureAwait(false);
                
                // Monitor every 30 seconds
                await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[DATA-MONITOR] Error in monitoring loop");
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken).ConfigureAwait(false);
            }
        }
        
        _logger.LogInformation("[DATA-MONITOR] Data flow monitoring service stopped");
    }

    /// <summary>
    /// Monitor all data flows and track metrics
    /// </summary>
    private async Task MonitorDataFlows(CancellationToken cancellationToken)
    {
        using var scope = _serviceProvider.CreateScope();
        
        // Monitor SignalR connections
        await MonitorSignalRConnections(scope, cancellationToken).ConfigureAwait(false);
        
        // Monitor data integration service
        await MonitorDataIntegrationService(scope, cancellationToken).ConfigureAwait(false);
        
        // Monitor backtest learning service
        await MonitorBacktestLearningService(scope, cancellationToken).ConfigureAwait(false);
        
        // Clean up old metrics (keep last hour only)
        CleanupOldMetrics();
    }

    /// <summary>
    /// Monitor SignalR connection status and data reception
    /// </summary>
    private async Task MonitorSignalRConnections(IServiceScope scope, CancellationToken cancellationToken)
    {
        try
        {
            var topstepXAdapter = scope.ServiceProvider.GetService<ITopstepXAdapterService>();
            if (topstepXAdapter == null)
            {
                RecordConnectionHealth("TopstepXAdapter", false, "Service not available");
                return;
            }

            var isConnected = topstepXAdapter.IsConnected;
            var health = topstepXAdapter.ConnectionHealth;
            
            RecordConnectionHealth("TopstepXAdapter", isConnected && health >= 80, 
                $"Connected: {isConnected}, Health: {health}%");
            
            // If adapter is connected but we're not receiving data, this indicates a data flow issue
            if (isConnected)
            {
                var timeSinceLastData = DateTime.UtcNow - _lastLiveDataReceived;
                if (timeSinceLastData > TimeSpan.FromMinutes(5) && _lastLiveDataReceived != DateTime.MinValue)
                {
                    _logger.LogWarning("[DATA-MONITOR] ⚠️ LIVE DATA ISSUE: Market hub connected but no data received for {Minutes} minutes", 
                        timeSinceLastData.TotalMinutes);
                    RecordConnectionHealth("MarketHubData", false, 
                        $"No live data received for {timeSinceLastData.TotalMinutes:F1} minutes");
                }
                else if (_totalLiveDataEvents > 0)
                {
                    RecordConnectionHealth("MarketHubData", true, "Receiving live data");
                }
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA-MONITOR] Error monitoring SignalR connections");
            RecordConnectionHealth("SignalRManager", false, $"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Monitor data integration service status
    /// </summary>
    private async Task MonitorDataIntegrationService(IServiceScope scope, CancellationToken cancellationToken)
    {
        try
        {
            var dataIntegrationService = scope.ServiceProvider.GetService<UnifiedDataIntegrationService>();
            if (dataIntegrationService == null)
            {
                RecordConnectionHealth("DataIntegration", false, "Service not available");
                return;
            }

            var historicalStatus = await dataIntegrationService.CheckHistoricalDataAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            var liveStatus = await dataIntegrationService.CheckLiveDataAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            RecordConnectionHealth("HistoricalData", historicalStatus, 
                historicalStatus ? "Connected" : "Disconnected");
            RecordConnectionHealth("LiveDataIntegration", liveStatus, 
                liveStatus ? "Connected" : "Disconnected");

            // Check if both are working concurrently
            if (historicalStatus && liveStatus)
            {
                RecordConnectionHealth("ConcurrentDataProcessing", true, 
                    "Both historical and live data processing active");
            }
            else
            {
                RecordConnectionHealth("ConcurrentDataProcessing", false, 
                    $"Incomplete data processing - Historical: {historicalStatus}, Live: {liveStatus}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA-MONITOR] Error monitoring data integration service");
            RecordConnectionHealth("DataIntegration", false, $"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Monitor backtest learning service for historical data processing
    /// </summary>
    private async Task MonitorBacktestLearningService(IServiceScope scope, CancellationToken cancellationToken)
    {
        try
        {
            // Check if BacktestLearningService is running by looking for its effects
            var timeSinceLastHistorical = DateTime.UtcNow - _lastHistoricalDataProcessed;
            
            if (_lastHistoricalDataProcessed == DateTime.MinValue)
            {
                RecordConnectionHealth("HistoricalProcessing", false, "No historical data processing detected yet");
            }
            else if (timeSinceLastHistorical > TimeSpan.FromHours(6))
            {
                RecordConnectionHealth("HistoricalProcessing", false, 
                    $"No historical processing for {timeSinceLastHistorical.TotalHours:F1} hours");
            }
            else
            {
                RecordConnectionHealth("HistoricalProcessing", true, 
                    $"Last processed {timeSinceLastHistorical.TotalMinutes:F1} minutes ago");
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA-MONITOR] Error monitoring backtest learning service");
            RecordConnectionHealth("HistoricalProcessing", false, $"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Check overall connection health and identify issues
    /// </summary>
    private async Task CheckConnectionHealth(CancellationToken cancellationToken)
    {
        var healthyConnections = _connectionHealth.Values.Count(h => h.IsHealthy);
        var totalConnections = _connectionHealth.Count;
        var healthPercentage = totalConnections > 0 ? (double)healthyConnections / totalConnections : 0;

        if (healthPercentage < 0.7) // Less than 70% healthy
        {
            _logger.LogWarning("[DATA-MONITOR] ⚠️ SYSTEM HEALTH DEGRADED: {Healthy}/{Total} connections healthy ({Percentage:P0})", 
                healthyConnections, totalConnections, healthPercentage);
            
            // Log specific issues
            var unhealthyConnections = _connectionHealth
                .Where(kvp => !kvp.Value.IsHealthy)
                .ToList();
                
            foreach (var connection in unhealthyConnections)
            {
                _logger.LogWarning("[DATA-MONITOR] 🔴 {ConnectionName}: {Status}", 
                    connection.Key, connection.Value.StatusMessage);
            }
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Report comprehensive data flow status
    /// </summary>
    private async Task ReportDataFlowStatus(CancellationToken cancellationToken)
    {
        var now = DateTime.UtcNow;
        var liveDataCount = _liveDataMetrics.Count;
        var historicalDataCount = _historicalDataMetrics.Count;
        
        // Calculate data rates
        var liveDataRate = CalculateDataRate(_liveDataMetrics, TimeSpan.FromMinutes(1));
        var historicalDataRate = CalculateDataRate(_historicalDataMetrics, TimeSpan.FromMinutes(5));
        
        _logger.LogInformation(
            "[DATA-MONITOR] 📊 STATUS - Live: {LiveCount} events ({LiveRate:F1}/min), Historical: {HistoricalCount} events ({HistoricalRate:F1}/5min), Health: {Healthy}/{Total}",
            liveDataCount, liveDataRate, historicalDataCount, historicalDataRate,
            _connectionHealth.Values.Count(h => h.IsHealthy), _connectionHealth.Count);

        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Record live data reception event
    /// </summary>
    public void RecordLiveDataEvent(string source, string eventType, object data)
    {
        _totalLiveDataEvents++;
        _lastLiveDataReceived = DateTime.UtcNow;
        
        _liveDataMetrics.Enqueue(new DataFlowMetric
        {
            Timestamp = DateTime.UtcNow,
            Source = source,
            EventType = eventType,
            Data = data
        });
        
        _logger.LogDebug("[DATA-MONITOR] 📈 Live data event: {Source} - {EventType}", source, eventType);
    }

    /// <summary>
    /// Record historical data processing event
    /// </summary>
    public void RecordHistoricalDataEvent(string source, string eventType, object data)
    {
        _totalHistoricalDataEvents++;
        _lastHistoricalDataProcessed = DateTime.UtcNow;
        
        _historicalDataMetrics.Enqueue(new DataFlowMetric
        {
            Timestamp = DateTime.UtcNow,
            Source = source,
            EventType = eventType,
            Data = data
        });
        
        _logger.LogDebug("[DATA-MONITOR] 📚 Historical data event: {Source} - {EventType}", source, eventType);
    }

    /// <summary>
    /// Record connection health status
    /// </summary>
    private void RecordConnectionHealth(string connectionName, bool isHealthy, string statusMessage)
    {
        _connectionHealth.AddOrUpdate(connectionName, 
            new ConnectionHealth
            {
                ConnectionName = connectionName,
                IsHealthy = isHealthy,
                StatusMessage = statusMessage,
                LastChecked = DateTime.UtcNow
            },
            (key, existingHealth) =>
            {
                existingHealth.IsHealthy = isHealthy;
                existingHealth.StatusMessage = statusMessage;
                existingHealth.LastChecked = DateTime.UtcNow;
                return existingHealth;
            });
    }

    /// <summary>
    /// Calculate data rate per time period
    /// </summary>
    private double CalculateDataRate(ConcurrentQueue<DataFlowMetric> metrics, TimeSpan timeWindow)
    {
        var cutoff = DateTime.UtcNow - timeWindow;
        var recentMetrics = metrics.Where(m => m.Timestamp > cutoff).Count();
        return recentMetrics;
    }

    /// <summary>
    /// Clean up old metrics to prevent memory issues
    /// </summary>
    private void CleanupOldMetrics()
    {
        var cutoff = DateTime.UtcNow.AddHours(-1);
        
        // Clean live data metrics
        while (_liveDataMetrics.TryPeek(out var liveMetric) && liveMetric.Timestamp < cutoff)
        {
            _liveDataMetrics.TryDequeue(out _);
        }
        
        // Clean historical data metrics
        while (_historicalDataMetrics.TryPeek(out var historicalMetric) && historicalMetric.Timestamp < cutoff)
        {
            _historicalDataMetrics.TryDequeue(out _);
        }
    }

    /// <summary>
    /// Get comprehensive data flow status report
    /// </summary>
    public DataFlowStatusReport GetStatusReport()
    {
        var now = DateTime.UtcNow;
        var liveDataCount = _liveDataMetrics.Count;
        var historicalDataCount = _historicalDataMetrics.Count;
        var liveDataRate = CalculateDataRate(_liveDataMetrics, TimeSpan.FromMinutes(1));
        var historicalDataRate = CalculateDataRate(_historicalDataMetrics, TimeSpan.FromMinutes(5));
        
        return new DataFlowStatusReport
        {
            GeneratedAt = now,
            TotalLiveDataEvents = _totalLiveDataEvents,
            TotalHistoricalDataEvents = _totalHistoricalDataEvents,
            LastLiveDataReceived = _lastLiveDataReceived,
            LastHistoricalDataProcessed = _lastHistoricalDataProcessed,
            LiveDataRatePerMinute = liveDataRate,
            HistoricalDataRatePerFiveMinutes = historicalDataRate,
            ConnectionHealthStatus = _connectionHealth.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
            HealthyConnectionCount = _connectionHealth.Values.Count(h => h.IsHealthy),
            TotalConnectionCount = _connectionHealth.Count
        };
    }
}

/// <summary>
/// Data flow metric for tracking events
/// </summary>
public class DataFlowMetric
{
    public DateTime Timestamp { get; set; }
    public string Source { get; set; } = string.Empty;
    public string EventType { get; set; } = string.Empty;
    public object? Data { get; set; }
}

/// <summary>
/// Connection health information
/// </summary>
public class ConnectionHealth
{
    public string ConnectionName { get; set; } = string.Empty;
    public bool IsHealthy { get; set; }
    public string StatusMessage { get; set; } = string.Empty;
    public DateTime LastChecked { get; set; }
}

/// <summary>
/// Comprehensive data flow status report
/// </summary>
public class DataFlowStatusReport
{
    public DateTime GeneratedAt { get; set; }
    public int TotalLiveDataEvents { get; set; }
    public int TotalHistoricalDataEvents { get; set; }
    public DateTime LastLiveDataReceived { get; set; }
    public DateTime LastHistoricalDataProcessed { get; set; }
    public double LiveDataRatePerMinute { get; set; }
    public double HistoricalDataRatePerFiveMinutes { get; set; }
    public Dictionary<string, ConnectionHealth> ConnectionHealthStatus { get; set; } = new();
    public int HealthyConnectionCount { get; set; }
    public int TotalConnectionCount { get; set; }
}