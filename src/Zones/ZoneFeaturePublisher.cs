using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using System.Diagnostics.CodeAnalysis;
using Zones;

namespace Zones;

public interface IFeatureBus 
{ 
    void Publish(string symbol, DateTime utc, string name, decimal value);
    void Publish(string symbol, DateTime utc, string name, double value);
}

/// <summary>
/// Event-driven zone feature publisher that reacts to zone updates and bar events
/// instead of polling on a timer. Part of Phase 4 reactive architecture refactoring.
/// </summary>
public sealed class ZoneFeaturePublisher : IHostedService, IDisposable
{
    private readonly IZoneFeatureSource _zones; 
    private readonly IFeatureBus? _bus; 
    private readonly ILogger<ZoneFeaturePublisher> _log;
    private readonly IServiceProvider _serviceProvider;
    private readonly int _emitEvery;
    private int _barCount;
    private object? _marketDataService;
    private bool _disposed;
    
    public ZoneFeaturePublisher(
        IZoneFeatureSource zones, 
        IFeatureBus? bus, 
        ILogger<ZoneFeaturePublisher> log, 
        IServiceProvider serviceProvider,
        [NotNull] IConfiguration cfg)
    { 
        ArgumentNullException.ThrowIfNull(cfg);
        
        _zones = zones; 
        _bus = bus; 
        _log = log;
        _serviceProvider = serviceProvider;
        _emitEvery = cfg.GetValue("Zone:EmitFeatureEveryBars", 1); 
        _barCount = 0;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (_bus == null)
        {
            LogNoBusConfigured(_log, null);
            return Task.CompletedTask;
        }

        // Subscribe to market data events to trigger zone feature publishing
        // This makes the service reactive instead of polling-based
        try
        {
            // Get the enhanced market data service dynamically to avoid circular dependencies
            var marketDataServiceType = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(a => a.GetTypes())
                .FirstOrDefault(t => t.Name == "IEnhancedMarketDataFlowService");

            if (marketDataServiceType != null)
            {
                _marketDataService = _serviceProvider.GetService(marketDataServiceType);
                if (_marketDataService != null)
                {
                    // Subscribe to OnMarketDataReceived event using reflection
                    var eventInfo = _marketDataService.GetType().GetEvent("OnMarketDataReceived");
                    if (eventInfo != null)
                    {
                        var handler = new Action<string, object>(OnMarketDataReceived);
                        eventInfo.AddEventHandler(_marketDataService, handler);
                        _log.LogInformation("[ZONE-FEATURES] Event-driven publisher activated - listening for market data events");
                    }
                }
            }

            if (_marketDataService == null)
            {
                _log.LogWarning("[ZONE-FEATURES] Market data service not available - zone features will not be published");
            }
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[ZONE-FEATURES] Error subscribing to market data events");
        }

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        Dispose();
        return Task.CompletedTask;
    }

    /// <summary>
    /// Event handler for market data updates - publishes zone features reactively
    /// </summary>
    private void OnMarketDataReceived(string symbol, object data)
    {
        // Only publish on configured intervals (every N bars)
        _barCount++;
        if (_barCount % _emitEvery != 0) return;

        try
        {
            PublishZoneFeatures();
        }
        catch (InvalidOperationException ex)
        {
            LogPublishError(_log, ex);
        }
        catch (ArgumentException ex)
        {
            LogPublishError(_log, ex);
        }
        catch (TimeoutException ex)
        {
            LogPublishError(_log, ex);
        }
    }

    private void PublishZoneFeatures()
    {
        if (_bus == null) return;
        
        foreach (var symbol in _tracked)
        {
            var (dmd, sup, breakout, press) = _zones.GetFeatures(symbol);
            var now = DateTime.UtcNow;
            _bus.Publish(symbol, now, "zone.dist_to_demand_atr", dmd);
            _bus.Publish(symbol, now, "zone.dist_to_supply_atr", sup);
            _bus.Publish(symbol, now, "zone.breakout_score", breakout);
            _bus.Publish(symbol, now, "zone.pressure", press);
        }
    }

    private readonly string[] _tracked = new[] { "ES", "NQ" };
    
    public void Dispose()
    {
        if (!_disposed)
        {
            if (_marketDataService != null)
            {
                // Unsubscribe from events
                try
                {
                    var eventInfo = _marketDataService.GetType().GetEvent("OnMarketDataReceived");
                    if (eventInfo != null)
                    {
                        var handler = new Action<string, object>(OnMarketDataReceived);
                        eventInfo.RemoveEventHandler(_marketDataService, handler);
                    }
                }
                catch (Exception ex)
                {
                    _log.LogError(ex, "[ZONE-FEATURES] Error unsubscribing from market data events");
                }
            }
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
    
    // Logger message delegates for performance
    private static readonly Action<ILogger, Exception?> LogNoBusConfigured = 
        LoggerMessage.Define(LogLevel.Information, new EventId(1), "[ZONE-FEATURES] No feature bus configured, zone features not published");
    
    private static readonly Action<ILogger, Exception?> LogPublishError = 
        LoggerMessage.Define(LogLevel.Error, new EventId(2), "[ZONE-FEATURES] Error publishing zone features");
}