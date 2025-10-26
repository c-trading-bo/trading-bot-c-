using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Diagnostics.CodeAnalysis;
using Zones;

namespace Zones;

public interface IFeatureBus 
{ 
    void Publish(string symbol, DateTime utc, string name, decimal value);
    void Publish(string symbol, DateTime utc, string name, double value);
}

public sealed class ZoneFeaturePublisher : BackgroundService
{
    private const int DefaultBarTimeframeMinutes = 5;
    
    private readonly IZoneFeatureSource _zones; 
    private readonly IFeatureBus? _bus; 
    private readonly ILogger<ZoneFeaturePublisher> _log; 
    private readonly TimeSpan _tf; 
    private readonly int _emitEvery;
    
    public ZoneFeaturePublisher(IZoneFeatureSource zones, IFeatureBus? bus, ILogger<ZoneFeaturePublisher> log, [NotNull] IConfiguration cfg)
    { 
        ArgumentNullException.ThrowIfNull(cfg);
        
        _zones = zones; 
        _bus = bus; 
        _log = log; 
        _tf = TimeSpan.FromMinutes(cfg.GetValue("Zone:BarTimeframeMinutes", DefaultBarTimeframeMinutes)); 
        _emitEvery = cfg.GetValue("Zone:EmitFeatureEveryBars", 1); 
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (_bus == null)
        {
            LogNoBusConfigured(_log, null);
            // FIXED: Wait indefinitely instead of returning to prevent shutdown signal
            await Task.Delay(Timeout.Infinite, stoppingToken).ConfigureAwait(false);
            return;
        }

        int k = 0; 
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(_tf, stoppingToken).ConfigureAwait(false); 
                k++;
                if (k % _emitEvery != 0) continue;
                
                await PublishZoneFeaturesAsync().ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Expected when cancellation is requested
                break;
            }
            catch (InvalidOperationException ex)
            {
                LogPublishError(_log, ex);
                await HandleRecoverableErrorAsync(stoppingToken).ConfigureAwait(false);
            }
            catch (ArgumentException ex)
            {
                LogPublishError(_log, ex);
                await HandleRecoverableErrorAsync(stoppingToken).ConfigureAwait(false);
            }
            catch (TimeoutException ex)
            {
                LogPublishError(_log, ex);
                await HandleRecoverableErrorAsync(stoppingToken).ConfigureAwait(false);
            }
        }
    }

    private async Task PublishZoneFeaturesAsync()
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
        
        await Task.CompletedTask.ConfigureAwait(false);
    }

    private static Task HandleRecoverableErrorAsync(CancellationToken stoppingToken)
    {
        const int ErrorRecoveryDelayMinutes = 1;
        return Task.Delay(TimeSpan.FromMinutes(ErrorRecoveryDelayMinutes), stoppingToken);
    }

    private readonly string[] _tracked = new[] { "ES", "NQ" };
    
    // Logger message delegates for performance
    private static readonly Action<ILogger, Exception?> LogNoBusConfigured = 
        LoggerMessage.Define(LogLevel.Information, new EventId(1), "[ZONE-FEATURES] No feature bus configured, zone features not published");
    
    private static readonly Action<ILogger, Exception?> LogPublishError = 
        LoggerMessage.Define(LogLevel.Error, new EventId(2), "[ZONE-FEATURES] Error publishing zone features");
}