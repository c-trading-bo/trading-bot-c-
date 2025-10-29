using Microsoft.Extensions.Logging;
using BotCore.Market;
using Zones;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Features;

/// <summary>
/// BACKTEST PARITY: Feature resolver that feeds bars to ZoneService and publishes zone features
/// Ensures historical and live bars both trigger zone detection
/// </summary>
public sealed class ZoneBarFeatureResolver : IFeatureResolver
{
    private readonly ILogger<ZoneBarFeatureResolver> _logger;
    private readonly IZoneService _zoneService;
    private readonly IFeatureBus _featureBus;
    private readonly Dictionary<string, double> _latestFeatures = new();
    private readonly object _lock = new object();

    private static readonly string[] FeatureKeys = new[]
    {
        "zone.dist_to_supply_atr",
        "zone.dist_to_demand_atr",
        "zone.breakout_score",
        "zone.pressure"
    };

    public ZoneBarFeatureResolver(
        ILogger<ZoneBarFeatureResolver> logger,
        IZoneService zoneService,
        IFeatureBus featureBus)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _zoneService = zoneService ?? throw new ArgumentNullException(nameof(zoneService));
        _featureBus = featureBus ?? throw new ArgumentNullException(nameof(featureBus));
    }

    public Task OnBarAsync(string symbol, object barData, CancellationToken cancellationToken = default)
    {
        try
        {
            if (barData is not Bar bar)
            {
                _logger.LogWarning("[ZONE-BAR-RESOLVER] Invalid bar data type for {Symbol}", symbol);
                return Task.CompletedTask;
            }

            // Feed bar to ZoneService for supply/demand analysis
            _zoneService.OnBar(
                symbol,
                bar.Start,
                bar.Open,
                bar.High,
                bar.Low,
                bar.Close,
                bar.Volume);

            // Get updated zone snapshot
            var snapshot = _zoneService.GetSnapshot(symbol);
            
            // Publish zone features to feature bus for Brain consumption
            var now = DateTime.UtcNow;
            
            // Distance to nearest zones (in ATR units)
            _featureBus.Publish(symbol, now, "zone.dist_to_supply_atr", snapshot.DistToSupplyAtr);
            _featureBus.Publish(symbol, now, "zone.dist_to_demand_atr", snapshot.DistToDemandAtr);
            
            // Zone quality scores
            _featureBus.Publish(symbol, now, "zone.breakout_score", snapshot.BreakoutScore);
            _featureBus.Publish(symbol, now, "zone.pressure", snapshot.ZonePressure);
            
            // Cache latest features for TryGetAsync
            lock (_lock)
            {
                var key = symbol;
                _latestFeatures[$"{key}::zone.dist_to_supply_atr"] = (double)snapshot.DistToSupplyAtr;
                _latestFeatures[$"{key}::zone.dist_to_demand_atr"] = (double)snapshot.DistToDemandAtr;
                _latestFeatures[$"{key}::zone.breakout_score"] = (double)snapshot.BreakoutScore;
                _latestFeatures[$"{key}::zone.pressure"] = (double)snapshot.ZonePressure;
            }

            _logger.LogTrace("[ZONE-BAR-RESOLVER] {Symbol}: DemandDist={Demand:F2}ATR, SupplyDist={Supply:F2}ATR, Pressure={Pressure:F3}",
                symbol, snapshot.DistToDemandAtr, snapshot.DistToSupplyAtr, snapshot.ZonePressure);

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-BAR-RESOLVER] Error processing bar for {Symbol}", symbol);
            throw;
        }
    }

    public Task<double?> TryGetAsync(string symbol, string featureKey, CancellationToken cancellationToken = default)
    {
        lock (_lock)
        {
            var key = $"{symbol}::{featureKey}";
            if (_latestFeatures.TryGetValue(key, out var value))
            {
                return Task.FromResult<double?>(value);
            }
        }
        return Task.FromResult<double?>(null);
    }

    public string[] GetAvailableFeatureKeys()
    {
        return FeatureKeys;
    }
}
