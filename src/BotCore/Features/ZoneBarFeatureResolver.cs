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
        "zone.distance_to_supply",
        "zone.distance_to_demand",
        "zone.supply_count",
        "zone.demand_count",
        "zone.avg_supply_strength",
        "zone.avg_demand_strength",
        "zone.net_pressure",
        "zone.in_supply",
        "zone.in_demand"
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
            
            // Distance to nearest zones
            _featureBus.Publish(symbol, now, "zone.distance_to_supply", (decimal)snapshot.DistanceToNearestSupply);
            _featureBus.Publish(symbol, now, "zone.distance_to_demand", (decimal)snapshot.DistanceToNearestDemand);
            
            // Zone counts
            _featureBus.Publish(symbol, now, "zone.supply_count", snapshot.ActiveSupplyCount);
            _featureBus.Publish(symbol, now, "zone.demand_count", snapshot.ActiveDemandCount);
            
            // Zone strength
            if (snapshot.ActiveSupplyCount > 0)
            {
                _featureBus.Publish(symbol, now, "zone.avg_supply_strength", (decimal)snapshot.AvgSupplyStrength);
            }
            if (snapshot.ActiveDemandCount > 0)
            {
                _featureBus.Publish(symbol, now, "zone.avg_demand_strength", (decimal)snapshot.AvgDemandStrength);
            }
            
            // Zone pressure
            _featureBus.Publish(symbol, now, "zone.net_pressure", (decimal)snapshot.NetPressure);
            
            // Zone interaction state
            _featureBus.Publish(symbol, now, "zone.in_supply", snapshot.InSupplyZone ? 1.0m : 0.0m);
            _featureBus.Publish(symbol, now, "zone.in_demand", snapshot.InDemandZone ? 1.0m : 0.0m);
            
            // Cache latest features for TryGetAsync
            lock (_lock)
            {
                var key = symbol;
                _latestFeatures[$"{key}::zone.distance_to_supply"] = snapshot.DistanceToNearestSupply;
                _latestFeatures[$"{key}::zone.distance_to_demand"] = snapshot.DistanceToNearestDemand;
                _latestFeatures[$"{key}::zone.supply_count"] = snapshot.ActiveSupplyCount;
                _latestFeatures[$"{key}::zone.demand_count"] = snapshot.ActiveDemandCount;
                _latestFeatures[$"{key}::zone.net_pressure"] = snapshot.NetPressure;
                _latestFeatures[$"{key}::zone.in_supply"] = snapshot.InSupplyZone ? 1.0 : 0.0;
                _latestFeatures[$"{key}::zone.in_demand"] = snapshot.InDemandZone ? 1.0 : 0.0;
                
                if (snapshot.ActiveSupplyCount > 0)
                    _latestFeatures[$"{key}::zone.avg_supply_strength"] = snapshot.AvgSupplyStrength;
                if (snapshot.ActiveDemandCount > 0)
                    _latestFeatures[$"{key}::zone.avg_demand_strength"] = snapshot.AvgDemandStrength;
            }

            _logger.LogTrace("[ZONE-BAR-RESOLVER] {Symbol}: Supply={Supply}, Demand={Demand}, Pressure={Pressure:F3}",
                symbol, snapshot.ActiveSupplyCount, snapshot.ActiveDemandCount, snapshot.NetPressure);

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
