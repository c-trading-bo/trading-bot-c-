using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace BotCore.Services;

/// <summary>
/// Interface for zone service
/// </summary>
public interface IZoneService
{
    Task<ZoneData?> GetLatestZonesAsync(string symbol = "ES");
    decimal GetNearestSupport(string symbol, decimal currentPrice);
    decimal GetNearestResistance(string symbol, decimal currentPrice);
    decimal GetOptimalStopLevel(string symbol, decimal entryPrice, bool isLong);
    decimal GetOptimalTargetLevel(string symbol, decimal entryPrice, bool isLong);
    decimal GetZoneBasedPositionSize(string symbol, decimal baseSize, decimal entryPrice, bool isLong);
    bool IsNearZone(string symbol, decimal price, decimal tolerance = 0.001m);
}

/// <summary>
/// Zone data model
/// </summary>
public class ZoneData
{
    public string Symbol { get; set; } = string.Empty;
    public decimal CurrentPrice { get; set; }
    public List<Zone> SupplyZones { get; set; } = new();
    public List<Zone> DemandZones { get; set; } = new();
    public decimal POC { get; set; }
    public ValueArea ValueArea { get; set; } = new();
    public KeyLevels KeyLevels { get; set; } = new();
    public DateTime GeneratedAt { get; set; }
}

public class Zone
{
    public decimal Price { get; set; }
    public List<decimal> Range { get; set; } = new();
    public int Strength { get; set; }
    public DateTime LastTest { get; set; }
    public int TouchCount { get; set; }
}

public class ValueArea
{
    public decimal High { get; set; }
    public decimal Low { get; set; }
}

public class KeyLevels
{
    public decimal NearestSupport { get; set; }
    public decimal NearestResistance { get; set; }
    public decimal StrongestSupport { get; set; }
    public decimal StrongestResistance { get; set; }
}

/// <summary>
/// Service for loading and using supply/demand zones for optimal trade placement
/// </summary>
public class ZoneService : IZoneService
{
    private readonly string _zonesPath;
    private readonly ILogger<ZoneService> _logger;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly Dictionary<string, ZoneData> _zoneCache = new();
    private DateTime _lastCacheUpdate = DateTime.MinValue;

    public ZoneService(ILogger<ZoneService> logger, string? zonesPath = null)
    {
        _logger = logger;
        _zonesPath = zonesPath ?? "Intelligence/data/zones/latest_zones.json";
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        };
    }

    /// <summary>
    /// Gets the latest zones for a symbol
    /// </summary>
    public async Task<ZoneData?> GetLatestZonesAsync(string symbol = "ES")
    {
        try
        {
            // Check cache first (refresh every 5 minutes)
            if (_zoneCache.ContainsKey(symbol) && 
                DateTime.UtcNow - _lastCacheUpdate < TimeSpan.FromMinutes(5))
            {
                return _zoneCache[symbol];
            }

            // Try symbol-specific file first
            var symbolZonesPath = Path.GetDirectoryName(_zonesPath) + $"/active_zones_{symbol}.json";
            var jsonPath = File.Exists(symbolZonesPath) ? symbolZonesPath : _zonesPath;

            if (!File.Exists(jsonPath))
            {
                _logger.LogDebug("[ZONES] Zones file not found: {Path}", jsonPath);
                return null;
            }

            var json = await File.ReadAllTextAsync(jsonPath);
            if (string.IsNullOrWhiteSpace(json))
            {
                _logger.LogDebug("[ZONES] Zones file is empty");
                return null;
            }

            var zoneData = JsonSerializer.Deserialize<ZoneData>(json, _jsonOptions);
            if (zoneData != null)
            {
                _zoneCache[symbol] = zoneData;
                _lastCacheUpdate = DateTime.UtcNow;
                
                _logger.LogInformation("[ZONES] Loaded zones for {Symbol}: {Supply} supply, {Demand} demand zones, POC={POC}",
                    symbol, zoneData.SupplyZones.Count, zoneData.DemandZones.Count, zoneData.POC);
            }

            return zoneData;
        }
        catch (JsonException ex)
        {
            _logger.LogWarning("[ZONES] Failed to parse zones file: {Error}", ex.Message);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Zones unavailable: {Error}", ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Gets the nearest support level below current price
    /// </summary>
    public decimal GetNearestSupport(string symbol, decimal currentPrice)
    {
        try
        {
            var zones = GetLatestZonesAsync(symbol).Result;
            if (zones == null) return currentPrice - GetDefaultOffset(symbol);

            // Find nearest demand zone below current price
            var nearestDemand = zones.DemandZones
                .Where(z => z.Price < currentPrice)
                .OrderByDescending(z => z.Price)
                .FirstOrDefault();

            if (nearestDemand != null)
                return nearestDemand.Price;

            // Fallback to key levels
            return zones.KeyLevels.NearestSupport > 0 ? zones.KeyLevels.NearestSupport : currentPrice - GetDefaultOffset(symbol);
        }
        catch
        {
            return currentPrice - GetDefaultOffset(symbol);
        }
    }

    /// <summary>
    /// Gets the nearest resistance level above current price
    /// </summary>
    public decimal GetNearestResistance(string symbol, decimal currentPrice)
    {
        try
        {
            var zones = GetLatestZonesAsync(symbol).Result;
            if (zones == null) return currentPrice + GetDefaultOffset(symbol);

            // Find nearest supply zone above current price
            var nearestSupply = zones.SupplyZones
                .Where(z => z.Price > currentPrice)
                .OrderBy(z => z.Price)
                .FirstOrDefault();

            if (nearestSupply != null)
                return nearestSupply.Price;

            // Fallback to key levels
            return zones.KeyLevels.NearestResistance > 0 ? zones.KeyLevels.NearestResistance : currentPrice + GetDefaultOffset(symbol);
        }
        catch
        {
            return currentPrice + GetDefaultOffset(symbol);
        }
    }

    /// <summary>
    /// Gets optimal stop level BEYOND the nearest zone (to avoid stop hunting)
    /// </summary>
    public decimal GetOptimalStopLevel(string symbol, decimal entryPrice, bool isLong)
    {
        try
        {
            var zones = GetLatestZonesAsync(symbol).Result;
            if (zones == null) 
            {
                // Fallback to standard stop distance
                var defaultOffset = GetDefaultOffset(symbol);
                return isLong ? entryPrice - defaultOffset : entryPrice + defaultOffset;
            }

            if (isLong)
            {
                // For long positions, place stop BELOW the nearest demand zone
                var nearestDemand = zones.DemandZones
                    .Where(z => z.Price < entryPrice)
                    .OrderByDescending(z => z.Price)
                    .FirstOrDefault();

                if (nearestDemand != null)
                {
                    // Place stop below the zone's lower boundary
                    var buffer = GetTickSize(symbol) * 2; // 2 tick buffer
                    return nearestDemand.Range[0] - buffer; // Range[0] is lower bound
                }
            }
            else
            {
                // For short positions, place stop ABOVE the nearest supply zone
                var nearestSupply = zones.SupplyZones
                    .Where(z => z.Price > entryPrice)
                    .OrderBy(z => z.Price)
                    .FirstOrDefault();

                if (nearestSupply != null)
                {
                    // Place stop above the zone's upper boundary
                    var buffer = GetTickSize(symbol) * 2; // 2 tick buffer
                    return nearestSupply.Range[1] + buffer; // Range[1] is upper bound
                }
            }

            // Fallback to standard stop
            var fallbackOffset = GetDefaultOffset(symbol);
            return isLong ? entryPrice - fallbackOffset : entryPrice + fallbackOffset;
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error calculating stop level: {Error}", ex.Message);
            var fallbackStopOffset = GetDefaultOffset(symbol);
            return isLong ? entryPrice - fallbackStopOffset : entryPrice + fallbackStopOffset;
        }
    }

    /// <summary>
    /// Gets optimal target level at the next significant zone
    /// </summary>
    public decimal GetOptimalTargetLevel(string symbol, decimal entryPrice, bool isLong)
    {
        try
        {
            var zones = GetLatestZonesAsync(symbol).Result;
            if (zones == null)
            {
                var fallbackTargetOffset = GetDefaultOffset(symbol) * 2; // 2:1 RR by default
                return isLong ? entryPrice + fallbackTargetOffset : entryPrice - fallbackTargetOffset;
            }

            if (isLong)
            {
                // For long positions, target the nearest supply zone above entry
                var targetSupply = zones.SupplyZones
                    .Where(z => z.Price > entryPrice)
                    .OrderBy(z => z.Price)
                    .FirstOrDefault();

                if (targetSupply != null)
                {
                    // Target the lower edge of the supply zone for better fill probability
                    return targetSupply.Range[0];
                }

                // Fallback to nearest resistance
                return GetNearestResistance(symbol, entryPrice);
            }
            else
            {
                // For short positions, target the nearest demand zone below entry
                var targetDemand = zones.DemandZones
                    .Where(z => z.Price < entryPrice)
                    .OrderByDescending(z => z.Price)
                    .FirstOrDefault();

                if (targetDemand != null)
                {
                    // Target the upper edge of the demand zone
                    return targetDemand.Range[1];
                }

                // Fallback to nearest support
                return GetNearestSupport(symbol, entryPrice);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error calculating target level: {Error}", ex.Message);
            var fallbackTargetOffset2 = GetDefaultOffset(symbol) * 2;
            return isLong ? entryPrice + fallbackTargetOffset2 : entryPrice - fallbackTargetOffset2;
        }
    }

    /// <summary>
    /// Gets zone-based position sizing multiplier
    /// </summary>
    public decimal GetZoneBasedPositionSize(string symbol, decimal baseSize, decimal entryPrice, bool isLong)
    {
        try
        {
            var zones = GetLatestZonesAsync(symbol).Result;
            if (zones == null) return baseSize;

            decimal multiplier = 1.0m;

            // Check if entry is near a strong zone
            var nearbyZones = (isLong ? zones.DemandZones : zones.SupplyZones)
                .Where(z => Math.Abs(z.Price - entryPrice) / entryPrice < 0.005m) // Within 0.5%
                .ToList();

            if (nearbyZones.Any())
            {
                var avgStrength = nearbyZones.Average(z => z.Strength);
                
                if (avgStrength >= 80)
                    multiplier = 1.5m; // 150% size near strong zones
                else if (avgStrength >= 70)
                    multiplier = 1.25m; // 125% size near medium zones
                else if (avgStrength < 50)
                    multiplier = 0.5m; // 50% size near weak zones
            }

            // Check if we're in "no man's land" between zones
            var nearestSupport = GetNearestSupport(symbol, entryPrice);
            var nearestResistance = GetNearestResistance(symbol, entryPrice);
            
            var supportDistance = Math.Abs(entryPrice - nearestSupport) / entryPrice;
            var resistanceDistance = Math.Abs(nearestResistance - entryPrice) / entryPrice;
            
            // If we're far from any zones (> 1% from both support and resistance)
            if (supportDistance > 0.01m && resistanceDistance > 0.01m)
            {
                multiplier *= 0.6m; // Reduce size in no man's land
                _logger.LogDebug("[ZONES] Reducing size in no man's land: {Symbol} @ {Price}", symbol, entryPrice);
            }

            return Math.Max(0.2m, Math.Min(baseSize * multiplier, baseSize * 2.0m));
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error calculating position size: {Error}", ex.Message);
            return baseSize;
        }
    }

    /// <summary>
    /// Checks if price is near a significant zone
    /// </summary>
    public bool IsNearZone(string symbol, decimal price, decimal tolerance = 0.001m)
    {
        try
        {
            var zones = GetLatestZonesAsync(symbol).Result;
            if (zones == null) return false;

            var allZones = zones.SupplyZones.Concat(zones.DemandZones);
            
            return allZones.Any(z => Math.Abs(z.Price - price) / price <= tolerance);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Gets default price offset for symbol
    /// </summary>
    private decimal GetDefaultOffset(string symbol)
    {
        return symbol.ToUpperInvariant() switch
        {
            "ES" => 20.0m,     // 20 points for ES
            "NQ" => 80.0m,     // 80 points for NQ
            "YM" => 200.0m,    // 200 points for YM
            "RTY" => 40.0m,    // 40 points for RTY
            _ => 20.0m
        };
    }

    /// <summary>
    /// Gets tick size for symbol
    /// </summary>
    private decimal GetTickSize(string symbol)
    {
        return symbol.ToUpperInvariant() switch
        {
            "ES" => 0.25m,
            "NQ" => 0.25m,
            "YM" => 1.0m,
            "RTY" => 0.1m,
            _ => 0.25m
        };
    }
}