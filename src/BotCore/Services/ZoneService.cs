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
    // Enhanced methods from problem statement
    Zone GetNearestZone(decimal price, string zoneType);
    string GetZoneContext(decimal price);
    decimal GetZoneAdjustedStopLoss(decimal entryPrice, string direction);
    decimal GetZoneAdjustedTarget(decimal entryPrice, string direction);
    Task RecordZoneInteraction(decimal price, string outcome);
}

/// <summary>
/// Enhanced Supply/Demand Service interface (as specified in problem statement)
/// </summary>
public interface ISupplyDemandService
{
    Task<ZoneData> LoadZonesAsync();
    Zone GetNearestZone(decimal price, string zoneType);
    bool IsNearZone(decimal price, decimal threshold = 0.002m);
    string GetZoneContext(decimal price);
    decimal GetZoneAdjustedStopLoss(decimal entryPrice, string direction);
    decimal GetZoneAdjustedTarget(decimal entryPrice, string direction);
    Task RecordZoneInteraction(decimal price, string outcome);
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
    public string Timestamp { get; set; } = string.Empty;  // For compatibility
    public NearestZone? NearestSupply { get; set; }
    public NearestZone? NearestDemand { get; set; }
    public ZoneStatistics? Statistics { get; set; }
}

public class NearestZone
{
    public decimal Price { get; set; }
    public decimal Distance { get; set; }
    public double DistancePercent { get; set; }
    public double Strength { get; set; }
}

public class ZoneStatistics
{
    public int TotalSupplyZones { get; set; }
    public int TotalDemandZones { get; set; }
    public int ActiveSupply { get; set; }
    public int ActiveDemand { get; set; }
}

/// <summary>
/// Zone interaction tracking for learning
/// </summary>
public class ZoneInteraction
{
    public DateTime Timestamp { get; set; }
    public decimal Price { get; set; }
    public string ZoneType { get; set; } = string.Empty;
    public decimal ZonePrice { get; set; }
    public string Outcome { get; set; } = string.Empty;
    public double ZoneStrength { get; set; }
    public string Direction { get; set; } = string.Empty;
}

public class Zone
{
    public string Type { get; set; } = string.Empty;  // "supply" or "demand"
    public decimal Price { get; set; }
    public List<decimal> Range { get; set; } = new();
    public decimal PriceLevel { get; set; }  // For compatibility with new format
    public decimal ZoneTop { get; set; }
    public decimal ZoneBottom { get; set; }
    public double Strength { get; set; }
    public DateTime LastTest { get; set; }
    public int TouchCount { get; set; }
    public int Touches { get; set; }  // For compatibility
    public int Holds { get; set; }
    public int Breaks { get; set; }
    public bool Active { get; set; } = true;
    public decimal Volume { get; set; }
    public DateTime CreatedDate { get; set; }
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
/// Enhanced with institutional-grade zone interaction tracking
/// </summary>
public class ZoneService : IZoneService, ISupplyDemandService
{
    private readonly string _zonesPath;
    private readonly ILogger<ZoneService> _logger;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly Dictionary<string, ZoneData> _zoneCache = new();
    private DateTime _lastCacheUpdate = DateTime.MinValue;
    private readonly List<ZoneInteraction> _interactions = new();
    private ZoneData? _currentZones;

    public ZoneService(ILogger<ZoneService> logger, string? zonesPath = null)
    {
        _logger = logger;
        _zonesPath = zonesPath ?? "Intelligence/data/zones/active_zones.json";
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

            // Try to parse as enhanced format first
            var zoneData = await ParseEnhancedZoneData(json, symbol);
            if (zoneData == null)
            {
                // Fallback to original format
                zoneData = JsonSerializer.Deserialize<ZoneData>(json, _jsonOptions);
            }

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
    /// Parse enhanced zone data format
    /// </summary>
    private Task<ZoneData?> ParseEnhancedZoneData(string json, string symbol)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;

            var zoneData = new ZoneData
            {
                Symbol = symbol,
                Timestamp = GetStringValue(root, "timestamp") ?? DateTime.UtcNow.ToString("O")
            };

            // Parse current price
            if (root.TryGetProperty("current_price", out var currentPriceElement))
            {
                zoneData.CurrentPrice = currentPriceElement.GetDecimal();
            }

            // Parse POC
            if (root.TryGetProperty("poc", out var pocElement) && pocElement.ValueKind != JsonValueKind.Null)
            {
                zoneData.POC = pocElement.GetDecimal();
            }

            // Parse supply zones
            if (root.TryGetProperty("supply_zones", out var supplyElement))
            {
                foreach (var zoneElement in supplyElement.EnumerateArray())
                {
                    var zone = ParseZoneElement(zoneElement, "supply");
                    if (zone != null) zoneData.SupplyZones.Add(zone);
                }
            }

            // Parse demand zones
            if (root.TryGetProperty("demand_zones", out var demandElement))
            {
                foreach (var zoneElement in demandElement.EnumerateArray())
                {
                    var zone = ParseZoneElement(zoneElement, "demand");
                    if (zone != null) zoneData.DemandZones.Add(zone);
                }
            }

            // Parse nearest zones
            if (root.TryGetProperty("nearest_supply", out var nearestSupplyElement) && nearestSupplyElement.ValueKind != JsonValueKind.Null)
            {
                zoneData.NearestSupply = ParseNearestZone(nearestSupplyElement);
            }

            if (root.TryGetProperty("nearest_demand", out var nearestDemandElement) && nearestDemandElement.ValueKind != JsonValueKind.Null)
            {
                zoneData.NearestDemand = ParseNearestZone(nearestDemandElement);
            }

            // Parse key levels (fallback from existing zones)
            zoneData.KeyLevels = new KeyLevels
            {
                NearestSupport = zoneData.NearestDemand?.Price ?? 0,
                NearestResistance = zoneData.NearestSupply?.Price ?? 0,
                StrongestSupport = zoneData.DemandZones.OrderByDescending(z => z.Strength).FirstOrDefault()?.PriceLevel ?? 0,
                StrongestResistance = zoneData.SupplyZones.OrderByDescending(z => z.Strength).FirstOrDefault()?.PriceLevel ?? 0
            };

            // Parse key_levels if present
            if (root.TryGetProperty("key_levels", out var keyLevelsElement))
            {
                if (keyLevelsElement.TryGetProperty("nearest_support", out var nsElement))
                    zoneData.KeyLevels.NearestSupport = nsElement.GetDecimal();
                if (keyLevelsElement.TryGetProperty("nearest_resistance", out var nrElement))
                    zoneData.KeyLevels.NearestResistance = nrElement.GetDecimal();
                if (keyLevelsElement.TryGetProperty("strongest_support", out var ssElement))
                    zoneData.KeyLevels.StrongestSupport = ssElement.GetDecimal();
                if (keyLevelsElement.TryGetProperty("strongest_resistance", out var srElement))
                    zoneData.KeyLevels.StrongestResistance = srElement.GetDecimal();
            }

            // Parse statistics
            if (root.TryGetProperty("statistics", out var statsElement))
            {
                zoneData.Statistics = new ZoneStatistics
                {
                    TotalSupplyZones = GetIntValue(statsElement, "total_supply_zones"),
                    TotalDemandZones = GetIntValue(statsElement, "total_demand_zones"),
                    ActiveSupply = GetIntValue(statsElement, "active_supply"),
                    ActiveDemand = GetIntValue(statsElement, "active_demand")
                };
            }

            // Set generated at
            if (DateTime.TryParse(zoneData.Timestamp, out var generatedAt))
            {
                zoneData.GeneratedAt = generatedAt;
            }

            return Task.FromResult(zoneData);
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Failed to parse enhanced format: {Error}", ex.Message);
            return Task.FromResult<ZoneData?>(null);
        }
    }

    /// <summary>
    /// Parse individual zone element
    /// </summary>
    private Zone? ParseZoneElement(JsonElement element, string type)
    {
        try
        {
            var zone = new Zone
            {
                Type = type,
                PriceLevel = GetDecimalValue(element, "price_level"),
                ZoneTop = GetDecimalValue(element, "zone_top"),
                ZoneBottom = GetDecimalValue(element, "zone_bottom"),
                Strength = GetDoubleValue(element, "strength"),
                Touches = GetIntValue(element, "touches"),
                Holds = GetIntValue(element, "holds"),
                Breaks = GetIntValue(element, "breaks"),
                Active = GetBoolValue(element, "active"),
                Volume = GetDecimalValue(element, "volume")
            };

            // Set Price for compatibility
            zone.Price = zone.PriceLevel;

            // Set Range for compatibility
            zone.Range = new List<decimal> { zone.ZoneBottom, zone.ZoneTop };

            // Set TouchCount for compatibility
            zone.TouchCount = zone.Touches;

            // Parse dates
            var lastTestStr = GetStringValue(element, "last_test");
            if (!string.IsNullOrEmpty(lastTestStr) && DateTime.TryParse(lastTestStr, out var lastTest))
            {
                zone.LastTest = lastTest;
            }

            var createdDateStr = GetStringValue(element, "created_date");
            if (!string.IsNullOrEmpty(createdDateStr) && DateTime.TryParse(createdDateStr, out var createdDate))
            {
                zone.CreatedDate = createdDate;
            }

            return zone;
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Failed to parse zone element: {Error}", ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Parse nearest zone element
    /// </summary>
    private NearestZone? ParseNearestZone(JsonElement element)
    {
        try
        {
            return new NearestZone
            {
                Price = GetDecimalValue(element, "price"),
                Distance = GetDecimalValue(element, "distance"),
                DistancePercent = GetDoubleValue(element, "distance_percent"),
                Strength = GetDoubleValue(element, "strength")
            };
        }
        catch
        {
            return null;
        }
    }

    // Helper methods for JSON parsing
    private string? GetStringValue(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var prop) && prop.ValueKind != JsonValueKind.Null
            ? prop.GetString() : null;
    }

    private decimal GetDecimalValue(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var prop) && prop.ValueKind != JsonValueKind.Null
            ? prop.GetDecimal() : 0m;
    }

    private double GetDoubleValue(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var prop) && prop.ValueKind != JsonValueKind.Null
            ? prop.GetDouble() : 0.0;
    }

    private int GetIntValue(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var prop) && prop.ValueKind != JsonValueKind.Null
            ? prop.GetInt32() : 0;
    }

    private bool GetBoolValue(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var prop) && prop.ValueKind != JsonValueKind.Null
            ? prop.GetBoolean() : true;
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

    // Enhanced methods from problem statement

    /// <summary>
    /// Get nearest zone to price (enhanced implementation)
    /// </summary>
    public Zone GetNearestZone(decimal price, string zoneType)
    {
        try
        {
            _currentZones ??= GetLatestZonesAsync("ES").Result;
            if (_currentZones == null) return new Zone();

            var zones = zoneType.ToLowerInvariant() == "supply"
                ? _currentZones.SupplyZones
                : _currentZones.DemandZones;

            if (!zones.Any()) return new Zone();

            if (zoneType.ToLowerInvariant() == "supply")
            {
                // Find closest supply above current price
                var aboveZones = zones.Where(z => z.PriceLevel > price).ToList();
                if (aboveZones.Any())
                {
                    return aboveZones.OrderBy(z => z.PriceLevel - price).First();
                }
            }
            else
            {
                // Find closest demand below current price
                var belowZones = zones.Where(z => z.PriceLevel < price).ToList();
                if (belowZones.Any())
                {
                    return belowZones.OrderByDescending(z => z.PriceLevel).First();
                }
            }

            return zones.OrderBy(z => Math.Abs(z.PriceLevel - price)).First();
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error getting nearest zone: {Error}", ex.Message);
            return new Zone();
        }
    }

    /// <summary>
    /// Check if price is near a zone (enhanced implementation)
    /// </summary>
    public bool IsNearZone(decimal price, decimal threshold = 0.002m)
    {
        try
        {
            _currentZones ??= GetLatestZonesAsync("ES").Result;
            if (_currentZones == null) return false;

            var allZones = _currentZones.SupplyZones.Concat(_currentZones.DemandZones);
            return allZones.Any(z => Math.Abs(z.PriceLevel - price) / price <= threshold);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Get zone context for current price
    /// </summary>
    public string GetZoneContext(decimal price)
    {
        try
        {
            _currentZones ??= GetLatestZonesAsync("ES").Result;
            if (_currentZones == null) return "No zone data available";

            var nearestSupply = GetNearestZone(price, "supply");
            var nearestDemand = GetNearestZone(price, "demand");

            var supplyDistance = Math.Abs(nearestSupply.PriceLevel - price);
            var demandDistance = Math.Abs(nearestDemand.PriceLevel - price);

            if (IsNearZone(price, 0.001m))
            {
                var nearestZone = supplyDistance < demandDistance ? nearestSupply : nearestDemand;
                return $"At {nearestZone.Type} zone (Strength: {nearestZone.Strength:F0})";
            }

            if (nearestSupply.PriceLevel > price && nearestDemand.PriceLevel < price)
            {
                return $"Between zones - Support: {nearestDemand.PriceLevel:F2} ({demandDistance:F2} away), Resistance: {nearestSupply.PriceLevel:F2} ({supplyDistance:F2} away)";
            }

            return "No significant zones nearby";
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error getting zone context: {Error}", ex.Message);
            return "Zone context unavailable";
        }
    }

    /// <summary>
    /// Get zone-adjusted stop loss level
    /// </summary>
    public decimal GetZoneAdjustedStopLoss(decimal entryPrice, string direction)
    {
        try
        {
            var isLong = direction.ToLowerInvariant() == "long" || direction.ToLowerInvariant() == "buy";

            _currentZones ??= GetLatestZonesAsync("ES").Result;
            if (_currentZones == null)
            {
                var defaultOffset = GetDefaultOffset("ES");
                return isLong ? entryPrice - defaultOffset : entryPrice + defaultOffset;
            }

            if (isLong)
            {
                // For long positions, place stop below nearest demand zone
                var nearestDemand = GetNearestZone(entryPrice, "demand");
                if (nearestDemand.PriceLevel > 0 && nearestDemand.PriceLevel < entryPrice)
                {
                    var buffer = GetTickSize("ES") * 3; // 3 tick buffer beyond zone
                    return nearestDemand.ZoneBottom - buffer;
                }
            }
            else
            {
                // For short positions, place stop above nearest supply zone
                var nearestSupply = GetNearestZone(entryPrice, "supply");
                if (nearestSupply.PriceLevel > 0 && nearestSupply.PriceLevel > entryPrice)
                {
                    var buffer = GetTickSize("ES") * 3; // 3 tick buffer beyond zone
                    return nearestSupply.ZoneTop + buffer;
                }
            }

            // Fallback to standard stop
            var fallbackOffset = GetDefaultOffset("ES");
            return isLong ? entryPrice - fallbackOffset : entryPrice + fallbackOffset;
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error calculating zone-adjusted stop: {Error}", ex.Message);
            var fallback = GetDefaultOffset("ES");
            var isLong = direction.ToLowerInvariant() == "long" || direction.ToLowerInvariant() == "buy";
            return isLong ? entryPrice - fallback : entryPrice + fallback;
        }
    }

    /// <summary>
    /// Get zone-adjusted target level
    /// </summary>
    public decimal GetZoneAdjustedTarget(decimal entryPrice, string direction)
    {
        try
        {
            var isLong = direction.ToLowerInvariant() == "long" || direction.ToLowerInvariant() == "buy";

            _currentZones ??= GetLatestZonesAsync("ES").Result;
            if (_currentZones == null)
            {
                var defaultTargetOffset = GetDefaultOffset("ES") * 2;
                return isLong ? entryPrice + defaultTargetOffset : entryPrice - defaultTargetOffset;
            }

            if (isLong)
            {
                // For long positions, target nearest supply zone above
                var nearestSupply = GetNearestZone(entryPrice, "supply");
                if (nearestSupply.PriceLevel > 0 && nearestSupply.PriceLevel > entryPrice)
                {
                    // Target slightly below zone for better fill probability
                    return nearestSupply.ZoneBottom;
                }
            }
            else
            {
                // For short positions, target nearest demand zone below
                var nearestDemand = GetNearestZone(entryPrice, "demand");
                if (nearestDemand.PriceLevel > 0 && nearestDemand.PriceLevel < entryPrice)
                {
                    // Target slightly above zone for better fill probability
                    return nearestDemand.ZoneTop;
                }
            }

            // Fallback to standard target
            var fallbackTargetOffset = GetDefaultOffset("ES") * 2;
            return isLong ? entryPrice + fallbackTargetOffset : entryPrice - fallbackTargetOffset;
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error calculating zone-adjusted target: {Error}", ex.Message);
            var fallback = GetDefaultOffset("ES") * 2;
            var isLong = direction.ToLowerInvariant() == "long" || direction.ToLowerInvariant() == "buy";
            return isLong ? entryPrice + fallback : entryPrice - fallback;
        }
    }

    /// <summary>
    /// Record zone interaction for learning
    /// </summary>
    public async Task RecordZoneInteraction(decimal price, string outcome)
    {
        try
        {
            _currentZones ??= GetLatestZonesAsync("ES").Result;
            if (_currentZones == null) return;

            // Find the zone that was touched
            var touchedZones = _currentZones.SupplyZones.Concat(_currentZones.DemandZones)
                .Where(z => price >= z.ZoneBottom && price <= z.ZoneTop)
                .ToList();

            foreach (var zone in touchedZones)
            {
                var interaction = new ZoneInteraction
                {
                    Timestamp = DateTime.UtcNow,
                    Price = price,
                    ZoneType = zone.Type,
                    ZonePrice = zone.PriceLevel,
                    Outcome = outcome,
                    ZoneStrength = zone.Strength,
                    Direction = price > zone.PriceLevel ? "above" : "below"
                };

                _interactions.Add(interaction);

                _logger.LogInformation("[ZONE INTERACTION] {ZoneType} zone at {ZonePrice:F2} touched at {Price:F2}, Outcome: {Outcome}",
                    zone.Type, zone.PriceLevel, price, outcome);
            }

            // Save interactions if we have enough data
            if (_interactions.Count >= 10)
            {
                await SaveZoneInteractions();
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error recording zone interaction: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Load zones (SupplyDemandService interface)
    /// </summary>
    public async Task<ZoneData> LoadZonesAsync()
    {
        return await GetLatestZonesAsync("ES") ?? new ZoneData();
    }

    /// <summary>
    /// Save zone interactions for learning
    /// </summary>
    private async Task SaveZoneInteractions()
    {
        try
        {
            var outputDir = "Intelligence/data/zones/learning";
            Directory.CreateDirectory(outputDir);

            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            var filePath = Path.Combine(outputDir, $"zone_interactions_{timestamp}.json");

            var interactionData = new
            {
                Timestamp = DateTime.UtcNow.ToString("O"),
                InteractionCount = _interactions.Count,
                Interactions = _interactions.TakeLast(50).ToList() // Keep last 50 interactions
            };

            var json = JsonSerializer.Serialize(interactionData, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            await File.WriteAllTextAsync(filePath, json);

            _logger.LogInformation("[ZONES] Saved {Count} zone interactions to {Path}",
                _interactions.Count, filePath);

            // Keep only recent interactions in memory
            if (_interactions.Count > 100)
            {
                _interactions.RemoveRange(0, _interactions.Count - 50);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug("[ZONES] Error saving zone interactions: {Error}", ex.Message);
        }
    }
}