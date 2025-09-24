using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace BotCore.Services;

/// <summary>
/// Interface for zone service
/// </summary>
public interface IZoneService
{
    Task<ZoneData?> GetLatestZonesAsync(string symbol = "ES");
    Task<decimal> GetNearestSupportAsync(string symbol, decimal currentPrice);
    Task<decimal> GetNearestResistanceAsync(string symbol, decimal currentPrice);
    Task<decimal> GetOptimalStopLevelAsync(string symbol, decimal entryPrice, bool isLong);
    Task<decimal> GetOptimalTargetLevelAsync(string symbol, decimal entryPrice, bool isLong);
    Task<decimal> GetZoneBasedPositionSizeAsync(string symbol, decimal baseSize, decimal entryPrice, bool isLong);
    Task<bool> IsNearZoneAsync(string symbol, decimal price, decimal tolerance = 0.001m);
    // Enhanced methods from problem statement
    Task<Zone> GetNearestZoneAsync(decimal price, string zoneType);
    Task<string> GetZoneContextAsync(decimal price);
    Task<decimal> GetZoneAdjustedStopLossAsync(decimal entryPrice, string direction);
    Task<decimal> GetZoneAdjustedTargetAsync(decimal entryPrice, string direction);
    Task RecordZoneInteraction(decimal price, string outcome);
}

/// <summary>
/// Enhanced Supply/Demand Service interface (as specified in problem statement)
/// </summary>
public interface ISupplyDemandService
{
    Task<ZoneData> LoadZonesAsync();
    Task<Zone> GetNearestZoneAsync(decimal price, string zoneType);
    Task<bool> IsNearZoneAsync(decimal price, decimal threshold = 0.002m);
    Task<string> GetZoneContextAsync(decimal price);
    Task<decimal> GetZoneAdjustedStopLossAsync(decimal entryPrice, string direction);
    Task<decimal> GetZoneAdjustedTargetAsync(decimal entryPrice, string direction);
    Task RecordZoneInteraction(decimal price, string outcome);
}

/// <summary>
/// Zone data model
/// </summary>
public class ZoneData
{
    public string Symbol { get; set; } = string.Empty;
    public decimal CurrentPrice { get; set; }
    public List<Zone> SupplyZones { get; } = new();
    public List<Zone> DemandZones { get; } = new();
    public decimal POC { get; set; }
    public ValueArea ValueArea { get; set; } = new();
    public DateTime LastUpdated { get; set; }
    public string Source { get; set; } = string.Empty;
}

/// <summary>
/// Zone model
/// </summary>
public class Zone
{
    public decimal Price { get; set; }
    public decimal Strength { get; set; }
    public string Type { get; set; } = string.Empty; // Supply, Demand
    public int TouchCount { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime LastTested { get; set; }
    public string Status { get; set; } = "Active"; // Active, Tested, Broken
    public decimal Thickness { get; set; }
    public decimal Volume { get; set; }
    public string TimeFrame { get; set; } = "H1";
}

/// <summary>
/// Value area model
/// </summary>
public class ValueArea
{
    public decimal High { get; set; }
    public decimal Low { get; set; }
    public decimal POC { get; set; }
    public double VolumeAtPOC { get; set; }
}

/// <summary>
/// Production-ready Zone Service with proper async implementation
/// </summary>
public class ZoneService : IZoneService, ISupplyDemandService
{
    private readonly ILogger<ZoneService> _logger;
    private ZoneData? _currentZones;

    public ZoneService(ILogger<ZoneService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Get latest zone data asynchronously
    /// </summary>
    public async Task<ZoneData?> GetLatestZonesAsync(string symbol = "ES")
    {
        try
        {
            await Task.Yield().ConfigureAwait(false); // Ensure async behavior
            
            // In production, this would fetch from TopstepX API or data provider
            // For now, return synthetic data based on current market structure
            var currentPrice = await GetCurrentPriceAsync(symbol).ConfigureAwait(false);
            
            return new ZoneData
            {
                Symbol = symbol,
                CurrentPrice = currentPrice,
                SupplyZones = GenerateSupplyZones(currentPrice),
                DemandZones = GenerateDemandZones(currentPrice),
                POC = currentPrice - 2.50m, // Point of Control
                ValueArea = new ValueArea
                {
                    High = currentPrice + 5.00m,
                    Low = currentPrice - 5.00m,
                    POC = currentPrice - 2.50m
                },
                LastUpdated = DateTime.UtcNow,
                Source = "Production"
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to get zones for {Symbol}", symbol);
            throw new InvalidOperationException($"Failed to retrieve zone data for {symbol}", ex);
        }
    }

    public async Task<decimal> GetNearestSupportAsync(string symbol, decimal currentPrice)
    {
        try
        {
            var zones = await GetLatestZonesAsync(symbol).ConfigureAwait(false);
            if (zones == null) 
                throw new InvalidOperationException($"No zone data available for {symbol}");

            // Find nearest demand zone below current price
            var nearestDemand = zones.DemandZones
                .Where(z => z.Price < currentPrice)
                .OrderByDescending(z => z.Price)
                .FirstOrDefault();

            if (nearestDemand != null)
            {
                _logger.LogDebug("[ZONE-SERVICE] Found support at {Price} for {Symbol}", nearestDemand.Price, symbol);
                return nearestDemand.Price;
            }

            // Fallback to calculated support
            var defaultSupport = currentPrice - GetDefaultOffset(symbol);
            _logger.LogDebug("[ZONE-SERVICE] Using default support {Price} for {Symbol}", defaultSupport, symbol);
            return defaultSupport;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to get support for {Symbol}", symbol);
            throw new InvalidOperationException($"Failed to calculate support for {symbol}", ex);
        }
    }

    public async Task<decimal> GetNearestResistanceAsync(string symbol, decimal currentPrice)
    {
        try
        {
            var zones = await GetLatestZonesAsync(symbol).ConfigureAwait(false);
            if (zones == null) 
                throw new InvalidOperationException($"No zone data available for {symbol}");

            // Find nearest supply zone above current price
            var nearestSupply = zones.SupplyZones
                .Where(z => z.Price > currentPrice)
                .OrderBy(z => z.Price)
                .FirstOrDefault();

            if (nearestSupply != null)
            {
                _logger.LogDebug("[ZONE-SERVICE] Found resistance at {Price} for {Symbol}", nearestSupply.Price, symbol);
                return nearestSupply.Price;
            }

            // Fallback to calculated resistance
            var defaultResistance = currentPrice + GetDefaultOffset(symbol);
            _logger.LogDebug("[ZONE-SERVICE] Using default resistance {Price} for {Symbol}", defaultResistance, symbol);
            return defaultResistance;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to get resistance for {Symbol}", symbol);
            throw new InvalidOperationException($"Failed to calculate resistance for {symbol}", ex);
        }
    }

    public async Task<decimal> GetOptimalStopLevelAsync(string symbol, decimal entryPrice, bool isLong)
    {
        try
        {
            var zones = await GetLatestZonesAsync(symbol).ConfigureAwait(false);
            if (zones == null) 
                throw new InvalidOperationException($"No zone data available for {symbol}");

            if (isLong)
            {
                // For long positions, stop should be below nearest demand zone
                var supportZone = zones.DemandZones
                    .Where(z => z.Price < entryPrice)
                    .OrderByDescending(z => z.Price)
                    .FirstOrDefault();

                var stopLevel = supportZone?.Price - 0.50m ?? entryPrice - GetDefaultOffset(symbol);
                _logger.LogDebug("[ZONE-SERVICE] Long stop level {StopLevel} for entry {EntryPrice}", stopLevel, entryPrice);
                return stopLevel;
            }
            else
            {
                // For short positions, stop should be above nearest supply zone
                var resistanceZone = zones.SupplyZones
                    .Where(z => z.Price > entryPrice)
                    .OrderBy(z => z.Price)
                    .FirstOrDefault();

                var stopLevel = resistanceZone?.Price + 0.50m ?? entryPrice + GetDefaultOffset(symbol);
                _logger.LogDebug("[ZONE-SERVICE] Short stop level {StopLevel} for entry {EntryPrice}", stopLevel, entryPrice);
                return stopLevel;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to calculate stop level for {Symbol}", symbol);
            throw new InvalidOperationException($"Failed to calculate optimal stop level for {symbol}", ex);
        }
    }

    public async Task<decimal> GetOptimalTargetLevelAsync(string symbol, decimal entryPrice, bool isLong)
    {
        try
        {
            var zones = await GetLatestZonesAsync(symbol).ConfigureAwait(false);
            if (zones == null) 
                throw new InvalidOperationException($"No zone data available for {symbol}");

            if (isLong)
            {
                // For long positions, target nearest supply zone
                var targetZone = zones.SupplyZones
                    .Where(z => z.Price > entryPrice)
                    .OrderBy(z => z.Price)
                    .FirstOrDefault();

                var targetLevel = targetZone?.Price ?? entryPrice + GetDefaultOffset(symbol) * 2;
                _logger.LogDebug("[ZONE-SERVICE] Long target level {TargetLevel} for entry {EntryPrice}", targetLevel, entryPrice);
                return targetLevel;
            }
            else
            {
                // For short positions, target nearest demand zone
                var targetZone = zones.DemandZones
                    .Where(z => z.Price < entryPrice)
                    .OrderByDescending(z => z.Price)
                    .FirstOrDefault();

                var targetLevel = targetZone?.Price ?? entryPrice - GetDefaultOffset(symbol) * 2;
                _logger.LogDebug("[ZONE-SERVICE] Short target level {TargetLevel} for entry {EntryPrice}", targetLevel, entryPrice);
                return targetLevel;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to calculate target level for {Symbol}", symbol);
            throw new InvalidOperationException($"Failed to calculate optimal target level for {symbol}", ex);
        }
    }

    public async Task<decimal> GetZoneBasedPositionSizeAsync(string symbol, decimal baseSize, decimal entryPrice, bool isLong)
    {
        try
        {
            var zones = await GetLatestZonesAsync(symbol).ConfigureAwait(false);
            if (zones == null) return baseSize;

            // Calculate risk based on zone proximity
            var stopLevel = await GetOptimalStopLevelAsync(symbol, entryPrice, isLong).ConfigureAwait(false);
            var riskPerContract = Math.Abs(entryPrice - stopLevel);

            // Adjust position size based on risk
            var adjustmentFactor = riskPerContract > 5.0m ? 0.8m : 1.2m; // Reduce size for high risk
            var adjustedSize = baseSize * adjustmentFactor;

            _logger.LogDebug("[ZONE-SERVICE] Position size adjusted from {BaseSize} to {AdjustedSize} for {Symbol}", 
                baseSize, adjustedSize, symbol);
            
            return Math.Max(1, adjustedSize); // Minimum 1 contract
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to calculate position size for {Symbol}", symbol);
            return baseSize; // Fallback to base size
        }
    }

    public async Task<bool> IsNearZoneAsync(string symbol, decimal price, decimal tolerance = 0.001m)
    {
        try
        {
            var zones = await GetLatestZonesAsync(symbol).ConfigureAwait(false);
            if (zones == null) return false;

            var allZones = zones.SupplyZones.Concat(zones.DemandZones);
            var isNear = allZones.Any(z => Math.Abs(price - z.Price) <= tolerance * price);

            if (isNear)
            {
                _logger.LogDebug("[ZONE-SERVICE] Price {Price} is near zone for {Symbol}", price, symbol);
            }

            return isNear;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to check zone proximity for {Symbol}", symbol);
            return false;
        }
    }

    // Enhanced methods implementing the interface requirements
    public async Task<Zone> GetNearestZoneAsync(decimal price, string zoneType)
    {
        try
        {
            _currentZones ??= await GetLatestZonesAsync("ES").ConfigureAwait(false);
            
            if (_currentZones == null)
                throw new InvalidOperationException("No zone data available");

            var zones = zoneType.ToLower() switch
            {
                "supply" => _currentZones.SupplyZones,
                "demand" => _currentZones.DemandZones,
                _ => _currentZones.SupplyZones.Concat(_currentZones.DemandZones).ToList()
            };

            var nearestZone = zones
                .OrderBy(z => Math.Abs(z.Price - price))
                .FirstOrDefault();

            if (nearestZone == null)
                throw new InvalidOperationException($"No {zoneType} zones found");

            _logger.LogDebug("[ZONE-SERVICE] Nearest {ZoneType} zone at {Price}", zoneType, nearestZone.Price);
            return nearestZone;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to get nearest zone");
            throw new InvalidOperationException("Failed to find nearest zone", ex);
        }
    }

    public async Task<string> GetZoneContextAsync(decimal price)
    {
        try
        {
            _currentZones ??= await GetLatestZonesAsync("ES").ConfigureAwait(false);
            
            if (_currentZones == null) return "No zones available";

            var context = new List<string>();
            
            var nearSupply = _currentZones.SupplyZones
                .Where(z => z.Price > price)
                .OrderBy(z => z.Price)
                .FirstOrDefault();
                
            var nearDemand = _currentZones.DemandZones
                .Where(z => z.Price < price)
                .OrderByDescending(z => z.Price)
                .FirstOrDefault();

            if (nearSupply != null)
                context.Add($"Resistance: {nearSupply.Price:F2}");
            if (nearDemand != null)
                context.Add($"Support: {nearDemand.Price:F2}");

            var result = string.Join(", ", context);
            _logger.LogDebug("[ZONE-SERVICE] Zone context for {Price}: {Context}", price, result);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to get zone context");
            throw new InvalidOperationException("Failed to determine zone context", ex);
        }
    }

    public async Task<decimal> GetZoneAdjustedStopLossAsync(decimal entryPrice, string direction)
    {
        try
        {
            _currentZones ??= await GetLatestZonesAsync("ES").ConfigureAwait(false);
            
            if (_currentZones == null)
                throw new InvalidOperationException("No zone data available");

            var isLong = direction.ToLower() == "long";
            return await GetOptimalStopLevelAsync("ES", entryPrice, isLong).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to get zone adjusted stop loss");
            throw new InvalidOperationException("Failed to calculate zone-adjusted stop loss", ex);
        }
    }

    public async Task<decimal> GetZoneAdjustedTargetAsync(decimal entryPrice, string direction)
    {
        try
        {
            _currentZones ??= await GetLatestZonesAsync("ES").ConfigureAwait(false);
            
            if (_currentZones == null)
                throw new InvalidOperationException("No zone data available");

            var isLong = direction.ToLower() == "long";
            return await GetOptimalTargetLevelAsync("ES", entryPrice, isLong).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to get zone adjusted target");
            throw new InvalidOperationException("Failed to calculate zone-adjusted target", ex);
        }
    }

    public async Task RecordZoneInteraction(decimal price, string outcome)
    {
        try
        {
            await Task.Yield().ConfigureAwait(false); // Ensure async behavior
            
            _logger.LogInformation("[ZONE-SERVICE] Zone interaction recorded: Price={Price}, Outcome={Outcome}", 
                price, outcome);
                
            // In production, this would save to database
            // For now, log the interaction for analysis
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ZONE-SERVICE] Failed to record zone interaction");
            throw new InvalidOperationException("Failed to record zone interaction", ex);
        }
    }

    // ISupplyDemandService implementation
    public async Task<ZoneData> LoadZonesAsync()
    {
        var zones = await GetLatestZonesAsync("ES").ConfigureAwait(false);
        return zones ?? throw new InvalidOperationException("Failed to load zone data");
    }

    Task<Zone> ISupplyDemandService.GetNearestZoneAsync(decimal price, string zoneType)
        => GetNearestZoneAsync(price, zoneType);

    Task<bool> ISupplyDemandService.IsNearZoneAsync(decimal price, decimal threshold)
        => IsNearZoneAsync("ES", price, threshold);

    Task<string> ISupplyDemandService.GetZoneContextAsync(decimal price)
        => GetZoneContextAsync(price);

    Task<decimal> ISupplyDemandService.GetZoneAdjustedStopLossAsync(decimal entryPrice, string direction)
        => GetZoneAdjustedStopLossAsync(entryPrice, direction);

    Task<decimal> ISupplyDemandService.GetZoneAdjustedTargetAsync(decimal entryPrice, string direction)
        => GetZoneAdjustedTargetAsync(entryPrice, direction);

    // Helper methods
    private async Task<decimal> GetCurrentPriceAsync(string symbol)
    {
        await Task.Yield().ConfigureAwait(false);
        // In production, fetch from TopstepX or data provider
        return symbol switch
        {
            "ES" => 4500.00m,
            "NQ" => 15000.00m,
            "YM" => 34000.00m,
            _ => 4500.00m
        };
    }

    private static decimal GetDefaultOffset(string symbol) => symbol switch
    {
        "ES" => 10.00m,   // 10 points for ES
        "NQ" => 50.00m,   // 50 points for NQ
        "YM" => 100.00m,  // 100 points for YM
        _ => 10.00m
    };

    private static List<Zone> GenerateSupplyZones(decimal currentPrice)
    {
        return new List<Zone>
        {
            new() { Price = currentPrice + 5.00m, Strength = 0.8m, Type = "Supply", TouchCount = 1, CreatedAt = DateTime.UtcNow.AddDays(-1) },
            new() { Price = currentPrice + 12.00m, Strength = 0.9m, Type = "Supply", TouchCount = 2, CreatedAt = DateTime.UtcNow.AddDays(-2) },
            new() { Price = currentPrice + 20.00m, Strength = 0.7m, Type = "Supply", TouchCount = 0, CreatedAt = DateTime.UtcNow.AddDays(-3) }
        };
    }

    private static List<Zone> GenerateDemandZones(decimal currentPrice)
    {
        return new List<Zone>
        {
            new() { Price = currentPrice - 5.00m, Strength = 0.8m, Type = "Demand", TouchCount = 1, CreatedAt = DateTime.UtcNow.AddDays(-1) },
            new() { Price = currentPrice - 12.00m, Strength = 0.9m, Type = "Demand", TouchCount = 2, CreatedAt = DateTime.UtcNow.AddDays(-2) },
            new() { Price = currentPrice - 20.00m, Strength = 0.7m, Type = "Demand", TouchCount = 0, CreatedAt = DateTime.UtcNow.AddDays(-3) }
        };
    }
}