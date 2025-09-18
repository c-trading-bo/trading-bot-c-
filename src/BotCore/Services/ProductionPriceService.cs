using System;
using Microsoft.Extensions.Logging;
using System.Globalization;

namespace BotCore.Services;

/// <summary>
/// Production-ready price rounding and risk calculation service
/// Following guardrails: "ES/MES tick size: Round any ES/MES price to 0.25. Print two decimals."
/// "Risk math: Compute R multiple from tick-rounded values. If risk ≤ 0 → reject."
/// </summary>
public static class ProductionPriceService
{
    // Tick size constants following guardrails
    public const decimal ES_TICK = 0.25m;
    public const decimal MES_TICK = 0.25m;
    
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    /// <summary>
    /// Round price to tick size (ES/MES = 0.25)
    /// </summary>
    public static decimal RoundToTick(decimal price, decimal tick = ES_TICK)
    {
        return Math.Round(price / tick, 0, MidpointRounding.AwayFromZero) * tick;
    }

    /// <summary>
    /// Format price with two decimals as required by guardrails
    /// </summary>
    public static string F2(decimal value)
    {
        return value.ToString("0.00", Invariant);
    }

    /// <summary>
    /// Calculate R multiple from tick-rounded values following guardrails
    /// Returns null if risk ≤ 0 (rejection case)
    /// </summary>
    public static decimal? RMultiple(decimal entry, decimal stop, decimal target, bool isLong, ILogger? logger = null)
    {
        // Round all prices to tick first
        entry = RoundToTick(entry);
        stop = RoundToTick(stop);
        target = RoundToTick(target);

        // Calculate risk and reward
        var risk = isLong ? entry - stop : stop - entry;
        var reward = isLong ? target - entry : entry - target;

        logger?.LogDebug("💰 [PRICE-SERVICE] Risk calculation: entry={Entry}, stop={Stop}, target={Target}, isLong={IsLong}", 
            F2(entry), F2(stop), F2(target), isLong);
        logger?.LogDebug("💰 [PRICE-SERVICE] Risk={Risk}, Reward={Reward}", F2(risk), F2(reward));

        // Guardrail: If risk ≤ 0 → reject
        if (risk <= 0)
        {
            logger?.LogCritical("🔴 [PRICE-SERVICE] GUARDRAIL VIOLATION: Risk ≤ 0 ({Risk}) - REJECTING", F2(risk));
            return null;
        }

        if (reward < 0)
        {
            logger?.LogWarning("⚠️ [PRICE-SERVICE] Negative reward ({Reward}) - unusual but allowed", F2(reward));
        }

        var rMultiple = reward / risk;
        logger?.LogInformation("📊 [PRICE-SERVICE] R Multiple calculated: {RMultiple:0.00} (Risk: {Risk}, Reward: {Reward})", 
            rMultiple, F2(risk), F2(reward));

        return rMultiple;
    }

    /// <summary>
    /// Validate if symbol requires ES/MES tick rounding
    /// </summary>
    public static bool RequiresEsTickRounding(string symbol)
    {
        if (string.IsNullOrWhiteSpace(symbol)) return false;
        
        var upperSymbol = symbol.ToUpperInvariant();
        return upperSymbol.Contains("ES") || upperSymbol.Contains("MES") || 
               upperSymbol.Contains("E-MINI") || upperSymbol.Contains("EMINI");
    }

    /// <summary>
    /// Get appropriate tick size for symbol
    /// </summary>
    public static decimal GetTickSize(string symbol)
    {
        if (RequiresEsTickRounding(symbol))
        {
            return ES_TICK;
        }
        
        // Add other instruments here as needed
        return 0.01m; // Default 1 cent tick
    }

    /// <summary>
    /// Comprehensive price validation and rounding for a trade setup
    /// </summary>
    public static TradeSetupResult ValidateAndRoundTradeSetup(
        string symbol, 
        decimal entry, 
        decimal stop, 
        decimal target, 
        bool isLong,
        ILogger? logger = null)
    {
        var result = new TradeSetupResult
        {
            Symbol = symbol,
            IsLong = isLong,
            OriginalEntry = entry,
            OriginalStop = stop,
            OriginalTarget = target
        };

        try
        {
            var tickSize = GetTickSize(symbol);
            
            // Round all prices to appropriate tick
            result.RoundedEntry = RoundToTick(entry, tickSize);
            result.RoundedStop = RoundToTick(stop, tickSize);
            result.RoundedTarget = RoundToTick(target, tickSize);

            // Calculate R multiple
            result.RMultiple = RMultiple(result.RoundedEntry, result.RoundedStop, result.RoundedTarget, isLong, logger);
            
            // Check if valid according to guardrails
            result.IsValid = result.RMultiple.HasValue;
            
            if (result.IsValid)
            {
                logger?.LogInformation("✅ [PRICE-SERVICE] Valid trade setup: {Symbol} entry={Entry} stop={Stop} target={Target} R={R:0.00}", 
                    symbol, F2(result.RoundedEntry), F2(result.RoundedStop), F2(result.RoundedTarget), result.RMultiple.Value);
            }
            else
            {
                logger?.LogCritical("🔴 [PRICE-SERVICE] INVALID trade setup: {Symbol} - Risk ≤ 0", symbol);
                result.ValidationError = "Risk is zero or negative";
            }

            return result;
        }
        catch (Exception ex)
        {
            logger?.LogError(ex, "❌ [PRICE-SERVICE] Error validating trade setup for {Symbol}", symbol);
            result.IsValid;
            result.ValidationError = ex.Message;
            return result;
        }
    }
}

/// <summary>
/// Result of trade setup validation and rounding
/// </summary>
public class TradeSetupResult
{
    public string Symbol { get; set; } = string.Empty;
    public bool IsLong { get; set; }
    
    // Original values
    public decimal OriginalEntry { get; set; }
    public decimal OriginalStop { get; set; }
    public decimal OriginalTarget { get; set; }
    
    // Tick-rounded values
    public decimal RoundedEntry { get; set; }
    public decimal RoundedStop { get; set; }
    public decimal RoundedTarget { get; set; }
    
    // Calculated metrics
    public decimal? RMultiple { get; set; }
    
    // Validation results
    public bool IsValid { get; set; }
    public string? ValidationError { get; set; }
}