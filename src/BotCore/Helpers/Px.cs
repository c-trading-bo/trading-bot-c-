using System;
using System.Globalization;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Price and risk calculation helpers for ES/MES trading
/// Implements tick rounding, R-multiple calculations, and formatting utilities
/// </summary>
public static class Px
{
    public const decimal ES_TICK = 0.25m;
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    /// <summary>
    /// Round price to tick size (default ES/MES 0.25)
    /// </summary>
    public static decimal RoundToTick(decimal price, decimal tick = ES_TICK) =>
        Math.Round(price / tick, 0, MidpointRounding.AwayFromZero) * tick;

    /// <summary>
    /// Format decimal to 2 decimal places using invariant culture
    /// </summary>
    public static string F2(decimal value) => value.ToString("0.00", Invariant);

    /// <summary>
    /// Calculate R-multiple (reward/risk ratio)
    /// Returns 0 if risk <= 0 (invalid risk scenarios)
    /// </summary>
    public static decimal RMultiple(decimal entry, decimal stop, decimal target, bool isLong)
    {
        var risk = isLong ? entry - stop : stop - entry;     // must be > 0
        var reward = isLong ? target - entry : entry - target; // must be >= 0
        
        if (risk <= 0) return 0m; // Invalid risk scenarios
        if (reward <= 0) return 0m; // Invalid reward scenarios
        
        return reward / risk;
    }
}