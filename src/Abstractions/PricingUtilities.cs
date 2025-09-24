using System;
using System.Globalization;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Price and risk calculation utilities for trading operations
    /// Critical for ES/MES tick rounding (0.25) and R-multiple validation
    /// </summary>
    public static class Px
    {
        /// <summary>
        /// Standard ES/MES tick size
        /// </summary>
        private const decimal StandardTickSize = 0.25m;

        /// <summary>
        /// Round price to ES/MES tick size (0.25)
        /// Critical production requirement for ES/MES contracts
        /// </summary>
        public static decimal RoundToTick(decimal price, decimal tickSize = StandardTickSize)
        {
            if (tickSize <= 0) tickSize = StandardTickSize;
            return Math.Round(price / tickSize, 0, MidpointRounding.AwayFromZero) * tickSize;
        }

        /// <summary>
        /// Calculate R-multiple for risk validation
        /// Must return > 0 for production trading (per safety requirements)
        /// </summary>
        public static decimal RMultiple(decimal entryPrice, decimal stopPrice, decimal targetPrice, bool isLong)
        {
            if (entryPrice <= 0 || stopPrice <= 0 || targetPrice <= 0)
                return 0;

            decimal risk = isLong ? (entryPrice - stopPrice) : (stopPrice - entryPrice);
            decimal reward = isLong ? (targetPrice - entryPrice) : (entryPrice - targetPrice);

            if (risk <= 0) return 0;
            return reward / risk;
        }

        /// <summary>
        /// Format decimal to 2 decimal places for price display
        /// </summary>
        public static string F2(decimal value)
        {
            return value.ToString("F2", CultureInfo.InvariantCulture);
        }
    }
}