using System;
using System.Collections.Generic;
using System.Linq;

namespace RiskAgent
{
    public static class Atr
    {
        // Wilder's ATR
        public static decimal Compute(IList<decimal> highs, IList<decimal> lows, IList<decimal> closes, int len = 14)
        {
            if (highs == null || lows == null || closes == null) return 0m;
            int n = Math.Min(highs.Count, Math.Min(lows.Count, closes.Count));
            if (n < Math.Max(2, len + 1)) return 0m;
            var tr = new List<decimal>(n);
            tr.Add(highs[0] - lows[0]);
            for (int i = 1; i < n; i++)
            {
                var h = highs[i]; var l = lows[i]; var pc = closes[i - 1];
                var t = Math.Max(h - l, Math.Max(Math.Abs(h - pc), Math.Abs(l - pc)));
                tr.Add(t);
            }
            // Wilder smoothing
            decimal atr = tr.Take(len).Average();
            for (int i = len; i < tr.Count; i++) atr = (atr * (len - 1) + tr[i]) / len;
            return atr;
        }
    }

    public sealed class RiskParams
    {
        public decimal MaxDailyLoss { get; init; } = 1000m;
        public int MaxContractsPerSymbol { get; init; } = 1;
        public int MaxTotalContracts { get; init; } = 2;
        public decimal RiskPerTradeUsd { get; init; } = 50m;
    }

    public static class RiskSizer
    {
        // Computes integer size based on dollar risk per trade and stop distance (points) and tick value.
        public static int CalcSize(decimal stopDistancePoints, decimal tickSize, decimal tickValueUsd, RiskParams p)
        {
            if (stopDistancePoints <= 0 || tickSize <= 0 || tickValueUsd <= 0) return 0;
            var ticks = stopDistancePoints / tickSize;
            if (ticks <= 0) return 0;
            var riskPerContract = ticks * tickValueUsd;
            if (riskPerContract <= 0) return 0;
            var size = (int)Math.Floor(p.RiskPerTradeUsd / riskPerContract);
            return Math.Max(0, Math.Min(size, p.MaxContractsPerSymbol));
        }
    }
}
