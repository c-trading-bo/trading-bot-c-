using System;
using System.Collections.Generic;
using System.Linq;
using BotCore.Models;

namespace BotCore.Strategy
{
    // Minimal helpers to enhance S2 without overlapping existing upgrades
    internal static class S2Upg
    {
        // Volume imbalance of up vs down bars over lookback window
        public static decimal UpDownImbalance(IList<Bar> bars, int look = 10)
        {
            if (bars is null || bars.Count < 2) return 1m;
            int start = Math.Max(0, bars.Count - look);
            decimal up = 0, dn = 0;
            for (int i = start; i < bars.Count; i++)
            {
                var b = bars[i];
                var vol = Math.Max(0, b.Volume);
                if (b.Close > b.Open) up += vol;
                else if (b.Close < b.Open) dn += vol;
            }
            return dn <= 0 ? 1.5m : up / dn; // >1 buyers dominant; <1 sellers dominant
        }

        // Adaptive sigma threshold: lift threshold on strong 5-bar slope / high volz / NQ; small relax late morning
        public static decimal DynamicSigmaThreshold(decimal baseSigma, decimal volz, decimal slope5, DateTime nowLocal, string sym)
        {
            decimal adj = 0m;
            var absSlope = Math.Abs(slope5);
            if (absSlope > 0.25m) adj += 0.3m;       // strong trend on 1m EMA20 slope proxy
            if (volz > 1.5m)        adj += 0.2m;       // high-vol regime
            if (sym.Contains("NQ", StringComparison.OrdinalIgnoreCase)) adj += 0.2m; // NQ spikier
            var mins = nowLocal.Hour * 60 + nowLocal.Minute;
            if (mins >= 680 && mins <= 720) adj -= 0.1m; // ~11:20–12:00 slight relax
            return Math.Max(baseSigma, baseSigma + adj);
        }

        // Require at least 0.5*ATR distance away from nearest opposite pivot (using 1m as proxy)
        public static bool PivotDistanceOK(IList<Bar> bars, decimal price, decimal atr, decimal tickSize, bool longSide)
        {
            var (hi, lo) = LastHtfPivots(bars, 48);
            if (hi == 0m && lo == 0m) return true;
            var dist = longSide ? price - lo : hi - price;
            return dist >= 0.5m * Math.Max(tickSize, atr);
        }

        // Simple swing pivot scan over recent window (1m proxy)
        private static (decimal hi, decimal lo) LastHtfPivots(IList<Bar> b, int lookback)
        {
            if (b == null || b.Count < 5) return (0m, 0m);
            int start = Math.Max(2, b.Count - lookback);
            decimal lastHi = 0m, lastLo = 0m;
            for (int i = start; i < b.Count - 2; i++)
            {
                bool swingHi = b[i].High > b[i-1].High && b[i].High > b[i-2].High && b[i].High > b[i+1].High && b[i].High > b[i+2].High;
                bool swingLo = b[i].Low  < b[i-1].Low  && b[i].Low  < b[i-2].Low  && b[i].Low  < b[i+1].Low  && b[i].Low  < b[i+2].Low;
                if (swingHi) lastHi = b[i].High;
                if (swingLo) lastLo = b[i].Low;
            }
            return (lastHi, lastLo);
        }

        // Optional size scale from |z|, not wired to sizing here (risk engine controls size). Provided for future use.
        public static decimal SizeScaleFromStretch(decimal absZ)
        {
            decimal s = 0.75m + 0.25m * absZ;
            if (s < 0.5m) s = 0.5m; if (s > 1.5m) s = 1.5m;
            return s;
        }
    }
}
