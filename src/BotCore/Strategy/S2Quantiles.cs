using System;
using System.Collections.Generic;

namespace BotCore.Strategy
{
    // Rolling quantiles of |z| extremes per instrument & hour bucket (lightweight, in-memory)
    internal static class S2Quantiles
    {
        // EMA-tracked threshold per (symbol:hour). Using a Robbins-Monro style update.
        private static readonly Dictionary<string, decimal> _q80 = new(StringComparer.OrdinalIgnoreCase);
        private static readonly object _sync = new();

        public static void Observe(string sym, DateTime nowLocal, decimal absZ)
        {
            var key = Key(sym, nowLocal);
            lock (_sync)
            {
                var current = _q80.TryGetValue(key, out var v) ? v : 2.0m; // sensible default
                var alpha = 0.02m; // slow adapt
                var step = absZ >= current ? alpha : -alpha * 0.5m;
                var next = Math.Max(1.5m, Math.Min(3.5m, current + step));
                _q80[key] = next;
            }
        }

        public static decimal GetSigmaFor(string sym, DateTime nowLocal, decimal fallback)
        {
            var key = Key(sym, nowLocal);
            lock (_sync)
            {
                return _q80.TryGetValue(key, out var v) ? v : fallback;
            }
        }

        private static string Key(string sym, DateTime nowLocal) => sym + ":" + nowLocal.Hour.ToString();
    }
}
