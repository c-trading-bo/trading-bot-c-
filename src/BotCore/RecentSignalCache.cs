#nullable enable
using System;
using System.Collections.Concurrent;

namespace BotCore
{
    /// <summary>Prevents re-emitting the same (Sp,Tp,Sl) for a given (strategy,symbol,side) within a window.</summary>
    public static class RecentSignalCache
    {
        private sealed record Key(string Strat, string Symbol, string Side);
        private sealed record Val(DateTime Utc, decimal Sp, decimal Tp, decimal Sl);

        private static readonly ConcurrentDictionary<Key, Val> _last = new();

        public static bool ShouldEmit(
            string strategy, string symbol, string side,
            decimal sp, decimal tp, decimal sl, int cooldownSeconds)
        {
            var key = new Key(strategy, symbol, side);
            var now = DateTime.UtcNow;

            if (_last.TryGetValue(key, out var prev))
            {
                var within = (now - prev.Utc).TotalSeconds < cooldownSeconds;
                var identical = prev.Sp == sp && prev.Tp == tp && prev.Sl == sl;
                if (within && identical) return false;
            }

            _last[key] = new Val(now, sp, tp, sl);
            return true;
        }
    }
}
