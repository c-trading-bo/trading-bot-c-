#nullable enable
using System;
using System.Collections.Concurrent;

namespace BotCore
{
    /// <summary>Simple in-memory duplicate suppressor for identical ticks within a short window.</summary>
    public static class TradeDeduper
    {
        private static readonly ConcurrentDictionary<string, DateTimeOffset> _seen = new();
        private static readonly TimeSpan Window = TimeSpan.FromSeconds(5);

        public static bool Seen(string symbolId, decimal price, decimal volume, DateTimeOffset ts)
        {
            var key = $"{symbolId}|{price}|{volume}|{ts:O}";
            var now = DateTimeOffset.UtcNow;

            if (_seen.TryGetValue(key, out var when) && (now - when) < Window)
                return true;

            _seen[key] = now;

            if (_seen.Count > 5000)
            {
                foreach (var kv in _seen)
                    if (now - kv.Value > Window) _seen.TryRemove(kv.Key, out _);
            }
            return false;
        }
    }
}
