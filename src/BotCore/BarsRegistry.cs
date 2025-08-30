
#nullable enable
using System.Collections.Concurrent;
using BotCore.Models;

namespace BotCore
{
    public sealed class BarsRegistry(int maxKeep = 5000)
    {
        private readonly ConcurrentDictionary<string, List<Bar>> _bars
            = new(StringComparer.OrdinalIgnoreCase);

        private readonly int _maxKeep = Math.Max(100, maxKeep);

        public void Append(string symbol, Bar bar)
        {
            var list = _bars.GetOrAdd(symbol, _ => new List<Bar>(256));
            lock (list)
            {
                list.Add(bar);
                if (list.Count > _maxKeep)
                    list.RemoveRange(0, list.Count - _maxKeep);
            }
        }

        public IReadOnlyList<Bar> Get(string symbol)
        {
            if (!_bars.TryGetValue(symbol, out var list)) return [];
            lock (list) { return [.. list]; }
        }

        public int Count(string symbol)
        {
            if (!_bars.TryGetValue(symbol, out var list)) return 0;
            lock (list) { return list.Count; }
        }
    }
}
