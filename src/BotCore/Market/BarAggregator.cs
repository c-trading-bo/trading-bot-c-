using System;
using System.Collections.Generic;

namespace BotCore.Market
{
    public sealed record Bar(DateTime Start, DateTime End, decimal Open, decimal High, decimal Low, decimal Close, long Volume);

    public sealed class BarAggregator
    {
        readonly int _tfSec; // 60, 300, 1800
        readonly Dictionary<string, Bar> _working = new();     // contractId -> current bar
        readonly Dictionary<string, List<Bar>> _history = new(); // contractId -> recent bars (FIFO)
        readonly int _maxKeep;

        public event Action<string, Bar>? OnBarClosed; // contractId, bar

        public BarAggregator(TimeSpan timeframe, int maxKeep = 1000)
        { _tfSec = (int)timeframe.TotalSeconds; _maxKeep = Math.Max(100, maxKeep); }

        static DateTime BucketStartUtc(DateTime tsUtc, int tfSec)
        {
            var s = tsUtc.Ticks / TimeSpan.TicksPerSecond;
            var bucket = (s / tfSec) * tfSec;
            return new DateTime(bucket * TimeSpan.TicksPerSecond, DateTimeKind.Utc);
        }

        public IReadOnlyList<Bar> GetHistory(string cid) => _history.TryGetValue(cid, out var lst) ? lst : Array.Empty<Bar>();

        public void Seed(string cid, IEnumerable<Bar> bars)
        {
            var list = _history.TryGetValue(cid, out var lst) ? lst : (_history[cid] = new List<Bar>(512));
            list.Clear();
            list.AddRange(bars);
            Trim(list);
            if (list.Count > 0) _working[cid] = list[^1]; // allow live to continue from last partial if you included it
        }

        public void OnTrade(string cid, DateTime tsUtc, decimal price, long qty)
        {
            var start = BucketStartUtc(tsUtc, _tfSec);
            var end   = start.AddSeconds(_tfSec);

            _working.TryGetValue(cid, out var bar);
            if (bar is null || bar.Start != start)
            {
                // close previous
                if (bar is not null)
                {
                    AddHistory(cid, bar);
                    OnBarClosed?.Invoke(cid, bar);
                }
                // open new
                _working[cid] = new Bar(start, end, price, price, price, price, qty);
                return;
            }

            // update current
            var open = bar.Open;
            var high = price > bar.High ? price : bar.High;
            var low  = price < bar.Low  ? price : bar.Low;
            var vol  = bar.Volume + qty;
            _working[cid] = bar with { High = high, Low = low, Close = price, Volume = vol };

            // optional: if ts ≥ end (late ticks), force close
            if (tsUtc >= end)
            {
                var b = _working[cid];
                AddHistory(cid, b);
                OnBarClosed?.Invoke(cid, b);
                _working.Remove(cid);
            }
        }

        void AddHistory(string cid, Bar bar)
        {
            var list = _history.TryGetValue(cid, out var lst) ? lst : (_history[cid] = new List<Bar>(512));
            list.Add(bar);
            Trim(list);
        }

        void Trim(List<Bar> list)
        {
            while (list.Count > _maxKeep) list.RemoveAt(0);
        }
    }
}
