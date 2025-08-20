using System;
using BotCore.Models;

namespace BotCore
{
    public sealed class BarAggregator
    {
        public readonly TimeSpan Interval;
        public event Action<Bar>? OnBar;
        private Bar? _cur;

        public BarAggregator(TimeSpan interval) => Interval = interval;

        static DateTime FloorUtc(DateTime t, TimeSpan step) =>
            new DateTime((t.Ticks / step.Ticks) * step.Ticks, DateTimeKind.Utc);

        public void OnTrade(DateTime tsUtc, decimal price, int volume)
        {
            var bucket = FloorUtc(tsUtc, Interval);
            if (_cur == null || bucket > _cur.Start)
            {
                if (_cur != null) OnBar?.Invoke(_cur);
                _cur = new Bar { Start = bucket, Open = price, High = price, Low = price, Close = price, Volume = volume };
                return;
            }

            if (price > _cur.High) _cur.High = price;
            if (price < _cur.Low)  _cur.Low  = price;
            _cur.Close  = price;
            _cur.Volume += volume;
        }
    }
}
