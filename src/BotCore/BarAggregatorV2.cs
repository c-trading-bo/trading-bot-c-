#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace BotCore
{
    public sealed class FootprintBar
    {
        public DateTimeOffset OpenTime { get; init; } // UTC minute
        public decimal O { get; set; }
        public decimal H { get; set; }
        public decimal L { get; set; }
        public decimal C { get; set; }
        public decimal V { get; set; }
        public decimal BuyVol { get; set; }
        public decimal SellVol { get; set; }
        public decimal Delta => BuyVol - SellVol;
        public override string ToString()
            => $"{OpenTime:HH:mm} O:{O} H:{H} L:{L} C:{C} V:{V}  Buy:{BuyVol} Sell:{SellVol} Δ:{Delta}";
    }


    public sealed class FootprintBarAggregator
    {
        private readonly Dictionary<string, (DateTimeOffset bucket, FootprintBar bar)> _state = new();
        public event Action<string, string, FootprintBar>? BarClosed;

        public void OnTrade(string contractId, string symbolId, DateTimeOffset tsUtc, decimal px, decimal vol, int side)
        {
            var bucket = new DateTimeOffset(tsUtc.Year, tsUtc.Month, tsUtc.Day, tsUtc.Hour, tsUtc.Minute, 0, TimeSpan.Zero);
            if (!_state.TryGetValue(symbolId, out var s))
            {
                var b = NewBar(bucket, px, vol, side);
                _state[symbolId] = (bucket, b);
                return;
            }
            if (bucket > s.bucket)
            {
                BarClosed?.Invoke(contractId, symbolId, s.bar);
                _state[symbolId] = (bucket, NewBar(bucket, px, vol, side));
                return;
            }
            var bar = s.bar;
            if (px > bar.H) bar.H = px;
            if (px < bar.L) bar.L = px;
            bar.C = px;
            bar.V += vol;
            if (side == 0) bar.BuyVol += vol; else bar.SellVol += vol;
        }

        private static FootprintBar NewBar(DateTimeOffset bucket, decimal px, decimal vol, int side)
            => new()
            {
                OpenTime = bucket, O = px, H = px, L = px, C = px, V = vol,
                BuyVol = side == 0 ? vol : 0m, SellVol = side == 1 ? vol : 0m
            };
    }
}
