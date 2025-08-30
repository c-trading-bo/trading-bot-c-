using System;

namespace BotCore.Market
{
    public sealed class BarPyramid
    {
        public readonly BarAggregator M1 = new(TimeSpan.FromMinutes(1));
        public readonly BarAggregator M5 = new(TimeSpan.FromMinutes(5));
        public readonly BarAggregator M30 = new(TimeSpan.FromMinutes(30));

        public BarPyramid()
        {
            M1.OnBarClosed += (cid, b1) =>
            {
                // collapse 1m into 5m / 30m by re-feeding synthetic trades at bar close
                M5.OnTrade(cid, b1.End.AddMilliseconds(-1), b1.Close, (long)Math.Max(1, b1.Volume));
                M30.OnTrade(cid, b1.End.AddMilliseconds(-1), b1.Close, (long)Math.Max(1, b1.Volume));
            };
        }
    }
}
