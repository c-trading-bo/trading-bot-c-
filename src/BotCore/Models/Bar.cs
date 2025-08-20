using System;

namespace BotCore.Models
{
    public sealed record Bar
    {
        public DateTime Start { get; init; }
        public long Ts { get; init; } // Unix ms timestamp
        public string Symbol { get; init; } = "";
        public decimal Open { get; set; }
        public decimal High { get; set; }
        public decimal Low { get; set; }
        public decimal Close { get; set; }
        public int Volume { get; set; }
    }
}
