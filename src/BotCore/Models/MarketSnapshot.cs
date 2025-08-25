using System;
namespace BotCore.Models
{
	public class MarketSnapshot
	{
		public string Symbol { get; set; } = string.Empty;
		public DateTime UtcNow { get; set; } // Changed to DateTime for compatibility
		public int SpreadTicks { get; set; }
		public int VolumePct5m { get; set; }
		public decimal AggIAbs { get; set; }
		public decimal SignalBarAtrMult { get; set; }
		public int Adx5m { get; set; }
		public bool Ema9Over21_5m { get; set; }
		public decimal Z5mReturnDiff { get; set; }
		public decimal LastPrice { get; set; }
		public decimal Bid { get; set; }
		public decimal Ask { get; set; }
		public long Volume { get; set; }
		public DateTimeOffset LastTradeTime { get; set; }
		public DateTimeOffset LastQuoteTime { get; set; }
		public string SessionWindowEt { get; set; } = string.Empty;
		public decimal Bias { get; set; } // directional bias sign (-1..1), optional
		public bool IsMajorNewsNow { get; set; } // news window currently active
		public bool IsHoliday { get; set; } // holiday schedule
		public int IsMajorNewsSoonWithinSec { get; set; } // <= N seconds until news
														  // Extend with additional fields for diagnostics, risk, etc.
	}
}
