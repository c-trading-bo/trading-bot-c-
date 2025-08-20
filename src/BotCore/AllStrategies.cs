using System;
using System.Globalization;
using System.Collections.Generic;

namespace BotCore
{
	public static class Px
	{
		public const decimal ES_TICK = 0.25m;
		private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

		public static decimal RoundToTick(decimal price, decimal tick = ES_TICK) =>
			Math.Round(price / tick, 0, MidpointRounding.AwayFromZero) * tick;

		public static string F2(decimal value) => value.ToString("0.00", Invariant);

		public static decimal RMultiple(decimal entry, decimal stop, decimal target, bool isLong)
		{
			var risk   = isLong ? entry - stop : stop - entry;     // must be > 0
			var reward = isLong ? target - entry : entry - target; // must be >= 0
			if (risk <= 0 || reward < 0) return 0m;
			return reward / risk;
		}
	}
}
