using System;
using System.Collections.Generic;

namespace BotCore
{
	/// <summary>
	/// Defines risk models and helper functions for position/risk tracking.
	/// </summary>
	public static class ModelsAndRisk
	{
		public static decimal CalculateRisk(decimal entry, decimal stop, bool isLong)
		{
			return isLong ? entry - stop : stop - entry;
		}

		public static decimal CalculateReward(decimal entry, decimal target, bool isLong)
		{
			return isLong ? target - entry : entry - target;
		}

		public static decimal RMultiple(decimal entry, decimal stop, decimal target, bool isLong)
		{
			var risk = CalculateRisk(entry, stop, isLong);
			var reward = CalculateReward(entry, target, isLong);
			if (risk <= 0 || reward < 0) return 0m;
			return reward / risk;
		}
	}
}
