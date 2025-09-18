
#nullable enable
using BotCore.Models;
namespace BotCore
{
    public static class EmaCrossStrategy
    {
        public static int TrySignal(IReadOnlyList<Bar> bars, int fast = 8, int slow = 21)
        {
            if (bars.Count < Math.Max(fast, slow) + 2) return 0;
            decimal emaFastPrev = 0, emaSlowPrev;
            var alphaF = 2m / (fast + 1);
            var alphaS = 2m / (slow + 1);
            decimal emaSlow;
            // initialize with the first close
            decimal emaFast = emaSlow = bars[0].Close;

            for (int i = 1; i < bars.Count; i++)
            {
                emaFastPrev = emaFast;
                emaSlowPrev = emaSlow;
                var c = bars[i].Close;
                emaFast = alphaF * c + (1 - alphaF) * emaFast;
                emaSlow = alphaS * c + (1 - alphaS) * emaSlow;
            }

            var prevCrossUp = emaFastPrev <= emaSlowPrev && emaFast > emaSlow;
            var prevCrossDown = emaFastPrev >= emaSlowPrev && emaFast < emaSlow;

            if (prevCrossUp) return +1;
            if (prevCrossDown) return -1;
            return 0;
        }
    }
}
