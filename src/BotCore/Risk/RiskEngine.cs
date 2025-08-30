namespace BotCore.Risk
{
    public sealed class RiskEngine
    {
        public RiskConfig cfg { get; set; } = new RiskConfig();

        public static decimal ComputeRisk(decimal entry, decimal stop, decimal target, bool isLong)
        {
            var risk = isLong ? entry - stop : stop - entry;
            var reward = isLong ? target - entry : entry - target;
            if (risk <= 0 || reward < 0) return 0m;
            return reward / risk;
        }

        public static decimal size_for(decimal riskPerTrade, decimal dist, decimal pointValue)
        {
            if (dist <= 0 || pointValue <= 0) return 0m;
            return Math.Floor(riskPerTrade / (dist * pointValue));
        }

        // NEW: Equity-% aware sizing helper (backwards-compatible)
        public (int Qty, decimal UsedRpt) ComputeSize(string symbol, decimal entry, decimal stop, decimal accountEquity)
        {
            var dist = Math.Abs(entry - stop);
            if (dist <= 0) return (0, 0);
            var pv = BotCore.Models.InstrumentMeta.PointValue(symbol);
            if (pv <= 0) return (0, 0);

            // If equity% configured and equity provided, use it, else fall back to fixed RPT
            var usePct = cfg.risk_pct_of_equity > 0m && accountEquity > 0m;
            var rpt = usePct ? Math.Round(accountEquity * cfg.risk_pct_of_equity, 2) : cfg.risk_per_trade;
            var raw = (int)System.Math.Floor((double)(rpt / (dist * pv)));
            var lot = BotCore.Models.InstrumentMeta.LotStep(symbol);
            var qty = System.Math.Max(0, raw - (raw % System.Math.Max(1, lot)));
            return (qty, rpt);
        }

        public bool ShouldHaltDay(decimal realizedPnlToday) => cfg.max_daily_drawdown > 0 && -realizedPnlToday >= cfg.max_daily_drawdown;
        public bool ShouldHaltWeek(decimal realizedPnlWeek) => cfg.max_weekly_drawdown > 0 && -realizedPnlWeek >= cfg.max_weekly_drawdown;
    }
}
