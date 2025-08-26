namespace BotCore.Risk
{
    public sealed class RiskConfig
    {
        public decimal risk_per_trade { get; set; } = 100m; // existing
        // Optional equity % and halts
        public decimal risk_pct_of_equity { get; set; } = 0.0m; // e.g., 0.0025m = 0.25%
        public decimal max_daily_drawdown { get; set; } = 1000m;
        public decimal max_weekly_drawdown { get; set; } = 3000m;
        public int     max_consecutive_losses { get; set; } = 3;
        public int     cooldown_minutes_after_streak { get; set; } = 30;
        public int     max_open_positions { get; set; } = 1;
    }
}
