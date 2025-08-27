using System.Collections.Generic;

namespace BotCore.Config
{
    public class HighWinRateProfile
    {
        public string Profile => "high_win_rate";
        public Dictionary<string, int> AttemptCaps => new()
        {
            { "S1", 0 }, { "S2", 2 }, { "S3", 2 }, { "S4", 0 }, { "S5", 0 }, { "S6", 2 }, { "S7", 0 }, { "S8", 0 }, { "S9", 0 }, { "S10", 0 }, { "S11", 2 }, { "S12", 0 }, { "S13", 0 }
        };
        public Dictionary<string, int> Buffers => new() { { "ES_ticks", 1 }, { "NQ_ticks", 2 } };
        public Dictionary<string, object> GlobalFilters => new()
        {
            { "spread_ticks_max", 8 }, { "spread_ticks_max_bo", 10 }, { "volume_pct_min_mr", 10 }, { "volume_pct_min_bo", 10 },
            { "volume_pct_strong", 50 }, { "aggI_min_bo", 0.0m }, { "aggI_max_mr_abs", 999m }, { "signal_bar_max_atr_mult", 4.0m }, { "three_stop_rule", true }
        };
        public Dictionary<string, object> Concurrency => new() { { "max_positions_total", 2 }, { "one_fresh_entry_per_symbol", true } };
        public Dictionary<string, object> Hysteresis => new() { { "bars", 3 }, { "min_dwell_minutes", 15 } };
        public Dictionary<string, object> Timeframes => new() { { "signal", "1m" }, { "filter", "5m" }, { "regime", "30m" }, { "daily", true } };
    }
}
