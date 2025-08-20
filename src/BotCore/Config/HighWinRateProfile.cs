using System.Collections.Generic;

namespace BotCore.Config
{
    public class HighWinRateProfile
    {
        public string Profile => "high_win_rate";
        public Dictionary<string, int> AttemptCaps => new()
        {
            { "S1", 2 }, { "S2", 6 }, { "S3", 2 }, { "S4", 3 }, { "S5", 3 }, { "S6", 2 }, { "S7", 0 }, { "S8", 3 }, { "S9", 3 }, { "S10", 2 }, { "S11", 2 }, { "S12", 3 }, { "S13", 3 }
        };
        public Dictionary<string, int> Buffers => new()
        {
            { "ES_ticks", 1 }, { "NQ_ticks", 2 }
        };
        public Dictionary<string, object> GlobalFilters => new()
        {
            { "volume_pct_min_mr", 20 }, { "volume_pct_min_bo", 35 }, { "volume_pct_strong", 70 }, { "spike_volume_pct", 85 },
            { "aggI_min_bo", 0.20 }, { "aggI_max_mr_abs", 0.15 }, { "signal_bar_max_atr_mult", 2.5 }, { "spread_ticks_max", 1 },
            { "spread_ticks_max_bo", 2 }, { "three_stop_rule", true }
        };
        public Dictionary<string, object> RsGate => new()
        {
            { "z5m_threshold_mid", 0.8 }, { "z5m_threshold_high", 1.0 }, { "z5m_threshold_low_mr_only", 0.6 }, { "capital_bias", new int[] { 70, 30 } }
        };
        public Dictionary<string, object> Concurrency => new()
        {
            { "max_positions_total", 2 }, { "one_fresh_entry_per_symbol", true }
        };
        public Dictionary<string, object> Hysteresis => new()
        {
            { "bars", 3 }, { "min_dwell_minutes", 15 }
        };
        public Dictionary<string, object> Timeframes => new()
        {
            { "signal", "1m" }, { "filter", "5m" }, { "regime", "30m" }, { "daily", true }
        };
    }
}
