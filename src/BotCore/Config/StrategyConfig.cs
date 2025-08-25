using System.Text.Json;
using System.Text.Json.Serialization;

namespace BotCore.Config;

public sealed class TradingProfileConfig
{
    [JsonPropertyName("profile")] public string Profile { get; set; } = "";
    [JsonPropertyName("schema_version")] public int SchemaVersion { get; set; } = 1;
    [JsonPropertyName("timeframes")] public Timeframes Timeframes { get; set; } = new();
    [JsonPropertyName("rs_gate")] public RsGate RsGate { get; set; } = new();
    [JsonPropertyName("global_filters")] public GlobalFilters GlobalFilters { get; set; } = new();
    [JsonPropertyName("concurrency")] public Concurrency Concurrency { get; set; } = new();
    [JsonPropertyName("hysteresis")] public Hysteresis Hysteresis { get; set; } = new();
    [JsonPropertyName("attempt_caps")] public Dictionary<string,int> AttemptCaps { get; set; } = new();
    [JsonPropertyName("buffers")] public Buffers Buffers { get; set; } = new();
    [JsonPropertyName("strategies")] public List<StrategyDef> Strategies { get; set; } = new();

    // NEW: Always-on bias and News tuning
    [JsonPropertyName("AlwaysOn")] public AlwaysOnConfig AlwaysOn { get; set; } = new();
    [JsonPropertyName("News")] public NewsTuning News { get; set; } = new();
}

public sealed class Timeframes
{
    [JsonPropertyName("signal")] public string Signal { get; set; } = "1m";
    [JsonPropertyName("filter")] public string Filter { get; set; } = "5m";
    [JsonPropertyName("regime")] public string Regime { get; set; } = "30m";
    [JsonPropertyName("daily")] public bool Daily { get; set; }
}

public sealed class RsGate
{
    [JsonPropertyName("z5m_threshold_mid")] public decimal ThresholdMid { get; set; }
    [JsonPropertyName("z5m_threshold_high")] public decimal ThresholdHigh { get; set; }
    [JsonPropertyName("z5m_threshold_low_mr_only")] public decimal ThresholdLowMrOnly { get; set; }
    [JsonPropertyName("capital_bias")] public List<int> CapitalBias { get; set; } = new();
    [JsonPropertyName("align_with_bias")] public bool AlignWithBias { get; set; }
}

public sealed class GlobalFilters
{
    [JsonPropertyName("volume_pct_min_mr")] public int VolumePctMinMr { get; set; }
    [JsonPropertyName("volume_pct_min_bo")] public int VolumePctMinBo { get; set; }
    [JsonPropertyName("volume_pct_strong")] public int VolumePctStrong { get; set; }
    [JsonPropertyName("spike_volume_pct")] public int SpikeVolumePct { get; set; }
    [JsonPropertyName("aggI_min_bo")] public decimal AggIMinBo { get; set; }
    [JsonPropertyName("aggI_max_mr_abs")] public decimal AggIMaxMrAbs { get; set; }
    [JsonPropertyName("signal_bar_max_atr_mult")] public decimal SignalBarMaxAtrMult { get; set; }
    [JsonPropertyName("spread_ticks_max")] public int SpreadTicksMax { get; set; }
    [JsonPropertyName("spread_ticks_max_bo")] public int SpreadTicksMaxBo { get; set; }
    [JsonPropertyName("three_stop_rule")] public bool ThreeStopRule { get; set; }
    [JsonPropertyName("min_seconds_between_entries")] public int MinSecondsBetweenEntries { get; set; }
}

public sealed class Concurrency
{
    [JsonPropertyName("max_positions_total")] public int MaxPositionsTotal { get; set; }
    [JsonPropertyName("one_fresh_entry_per_symbol")] public bool OneFreshEntryPerSymbol { get; set; }
}

public sealed class Hysteresis
{
    [JsonPropertyName("bars")] public int Bars { get; set; }
    [JsonPropertyName("min_dwell_minutes")] public int MinDwellMinutes { get; set; }
}

public sealed class Buffers
{
    [JsonPropertyName("ES_ticks")] public int ES_Ticks { get; set; }
    [JsonPropertyName("NQ_ticks")] public int NQ_Ticks { get; set; }
}

public sealed class AlwaysOnConfig
{
    // Never block a trade because of filters; use scoring/sizing instead
    [JsonPropertyName("Enabled")] public bool Enabled { get; set; } = true;
    // Keep position size within [SizeFloor, SizeCap] of baseline sizing
    [JsonPropertyName("SizeFloor")] public decimal SizeFloor { get; set; } = 1.00m;
    [JsonPropertyName("SizeCap")]   public decimal SizeCap   { get; set; } = 1.25m;
    // If true, indicators seed immediately (no warmup bars)
    [JsonPropertyName("ZeroWarmupIndicators")] public bool ZeroWarmupIndicators { get; set; } = true;
}

public sealed class NewsTuning
{
    // Don’t block on news/holidays; optionally boost breakouts
    [JsonPropertyName("BoostOnMajorNews")] public bool BoostOnMajorNews { get; set; } = true;
    [JsonPropertyName("BreakoutScoreBoost")] public decimal BreakoutScoreBoost { get; set; } = 1.5m;
    [JsonPropertyName("MeanRevScoreBias")]   public decimal MeanRevScoreBias   { get; set; } = 0.9m;
    [JsonPropertyName("SizeBoostOnNews")]    public decimal SizeBoostOnNews    { get; set; } = 1.10m;
}

public sealed class StrategyDef
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("enabled")] public bool Enabled { get; set; }
    [JsonPropertyName("family")] public string? Family { get; set; } // e.g. "breakout" | "meanrev"
    [JsonPropertyName("version")] public string? Version { get; set; }
    [JsonPropertyName("session")] public string? Session { get; set; }
    [JsonPropertyName("session_window_et")] public string? SessionWindowEt { get; set; }
    [JsonPropertyName("on_range_window_et")] public string? OnRangeWindowEt { get; set; }
    [JsonPropertyName("flat_by_et")] public string? FlatByEt { get; set; }
    [JsonPropertyName("regimes")] public string[]? Regimes { get; set; }
    [JsonPropertyName("levels")] public string[]? Levels { get; set; }
    [JsonExtensionData] public Dictionary<string, JsonElement> Extra { get; set; } = new();
}
