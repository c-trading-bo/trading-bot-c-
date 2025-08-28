using System;
using System.Text.Json;
using BotCore.Config;

namespace BotCore.Strategy;

// Lightweight runtime configuration holder for S2 (VWAP Mean-Reversion)
public static class S2RuntimeConfig
{
    // Defaults match the provided JSON requirements
    public static string Tf1 { get; private set; } = "M1";
    public static string Tf2 { get; private set; } = "M5";
    public static int AtrLen { get; private set; } = 14;
    public static decimal SigmaEnter { get; private set; } = 2.0m;
    public static decimal AtrEnter { get; private set; } = 1.0m;
    public static decimal SigmaForceTrend { get; private set; } = 2.8m;
    public static decimal MinSlopeTf2 { get; private set; } = 0.18m; // ticks/bar on EMA20 TF2 (proxy)
    public static decimal VolZMin { get; private set; } = -0.3m;
    public static decimal VolZMax { get; private set; } = 2.2m;
    public static int ConfirmLookback { get; private set; } = 3;
    public static int ValidityBars { get; private set; } = 3;
    public static int CooldownBars { get; private set; } = 5;
    public static int MaxBarsInTrade { get; private set; } = 45;
    public static decimal StopAtrMult { get; private set; } = 0.75m;
    public static decimal TrailAtrMult { get; private set; } = 1.0m;
    public static int IbEndMinute { get; private set; } = 10 * 60 + 30; // 630
    public static decimal EsSigma { get; private set; } = 2.0m;
    public static decimal NqSigma { get; private set; } = 2.6m;
    public static decimal OvernightScale { get; private set; } = 0.5m;
    public static int MinVolume { get; private set; } = 3000;
    public static int MaxSpreadTicks { get; private set; } = 2;
    // Patch B: session-level tunables
    public static int MaxTradesPerSession { get; private set; } = 3;
    public static string EntryMode { get; private set; } = "retest";
    public static int RetestOffsetTicks { get; private set; } = 1;
    public static decimal IbAtrGuardMult { get; private set; } = 1.5m;
    public static decimal OrAtrGuardMult { get; private set; } = 2.0m;
    public static decimal GapGuardThreshold { get; private set; } = 8.0m; // ticks
    public static decimal RsPeerThreshold { get; private set; } = 0.7m;
    public static decimal RollWeekSigmaBump { get; private set; } = 0.3m;
    public static bool PriorDayVwapVeto { get; private set; } = true;
    public static bool PriorDayCloseVeto { get; private set; } = true;
    // ADR guards
    public static int AdrLookbackDays { get; private set; } = 20;
    public static decimal AdrRoomFrac { get; private set; } = 0.25m;
    public static decimal AdrExhaustionCap { get; private set; } = 1.20m;

    public static void ApplyFrom(StrategyDef def)
    {
        if (def is null) return;
        var extra = def.Extra;
        TryString(extra, "tf1", s => Tf1 = s);
        TryString(extra, "tf2", s => Tf2 = s);
        TryInt(extra, "atr_len", v => AtrLen = v);
        TryDec(extra, "sigma_enter", v => SigmaEnter = v);
        TryDec(extra, "atr_enter", v => AtrEnter = v);
        TryDec(extra, "sigma_force_trend", v => SigmaForceTrend = v);
        TryDec(extra, "min_slope_tf2", v => MinSlopeTf2 = v);
        if (extra.TryGetValue("volz", out var volzEl) && volzEl.ValueKind == JsonValueKind.Object)
        {
            if (volzEl.TryGetProperty("min", out var mn) && mn.TryGetDecimal(out var vMin)) VolZMin = vMin;
            if (volzEl.TryGetProperty("max", out var mx) && mx.TryGetDecimal(out var vMax)) VolZMax = vMax;
        }
        TryInt(extra, "confirm_lookback", v => ConfirmLookback = v);
        TryInt(extra, "validity_bars", v => ValidityBars = v);
        TryInt(extra, "cooldown_bars", v => CooldownBars = v);
        TryInt(extra, "max_bars_in_trade", v => MaxBarsInTrade = v);
        TryDec(extra, "stop_atr_mult", v => StopAtrMult = v);
        TryDec(extra, "trail_atr_mult", v => TrailAtrMult = v);
        TryInt(extra, "ib_end_minute", v => IbEndMinute = v);
        TryDec(extra, "es_sigma", v => EsSigma = v);
        TryDec(extra, "nq_sigma", v => NqSigma = v);
        TryDec(extra, "overnight_scale", v => OvernightScale = v);
        TryInt(extra, "min_volume", v => MinVolume = v);
        TryInt(extra, "max_spread_ticks", v => MaxSpreadTicks = v);
        // Patch B: session-level tunables
        TryInt(extra, "max_trades_per_session", v => MaxTradesPerSession = v);
        TryString(extra, "entry_mode", v => EntryMode = v);
        TryInt(extra, "retest_offset_ticks", v => RetestOffsetTicks = v);
        TryDec(extra, "ib_atr_guard_mult", v => IbAtrGuardMult = v);
        TryDec(extra, "or_atr_guard_mult", v => OrAtrGuardMult = v);
        TryDec(extra, "gap_guard_threshold", v => GapGuardThreshold = v);
        TryDec(extra, "rs_peer_threshold", v => RsPeerThreshold = v);
        TryDec(extra, "roll_week_sigma_bump", v => RollWeekSigmaBump = v);
        TryBool(extra, "prior_day_vwap_veto", v => PriorDayVwapVeto = v);
        TryBool(extra, "prior_day_close_veto", v => PriorDayCloseVeto = v);
        // ADR guards
        TryInt(extra, "adr_lookback_days", v => AdrLookbackDays = v);
        TryDec(extra, "adr_room_frac", v => AdrRoomFrac = v);
        TryDec(extra, "adr_exhaustion_cap", v => AdrExhaustionCap = v);
    }

    private static void TryInt(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<int> set)
    { if (extra.TryGetValue(key, out var el) && el.TryGetInt32(out var v)) set(v); }
    private static void TryDec(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<decimal> set)
    { if (extra.TryGetValue(key, out var el) && el.TryGetDecimal(out var v)) set(v); }
    private static void TryString(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<string> set)
    {
        if (extra.TryGetValue(key, out var el) && el.ValueKind == JsonValueKind.String)
        {
            var s = el.GetString();
            if (!string.IsNullOrWhiteSpace(s)) set(s!);
        }
    }
    private static void TryBool(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<bool> set)
    {
        if (extra.TryGetValue(key, out var el))
        {
            if (el.ValueKind == JsonValueKind.True) set(true);
            else if (el.ValueKind == JsonValueKind.False) set(false);
        }
    }
}
