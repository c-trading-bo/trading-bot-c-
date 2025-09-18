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
    public static decimal AdrRoomFrac { get; private set; } = 0.25m;    // require ≥25% ADR room to target
    public static decimal AdrExhaustionCap { get; private set; } = 1.20m; // block if today range > 1.2×ADR

    // Patch C extras (optional; default-off semantics where reasonable)
    public static bool AdrGuardEnabled { get; private set; };
    public static int AdrGuardLen { get; private set; } = 14;
    public static decimal AdrGuardMaxUsed { get; private set; } = 0.0m; // fraction of ADR used today to block, 0 = ignore
    public static decimal AdrGuardWarnUsed { get; private set; } = 0.0m; // not used in logic; reserved for logging/UI

    public static bool PdExtremeGuardEnabled { get; private set; };
    public static decimal PdExtremeMinRoomAtr { get; private set; } = 0.40m;

    public static bool VwapSlopeGuardEnabled { get; private set; };
    public static decimal VwapMaxSigmaPerMin { get; private set; } = 0.12m;

    public static bool ZDecelerateEnabled { get; private set; };
    public static int ZDecelNeed { get; private set; } = 2;

    // Patch C: optional day PnL kill-switch (in R units) and curfew helper
    public static bool DayPnlKillEnabled { get; private set; };
    public static decimal DayPnlStopGainR { get; private set; }; // e.g., 8.0
    public static decimal DayPnlStopLossR { get; private set; }; // e.g., -6.0

    public static bool CurfewEnabled { get; private set; };
    public static string CurfewNoNewHHMM { get; private set; } = "";      // e.g., 09:15
    public static string CurfewForceFlatHHMM { get; private set; } = "";  // e.g., 09:23:30

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

        // Patch C: nested guard objects
        if (extra.TryGetValue("adr_guard", out var adrGuard) && adrGuard.ValueKind == JsonValueKind.Object)
        {
            AdrGuardEnabled = true;
            if (adrGuard.TryGetProperty("len", out var gLen) && gLen.TryGetInt32(out var v1)) AdrGuardLen = v1;
            if (adrGuard.TryGetProperty("max_used", out var gMax) && gMax.TryGetDecimal(out var v2)) AdrGuardMaxUsed = v2;
            if (adrGuard.TryGetProperty("warn_used", out var gWarn) && gWarn.TryGetDecimal(out var v3)) AdrGuardWarnUsed = v3;
        }
        if (extra.TryGetValue("pd_extreme_guard", out var pdx) && pdx.ValueKind == JsonValueKind.Object)
        {
            PdExtremeGuardEnabled = true;
            if (pdx.TryGetProperty("min_room_atr", out var mra) && mra.TryGetDecimal(out var v)) PdExtremeMinRoomAtr = v;
        }
        if (extra.TryGetValue("vwap_slope_guard", out var vsg) && vsg.ValueKind == JsonValueKind.Object)
        {
            VwapSlopeGuardEnabled = true;
            if (vsg.TryGetProperty("max_sigma_per_min", out var msm) && msm.TryGetDecimal(out var v)) VwapMaxSigmaPerMin = v;
        }
        if (extra.TryGetValue("z_decelerate", out var zdz) && zdz.ValueKind == JsonValueKind.Object)
        {
            ZDecelerateEnabled = true;
            if (zdz.TryGetProperty("need", out var needEl) && needEl.TryGetInt32(out var need)) ZDecelNeed = need;
        }

        // Optional: day PnL kill-switch (R units)
        if (extra.TryGetValue("day_pnl_kill", out var dpk) && dpk.ValueKind == JsonValueKind.Object)
        {
            DayPnlKillEnabled = true;
            if (dpk.TryGetProperty("stop_gain", out var sg) && sg.TryGetDecimal(out var vg)) DayPnlStopGainR = vg;
            if (dpk.TryGetProperty("stop_loss", out var sl) && sl.TryGetDecimal(out var vl)) DayPnlStopLossR = vl;
        }
        // Optional: curfew helper (no-new and/or force-flat clock marks)
        if (extra.TryGetValue("curfew", out var cf) && cf.ValueKind == JsonValueKind.Object)
        {
            CurfewEnabled = true;
            if (cf.TryGetProperty("no_new_hhmm", out var nn) && nn.ValueKind == JsonValueKind.String)
            {
                var s = nn.GetString(); if (!string.IsNullOrWhiteSpace(s)) CurfewNoNewHHMM = s!;
            }
            if (cf.TryGetProperty("force_flat_hhmm", out var ff) && ff.ValueKind == JsonValueKind.String)
            {
                var s = ff.GetString(); if (!string.IsNullOrWhiteSpace(s)) CurfewForceFlatHHMM = s!;
            }
        }
    }

    private static void TryInt(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<int> set)
    {
        if (extra.TryGetValue(key, out var el) && el.TryGetInt32(out var v)) set(v);
    }
    private static void TryDec(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<decimal> set)
    {
        if (extra.TryGetValue(key, out var el) && el.TryGetDecimal(out var v)) set(v);
    }
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
