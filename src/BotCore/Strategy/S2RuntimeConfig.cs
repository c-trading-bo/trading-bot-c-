using System;
using System.Text.Json;
using BotCore.Config;
using Microsoft.Extensions.Logging;

namespace BotCore.Strategy
{
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

        // Patch B: S2 tunables for enhanced strategy control
        public static int AttemptCapsPerSession { get; private set; } = 3;
        public static int AttemptCapsPerDay { get; private set; } = 5;
        public static int MaxTradesPerSession { get; private set; } = 2;
        public static string EntryMode { get; private set; } = "retest";
        public static decimal RetestOffsetTicks { get; private set; } = 1.0m;
        public static decimal IbOrRoomGuardAtrMult { get; private set; } = 1.5m;
        public static decimal GapGuardThreshold { get; private set; } = 8.0m;
        public static decimal RsPeerFilterThreshold { get; private set; } = 0.3m;
        public static decimal RollWeekSigmaBump { get; private set; } = 0.2m;
        public static bool PriorDayMagnetVetoEnabled { get; private set; } = true;
        public static decimal PriorDayVwapThreshold { get; private set; } = 2.0m;
        public static decimal PriorDayCloseThreshold { get; private set; } = 4.0m;

        public static void ApplyFrom(StrategyDef def)
        {
            if (def is null) return;
            var extra = def.Extra;
            try { if (extra.TryGetValue("tf1", out var el) && el.ValueKind == JsonValueKind.String) Tf1 = el.GetString() ?? Tf1; } catch { }
            try { if (extra.TryGetValue("tf2", out var el) && el.ValueKind == JsonValueKind.String) Tf2 = el.GetString() ?? Tf2; } catch { }
            TryInt(extra, "atr_len", v => AtrLen = v);
            TryDec(extra, "sigma_enter", v => SigmaEnter = v);
            TryDec(extra, "atr_enter", v => AtrEnter = v);
            TryDec(extra, "sigma_force_trend", v => SigmaForceTrend = v);
            TryDec(extra, "min_slope_tf2", v => MinSlopeTf2 = v);
            if (extra.TryGetValue("volz", out var volzEl) && volzEl.ValueKind == JsonValueKind.Object)
            {
                try { if (volzEl.TryGetProperty("min", out var mn) && mn.TryGetDecimal(out var v)) VolZMin = v; } catch { }
                try { if (volzEl.TryGetProperty("max", out var mx) && mx.TryGetDecimal(out var v)) VolZMax = v; } catch { }
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
            // news_block not used here; global gates already cover minute gates
            TryInt(extra, "min_volume", v => MinVolume = v);
            TryInt(extra, "max_spread_ticks", v => MaxSpreadTicks = v);

            // Patch B: Parse S2 tunables
            TryInt(extra, "attempt_caps_per_session", v => AttemptCapsPerSession = v);
            TryInt(extra, "attempt_caps_per_day", v => AttemptCapsPerDay = v);
            TryInt(extra, "max_trades_per_session", v => MaxTradesPerSession = v);
            try { if (extra.TryGetValue("entry_mode", out var el) && el.ValueKind == JsonValueKind.String) EntryMode = el.GetString() ?? EntryMode; } catch { }
            TryDec(extra, "retest_offset_ticks", v => RetestOffsetTicks = v);
            TryDec(extra, "ib_or_room_guard_atr_mult", v => IbOrRoomGuardAtrMult = v);
            TryDec(extra, "gap_guard_threshold", v => GapGuardThreshold = v);
            TryDec(extra, "rs_peer_filter_threshold", v => RsPeerFilterThreshold = v);
            TryDec(extra, "roll_week_sigma_bump", v => RollWeekSigmaBump = v);
            TryBool(extra, "prior_day_magnet_veto_enabled", v => PriorDayMagnetVetoEnabled = v);
            TryDec(extra, "prior_day_vwap_threshold", v => PriorDayVwapThreshold = v);
            TryDec(extra, "prior_day_close_threshold", v => PriorDayCloseThreshold = v);
        }

        private static void TryInt(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<int> set)
        {
            try { if (extra.TryGetValue(key, out var el) && el.TryGetInt32(out var v)) set(v); } catch { }
        }
        private static void TryDec(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<decimal> set)
        {
            try { if (extra.TryGetValue(key, out var el) && el.TryGetDecimal(out var v)) set(v); } catch { }
        }
        private static void TryBool(System.Collections.Generic.Dictionary<string, JsonElement> extra, string key, Action<bool> set)
        {
            try { if (extra.TryGetValue(key, out var el) && el.ValueKind == JsonValueKind.True) set(true); else if (el.ValueKind == JsonValueKind.False) set(false); } catch { }
        }

        public static void LogPatchBSettings(Microsoft.Extensions.Logging.ILogger logger)
        {
            logger.LogInformation("[S2] Patch B Settings: attemptCaps={SessionCaps}/{DayCaps}, maxTrades={MaxTrades}, entryMode={EntryMode}, retestOffset={RetestOffset:F1}",
                AttemptCapsPerSession, AttemptCapsPerDay, MaxTradesPerSession, EntryMode, RetestOffsetTicks);
            logger.LogInformation("[S2] Patch B Guards: ibOrRoomGuard={IbOrGuard:F1}xATR, gapGuard={GapGuard:F1}, rsPeerFilter={RsPeer:F2}, rollWeekBump={RollBump:F2}",
                IbOrRoomGuardAtrMult, GapGuardThreshold, RsPeerFilterThreshold, RollWeekSigmaBump);
            logger.LogInformation("[S2] Patch B Prior-Day: veto={VetoEnabled}, vwapThreshold={VwapThreshold:F1}, closeThreshold={CloseThreshold:F1}",
                PriorDayMagnetVetoEnabled, PriorDayVwapThreshold, PriorDayCloseThreshold);
        }
    }
}
