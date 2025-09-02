// PURPOSE: AllStrategies as strategy engine for candidate generation.
using BotCore.Config;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Collections.Concurrent;
using System.Text.Json;
using BotCore.Models;
using BotCore.Risk;
using Microsoft.Extensions.Logging;

namespace BotCore.Strategy
{
    public static class AllStrategies
    {
        private static readonly HighWinRateProfile Profile = new();
        private static readonly TimeZoneInfo Et = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");

        // (attempt accounting moved to specific strategies as needed)
        static decimal rr_quality(decimal entry, decimal stop, decimal t1)
        {
            var r = Math.Abs(entry - stop);
            if (r <= 0) return 0m;
            return Math.Abs(t1 - entry) / r;
        }

        public static List<Candidate> generate_candidates(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var cands = new List<Candidate>();
            // Ensure env.volz is computed from history (regime proxy)
            try { env.volz = VolZ(bars); } catch { env.volz ??= 0m; }
            var attemptCaps = HighWinRateProfile.AttemptCaps;
            bool noAttemptCaps = (Environment.GetEnvironmentVariable("NO_ATTEMPT_CAPS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            var strategyMethods = new List<(string, Func<string, Env, Levels, IList<Bar>, RiskEngine, List<Candidate>>)> {
                ("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4), ("S5", S5), ("S6", S6), ("S7", S7), ("S8", S8), ("S9", S9), ("S10", S10), ("S11", S11), ("S12", S12), ("S13", S13), ("S14", S14)
            };
            foreach (var (id, method) in strategyMethods)
            {
                bool hasCap = attemptCaps.TryGetValue(id, out int cap);
                if (!noAttemptCaps && hasCap && cap == 0) continue;
                var candidates = method(symbol, env, levels, bars, risk);
                if (!noAttemptCaps && hasCap && cap > 0 && candidates.Count > cap) candidates = [.. candidates.Take(cap)];
                cands.AddRange(candidates);
            }
            return cands;
        }

        // Config-aware method for StrategyAgent
        public static List<Signal> generate_candidates(string symbol, TradingProfileConfig cfg, StrategyDef def, List<Bar> bars, object risk, BotCore.Models.MarketSnapshot snap)
        {
            // Warm-up disabled: always allow indicator use immediately

            // Dispatch to the specific strategy function based on def.Name (S1..S14)
            var env = new Env { atr = bars.Count > 0 ? (decimal?)Math.Abs(bars[^1].High - bars[^1].Low) : null, volz = VolZ(bars) };
            var levels = new Levels();
            var riskEngine = risk as RiskEngine ?? new RiskEngine();

            var map = new Dictionary<string, Func<string, Env, Levels, IList<Bar>, RiskEngine, List<Candidate>>>(StringComparer.OrdinalIgnoreCase)
            {
                ["S1"] = S1,
                ["S2"] = S2,
                ["S3"] = S3,
                ["S4"] = S4,
                ["S5"] = S5,
                ["S6"] = S6,
                ["S7"] = S7,
                ["S8"] = S8,
                ["S9"] = S9,
                ["S10"] = S10,
                ["S11"] = S11,
                ["S12"] = S12,
                ["S13"] = S13,
                ["S14"] = S14,
            };

            if (!def.Enabled) return [];
            if (!map.TryGetValue(def.Name, out var fn)) return [];

            var candidates = fn(symbol, env, levels, bars, riskEngine);

            // Family weighting
            var family = def.Family ?? "breakout";
            var famW = StrategyGates.ScoreWeight(cfg, snap, family);

            var signals = new List<Signal>();
            // Session-aware QScore thresholds (env-overridable)
            static decimal EnvDec(string key, decimal dv)
            {
                var v = Environment.GetEnvironmentVariable(key);
                return (v != null && decimal.TryParse(v, out var x)) ? x : dv;
            }
            var qThNight = EnvDec("QTH_NIGHT", 0.80m);
            var qThOpen = EnvDec("QTH_OPEN", 0.85m);
            var qThRth = EnvDec("QTH_RTH", 0.75m);
            // Map last bar time to ET and choose threshold
            DateTime lastEt;
            try
            {
                var last = bars.Count > 0 ? bars[^1].Start : DateTime.UtcNow;
                var lastUtc = last.Kind == DateTimeKind.Utc ? last : DateTime.SpecifyKind(last, DateTimeKind.Utc);
                lastEt = TimeZoneInfo.ConvertTimeFromUtc(lastUtc, Et);
            }
            catch { lastEt = DateTime.UtcNow; }
            var tod = lastEt.TimeOfDay;
            decimal qTh = qThRth;
            if (tod >= new TimeSpan(20, 0, 0) || tod < new TimeSpan(8, 0, 0)) qTh = qThNight; // 20:00–08:00
            else if (tod >= new TimeSpan(9, 28, 0) && tod <= new TimeSpan(10, 30, 0)) qTh = qThOpen; // 09:28–10:30
            foreach (var c in candidates)
            {
                // Quality-first gate
                if (c.QScore < qTh) continue;
                var baseQty = (int)c.qty;
                var scaledQty = (int)Math.Floor((double)(baseQty * StrategyGates.SizeScale(cfg, snap)));
                if (scaledQty < 1) scaledQty = 1;
                signals.Add(new Signal
                {
                    StrategyId = c.strategy_id,
                    Symbol = c.symbol,
                    Side = c.side.ToString(),
                    Entry = InstrumentMeta.RoundToTick(c.symbol, c.entry),
                    Stop = InstrumentMeta.RoundToTick(c.symbol, c.stop),
                    Target = InstrumentMeta.RoundToTick(c.symbol, c.t1),
                    ExpR = c.expR,
                    Score = c.Score * famW,
                    QScore = c.QScore,
                    Size = Math.Max(InstrumentMeta.LotStep(c.symbol), scaledQty - (scaledQty % InstrumentMeta.LotStep(c.symbol))),
                    AccountId = c.accountId,
                    ContractId = c.contractId,
                    Tag = c.Tag,
                    StrategyVersion = def.Version ?? "1.0.0",
                    ProfileName = string.IsNullOrWhiteSpace(cfg.Profile) ? "default" : cfg.Profile,
                    EmittedUtc = DateTime.UtcNow
                });
            }
            return signals;
        }

        // Deterministic combined candidate flow (no forced trade); config-aware with defs list
        public static List<Signal> generate_candidates(
            string symbol, Env env, Levels levels, IList<Bar> bars,
            IList<StrategyDef> defs, RiskEngine risk, TradingProfileConfig profile, BotCore.Models.MarketSnapshot snap, int max = 10)
        {
            var map = new Dictionary<string, Func<string, Env, Levels, IList<Bar>, RiskEngine, List<Candidate>>>(StringComparer.OrdinalIgnoreCase)
            {
                ["S1"] = S1,
                ["S2"] = S2,
                ["S3"] = S3,
                ["S4"] = S4,
                ["S5"] = S5,
                ["S6"] = S6,
                ["S7"] = S7,
                ["S8"] = S8,
                ["S9"] = S9,
                ["S10"] = S10,
                ["S11"] = S11,
                ["S12"] = S12,
                ["S13"] = S13,
                ["S14"] = S14,
            };
            var signals = new List<Signal>();
            foreach (var def in defs.Where(d => d.Enabled))
            {
                if (!map.TryGetValue(def.Name, out var fn)) continue;
                if (!StrategyGates.PassesGlobal(profile, snap)) continue; // AlwaysOn => true
                var raw = fn(symbol, env, levels, bars, risk);
                var family = def.Family ?? "breakout";
                var w = StrategyGates.ScoreWeight(profile, snap, family);
                foreach (var c in raw)
                {
                    // Light quality gate using same env thresholds
                    static decimal EnvDec(string key, decimal dv)
                    {
                        var v = Environment.GetEnvironmentVariable(key);
                        return (v != null && decimal.TryParse(v, out var x)) ? x : dv;
                    }
                    var qThNight = EnvDec("QTH_NIGHT", 0.80m);
                    var qThOpen = EnvDec("QTH_OPEN", 0.85m);
                    var qThRth = EnvDec("QTH_RTH", 0.75m);
                    DateTime lastEt;
                    try
                    {
                        var last = bars.Count > 0 ? bars[^1].Start : DateTime.UtcNow;
                        var lastUtc = last.Kind == DateTimeKind.Utc ? last : DateTime.SpecifyKind(last, DateTimeKind.Utc);
                        lastEt = TimeZoneInfo.ConvertTimeFromUtc(lastUtc, Et);
                    }
                    catch { lastEt = DateTime.UtcNow; }
                    var tod = lastEt.TimeOfDay; decimal qTh = qThRth;
                    if (tod >= new TimeSpan(20, 0, 0) || tod < new TimeSpan(8, 0, 0)) qTh = qThNight; else if (tod >= new TimeSpan(9, 28, 0) && tod <= new TimeSpan(10, 30, 0)) qTh = qThOpen;
                    if (c.QScore < qTh) continue;
                    var s = new Signal
                    {
                        StrategyId = def.Name,
                        Symbol = symbol,
                        Side = c.side.ToString(),
                        Entry = c.entry,
                        Stop = c.stop,
                        Target = c.t1,
                        ExpR = c.expR,
                        Score = c.Score * w,
                        QScore = c.QScore,
                        Size = (int)c.qty,
                        Tag = c.Tag
                    };
                    signals.Add(s);
                }
            }
            return [.. signals
                .OrderByDescending(x => x.Score)
                .DistinctBy(x => (x.Side, x.StrategyId, Math.Round(x.Entry, 2), Math.Round(x.Target, 2), Math.Round(x.Stop, 2)))
                .Take(max)];
        }

        // Config-aware method for StrategyAgent
        public static List<Signal> generate_signals(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk, long accountId, string contractId)
        {
            // Use the enumerating candidate method to include S1..S14
            var candidates = generate_candidates(symbol, env, levels, bars, risk);
            var signals = new List<Signal>();
            foreach (var c in candidates)
            {
                signals.Add(new Signal
                {
                    StrategyId = c.strategy_id,
                    Symbol = c.symbol,
                    Side = c.side.ToString(),
                    Entry = InstrumentMeta.RoundToTick(c.symbol, c.entry),
                    Stop = InstrumentMeta.RoundToTick(c.symbol, c.stop),
                    Target = InstrumentMeta.RoundToTick(c.symbol, c.t1),
                    ExpR = c.expR,
                    Score = c.Score,
                    QScore = c.QScore,
                    Size = Math.Max(InstrumentMeta.LotStep(c.symbol), ((int)c.qty) - (((int)c.qty) % InstrumentMeta.LotStep(c.symbol))),
                    AccountId = accountId,
                    ContractId = contractId,
                    Tag = c.Tag,
                    StrategyVersion = "1.0.0",
                    ProfileName = HighWinRateProfile.Profile,
                    EmittedUtc = DateTime.UtcNow
                });
            }
            return signals;
        }

        // S1–S14 strategies
        public static List<Candidate> S1(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            // Zero-warmup versions allow immediate use; need at least 2 bars for cross checks
            const int fastLen = 9;
            const int slowLen = 21;
            const int atrLen = 14;
            if (bars is null || bars.Count < 2) return lst;

            // Compute EMAs and ATR (no warmup seed)
            var emaFast = EmaNoWarmup(bars, fastLen);
            var emaSlow = EmaNoWarmup(bars, slowLen);
            var atr = AtrNoWarmup(bars, atrLen);
            int n = bars.Count - 1;
            var last = bars[n].Close;

            bool bullCross = emaFast[n - 1] <= emaSlow[n - 1] && emaFast[n] > emaSlow[n];
            bool bearCross = emaFast[n - 1] >= emaSlow[n - 1] && emaFast[n] < emaSlow[n];
            bool fastUp = emaFast[n] > emaFast[n - 1];
            bool fastDown = emaFast[n] < emaFast[n - 1];

            // Simple RS gate proxy using env.volz if present (usable band: [0.5, 2.0))
            var rsOk = !env.volz.HasValue || (Math.Abs(env.volz.Value) >= 0.5m && Math.Abs(env.volz.Value) < 2.0m);
            // Basic ATR floor
            var atrOk = atr > 0m;

            if (rsOk && atrOk && bullCross && fastUp)
            {
                var stop = last - 1.5m * atr;
                var t1 = last + 2.0m * atr;
                if (t1 > last && stop < last)
                {
                    var e = new Env { Symbol = symbol, atr = atr, volz = env.volz };
                    add_cand(lst, "S1", symbol, "BUY", last, stop, t1, e, risk);
                }
            }
            if (rsOk && atrOk && bearCross && fastDown)
            {
                var stop = last + 1.5m * atr;
                var t1 = last - 2.0m * atr;
                if (t1 < last && stop > last)
                {
                    var e = new Env { Symbol = symbol, atr = atr, volz = env.volz };
                    add_cand(lst, "S1", symbol, "SELL", last, stop, t1, e, risk);
                }
            }
            return lst;
        }

        // --- helpers ---
        private static List<decimal> EmaNoWarmup(IList<Bar> bars, int len)
        {
            var ema = new List<decimal>(new decimal[bars.Count]);
            if (bars.Count == 0) return ema;
            var k = 2m / (len + 1m);
            ema[0] = bars[0].Close;
            for (int i = 1; i < bars.Count; i++)
                ema[i] = ema[i - 1] + k * (bars[i].Close - ema[i - 1]);
            return ema;
        }

        private static decimal AtrNoWarmup(IList<Bar> bars, int len)
        {
            if (bars.Count == 0) return 0m;
            decimal atr = bars[0].High - bars[0].Low;
            for (int i = 1; i < bars.Count; i++)
            {
                var h = bars[i].High; var l = bars[i].Low; var pc = bars[i - 1].Close;
                var tr = Math.Max(h - l, Math.Max(Math.Abs(h - pc), Math.Abs(l - pc)));
                atr += (tr - atr) / len; // Wilder smoothing seeded by first TR
            }
            return atr;
        }

        // Regime proxy: z-score of recent returns (default lookback = 50 bars)
        private static decimal VolZ(IList<Bar> bars, int lookback = 50)
        {
            if (bars is null || bars.Count < lookback + 1) return 0m;
            var rets = new List<decimal>(lookback);
            for (int i = bars.Count - lookback; i < bars.Count; i++)
            {
                var p0 = bars[i - 1].Close; var p1 = bars[i].Close;
                if (p0 == 0) continue;
                rets.Add((p1 - p0) / p0);
            }
            if (rets.Count == 0) return 0m;
            var mean = rets.Average();
            var varv = rets.Select(r => (r - mean) * (r - mean)).Average();
            var std = (decimal)Math.Sqrt((double)Math.Max(1e-12m, varv));
            var last = rets[^1];
            return std == 0m ? 0m : (last - mean) / std;
        }

        // === S2 helpers anchored to local 09:30 session ===
        private static DateTime AnchorToday(DateTime localDate, TimeSpan time) => localDate.Date + time;

        private static int IndexFromLocalAnchor(IList<Bar> bars, DateTime anchorLocal)
        {
            for (int i = 0; i < bars.Count; i++)
            {
                var t = bars[i].Start; // Start assumed local; if UTC adjust upstream
                if (t >= anchorLocal) return i;
            }
            return -1;
        }

        private static (decimal vwap, decimal wvar, decimal wvol) SessionVwapAndVar(IList<Bar> bars, DateTime anchorLocal)
        {
            decimal wv = 0m, vol = 0m;
            var idx0 = IndexFromLocalAnchor(bars, anchorLocal);
            if (idx0 < 0) return (0m, 0m, 0m);
            for (int i = idx0; i < bars.Count; i++)
            {
                var b = bars[i];
                var tp = (b.High + b.Low + b.Close) / 3m;
                var v = Math.Max(0, b.Volume);
                wv += tp * v;
                vol += v;
            }
            if (vol <= 0) return (0m, 0m, 0m);
            var vwap = wv / vol;
            decimal num = 0m;
            for (int i = idx0; i < bars.Count; i++)
            {
                var b = bars[i];
                var tp = (b.High + b.Low + b.Close) / 3m;
                var v = Math.Max(0, b.Volume);
                var d = tp - vwap;
                num += d * d * v;
            }
            var wvar = vol > 0 ? num / vol : 0m;
            return (vwap, wvar, vol);
        }

        // Smart ON/RTH anchor: if before 09:30 local, anchor to 20:00 (prev/today) else 09:30 today
        private static readonly TimeSpan NightStart = new(20, 0, 0);
        private static readonly TimeSpan NightEnd = new(2, 0, 0);
        private static readonly TimeSpan RthOpen = new(9, 30, 0);
        private static DateTime SmartAnchor(DateTime nowLocal)
        {
            var rthStart = nowLocal.Date + RthOpen;
            var onStartDate = (nowLocal.TimeOfDay >= NightStart ? nowLocal.Date : nowLocal.Date.AddDays(-1));
            var onStart = onStartDate + NightStart;
            return nowLocal < rthStart ? onStart : rthStart;
        }

        private static (decimal high, decimal low) InitialBalance(IList<Bar> bars, DateTime startLocal, DateTime endLocal)
        {
            decimal hi = 0m, lo = 0m; bool init = false;
            for (int i = 0; i < bars.Count; i++)
            {
                var t = bars[i].Start;
                if (t < startLocal) continue;
                if (t >= endLocal) break;
                if (!init) { hi = bars[i].High; lo = bars[i].Low; init = true; }
                else { if (bars[i].High > hi) hi = bars[i].High; if (bars[i].Low < lo) lo = bars[i].Low; }
            }
            return init ? (hi, lo) : (0m, 0m);
        }

        private static int AboveVWAPCount(IList<Bar> b, decimal vwap, int look)
        { int cnt = 0; for (int i = Math.Max(0, b.Count - look); i < b.Count; i++) if (b[i].Close > vwap) cnt++; return cnt; }
        private static int BelowVWAPCount(IList<Bar> b, decimal vwap, int look)
        { int cnt = 0; for (int i = Math.Max(0, b.Count - look); i < b.Count; i++) if (b[i].Close < vwap) cnt++; return cnt; }

        private static bool BullConfirm(IList<Bar> b)
        {
            if (b.Count < 3) return false; var a = b[^2]; var c = b[^1];
            bool engulf = c.Close > a.Open && c.Open < a.Close && c.Close > c.Open;
            bool hammer = (c.Close >= c.Open) && (c.Open - c.Low) >= 0.5m * (c.High - c.Low) && (c.High - c.Close) <= 0.3m * (c.High - c.Low);
            return engulf || hammer;
        }
        private static bool BearConfirm(IList<Bar> b)
        {
            if (b.Count < 3) return false; var a = b[^2]; var c = b[^1];
            bool engulf = c.Close < a.Open && c.Open > a.Close && c.Close < c.Open;
            bool shoot = (c.Close <= c.Open) && (c.High - c.Open) >= 0.5m * (c.High - c.Low) && (c.Close - c.Low) <= 0.3m * (c.High - c.Low);
            return engulf || shoot;
        }

        // Keltner Channel helper: EMA mid and ATR band
        private static (decimal mid, decimal up, decimal dn) Keltner(IList<Bar> bars, int emaLen = 20, int atrLen = 20, decimal mult = 1.5m)
        {
            var e = EmaNoWarmup(bars, emaLen);
            var a = AtrNoWarmup(bars, atrLen);
            var mid = e[^1];
            return (mid, mid + mult * a, mid - mult * a);
        }

        public static List<Candidate> S2(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars is null || bars.Count < 60) return lst;
            // Hard microstructure: require minimum 1m volume per config
            if (bars[^1].Volume < S2RuntimeConfig.MinVolume) return lst;
            // Normalize timestamps to ET for session anchoring
            static DateTime ToEt(Bar b, TimeZoneInfo et)
            {
                try
                {
                    DateTime utc = b.Ts > 0
                        ? DateTimeOffset.FromUnixTimeMilliseconds(b.Ts).UtcDateTime
                        : (b.Start.Kind == DateTimeKind.Utc ? b.Start : DateTime.SpecifyKind(b.Start, DateTimeKind.Utc));
                    return TimeZoneInfo.ConvertTimeFromUtc(utc, et);
                }
                catch { return b.Start; }
            }
            var barsEt = new List<Bar>(bars.Count);
            foreach (var b in bars)
            {
                barsEt.Add(new Bar
                {
                    Start = ToEt(b, Et),
                    Ts = b.Ts,
                    Symbol = b.Symbol,
                    Open = b.Open,
                    High = b.High,
                    Low = b.Low,
                    Close = b.Close,
                    Volume = b.Volume
                });
            }
            bars = barsEt;

            // Compute session VWAP/σ anchored to 09:30 using the last bar's time as "now" (backtest-friendly)
            var nowLocal = bars[^1].Start; var localDate = nowLocal.Date;
            var anchor = SmartAnchor(nowLocal);
            var (vwap, wvar, wvol) = SessionVwapAndVar(bars, anchor);
            if (wvol <= 0 || vwap <= 0) return lst;
            var sigma = (decimal)Math.Sqrt((double)Math.Max(1e-12m, wvar));

            var atr = (env.atr.HasValue && env.atr.Value > 0m) ? env.atr.Value : AtrNoWarmup(bars, S2RuntimeConfig.AtrLen);
            if (atr <= 0) return lst;
            var volz = VolZ(bars, 50);
            if (!(volz >= S2RuntimeConfig.VolZMin && volz <= S2RuntimeConfig.VolZMax)) return lst; // regime band

            var px = bars[^1].Close;
            var d = px - vwap;
            var z = sigma > 0 ? d / sigma : 0m;
            var a = atr > 0 ? d / atr : 0m;
            try { S2Quantiles.Observe(symbol, nowLocal, Math.Abs(z)); } catch { }

            // ADR guards: compute simple rolling ADR and today's realized range
            decimal adr = 0m;
            int look = Math.Max(5, S2RuntimeConfig.AdrGuardEnabled ? S2RuntimeConfig.AdrGuardLen : S2RuntimeConfig.AdrLookbackDays);
            int daysCounted = 0; decimal sumAdr = 0m;
            for (int i = bars.Count - 1; i >= 0 && daysCounted < look; i--)
            {
                var day = bars[i].Start.Date;
                var dayStartIdx = i;
                while (dayStartIdx - 1 >= 0 && bars[dayStartIdx - 1].Start.Date == day) dayStartIdx--;
                var dayEndIdx = i;
                decimal hi = bars[dayStartIdx].High, lo = bars[dayStartIdx].Low;
                for (int j = dayStartIdx; j <= dayEndIdx; j++) { if (bars[j].High > hi) hi = bars[j].High; if (bars[j].Low < lo) lo = bars[j].Low; }
                sumAdr += Math.Max(0m, hi - lo);
                daysCounted++;
                i = dayStartIdx; // for-loop will i-- again
            }
            if (daysCounted > 0) adr = sumAdr / daysCounted;
            // Today's realized range
            decimal todayHi = 0m, todayLo = 0m; bool todayInit = false;
            var today = nowLocal.Date;
            for (int i = 0; i < bars.Count; i++)
            {
                if (bars[i].Start.Date != today) continue;
                if (!todayInit) { todayHi = bars[i].High; todayLo = bars[i].Low; todayInit = true; }
                else { if (bars[i].High > todayHi) todayHi = bars[i].High; if (bars[i].Low < todayLo) todayLo = bars[i].Low; }
            }
            var todayRange = todayInit ? (todayHi - todayLo) : 0m;
            if (adr > 0m)
            {
                // Exhaustion cap
                if (todayRange > S2RuntimeConfig.AdrExhaustionCap * adr) return lst;
                // Optional: used% guard (Patch C)
                if (S2RuntimeConfig.AdrGuardEnabled && S2RuntimeConfig.AdrGuardMaxUsed > 0m)
                {
                    var used = adr == 0m ? 0m : (todayRange / adr);
                    if (used >= S2RuntimeConfig.AdrGuardMaxUsed) return lst;
                }
            }

            // Instrument-specific σ triggers
            bool isNq = symbol.Contains("NQ", StringComparison.OrdinalIgnoreCase);
            decimal esSigma = S2RuntimeConfig.EsSigma, nqSigma = S2RuntimeConfig.NqSigma;
            decimal needSigma = isNq ? nqSigma : esSigma;
            decimal baseSigma = Math.Max(needSigma, S2RuntimeConfig.SigmaEnter);
            decimal baseAtr = S2RuntimeConfig.AtrEnter;

            // Roll-week bump (env switch + simple auto-detect fallback)
            try
            {
                bool isRoll = false;
                var roll = Environment.GetEnvironmentVariable("ROLL_WEEK");
                isRoll = !string.IsNullOrWhiteSpace(roll) && (roll.Equals("1", StringComparison.OrdinalIgnoreCase) || roll.Equals("true", StringComparison.OrdinalIgnoreCase));
                if (!isRoll)
                {
                    var rollDay = DateTime.UtcNow.Date;
                    if (rollDay.Month is 3 or 6 or 9 or 12)
                    {
                        DateTime first = new(rollDay.Year, rollDay.Month, 1);
                        int add = ((int)DayOfWeek.Friday - (int)first.DayOfWeek + 7) % 7;
                        DateTime firstFri = first.AddDays(add);
                        DateTime secondFri = firstFri.AddDays(7);
                        DateTime thuBefore = secondFri.AddDays(-1);
                        if ((rollDay - thuBefore).Duration() <= TimeSpan.FromDays(1.5)) isRoll = true;
                    }
                }
                if (isRoll && S2RuntimeConfig.RollWeekSigmaBump > 0m)
                    needSigma += S2RuntimeConfig.RollWeekSigmaBump;
            }
            catch { }

            // Curfew: optional no-new window (Patch C)
            if (S2RuntimeConfig.CurfewEnabled && !string.IsNullOrWhiteSpace(S2RuntimeConfig.CurfewNoNewHHMM))
            {
                if (TimeSpan.TryParse(S2RuntimeConfig.CurfewNoNewHHMM, out var tNoNew))
                {
                    var tNow = nowLocal.TimeOfDay;
                    // Only apply before U.S. open; we don't flatten here (router handles flattening if needed)
                    if (tNow >= tNoNew && tNow < new TimeSpan(9, 30, 0)) return lst;
                }
            }

            // Trend day detection proxy: EMA20 slope over last 5 bars + VWAP streak
            var ema20 = EmaNoWarmup(bars, 20).ToArray();
            decimal slope = ema20.Length > 5 ? (ema20[^1] - ema20[^6]) / 5m : 0m;
            var tickSz = InstrumentMeta.Tick(symbol);
            var slopeTicks = tickSz > 0 ? slope / tickSz : slope; // ticks per bar
            // VWAP slope guard (Patch C): too steep weighted move → skip
            if (S2RuntimeConfig.VwapSlopeGuardEnabled)
            {
                // Approximate VWAP slope using EMA20 slope scaled by σ units per bar; fallback if sigma <= 0
                var sigmaPerBar = sigma > 0 ? (slope / sigma) : 0m;
                var sigmaPerMin = sigmaPerBar; // 1m bars
                if (Math.Abs(sigmaPerMin) > S2RuntimeConfig.VwapMaxSigmaPerMin) return lst;
            }
            var strongUp = slopeTicks > S2RuntimeConfig.MinSlopeTf2 && AboveVWAPCount(bars, vwap, 12) >= 9; // 75% bars above
            var strongDn = slopeTicks < -S2RuntimeConfig.MinSlopeTf2 && BelowVWAPCount(bars, vwap, 12) >= 9;
            if (strongUp && z < 0) needSigma = Math.Max(needSigma, S2RuntimeConfig.SigmaForceTrend);
            if (strongDn && z > 0) needSigma = Math.Max(needSigma, S2RuntimeConfig.SigmaForceTrend);

            // RS peer directional veto (Patch B): use other index short-term slope to block fades against strong move
            try
            {
                string peer = symbol.Contains("ES", StringComparison.OrdinalIgnoreCase) ? "NQ" : symbol.Contains("NQ", StringComparison.OrdinalIgnoreCase) ? "ES" : string.Empty;
                if (!string.IsNullOrWhiteSpace(peer) && S2RuntimeConfig.RsPeerThreshold > 0m && ExternalGetBars != null)
                {
                    var pb = ExternalGetBars(peer) ?? [];
                    var pbl = pb as IList<Bar> ?? [.. pb];
                    if (pbl.Count >= 10)
                    {
                        var pEma = EmaNoWarmup(pbl, 20).ToArray();
                        var pslope = pEma.Length > 5 ? (pEma[^1] - pEma[^6]) / 5m : 0m;
                        var ptick = InstrumentMeta.Tick(peer);
                        var pslopeTicks = ptick > 0 ? pslope / ptick : pslope;
                        // If peer strongly trending up, veto new shorts; vice versa
                        if (pslopeTicks >= S2RuntimeConfig.RsPeerThreshold && z > 0)
                            return lst;
                        if (pslopeTicks <= -S2RuntimeConfig.RsPeerThreshold && z < 0)
                            return lst;
                    }
                }
            }
            catch { }

            // IB continuation filter after 10:30: avoid small fades unless extreme (use last bar's clock)
            var now = nowLocal; var nowMin = now.Hour * 60 + now.Minute;
            if (nowMin >= S2RuntimeConfig.IbEndMinute)
            {
                var (ibh, ibl) = InitialBalance(bars, AnchorToday(localDate, new TimeSpan(9, 30, 0)), AnchorToday(localDate, new TimeSpan(10, 30, 0)));
                if (ibh > 0 && ibl > 0)
                {
                    bool brokeUp = bars[^1].Close > ibh && bars.Skip(Math.Max(0, bars.Count - 6)).Min(b => b.Low) > ibh - 0.25m * atr;
                    bool brokeDn = bars[^1].Close < ibl && bars.Skip(Math.Max(0, bars.Count - 6)).Max(b => b.High) < ibl + 0.25m * atr;
                    if ((brokeUp && z < 0 && Math.Abs(z) < S2RuntimeConfig.SigmaForceTrend) || (brokeDn && z > 0 && Math.Abs(z) < S2RuntimeConfig.SigmaForceTrend))
                        return lst;
                }
            }

            // Reclaim/reject checks at ±2σ
            var up2 = vwap + 2m * sigma; var dn2 = vwap - 2m * sigma;
            bool reclaimDown = bars.Count >= 2 && bars[^2].Low <= dn2 && bars[^1].Close > dn2;
            bool rejectUp = bars.Count >= 2 && bars[^2].High >= up2 && bars[^1].Close < up2;

            // Dynamic sigma threshold + microstructure safety
            var qAdj = S2Quantiles.GetSigmaFor(symbol, nowLocal, needSigma);
            decimal dynSigma = S2Upg.DynamicSigmaThreshold(Math.Max(needSigma, qAdj), volz, slopeTicks, nowLocal, symbol);
            var imb = S2Upg.UpDownImbalance(bars, 10);
            var tickSize = InstrumentMeta.Tick(symbol);
            bool pivotOKLong = S2Upg.PivotDistanceOK(bars, px, atr, tickSize, true);
            bool pivotOKShort = S2Upg.PivotDistanceOK(bars, px, atr, tickSize, false);
            // Additional room vs prior-day extremes and deceleration toward VWAP
            var minRoomAtr = S2RuntimeConfig.PdExtremeGuardEnabled ? Math.Max(0.10m, S2RuntimeConfig.PdExtremeMinRoomAtr) : 0.25m;
            bool roomLong = S2Upg.HasRoomVsPriorExtremes(bars, nowLocal, px, tickSize, atr, true, (minRoomAtr, 4));
            bool roomShort = S2Upg.HasRoomVsPriorExtremes(bars, nowLocal, px, tickSize, atr, false, (minRoomAtr, 4));
            int decelNeed = S2RuntimeConfig.ZDecelerateEnabled ? Math.Max(2, S2RuntimeConfig.ZDecelNeed) : S2RuntimeConfig.ConfirmLookback;
            bool decel = S2Upg.ZDecelerating(bars, vwap, sigma, decelNeed);

            // Prior-day magnet veto near PD VWAP/CLOSE (Patch B toggles)
            try
            {
                if (S2RuntimeConfig.PriorDayVwapVeto || S2RuntimeConfig.PriorDayCloseVeto)
                {
                    var (pdVwap, pdClose) = S2Upg.PriorDayVwapClose(bars, nowLocal);
                    var atrRef = Math.Max(tickSz > 0 ? tickSz : 0.25m, atr);
                    var nearThresh = 0.30m * atrRef;
                    if (pdVwap > 0m && S2RuntimeConfig.PriorDayVwapVeto && Math.Abs(px - pdVwap) <= nearThresh)
                        return lst;
                    if (pdClose > 0m && S2RuntimeConfig.PriorDayCloseVeto && Math.Abs(px - pdClose) <= nearThresh)
                        return lst;
                }
            }
            catch { }

            // LONG: fade below VWAP
            if ((z <= -dynSigma || a <= -baseAtr) && imb >= 0.9m && pivotOKLong && roomLong && decel)
            {
                if (BullConfirm(bars) || reclaimDown)
                {
                    var entry = px;
                    var dn3 = vwap - 3m * sigma;
                    var swing = bars.Skip(Math.Max(0, bars.Count - S2RuntimeConfig.ConfirmLookback)).Min(b => b.Low);
                    var stop = Math.Min(swing, dn3);
                    if (stop >= entry) stop = entry - 0.25m * atr;
                    var r = entry - stop;
                    var t1 = vwap;
                    // Retest entry mode: wait for micro pullback toward VWAP by offset ticks
                    if (string.Equals(S2RuntimeConfig.EntryMode, "retest", StringComparison.OrdinalIgnoreCase))
                    {
                        var off = Math.Max(0, S2RuntimeConfig.RetestOffsetTicks);
                        var t = InstrumentMeta.Tick(symbol);
                        if (t > 0 && off > 0) entry = Math.Min(t1 - t, px + off * t);
                    }
                    // Require minimum room to target vs ADR
                    if (adr > 0m)
                    {
                        var room = Math.Abs(t1 - entry);
                        if (room < S2RuntimeConfig.AdrRoomFrac * adr) return lst;
                    }
                    if (t1 - entry < 0.8m * r) t1 = entry + 0.9m * r;
                    add_cand(lst, "S2", symbol, "BUY", entry, stop, t1, env, risk);
                }
            }
            // SHORT: fade above VWAP
            else if ((z >= dynSigma || a >= baseAtr) && imb <= 1.1m && pivotOKShort && roomShort && decel)
            {
                if (BearConfirm(bars) || rejectUp)
                {
                    var entry = px;
                    var up3 = vwap + 3m * sigma;
                    var swing = bars.Skip(Math.Max(0, bars.Count - S2RuntimeConfig.ConfirmLookback)).Max(b => b.High);
                    var stop = Math.Max(swing, up3);
                    if (stop <= entry) stop = entry + 0.25m * atr;
                    var r = stop - entry;
                    var t1 = vwap;
                    // Retest entry pricing
                    if (string.Equals(S2RuntimeConfig.EntryMode, "retest", StringComparison.OrdinalIgnoreCase))
                    {
                        var off = Math.Max(0, S2RuntimeConfig.RetestOffsetTicks);
                        var t = InstrumentMeta.Tick(symbol);
                        if (t > 0 && off > 0) entry = Math.Max(t1 + t, px - off * t);
                    }
                    // Require minimum room to target vs ADR
                    if (adr > 0m)
                    {
                        var room = Math.Abs(entry - t1);
                        if (room < S2RuntimeConfig.AdrRoomFrac * adr) return lst;
                    }
                    if (entry - t1 < 0.8m * r) t1 = entry - 0.9m * r;
                    add_cand(lst, "S2", symbol, "SELL", entry, stop, t1, env, risk);
                }
            }
            return lst;
        }

        // Optional external providers (can be set by hosting agent); coded but optional
        public static Func<string, IReadOnlyList<Bar>>? ExternalGetBars { get; set; }
        public static Func<string, int>? ExternalSpreadTicks { get; set; }
        public static Func<string, decimal>? ExternalTickSize { get; set; }

        public static List<Candidate> S3(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
            => S3Strategy.S3(symbol, env, levels, bars, risk);

        // S3 internals moved to S3Strategy.cs

        public static List<Candidate> S4(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.6m)
            {
                var entry = bars[^1].Close;
                var stop = entry - env.atr.Value * 1.5m;
                var t1 = entry + env.atr.Value * 3.0m;
                add_cand(lst, "S4", symbol, "BUY", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S5(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.6m)
            {
                var entry = bars[^1].Close;
                var stop = entry + env.atr.Value * 1.5m;
                var t1 = entry - env.atr.Value * 3.0m;
                add_cand(lst, "S5", symbol, "SELL", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S6(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            var minAtr = S6RuntimeConfig.MinAtr;
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > minAtr)
            {
                var entry = bars[^1].Close;
                var stop = entry - env.atr.Value * S6RuntimeConfig.StopAtrMult;
                var t1 = entry + env.atr.Value * S6RuntimeConfig.TargetAtrMult;
                add_cand(lst, "S6", symbol, "BUY", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S7(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.7m)
            {
                var entry = bars[^1].Close;
                var stop = entry + env.atr.Value * 2.0m;
                var t1 = entry - env.atr.Value * 4.0m;
                add_cand(lst, "S7", symbol, "SELL", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S8(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars is null || bars.Count < 30) return lst;
            var (mid, up, dn) = Keltner(bars, 20, 20, 1.5m);
            var px = bars[^1].Close;
            var ema20 = EmaNoWarmup(bars, 20);
            bool midRising = ema20[^1] > ema20[^2];
            if (midRising && px > mid)
            {
                var entry = px;
                var stop = Math.Min(entry, dn);
                var t1 = up;
                add_cand(lst, "S8", symbol, "BUY", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S9(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.8m)
            {
                var entry = bars[^1].Close;
                var stop = entry + env.atr.Value * 2.5m;
                var t1 = entry - env.atr.Value * 5.0m;
                add_cand(lst, "S9", symbol, "SELL", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S10(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.9m)
            {
                var entry = bars[^1].Close;
                var stop = entry - env.atr.Value * 3.0m;
                var t1 = entry + env.atr.Value * 6.0m;
                add_cand(lst, "S10", symbol, "BUY", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S11(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            var minAtr = S11RuntimeConfig.MinAtr;
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > minAtr)
            {
                var entry = bars[^1].Close;
                var stop = entry + env.atr.Value * S11RuntimeConfig.StopAtrMult;
                var t1 = entry - env.atr.Value * S11RuntimeConfig.TargetAtrMult;
                add_cand(lst, "S11", symbol, "SELL", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S12(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 1.0m)
            {
                var entry = bars[^1].Close;
                var stop = entry - env.atr.Value * 3.5m;
                var t1 = entry + env.atr.Value * 7.0m;
                add_cand(lst, "S12", symbol, "BUY", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S13(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 1.0m)
            {
                var entry = bars[^1].Close;
                var stop = entry + env.atr.Value * 3.5m;
                var t1 = entry - env.atr.Value * 7.0m;
                add_cand(lst, "S13", symbol, "SELL", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S14(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 1.1m)
            {
                var entry = bars[^1].Close;
                var stop = entry - env.atr.Value * 4.0m;
                var t1 = entry + env.atr.Value * 8.0m;
                add_cand(lst, "S14", symbol, "BUY", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static void add_cand(List<Candidate> lst, string sid, string symbol, string sideTxt,
                                 decimal entry, decimal stop, decimal t1, Env env, RiskEngine risk, string? tag = null)
        {
            var pv = InstrumentMeta.PointValue(symbol);
            var tick = InstrumentMeta.Tick(symbol);
            var dist = Math.Max(Math.Abs(entry - stop), tick); // ≥ 1 tick
            // Prefer equity-% aware sizing if configured; pass 0 equity to fallback to fixed RPT when not provided
            var (Qty, UsedRpt) = risk.ComputeSize(symbol, entry, stop, 0m);
            var qty = Qty > 0 ? Qty : (int)RiskEngine.size_for(risk.cfg.risk_per_trade, dist, pv);
            if (qty <= 0) return;

            var expR = rr_quality(entry, stop, t1);
            // Quality score in [0..1]: combine normalized ExpR and regime suitability; clamp for stability
            var expRNorm = Math.Clamp(expR / 3m, 0m, 1m); // consider 3R as near-top quality
            var regime = env.volz.HasValue ? Math.Clamp(1m - Math.Abs(env.volz.Value - 1m) / 3m, 0m, 1m) : 0.5m; // center around ~1
            var qScore = Math.Clamp(0.7m * expRNorm + 0.3m * regime, 0m, 1m);
            // Ranking score retains legacy behavior
            var volBoost = env.volz.HasValue ? Math.Clamp(Math.Abs(env.volz.Value), 0m, 2m) * 0.25m : 0m;
            var score = expR + volBoost;

            var c = new Candidate
            {
                strategy_id = sid,
                symbol = symbol,
                side = sideTxt.Equals("BUY", StringComparison.OrdinalIgnoreCase) ? Side.BUY : Side.SELL,
                entry = entry,
                stop = stop,
                t1 = t1,
                expR = expR,
                qty = qty,
                atr_ok = (env.atr ?? tick) >= tick,
                vol_z = env.volz,
                Score = score,
                QScore = qScore
            };
            lst.Add(c);

            // **ML/RL Integration**: Log signal for training data collection
            try
            {
                // Get bars for technical indicator calculation - this would need to be passed in ideally
                // For now, create minimal data for logging
                var bars = new List<Bar>(); // TODO: Get actual bars from context

                StrategyMlIntegration.LogStrategySignal(
                    // Use a simple console logger for now - in production this would be injected
                    new Microsoft.Extensions.Logging.Abstractions.NullLogger<object>(),
                    sid,
                    symbol,
                    c.side,
                    entry,
                    stop,
                    t1,
                    score,
                    qScore,
                    bars,
                    $"{sid}-{symbol}-{DateTime.UtcNow:yyyyMMddHHmmss}"
                );
            }
            catch (Exception ex)
            {
                // Don't let ML logging break strategy execution
                Console.WriteLine($"[ML-Integration] Failed to log signal for {sid}: {ex.Message}");
            }
        }
    }
}
