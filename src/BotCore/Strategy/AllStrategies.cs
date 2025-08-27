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

namespace BotCore.Strategy
{
    public static class AllStrategies
    {
        private static readonly HighWinRateProfile Profile = new();

        // S3 state and constants
        private static readonly ConcurrentDictionary<string, SegmentState> _segState = new(StringComparer.OrdinalIgnoreCase);
        private static readonly ConcurrentDictionary<(string Sym, DateOnly Day, string Sess, Side Side), int> _attempts = new();
        private static readonly TimeSpan OvernightWinStart = new(2, 55, 0);
        private static readonly TimeSpan OvernightWinEnd   = new(4, 10, 0);
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
            try { env.volz = VolZ(bars); } catch { env.volz = env.volz ?? 0m; }
            var attemptCaps = Profile.AttemptCaps;
            var strategyMethods = new List<(string, Func<string, Env, Levels, IList<Bar>, RiskEngine, List<Candidate>>)> {
                ("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4), ("S5", S5), ("S6", S6), ("S7", S7), ("S8", S8), ("S9", S9), ("S10", S10), ("S11", S11), ("S12", S12), ("S13", S13), ("S14", S14)
            };
            foreach (var (id, method) in strategyMethods) {
                if (attemptCaps.TryGetValue(id, out var cap) && cap == 0) continue;
                var candidates = method(symbol, env, levels, bars, risk);
                if (cap > 0 && candidates.Count > cap) candidates = candidates.Take(cap).ToList();
                cands.AddRange(candidates);
            }
            return cands;
        }

        // Config-aware method for StrategyAgent
        public static List<Signal> generate_candidates(string symbol, TradingProfileConfig cfg, StrategyDef def, List<Bar> bars, object risk, BotCore.Models.MarketSnapshot snap)
        {
            // Warm-up: allow immediate indicators if AlwaysOn.ZeroWarmupIndicators
            int warmup = 20;
            if (def.Extra.TryGetValue("warmup_n", out var wEl) && wEl.TryGetInt32(out var w) && w > 0)
                warmup = w;
            if (!(cfg.AlwaysOn?.ZeroWarmupIndicators ?? false))
            {
                if (bars is null || bars.Count < warmup)
                    return new List<Signal>();
            }

            // Dispatch to the specific strategy function based on def.Name (S1..S14)
            var env = new Env { atr = bars.Count > 0 ? (decimal?)Math.Abs(bars[^1].High - bars[^1].Low) : null, volz = VolZ(bars) };
            var levels = new Levels();
            var riskEngine = risk as RiskEngine ?? new RiskEngine();

            var map = new Dictionary<string, Func<string, Env, Levels, IList<Bar>, RiskEngine, List<Candidate>>>(StringComparer.OrdinalIgnoreCase)
            {
                ["S1"] = S1,  ["S2"] = S2,  ["S3"] = S3,  ["S4"] = S4,  ["S5"] = S5,
                ["S6"] = S6,  ["S7"] = S7,  ["S8"] = S8,  ["S9"] = S9,  ["S10"] = S10,
                ["S11"] = S11,["S12"] = S12,["S13"] = S13,["S14"] = S14,
            };

            if (!def.Enabled) return new List<Signal>();
            if (!map.TryGetValue(def.Name, out var fn)) return new List<Signal>();

            var candidates = fn(symbol, env, levels, bars, riskEngine);

            // Family weighting
            var family = def.Family ?? "breakout";
            var famW = StrategyGates.ScoreWeight(cfg, snap, family);

            var signals = new List<Signal>();
            foreach (var c in candidates)
            {
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
                ["S1"]=S1,["S2"]=S2,["S3"]=S3,["S4"]=S4,["S5"]=S5,["S6"]=S6,["S7"]=S7,
                ["S8"]=S8,["S9"]=S9,["S10"]=S10,["S11"]=S11,["S12"]=S12,["S13"]=S13,["S14"]=S14,
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
                        Size = (int)c.qty,
                        Tag = c.Tag
                    };
                    signals.Add(s);
                }
            }
            return signals
                .OrderByDescending(x => x.Score)
                .DistinctBy(x => (x.Side, x.StrategyId, Math.Round(x.Entry, 2), Math.Round(x.Target, 2), Math.Round(x.Stop, 2)))
                .Take(max)
                .ToList();
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
                    Size = Math.Max(InstrumentMeta.LotStep(c.symbol), ((int)c.qty) - (((int)c.qty) % InstrumentMeta.LotStep(c.symbol))),
                    AccountId = accountId,
                    ContractId = contractId,
                    Tag = c.Tag,
                    StrategyVersion = "1.0.0",
                    ProfileName = new HighWinRateProfile().Profile,
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
            const int atrLen  = 14;
            if (bars is null || bars.Count < 2) return lst;

            // Compute EMAs and ATR (no warmup seed)
            var emaFast = EmaNoWarmup(bars, fastLen);
            var emaSlow = EmaNoWarmup(bars, slowLen);
            var atr = AtrNoWarmup(bars, atrLen);
            int n = bars.Count - 1;
            var last = bars[n].Close;

            bool bullCross = emaFast[n - 1] <= emaSlow[n - 1] && emaFast[n] > emaSlow[n];
            bool bearCross = emaFast[n - 1] >= emaSlow[n - 1] && emaFast[n] < emaSlow[n];
            bool fastUp    = emaFast[n] > emaFast[n - 1];
            bool fastDown  = emaFast[n] < emaFast[n - 1];

            // Simple RS gate proxy using env.volz if present (usable band: [0.5, 2.0))
            var rsOk = !env.volz.HasValue || (Math.Abs(env.volz.Value) >= 0.5m && Math.Abs(env.volz.Value) < 2.0m);
            // Basic ATR floor
            var atrOk = atr > 0m;

            if (rsOk && atrOk && bullCross && fastUp)
            {
                var stop = last - 1.5m * atr;
                var t1   = last + 2.0m * atr;
                if (t1 > last && stop < last)
            {
                var e = new Env { Symbol = symbol, atr = atr, volz = env.volz };
                add_cand(lst, "S1", symbol, "BUY", last, stop, t1, e, risk);
            }
            }
            if (rsOk && atrOk && bearCross && fastDown)
            {
                var stop = last + 1.5m * atr;
                var t1   = last - 2.0m * atr;
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
                atr = atr + (tr - atr) / len; // Wilder smoothing seeded by first TR
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
        private static readonly TimeSpan NightEnd   = new(2, 0, 0);
        private static readonly TimeSpan RthOpen    = new(9, 30, 0);
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

            // Compute session VWAP/σ anchored to 09:30 local (same as ET for Santo Domingo)
            var nowLocal = DateTime.Now; var localDate = nowLocal.Date; // wall-time
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

            // Instrument-specific σ triggers
            bool isNq = symbol.Contains("NQ", StringComparison.OrdinalIgnoreCase);
            decimal esSigma = S2RuntimeConfig.EsSigma, nqSigma = S2RuntimeConfig.NqSigma;
            decimal needSigma = isNq ? nqSigma : esSigma;
            decimal baseSigma = Math.Max(needSigma, S2RuntimeConfig.SigmaEnter);
            decimal baseAtr = S2RuntimeConfig.AtrEnter;

            // Trend day detection proxy: EMA20 slope over last 5 bars + VWAP streak
            var ema20 = EmaNoWarmup(bars, 20).ToArray();
            decimal slope = ema20.Length > 5 ? (ema20[^1] - ema20[^6]) / 5m : 0m;
            var tickSz = InstrumentMeta.Tick(symbol);
            var slopeTicks = tickSz > 0 ? slope / tickSz : slope; // ticks per bar
            var strongUp = slopeTicks > S2RuntimeConfig.MinSlopeTf2 && AboveVWAPCount(bars, vwap, 12) >= 9; // 75% bars above
            var strongDn = slopeTicks < -S2RuntimeConfig.MinSlopeTf2 && BelowVWAPCount(bars, vwap, 12) >= 9;
            if (strongUp && z < 0) needSigma = Math.Max(needSigma, S2RuntimeConfig.SigmaForceTrend);
            if (strongDn && z > 0) needSigma = Math.Max(needSigma, S2RuntimeConfig.SigmaForceTrend);

            // IB continuation filter after 10:30: avoid small fades unless extreme
            var now = DateTime.Now; var nowMin = now.Hour * 60 + now.Minute;
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
            bool rejectUp    = bars.Count >= 2 && bars[^2].High >= up2 && bars[^1].Close < up2;

            // Dynamic sigma threshold + microstructure safety
            var qAdj = S2Quantiles.GetSigmaFor(symbol, nowLocal, needSigma);
            decimal dynSigma = S2Upg.DynamicSigmaThreshold(Math.Max(needSigma, qAdj), volz, slopeTicks, nowLocal, symbol);
            var imb = S2Upg.UpDownImbalance(bars, 10);
            var tickSize = InstrumentMeta.Tick(symbol);
            bool pivotOKLong  = S2Upg.PivotDistanceOK(bars, px, atr, tickSize, true);
            bool pivotOKShort = S2Upg.PivotDistanceOK(bars, px, atr, tickSize, false);

            // LONG: fade below VWAP
            if ((z <= -dynSigma || a <= -baseAtr) && imb >= 0.9m && pivotOKLong)
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
                    if (t1 - entry < 0.8m * r) t1 = entry + 0.9m * r;
                    add_cand(lst, "S2", symbol, "BUY", entry, stop, t1, env, risk);
                }
            }
            // SHORT: fade above VWAP
            else if ((z >= dynSigma || a >= baseAtr) && imb <= 1.1m && pivotOKShort)
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
        {
            var lst = new List<Candidate>();
            if (bars is null || bars.Count < 80) return lst; // need enough for pre-squeeze, TF2 agg, etc.

            var cfg = S3RuntimeConfig.Instance;
            var last = bars[^1];

            // News block
            if (InNewsWindow(last.Start, cfg.NewsOnMinutes, cfg.NewsBlockBeforeMin, cfg.NewsBlockAfterMin)) return lst;

            // Volume gate
            if (last.Volume < cfg.MinVolume) return lst;

            // Spread gate (optional provider)
            var spread = ExternalSpreadTicks?.Invoke(symbol);
            if (spread.HasValue)
            {
                var spreadCap = symbol.Contains("NQ", StringComparison.OrdinalIgnoreCase) ? Math.Max(cfg.MaxSpreadTicks, 3) : cfg.MaxSpreadTicks;
                if (spread.Value > spreadCap) return lst;
            }

            // VolZ regime gate if available
            if (env.volz.HasValue)
            {
                if (env.volz.Value < cfg.VolZMin || env.volz.Value > cfg.VolZMax) return lst;
            }

            // Indicator bases on TF1 (1m)
            var closes = bars.Select(b => b.Close).ToArray();
            var emaKc = EMA(closes, cfg.KcEma);
            var mid = emaKc[^1];
            var sd = Stdev(closes, cfg.BbLen);
            var bbUp = mid + cfg.BbMult * sd;
            var bbDn = mid - cfg.BbMult * sd;
            var atr = ATR(bars, cfg.AtrLen);
            if (atr <= 0) return lst;

            var kcAtr = ATR(bars, cfg.KcAtrLen);
            var kcUp = mid + cfg.KcMult * kcAtr;
            var kcDn = mid - cfg.KcMult * kcAtr;

            // Width-rank based squeeze
            var widthRank = PercentRankOfWidth(bars, cfg.BbLen, cfg.PreSqueezeLookback, cfg.BbMult);
            var rankThresh = cfg.WidthRankEnter;
            if (cfg.HourlyRankAdapt)
            {
                rankThresh = AdaptedRankThreshold(symbol, last.Start, cfg.WidthRankEnter);
            }
            bool squeezeOnRank = widthRank <= rankThresh;

            // TTM squeeze
            bool squeezeOnTTM = (bbUp <= kcUp) && (bbDn >= kcDn);

            // Squeeze run length using TTM condition
            int squeezeRun = SqueezeRunLength(bars, cfg.BbLen, cfg.BbMult, cfg.KcEma, cfg.KcAtrLen, cfg.KcMult);

            // Width slope down and narrow-range cluster
            bool widthSlopeOk = WidthSlopeDown(bars, cfg.BbLen, cfg.BbMult, cfg.WidthSlopeDownBars, cfg.WidthSlopeTol);
            bool hasNrCluster = HasNarrowRangeCluster(bars, cfg.PreSqueezeLookback, cfg.NrClusterRatio, cfg.NrClusterMinBars);

            bool squeezeArmed = (squeezeOnTTM || squeezeOnRank) && squeezeRun >= cfg.MinSqueezeBars && hasNrCluster && widthSlopeOk;
            if (!squeezeArmed) return lst;

            // TF2 slope bias via 5m aggregation
            var bars5 = Aggregate(bars, 5);
            var ema20_5 = EMA(bars5.Select(b => b.Close).ToArray(), 20);
            var slope5 = LinearSlope(ema20_5, Math.Min(5, ema20_5.Length));
            bool biasUp = slope5 > cfg.MinSlopeTf2;
            bool biasDn = slope5 < -cfg.MinSlopeTf2;

            // Compression box with segment index
            var (boxHi, boxLo, segStartIdx) = CompressionBoxWithIndex(bars, cfg.BbLen, cfg.BbMult, cfg.KcEma, cfg.KcAtrLen, cfg.KcMult, cfg.PreSqueezeLookback);
            var boxW = Math.Max(1e-9m, boxHi - boxLo);

            // Segment state and cooldown
            var st = _segState.GetOrAdd(symbol, _ => new SegmentState());
            st.UpdateIfNewSegment(segStartIdx, last.Start, boxHi, boxLo);
            if (cfg.OnePerSegment && st.FilledThisSegment) return lst;
            if (st.OnCooldown(last.Start, cfg.SegmentCooldownMinutes)) return lst;

            // Opening Range guard (OR guard; first N minutes post 09:30 local)
            if (cfg.OrGuardEnabled)
            {
                var orStart = AnchorToday(last.Start, new TimeSpan(9, 30, 0));
                var orEnd = orStart.AddMinutes(cfg.OrMinutes);
                var minsSinceOpen = (int)(last.Start - orStart).TotalMinutes;
                if (minsSinceOpen >= 0 && minsSinceOpen <= cfg.OrMinutes)
                {
                    var (orh, orl) = InitialBalance(bars, orStart, orEnd);
                    if (cfg.OrAvoidBreakInto.Equals("opposite", StringComparison.OrdinalIgnoreCase))
                    {
                        if (last.Close < orl && bbUp > orl) return lst;
                        if (last.Close > orh && bbDn < orh) return lst;
                    }
                }
            }

            // Initial Balance (IB) guard
            if (cfg.IbGuardEnabled)
            {
                var ibStart = AnchorToday(last.Start, new TimeSpan(9, 30, 0));
                var ibEnd = last.Start.Date.AddMinutes(cfg.IbEndMinute);
                var (ibh, ibl) = InitialBalance(bars, ibStart, ibEnd);
                if (ibh > 0 && ibl > 0 && cfg.IbAvoidBreakInto.Equals("opposite", StringComparison.OrdinalIgnoreCase))
                {
                    if (last.Close < ibl && bbUp > ibl) return lst;
                    if (last.Close > ibh && bbDn < ibh) return lst;
                }
            }

            // Roll guard tighten (tighter rank requirement around roll week)
            if (cfg.RollEnabled && IsWithinRollWindow(last.Start.Date, cfg.RollDaysBefore, cfg.RollDaysAfter))
            {
                rankThresh = Math.Max(0.05m, rankThresh - cfg.RollRankTighten);
                if (!(widthRank <= rankThresh)) return lst; // re-check if adapting tightened threshold disqualifies
            }

            // RS filter (optional, needs ExternalGetBars)
            if (cfg.RsEnabled && ExternalGetBars != null)
            {
                var peer = TryGetPeer(symbol, cfg); // ES<->NQ
                if (!string.IsNullOrEmpty(peer))
                {
                    var peerBars = ExternalGetBars(peer!);
                    if (peerBars != null && peerBars.Count >= cfg.RsWindowBars + 2)
                    {
                        var rs = RelativeStrength(bars, peerBars, cfg.RsWindowBars);
                        if (cfg.RsDirectionalOnly)
                        {
                            if (biasUp && rs < cfg.RsThreshold) return lst;
                            if (biasDn && rs > -cfg.RsThreshold) return lst;
                        }
                        else if (Math.Abs(rs) < cfg.RsThreshold) return lst;
                    }
                }
            }

            // Impulse
            var impulseScore = ImpulseScore(bars, 3) / boxW;
            if (impulseScore < cfg.ImpulseScoreMin) return lst;

            // Buffers
            decimal buf = cfg.ConfirmBreakAtrMult * atr;
            if ((biasUp && last.Close < boxHi) || (biasDn && last.Close > boxLo)) buf *= cfg.ContraBufferMult;
            if (InOvernightWindow(last.Start.TimeOfDay)) buf += cfg.OvernightBufferAdd * atr;

            // Break bar quality
            if (!BreakBarQualityOk(last, cfg.BreakQ_MinClosePos, cfg.BreakQ_MaxOppWick, out var barq)) return lst;

            // Anchored VWAP from segment start
            var segAnchor = bars[Math.Max(0, segStartIdx)].Start;
            var (segVwap, _, _) = AnchoredVwap(bars, segAnchor);

            // Determine session and attempt cap
            var session = InOvernightWindow(last.Start.TimeOfDay) ? "ON" : "RTH";
            int cap = session == "ON" ? cfg.AttemptCapOvernight : cfg.AttemptCapRTH;

            // Breakout detection (allow kc bands if biased)
            bool brokeUp = last.High > boxHi + buf || (biasUp && last.Close > kcUp + buf);
            bool brokeDn = last.Low  < boxLo - buf || (biasDn && last.Close < kcDn - buf);

            // Early invalidate tick
            st.TickInvalidate(bars, mid, boxHi, boxLo, cfg.EarlyInvalidateBars);
            if (st.IsInvalid) return lst;

            var px = last.Close;
            var tick = ExternalTickSize?.Invoke(symbol) ?? GuessTickSize(symbol);

            // LONG side
            if (brokeUp && CanAttempt(symbol, session, Side.BUY, cap))
            {
                decimal entry;
                if (cfg.EntryMode.Equals("retest", StringComparison.OrdinalIgnoreCase))
                {
                    var backoff = cfg.RetestBackoffTicks * tick;
                    entry = Math.Max(px, boxHi - backoff);
                }
                else // breakstop
                {
                    entry = Math.Max(px, Math.Max(last.High + tick, boxHi + buf));
                }
                if (entry <= 0) return lst;
                if (!HoldsAroundVwap(bars, segVwap, true, 2)) return lst;

                var isl = Math.Min(boxLo - cfg.StopAtrMult * atr, SwingLow(bars, 5));
                var r = entry - isl;
                if (r <= 0) return lst;
                var (t1, _) = Targets(cfg, symbol, last.Start, r, entry, boxW, true);
                add_cand(lst, "S3", symbol, "BUY", entry, isl, t1, env, risk);
                st.MarkFilled(segStartIdx, Side.BUY, last.Start);
                RegisterAttempt(symbol, session, Side.BUY);
            }

            // SHORT side
            if (brokeDn && CanAttempt(symbol, session, Side.SELL, cap))
            {
                decimal entry;
                if (cfg.EntryMode.Equals("retest", StringComparison.OrdinalIgnoreCase))
                {
                    var backoff = cfg.RetestBackoffTicks * tick;
                    entry = Math.Min(px, boxLo + backoff);
                }
                else
                {
                    entry = Math.Min(px, Math.Min(last.Low - tick, boxLo - buf));
                }
                if (entry <= 0) return lst;
                if (!HoldsAroundVwap(bars, segVwap, false, 2)) return lst;

                var ish = Math.Max(boxHi + cfg.StopAtrMult * atr, SwingHigh(bars, 5));
                var r = ish - entry;
                if (r <= 0) return lst;
                var (t1, _) = Targets(cfg, symbol, last.Start, r, entry, boxW, false);
                add_cand(lst, "S3", symbol, "SELL", entry, ish, t1, env, risk,
                    tag: $"rank={widthRank:F2} run={squeezeRun} nrOK={hasNrCluster} slope5={slope5:F3} barq={barq:F2}");
                st.MarkFilled(segStartIdx, Side.Sell, last.Start);
                RegisterAttempt(symbol, session, Side.Sell);
            }

            return lst;
        }

        // ---- S3 helpers (local, full) ----
        private static decimal[] EMA(decimal[] x, int n)
        {
            if (x.Length == 0) return Array.Empty<decimal>();
            var a = 2m / (n + 1);
            var y = new decimal[x.Length];
            y[0] = x[0];
            for (int i = 1; i < x.Length; i++) y[i] = a * x[i] + (1 - a) * y[i - 1];
            return y;
        }
        private static decimal Stdev(decimal[] x, int n)
        {
            if (x.Length < n) return 0m;
            decimal sum = 0m, sum2 = 0m;
            for (int i = x.Length - n; i < x.Length; i++) { sum += x[i]; sum2 += x[i] * x[i]; }
            var mean = sum / n; var varv = (sum2 / n) - mean * mean;
            return varv <= 0 ? 0m : (decimal)Math.Sqrt((double)varv);
        }
        private static decimal ATR(IList<Bar> b, int n)
        {
            if (b.Count < n + 1) return 0m;
            decimal atr = 0m; int start = b.Count - n;
            for (int i = start; i < b.Count; i++)
            {
                var c = b[i]; var p = b[i - 1];
                var tr = Math.Max((float)(c.High - c.Low), Math.Max((float)Math.Abs(c.High - p.Close), (float)Math.Abs(c.Low - p.Close)));
                atr = i == start ? tr : (atr * (n - 1) + (decimal)tr) / n;
            }
            return atr;
        }
        private static decimal PercentRankOfWidth(IList<Bar> bars, int bbLen, int lookback, decimal bbMult)
        {
            int start = Math.Max(bbLen + 2, bars.Count - lookback);
            var closes = bars.Select(b => b.Close).ToArray();
            var widths = new List<decimal>();
            for (int i = start; i < bars.Count; i++)
            {
                var arr = closes.Take(i + 1).ToArray();
                var ema = EMA(arr, bbLen);
                var mid = ema[^1];
                var sd = Stdev(arr, bbLen);
                var up = mid + bbMult * sd; var dn = mid - bbMult * sd;
                var w = (up - dn) / Math.Max(1e-9m, Math.Abs(mid));
                widths.Add(w);
            }
            if (widths.Count == 0) return 1m;
            var last = widths[^1];
            var cnt = widths.Count(w => w <= last);
            return (decimal)cnt / widths.Count;
        }
        private static bool WidthSlopeDown(IList<Bar> bars, int bbLen, decimal bbMult, int look, decimal tol)
        {
            var closes = bars.Select(b => b.Close).ToArray();
            var list = new List<decimal>();
            for (int i = Math.Max(bbLen, closes.Length - (look + 6)); i < closes.Length; i++)
            {
                var arr = closes.Take(i + 1).ToArray();
                var ema = EMA(arr, bbLen);
                var mid = ema[^1]; var sd = Stdev(arr, bbLen);
                var up = mid + bbMult * sd; var dn = mid - bbMult * sd;
                list.Add((up - dn) / Math.Max(1e-9m, Math.Abs(mid)));
            }
            if (list.Count < look + 2) return false;
            int desc = 0;
            for (int i = list.Count - look + 1; i < list.Count; i++) if (list[i] <= list[i - 1] + tol) desc++;
            return desc >= look - 1;
        }
        private static bool HasNarrowRangeCluster(IList<Bar> bars, int lookback, decimal ratio, int minBars)
        {
            int s = Math.Max(1, bars.Count - lookback);
            var trs = new List<decimal>();
            for (int i = s; i < bars.Count; i++)
            {
                var c = bars[i]; var p = bars[Math.Max(i - 1, 0)];
                trs.Add((decimal)Math.Max((float)(c.High - c.Low), Math.Max((float)Math.Abs(c.High - p.Close), (float)Math.Abs(c.Low - p.Close))));
            }
            if (trs.Count < minBars) return false;
            var sorted = trs.OrderBy(x => x).ToList();
            var med = sorted[sorted.Count / 2];
            int cnt = 0;
            for (int i = trs.Count - 1; i >= 0 && cnt < minBars; i--)
            {
                if (trs[i] <= ratio * med) cnt++; else break;
            }
            return cnt >= minBars;
        }
        private static int SqueezeRunLength(IList<Bar> bars, int bbLen, decimal bbMult, int kcEma, int kcAtrLen, decimal kcMult)
        {
            int run = 0;
            var closes = bars.Select(b => b.Close).ToArray();
            for (int i = bars.Count - 1; i >= Math.Max(0, bars.Count - 60); i--)
            {
                var arr = closes.Take(i + 1).ToArray();
                var ema = EMA(arr, kcEma); var mid = ema[^1];
                var sd = Stdev(arr, bbLen); var bbUp = mid + bbMult * sd; var bbDn = mid - bbMult * sd;
                var atr = ATR(bars.Take(i + 1).ToList(), kcAtrLen);
                var kcUp = mid + kcMult * atr; var kcDn = mid - kcMult * atr;
                bool squeeze = bbUp <= kcUp && bbDn >= kcDn;
                if (squeeze) run++; else break;
            }
            return run;
        }
        private static (decimal hi, decimal lo, int startIdx) CompressionBoxWithIndex(IList<Bar> bars, int bbLen, decimal bbMult, int kcEma, int kcAtrLen, decimal kcMult, int maxLookback)
        {
            int end = bars.Count - 1; int start = end;
            var closes = bars.Select(b => b.Close).ToArray();
            for (int i = end; i >= Math.Max(0, end - maxLookback); i--)
            {
                var arr = closes.Take(i + 1).ToArray();
                var ema = EMA(arr, kcEma); var mid = ema[^1];
                var sd = Stdev(arr, bbLen); var bbUp = mid + bbMult * sd; var bbDn = mid - bbMult * sd;
                var atr = ATR(bars.Take(i + 1).ToList(), kcAtrLen); var kcUp = mid + kcMult * atr; var kcLo = mid - kcMult * atr;
                bool squeeze = bbUp <= kcUp && bbDn >= kcLo;
                if (squeeze) start = i; else if (start != end) break;
            }
            decimal hi = decimal.MinValue, lo = decimal.MaxValue;
            for (int i = start; i <= end; i++) { hi = Math.Max(hi, bars[i].High); lo = Math.Min(lo, bars[i].Low); }
            if (hi == decimal.MinValue || lo == decimal.MaxValue)
            {
                for (int i = Math.Max(0, end - 60); i <= end; i++) { hi = Math.Max(hi, bars[i].High); lo = Math.Min(lo, bars[i].Low); }
            }
            return (hi, lo, start);
        }
        private static List<Bar> Aggregate(IList<Bar> bars, int n)
        {
            if (bars == null || bars.Count == 0 || n <= 1) return bars.ToList();
            var res = new List<Bar>();
            int i = 0;
            while (i < bars.Count)
            {
                int j = Math.Min(i + n, bars.Count) - 1;
                var o = bars[i]; var c = bars[j];
                var high = bars.Skip(i).Take(j - i + 1).Max(b => b.High);
                var low = bars.Skip(i).Take(j - i + 1).Min(b => b.Low);
                var vol = bars.Skip(i).Take(j - i + 1).Sum(b => b.Volume);
                res.Add(new Bar { Start = o.Start, Ts = o.Ts, Symbol = o.Symbol, Open = o.Open, High = high, Low = low, Close = c.Close, Volume = vol });
                i += n;
            }
            return res;
        }
        private static decimal LinearSlope(decimal[] arr, int n)
        {
            if (arr.Length < n + 1) return 0m;
            return (arr[^1] - arr[^1 - n]) / n;
        }
        private static decimal RelativeStrength(IList<Bar> a, IList<Bar> b, int look)
        {
            int sa = Math.Max(0, a.Count - look);
            int sb = Math.Max(0, b.Count - look);
            var ra = a[sa].Close == 0 ? 0m : (a[^1].Close - a[sa].Close) / a[sa].Close;
            var rb = b[sb].Close == 0 ? 0m : (b[^1].Close - b[sb].Close) / b[sb].Close;
            return ra - rb;
        }
        private static (decimal vwap, decimal wvar, decimal wvol) AnchoredVwap(IList<Bar> bars, DateTime anchorLocal)
        {
            decimal wv = 0m, vol = 0m; int idx0 = 0;
            for (int i = 0; i < bars.Count; i++) { if (bars[i].Start >= anchorLocal) { idx0 = i; break; } }
            for (int i = idx0; i < bars.Count; i++) { var b = bars[i]; var tp = (b.High + b.Low + b.Close) / 3m; var v = Math.Max(0, b.Volume); wv += tp * v; vol += v; }
            if (vol <= 0) return (0m, 0m, 0m);
            var vwap = wv / vol; decimal num = 0m;
            for (int i = idx0; i < bars.Count; i++) { var b = bars[i]; var tp = (b.High + b.Low + b.Close) / 3m; var v = Math.Max(0, b.Volume); var d = tp - vwap; num += d * d * v; }
            var wvar = num / vol; return (vwap, wvar, vol);
        }
        private static bool HoldsAroundVwap(IList<Bar> b, decimal vwap, bool above, int need)
        { int ok = 0; for (int i = b.Count - need; i < b.Count; i++) { var c = b[i].Close; if (above ? c >= vwap : c <= vwap) ok++; } return ok >= need; }
        private static decimal SwingLow(IList<Bar> b, int look) => Enumerable.Range(1, Math.Min(look, b.Count)).Select(i => b[^i].Low).Min();
        private static decimal SwingHigh(IList<Bar> b, int look) => Enumerable.Range(1, Math.Min(look, b.Count)).Select(i => b[^i].High).Max();
        private static bool BreakBarQualityOk(Bar c, decimal minClosePos, decimal maxOppWick, out decimal score)
        {
            var rng = Math.Max(1e-9m, c.High - c.Low);
            var closePos = (c.Close - c.Low) / rng;
            var oppWick = c.Close >= c.Open ? (c.High - c.Close) / rng : (c.Open - c.Low) / rng;
            bool okLong = c.Close >= c.Open && closePos >= minClosePos && oppWick <= maxOppWick;
            bool okShort = c.Close <= c.Open && (1m - closePos) >= minClosePos && oppWick <= maxOppWick;
            score = closePos - oppWick; return okLong || okShort;
        }
        private static decimal ImpulseScore(IList<Bar> bars, int look)
        {
            if (bars.Count < look) return 0m; decimal hi = decimal.MinValue, lo = decimal.MaxValue;
            for (int i = bars.Count - look; i < bars.Count; i++) { hi = Math.Max(hi, bars[i].High); lo = Math.Min(lo, bars[i].Low); }
            return hi - lo;
        }
        private static bool InNewsWindow(DateTime nowLocal, int[] onMinutes, int beforeMin, int afterMin)
        {
            int m = nowLocal.Minute;
            foreach (var x in onMinutes)
            {
                if (m <= x && (x - m) <= beforeMin) return true;
                if (m >= x && (m - x) <= afterMin) return true;
            }
            return false;
        }
        private static bool InOvernightWindow(TimeSpan t) => t >= OvernightWinStart && t <= OvernightWinEnd;
        private static string? TryGetPeer(string sym, S3RuntimeConfig cfg)
        {
            if (cfg.Peers.TryGetValue(sym, out var p) && !string.IsNullOrWhiteSpace(p)) return p;
            // simple symmetric map
            if (sym.Equals("ES", StringComparison.OrdinalIgnoreCase)) return cfg.Peers.TryGetValue("ES", out var v) ? v : "NQ";
            if (sym.Equals("NQ", StringComparison.OrdinalIgnoreCase)) return cfg.Peers.TryGetValue("NQ", out var v2) ? v2 : "ES";
            return null;
        }
        private static (decimal t1, decimal t2) Targets(S3RuntimeConfig cfg, string sym, DateTime nowLocal, decimal r, decimal entry, decimal boxW, bool isLong)
        {
            if (cfg.TargetsMode.Equals("expansion", StringComparison.OrdinalIgnoreCase))
            {
                var exp = 1.6m; // fallback expected expansion factor of box
                var move = exp * boxW;
                var t1 = isLong ? entry + Math.Max(cfg.TargetR1 * r, 0.5m * move) : entry - Math.Max(cfg.TargetR1 * r, 0.5m * move);
                var t2 = isLong ? entry + Math.Max(cfg.TargetR2 * r, move) : entry - Math.Max(cfg.TargetR2 * r, move);
                return (t1, t2);
            }
            else
            {
                return (isLong ? entry + cfg.TargetR1 * r : entry - cfg.TargetR1 * r,
                        isLong ? entry + cfg.TargetR2 * r : entry - cfg.TargetR2 * r);
            }
        }
        private static decimal GuessTickSize(string sym)
        { try { return InstrumentMeta.Tick(sym); } catch { return 0.25m; } }
        private static bool CanAttempt(string sym, string sess, Side side, int cap)
        {
            var key = (sym, DateOnly.FromDateTime(DateTime.Now), sess, side);
            var n = _attempts.AddOrUpdate(key, 1, (_, v) => v + 1);
            return n <= cap;
        }
        private static void RegisterAttempt(string sym, string sess, Side side) { /* account for metrics in future */ }
        private static bool IsWithinRollWindow(DateTime today, int daysBefore, int daysAfter)
        {
            int m = today.Month; if (m % 3 != 0) return false; // Mar/Jun/Sep/Dec only
            int firstThu = FirstWeekdayOfMonth(today.Year, m, DayOfWeek.Thursday);
            int secondThu = firstThu + 7;
            var center = new DateTime(today.Year, m, secondThu);
            return (today - center).TotalDays <= daysBefore && (center - today).TotalDays <= daysAfter;
        }
        private static int FirstWeekdayOfMonth(int y, int m, DayOfWeek dow)
        { var d = new DateTime(y, m, 1); while (d.DayOfWeek != dow) d = d.AddDays(1); return d.Day; }
        private static decimal AdaptedRankThreshold(string sym, DateTime local, decimal baseRank)
        {
            // Simple hourly adaptation: slightly tighter just after RTH open
            var minuteOfDay = local.Hour * 60 + local.Minute;
            // Tighten by 0.02 between 09:30-10:30, otherwise base
            if (minuteOfDay >= 570 && minuteOfDay <= 630) return Math.Max(0.05m, baseRank - 0.02m);
            return baseRank;
        }

        private sealed class SegmentState
        {
            public int SegmentId = -1;
            public DateTime SegmentStartLocal = DateTime.MinValue;
            public decimal BoxHigh;
            public decimal BoxLow;
            public bool FilledThisSegment;
            public DateTime LastFillLocal = DateTime.MinValue;
            public bool IsInvalid;
            public int LastBreakBarIndex = -1;
            public Side LastSide = Side.BUY;

            public void UpdateIfNewSegment(int segId, DateTime nowLocal, decimal hi, decimal lo)
            {
                if (segId != SegmentId)
                {
                    SegmentId = segId;
                    SegmentStartLocal = nowLocal;
                    BoxHigh = hi; BoxLow = lo;
                    FilledThisSegment = false; IsInvalid = false; LastBreakBarIndex = -1;
                }
            }
            public void MarkFilled(int segId, Side side, DateTime nowLocal)
            {
                if (segId == SegmentId) { FilledThisSegment = true; LastFillLocal = nowLocal; LastSide = side; }
            }
            public bool OnCooldown(DateTime nowLocal, int minutes) => (nowLocal - LastFillLocal).TotalMinutes < minutes;
            public void TickInvalidate(IList<Bar> bars, decimal kcMid, decimal boxHigh, decimal boxLow, int earlyInvalidateBars)
            {
                if (bars == null || bars.Count == 0) return;
                bool backInside = bars[^1].Close < boxHigh && bars[^1].Close > boxLow;
                bool midFlipUp = kcMid > bars[^1].Close; bool midFlipDn = kcMid < bars[^1].Close;
                if (backInside && (midFlipUp || midFlipDn)) { IsInvalid = true; LastBreakBarIndex = bars.Count - 1; }
                if (IsInvalid && (bars.Count - 1 - LastBreakBarIndex) > earlyInvalidateBars) IsInvalid = false;
            }
            public void ResetBreak() { LastBreakBarIndex = -1; }
        }

        private sealed class S3RuntimeConfig
        {
            public int BbLen { get; init; } = 20;
            public decimal BbMult { get; init; } = 2.0m;
            public int KcEma { get; init; } = 20;
            public int KcAtrLen { get; init; } = 20;
            public decimal KcMult { get; init; } = 1.5m;
            public int AtrLen { get; init; } = 14;

            public int MinSqueezeBars { get; init; } = 6;
            public int PreSqueezeLookback { get; init; } = 60;
            public decimal WidthRankEnter { get; init; } = 0.15m;
            public bool HourlyRankAdapt { get; init; } = true;
            public decimal NrClusterRatio { get; init; } = 0.60m;
            public int NrClusterMinBars { get; init; } = 5;
            public int WidthSlopeDownBars { get; init; } = 8;
            public decimal WidthSlopeTol { get; init; } = 0.0m;

            public decimal ConfirmBreakAtrMult { get; init; } = 0.15m;
            public decimal ContraBufferMult { get; init; } = 1.5m;
            public decimal OvernightBufferAdd { get; init; } = 0.05m;
            public decimal StopAtrMult { get; init; } = 1.1m;
            public decimal TargetR1 { get; init; } = 1.2m;
            public decimal TargetR2 { get; init; } = 2.2m;

            public string EntryMode { get; init; } = "retest";
            public int RetestBars { get; init; } = 5;
            public int RetestBackoffTicks { get; init; } = 1;

            public bool OrGuardEnabled { get; init; } = true;
            public int OrMinutes { get; init; } = 10;
            public string OrAvoidBreakInto { get; init; } = "opposite";

            public bool IbGuardEnabled { get; init; } = true;
            public int IbEndMinute { get; init; } = 630;
            public string IbAvoidBreakInto { get; init; } = "opposite";

            public bool RsEnabled { get; init; } = true;
            public int RsWindowBars { get; init; } = 60;
            public decimal RsThreshold { get; init; } = 0.10m;
            public bool RsDirectionalOnly { get; init; } = true;
            public Dictionary<string,string> Peers { get; init; } = new(StringComparer.OrdinalIgnoreCase);

            public bool RollEnabled { get; init; } = true;
            public int RollDaysBefore { get; init; } = 2;
            public int RollDaysAfter { get; init; } = 1;
            public decimal RollRankTighten { get; init; } = 0.02m;

            public decimal BreakQ_MinClosePos { get; init; } = 0.65m;
            public decimal BreakQ_MaxOppWick { get; init; } = 0.35m;

            public int ValidityBars { get; init; } = 3;
            public int CooldownBars { get; init; } = 5;
            public int MaxBarsInTrade { get; init; } = 45;

            public decimal TrailAtrMult { get; init; } = 1.0m;

            public int MinVolume { get; init; } = 3000;
            public int MaxSpreadTicks { get; init; } = 2;
            public int NewsBlockBeforeMin { get; init; } = 2;
            public int NewsBlockAfterMin { get; init; } = 3;
            public int[] NewsOnMinutes { get; init; } = new[] { 0, 30 };

            public int AttemptCapRTH { get; init; } = 2;
            public int AttemptCapOvernight { get; init; } = 1;

            public bool OnePerSegment { get; init; } = true;
            public int SegmentCooldownMinutes { get; init; } = 15;
            public int EarlyInvalidateBars { get; init; } = 6;

            public decimal ImpulseScoreMin { get; init; } = 1.0m;
            public string TargetsMode { get; init; } = "expansion";
            public decimal ExpansionQuantile { get; init; } = 0.60m;
            public decimal GivebackAfterT1R { get; init; } = 0.20m;

            public decimal MinSlopeTf2 { get; init; } = 0.15m;
            public decimal VolZMin { get; init; } = -0.5m;
            public decimal VolZMax { get; init; } = 2.5m;

            private static S3RuntimeConfig? _instance;
            public static S3RuntimeConfig Instance => _instance ??= Load();

            private static S3RuntimeConfig Load()
            {
                try
                {
                    // 1) Allow override via env var
                    var envPath = Environment.GetEnvironmentVariable("S3_CONFIG_PATH");
                    var candidates = new List<string>();
                    if (!string.IsNullOrWhiteSpace(envPath)) candidates.Add(envPath!);

                    // 2) Probe from BaseDirectory and CurrentDirectory walking up for common repo layouts
                    string[] bases = new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() };
                    foreach (var b in bases.Distinct())
                    {
                        var dir = new DirectoryInfo(b);
                        for (int i = 0; i < 7 && dir != null; i++, dir = dir.Parent)
                        {
                            var c1 = Path.Combine(dir.FullName, "src", "BotCore", "Strategy", "S3-StrategyConfig.json");
                            var c2 = Path.Combine(dir.FullName, "BotCore", "Strategy", "S3-StrategyConfig.json");
                            var c3 = Path.Combine(dir.FullName, "S3-StrategyConfig.json");
                            candidates.Add(c1); candidates.Add(c2); candidates.Add(c3);
                        }
                    }

                    // 3) Try previously used fallbacks for dev
                    candidates.Add(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "BotCore", "Strategy", "S3-StrategyConfig.json"));
                    candidates.Add(Path.Combine(AppContext.BaseDirectory, "src", "BotCore", "Strategy", "S3-StrategyConfig.json"));
                    candidates.Add("src\\BotCore\\Strategy\\S3-StrategyConfig.json");

                    var path = candidates.FirstOrDefault(File.Exists);
                    if (string.IsNullOrEmpty(path)) return new S3RuntimeConfig();

                    var json = File.ReadAllText(path);
                    using var doc = JsonDocument.Parse(json, new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });
                    var root = doc.RootElement;
                    if (root.TryGetProperty("Strategies", out var arr) && arr.ValueKind == JsonValueKind.Array && arr.GetArrayLength() > 0)
                    {
                        var s3 = arr[0];
                        // Nested helpers
                        static int[] GetIntArray(JsonElement e)
                        {
                            if (e.ValueKind != JsonValueKind.Array) return Array.Empty<int>();
                            var list = new List<int>();
                            foreach (var it in e.EnumerateArray()) if (it.TryGetInt32(out var v)) list.Add(v);
                            return list.ToArray();
                        }
                        var peers = new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
                        if (s3.TryGetProperty("rs_filter", out var rs) && rs.ValueKind == JsonValueKind.Object)
                        {
                            if (rs.TryGetProperty("peers", out var pe) && pe.ValueKind == JsonValueKind.Object)
                            {
                                foreach (var kv in pe.EnumerateObject()) peers[kv.Name] = kv.Value.GetString() ?? "";
                            }
                        }
                        var newsMins = new[] { 0, 30 };
                        if (s3.TryGetProperty("news_block", out var nb) && nb.ValueKind == JsonValueKind.Object)
                        {
                            if (nb.TryGetProperty("on_minutes", out var onm)) newsMins = GetIntArray(onm);
                        }
                        var cfg = new S3RuntimeConfig
                        {
                            BbLen = s3.TryGetProperty("bb_len", out var v1) && v1.TryGetInt32(out var i1) ? i1 : 20,
                            BbMult = s3.TryGetProperty("bb_mult", out var v2) && v2.TryGetDecimal(out var d2) ? d2 : 2.0m,
                            KcEma = s3.TryGetProperty("kc_ema", out var v3) && v3.TryGetInt32(out var i3) ? i3 : 20,
                            KcAtrLen = s3.TryGetProperty("kc_atr_len", out var v4) && v4.TryGetInt32(out var i4) ? i4 : 20,
                            KcMult = s3.TryGetProperty("kc_mult", out var v5) && v5.TryGetDecimal(out var d5) ? d5 : 1.5m,
                            AtrLen = s3.TryGetProperty("atr_len", out var v6) && v6.TryGetInt32(out var i6) ? i6 : 14,
                            MinSqueezeBars = s3.TryGetProperty("min_squeeze_bars", out var v7) && v7.TryGetInt32(out var i7) ? i7 : 6,
                            PreSqueezeLookback = s3.TryGetProperty("pre_squeeze_lookback", out var v8) && v8.TryGetInt32(out var i8) ? i8 : 60,
                            ConfirmBreakAtrMult = s3.TryGetProperty("confirm_break_mult", out var v9) && v9.TryGetDecimal(out var d9) ? d9 : 0.15m,
                            StopAtrMult = s3.TryGetProperty("stop_atr_mult", out var v10) && v10.TryGetDecimal(out var d10) ? d10 : 1.1m,
                            TargetR1 = s3.TryGetProperty("target_r1", out var v11) && v11.TryGetDecimal(out var d11) ? d11 : 1.2m,
                            TargetR2 = s3.TryGetProperty("target_r2", out var v11b) && v11b.TryGetDecimal(out var d11b) ? d11b : 2.2m,
                            MinVolume = s3.TryGetProperty("min_volume", out var v12) && v12.TryGetInt32(out var i12) ? i12 : 3000,
                            MaxSpreadTicks = s3.TryGetProperty("max_spread_ticks", out var v13) && v13.TryGetInt32(out var i13) ? i13 : 2,
                            AttemptCapRTH = s3.TryGetProperty("attempt_cap", out var ac) && ac.TryGetProperty("RTH", out var rth) && rth.TryGetInt32(out var iac1) ? iac1 : 2,
                            AttemptCapOvernight = s3.TryGetProperty("attempt_cap", out var ac2) && ac2.TryGetProperty("overnight", out var on) && on.TryGetInt32(out var iac2) ? iac2 : 1,
                            WidthRankEnter = s3.TryGetProperty("width_rank_enter", out var wr) && wr.TryGetDecimal(out var dwr) ? dwr : 0.15m,
                            HourlyRankAdapt = s3.TryGetProperty("hourly_rank_adapt", out var hra) && hra.ValueKind == JsonValueKind.True || (hra.ValueKind == JsonValueKind.False ? false : true),
                            NrClusterRatio = s3.TryGetProperty("nr_cluster_ratio", out var nrr) && nrr.TryGetDecimal(out var dnrr) ? dnrr : 0.60m,
                            NrClusterMinBars = s3.TryGetProperty("nr_cluster_min_bars", out var nrb) && nrb.TryGetInt32(out var inrb) ? inrb : 5,
                            WidthSlopeDownBars = s3.TryGetProperty("width_slope_down_bars", out var wsb) && wsb.TryGetInt32(out var iwsb) ? iwsb : 8,
                            WidthSlopeTol = s3.TryGetProperty("width_slope_tol", out var wst) && wst.TryGetDecimal(out var dwst) ? dwst : 0.0m,
                            ContraBufferMult = s3.TryGetProperty("contra_buffer_mult", out var cb) && cb.TryGetDecimal(out var dcb) ? dcb : 1.5m,
                            OvernightBufferAdd = s3.TryGetProperty("overnight_buffer_add", out var oba) && oba.TryGetDecimal(out var doba) ? doba : 0.05m,
                            EntryMode = s3.TryGetProperty("entry_mode", out var em) && em.ValueKind == JsonValueKind.String ? em.GetString() ?? "retest" : "retest",
                            RetestBackoffTicks = s3.TryGetProperty("retest_backoff_ticks", out var rbt) && rbt.TryGetInt32(out var irbt) ? irbt : 1,
                            OrGuardEnabled = s3.TryGetProperty("or_guard", out var org) && org.TryGetProperty("enabled", out var oge) && oge.ValueKind == JsonValueKind.True,
                            OrMinutes = s3.TryGetProperty("or_guard", out var org2) && org2.TryGetProperty("minutes", out var ogm) && ogm.TryGetInt32(out var iogm) ? iogm : 10,
                            OrAvoidBreakInto = s3.TryGetProperty("or_guard", out var org3) && org3.TryGetProperty("avoid_break_into", out var abi) && abi.ValueKind == JsonValueKind.String ? abi.GetString() ?? "opposite" : "opposite",
                            RsEnabled = s3.TryGetProperty("rs_filter", out var rs) && rs.TryGetProperty("enabled", out var rse) && rse.ValueKind == JsonValueKind.True,
                            RsWindowBars = s3.TryGetProperty("rs_filter", out var rs2) && rs2.TryGetProperty("window_bars", out var rwb) && rwb.TryGetInt32(out var irwb) ? irwb : 60,
                            RsThreshold = s3.TryGetProperty("rs_filter", out var rs3) && rs3.TryGetProperty("threshold", out var rst) && rst.TryGetDecimal(out var drst) ? drst : 0.10m,
                            RsDirectionalOnly = s3.TryGetProperty("rs_filter", out var rs4) && rs4.TryGetProperty("directional_only", out var rdo) && rdo.ValueKind == JsonValueKind.True,
                            Peers = peers,
                            RollEnabled = s3.TryGetProperty("roll_guard", out var rg) && rg.TryGetProperty("enabled", out var rge) && rge.ValueKind == JsonValueKind.True,
                            RollDaysBefore = s3.TryGetProperty("roll_guard", out var rg2) && rg2.TryGetProperty("days_before", out var rdb) && rdb.TryGetInt32(out var irdb) ? irdb : 2,
                            RollDaysAfter = s3.TryGetProperty("roll_guard", out var rg3) && rg3.TryGetProperty("days_after", out var rda) && rda.TryGetInt32(out var irda) ? irda : 1,
                            RollRankTighten = s3.TryGetProperty("roll_guard", out var rg4) && rg4.TryGetProperty("rank_tighten", out var rrt) && rrt.TryGetDecimal(out var drrt) ? drrt : 0.02m,
                            BreakQ_MinClosePos = s3.TryGetProperty("break_bar_quality", out var bbq) && bbq.TryGetProperty("min_close_pos", out var mcp) && mcp.TryGetDecimal(out var dmcp) ? dmcp : 0.65m,
                            BreakQ_MaxOppWick = s3.TryGetProperty("break_bar_quality", out var bbq2) && bbq2.TryGetProperty("max_opp_wick", out var mow) && mow.TryGetDecimal(out var dmow) ? dmow : 0.35m,
                            ValidityBars = s3.TryGetProperty("validity_bars", out var vb) && vb.TryGetInt32(out var ivb) ? ivb : 3,
                            CooldownBars = s3.TryGetProperty("cooldown_bars", out var cb2) && cb2.TryGetInt32(out var icb) ? icb : 5,
                            MaxBarsInTrade = s3.TryGetProperty("max_bars_in_trade", out var mb) && mb.TryGetInt32(out var imb) ? imb : 45,
                            TrailAtrMult = s3.TryGetProperty("trail_atr_mult", out var tam) && tam.TryGetDecimal(out var dtam) ? dtam : 1.0m,
                            NewsBlockBeforeMin = s3.TryGetProperty("news_block", out var nb) && nb.TryGetProperty("minutes_before", out var nbB) && nbB.TryGetInt32(out var inbB) ? inbB : 2,
                            NewsBlockAfterMin = s3.TryGetProperty("news_block", out var nb2) && nb2.TryGetProperty("minutes_after", out var nbA) && nbA.TryGetInt32(out var inbA) ? inbA : 3,
                            NewsOnMinutes = newsMins,
                            OnePerSegment = s3.TryGetProperty("one_per_segment", out var ops) && ops.ValueKind == JsonValueKind.True,
                            SegmentCooldownMinutes = s3.TryGetProperty("segment_cooldown_minutes", out var scm) && scm.TryGetInt32(out var iscm) ? iscm : 15,
                            EarlyInvalidateBars = s3.TryGetProperty("early_invalidate_bars", out var eib) && eib.TryGetInt32(out var ieib) ? ieib : 6,
                            ImpulseScoreMin = s3.TryGetProperty("impulse_score_min", out var ism) && ism.TryGetDecimal(out var dism) ? dism : 1.0m,
                            TargetsMode = s3.TryGetProperty("targets_mode", out var tm) && tm.ValueKind == JsonValueKind.String ? tm.GetString() ?? "expansion" : "expansion",
                            ExpansionQuantile = s3.TryGetProperty("expansion_quantile", out var eq) && eq.TryGetDecimal(out var deq) ? deq : 0.60m,
                            GivebackAfterT1R = s3.TryGetProperty("giveback_after_t1_R", out var gb) && gb.TryGetDecimal(out var dgb) ? dgb : 0.20m,
                            MinSlopeTf2 = s3.TryGetProperty("min_slope_tf2", out var ms) && ms.TryGetDecimal(out var dms) ? dms : 0.15m,
                            VolZMin = s3.TryGetProperty("volz", out var vz) && vz.TryGetProperty("min", out var vmin) && vmin.TryGetDecimal(out var dvmin) ? dvmin : -0.5m,
                            VolZMax = s3.TryGetProperty("volz", out var vz2) && vz2.TryGetProperty("max", out var vmax) && vmax.TryGetDecimal(out var dvmax) ? dvmax : 2.5m,
                        };
                        return cfg;
                    }
                }
                catch { }
                return new S3RuntimeConfig();
            }
        }

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
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.7m)
            {
                var entry = bars[^1].Close;
                var stop = entry - env.atr.Value * 2.0m;
                var t1 = entry + env.atr.Value * 4.0m;
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
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.9m)
            {
                var entry = bars[^1].Close;
                var stop = entry + env.atr.Value * 3.0m;
                var t1 = entry - env.atr.Value * 6.0m;
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
            var pv   = InstrumentMeta.PointValue(symbol);
            var tick = InstrumentMeta.Tick(symbol);
            var dist = Math.Max(Math.Abs(entry - stop), tick); // ≥ 1 tick
            // Prefer equity-% aware sizing if configured; pass 0 equity to fallback to fixed RPT when not provided
            var sz = risk.ComputeSize(symbol, entry, stop, 0m);
            var qty = sz.Qty > 0 ? sz.Qty : (int)risk.size_for(risk.cfg.risk_per_trade, dist, pv);
            if (qty <= 0) return;

            var expR = rr_quality(entry, stop, t1);
            // Minimal, deterministic score: baseline from ExpR plus modest boost when volatility proxy is supportive
            var volBoost = env.volz.HasValue ? Math.Clamp(Math.Abs(env.volz.Value), 0m, 2m) * 0.25m : 0m;
            var score = expR + volBoost;

            var c = new Candidate
            {
                strategy_id = sid,
                symbol = symbol,
                side = sideTxt.Equals("BUY", StringComparison.OrdinalIgnoreCase) ? Side.BUY : Side.SELL,
                entry = entry,
                stop  = stop,
                t1    = t1,
                expR  = expR,
                qty   = qty,
                atr_ok = (env.atr ?? tick) >= tick,
                vol_z  = env.volz,
                Score  = score
            };
            lst.Add(c);
        }
    }
}
