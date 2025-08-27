// PURPOSE: AllStrategies as strategy engine for candidate generation.
using BotCore.Config;
using System;
using System.Collections.Generic;
using System.Linq;
using BotCore.Models;
using BotCore.Risk;

namespace BotCore.Strategy
{
    public static class AllStrategies
    {
        private static readonly HighWinRateProfile Profile = new();
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

        // Session VWAP computed from today's UTC midnight (approximation)
        private static (decimal vwap, decimal dist) SessionVWAP(IList<Bar> bars)
        {
            if (bars is null || bars.Count == 0) return (0m, 0m);
            var startUtc = DateTime.UtcNow.Date;
            decimal pv = 0m; decimal vol = 0m;
            foreach (var b in bars)
            {
                try
                {
                    var ts = DateTimeOffset.FromUnixTimeMilliseconds(b.Ts).UtcDateTime;
                    if (ts < startUtc) continue;
                }
                catch { }
                var tp = (b.High + b.Low + b.Close) / 3m;
                pv += tp * b.Volume;
                vol += b.Volume;
            }
            var vwap = vol > 0 ? pv / Math.Max(1m, vol) : bars[^1].Close;
            var dist = bars[^1].Close - vwap;
            return (vwap, dist);
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
            if (bars is null || bars.Count < 50) return lst;
            var atr = (env.atr.HasValue && env.atr.Value > 0m) ? env.atr.Value : AtrNoWarmup(bars, 14);
            if (atr <= 0) return lst;
            var (vwap, dist) = SessionVWAP(bars);
            var px = bars[^1].Close;
            if (dist >= 1.5m * atr)
            {
                // stretched above VWAP → fade short back to VWAP
                var entry = px;
                var stop = entry + 0.75m * atr;
                var t1 = vwap;
                add_cand(lst, "S2", symbol, "SELL", entry, stop, t1, env, risk);
            }
            else if (dist <= -1.5m * atr)
            {
                // stretched below VWAP → fade long back to VWAP
                var entry = px;
                var stop = entry - 0.75m * atr;
                var t1 = vwap;
                add_cand(lst, "S2", symbol, "BUY", entry, stop, t1, env, risk);
            }
            return lst;
        }

        public static List<Candidate> S3(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.3m && env.volz.HasValue && env.volz.Value < -0.3m)
            {
                var entry = bars[^1].Close;
                var stop = entry + env.atr.Value * 1.0m;
                var t1 = entry - env.atr.Value * 2.0m;
                add_cand(lst, "S3", symbol, "SELL", entry, stop, t1, env, risk);
            }
            return lst;
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
                                 decimal entry, decimal stop, decimal t1, Env env, RiskEngine risk)
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
