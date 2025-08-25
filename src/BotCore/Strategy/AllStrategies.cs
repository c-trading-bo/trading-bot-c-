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
        public static List<Signal> generate_candidates(string symbol, TradingProfileConfig cfg, StrategyDef def, List<Bar> bars, object risk)
        {
            // Dispatch to the specific strategy function based on def.Name (S1..S14)
            var env = new Env { atr = bars.Count > 0 ? (decimal?)Math.Abs(bars[^1].High - bars[^1].Low) : null, volz = 1.0m };
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
            var signals = new List<Signal>();
            foreach (var c in candidates)
            {
                signals.Add(new Signal
                {
                    StrategyId = c.strategy_id,
                    Symbol = c.symbol,
                    Side = c.side.ToString(),
                    Entry = c.entry,
                    Stop = c.stop,
                    Target = c.t1,
                    ExpR = c.expR,
                    Size = (int)c.qty,
                    AccountId = c.accountId,
                    ContractId = c.contractId,
                    Tag = c.Tag
                });
            }
            return signals;
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
                    Entry = c.entry,
                    Stop = c.stop,
                    Target = c.t1,
                    ExpR = c.expR,
                    Size = (int)c.qty,
                    AccountId = accountId,
                    ContractId = contractId,
                    Tag = c.Tag
                });
            }
            return signals;
        }

        // S1–S14 strategies
        public static List<Candidate> S1(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            // Require enough history for EMAs and ATR
            const int fastLen = 9;
            const int slowLen = 21;
            const int atrLen  = 14;
            if (bars is null || bars.Count < Math.Max(slowLen, atrLen) + 1) return lst;

            // Compute EMAs and ATR
            var emaFast = Ema(bars, fastLen);
            var emaSlow = Ema(bars, slowLen);
            var atr = Atr(bars, atrLen);
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
        private static List<decimal> Ema(IList<Bar> bars, int len)
        {
            var k = 2m / (len + 1);
            var ema = new List<decimal>(new decimal[bars.Count]);
            ema[0] = bars[0].Close;
            for (int i = 1; i < bars.Count; i++)
                ema[i] = bars[i].Close * k + ema[i - 1] * (1 - k);
            return ema;
        }

        private static decimal Atr(IList<Bar> bars, int len)
        {
            if (bars.Count < len + 1) return 0m;
            decimal sumTr = 0m;
            for (int i = bars.Count - len; i < bars.Count; i++)
            {
                var h = bars[i].High; var l = bars[i].Low; var pc = bars[i - 1].Close;
                var tr = Math.Max(h - l, Math.Max(Math.Abs(h - pc), Math.Abs(l - pc)));
                sumTr += tr;
            }
            return sumTr / len;
        }

        public static List<Candidate> S2(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.4m && env.volz.HasValue && env.volz.Value > 0.3m)
            {
                var entry = bars[^1].Close;
                var stop = entry - env.atr.Value * 1.0m;
                var t1 = entry + env.atr.Value * 2.0m;
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
            if (bars.Count > 0 && env.atr.HasValue && env.atr.Value > 0.8m)
            {
                var entry = bars[^1].Close;
                var stop = entry - env.atr.Value * 2.5m;
                var t1 = entry + env.atr.Value * 5.0m;
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
            var qty  = risk.size_for(risk.cfg.risk_per_trade, dist, pv);
            if (qty <= 0) return;

            var c = new Candidate
            {
                strategy_id = sid,
                symbol = symbol,
                side = sideTxt.Equals("BUY", StringComparison.OrdinalIgnoreCase) ? Side.BUY : Side.SELL,
                entry = entry,
                stop  = stop,
                t1    = t1,
                expR  = rr_quality(entry, stop, t1),
                qty   = qty,
                atr_ok = (env.atr ?? tick) >= tick,
                vol_z  = env.volz
            };
            lst.Add(c);
        }
    }
}
