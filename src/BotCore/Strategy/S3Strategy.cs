using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using BotCore.Models;
using BotCore.Risk;

namespace BotCore.Strategy
{
    public static class S3Strategy
    {
        private static bool BtBypass(string gate)
        {
            string k = gate switch { "news" => "BT_IGNORE_NEWS", "spread" => "BT_IGNORE_SPREAD", _ => string.Empty };
            if (string.IsNullOrEmpty(k)) return false;
            var v = Environment.GetEnvironmentVariable(k);
            return v is not null && (v.Equals("1", StringComparison.OrdinalIgnoreCase) || v.Equals("true", StringComparison.OrdinalIgnoreCase));
        }
        // S3 state and constants
        private static readonly ConcurrentDictionary<string, SegmentState> _segState = new(StringComparer.OrdinalIgnoreCase);
        private static readonly ConcurrentDictionary<(string Sym, DateOnly Day, string Sess, Side Side), int> _attempts = new();
        private static readonly TimeSpan OvernightWinStart = new(2, 55, 0);
        private static readonly TimeSpan OvernightWinEnd = new(4, 10, 0);
        private static readonly TimeZoneInfo Et = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");

        // Optional: debug counters to understand why entries are rejected in backtests
        private static readonly ConcurrentDictionary<string, int> _rejects = new(StringComparer.OrdinalIgnoreCase);
        private static bool DebugOn
        {
            get
            {
                var v = Environment.GetEnvironmentVariable("S3_DEBUG_REASONS") ?? string.Empty;
                return v.Equals("1", StringComparison.OrdinalIgnoreCase) || v.Equals("true", StringComparison.OrdinalIgnoreCase);
            }
        }
        private static void Reject(string key)
        {
            if (!DebugOn) return;
            _rejects.AddOrUpdate(key, 1, static (_, c) => c + 1);
        }
        public static void ResetDebugCounters() => _rejects.Clear();
        public static IReadOnlyDictionary<string, int> GetDebugCounters() => new Dictionary<string, int>(_rejects);

        public static List<Candidate> S3(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            var lst = new List<Candidate>();
            if (bars is null || bars.Count < 80) return lst; // need enough for pre-squeeze, TF2 agg, etc.
            // Normalize bar timestamps to ET for all session/time-of-day logic (bars arrive as UTC)
            static DateTime ToEt(Bar b)
            {
                try
                {
                    DateTime utc = b.Ts > 0
                        ? DateTimeOffset.FromUnixTimeMilliseconds(b.Ts).UtcDateTime
                        : (b.Start.Kind == DateTimeKind.Utc ? b.Start : DateTime.SpecifyKind(b.Start, DateTimeKind.Utc));
                    return TimeZoneInfo.ConvertTimeFromUtc(utc, Et);
                }
                catch { return b.Start; }
            }
            var barsEt = new List<Bar>(bars.Count);
            foreach (var b in bars)
            {
                barsEt.Add(new Bar
                {
                    Start = ToEt(b),
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

            var cfg = S3RuntimeConfig.Instance;
            var last = bars[^1];

            // News block (bypass in backtest)
            if (!BtBypass("news"))
            {
                if (InNewsWindow(last.Start, cfg.NewsOnMinutes, cfg.NewsBlockBeforeMin, cfg.NewsBlockAfterMin))
                {
                    Reject("news_window");
                    return lst;
                }
            }

            // Volume gate
            if (last.Volume < cfg.MinVolume)
            {
                Reject("min_volume");
                return lst;
            }

            // Spread gate (optional provider)
            var spread = AllStrategies.ExternalSpreadTicks?.Invoke(symbol);
            // instrument overrides (pre-derive locals)
            int localMaxSpread = cfg.MaxSpreadTicks;
            decimal localWidthRankEnter = cfg.WidthRankEnter;
            int localNrClusterMinBars = cfg.NrClusterMinBars;
            decimal localBreakQ_MinClosePos = cfg.BreakQ_MinClosePos;
            decimal localBreakQ_MaxOppWick = cfg.BreakQ_MaxOppWick;
            if (S3RuntimeConfig.TryGetOverride(symbol, out var ov) && ov is not null)
            {
                if (ov.MaxSpreadTicks.HasValue) localMaxSpread = ov.MaxSpreadTicks.Value;
                if (ov.WidthRankEnter.HasValue) localWidthRankEnter = ov.WidthRankEnter.Value;
                if (ov.NrClusterMinBars.HasValue) localNrClusterMinBars = ov.NrClusterMinBars.Value;
                if (ov.BreakQ_MinClosePos.HasValue) localBreakQ_MinClosePos = ov.BreakQ_MinClosePos.Value;
                if (ov.BreakQ_MaxOppWick.HasValue) localBreakQ_MaxOppWick = ov.BreakQ_MaxOppWick.Value;
            }
            if (!BtBypass("spread") && spread.HasValue)
            {
                var spreadCap = symbol.Contains("NQ", StringComparison.OrdinalIgnoreCase) ? Math.Max(localMaxSpread, 3) : localMaxSpread;
                if (spread.Value > spreadCap)
                {
                    Reject("spread_cap");
                    return lst;
                }
            }

            // VolZ regime gate if available
            if (env.volz.HasValue)
            {
                if (env.volz.Value < cfg.VolZMin || env.volz.Value > cfg.VolZMax)
                {
                    Reject("volz_regime");
                    return lst;
                }
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
            var rankThresh = localWidthRankEnter;
            if (cfg.HourlyRankAdapt)
            {
                rankThresh = AdaptedRankThreshold(last.Start, rankThresh);
            }
            bool squeezeOnRank = widthRank <= rankThresh;

            // TTM squeeze
            bool squeezeOnTTM = (bbUp <= kcUp) && (bbDn >= kcDn);

            // Squeeze run length using TTM condition
            int squeezeRun = SqueezeRunLength(bars, cfg.BbLen, cfg.BbMult, cfg.KcEma, cfg.KcAtrLen, cfg.KcMult);

            // Width slope down and narrow-range cluster
            bool widthSlopeOk = WidthSlopeDown(bars, cfg.BbLen, cfg.BbMult, cfg.WidthSlopeDownBars, cfg.WidthSlopeTol);
            bool hasNrCluster = HasNarrowRangeCluster(bars, cfg.PreSqueezeLookback, cfg.NrClusterRatio, localNrClusterMinBars);

            bool squeezeArmed = (squeezeOnTTM || squeezeOnRank) && squeezeRun >= cfg.MinSqueezeBars && hasNrCluster && widthSlopeOk;
            if (!squeezeArmed)
            {
                if (!squeezeOnTTM && !squeezeOnRank) Reject("squeeze_off");
                if (squeezeRun < cfg.MinSqueezeBars) Reject("squeeze_len");
                if (!hasNrCluster) Reject("nr_cluster");
                if (!widthSlopeOk) Reject("width_slope");
                return lst;
            }

            // TF2 slope bias via 5m aggregation
            var bars5 = Aggregate(bars, 5);
            var ema20_5 = EMA([.. bars5.Select(b => b.Close)], 20);
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
                        if (last.Close < orl && bbUp > orl) { Reject("or_guard"); return lst; }
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
                    if (last.Close < ibl && bbUp > ibl) { Reject("ib_guard"); return lst; }
                    if (last.Close > ibh && bbDn < ibh) return lst;
                }
            }

            // Roll guard tighten (tighter rank requirement around roll week)
            if (cfg.RollEnabled && IsWithinRollWindow(last.Start.Date, cfg.RollDaysBefore, cfg.RollDaysAfter))
            {
                rankThresh = Math.Max(0.05m, rankThresh - cfg.RollRankTighten);
                if (widthRank > rankThresh) { Reject("roll_tighten"); return lst; } // re-check if adapting tightened threshold disqualifies
            }

            // RS filter (optional, needs ExternalGetBars)
            if (cfg.RsEnabled && AllStrategies.ExternalGetBars != null)
            {
                var peer = TryGetPeer(symbol, cfg); // ES<->NQ
                if (!string.IsNullOrEmpty(peer))
                {
                    var peerBars = AllStrategies.ExternalGetBars(peer!);
                    if (peerBars != null && peerBars.Count >= cfg.RsWindowBars + 2)
                    {
                        var rs = RelativeStrength(bars, [.. peerBars], cfg.RsWindowBars);
                        if (cfg.RsDirectionalOnly)
                        {
                            if (biasUp && rs < cfg.RsThreshold) { Reject("rs_filter"); return lst; }
                            if (biasDn && rs > -cfg.RsThreshold) { Reject("rs_filter"); return lst; }
                        }
                        else if (Math.Abs(rs) < cfg.RsThreshold) { Reject("rs_filter"); return lst; }
                    }
                }
            }

            // Impulse
            var impulseScore = ImpulseScore(bars, 3) / boxW;
            if (impulseScore < cfg.ImpulseScoreMin) { Reject("impulse_low"); return lst; }

            // Buffers
            decimal buf = cfg.ConfirmBreakAtrMult * atr;
            if ((biasUp && last.Close < boxHi) || (biasDn && last.Close > boxLo)) buf *= cfg.ContraBufferMult;
            if (InOvernightWindow(last.Start.TimeOfDay)) buf += cfg.OvernightBufferAdd * atr;

            // Break bar quality (apply instrument overrides if any)
            if (!BreakBarQualityOk(last, localBreakQ_MinClosePos, localBreakQ_MaxOppWick, out var barq)) { Reject("break_bar_quality"); return lst; }

            // Anchored VWAP from segment start
            var segAnchor = bars[Math.Max(0, segStartIdx)].Start;
            var (segVwap, _, _) = AnchoredVwap(bars, segAnchor);

            // Determine session and attempt cap
            var session = InOvernightWindow(last.Start.TimeOfDay) ? "ON" : "RTH";
            int cap = session == "ON" ? cfg.AttemptCapOvernight : cfg.AttemptCapRTH;

            // Breakout detection (allow kc bands if biased)
            bool brokeUp = last.High > boxHi + buf || (biasUp && last.Close > kcUp + buf);
            bool brokeDn = last.Low < boxLo - buf || (biasDn && last.Close < kcDn - buf);

            // Early invalidate tick
            st.TickInvalidate(bars, mid, boxHi, boxLo, cfg.EarlyInvalidateBars);
            if (st.IsInvalid) { Reject("early_invalidate"); return lst; }

            var px = last.Close;
            var tick = AllStrategies.ExternalTickSize?.Invoke(symbol) ?? GuessTickSize(symbol);

            // LONG side
            if (brokeUp && CanAttempt(symbol, session, Side.BUY, cap))
            {
                decimal entry;
                if (cfg.EntryMode.Equals("retest", StringComparison.OrdinalIgnoreCase))
                {
                    var backoff = cfg.RetestBackoffTicks * tick;
                    if (!DidThrowback(bars, boxHi, true, cfg.RetestBars, backoff)) { Reject("retest_throwback_long"); return lst; }
                    entry = Math.Max(px, boxHi - backoff);
                }
                else // breakstop
                {
                    entry = Math.Max(px, Math.Max(last.High + tick, boxHi + buf));
                }
                if (entry <= 0) return lst;
                if (!HoldsAroundVwap(bars, segVwap, true, 2)) { Reject("vwap_hold_long"); return lst; }

                var isl = Math.Min(boxLo - cfg.StopAtrMult * atr, SwingLow(bars, 5));
                var r = entry - isl;
                if (r <= 0) { Reject("risk_nonpos_long"); return lst; }
                var expF = ExpectedExpansionFactor(bars, Math.Max(60, cfg.PreSqueezeLookback), boxW, cfg.ExpansionQuantile);
                var (t1, _) = Targets(cfg, r, entry, boxW, true, expF);
                AllStrategies.add_cand(lst, "S3", symbol, "BUY", entry, isl, t1, env, risk);
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
                    if (!DidThrowback(bars, boxLo, false, cfg.RetestBars, backoff)) { Reject("retest_throwback_short"); return lst; }
                    entry = Math.Min(px, boxLo + backoff);
                }
                else
                {
                    entry = Math.Min(px, Math.Min(last.Low - tick, boxLo - buf));
                }
                if (entry <= 0) { Reject("entry_nonpos_short"); return lst; }
                if (!HoldsAroundVwap(bars, segVwap, false, 2)) { Reject("vwap_hold_short"); return lst; }

                var ish = Math.Max(boxHi + cfg.StopAtrMult * atr, SwingHigh(bars, 5));
                var r = ish - entry;
                if (r <= 0) { Reject("risk_nonpos_short"); return lst; }
                var expF = ExpectedExpansionFactor(bars, Math.Max(60, cfg.PreSqueezeLookback), boxW, cfg.ExpansionQuantile);
                var (t1, _) = Targets(cfg, r, entry, boxW, false, expF);
                AllStrategies.add_cand(lst, "S3", symbol, "SELL", entry, ish, t1, env, risk,
                    tag: $"rank={widthRank:F2} run={squeezeRun} nrOK={hasNrCluster} slope5={slope5:F3} barq={barq:F2}");
                st.MarkFilled(segStartIdx, Side.SELL, last.Start);
                RegisterAttempt(symbol, session, Side.SELL);
            }

            // Attempt caps blocked a side?
            if (brokeUp && !CanAttempt(symbol, session, Side.BUY, cap)) Reject("attempt_cap_long");
            if (brokeDn && !CanAttempt(symbol, session, Side.SELL, cap)) Reject("attempt_cap_short");

            return lst;
        }

        // ---- S3 helpers (local, full) ----
        private static DateTime AnchorToday(DateTime localDate, TimeSpan time) => localDate.Date + time;
        private static (decimal high, decimal low) InitialBalance(IList<Bar> bars, DateTime startLocal, DateTime endLocal)
        {
            decimal hi = 0, lo; bool seen;
            foreach (var b in bars)
            {
                if (b.Start >= startLocal && b.Start < endLocal)
                {
                    if (!seen) { hi = b.High; lo = b.Low; seen = true; }
                    else { if (b.High > hi) hi = b.High; if (b.Low < lo) lo = b.Low; }
                }
            }
            return seen ? (hi, lo) : (0m, 0m);
        }
        private static decimal[] EMA(decimal[] x, int n)
        {
            if (x.Length == 0) return [];
            var a = 2m / (n + 1);
            var y = new decimal[x.Length];
            y[0] = x[0];
            for (int i = 1; i < x.Length; i++) y[i] = a * x[i] + (1 - a) * y[i - 1];
            return y;
        }
        private static decimal Stdev(decimal[] x, int n)
        {
            if (x.Length < n) return 0m;
            decimal sum = 0m, sum2;
            for (int i = x.Length - n; i < x.Length; i++) { sum += x[i]; sum2 += x[i] * x[i]; }
            var mean = sum / n; var varv = (sum2 / n) - mean * mean;
            return varv <= 0 ? 0m : (decimal)Math.Sqrt((double)varv);
        }
        private static decimal ATR(IList<Bar> b, int n)
        {
            if (b.Count < n + 1) return 0m;
            decimal atr; int start = b.Count - n;
            for (int i = start; i < b.Count; i++)
            {
                var c = b[i]; var p = b[i - 1];
                var tr = Math.Max(c.High - c.Low, Math.Max(Math.Abs(c.High - p.Close), Math.Abs(c.Low - p.Close)));
                atr = i == start ? tr : (atr * (n - 1) + tr) / n;
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
            int desc;
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
                trs.Add(Math.Max(c.High - c.Low, Math.Max(Math.Abs(c.High - p.Close), Math.Abs(c.Low - p.Close))));
            }
            if (trs.Count < minBars) return false;
            var sorted = trs.OrderBy(x => x).ToList();
            var med = sorted[sorted.Count / 2];
            int cnt;
            for (int i = trs.Count - 1; i >= 0 && cnt < minBars; i--)
            {
                if (trs[i] <= ratio * med) cnt++; else break;
            }
            return cnt >= minBars;
        }
        private static int SqueezeRunLength(IList<Bar> bars, int bbLen, decimal bbMult, int kcEma, int kcAtrLen, decimal kcMult)
        {
            int run;
            var closes = bars.Select(b => b.Close).ToArray();
            for (int i = bars.Count - 1; i >= Math.Max(0, bars.Count - 60); i--)
            {
                var arr = closes.Take(i + 1).ToArray();
                var ema = EMA(arr, kcEma); var mid = ema[^1];
                var sd = Stdev(arr, bbLen); var bbUp = mid + bbMult * sd; var bbDn = mid - bbMult * sd;
                var atr = ATR([.. bars.Take(i + 1)], kcAtrLen);
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
                var atr = ATR([.. bars.Take(i + 1)], kcAtrLen); var kcUp = mid + kcMult * atr; var kcLo = mid - kcMult * atr;
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
            if (bars == null || bars.Count == 0 || n <= 1) return bars?.ToList() ?? [];
            var res = new List<Bar>();
            int i;
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
            int last = arr.Length - 1;
            return (arr[last] - arr[last - n]) / n;
        }
        private static decimal RelativeStrength(IList<Bar> a, IList<Bar> b, int look)
        {
            int sa = Math.Max(0, a.Count - look);
            int sb = Math.Max(0, b.Count - look);
            var ra = a[sa].Close == 0 ? 0m : (a[a.Count - 1].Close - a[sa].Close) / a[sa].Close;
            var rb = b[sb].Close == 0 ? 0m : (b[b.Count - 1].Close - b[sb].Close) / b[sb].Close;
            return ra - rb;
        }
        private static (decimal vwap, decimal wvar, decimal wvol) AnchoredVwap(IList<Bar> bars, DateTime anchorLocal)
        {
            decimal wv = 0m, vol; int idx0;
            for (int i; i < bars.Count; i++) { if (bars[i].Start >= anchorLocal) { idx0 = i; break; } }
            for (int i = idx0; i < bars.Count; i++) { var b = bars[i]; var tp = (b.High + b.Low + b.Close) / 3m; var v = Math.Max(0, b.Volume); wv += tp * v; vol += v; }
            if (vol <= 0) return (0m, 0m, 0m);
            var vwap = wv / vol; decimal num;
            for (int i = idx0; i < bars.Count; i++) { var b = bars[i]; var tp = (b.High + b.Low + b.Close) / 3m; var v = Math.Max(0, b.Volume); var d = tp - vwap; num += d * d * v; }
            var wvar = num / vol; return (vwap, wvar, vol);
        }
        private static bool HoldsAroundVwap(IList<Bar> b, decimal vwap, bool above, int need)
        { int ok; for (int i = b.Count - need; i < b.Count; i++) { var c = b[i].Close; if (above ? c >= vwap : c <= vwap) ok++; } return ok >= need; }
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
        private static bool DidThrowback(IList<Bar> bars, decimal level, bool longSide, int lookbackBars, decimal tol)
        {
            int s = Math.Max(0, bars.Count - lookbackBars);
            if (longSide)
                return bars.Skip(s).Any(b => b.Low <= level + tol);
            else
                return bars.Skip(s).Any(b => b.High >= level - tol);
        }
        private static decimal Quantile(List<decimal> a, decimal p)
        {
            if (a == null || a.Count == 0) return 0m; var t = a.OrderBy(x => x).ToList();
            p = Math.Clamp(p, 0m, 1m);
            var idx = (t.Count - 1) * (double)p;
            int lo = (int)Math.Floor(idx); int hi = (int)Math.Ceiling(idx);
            if (lo == hi) return t[lo];
            var w = (decimal)(idx - lo);
            return t[lo] + w * (t[hi] - t[lo]);
        }
        private static decimal ExpectedExpansionFactor(IList<Bar> bars, int lookback, decimal boxW, decimal qLevel)
        {
            int s = Math.Max(1, bars.Count - lookback);
            var trs = new List<decimal>();
            for (int i = s; i < bars.Count; i++)
            {
                var c = bars[i]; var p = bars[i - 1];
                var tr = Math.Max(c.High - c.Low, Math.Max(Math.Abs(c.High - p.Close), Math.Abs(c.Low - p.Close)));
                trs.Add(tr);
            }
            if (trs.Count == 0 || boxW <= 0) return 1.6m;
            var q = Quantile(trs, qLevel);
            var factor = q / Math.Max(1e-9m, boxW / 2m);
            return Math.Clamp(factor, 0.8m, 2.5m);
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
        private static (decimal t1, decimal t2) Targets(S3RuntimeConfig cfg, decimal r, decimal entry, decimal boxW, bool isLong, decimal? expOverride = null)
        {
            if (cfg.TargetsMode.Equals("expansion", StringComparison.OrdinalIgnoreCase))
            {
                var exp = expOverride ?? 1.6m; // fallback expected expansion factor of box
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
        private static decimal AdaptedRankThreshold(DateTime local, decimal baseRank)
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
                    FilledThisSegment; IsInvalid; LastBreakBarIndex = -1;
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
                if (IsInvalid && (bars.Count - 1 - LastBreakBarIndex) > earlyInvalidateBars) IsInvalid;
            }
        }

        private sealed class S3RuntimeConfig
        {
            public sealed class InstrumentOverride
            {
                public int? MaxSpreadTicks { get; init; }
                public decimal? WidthRankEnter { get; init; }
                public int? NrClusterMinBars { get; init; }
                public decimal? BreakQ_MinClosePos { get; init; }
                public decimal? BreakQ_MaxOppWick { get; init; }
            }
            private static readonly Dictionary<string, InstrumentOverride> _instrumentOverrides = new(StringComparer.OrdinalIgnoreCase);
            public static bool TryGetOverride(string sym, out InstrumentOverride? ov) => _instrumentOverrides.TryGetValue(sym, out ov);
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
            public Dictionary<string, string> Peers { get; init; } = new(StringComparer.OrdinalIgnoreCase);

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
            public int[] NewsOnMinutes { get; init; } = [0, 30];

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
                    string[] bases = [AppContext.BaseDirectory, Directory.GetCurrentDirectory()];
                    foreach (var b in bases.Distinct())
                    {
                        var dir = new DirectoryInfo(b);
                        for (int i; i < 7 && dir != null; i++, dir = dir.Parent)
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
                            if (e.ValueKind != JsonValueKind.Array) return [];
                            var list = new List<int>();
                            foreach (var it in e.EnumerateArray()) if (it.TryGetInt32(out var v)) list.Add(v);
                            return [.. list];
                        }
                        var peers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                        if (s3.TryGetProperty("rs_filter", out var rsNode) && rsNode.ValueKind == JsonValueKind.Object)
                        {
                            if (rsNode.TryGetProperty("peers", out var pe) && pe.ValueKind == JsonValueKind.Object)
                            {
                                foreach (var kv in pe.EnumerateObject()) peers[kv.Name] = kv.Value.GetString() ?? "";
                            }
                        }
                        var newsMins = new[] { 0, 30 };
                        if (s3.TryGetProperty("news_block", out var nbNode) && nbNode.ValueKind == JsonValueKind.Object)
                        {
                            if (nbNode.TryGetProperty("on_minutes", out var onm)) newsMins = GetIntArray(onm);
                        }
                        // Parse instrument_overrides
                        if (s3.TryGetProperty("instrument_overrides", out var ios) && ios.ValueKind == JsonValueKind.Object)
                        {
                            _instrumentOverrides.Clear();
                            foreach (var kvp in ios.EnumerateObject())
                            {
                                var sym = kvp.Name;
                                var node = kvp.Value;
                                int? maxSpread = null; decimal? wre = null; int? nrb_override = null; decimal? qMinClose = null; decimal? qMaxOpp = null;
                                if (node.ValueKind == JsonValueKind.Object)
                                {
                                    if (node.TryGetProperty("max_spread_ticks", out var p1o) && p1o.TryGetInt32(out var i1o)) maxSpread = i1o;
                                    if (node.TryGetProperty("width_rank_enter", out var p2o) && p2o.TryGetDecimal(out var d2o)) wre = d2o;
                                    if (node.TryGetProperty("nr_cluster_min_bars", out var p3o) && p3o.TryGetInt32(out var i3o)) nrb_override = i3o;
                                    if (node.TryGetProperty("break_bar_quality", out var bbqo) && bbqo.ValueKind == JsonValueKind.Object)
                                    {
                                        if (bbqo.TryGetProperty("min_close_pos", out var mcp_o) && mcp_o.TryGetDecimal(out var dmcp_o)) qMinClose = dmcp_o;
                                        if (bbqo.TryGetProperty("max_opp_wick", out var mow_o) && mow_o.TryGetDecimal(out var dmow_o)) qMaxOpp = dmow_o;
                                    }
                                }
                                _instrumentOverrides[sym] = new InstrumentOverride
                                {
                                    MaxSpreadTicks = maxSpread,
                                    WidthRankEnter = wre,
                                    NrClusterMinBars = nrb_override,
                                    BreakQ_MinClosePos = qMinClose,
                                    BreakQ_MaxOppWick = qMaxOpp
                                };
                            }
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
                            HourlyRankAdapt = s3.TryGetProperty("hourly_rank_adapt", out var hra) && hra.ValueKind == JsonValueKind.True || (hra.ValueKind != JsonValueKind.False),
                            NrClusterRatio = s3.TryGetProperty("nr_cluster_ratio", out var nrr) && nrr.TryGetDecimal(out var dnrr) ? dnrr : 0.60m,
                            NrClusterMinBars = s3.TryGetProperty("nr_cluster_min_bars", out var nrb) && nrb.TryGetInt32(out var inrb) ? inrb : 5,
                            WidthSlopeDownBars = s3.TryGetProperty("width_slope_down_bars", out var wsb) && wsb.TryGetInt32(out var iwsb) ? iwsb : 8,
                            WidthSlopeTol = s3.TryGetProperty("width_slope_tol", out var wst) && wst.TryGetDecimal(out var dwst) ? dwst : 0.0m,
                            ContraBufferMult = s3.TryGetProperty("contra_buffer_mult", out var cb) && cb.TryGetDecimal(out var dcb) ? dcb : 1.5m,
                            OvernightBufferAdd = s3.TryGetProperty("overnight_buffer_add", out var oba) && oba.TryGetDecimal(out var doba) ? doba : 0.05m,
                            EntryMode = s3.TryGetProperty("entry_mode", out var em) && em.ValueKind == JsonValueKind.String ? em.GetString() ?? "retest" : "retest",
                            RetestBars = s3.TryGetProperty("retest_bars", out var rbs) && rbs.TryGetInt32(out var irbs) ? irbs : 5,
                            RetestBackoffTicks = s3.TryGetProperty("retest_backoff_ticks", out var rbt) && rbt.TryGetInt32(out var irbt) ? irbt : 1,
                            OrGuardEnabled = s3.TryGetProperty("or_guard", out var org) && org.TryGetProperty("enabled", out var oge) && oge.ValueKind == JsonValueKind.True,
                            OrMinutes = s3.TryGetProperty("or_guard", out var org2) && org2.TryGetProperty("minutes", out var ogm) && ogm.TryGetInt32(out var iogm) ? iogm : 10,
                            OrAvoidBreakInto = s3.TryGetProperty("or_guard", out var org3) && org3.TryGetProperty("avoid_break_into", out var abi) && abi.ValueKind == JsonValueKind.String ? abi.GetString() ?? "opposite" : "opposite",
                            RsEnabled = s3.TryGetProperty("rs_filter", out var rsA) && rsA.TryGetProperty("enabled", out var rse) && rse.ValueKind == JsonValueKind.True,
                            RsWindowBars = s3.TryGetProperty("rs_filter", out var rsB) && rsB.TryGetProperty("window_bars", out var rwb) && rwb.TryGetInt32(out var irwb) ? irwb : 60,
                            RsThreshold = s3.TryGetProperty("rs_filter", out var rsC) && rsC.TryGetProperty("threshold", out var rst) && rst.TryGetDecimal(out var drst) ? drst : 0.10m,
                            RsDirectionalOnly = s3.TryGetProperty("rs_filter", out var rsD) && rsD.TryGetProperty("directional_only", out var rdo) && rdo.ValueKind == JsonValueKind.True,
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

            // Internal override for tuning: apply a JSON blob shaped like S3-StrategyConfig.json
            public static void OverrideFromJson(string json)
            {
                try
                {
                    using var doc = JsonDocument.Parse(json, new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });
                    var root = doc.RootElement;
                    if (root.TryGetProperty("Strategies", out var arr) && arr.ValueKind == JsonValueKind.Array && arr.GetArrayLength() > 0)
                    {
                        // Reuse loader by writing temp file content into a new config instance
                        var cfg = _instance ?? new S3RuntimeConfig();
                        // Simple approach: write json to a temp file and reuse parsing path is overkill; instead re-map a subset
                        var s3 = arr[0];
                        var peers = cfg.Peers;
                        if (s3.TryGetProperty("rs_filter", out var rsNode) && rsNode.ValueKind == JsonValueKind.Object)
                        {
                            if (rsNode.TryGetProperty("peers", out var pe) && pe.ValueKind == JsonValueKind.Object)
                            {
                                peers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                                foreach (var kv in pe.EnumerateObject()) peers[kv.Name] = kv.Value.GetString() ?? "";
                            }
                        }
                        _instance = new S3RuntimeConfig
                        {
                            BbLen = s3.TryGetProperty("bb_len", out var v1) && v1.TryGetInt32(out var i1) ? i1 : cfg.BbLen,
                            BbMult = s3.TryGetProperty("bb_mult", out var v2) && v2.TryGetDecimal(out var d2) ? d2 : cfg.BbMult,
                            KcEma = s3.TryGetProperty("kc_ema", out var v3) && v3.TryGetInt32(out var i3) ? i3 : cfg.KcEma,
                            KcAtrLen = s3.TryGetProperty("kc_atr_len", out var v4) && v4.TryGetInt32(out var i4) ? i4 : cfg.KcAtrLen,
                            KcMult = s3.TryGetProperty("kc_mult", out var v5) && v5.TryGetDecimal(out var d5) ? d5 : cfg.KcMult,
                            AtrLen = s3.TryGetProperty("atr_len", out var v6) && v6.TryGetInt32(out var i6) ? i6 : cfg.AtrLen,
                            MinSqueezeBars = s3.TryGetProperty("min_squeeze_bars", out var v7) && v7.TryGetInt32(out var i7) ? i7 : cfg.MinSqueezeBars,
                            PreSqueezeLookback = s3.TryGetProperty("pre_squeeze_lookback", out var v8) && v8.TryGetInt32(out var i8) ? i8 : cfg.PreSqueezeLookback,
                            ConfirmBreakAtrMult = s3.TryGetProperty("confirm_break_mult", out var v9) && v9.TryGetDecimal(out var d9) ? d9 : cfg.ConfirmBreakAtrMult,
                            StopAtrMult = s3.TryGetProperty("stop_atr_mult", out var v10) && v10.TryGetDecimal(out var d10) ? d10 : cfg.StopAtrMult,
                            TargetR1 = s3.TryGetProperty("target_r1", out var v11) && v11.TryGetDecimal(out var d11) ? d11 : cfg.TargetR1,
                            TargetR2 = s3.TryGetProperty("target_r2", out var v11b) && v11b.TryGetDecimal(out var d11b) ? d11b : cfg.TargetR2,
                            MinVolume = s3.TryGetProperty("min_volume", out var v12) && v12.TryGetInt32(out var i12) ? i12 : cfg.MinVolume,
                            MaxSpreadTicks = s3.TryGetProperty("max_spread_ticks", out var v13) && v13.TryGetInt32(out var i13) ? i13 : cfg.MaxSpreadTicks,
                            AttemptCapRTH = s3.TryGetProperty("attempt_cap", out var ac) && ac.TryGetProperty("RTH", out var rth) && rth.TryGetInt32(out var iac1) ? iac1 : cfg.AttemptCapRTH,
                            AttemptCapOvernight = s3.TryGetProperty("attempt_cap", out var ac2) && ac2.TryGetProperty("overnight", out var on) && on.TryGetInt32(out var iac2) ? iac2 : cfg.AttemptCapOvernight,
                            WidthRankEnter = s3.TryGetProperty("width_rank_enter", out var wr) && wr.TryGetDecimal(out var dwr) ? dwr : cfg.WidthRankEnter,
                            HourlyRankAdapt = s3.TryGetProperty("hourly_rank_adapt", out var hra) && hra.ValueKind == JsonValueKind.True || (hra.ValueKind != JsonValueKind.False && cfg.HourlyRankAdapt),
                            NrClusterRatio = s3.TryGetProperty("nr_cluster_ratio", out var nrr) && nrr.TryGetDecimal(out var dnrr) ? dnrr : cfg.NrClusterRatio,
                            NrClusterMinBars = s3.TryGetProperty("nr_cluster_min_bars", out var nrb) && nrb.TryGetInt32(out var inrb) ? inrb : cfg.NrClusterMinBars,
                            WidthSlopeDownBars = s3.TryGetProperty("width_slope_down_bars", out var wsb) && wsb.TryGetInt32(out var iwsb) ? iwsb : cfg.WidthSlopeDownBars,
                            WidthSlopeTol = s3.TryGetProperty("width_slope_tol", out var wst) && wst.TryGetDecimal(out var dwst) ? dwst : cfg.WidthSlopeTol,
                            ContraBufferMult = s3.TryGetProperty("contra_buffer_mult", out var cb) && cb.TryGetDecimal(out var dcb) ? dcb : cfg.ContraBufferMult,
                            OvernightBufferAdd = s3.TryGetProperty("overnight_buffer_add", out var oba) && oba.TryGetDecimal(out var doba) ? doba : cfg.OvernightBufferAdd,
                            EntryMode = s3.TryGetProperty("entry_mode", out var em) && em.ValueKind == JsonValueKind.String ? (em.GetString() ?? cfg.EntryMode) : cfg.EntryMode,
                            RetestBars = s3.TryGetProperty("retest_bars", out var rbs) && rbs.TryGetInt32(out var irbs) ? irbs : cfg.RetestBars,
                            RetestBackoffTicks = s3.TryGetProperty("retest_backoff_ticks", out var rbt) && rbt.TryGetInt32(out var irbt) ? irbt : cfg.RetestBackoffTicks,
                            OrGuardEnabled = s3.TryGetProperty("or_guard", out var org) && org.TryGetProperty("enabled", out var oge) && oge.ValueKind == JsonValueKind.True,
                            OrMinutes = s3.TryGetProperty("or_guard", out var org2) && org2.TryGetProperty("minutes", out var ogm) && ogm.TryGetInt32(out var iogm) ? iogm : cfg.OrMinutes,
                            OrAvoidBreakInto = s3.TryGetProperty("or_guard", out var org3) && org3.TryGetProperty("avoid_break_into", out var abi) && abi.ValueKind == JsonValueKind.String ? (abi.GetString() ?? cfg.OrAvoidBreakInto) : cfg.OrAvoidBreakInto,
                            RsEnabled = s3.TryGetProperty("rs_filter", out var rsA) && rsA.TryGetProperty("enabled", out var rse) && rse.ValueKind == JsonValueKind.True,
                            RsWindowBars = s3.TryGetProperty("rs_filter", out var rsB) && rsB.TryGetProperty("window_bars", out var rwb) && rwb.TryGetInt32(out var irwb) ? irwb : cfg.RsWindowBars,
                            RsThreshold = s3.TryGetProperty("rs_filter", out var rsC) && rsC.TryGetProperty("threshold", out var rst) && rst.TryGetDecimal(out var drst) ? drst : cfg.RsThreshold,
                            RsDirectionalOnly = s3.TryGetProperty("rs_filter", out var rsD) && rsD.TryGetProperty("directional_only", out var rdo) && rdo.ValueKind == JsonValueKind.True,
                            Peers = peers,
                            RollEnabled = s3.TryGetProperty("roll_guard", out var rg) && rg.TryGetProperty("enabled", out var rge) && rge.ValueKind == JsonValueKind.True,
                            RollDaysBefore = s3.TryGetProperty("roll_guard", out var rg2) && rg2.TryGetProperty("days_before", out var rdb) && rdb.TryGetInt32(out var irdb) ? irdb : cfg.RollDaysBefore,
                            RollDaysAfter = s3.TryGetProperty("roll_guard", out var rg3) && rg3.TryGetProperty("days_after", out var rda) && rda.TryGetInt32(out var irda) ? irda : cfg.RollDaysAfter,
                            RollRankTighten = s3.TryGetProperty("roll_guard", out var rg4) && rg4.TryGetProperty("rank_tighten", out var rrt) && rrt.TryGetDecimal(out var drrt) ? drrt : cfg.RollRankTighten,
                            BreakQ_MinClosePos = s3.TryGetProperty("break_bar_quality", out var bbq) && bbq.TryGetProperty("min_close_pos", out var mcp) && mcp.TryGetDecimal(out var dmcp) ? dmcp : cfg.BreakQ_MinClosePos,
                            BreakQ_MaxOppWick = s3.TryGetProperty("break_bar_quality", out var bbq2) && bbq2.TryGetProperty("max_opp_wick", out var mow) && mow.TryGetDecimal(out var dmow) ? dmow : cfg.BreakQ_MaxOppWick,
                            ValidityBars = s3.TryGetProperty("validity_bars", out var vb) && vb.TryGetInt32(out var ivb) ? ivb : cfg.ValidityBars,
                            CooldownBars = s3.TryGetProperty("cooldown_bars", out var cb2) && cb2.TryGetInt32(out var icb) ? icb : cfg.CooldownBars,
                            MaxBarsInTrade = s3.TryGetProperty("max_bars_in_trade", out var mb) && mb.TryGetInt32(out var imb) ? imb : cfg.MaxBarsInTrade,
                            TrailAtrMult = s3.TryGetProperty("trail_atr_mult", out var tam) && tam.TryGetDecimal(out var dtam) ? dtam : cfg.TrailAtrMult,
                            NewsBlockBeforeMin = s3.TryGetProperty("news_block", out var nb) && nb.TryGetProperty("minutes_before", out var nbB) && nbB.TryGetInt32(out var inbB) ? inbB : cfg.NewsBlockBeforeMin,
                            NewsBlockAfterMin = s3.TryGetProperty("news_block", out var nb2) && nb2.TryGetProperty("minutes_after", out var nbA) && nbA.TryGetInt32(out var inbA) ? inbA : cfg.NewsBlockAfterMin,
                            NewsOnMinutes = s3.TryGetProperty("news_block", out var nb3) && nb3.TryGetProperty("on_minutes", out var onm) && onm.ValueKind == JsonValueKind.Array ? [.. onm.EnumerateArray().Where(e => e.TryGetInt32(out _)).Select(e => e.GetInt32())] : cfg.NewsOnMinutes,
                            OnePerSegment = s3.TryGetProperty("one_per_segment", out var ops) && ops.ValueKind == JsonValueKind.True,
                            SegmentCooldownMinutes = s3.TryGetProperty("segment_cooldown_minutes", out var scm) && scm.TryGetInt32(out var iscm) ? iscm : cfg.SegmentCooldownMinutes,
                            EarlyInvalidateBars = s3.TryGetProperty("early_invalidate_bars", out var eib) && eib.TryGetInt32(out var ieib) ? ieib : cfg.EarlyInvalidateBars,
                            ImpulseScoreMin = s3.TryGetProperty("impulse_score_min", out var ism) && ism.TryGetDecimal(out var dism) ? dism : cfg.ImpulseScoreMin,
                            TargetsMode = s3.TryGetProperty("targets_mode", out var tm) && tm.ValueKind == JsonValueKind.String ? (tm.GetString() ?? cfg.TargetsMode) : cfg.TargetsMode,
                            ExpansionQuantile = s3.TryGetProperty("expansion_quantile", out var eq) && eq.TryGetDecimal(out var deq) ? deq : cfg.ExpansionQuantile,
                            GivebackAfterT1R = s3.TryGetProperty("giveback_after_t1_R", out var gb) && gb.TryGetDecimal(out var dgb) ? dgb : cfg.GivebackAfterT1R,
                            MinSlopeTf2 = s3.TryGetProperty("min_slope_tf2", out var ms) && ms.TryGetDecimal(out var dms) ? dms : cfg.MinSlopeTf2,
                            VolZMin = s3.TryGetProperty("volz", out var vz) && vz.TryGetProperty("min", out var vmin) && vmin.TryGetDecimal(out var dvmin) ? dvmin : cfg.VolZMin,
                            VolZMax = s3.TryGetProperty("volz", out var vz2) && vz2.TryGetProperty("max", out var vmax) && vmax.TryGetDecimal(out var dvmax) ? dvmax : cfg.VolZMax,
                        };
                    }
                }
                catch { /* keep prior config on parse errors */ }
            }
        }

        // Public shim for tuning to apply a JSON override at runtime
        public static void ApplyTuningJson(string json)
        {
            S3RuntimeConfig.OverrideFromJson(json);
        }
    }
}
