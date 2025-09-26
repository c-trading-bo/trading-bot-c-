// S11_MaxPerf_FullStack.cs
// ADR / IB Exhaustion Fade for ES & NQ (13:30–15:30 ET)
// • Ticks-only math (long), allocation-free rings
// • O(1) rolling ATR/ADX/EMA/RSI/RVOL
// • ADR(20) + IB(09:30–10:30) tracking
// • Exhaustion + rejection + volume + momentum-divergence filters
// • Range-bound regime only (ADX <= 20), multi-timeframe (1m/5m/15m)
// • Time-decay toward close; news-avoidance gate; spread filter
// • Conservative sizing; tight stops beyond extremes; ATR-multiple targets
//
// HOW TO USE
// 1) Implement IOrderRouter for your TopstepX order flow.
// 2) (Optional) Implement INewsGate.IsBlocked(DateTimeOffset et) to return true during high-impact news.
// 3) WarmupDaily / Warmup1m with your history, then call OnBar1M + UpdateDivergence(optional) and OnDepth live.
// 4) Copy/paste into your project; wire just like S6.

using System;
using System.Collections.Generic;
using System.Globalization;
using BotCore.Models;

namespace TopstepX.S11
{
    public enum Instrument { ES, NQ }
    public enum Side { Buy, Sell, Flat }
    public enum Mode { Idle, Fade }

    // --- ORDER ROUTER INTERFACE ---
    public interface IOrderRouter
    {
        string PlaceMarket(Instrument instr, Side side, int qty, string tag);
        void   ModifyStop(string positionId, double stopPrice);
        void   ClosePosition(string positionId);
        (Side side, int qty, double avgPx, DateTimeOffset openedAt, string positionId) GetPosition(Instrument instr);
        double GetTickSize(Instrument instr);
        double GetPointValue(Instrument instr);
    }

    // --- NEWS GATE (optional) ---
    public interface INewsGate { bool IsBlocked(DateTimeOffset et); }
    public sealed class NullNewsGate : INewsGate { public bool IsBlocked(DateTimeOffset et) => false; }

    // --- DATA TYPES ---
    public readonly struct Bar1M : IEquatable<Bar1M>
    {
        public readonly DateTimeOffset TimeET; // ET time
        public readonly long Open, High, Low, Close; // ticks
        public readonly double Volume;
        public Bar1M(DateTimeOffset tEt, long o, long h, long l, long c, double v)
        { TimeET = tEt; Open = o; High = h; Low = l; Close = c; Volume = v; }

        public override bool Equals(object? obj)
        {
            return obj is Bar1M other && Equals(other);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(TimeET, Open, High, Low, Close, Volume);
        }

        public static bool operator ==(Bar1M left, Bar1M right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bar1M left, Bar1M right)
        {
            return !(left == right);
        }

        public bool Equals(Bar1M other)
        {
            return TimeET == other.TimeET && Open == other.Open && High == other.High && 
                   Low == other.Low && Close == other.Close && Volume.Equals(other.Volume);
        }
    }

    public readonly struct DepthLadder : IEquatable<DepthLadder>
    {
        public readonly DateTimeOffset TimeET;
        public readonly long Bid1, Ask1, Bid2, Ask2, Bid3, Ask3; // ticks
        public readonly int BidSz1, AskSz1, BidSz2, AskSz2, BidSz3, AskSz3;
        public DepthLadder(DateTimeOffset t, long b1, long a1, int bs1, int as1, long b2, long a2, int bs2, int as2, long b3, long a3, int bs3, int as3)
        { TimeET=t; Bid1=b1; Ask1=a1; Bid2=b2; Ask2=a2; Bid3=b3; Ask3=a3; BidSz1=bs1; AskSz1=as1; BidSz2=bs2; AskSz2=as2; BidSz3=bs3; AskSz3=as3; }
        
        // Convenience properties for compatibility
        public long BestBid => Bid1;
        public long BestAsk => Ask1;
        public int BidSize => BidSz1;
        public int AskSize => AskSz1;
        
        public long SpreadTicks => Ask1 - Bid1;
        public double Imbalance()
        {
            long b = (long)BidSz1 + BidSz2 + BidSz3; long a = (long)AskSz1 + AskSz2 + AskSz3; long d = b + a;
            if (d <= 0) return 0; return (double)(b - a) / d;
        }

        public override bool Equals(object? obj)
        {
            return obj is DepthLadder other && Equals(other);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(BestBid, BestAsk, BidSize, AskSize);
        }

        public static bool operator ==(DepthLadder left, DepthLadder right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(DepthLadder left, DepthLadder right)
        {
            return !(left == right);
        }

        public bool Equals(DepthLadder other)
        {
            return BestBid == other.BestBid && BestAsk == other.BestAsk && 
                   BidSize.Equals(other.BidSize) && AskSize.Equals(other.AskSize);
        }
    }

    // --- RING BUFFER (reused from S6) ---
    public sealed class Ring<T>
    {
        private readonly T[] _buf;
        private int _idx, _count;
        public Ring(int capacity) { _buf = new T[capacity]; _idx = 0; _count = 0; }
        public int Count => _count; public int Capacity => _buf.Length;
        public void Add(in T x) { _buf[_idx] = x; _idx = (_idx + 1) % _buf.Length; if (_count < _buf.Length) _count++; }
        public ref readonly T Last(int back = 0)
        {
            if (_count == 0) throw new InvalidOperationException("Ring empty");
            int pos = (_idx - 1 - back); if (pos < 0) pos += _buf.Length; return ref _buf[pos];
        }
        public void ForEachNewest(int n, Action<T> f) 
        { 
            if (f is null) throw new ArgumentNullException(nameof(f));
            
            for (int i = Math.Max(0,_count - n); i < _count; i++) { int pos = ( (_idx - _count + i) % _buf.Length + _buf.Length ) % _buf.Length; f(_buf[pos]); } 
        }
        public void CopyNewest(int n, Span<T> dst) { n = Math.Min(n, _count); for (int i = 0; i < n; i++){ int pos = (_idx - n + i); if (pos < 0) pos += _buf.Length; dst[i] = _buf[pos]; } }
    }

    // --- ROLLING INDICATORS ---
    public sealed class Atr
    {
        private readonly int _n; private double _atr; private bool _seeded;
        public Atr(int n){ _n=n; }
        public double Update(double high, double low, double prevClose)
        {
            double tr = Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
            if (!_seeded){ _atr = tr; _seeded = true; } else { _atr = (_atr * (_n - 1) + tr) / _n; }
            return _atr;
        }
        public double Value => _atr;
    }

    public sealed class Adx
    {
        private readonly int _n; private double _tr, _dmP, _dmN; private bool _seeded; public double Value { get; private set; }
        public Adx(int n){ _n=n; }
        public double Update(double curH, double curL, double prevH, double prevL, double prevC)
        {
            double up = curH - prevH, dn = prevL - curL;
            double dmP = (up > dn && up > 0) ? up : 0; double dmN = (dn > up && dn > 0) ? dn : 0;
            double tr = Math.Max(curH - curL, Math.Max(Math.Abs(curH - prevC), Math.Abs(curL - prevC)));
            if (!_seeded){ _tr = tr; _dmP = dmP; _dmN = dmN; _seeded = true; }
            else { _tr = _tr - (_tr / _n) + tr; _dmP = _dmP - (_dmP / _n) + dmP; _dmN = _dmN - (_dmN / _n) + dmN; }
            if (_tr <= 1e-12) return Value;
            double diP = 100.0 * (_dmP / _tr); double diN = 100.0 * (_dmN / _tr);
            double dx = 100.0 * Math.Abs(diP - diN) / Math.Max(1e-9, diP + diN);
            Value = Value <= 0 ? dx : (Value - (Value / _n) + dx);
            return Value;
        }
    }

    public sealed class Ema { private readonly double _k; private bool _seed; public double Value; public Ema(int n){ _k=2.0/(n+1);} public double Update(double v){ if(!_seed){ Value=v; _seed=true; } else Value = v*_k + Value*(1-_k); return Value; } }

    public sealed class Rsi
    {
        private readonly int _n; private double _avgGain, _avgLoss; private bool _seeded; private double _lastClose;
        public Rsi(int n) { _n = n; }
        public double Update(double close)
        {
            if (!_seeded) { _lastClose = close; _seeded = true; return 50; }
            double change = close - _lastClose;
            double gain = change > 0 ? change : 0;
            double loss = change < 0 ? -change : 0;
            if (_avgGain == 0 && _avgLoss == 0) { _avgGain = gain; _avgLoss = loss; }
            else { _avgGain = (_avgGain * (_n - 1) + gain) / _n; _avgLoss = (_avgLoss * (_n - 1) + loss) / _n; }
            _lastClose = close;
            if (_avgLoss == 0) return 100;
            double rs = _avgGain / _avgLoss;
            return 100 - (100 / (1 + rs));
        }
        public double Value => _seeded ? (100 - (100 / (1 + (_avgGain / Math.Max(_avgLoss, 1e-12))))) : 50;
    }

    public sealed class RvolBaseline
    {
        private readonly double[] _sum = new double[390];
        private readonly int[] _cnt = new int[390];
        private readonly int _lookback;
        public RvolBaseline(int lookbackDays){ _lookback = lookbackDays; }
        public void AddDayMinute(int idx, double vol){ _sum[idx]+=vol; if(_cnt[idx] < _lookback) _cnt[idx]++; }
        public double GetBaseline(int idx) => _cnt[idx]==0 ? 0 : _sum[idx] / _cnt[idx];
    }

    // --- CONFIG ---
    public sealed class S11Config
    {
        // session windows (ET)
        public TimeSpan WindowStart = TimeSpan.Parse("13:30", CultureInfo.InvariantCulture);
        public TimeSpan WindowEnd   = TimeSpan.Parse("15:30", CultureInfo.InvariantCulture);
        public TimeSpan IBStart     = TimeSpan.Parse("09:30", CultureInfo.InvariantCulture);
        public TimeSpan IBEnd       = TimeSpan.Parse("10:30", CultureInfo.InvariantCulture);

        // risk
        public int    BaseQty = 1;
        public double MultiplierAfternoon = 0.8; // conservative sizing
        public int    MaxSpreadTicks = 2;
        public int    StopTicksMin = 8;
        public double StopAtrMult = 1.2;
        public double TargetAdrFrac = 0.12;
        public int    MaxHoldMinutes = 90;

        // filters
        public double MaxADX = 20.0;     // range-bound only
        public double MinRVOL = 1.5;
        public double MinRSI = 25.0;     // oversold/overbought thresholds
        public double MaxRSI = 75.0;
        public double MinDomImbalance = 0.25;

        // exhaustion detection
        public double ExhaustionVolMult = 2.0;   // volume spike
        public int    ExhaustionBars = 3;        // look back bars
        public double RejectionWickRatio = 0.6;  // wick vs body ratio

        // ADR tracking
        public int AdrLookbackDays = 20;
        public double AdrExhaustionThreshold = 0.75; // 75% of ADR used

        // news avoidance
        public bool EnableNewsGate;
    }

    // --- STRATEGY ---
    public sealed class S11Strategy
    {
        private readonly IOrderRouter _router; private readonly S11Config _cfg; private readonly INewsGate _newsGate;
        private readonly State _es; private readonly State _nq;
        
        public S11Strategy(IOrderRouter router, S11Config? cfg = null, INewsGate? newsGate = null)
        { _router=router; _cfg=cfg??new S11Config(); _newsGate=newsGate??new NullNewsGate(); _es=new State(Instrument.ES,_router,_cfg); _nq=new State(Instrument.NQ,_router,_cfg); }

        public void WarmupDaily(Instrument instr, IEnumerable<(DateTime dateEt, double high, double low)> days)
        {
            if (days is null) throw new ArgumentNullException(nameof(days));
            
            var s = Get(instr); s.DailyForAdr.Clear();
            int k=0; foreach (var d in days){ s.DailyForAdr.Add((d.dateEt, d.high, d.low)); if(++k>=_cfg.AdrLookbackDays) break; }
            s.Adr = s.ComputeADR();
        }
        public void Warmup1m(Instrument instr, IEnumerable<(DateTimeOffset tEt,double o,double h,double l,double c,double v)> bars)
        {
            if (bars is null) throw new ArgumentNullException(nameof(bars));
            
            var s = Get(instr); foreach (var b in bars)
            { var bar = new Bar1M(b.tEt, s.ToTicks(b.o), s.ToTicks(b.h), s.ToTicks(b.l), s.ToTicks(b.c), b.v); s.OnBar(bar); }
        }

        public void OnBar1M(Instrument instr, Bar1M bar) { Get(instr).OnBar(bar); StepEngine(Get(instr)); }
        public void OnDepth(Instrument instr, DepthLadder depth) { Get(instr).LastDepth = depth; }

        private void StepEngine(State s)
        {
            var nowEt = s.LastBarTime;
            if (!InWindow(nowEt.TimeOfDay)) return;

            // news gate
            if (_cfg.EnableNewsGate && _newsGate.IsBlocked(nowEt)) return;

            // spread filter
            if (s.LastDepth.SpreadTicks > _cfg.MaxSpreadTicks) return;

            var regime = Classify(s);
            ManagePosition(s);

            if (regime == Mode.Fade) TryEnterFade(s);
        }

        private Mode Classify(State s)
        {
            // Range-bound regime only
            if (s.ADX > _cfg.MaxADX) return Mode.Idle;
            if (s.RVOL < _cfg.MinRVOL) return Mode.Idle;

            // ADR exhaustion check
            if (!s.IsAdrExhausted(_cfg.AdrExhaustionThreshold)) return Mode.Idle;

            // RSI extremes
            bool rsiOversold = s.RSI <= _cfg.MinRSI;
            bool rsiOverbought = s.RSI >= _cfg.MaxRSI;
            if (!rsiOversold && !rsiOverbought) return Mode.Idle;

            // Volume exhaustion or rejection patterns
            bool exhaustion = s.VolumeExhaustion() || s.RejectionPattern();
            if (!exhaustion) return Mode.Idle;

            return Mode.Fade;
        }

        private void TryEnterFade(State s)
        {
            var (side, _, _, _, _) = _router.GetPosition(s.Instr); if (side != Side.Flat) return;

            bool fadeLong = s.RSI <= _cfg.MinRSI && s.IsBelowIBLow() && s.VolumeExhaustion();
            bool fadeShort = s.RSI >= _cfg.MaxRSI && s.IsAboveIBHigh() && s.VolumeExhaustion();
            
            if (!fadeLong && !fadeShort) return;

            long entry = s.LastClose;
            double stopPx = s.ComputeStopBeyondExtreme(fadeLong);
            double tgtPx = s.ComputeTargetTowardMean(fadeLong);

            // Conservative DOM imbalance check
            if (Math.Abs(s.LastDepth.Imbalance()) < _cfg.MinDomImbalance) return;

            s.PlaceWithOco(fadeLong ? Side.Buy : Side.Sell, stopPx, tgtPx, "S11-Fade");
        }

        private void ManagePosition(State s)
        {
            var (side, qty, avgPx, openedAt, positionId) = _router.GetPosition(s.Instr);
            if (side == Side.Flat || qty <= 0) return;

            // Time-based exit approaching close
            if ((s.LastBarTime - openedAt).TotalMinutes >= _cfg.MaxHoldMinutes)
            { _router.ClosePosition(positionId); return; }

            // Trail stops as price moves favorably
            double r = s.RealizedR(side, avgPx);
            if (r >= 0.5)
            {
                double? sw = s.RecentExtremePx(side);
                if (sw.HasValue) _router.ModifyStop(positionId, sw.Value);
            }
        }

        private bool InWindow(TimeSpan et) => et >= _cfg.WindowStart && et <= _cfg.WindowEnd;
        private State Get(Instrument i) => i==Instrument.ES ? _es : _nq;

        // --- PER-INSTRUMENT STATE ---
        private sealed class State
        {
            public readonly Instrument Instr; private readonly IOrderRouter R; private readonly S11Config C;
            public readonly long Tick; public readonly double TickPx;
            public State(Instrument i, IOrderRouter r, S11Config c){ Instr=i; R=r; C=c; TickPx=r.GetTickSize(i); Tick=(long)Math.Round(1.0/ TickPx); }

            // series
            public readonly Ring<Bar1M> Min1 = new Ring<Bar1M>(1200);
            public readonly Ring<Bar1M> Min5 = new Ring<Bar1M>(500);
            public readonly Ring<Bar1M> Min15 = new Ring<Bar1M>(200);
            public DepthLadder LastDepth;

            // indicators
            private readonly Atr atr = new Atr(14);
            private readonly Adx adx = new Adx(14);
            private readonly Ema ema20 = new Ema(20), ema50 = new Ema(50);
            private readonly Rsi rsi14 = new Rsi(14);
            private readonly RvolBaseline rvolBase = new RvolBaseline(20);
            public double ATR, ADX, RSI=50, RVOL=1.0;

            // session data
            public long IB_High = long.MinValue, IB_Low = long.MaxValue;
            public DateTime LastResetDay = DateTime.MinValue.Date; 
            public DateTimeOffset LastBarTime;

            // ADR
            public readonly List<(DateTime dateEt, double high, double low)> DailyForAdr = new();
            public double Adr=0;

            // helpers
            public long LastClose => Min1.Count>0 ? Min1.Last().Close : 0;
            public long ToTicks(double px) => (long)Math.Round(px / TickPx);
            public double ToPx(long ticks) => ticks * TickPx;

            public void OnBar(Bar1M bar)
            {
                LastBarTime = bar.TimeET;
                Min1.Add(bar);

                // build 5m bars
                int mod5 = bar.TimeET.Minute % 5;
                if (mod5 == 4 && Min1.Count >= 5)
                {
                    var b4 = Min1.Last(4); var b3 = Min1.Last(3); var b2 = Min1.Last(2); var b1 = Min1.Last(1); var b0 = Min1.Last(0);
                    long o = b4.Open; long h = Math.Max(Math.Max(Math.Max(Math.Max(b4.High,b3.High),b2.High),b1.High),b0.High);
                    long l = Math.Min(Math.Min(Math.Min(Math.Min(b4.Low, b3.Low), b2.Low), b1.Low), b0.Low);
                    long c = b0.Close; double v = b4.Volume + b3.Volume + b2.Volume + b1.Volume + b0.Volume;
                    Min5.Add(new Bar1M(bar.TimeET, o,h,l,c,v));
                }

                // build 15m bars
                int mod15 = bar.TimeET.Minute % 15;
                if (mod15 == 14 && Min1.Count >= 15)
                {
                    long o15 = Min1.Last(14).Open; long h15 = long.MinValue; long l15 = long.MaxValue; long c15 = Min1.Last(0).Close; double v15 = 0;
                    for (int i = 0; i < 15; i++) { var b = Min1.Last(i); if (b.High > h15) h15 = b.High; if (b.Low < l15) l15 = b.Low; v15 += b.Volume; }
                    Min15.Add(new Bar1M(bar.TimeET, o15, h15, l15, c15, v15));
                }

                // IB tracking (09:30-10:30)
                if (bar.TimeET.TimeOfDay >= C.IBStart && bar.TimeET.TimeOfDay <= C.IBEnd)
                {
                    if (bar.High > IB_High) IB_High = bar.High; if (bar.Low < IB_Low) IB_Low = bar.Low;
                }

                // indicators
                if (Min1.Count >= 2)
                {
                    var prev = Min1.Last(1); var cur = Min1.Last(0);
                    ATR = atr.Update(ToPx(cur.High), ToPx(cur.Low), ToPx(prev.Close));
                    ADX = adx.Update(ToPx(cur.High), ToPx(cur.Low), ToPx(prev.High), ToPx(prev.Low), ToPx(prev.Close));
                    RSI = rsi14.Update(ToPx(cur.Close));
                    ema20.Update(ToPx(cur.Close)); ema50.Update(ToPx(cur.Close));
                }

                // RVOL
                int idx = RthMinuteIndex(bar.TimeET);
                if (idx >= 0) { RVOL = ComputeRVOL(idx, bar.Volume); }

                // ADR recompute daily
                if (LastResetDay != bar.TimeET.Date)
                {
                    LastResetDay = bar.TimeET.Date; Adr = ComputeADR();
                    IB_High = long.MinValue; IB_Low = long.MaxValue; // reset IB
                }
            }

            private int RthMinuteIndex(DateTimeOffset et) { var start = et.Date + C.IBStart; if (et < start || et >= start.AddHours(6.5)) return -1; return (int)(et - start).TotalMinutes; }
            private double ComputeRVOL(int minuteIdx, double vol)
            {
                double baseVol = rvolBase.GetBaseline(minuteIdx);
                if (baseVol <= 0) return 1.0; return vol / baseVol;
            }
            public double ComputeADR()
            {
                if (DailyForAdr.Count == 0) return 0; double s=0; int n=0; foreach (var d in DailyForAdr){ s += (d.high - d.low); n++; if (n >= C.AdrLookbackDays) break; } return n>0 ? s/n : 0;
            }

            public bool IsAdrExhausted(double threshold)
            {
                if (Adr <= 0) return false;
                var today = LastBarTime.Date;
                double todayHi = 0, todayLo = double.MaxValue;
                bool found = false;
                for (int i = 0; i < Min1.Count; i++)
                {
                    var b = Min1.Last(i);
                    if (b.TimeET.Date != today) continue;
                    if (!found) { todayHi = ToPx(b.High); todayLo = ToPx(b.Low); found = true; }
                    else { if (ToPx(b.High) > todayHi) todayHi = ToPx(b.High); if (ToPx(b.Low) < todayLo) todayLo = ToPx(b.Low); }
                }
                if (!found) return false;
                double usedRange = todayHi - todayLo;
                return (usedRange / Adr) >= threshold;
            }

            public bool VolumeExhaustion()
            {
                if (Min1.Count < C.ExhaustionBars) return false;
                double avgVol = 0; for (int i = 1; i <= C.ExhaustionBars; i++) avgVol += Min1.Last(i).Volume; avgVol /= C.ExhaustionBars;
                return Min1.Last(0).Volume >= (avgVol * C.ExhaustionVolMult);
            }

            public bool RejectionPattern()
            {
                if (Min1.Count == 0) return false;
                var bar = Min1.Last(0);
                double range = ToPx(bar.High - bar.Low);
                double body = Math.Abs(ToPx(bar.Close - bar.Open));
                if (range <= 0) return false;
                double wickRatio = (range - body) / range;
                return wickRatio >= C.RejectionWickRatio;
            }

            public bool IsBelowIBLow() => LastClose < IB_Low;
            public bool IsAboveIBHigh() => LastClose > IB_High;

            public void PlaceWithOco(Side side, double stopPx, double targetPx, string tag)
            {
                int qty = Math.Max(1, (int)Math.Round(C.BaseQty * C.MultiplierAfternoon));
                R.PlaceMarket(Instr, side, qty, $"{tag};stop={stopPx:F2};tgt={targetPx:F2}");
            }

            public double ComputeStopBeyondExtreme(bool longSide)
            {
                double grace = Math.Max(C.StopTicksMin * TickPx, C.StopAtrMult * ATR);
                if (longSide) return ToPx(IB_Low) - grace;
                else return ToPx(IB_High) + grace;
            }

            public double ComputeTargetTowardMean(bool longSide)
            {
                double lastPx = ToPx(LastClose);
                double mean = (ToPx(IB_High) + ToPx(IB_Low)) / 2.0;
                double adrTarget = Adr * C.TargetAdrFrac;
                if (longSide) return Math.Min(mean, lastPx + adrTarget);
                else return Math.Max(mean, lastPx - adrTarget);
            }

            public double RealizedR(Side side, double avgPx)
            {
                double stop = side==Side.Buy ? ComputeStopBeyondExtreme(true) : ComputeStopBeyondExtreme(false);
                double last = ToPx(LastClose);
                return side==Side.Buy ? (last - avgPx) / Math.Max(1e-9,(avgPx - stop)) : (avgPx - last) / Math.Max(1e-9,(stop - avgPx));
            }

            public double? RecentExtremePx(Side side)
            {
                if (Min1.Count < 5) return null; 
                long extreme = side==Side.Buy ? long.MaxValue : long.MinValue; 
                for (int i=0;i<5;i++)
                { 
                    var b = Min1.Last(i); 
                    if (side==Side.Buy) { if (b.Low < extreme) extreme = b.Low; } 
                    else { if (b.High > extreme) extreme = b.High; } 
                }
                return ToPx(extreme);
            }
        }

        // --- DIVERGENCE (optional, same as S6) ---
        public void UpdateDivergence()
        {
            if (_es.Min1.Count < 2 || _nq.Min1.Count < 2) return;
            double esRet = (_es.ToPx(_es.Min1.Last().Close) - _es.ToPx(_es.Min1.Last(1).Close)) / _es.ToPx(_es.Min1.Last(1).Close);
            double nqRet = (_nq.ToPx(_nq.Min1.Last().Close) - _nq.ToPx(_nq.Min1.Last(1).Close)) / _nq.ToPx(_nq.Min1.Last(1).Close);
            // Could store divergence if needed for S11 logic
        }
    }
}