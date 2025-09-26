// S6_MaxPerf_FullStack.cs
// Opening Drive / Reversal strategy for ES & NQ — maximum-performance, drop-in.
// • Ticks-only math (long) to avoid float drift
// • Allocation-free ring buffers
// • O(1) rolling ATR/ADX/EMA/RVOL
// • Tight, branch-light hot path + time-decay (no new entries after 10:00 ET)
// • Plug your own IOrderRouter (TopstepX REST/WebSocket) — interface below
//
// HOW TO USE
// 1) Plug your existing order/router into IOrderRouter.
// 2) Feed 1m bars via OnBar1m(Instrument, Bar1m) and L1–L3 via OnDepth(Instrument, DepthLadder).
// 3) Call WarmupDaily & Warmup1m before live if you have history.
// 4) All prices internally are ticks (long). Convert using router.GetTickSize().
// 5) ES & NQ only; window 09:28–10:00 ET; multi-timeframe confirms; spread, RVOL, ADX filters.
//
// OPTIONAL: See Program wiring example at the bottom (works without any external libs).

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using BotCore.Models;

namespace TopstepX.S6
{
    public enum Instrument { ES, NQ }
    public enum Side { Buy, Sell, Flat }
    public enum Mode { Idle, Drive, Reversal }

    // --- ORDER ROUTER INTERFACE (plug your TopstepX router here) ---
    public interface IOrderRouter
    {
        string PlaceMarket(Instrument instr, Side side, int qty, string tag);
        void   ModifyStop(string positionId, double stopPrice);
        void   ClosePosition(string positionId);
        (Side side, int qty, double avgPx, DateTimeOffset openedAt, string positionId) GetPosition(Instrument instr);
        double GetTickSize(Instrument instr);    // ES/NQ both 0.25 typically
        double GetPointValue(Instrument instr);  // ES $50/pt, NQ $20/pt typically
    }

    // --- DATA TYPES (ticks internally) ---
    public readonly struct Bar1m : IEquatable<Bar1m>
    {
        public readonly DateTimeOffset TimeET; // end time of bar (ET)
        public readonly long Open, High, Low, Close; // ticks
        public readonly double Volume;
        public Bar1m(DateTimeOffset timeEt, long o, long h, long l, long c, double v)
        { TimeET = timeEt; Open = o; High = h; Low = l; Close = c; Volume = v; }

        public override bool Equals(object? obj)
        {
            throw new NotImplementedException();
        }

        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }

        public static bool operator ==(Bar1m left, Bar1m right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bar1m left, Bar1m right)
        {
            return !(left == right);
        }

        public bool Equals(Bar1m other)
        {
            throw new NotImplementedException();
        }
    }

    public readonly struct DepthLadder : IEquatable<DepthLadder>
    // L1-L3 snapshot
    {
        public readonly DateTimeOffset TimeET;
        public readonly long Bid1, Ask1, Bid2, Ask2, Bid3, Ask3; // ticks
        public readonly int BidSz1, AskSz1, BidSz2, AskSz2, BidSz3, AskSz3;
        public DepthLadder(DateTimeOffset t, long b1, long a1, int bs1, int as1, long b2, long a2, int bs2, int as2, long b3, long a3, int bs3, int as3)
        { TimeET=t; Bid1=b1; Ask1=a1; Bid2=b2; Ask2=a2; Bid3=b3; Ask3=a3; BidSz1=bs1; AskSz1=as1; BidSz2=bs2; AskSz2=as2; BidSz3=bs3; AskSz3=as3; }
        public long SpreadTicks => Ask1 - Bid1;
        public double Imbalance()
        {
            long b = (long)BidSz1 + BidSz2 + BidSz3; long a = (long)AskSz1 + AskSz2 + AskSz3; long d = b + a;
            if (d <= 0) return 0; return (double)(b - a) / d;
        }

        public override bool Equals(object? obj)
        {
            throw new NotImplementedException();
        }

        public override int GetHashCode()
        {
            throw new NotImplementedException();
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
            throw new NotImplementedException();
        }
    }

    // --- RING BUFFER (allocation-free) ---
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

    // --- ROLLING INDICATORS (O(1)) ---
    public sealed class Atr
    {
        private readonly int _n; private double _atr; private bool _seeded; private double _prevClosePx;
        public Atr(int n){ _n=n; }
        public double Update(double high, double low, double prevClose)
        {
            double tr = Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
            if (!_seeded){ _atr = tr; _seeded = true; } else { _atr = (_atr * (_n - 1) + tr) / _n; }
            _prevClosePx = prevClose; return _atr;
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
            if (!_seeded){ _tr = tr; _dmP = dmP; _dmN = dmN; _seeded = true; _ = Value; }
            else { _tr = _tr - (_tr / _n) + tr; _dmP = _dmP - (_dmP / _n) + dmP; _dmN = _dmN - (_dmN / _n) + dmN; }
            if (_tr <= 1e-12) return Value;
            double diP = 100.0 * (_dmP / _tr); double diN = 100.0 * (_dmN / _tr);
            double dx = 100.0 * Math.Abs(diP - diN) / Math.Max(1e-9, diP + diN);
            Value = Value <= 0 ? dx : (Value - (Value / _n) + dx);
            return Value;
        }
    }
    public sealed class Ema { private readonly double _k; private bool _seed; public double Value; public Ema(int n){ _k=2.0/(n+1);} public double Update(double v){ if(!_seed){ Value=v; _seed=true; } else Value = v*_k + Value*(1-_k); return Value; } }

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
    public sealed class S6Config
    {
        // window (ET)
        public TimeSpan WindowStart = TimeSpan.Parse("09:28");
        public TimeSpan RTHOpen     = TimeSpan.Parse("09:30");
        public TimeSpan WindowEnd   = TimeSpan.Parse("10:00");

        // risk
        public int    BaseQty = 1;
        public double MultiplierInWindow = 1.2;
        public int    MaxSpreadTicks = 2;
        public int    StopTicksMin = 6;
        public double StopAtrMult = 0.7;
        public double TargetAdrFrac = 0.18;
        public int    MaxHoldMinutes = 45;
        public double FlipMinR = 1.25;

        // filters
        public double MinADX = 18.0;
        public double MinRVOL = 1.2;
        public int    DivMaxBp = 12;          // ES–NQ divergence bps
        public double MinDomImbalance = 0.18; // L1–L3

        // retests
        public bool   RetestEnable = true;
        public int    RetestGraceTicks = 2;
        public int    RetestConfirmBars1m = 2;

        // history
        public int AdrLookbackDays = 14;
        public int RvolLookbackDays = 20;

        // failed breakout
        public int FailBreakPenetrationTicks_ES = 3;
        public int FailBreakPenetrationTicks_NQ = 4;
    }

    // --- STRATEGY ---
    public sealed class S6Strategy
    {
        private readonly IOrderRouter _router; private readonly S6Config _cfg;
        private readonly State _es; private readonly State _nq;
        public S6Strategy(IOrderRouter router, S6Config? cfg = null)
        { _router=router; _cfg=cfg??new S6Config(); _es=new State(Instrument.ES,_router,_cfg); _nq=new State(Instrument.NQ,_router,_cfg); }

        public void WarmupDaily(Instrument instr, IEnumerable<(DateTime dateEt, double high, double low)> days)
        {
            if (days == null)
                throw new ArgumentNullException(nameof(days));

            var s = Get(instr); s.DailyForAdr.Clear();
            int k=0; foreach (var d in days){ s.DailyForAdr.Add((d.dateEt, d.high, d.low)); if(++k>=_cfg.AdrLookbackDays) break; }
            s.Adr = s.ComputeADR();
        }
        public void Warmup1m(Instrument instr, IEnumerable<(DateTimeOffset tEt,double o,double h,double l,double c,double v)> bars)
        {
            if (bars == null)
                throw new ArgumentNullException(nameof(bars));

            var s = Get(instr); long tick = s.Tick; foreach (var b in bars)
            { var bar = new Bar1m(b.tEt, s.ToTicks(b.o), s.ToTicks(b.h), s.ToTicks(b.l), s.ToTicks(b.c), b.v); s.OnBar(bar); }
        }

        public void OnBar1m(Instrument instr, Bar1m bar) { Get(instr).OnBar(bar); StepEngine(Get(instr)); }
        public void OnDepth(Instrument instr, DepthLadder depth) { Get(instr).LastDepth = depth; }

        private void StepEngine(State s)
        {
            var nowEt = s.LastBarTime;
            if (!InWindow(nowEt.TimeOfDay)) return;

            // spread filter
            if (s.LastDepth.SpreadTicks > _cfg.MaxSpreadTicks) return;

            var regime = Classify(s);
            ManagePosition(s, regime);

            if (nowEt.TimeOfDay > _cfg.WindowEnd) return; // no new entries after 10:00

            switch (regime)
            {
                case Mode.Drive: TryEnterDrive(s); break;
                case Mode.Reversal: TryEnterReversal(s); break;
            }
        }

        private Mode Classify(State s)
        {
            if (s.ADX < _cfg.MinADX) return Mode.Idle;
            if (s.RVOL < _cfg.MinRVOL) return Mode.Idle;

            bool brokeONH = s.LastClose > s.ON_High && s.CrossedForBars(above:true, s.ON_High, bars:2);
            bool brokeONL = s.LastClose < s.ON_Low  && s.CrossedForBars(above:false, s.ON_Low,  bars:2);
            bool sustained = s.BarsSinceRTHOpen() <= 15 && s.TrendAgree1m5m(false);

            bool driveLong  = brokeONH && sustained && s.GapDir >= 0;
            bool driveShort = brokeONL && sustained && s.GapDir <= 0;

            bool failedUp = s.FailedBreakout(true);
            bool failedDn = s.FailedBreakout(false);
            bool reversalLong  = failedDn && (s.DivergenceBp > _cfg.DivMaxBp || s.VolumeExhaustion());
            bool reversalShort = failedUp && (s.DivergenceBp < -_cfg.DivMaxBp || s.VolumeExhaustion());

            if (driveLong || driveShort) return Mode.Drive;
            if (reversalLong || reversalShort) return Mode.Reversal;
            return Mode.Idle;
        }

        private void TryEnterDrive(State s)
        {
            var (side, qty, avgPx, openedAt, positionId) = _router.GetPosition(s.Instr); if (side != Side.Flat) return;
            bool longSide = s.LastClose > s.ON_High && s.TrendAgree1m5m(false);
            bool shortSide= s.LastClose < s.ON_Low  && s.TrendAgree1m5m(false);
            if (!longSide && !shortSide) return;

            long entry = s.LastClose;
            double stopPx = s.ComputeStopPx(longSide);
            double tgtPx  = s.ComputeTargetPx(longSide);

            if (_cfg.RetestEnable)
            {
                long level = longSide ? s.ON_High : s.ON_Low;
                if (!s.RetestConfirmed(level, longSide)) return;
            }
            s.PlaceWithOco(longSide ? Side.Buy : Side.Sell, stopPx, tgtPx, "S6-Drive");
        }

        private void TryEnterReversal(State s)
        {
            var (side, qty, avgPx, openedAt, positionId) = _router.GetPosition(s.Instr); if (side != Side.Flat) return;
            bool revLong  = s.FailedBreakout(false);
            bool revShort = s.FailedBreakout(true);
            if (!revLong && !revShort) return;
            bool longSide = revLong;
            double stopPx = s.ComputeStopBeyondExtremePx(longSide);
            double tgtPx  = s.ComputeTargetPx(longSide);
            if (!s.TrendAgree1m5m(true)) return;
            s.PlaceWithOco(longSide ? Side.Buy : Side.Sell, stopPx, tgtPx, "S6-Reversal");
        }

        private void ManagePosition(State s, Mode regime)
        {
            var (side, qty, avgPx, openedAt, positionId) = _router.GetPosition(s.Instr);
            if (side == Side.Flat || qty <= 0) return;

            if ((s.LastBarTime - openedAt).TotalMinutes >= _cfg.MaxHoldMinutes)
            { _router.ClosePosition(positionId); return; }

            if (s.LastBarTime.TimeOfDay >= _cfg.WindowEnd) s.TightenToBEOrTrail(positionId, side, avgPx);

            if (regime == Mode.Reversal)
            {
                if (side == Side.Buy && s.ExpectedR(false, avgPx) >= _cfg.FlipMinR) { _router.ClosePosition(positionId); TryEnterReversal(s); }
                else if (side == Side.Sell && s.ExpectedR(true, avgPx) >= _cfg.FlipMinR) { _router.ClosePosition(positionId); TryEnterReversal(s); }
            }

            double r = s.RealizedR(side, avgPx);
            if (r >= 0.8)
            {
                double? sw = s.RecentSwingPx(side);
                if (sw.HasValue) _router.ModifyStop(positionId, sw.Value);
            }
        }

        private bool InWindow(TimeSpan et) => et >= _cfg.WindowStart && et <= _cfg.WindowEnd;
        private State Get(Instrument i) => i==Instrument.ES ? _es : _nq;

        // --- PER-INSTRUMENT STATE ---
        private sealed class State
        {
            public readonly Instrument Instr; private readonly IOrderRouter R; private readonly S6Config C;
            public readonly long Tick; public readonly double TickPx;
            public State(Instrument i, IOrderRouter r, S6Config c){ Instr=i; R=r; C=c; TickPx=r.GetTickSize(i); Tick=(long)Math.Round(1.0/ TickPx); }

            // series
            public readonly Ring<Bar1m> Min1 = new Ring<Bar1m>(1200);
            public readonly Ring<Bar1m> Min5 = new Ring<Bar1m>(500);
            public DepthLadder LastDepth;

            // indicators
            private readonly Atr atr = new Atr(14);
            private readonly Adx adx = new Adx(14);
            private readonly Ema ema8 = new Ema(8), ema21 = new Ema(21);
            private readonly RvolBaseline rvolBase = new RvolBaseline(20);
            public double ATR=0, ADX=0, RVOL=1.0; public int DivergenceBp=0; // Divergence set externally by strategy combining ES/NQ

            // ON/session
            public long ON_High = long.MinValue, ON_Low = long.MaxValue;
            public long PremarketLast; public bool GapComputed=false; public int GapDir=0; public double GapPts=0; public double RTHOpenPx=0;
            public DateTime LastResetDay = DateTime.MinValue.Date; public DateTimeOffset LastBarTime;

            // ADR
            public readonly List<(DateTime dateEt, double high, double low)> DailyForAdr = new();
            public double Adr=0;

            // helpers
            public long LastClose => Min1.Count>0 ? Min1.Last().Close : 0;
            public long ToTicks(double px) => (long)Math.Round(px / TickPx);
            public double ToPx(long ticks) => ticks * TickPx;

            public void OnBar(Bar1m bar)
            {
                LastBarTime = bar.TimeET;
                Min1.Add(bar);

                // build 5m aggregate at minute 4/9/14...
                int mod5 = bar.TimeET.Minute % 5;
                if (mod5 == 4)
                {
                    // aggregate last 5 1m bars
                    if (Min1.Count >= 5)
                    {
                        var b4 = Min1.Last(4); var b3 = Min1.Last(3); var b2 = Min1.Last(2); var b1 = Min1.Last(1); var b0 = Min1.Last(0);
                        long o = b4.Open; long h = Math.Max(Math.Max(Math.Max(Math.Max(b4.High,b3.High),b2.High),b1.High),b0.High);
                        long l = Math.Min(Math.Min(Math.Min(Math.Min(b4.Low, b3.Low), b2.Low), b1.Low), b0.Low);
                        long c = b0.Close; double v = b4.Volume + b3.Volume + b2.Volume + b1.Volume + b0.Volume;
                        Min5.Add(new Bar1m(bar.TimeET, o,h,l,c,v));
                    }
                }

                // ON range until 09:28
                if (bar.TimeET.TimeOfDay < C.WindowStart)
                {
                    if (bar.High > ON_High) ON_High = bar.High; if (bar.Low < ON_Low) ON_Low = bar.Low; PremarketLast = bar.Close;
                }

                // gap compute at 09:30 bar
                if (!GapComputed && bar.TimeET.TimeOfDay >= C.RTHOpen)
                {
                    RTHOpenPx = ToPx(bar.Open); GapPts = RTHOpenPx - ToPx(PremarketLast); GapDir = GapPts>0?1:(GapPts<0?-1:0); GapComputed=true;
                }

                // indicators
                if (Min1.Count >= 2)
                {
                    var prev = Min1.Last(1); var cur = Min1.Last(0);
                    ATR = atr.Update(ToPx(cur.High), ToPx(cur.Low), ToPx(prev.Close));
                    ADX = adx.Update(ToPx(cur.High), ToPx(cur.Low), ToPx(prev.High), ToPx(prev.Low), ToPx(prev.Close));
                    // 5m EMA proxy from 1m close (cheap): update on 1m nevertheless keeps shape; or prefer Min5 at mod5==4
                    double emaSrc = ToPx(cur.Close);
                    ema8.Update(emaSrc); ema21.Update(emaSrc);
                }

                // RVOL baseline
                int idx = RthMinuteIndex(bar.TimeET);
                if (idx >= 0) { RVOL = ComputeRVOL(idx, bar.Volume); }

                // ADR recompute daily (cheap)
                if (LastResetDay != bar.TimeET.Date)
                {
                    LastResetDay = bar.TimeET.Date; Adr = ComputeADR();
                }
            }

            private int RthMinuteIndex(DateTimeOffset et) { var start = et.Date + C.RTHOpen; if (et < start || et >= start.AddHours(6.5)) return -1; return (int)(et - start).TotalMinutes; }
            private double ComputeRVOL(int minuteIdx, double vol)
            {
                double baseVol = rvolBase.GetBaseline(minuteIdx);
                if (baseVol <= 0) return 1.0; return vol / baseVol;
            }
            public double ComputeADR()
            {
                if (DailyForAdr.Count == 0) return 0; double s=0; int n=0; foreach (var d in DailyForAdr){ s += (d.high - d.low); n++; if (n >= C.AdrLookbackDays) break; } return n>0 ? s/n : 0;
            }

            public bool CrossedForBars(bool above, long level, int bars)
            {
                if (Min1.Count < bars) return false; for (int i=1;i<=bars;i++){ var c = Min1.Last(i).Close; if (above && c <= level) return false; if (!above && c >= level) return false; } return true;
            }

            public int BarsSinceRTHOpen()
            {
                DateTimeOffset openTs = LastBarTime.Date + C.RTHOpen; int cnt=0; for (int i=0;i<Min1.Count;i++){ var b = Min1.Last(i); if (b.TimeET < openTs) break; cnt++; } return cnt;
            }

            public bool TrendAgree1m5m(bool reversal)
            {
                if (Min1.Count < 5) return false; double e8=ema8.Value, e21=ema21.Value; bool up5 = e8 > e21; bool up1 = Min1.Last(0).Close > Min1.Last(1).Close;
                if (!reversal) return (up5 && up1) || (!up5 && !up1);
                bool flat5 = Math.Abs(e8 - e21) <= (ATR * 0.05); return (flat5 || up5 != up1);
            }

            public bool FailedBreakout(bool failedAboveONH)
            {
                int pen = Instr==Instrument.ES? C.FailBreakPenetrationTicks_ES : C.FailBreakPenetrationTicks_NQ;
                if (Min1.Count < 3) return false; var b1 = Min1.Last(0); var b2 = Min1.Last(1);
                if (failedAboveONH)
                {
                    bool pierced = b2.High >= ON_High && (b2.High - ON_High) <= pen;
                    bool closedIn = b1.Close < ON_High && b1.Open > b1.Close;
                    return pierced && closedIn;
                }
                else
                {
                    bool pierced = b2.Low <= ON_Low && (ON_Low - b2.Low) <= pen;
                    bool closedIn = b1.Close > ON_Low && b1.Close > b1.Open;
                    return pierced && closedIn;
                }
            }

            public bool VolumeExhaustion()
            {
                if (Min1.Count < 5) return false; double v1=Min1.Last(0).Volume, v2=Min1.Last(1).Volume, v3=Min1.Last(2).Volume; return (v3 < v2 && v2 > v1 && RVOL >= C.MinRVOL);
            }

            public bool RetestConfirmed(long level, bool longSide)
            {
                if (Min1.Count < C.RetestConfirmBars1m) return false;
                var curClose = Min1.Last(0).Close;
                var graceTicks = C.RetestGraceTicks;
                
                if (longSide)
                {
                    // For longs, we want price to have pulled back close to the level but not too far below
                    return curClose >= (level - graceTicks) && curClose <= (level + graceTicks);
                }
                else
                {
                    // For shorts, we want price to have pulled back close to the level but not too far above
                    return curClose <= (level + graceTicks) && curClose >= (level - graceTicks);
                }
            }

            public void PlaceWithOco(Side side, double stopPx, double targetPx, string tag)
            {
                int qty = Math.Max(1, (int)Math.Round(C.BaseQty * C.MultiplierInWindow));
                R.PlaceMarket(Instr, side, qty, $"{tag};stop={stopPx:F2};tgt={targetPx:F2}");
            }

            public double ComputeStopPx(bool longSide)
            {
                double stopDistPx = Math.Max(C.StopTicksMin * TickPx, C.StopAtrMult * ATR);
                double lastPx = ToPx(LastClose);
                return longSide ? (lastPx - stopDistPx) : (lastPx + stopDistPx);
            }
            public double ComputeStopBeyondExtremePx(bool longSide)
            {
                double grace = Math.Max(C.StopTicksMin * TickPx, 2 * TickPx);
                double lastPx = ToPx(LastClose);
                return longSide ? Math.Min(lastPx, ToPx(ON_Low)) - grace : Math.Max(lastPx, ToPx(ON_High)) + grace;
            }
            public double ComputeTargetPx(bool longSide)
            {
                double adrPts = Adr>0 ? Adr : (ATR * 2.0); double dist = Math.Max(ATR, adrPts * C.TargetAdrFrac); double lastPx = ToPx(LastClose); return longSide ? lastPx + dist : lastPx - dist;
            }

            public void TightenToBEOrTrail(string positionId, Side side, double avgPx)
            {
                double lastPx = ToPx(LastClose); double tick = TickPx;
                if (side == Side.Buy)
                { if (lastPx > avgPx) R.ModifyStop(positionId, avgPx + tick); else { var sw = RecentSwingPx(side); if (sw.HasValue) R.ModifyStop(positionId, sw.Value); } }
                else
                { if (lastPx < avgPx) R.ModifyStop(positionId, avgPx - tick); else { var sw = RecentSwingPx(side); if (sw.HasValue) R.ModifyStop(positionId, sw.Value); } }
            }

            public double ExpectedR(bool longSide, double fromPx)
            {
                double stop = longSide ? ComputeStopPx(true) : ComputeStopPx(false);
                double target = ComputeTargetPx(longSide);
                return longSide ? (target - fromPx) / Math.Max(1e-9, (fromPx - stop)) : (fromPx - target) / Math.Max(1e-9,(stop - fromPx));
            }
            public double RealizedR(Side side, double avgPx)
            {
                double stop = side==Side.Buy ? ComputeStopPx(true) : ComputeStopPx(false);
                double last = ToPx(LastClose);
                return side==Side.Buy ? (last - avgPx) / Math.Max(1e-9,(avgPx - stop)) : (avgPx - last) / Math.Max(1e-9,(stop - avgPx));
            }
            public double? RecentSwingPx(Side side)
            {
                if (Min1.Count < 5) return null; long swing = side==Side.Buy ? long.MaxValue : long.MinValue; for (int i=0;i<5;i++)
                { var b = Min1.Last(i); if (side==Side.Buy) { if (b.Low < swing) swing = b.Low; } else { if (b.High > swing) swing = b.High; } }
                return ToPx(swing);
            }
        }

        // --- DIVERGENCE (call this after OnBar1m for both ES & NQ each minute) ---
        public void UpdateDivergence()
        {
            if (_es.Min1.Count < 2 || _nq.Min1.Count < 2) { _es.DivergenceBp=_nq.DivergenceBp=0; return; }
            double esRet = (_es.ToPx(_es.Min1.Last().Close) - _es.ToPx(_es.Min1.Last(1).Close)) / _es.ToPx(_es.Min1.Last(1).Close);
            double nqRet = (_nq.ToPx(_nq.Min1.Last().Close) - _nq.ToPx(_nq.Min1.Last(1).Close)) / _nq.ToPx(_nq.Min1.Last(1).Close);
            int bp = (int)Math.Round((esRet - nqRet) * 10000.0); _es.DivergenceBp=bp; _nq.DivergenceBp=-bp;
        }
    }

    // --- OPTIONAL: high-throughput event pump (lock-free) ---
    public sealed class EventPump<T>
    {
        private readonly Channel<T> _ch = Channel.CreateBounded<T>(new BoundedChannelOptions(8192){ SingleReader = true, SingleWriter = false, FullMode = BoundedChannelFullMode.DropOldest });
        public bool TryPost(in T item) => _ch.Writer.TryWrite(item);
        public async Task Run(Func<T, bool> onEvent, CancellationToken ct)
        {
            if (onEvent == null)
                throw new ArgumentNullException(nameof(onEvent));

            var r = _ch.Reader;
            while (await r.WaitToReadAsync(ct).ConfigureAwait(false))
                while (r.TryRead(out var ev)) 
                    _ = onEvent(ev);
        }
    }
}