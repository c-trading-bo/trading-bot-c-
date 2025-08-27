using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Models;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent
{
    internal sealed class PaperBroker
    {
        private readonly SupervisorAgent.StatusService _status;
        private readonly ILogger _log;
        private readonly ConcurrentDictionary<string, PositionState> _pos = new(StringComparer.OrdinalIgnoreCase);
        private readonly ConcurrentDictionary<string, PendingOrder> _pending = new(StringComparer.OrdinalIgnoreCase);

        private sealed class PositionState
        {
            public string Symbol { get; init; } = "";
            public int Qty { get; set; }
            public decimal Avg { get; set; }
            public decimal? Stop { get; set; }
            public decimal? Target { get; set; }
            public decimal RealizedPnl { get; set; }
            public decimal UnrealizedPnl { get; set; }
        }
        private sealed class PendingOrder
        {
            public string Symbol { get; init; } = "";
            public string Side { get; init; } = "BUY"; // BUY/SELL
            public int Size { get; init; }
            public decimal Entry { get; init; }
            public decimal Stop { get; init; }
            public decimal Target { get; init; }
            public string Tag { get; init; } = "";
            public DateTimeOffset Ts { get; init; }
        }

        public PaperBroker(SupervisorAgent.StatusService status, ILogger log)
        {
            _status = status; _log = log;
        }

        public Task<bool> RouteAsync(Signal sig, CancellationToken ct)
        {
            if (sig is null) return Task.FromResult(false);
            var sym = sig.Symbol.ToUpperInvariant();
            // Default TP/SL in ticks if not provided
            decimal tick = InstrumentMeta.Tick(sym);
            int tpTicks = EnvInt($"BRACKET_TP_{sym}_TICKS", EnvInt("BRACKET_TP_TICKS", 20));
            int slTicks = EnvInt($"BRACKET_SL_{sym}_TICKS", EnvInt("BRACKET_SL_TICKS", 12));
            var target = sig.Target != 0 ? sig.Target : (sig.Side.Equals("SELL", StringComparison.OrdinalIgnoreCase) ? sig.Entry - tpTicks * tick : sig.Entry + tpTicks * tick);
            var stop = sig.Stop != 0 ? sig.Stop : (sig.Side.Equals("SELL", StringComparison.OrdinalIgnoreCase) ? sig.Entry + slTicks * tick : sig.Entry - slTicks * tick);

            // Narrative logs similar to live
            // Enforce global size cap
            var size = Math.Clamp(sig.Size, 1, 2);

            _log.LogInformation("[Router] PAPER Route: {Side} {Size} {Sym} @ {Entry} (stop {Stop}, target {Target}) tag={Tag}", sig.Side, size, sym, sig.Entry, stop, target, sig.Tag);
            _log.LogInformation("ORDER NEW  {Side} {Size} {Sym} @ {Entry} (paper)", sig.Side, size, sym, sig.Entry);
            _log.LogInformation("Accepted: O#{Tag} (paper)", sig.Tag);

            var pend = new PendingOrder
            {
                Symbol = sym,
                Side = sig.Side,
                Size = size,
                Entry = sig.Entry,
                Stop = stop,
                Target = target,
                Tag = string.IsNullOrWhiteSpace(sig.Tag) ? $"{sym}-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}" : sig.Tag,
                Ts = DateTimeOffset.UtcNow
            };
            _pending[sym] = pend;

            // Simulate latency then immediate fill at entry
            var latMs = EnvInt("PAPER_LATENCY_MS", 150);
            _ = Task.Run(async () =>
            {
                try { await Task.Delay(latMs, ct); } catch { }
                TryFillPending(sym, pend);
            }, ct);

            return Task.FromResult(true);
        }

        private void TryFillPending(string sym, PendingOrder pend)
        {
            // Fill now at entry
            _log.LogInformation("FILL       {Side} {Size} @ {Px} (paper) pos={Pos}", pend.Side, pend.Size, pend.Entry, pend.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase) ? "+" + pend.Size : "-" + pend.Size);
            var ps = _pos.GetOrAdd(sym, _ => new PositionState { Symbol = sym });
            int signedQty = pend.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase) ? pend.Size : -pend.Size;
            var newQty = ps.Qty + signedQty;
            if (ps.Qty == 0)
            {
                ps.Avg = pend.Entry;
            }
            else
            {
                // Weighted average if adding
                ps.Avg = (ps.Avg * Math.Abs(ps.Qty) + pend.Entry * Math.Abs(signedQty)) / (Math.Abs(ps.Qty + signedQty));
            }
            ps.Qty = newQty;
            ps.Stop = pend.Stop;
            ps.Target = pend.Target;
            UpdateStatus(sym, ps);
            _log.LogInformation("STOP NEW   {Stop}  (paper)", ps.Stop);
            _log.LogInformation("TARGET NEW {Tgt}  (paper)", ps.Target);
            // Pending completed
            _pending.TryRemove(sym, out _);
        }

        public void OnBar(string sym, BotCore.Models.Bar bar)
        {
            sym = sym.ToUpperInvariant();
            if (_pos.TryGetValue(sym, out var ps))
            {
                // Update uPnL using bar close
                var pv = InstrumentMeta.PointValue(sym);
                var dir = Math.Sign(ps.Qty);
                var upnlPts = dir * (bar.Close - ps.Avg);
                ps.UnrealizedPnl = upnlPts * pv * Math.Abs(ps.Qty);

                // Check stop/target hits within bar range
                if (ps.Qty != 0 && ps.Stop.HasValue && ps.Target.HasValue)
                {
                    bool hitTarget = false, hitStop = false;
                    if (ps.Qty > 0)
                    {
                        if (bar.High >= ps.Target.Value) hitTarget = true;
                        if (bar.Low <= ps.Stop.Value) hitStop = true;
                    }
                    else if (ps.Qty < 0)
                    {
                        if (bar.Low <= ps.Target.Value) hitTarget = true; // for short, target below
                        if (bar.High >= ps.Stop.Value) hitStop = true;
                    }

                    if (hitTarget || hitStop)
                    {
                        var exitPx = hitTarget ? ps.Target!.Value : ps.Stop!.Value;
                        var sideTxt = ps.Qty > 0 ? "Sell" : "Buy";
                        var qty = Math.Abs(ps.Qty);
                        var pnlPts = (ps.Qty > 0) ? (exitPx - ps.Avg) : (ps.Avg - exitPx);
                        var realized = pnlPts * pv * qty;
                        ps.RealizedPnl += realized;
                        _log.LogInformation(hitTarget ? "EXIT       {Side} {Qty} @ {Px}  (+{Pts:F2} pts avg)  pos=0" : "STOP HIT   {Side} {Qty} @ {Px}  (-{Pts:F2} pts avg)  pos=0",
                            sideTxt.ToUpper(), qty, exitPx, Math.Abs(pnlPts));
                        _log.LogInformation("OCO        Cancelled remaining targets/stops (paper)");
                        // Flat position
                        ps.Qty = 0;
                        ps.Avg = 0m;
                        ps.Stop = null;
                        ps.Target = null;
                        ps.UnrealizedPnl = 0m;
                        UpdateStatus(sym, ps);
                        return;
                    }

                    // Simple trail to breakeven once uPnL >= 0: move stop to avg
                    if (ps.Qty != 0 && ((ps.Qty > 0 && bar.Close >= ps.Avg) || (ps.Qty < 0 && bar.Close <= ps.Avg)))
                    {
                        if ((ps.Qty > 0 && (ps.Stop ?? decimal.MinValue) < ps.Avg) || (ps.Qty < 0 && (ps.Stop ?? decimal.MaxValue) > ps.Avg))
                        {
                            ps.Stop = ps.Avg;
                            _log.LogInformation("MOVE STOP  => {Px}  (breakeven)  trail=on (paper)", ps.Stop);
                        }
                    }
                }
                UpdateStatus(sym, ps);
            }
        }

        private void UpdateStatus(string sym, PositionState ps)
        {
            try
            {
                _status.Set($"pos.{sym}.qty", ps.Qty);
                _status.Set($"pos.{sym}.avg", ps.Avg);
                _status.Set($"pos.{sym}.upnl", ps.UnrealizedPnl);
                _status.Set($"pos.{sym}.rpnl", ps.RealizedPnl);
            }
            catch { }
        }

        private static int EnvInt(string key, int defVal)
        {
            try { var raw = Environment.GetEnvironmentVariable(key); return int.TryParse(raw, out var v) && v >= 0 ? v : defVal; } catch { return defVal; }
        }
    }
}
