using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;

namespace BotCore
{
    public static class TradeLog
    {
        public record ContractSpec(string Sym, int Decimals, decimal TickSize, decimal BigPointValue);
        public static readonly IReadOnlyDictionary<string, ContractSpec> Spec = new Dictionary<string, ContractSpec>
        {
            ["ES"] = new("ES", 2, 0.25m, 50m),
            ["NQ"] = new("NQ", 2, 0.25m, 20m)
        };

        static decimal RoundToTick(decimal px, decimal tick) =>
            (tick <= 0m) ? px : System.Math.Round(px / tick, 0, System.MidpointRounding.AwayFromZero) * tick;

        static string Fpx(string sym, decimal px)
        {
            if (!Spec.TryGetValue(sym, out var c)) c = new ContractSpec(sym, 2, 0.25m, 1m);
            return RoundToTick(px, c.TickSize).ToString($"F{c.Decimals}", CultureInfo.InvariantCulture);
        }
        static string Fpn(decimal v) => v.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);

        static readonly ConcurrentDictionary<string, string> _last = new();
        static void LogChange(ILogger log, string key, string line, LogLevel lvl = LogLevel.Information)
        {
            if (_last.TryGetValue(key, out var prev) && prev == line) return;
            _last[key] = line;
            log.Log(lvl, "{line}", line);
        }

        public static void Session(ILogger log, string mode, string acct, string[] syms) =>
            log.LogInformation("SESSION mode={mode} acct={acct} syms={syms}", mode, acct, string.Join(",", syms));

        public static void Signal(ILogger log, string sym, string strat, string side, int qty, decimal entry, decimal stop, decimal target, string reason, string tag)
            => log.LogInformation("[{sym}] SIGNAL {strat} {side} x{qty} @ {entry} (stop {stop}, t1 {t1}) tag={tag} {reason}",
                sym, strat, side, qty, Fpx(sym, entry), Fpx(sym, stop), Fpx(sym, target), tag, string.IsNullOrWhiteSpace(reason) ? "" : $"reason={reason}");

        public static void OrderNew(ILogger log, string sym, string side, int qty, decimal px, string tag)
            => log.LogInformation("[{sym}] ORDER NEW {side} x{qty} @ {px} tag={tag}", sym, side, qty, Fpx(sym, px), tag);

        public static void Fill(ILogger log, string sym, string side, int qty, decimal px, int pos, decimal avg, decimal mark, decimal uPnL, decimal rPnL, string tag)
            => log.LogInformation("[{sym}] FILL {side} x{qty} @ {px} => pos={pos} avg={avg} mark={mark} uPnL={u} rPnL={r} tag={tag}",
                sym, side, qty, Fpx(sym, px), pos, Fpx(sym, avg), Fpx(sym, mark), Fpn(uPnL), Fpn(rPnL), tag);

        public static void StopNew(ILogger log, string sym, decimal stop, string tag)
            => log.LogInformation("[{sym}] STOP NEW {stop} tag={tag}", sym, Fpx(sym, stop), tag);

        public static void StopMove(ILogger log, string sym, decimal stop, string reason, string tag)
            => log.LogInformation("[{sym}] STOP MOVE => {stop} {reason} tag={tag}", sym, Fpx(sym, stop), reason, tag);

        public static void StopHit(ILogger log, string sym, int qty, decimal px, int pos, decimal rPnL, string tag)
            => log.LogInformation("[{sym}] STOP HIT {qty} @ {px} => pos={pos} rPnL={r} tag={tag}", sym, qty, Fpx(sym, px), pos, Fpn(rPnL), tag);

        public static void TargetNew(ILogger log, string sym, decimal t1, string tag)
            => log.LogInformation("[{sym}] TARGET NEW {t1} tag={tag}", sym, Fpx(sym, t1), tag);

        public static void Exit(ILogger log, string sym, int qty, decimal px, int pos, decimal rPnL, string tag)
            => log.LogInformation("[{sym}] EXIT {qty} @ {px} => pos={pos} rPnL={r} tag={tag}", sym, qty, Fpx(sym, px), pos, Fpn(rPnL), tag);

        public static void Position(ILogger log, string sym, int pos, decimal avg, decimal mark, decimal uPnL, decimal rPnL)
            => LogChange(log, $"pos/{sym}",
                $"[{sym}] POS {pos:+#;-#;0} @ {Fpx(sym, avg)} mark={Fpx(sym, mark)} uPnL={Fpn(uPnL)} rPnL={Fpn(rPnL)}");

        public static void Skip(ILogger log, string sym, string code, string detail) =>
            LogChange(log, $"skip/{sym}/{code}", $"[{sym}] SKIP {code} {detail}");

        public static void SpreadGate(ILogger log, string sym, bool closed, int ticks, int allowTicks)
            => LogChange(log, $"gate/{sym}", $"[{sym}] GATE spread={(closed ? "CLOSED" : "OPEN")} {ticks}t allow={allowTicks}t");

        public static void Heartbeat(ILogger log, decimal dailyPnL, decimal maxDD, decimal remaining, string exposure)
            => log.LogInformation("HEARTBEAT dailyPnL={d} maxDailyLoss={m} remaining={r} exposure={x}",
                Fpn(dailyPnL), Fpn(maxDD), Fpn(remaining), exposure);
    }
}
