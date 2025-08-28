// Agent: StrategyAgent
// Role: Implements trading strategies and signal generation.
// Integration: Receives market data, sends signals to orchestrator and order agents.

using System;
using System.Collections.Generic;
using System.Linq;
using BotCore.Config;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;

namespace StrategyAgent
{
    public class StrategyAgent
    {
        private readonly TradingProfileConfig _cfg;
    private static readonly Dictionary<(string Strat, string Sym, string Side), long> _lastBarEnteredTs = new();

        public StrategyAgent(TradingProfileConfig cfg) => _cfg = cfg;

        public List<Signal> RunAll(BotCore.Models.MarketSnapshot snap, IReadOnlyList<Bar> bars, RiskEngine risk)
        {
            // Schema pinning to avoid silent profile shape regressions
            if (_cfg.SchemaVersion != 1)
                throw new InvalidOperationException($"Unsupported profile schema: v{_cfg.SchemaVersion}");

            // Early circuit breakers (simple env toggles); if hit, emit nothing
            var dailyLock = Environment.GetEnvironmentVariable("DAILY_LOCKOUT");
            var tradeLock = Environment.GetEnvironmentVariable("TRADE_LOCKOUT");
            if ((dailyLock?.Equals("1", StringComparison.OrdinalIgnoreCase) ?? false) ||
                (tradeLock?.Equals("1", StringComparison.OrdinalIgnoreCase) ?? false))
                return new List<Signal>();

            var outSignals = new List<Signal>();
            var lastBarTs = (bars != null && bars.Count > 0) ? bars[^1].Ts : 0L;

            foreach (var s in _cfg.Strategies)
            {
                if (!s.Enabled) continue;

                // Treat filter-family as gate only (no orders)
                if (!string.IsNullOrWhiteSpace(s.Family) && s.Family.Equals("filter", StringComparison.OrdinalIgnoreCase))
                    continue;

                // Enforce session window and flat-by regardless of AlwaysOn
                if (!string.IsNullOrWhiteSpace(s.SessionWindowEt) && !BotCore.Config.TimeWindows.IsNowWithinEt(s.SessionWindowEt!, snap.UtcNow))
                    continue;
                if (!string.IsNullOrWhiteSpace(s.FlatByEt))
                {
                    var until = $"00:00-{s.FlatByEt}";
                    if (!BotCore.Config.TimeWindows.IsNowWithinEt(until, snap.UtcNow))
                        continue;
                }

                // Minimal spread guard: 1 tick default, 2 for breakout/trend
                var fam = s.Family ?? string.Empty;
                bool isBo = fam.Equals("breakout", StringComparison.OrdinalIgnoreCase) || fam.Equals("trend", StringComparison.OrdinalIgnoreCase);
                var spreadMax = isBo ? _cfg.GlobalFilters.SpreadTicksMaxBo : _cfg.GlobalFilters.SpreadTicksMax;
                if (snap.SpreadTicks > spreadMax) continue;

                if (!StrategyGates.PassesGlobal(_cfg, snap)) continue; // AlwaysOn => always true

                List<Signal> candidates;
                try
                {
                    candidates = AllStrategies.generate_candidates(snap.Symbol, _cfg, s, new List<Bar>(bars), risk, snap);
                }
                catch (Exception)
                {
                    // Exception policy: skip this strategy for this bar
                    continue;
                }

                // Per-strategy cooldown and duplicate suppression
                var cooldownSec = 0;
                if (s.Extra.TryGetValue("cooldown_s", out var coolElem) && coolElem.TryGetInt32(out var c))
                    cooldownSec = Math.Max(0, c);
                if (cooldownSec <= 0)
                    cooldownSec = _cfg.GlobalFilters.MinSecondsBetweenEntries;

                foreach (var sig in candidates)
                {
                    // block same-bar re-entry per (strategy,symbol,side)
                    var key = (sig.StrategyId, sig.Symbol, sig.Side);
                    if (_lastBarEnteredTs.TryGetValue(key, out var prevTs) && prevTs == lastBarTs)
                        continue;

                    if (!BotCore.RecentSignalCache.ShouldEmit(sig.StrategyId, sig.Symbol, sig.Side, sig.Entry, sig.Target, sig.Stop, cooldownSec))
                        continue;

                    _lastBarEnteredTs[key] = lastBarTs;
                    outSignals.Add(sig);
                }
            }

            // Score & rank, de-dup by rounded tuple, then concurrency caps
            static decimal RoundToTick(string sym, decimal px)
            {
                var t = InstrumentMeta.Tick(sym);
                return Math.Round(px / t, 0, MidpointRounding.AwayFromZero) * t;
            }

            outSignals = outSignals
                .OrderByDescending(x => x.Score)
                // Cross-strategy de-dupe: drop StrategyId from the hash so identical trades from different Sx collapse
                .DistinctBy(x => (x.Side, RoundToTick(x.Symbol, x.Entry), RoundToTick(x.Symbol, x.Target), RoundToTick(x.Symbol, x.Stop)))
                .ToList();

            // Simple ES/NQ correlation guard: avoid doubling exposure in same direction
            var hasEsLong = outSignals.Any(s => s.Symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) && s.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase));
            var hasEsShort = outSignals.Any(s => s.Symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) && s.Side.Equals("SELL", StringComparison.OrdinalIgnoreCase));
            outSignals = outSignals.Where(s =>
                !s.Symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) ||
                !((s.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase) && hasEsLong) ||
                   (s.Side.Equals("SELL", StringComparison.OrdinalIgnoreCase) && hasEsShort))
            ).ToList();

            // Concurrency: enforce one fresh entry per symbol and total cap
            if (_cfg.Concurrency.OneFreshEntryPerSymbol)
            {
                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var filtered = new List<Signal>();
                foreach (var s in outSignals)
                {
                    if (seen.Add(s.Symbol)) filtered.Add(s);
                }
                outSignals = filtered;
            }
            var maxTotal = _cfg.Concurrency.MaxPositionsTotal;
            if (maxTotal > 0 && outSignals.Count > maxTotal)
                outSignals = outSignals.GetRange(0, maxTotal);

            return outSignals;
        }
    }
}
