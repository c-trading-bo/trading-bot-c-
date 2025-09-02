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
    public class StrategyAgent(TradingProfileConfig cfg)
    {
        private readonly TradingProfileConfig _cfg = cfg;
        private static readonly Dictionary<(string Strat, string Sym, string Side), long> _lastBarEnteredTs = [];

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
                return [];

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

                // Session-aware spread guard (ES:2/3, NQ:3/4) — do not be stricter than profile
                var fam = s.Family ?? string.Empty;
                bool isBo = fam.Equals("breakout", StringComparison.OrdinalIgnoreCase) || fam.Equals("trend", StringComparison.OrdinalIgnoreCase);
                int baseMax = isBo ? _cfg.GlobalFilters.SpreadTicksMaxBo : _cfg.GlobalFilters.SpreadTicksMax;

                bool inRth = BotCore.Config.TimeWindows.IsNowWithinEt("09:30-16:00", snap.UtcNow);
                int sessMax = snap.Symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) ? (inRth ? 3 : 4) : (inRth ? 2 : 3);
                int spreadMax = Math.Max(baseMax, sessMax);
                if (snap.SpreadTicks > spreadMax) continue;

                if (!StrategyGates.PassesGlobal(_cfg, snap)) continue; // AlwaysOn => always true

                List<Signal> candidates;
                try
                {
                    candidates = AllStrategies.generate_candidates(snap.Symbol, _cfg, s, [.. bars], risk, snap);
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

            outSignals = [.. outSignals
                .OrderByDescending(x => x.Score)
                // Cross-strategy de-dupe: drop StrategyId from the hash so identical trades from different Sx collapse
                .DistinctBy(x => (x.Side, RoundToTick(x.Symbol, x.Entry), RoundToTick(x.Symbol, x.Target), RoundToTick(x.Symbol, x.Stop)))];

            // Enhanced ES/NQ correlation guard with 24/7 session awareness
            var currentTime = snap.UtcNow.TimeOfDay;
            var currentSession = BotCore.Config.ES_NQ_TradingSchedule.GetCurrentSession(currentTime);
            
            if (currentSession != null)
            {
                // Filter signals based on session instrument allowance
                outSignals = [.. outSignals.Where(s => 
                    currentSession.Instruments.Contains(s.Symbol, StringComparer.OrdinalIgnoreCase))];
                
                // Apply session-specific position sizing
                var adjustedSignals = new List<Signal>();
                foreach (var signal in outSignals)
                {
                    var sizeMultiplier = BotCore.Config.ES_NQ_TradingSchedule.GetPositionSizeMultiplier(signal.Symbol, currentTime);
                    var adjustedSignal = signal with 
                    { 
                        Size = (int)(signal.Size * sizeMultiplier)
                    };
                    adjustedSignals.Add(adjustedSignal);
                }
                outSignals = adjustedSignals;
            }
            
            // Enhanced correlation guard for ES/NQ
            var hasEsLong = outSignals.Any(s => s.Symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) && s.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase));
            var hasEsShort = outSignals.Any(s => s.Symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) && s.Side.Equals("SELL", StringComparison.OrdinalIgnoreCase));
            
            // Only filter NQ if we're in high correlation periods and have ES positions
            if (currentSession?.PrimaryInstrument != "BOTH" && (hasEsLong || hasEsShort))
            {
                outSignals = [.. outSignals.Where(s =>
                    !s.Symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) ||
                    !((s.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase) && hasEsLong) ||
                       (s.Side.Equals("SELL", StringComparison.OrdinalIgnoreCase) && hasEsShort))
                )];
            }

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
