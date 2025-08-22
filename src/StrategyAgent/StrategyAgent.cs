// Agent: StrategyAgent
// Role: Implements trading strategies and signal generation.
// Integration: Receives market data, sends signals to orchestrator and order agents.

using System.Collections.Generic;
using BotCore.Config;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;

namespace StrategyAgent
{
    public class StrategyAgent
    {
        private readonly TradingProfileConfig _cfg;

        public StrategyAgent(TradingProfileConfig cfg) => _cfg = cfg;

        public List<Signal> RunAll(MarketSnapshot snap, IReadOnlyList<Bar> bars, RiskEngine risk)
        {
            var outSignals = new List<Signal>();

            foreach (var s in _cfg.Strategies)
            {
                if (!s.Enabled) continue;
                if (!StrategyGates.PassesRSGate(_cfg, snap)) continue;
                if (!StrategyGates.PassesGlobalFilters(_cfg, s, snap)) continue;

                foreach (var sig in AllStrategies.generate_candidates(snap.Symbol, _cfg, s, new List<Bar>(bars), risk))
                    outSignals.Add(sig);
            }

            return outSignals;
        }
    }
}
