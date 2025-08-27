
#nullable enable
using System.Reflection;
using Microsoft.Extensions.Logging;
using BotCore;
using BotCore.Models;

namespace OrchestratorAgent
{
    /// <summary>Discovers all IStrategy implementations and runs them on each bar.</summary>
    public sealed class StrategyManager
    {
        private readonly ILogger<StrategyManager> _log;
        private readonly List<IStrategy> _strategies = new();
        private readonly StrategyContext _ctx;
        private readonly OrderRouter _router;
        private readonly EvalGates _gates;
        private readonly IReadOnlyDictionary<string,string> _contractIds;

        public StrategyManager(ILogger<StrategyManager> log,
                               StrategyContext ctx,
                               OrderRouter router,
                               EvalGates gates,
                               IReadOnlyDictionary<string,string> contractIds)
        {
            _log = log;
            _ctx = ctx;
            _router = router;
            _gates = gates;
            _contractIds = contractIds;
        }

        public void DiscoverFrom(params Assembly[] assemblies)
        {
            if (assemblies == null || assemblies.Length == 0)
                assemblies = new[] { Assembly.GetExecutingAssembly() };

            foreach (var asm in assemblies)
            foreach (var t in asm.GetTypes())
            {
                if (t.IsAbstract || t.IsInterface) continue;
                if (typeof(IStrategy).IsAssignableFrom(t))
                {
                    if (Activator.CreateInstance(t) is IStrategy s)
                    {
                        _strategies.Add(s);
                        _log.LogInformation("Strategy registered: {Name} ({Type})", s.Name, t.FullName);
                    }
                }
            }
        }

        public async Task OnNewBarAsync(string symbol, Bar bar, CancellationToken ct = default)
        {
            foreach (var s in _strategies)
            {
                IEnumerable<StrategySignal> sigs;
                try
                {
                    sigs = s.OnBar(symbol, _ctx.GetBars(symbol), _ctx);
                }
                catch (Exception ex)
                {
                    _log.LogWarning(ex, "Strategy {S} threw during OnBar for {Sym}", s.Name, symbol);
                    continue;
                }

                foreach (var sig in sigs)
                {
                    // Normalize and gate
                    var root = string.IsNullOrWhiteSpace(sig.Symbol) ? symbol : sig.Symbol;
                    var normalized = new StrategySignal
                    {
                        Strategy = string.IsNullOrWhiteSpace(sig.Strategy) ? s.Name : sig.Strategy,
                        Symbol   = root,
                        Side     = sig.Side,
                        Size     = Math.Clamp(sig.Size, 1, 2),
                        LimitPrice = sig.LimitPrice,
                        Note     = sig.Note
                    };

                    if (normalized.Side == SignalSide.Flat) continue;

                    if (!_contractIds.TryGetValue(root, out var cid))
                    {
                        _log.LogWarning("No contractId for {Root}; skipping signal {Strat}", root, normalized.Strategy);
                        continue;
                    }

                    if (!_gates.Allow(normalized, out var reason))
                    {
                        _log.LogWarning("[BLOCKED] {Strat} {Root} {Side} x{Size}: {Reason}",
                            normalized.Strategy, root, normalized.Side, normalized.Size, reason);
                        continue;
                    }

                    await _router.RouteAsync(normalized, cid, ct);
                }
            }
        }
    }
}
