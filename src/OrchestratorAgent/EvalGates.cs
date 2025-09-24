#nullable enable
using BotCore;

namespace OrchestratorAgent
{
    /// <summary>Wraps your previously-added EvalPolicy/PnLTracker checks into a single gate call.</summary>
    internal sealed class EvalGates(EvalPolicy policy, PnLTracker pnl)
    {
        private readonly EvalPolicy _policy = policy;
        private readonly PnLTracker _pnl = pnl;

        public bool Allow(StrategySignal sig, out string reason)
        {
            reason = "";
            var root = SymbolMeta.RootFromName(sig.Symbol);
            var guard = new EvalGuard(_policy, _pnl);
            if (!guard.CanOpen(root, sig.Size, out var r))
            {
                reason = r ?? "eval guard";
                return false;
            }
            return true;
        }
    }
}
