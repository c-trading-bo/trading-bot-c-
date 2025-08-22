#nullable enable
using BotCore;

namespace OrchestratorAgent
{
    /// <summary>Wraps your previously-added EvalPolicy/PnLTracker checks into a single gate call.</summary>
    public sealed class EvalGates
    {
        private readonly EvalPolicy _policy;
        private readonly PnLTracker _pnl;

        public EvalGates(EvalPolicy policy, PnLTracker pnl) { _policy = policy; _pnl = pnl; }

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
