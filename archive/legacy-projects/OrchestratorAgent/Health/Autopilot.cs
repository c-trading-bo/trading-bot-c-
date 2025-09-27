using Microsoft.Extensions.Logging;
using OrchestratorAgent.Ops;

namespace OrchestratorAgent.Health
{
    internal sealed class Autopilot(ILogger log, ModeController mode)
    {
        private readonly ILogger _log = log;
        private readonly ModeController _mode = mode;

        public void MaybePromote((bool ok, string reason) health)
        {
            if (!_mode.Autopilot) return;
            if (_mode.IsShadow && health.ok)
                _mode.Set(TradeMode.Live);
            else if (!health.ok)
                _log.LogWarning("Preflight not ready: {Reason}", health.reason);
        }
    }
}
