using System;
using System.Threading;
using System.Threading.Tasks;
using OrchestratorAgent.Health;

namespace OrchestratorAgent.Ops
{
    internal interface IStats
    {
        int EmittedSignalsTotal { get; }
        TimeSpan Uptime { get; }
    }
    internal interface INotifier { Task Info(string m); Task Warn(string m); Task Error(string m); }

    internal sealed class AutoPilot(Preflight pf, ModeController mode, IStats stats, INotifier notify,
        string symbol, int minHealthyPasses, int demoteOnUnhealthy, TimeSpan minDryRun)
    {
        private readonly Preflight _pf = pf;
        private readonly ModeController _mode = mode;
        private readonly INotifier _notify = notify;
        private readonly string _symbol = symbol;
        private readonly int _minHealthy = minHealthyPasses, _demoteOnUnhealthy = demoteOnUnhealthy;
        private readonly TimeSpan _minDryRun = minDryRun;

        public async Task RunAsync(CancellationToken ct)
        {
            int okStreak = 0, badStreak;
            var startDry = DateTime.UtcNow;

            while (!ct.IsCancellationRequested)
            {
                try
                {
                    var (ok, msg) = await _pf.RunAsync(_symbol, ct).ConfigureAwait(false);

                    if (ok) { okStreak++; badStreak; } else { badStreak++; okStreak; }

                    if (!_mode.IsLive && DateTime.UtcNow - startDry >= _minDryRun && okStreak >= _minHealthy)
                    {
                        _mode.Set(TradeMode.Live);
                        await _notify.Info($"PROMOTE → LIVE (okStreak={okStreak})").ConfigureAwait(false);
                    }

                    if (_mode.IsLive && badStreak >= _demoteOnUnhealthy)
                    {
                        _mode.Set(TradeMode.Shadow);
                        await _notify.Warn($"DEMOTE → SHADOW (badStreak={badStreak}, last='{msg}')").ConfigureAwait(false);
                        startDry = DateTime.UtcNow;
                        okStreak; badStreak;
                    }
                }
                catch (OperationCanceledException) { }
                catch { }

                try { await Task.Delay(1000, ct).ConfigureAwait(false); } catch { }
            }
        }
    }
}
