using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Ops
{
    public sealed class Watchdog(int maxMb, int maxThreads, int periodSec, Func<Task> persistState)
    {
        private readonly int _maxMb = maxMb, _maxThreads = maxThreads, _periodSec = periodSec;
        private readonly Func<Task> _persist = persistState;

        public async Task RunLoopAsync(CancellationToken ct)
        {
            var proc = Process.GetCurrentProcess();
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    proc.Refresh();
                    var rssMb = proc.WorkingSet64 / (1024.0 * 1024.0);
                    var threads = proc.Threads.Count;
                    if (rssMb > _maxMb || threads > _maxThreads)
                    {
                        try { await _persist(); } catch { }
                        Environment.Exit(200);
                    }
                }
                catch { }
                try { await Task.Delay(TimeSpan.FromSeconds(_periodSec), ct); } catch { }
            }
        }
    }
}
