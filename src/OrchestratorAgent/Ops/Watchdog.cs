using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Ops
{
    public sealed class Watchdog
    {
        private readonly int _maxMb, _maxThreads, _periodSec;
        private readonly Func<Task> _persist;

        public Watchdog(int maxMb, int maxThreads, int periodSec, Func<Task> persistState)
        { _maxMb = maxMb; _maxThreads = maxThreads; _periodSec = periodSec; _persist = persistState; }

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
