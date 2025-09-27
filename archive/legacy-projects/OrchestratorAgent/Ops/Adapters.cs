using System;
using System.Threading.Tasks;

namespace OrchestratorAgent.Ops
{
    internal sealed class SimpleStats(DateTime startUtc) : IStats
    {
        private readonly DateTime _startUtc = startUtc;
        private int _emitted;

        public int EmittedSignalsTotal => System.Threading.Volatile.Read(ref _emitted);
        public TimeSpan Uptime => DateTime.UtcNow - _startUtc;
        public void Inc(int n = 1) => System.Threading.Interlocked.Add(ref _emitted, n);
    }

    internal sealed class NotifierAdapter(OrchestratorAgent.Infra.Notifier inner) : INotifier
    {
        private readonly OrchestratorAgent.Infra.Notifier _inner = inner;

        public Task Info(string m) => _inner.Info(m);
        public Task Warn(string m) => _inner.Warn(m);
        public Task Error(string m) => _inner.Error(m);
    }
}
