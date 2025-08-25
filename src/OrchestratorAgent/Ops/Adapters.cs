using System;
using System.Threading.Tasks;

namespace OrchestratorAgent.Ops
{
    internal sealed class SimpleStats : IStats
    {
        private readonly DateTime _startUtc;
        private int _emitted;
        public SimpleStats(DateTime startUtc) { _startUtc = startUtc; }
        public int EmittedSignalsTotal => System.Threading.Volatile.Read(ref _emitted);
        public TimeSpan Uptime => DateTime.UtcNow - _startUtc;
        public void Inc(int n = 1) => System.Threading.Interlocked.Add(ref _emitted, n);
    }

    internal sealed class NotifierAdapter : INotifier
    {
        private readonly OrchestratorAgent.Infra.Notifier _inner;
        public NotifierAdapter(OrchestratorAgent.Infra.Notifier inner) { _inner = inner; }
        public Task Info(string m) => _inner.Info(m);
        public Task Warn(string m) => _inner.Warn(m);
        public Task Error(string m) => _inner.Error(m);
    }
}
