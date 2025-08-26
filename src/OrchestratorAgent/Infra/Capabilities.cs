using System.Collections.Generic;

namespace OrchestratorAgent.Infra
{
    public static class Capabilities
    {
        private static readonly HashSet<string> _set = new();
        public static IReadOnlyCollection<string> All => _set;
        public static void Add(string name)
        {
            lock (_set) _set.Add(name);
        }
    }
}
