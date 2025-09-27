namespace OrchestratorAgent.Ops
{
    /// <summary>Process-scoped flags for routing behavior.</summary>
    internal sealed class AppState
    {
        /// <summary>When true, do not open NEW parent entries; still manage existing OCO/TP/SL.</summary>
        public volatile bool DrainMode;
    }
}
