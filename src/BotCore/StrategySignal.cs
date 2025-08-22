#nullable enable
namespace BotCore
{
    public enum SignalSide { Long = 1, Short = -1, Flat = 0 }

    /// <summary>A normalized signal your orchestrator can route.</summary>
    public sealed class StrategySignal
    {
        public string Strategy { get; init; } = "";
        public string Symbol   { get; init; } = "";
        public SignalSide Side { get; init; } = SignalSide.Flat;
        public int Size        { get; init; } = 1;
        public decimal? LimitPrice { get; init; } // optional; null means use market/last
        public string? Note    { get; init; }
    }
}
