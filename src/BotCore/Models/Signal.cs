namespace BotCore.Models
{
    public sealed record Signal
    {
        public string StrategyId { get; init; } = "";
        public string Symbol { get; init; } = "";
        public string Side { get; init; } = "BUY"; // "BUY" or "SELL"
        public decimal Entry { get; init; }
        public decimal Stop { get; init; }
        public decimal Target { get; init; }
        public decimal ExpR { get; init; }
        public decimal Score { get; init; } // computed ranking score
        public decimal QScore { get; init; } // normalized quality score used by session gates
        public int Size { get; init; } = 1;
        public long AccountId { get; init; }
        public string ContractId { get; init; } = "";
        public string Tag { get; init; } = "";
        // Audit / provenance
        public string StrategyVersion { get; init; } = "1.0.0";
        public string ProfileName { get; init; } = "";
        public DateTime EmittedUtc { get; init; } = DateTime.UtcNow;
    }
}
