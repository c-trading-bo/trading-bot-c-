namespace RiskAgent
{
    public sealed class RiskEngine
    {
        public RiskConfig cfg { get; set; } = new RiskConfig();

        public decimal ComputeRisk(decimal entry, decimal stop, decimal target, bool isLong)
        {
            var risk = isLong ? entry - stop : stop - entry;
            var reward = isLong ? target - entry : entry - target;
            if (risk <= 0 || reward < 0) return 0m;
            return reward / risk;
        }

        public decimal size_for(decimal riskPerTrade, decimal dist, decimal pointValue)
        {
            if (dist <= 0 || pointValue <= 0) return 0m;
            return Math.Floor(riskPerTrade / (dist * pointValue));
        }
    }
}
