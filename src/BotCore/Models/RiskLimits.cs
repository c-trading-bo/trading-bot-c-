namespace BotCore.Models
{
    /// <summary>
    /// Risk limits configuration for position tracking
    /// </summary>
    public class RiskLimits
    {
        private const decimal DEFAULT_MAX_DAILY_LOSS = -1000m;
        private const decimal DEFAULT_MAX_POSITION_SIZE = 5;
        private const decimal DEFAULT_MAX_DRAWDOWN = 0.15m;
        private const int DEFAULT_MAX_ORDERS_PER_MINUTE = 10;
        private const decimal DEFAULT_ACCOUNT_BALANCE = 25000m;
        private const decimal DEFAULT_MAX_RISK_PER_TRADE = 100m;
        
        public decimal MaxDailyLoss { get; set; } = DEFAULT_MAX_DAILY_LOSS;
        public decimal MaxPositionSize { get; set; } = DEFAULT_MAX_POSITION_SIZE;
        public decimal MaxDrawdown { get; set; } = DEFAULT_MAX_DRAWDOWN;
        public int MaxOrdersPerMinute { get; set; } = DEFAULT_MAX_ORDERS_PER_MINUTE;
        public decimal AccountBalance { get; set; } = DEFAULT_ACCOUNT_BALANCE;
        public decimal MaxRiskPerTrade { get; set; } = DEFAULT_MAX_RISK_PER_TRADE;
    }
}