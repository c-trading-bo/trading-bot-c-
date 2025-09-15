using System.ComponentModel.DataAnnotations;

namespace BotCore.Configuration
{
    /// <summary>
    /// Configuration for enhanced backtest realism with market friction
    /// Implements slippage, latency, and commission modeling for realistic backtesting
    /// </summary>
    public class BacktestEnhancementConfiguration
    {
        /// <summary>
        /// Enable market friction simulation in backtests
        /// </summary>
        public bool EnableMarketFriction { get; set; } = true;

        /// <summary>
        /// Slippage configuration for different scenarios
        /// </summary>
        public SlippageConfiguration SlippageConfig { get; set; } = new();

        /// <summary>
        /// Latency simulation configuration
        /// </summary>
        public LatencyConfiguration LatencyConfig { get; set; } = new();

        /// <summary>
        /// Commission and fee configuration
        /// </summary>
        public CommissionConfiguration CommissionConfig { get; set; } = new();
    }

    /// <summary>
    /// Slippage modeling configuration for realistic order execution
    /// </summary>
    public class SlippageConfiguration
    {
        /// <summary>
        /// Base slippage in basis points (1 bp = 0.01%)
        /// </summary>
        [Range(0.0, 10.0)]
        public double BaseSlippageBps { get; set; } = 0.5;

        /// <summary>
        /// Volatility multiplier for slippage calculation
        /// Higher volatility = more slippage
        /// </summary>
        [Range(1.0, 3.0)]
        public double VolatilityMultiplier { get; set; } = 1.2;

        /// <summary>
        /// Liquidity impact factor
        /// </summary>
        [Range(0.0, 1.0)]
        public double LiquidityImpact { get; set; } = 0.1;

        /// <summary>
        /// ES-specific slippage in basis points
        /// </summary>
        [Range(0.0, 2.0)]
        public double ESSlippageBps { get; set; } = 0.25;

        /// <summary>
        /// NQ-specific slippage in basis points
        /// </summary>
        [Range(0.0, 2.0)]
        public double NQSlippageBps { get; set; } = 0.5;



        /// <summary>
        /// Calculate slippage for a given symbol and market conditions
        /// </summary>
        public double CalculateSlippage(string symbol, double volatility = 1.0, double liquidityScore = 1.0)
        {
            var baseSlippage = symbol.ToUpper() switch
            {
                "ES" => ESSlippageBps,
                "NQ" => NQSlippageBps,
                _ => BaseSlippageBps
            };

            return baseSlippage * VolatilityMultiplier * volatility * (1 + LiquidityImpact * (1 - liquidityScore));
        }
    }

    /// <summary>
    /// Latency simulation configuration for realistic order timing
    /// </summary>
    public class LatencyConfiguration
    {
        /// <summary>
        /// Base latency in milliseconds
        /// </summary>
        [Range(5, 100)]
        public int BaseLatencyMs { get; set; } = 15;

        /// <summary>
        /// Network jitter variance in milliseconds
        /// </summary>
        [Range(0, 20)]
        public int NetworkJitterMs { get; set; } = 5;

        /// <summary>
        /// Order processing delay in milliseconds
        /// </summary>
        [Range(2, 50)]
        public int OrderProcessingMs { get; set; } = 8;

        /// <summary>
        /// Maximum allowable latency in milliseconds
        /// </summary>
        [Range(50, 500)]
        public int MaxLatencyMs { get; set; } = 100;

        /// <summary>
        /// Calculate total latency with random jitter
        /// </summary>
        public int CalculateLatency(Random? random = null)
        {
            random ??= new Random();
            var jitter = random.Next(-NetworkJitterMs, NetworkJitterMs + 1);
            var totalLatency = BaseLatencyMs + OrderProcessingMs + jitter;
            return Math.Min(totalLatency, MaxLatencyMs);
        }
    }

    /// <summary>
    /// Commission and fee configuration for realistic cost modeling
    /// </summary>
    public class CommissionConfiguration
    {
        /// <summary>
        /// ES commission per contract
        /// </summary>
        [Range(0.0, 5.0)]
        public decimal ESCommission { get; set; } = 0.62m;

        /// <summary>
        /// NQ commission per contract
        /// </summary>
        [Range(0.0, 5.0)]
        public decimal NQCommission { get; set; } = 0.62m;



        /// <summary>
        /// Whether to charge round-turn commission (entry + exit)
        /// </summary>
        public bool RoundTurnCommission { get; set; } = true;

        /// <summary>
        /// Calculate commission for a given symbol and quantity
        /// </summary>
        public decimal CalculateCommission(string symbol, int quantity, bool isEntry = true)
        {
            var commissionPerContract = symbol.ToUpper() switch
            {
                "ES" => ESCommission,
                "NQ" => NQCommission,
                _ => 0.62m // Default commission for standard futures
            };

            var multiplier = RoundTurnCommission ? 2 : (isEntry ? 1 : 1);
            return commissionPerContract * quantity * multiplier;
        }
    }
}