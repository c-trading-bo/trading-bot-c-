using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of execution cost configuration
    /// Replaces hardcoded cost budgets and slippage parameters
    /// </summary>
    public class ExecutionCostConfigService : IExecutionCostConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<ExecutionCostConfigService> _logger;

        public ExecutionCostConfigService(IConfiguration config, ILogger<ExecutionCostConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public decimal GetMaxSlippageUsd() => 
            _config.GetValue("ExecutionCost:MaxSlippageUsd", 25.0m);

        public decimal GetDailyExecutionBudgetUsd() => 
            _config.GetValue("ExecutionCost:DailyBudgetUsd", 500.0m);

        public decimal GetCommissionPerContract() => 
            _config.GetValue("ExecutionCost:CommissionPerContract", 2.50m);

        public double GetMarketImpactMultiplier() => 
            _config.GetValue("ExecutionCost:MarketImpactMultiplier", 0.1);

        public double GetExpectedSlippageTicks(string orderType) => orderType?.ToUpper() switch
        {
            "MARKET" => _config.GetValue("ExecutionCost:MarketOrderSlippageTicks", 1.0),
            "LIMIT" => _config.GetValue("ExecutionCost:LimitOrderSlippageTicks", 0.25),
            "STOP" => _config.GetValue("ExecutionCost:StopOrderSlippageTicks", 2.0),
            _ => _config.GetValue("ExecutionCost:DefaultSlippageTicks", 1.0)
        };

        public decimal GetRoutingCostThresholdUsd() => 
            _config.GetValue("ExecutionCost:RoutingCostThreshold", 5.0m);
    }
}