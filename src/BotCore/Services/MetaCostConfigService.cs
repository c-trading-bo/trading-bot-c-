using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of meta-cost configuration
    /// Replaces hardcoded cost blending parameters in RL/ML models
    /// </summary>
    public class MetaCostConfigService : IMetaCostConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<MetaCostConfigService> _logger;

        public MetaCostConfigService(IConfiguration config, ILogger<MetaCostConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public double GetExecutionCostWeight() => 
            _config.GetValue("MetaCost:ExecutionCostWeight", 0.3);

        public double GetMarketImpactWeight() => 
            _config.GetValue("MetaCost:MarketImpactWeight", 0.2);

        public double GetOpportunityCostWeight() => 
            _config.GetValue("MetaCost:OpportunityCostWeight", 0.25);

        public double GetTimingCostWeight() => 
            _config.GetValue("MetaCost:TimingCostWeight", 0.15);

        public double GetVolatilityRiskWeight() => 
            _config.GetValue("MetaCost:VolatilityRiskWeight", 0.1);

        public double GetCostBlendingTemperature() => 
            _config.GetValue("MetaCost:CostBlendingTemperature", 1.0);

        public bool NormalizeCostWeights() => 
            _config.GetValue("MetaCost:NormalizeCostWeights", true);

        public double GetAdaptiveWeightRate() => 
            _config.GetValue("MetaCost:AdaptiveWeightRate", 0.01);
    }
}