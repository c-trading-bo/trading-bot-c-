using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of sizer configuration
    /// Replaces hardcoded RL/ML hyperparameters and sizing parameters
    /// </summary>
    public class SizerConfigService : ISizerConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<SizerConfigService> _logger;

        public SizerConfigService(IConfiguration config, ILogger<SizerConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public double GetPpoLearningRate() => 
            _config.GetValue("Sizer:PPO:LearningRate", 3e-4);

        public double GetCqlAlpha() => 
            _config.GetValue("Sizer:CQL:Alpha", 0.2);

        public double GetMetaCostWeight(string costType) => costType?.ToLower() switch
        {
            "execution" => _config.GetValue("Sizer:MetaCost:ExecutionWeight", 0.3),
            "market_impact" => _config.GetValue("Sizer:MetaCost:MarketImpactWeight", 0.2),
            "opportunity" => _config.GetValue("Sizer:MetaCost:OpportunityWeight", 0.25),
            "timing" => _config.GetValue("Sizer:MetaCost:TimingWeight", 0.15),
            "volatility" => _config.GetValue("Sizer:MetaCost:VolatilityWeight", 0.1),
            _ => _config.GetValue("Sizer:MetaCost:DefaultWeight", 0.2)
        };

        public double GetPositionSizeMultiplierBaseline() => 
            _config.GetValue("Sizer:PositionSizeMultiplierBaseline", 1.0);

        public double GetMinPositionSizeMultiplier() => 
            _config.GetValue("Sizer:MinPositionSizeMultiplier", 0.1);

        public double GetMaxPositionSizeMultiplier() => 
            _config.GetValue("Sizer:MaxPositionSizeMultiplier", 2.5);

        public double GetExplorationRate() => 
            _config.GetValue("Sizer:ExplorationRate", 0.05);

        public double GetWeightFloor() => 
            _config.GetValue("Sizer:WeightFloor", 0.10);

        public int GetModelRefreshIntervalMinutes() => 
            _config.GetValue("Sizer:ModelRefreshIntervalMinutes", 120);
    }
}