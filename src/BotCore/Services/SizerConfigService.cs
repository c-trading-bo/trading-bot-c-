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
        // ML/RL Configuration Constants
        private const double DefaultPpoLearningRate = 0.0003; // 3e-4
        private const double DefaultCqlAlpha = 0.2;
        
        // Meta Cost Weight Constants
        private const double DefaultExecutionWeight = 0.3;
        private const double DefaultMarketImpactWeight = 0.2;
        private const double DefaultOpportunityWeight = 0.25;
        private const double DefaultTimingWeight = 0.15;
        private const double DefaultVolatilityWeight = 0.1;
        private const double DefaultMetaCostWeight = 0.2;
        
        // Position Sizing Constants
        private const double DefaultPositionSizeMultiplierBaseline = 1.0;
        private const double DefaultMinPositionSizeMultiplier = 0.1;
        private const double DefaultMaxPositionSizeMultiplier = 2.5;
        private const double DefaultExplorationRate = 0.05;
        private const double DefaultWeightFloor = 0.10;
        
        // Model Refresh Constants
        private const int DefaultModelRefreshIntervalMinutes = 120;
        
        private readonly IConfiguration _config;
        private readonly ILogger<SizerConfigService> _logger;

        public SizerConfigService(IConfiguration config, ILogger<SizerConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public double GetPpoLearningRate() => 
            _config.GetValue("Sizer:PPO:LearningRate", DefaultPpoLearningRate);

        public double GetCqlAlpha() => 
            _config.GetValue("Sizer:CQL:Alpha", DefaultCqlAlpha);

        public double GetMetaCostWeight(string costType) => costType?.ToLower() switch
        {
            "execution" => _config.GetValue("Sizer:MetaCost:ExecutionWeight", DefaultExecutionWeight),
            "market_impact" => _config.GetValue("Sizer:MetaCost:MarketImpactWeight", DefaultMarketImpactWeight),
            "opportunity" => _config.GetValue("Sizer:MetaCost:OpportunityWeight", DefaultOpportunityWeight),
            "timing" => _config.GetValue("Sizer:MetaCost:TimingWeight", DefaultTimingWeight),
            "volatility" => _config.GetValue("Sizer:MetaCost:VolatilityWeight", DefaultVolatilityWeight),
            _ => _config.GetValue("Sizer:MetaCost:DefaultWeight", DefaultMetaCostWeight)
        };

        public double GetPositionSizeMultiplierBaseline() => 
            _config.GetValue("Sizer:PositionSizeMultiplierBaseline", DefaultPositionSizeMultiplierBaseline);

        public double GetMinPositionSizeMultiplier() => 
            _config.GetValue("Sizer:MinPositionSizeMultiplier", DefaultMinPositionSizeMultiplier);

        public double GetMaxPositionSizeMultiplier() => 
            _config.GetValue("Sizer:MaxPositionSizeMultiplier", DefaultMaxPositionSizeMultiplier);

        public double GetExplorationRate() => 
            _config.GetValue("Sizer:ExplorationRate", DefaultExplorationRate);

        public double GetWeightFloor() => 
            _config.GetValue("Sizer:WeightFloor", DefaultWeightFloor);

        public int GetModelRefreshIntervalMinutes() => 
            _config.GetValue("Sizer:ModelRefreshIntervalMinutes", DefaultModelRefreshIntervalMinutes);
    }
}