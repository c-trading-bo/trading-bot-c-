using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of bracket configuration
    /// Replaces hardcoded TP/SL distances and bracket parameters
    /// </summary>
    public class BracketConfigService : IBracketConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<BracketConfigService> _logger;

        public BracketConfigService(IConfiguration config, ILogger<BracketConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public double GetDefaultTakeProfitAtrMultiple() => 
            _config.GetValue("Bracket:DefaultTakeProfitAtrMultiple", 2.0);

        public double GetDefaultStopLossAtrMultiple() => 
            _config.GetValue("Bracket:DefaultStopLossAtrMultiple", 1.0);

        public double GetMinRewardRiskRatio() => 
            _config.GetValue("Bracket:MinRewardRiskRatio", 1.2);

        public double GetMaxRewardRiskRatio() => 
            _config.GetValue("Bracket:MaxRewardRiskRatio", 5.0);

        public bool EnableTrailingStops() => 
            _config.GetValue("Bracket:EnableTrailingStops", false);

        public double GetTrailingStopAtrMultiple() => 
            _config.GetValue("Bracket:TrailingStopAtrMultiple", 0.8);

        public string GetDefaultBracketMode() => 
            _config.GetValue("Bracket:DefaultBracketMode", "OCO");

        public bool BracketOrdersReduceOnly() => 
            _config.GetValue("Bracket:BracketOrdersReduceOnly", true);

        public string GetPartialFillHandling() => 
            _config.GetValue("Bracket:PartialFillHandling", "SCALE_BRACKET");
    }
}