using Microsoft.Extensions.Configuration;
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

        // Trading bracket constants
        private const double DefaultTakeProfitAtrMultipleValue = 2.0;
        private const double DefaultStopLossAtrMultipleValue = 1.0;
        private const double MinRewardRiskRatioValue = 1.2;
        private const double MaxRewardRiskRatioValue = 5.0;
        private const double TrailingStopAtrMultipleValue = 0.8;

        public BracketConfigService(IConfiguration config)
        {
            _config = config;
        }

        public double GetDefaultTakeProfitAtrMultiple() => 
            _config.GetValue("Bracket:DefaultTakeProfitAtrMultiple", DefaultTakeProfitAtrMultipleValue);

        public double GetDefaultStopLossAtrMultiple() => 
            _config.GetValue("Bracket:DefaultStopLossAtrMultiple", DefaultStopLossAtrMultipleValue);

        public double GetMinRewardRiskRatio() => 
            _config.GetValue("Bracket:MinRewardRiskRatio", MinRewardRiskRatioValue);

        public double GetMaxRewardRiskRatio() => 
            _config.GetValue("Bracket:MaxRewardRiskRatio", MaxRewardRiskRatioValue);

        public bool EnableTrailingStops() => 
            _config.GetValue("Bracket:EnableTrailingStops", false);

        public double GetTrailingStopAtrMultiple() => 
            _config.GetValue("Bracket:TrailingStopAtrMultiple", TrailingStopAtrMultipleValue);

        public string GetDefaultBracketMode() => 
            _config.GetValue("Bracket:DefaultBracketMode", "OCO");

        public bool BracketOrdersReduceOnly() => 
            _config.GetValue("Bracket:BracketOrdersReduceOnly", true);

        public string GetPartialFillHandling() => 
            _config.GetValue("Bracket:PartialFillHandling", "SCALE_BRACKET");

        // Additional methods needed by consuming code
        public double GetDefaultStopAtrMultiple() =>
            _config.GetValue("Bracket:DefaultStopAtrMultiple", DefaultStopLossAtrMultipleValue);

        public double GetDefaultTargetAtrMultiple() =>
            _config.GetValue("Bracket:DefaultTargetAtrMultiple", DefaultTakeProfitAtrMultipleValue);

        public bool EnableTrailingStop => 
            _config.GetValue("Bracket:EnableTrailingStop", false);

        public bool ReduceOnlyMode => 
            _config.GetValue("Bracket:ReduceOnlyMode", true);
    }
}