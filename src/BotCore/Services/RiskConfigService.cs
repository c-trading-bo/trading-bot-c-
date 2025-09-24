using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of risk configuration
    /// Replaces hardcoded risk limits and position sizing parameters
    /// </summary>
    public class RiskConfigService : IRiskConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<RiskConfigService> _logger;

        public RiskConfigService(IConfiguration config, ILogger<RiskConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public decimal GetMaxDailyLossUsd() => 
            _config.GetValue("Risk:MaxDailyLossUsd", 1000.0m);

        public decimal GetMaxWeeklyLossUsd() => 
            _config.GetValue("Risk:MaxWeeklyLossUsd", 3000.0m);

        public double GetRiskPerTradePercent() => 
            _config.GetValue("Risk:RiskPerTradePercent", 0.0025); // 0.25%

        public decimal GetFixedRiskPerTradeUsd() => 
            _config.GetValue("Risk:FixedRiskPerTradeUsd", 100.0m);

        public int GetMaxOpenPositions() => 
            _config.GetValue("Risk:MaxOpenPositions", 1);

        public int GetMaxConsecutiveLosses() => 
            _config.GetValue("Risk:MaxConsecutiveLosses", 3);

        public double GetCvarConfidenceLevel() => 
            _config.GetValue("Risk:CvarConfidenceLevel", 0.95);

        public double GetCvarTargetRMultiple() => 
            _config.GetValue("Risk:CvarTargetRMultiple", 0.65);

        public double GetRegimeDrawdownMultiplier(string regimeType) => regimeType?.ToLower() switch
        {
            "bull" => _config.GetValue("Risk:RegimeMultipliers:Bull", 1.0),
            "bear" => _config.GetValue("Risk:RegimeMultipliers:Bear", 0.8),
            "sideways" => _config.GetValue("Risk:RegimeMultipliers:Sideways", 0.9),
            "volatile" => _config.GetValue("Risk:RegimeMultipliers:Volatile", 0.7),
            _ => _config.GetValue("Risk:RegimeMultipliers:Default", 0.85)
        };
    }
}