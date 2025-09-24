using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of roll configuration
    /// Replaces hardcoded futures contract roll parameters
    /// </summary>
    public class RollConfigService : IRollConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<RollConfigService> _logger;

        public RollConfigService(IConfiguration config, ILogger<RollConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public int GetRollDaysBeforeExpiration() => 
            _config.GetValue("Roll:DaysBeforeExpiration", 3);

        public long GetMinVolumeForRoll() => 
            _config.GetValue("Roll:MinVolumeForRoll", 10000L);

        public double GetMaxRollSpreadTicks() => 
            _config.GetValue("Roll:MaxRollSpreadTicks", 2.0);

        public bool EnableAutomaticRoll() => 
            _config.GetValue("Roll:EnableAutomaticRoll", true);

        public TimeSpan GetRollWindowStartUtc() => 
            TimeSpan.ParseExact(_config.GetValue("Roll:RollWindowStartUtc", "13:30"), @"hh\:mm", null);

        public TimeSpan GetRollWindowEndUtc() => 
            TimeSpan.ParseExact(_config.GetValue("Roll:RollWindowEndUtc", "14:30"), @"hh\:mm", null);

        public string GetRollHintsForSymbol(string symbol) => symbol?.ToUpper() switch
        {
            "ES" => _config.GetValue("Roll:Hints:ES", "H,M,U,Z"),
            "NQ" => _config.GetValue("Roll:Hints:NQ", "H,M,U,Z"),
            "YM" => _config.GetValue("Roll:Hints:YM", "H,M,U,Z"),
            "RTY" => _config.GetValue("Roll:Hints:RTY", "H,M,U,Z"),
            _ => _config.GetValue("Roll:Hints:Default", "H,M,U,Z")
        };

        public bool ForceRollOnLastTradingDay() => 
            _config.GetValue("Roll:ForceRollOnLastTradingDay", true);

        public int GetRollNotificationLeadHours() => 
            _config.GetValue("Roll:RollNotificationLeadHours", 24);
    }
}