using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of event tempering configuration
    /// Replaces hardcoded event handling and lockout parameters
    /// </summary>
    public class EventTemperingConfigService : IEventTemperingConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<EventTemperingConfigService> _logger;

        public EventTemperingConfigService(IConfiguration config, ILogger<EventTemperingConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public int GetNewsLockoutMinutesBefore() => 
            _config.GetValue("EventTempering:NewsLockoutMinutesBefore", 5);

        public int GetNewsLockoutMinutesAfter() => 
            _config.GetValue("EventTempering:NewsLockoutMinutesAfter", 2);

        public int GetHighImpactEventLockoutMinutes() => 
            _config.GetValue("EventTempering:HighImpactEventLockoutMinutes", 15);

        public bool ReducePositionSizesBeforeEvents() => 
            _config.GetValue("EventTempering:ReducePositionSizesBeforeEvents", true);

        public double GetEventPositionSizeReduction() => 
            _config.GetValue("EventTempering:EventPositionSizeReduction", 0.5);

        public bool EnableHolidayTradingLockout() => 
            _config.GetValue("EventTempering:EnableHolidayTradingLockout", true);

        public bool EnableEarningsLockout() => 
            _config.GetValue("EventTempering:EnableEarningsLockout", true);

        public bool EnableFomcLockout() => 
            _config.GetValue("EventTempering:EnableFomcLockout", true);

        public int GetPreMarketTradingMinutes() => 
            _config.GetValue("EventTempering:PreMarketTradingMinutes", 30);
    }
}