using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of session configuration
    /// Replaces hardcoded timezone and trading session parameters
    /// </summary>
    public class SessionConfigService : ISessionConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<SessionConfigService> _logger;

        public SessionConfigService(IConfiguration config, ILogger<SessionConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public string GetPrimaryTimezone() => 
            _config.GetValue("Session:PrimaryTimezone", "America/Chicago");

        public TimeSpan GetRthStartTime() => 
            TimeSpan.ParseExact(_config.GetValue("Session:RthStartTime", "09:30"), @"hh\:mm", null);

        public TimeSpan GetRthEndTime() => 
            TimeSpan.ParseExact(_config.GetValue("Session:RthEndTime", "16:00"), @"hh\:mm", null);

        public bool AllowExtendedHours() => 
            _config.GetValue("Session:AllowExtendedHours", false);

        public TimeSpan GetOvernightStartTime() => 
            TimeSpan.ParseExact(_config.GetValue("Session:OvernightStartTime", "20:00"), @"hh\:mm", null);

        public TimeSpan GetOvernightEndTime() => 
            TimeSpan.ParseExact(_config.GetValue("Session:OvernightEndTime", "02:00"), @"hh\:mm", null);

        public TimeSpan GetMaintenanceWindowStartUtc() => 
            TimeSpan.ParseExact(_config.GetValue("Session:MaintenanceWindowStartUtc", "05:00"), @"hh\:mm", null);

        public int GetMaintenanceWindowDurationMinutes() => 
            _config.GetValue("Session:MaintenanceWindowDurationMinutes", 30);

        public bool AllowWeekendTrading() => 
            _config.GetValue("Session:AllowWeekendTrading", false);
    }
}