using System;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for trading sessions and calendar management
    /// Replaces hardcoded timezone and session parameters
    /// </summary>
    public interface ISessionConfig
    {
        /// <summary>
        /// Primary trading timezone (e.g., "America/New_York", "America/Chicago")
        /// </summary>
        string GetPrimaryTimezone();

        /// <summary>
        /// Regular trading hours start time (local time)
        /// </summary>
        TimeSpan GetRthStartTime();

        /// <summary>
        /// Regular trading hours end time (local time)
        /// </summary>
        TimeSpan GetRthEndTime();

        /// <summary>
        /// Extended hours trading allowed
        /// </summary>
        bool AllowExtendedHours();

        /// <summary>
        /// Overnight session start time
        /// </summary>
        TimeSpan GetOvernightStartTime();

        /// <summary>
        /// Overnight session end time
        /// </summary>
        TimeSpan GetOvernightEndTime();

        /// <summary>
        /// Maintenance window start time (UTC)
        /// </summary>
        TimeSpan GetMaintenanceWindowStartUtc();

        /// <summary>
        /// Maintenance window duration in minutes
        /// </summary>
        int GetMaintenanceWindowDurationMinutes();

        /// <summary>
        /// Whether trading is allowed on weekends
        /// </summary>
        bool AllowWeekendTrading();
    }
}