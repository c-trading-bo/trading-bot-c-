using System;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for event tempering and lockouts
    /// Replaces hardcoded news/event handling parameters
    /// </summary>
    public interface IEventTemperingConfig
    {
        /// <summary>
        /// Minutes before news events to stop trading
        /// </summary>
        int GetNewsLockoutMinutesBefore();

        /// <summary>
        /// Minutes after news events to resume trading
        /// </summary>
        int GetNewsLockoutMinutesAfter();

        /// <summary>
        /// High-impact events lockout duration in minutes
        /// </summary>
        int GetHighImpactEventLockoutMinutes();

        /// <summary>
        /// Whether to reduce position sizes before events
        /// </summary>
        bool ReducePositionSizesBeforeEvents();

        /// <summary>
        /// Position size reduction factor before events (0.0-1.0)
        /// </summary>
        double GetEventPositionSizeReduction();

        /// <summary>
        /// Holiday trading lockout enabled
        /// </summary>
        bool EnableHolidayTradingLockout();

        /// <summary>
        /// Earnings announcement lockout enabled
        /// </summary>
        bool EnableEarningsLockout();

        /// <summary>
        /// FOMC meeting lockout enabled
        /// </summary>
        bool EnableFomcLockout();

        /// <summary>
        /// Minutes before market open to start trading
        /// </summary>
        int GetPreMarketTradingMinutes();
    }
}