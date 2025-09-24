using System;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for futures contract roll management
    /// Replaces hardcoded roll timing and parameters
    /// </summary>
    public interface IRollConfig
    {
        /// <summary>
        /// Days before expiration to start rolling contracts
        /// </summary>
        int GetRollDaysBeforeExpiration();

        /// <summary>
        /// Minimum volume threshold for rolling to new contract
        /// </summary>
        long GetMinVolumeForRoll();

        /// <summary>
        /// Maximum spread between front and back month for rolling
        /// </summary>
        double GetMaxRollSpreadTicks();

        /// <summary>
        /// Whether to automatically roll positions
        /// </summary>
        bool EnableAutomaticRoll();

        /// <summary>
        /// Time window for roll execution (start time UTC)
        /// </summary>
        TimeSpan GetRollWindowStartUtc();

        /// <summary>
        /// Time window for roll execution (end time UTC)
        /// </summary>
        TimeSpan GetRollWindowEndUtc();

        /// <summary>
        /// Roll hints for specific symbols (comma-separated)
        /// </summary>
        string GetRollHintsForSymbol(string symbol);

        /// <summary>
        /// Whether to force roll on last trading day
        /// </summary>
        bool ForceRollOnLastTradingDay();

        /// <summary>
        /// Roll notification lead time in hours
        /// </summary>
        int GetRollNotificationLeadHours();
    }
}