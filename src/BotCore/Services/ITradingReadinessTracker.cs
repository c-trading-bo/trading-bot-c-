using System;
using System.Threading.Tasks;

namespace BotCore.Services
{
    /// <summary>
    /// Interface for tracking trading readiness state
    /// Allows different components to update the BarsSeen counter and other readiness metrics
    /// </summary>
    public interface ITradingReadinessTracker
    {
        /// <summary>
        /// Increment the bars seen counter
        /// </summary>
        void IncrementBarsSeen(int count = 1);

        /// <summary>
        /// Increment seeded bars counter
        /// </summary>
        void IncrementSeededBars(int count = 1);

        /// <summary>
        /// Increment live ticks counter
        /// </summary>
        void IncrementLiveTicks(int count = 1);

        /// <summary>
        /// Update last market data timestamp
        /// </summary>
        void UpdateLastMarketDataTimestamp();

        /// <summary>
        /// Get current readiness context
        /// </summary>
        TradingReadinessContext GetReadinessContext();

        /// <summary>
        /// Validate if system is ready for trading
        /// </summary>
        Task<ReadinessValidationResult> ValidateReadinessAsync();

        /// <summary>
        /// Reset all counters (for testing or restart scenarios)
        /// </summary>
        void Reset();
        
        /// <summary>
        /// Mark the system as ready for trading with proper logging
        /// </summary>
        void SetSystemReady();
    }
}