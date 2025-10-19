using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Service for loading and validating historical market data seed files.
    /// </summary>
    public interface IHistoricalDataSeedService
    {
        /// <summary>
        /// Try to load and apply historical seed data with validation.
        /// Auto-refreshes if data is stale (during maintenance window: 5 PM ET daily, skip weekends).
        /// </summary>
        /// <param name="symbols">Symbols to load seed data for (e.g., "ES", "NQ")</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Result containing loaded bars or error message</returns>
        Task<SeedApplyResult> TryApplySeedAsync(string[] symbols, CancellationToken cancellationToken = default);
    }
}
