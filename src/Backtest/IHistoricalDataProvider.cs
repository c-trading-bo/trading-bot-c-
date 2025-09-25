using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Backtest
{
    /// <summary>
    /// Quote data structure for historical market data
    /// Maps to market data structure in the trading system
    /// </summary>
    public record Quote(
        DateTime Time,
        string Symbol,
        decimal Bid,
        decimal Ask,
        decimal Last,
        int Volume,
        decimal Open,
        decimal High,
        decimal Low,
        decimal Close
    );

    /// <summary>
    /// Interface for providing historical market data to backtesting engine
    /// Connects to existing historical data sources (TopstepX, databases)
    /// </summary>
    public interface IHistoricalDataProvider
    {
        /// <summary>
        /// Load historical quotes for a specific symbol and time range
        /// Used to replay market conditions during backtesting
        /// </summary>
        /// <param name="symbol">Trading symbol (e.g., "ES", "MES")</param>
        /// <param name="startTime">Start of historical data range</param>
        /// <param name="endTime">End of historical data range</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Stream of historical quotes in chronological order</returns>
        Task<IAsyncEnumerable<Quote>> GetHistoricalQuotesAsync(
            string symbol, 
            DateTime startTime, 
            DateTime endTime, 
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Check if historical data is available for the specified time range
        /// Prevents backtests from running with insufficient data
        /// </summary>
        /// <param name="symbol">Trading symbol</param>
        /// <param name="startTime">Start of required data range</param>
        /// <param name="endTime">End of required data range</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>True if data is available, false otherwise</returns>
        Task<bool> IsDataAvailableAsync(
            string symbol, 
            DateTime startTime, 
            DateTime endTime, 
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Get available data range for a specific symbol
        /// Used to determine valid backtesting periods
        /// </summary>
        /// <param name="symbol">Trading symbol</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Tuple of earliest and latest available data timestamps</returns>
        Task<(DateTime EarliestData, DateTime LatestData)> GetDataRangeAsync(
            string symbol, 
            CancellationToken cancellationToken = default);
    }
}