using System;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace TradingBot.Backtest
{
    /// <summary>
    /// Decision log entry capturing trading decisions made by the system
    /// Replaces fake decision tracking with real strategy decisions
    /// </summary>
    public record DecisionLog(
        DateTime Timestamp,
        string Symbol,
        string Strategy,
        TradingAction Decision,
        decimal Confidence,
        string Rationale,
        decimal? EntryPrice,
        decimal? StopLoss,
        decimal? TakeProfit,
        decimal RiskAmount,
        string MarketConditions
    );

    /// <summary>
    /// Fill log entry capturing actual order executions
    /// Records real simulated trade executions for analysis
    /// </summary>
    public record FillLog(
        DateTime Timestamp,
        string OrderId,
        string Symbol,
        OrderSide Side,
        decimal Quantity,
        decimal FillPrice,
        decimal Slippage,
        decimal Commission,
        string FillReason,
        decimal RealizedPnL,
        decimal UnrealizedPnL,
        decimal TotalPnL
    );

    /// <summary>
    /// Position closure log capturing complete trade lifecycle
    /// Records round-trip trade metrics for performance analysis
    /// </summary>
    public record PositionClosureLog(
        DateTime OpenTime,
        DateTime CloseTime,
        string Symbol,
        OrderSide Side,
        decimal Quantity,
        decimal EntryPrice,
        decimal ExitPrice,
        decimal GrossPnL,
        decimal Commission,
        decimal NetPnL,
        TimeSpan HoldTime,
        string ExitReason,
        decimal MaxFavorableExcursion,
        decimal MaxAdverseExcursion
    );

    /// <summary>
    /// Interface for capturing and storing backtest metrics
    /// Replaces fake metric generation with real trade tracking
    /// </summary>
    public interface IMetricSink
    {
        /// <summary>
        /// Record a trading decision made by the strategy
        /// Captures all decisions including no-action decisions
        /// </summary>
        /// <param name="decision">Decision details to record</param>
        /// <param name="cancellationToken">Cancellation token</param>
        Task RecordDecisionAsync(DecisionLog decision, CancellationToken cancellationToken = default);

        /// <summary>
        /// Record an order fill execution
        /// Captures actual simulated trade executions
        /// </summary>
        /// <param name="fill">Fill details to record</param>
        /// <param name="cancellationToken">Cancellation token</param>
        Task RecordFillAsync(FillLog fill, CancellationToken cancellationToken = default);

        /// <summary>
        /// Record a complete position closure (round-trip trade)
        /// Captures full trade lifecycle metrics
        /// </summary>
        /// <param name="closure">Position closure details to record</param>
        /// <param name="cancellationToken">Cancellation token</param>
        Task RecordPositionClosureAsync(PositionClosureLog closure, CancellationToken cancellationToken = default);

        /// <summary>
        /// Flush all pending metrics to persistent storage
        /// Ensures data is saved before backtest completion
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        Task FlushAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Get the storage location for generated metrics
        /// Used for post-backtest analysis and reporting
        /// </summary>
        /// <returns>Path to stored metrics files</returns>
        string GetStoragePath();
    }
}