using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace TradingBot.Backtest
{
    /// <summary>
    /// Order type enumeration for backtest simulation
    /// </summary>
    public enum OrderType
    {
        Market,
        Limit,
        Stop,
        StopLimit
    }

    /// <summary>
    /// Order side enumeration for backtest simulation
    /// </summary>
    public enum OrderSide
    {
        Buy,
        Sell
    }

    /// <summary>
    /// Time in force enumeration for backtest simulation
    /// </summary>
    public enum TimeInForce
    {
        Day,
        GTC, // Good Till Cancelled
        IOC, // Immediate Or Cancel
        FOK  // Fill Or Kill
    }

    /// <summary>
    /// Order specification for execution simulation
    /// Contains all details needed to simulate realistic order execution
    /// </summary>
    public record OrderSpec(
        string Symbol,
        OrderType Type,
        OrderSide Side,
        decimal Quantity,
        decimal? LimitPrice,
        decimal? StopPrice,
        TimeInForce TimeInForce,
        DateTime PlacedAt
    );

    /// <summary>
    /// Result of simulated order execution
    /// Models realistic fill outcomes including partial fills and slippage
    /// </summary>
    public record FillResult(
        string OrderId,
        decimal FilledQuantity,
        decimal FillPrice,
        DateTime FillTime,
        decimal Slippage,
        string Reason
    );

    /// <summary>
    /// Current state of the execution simulator
    /// Tracks positions, OCO brackets, and PnL using existing Runtime.Accounting
    /// </summary>
    public class SimState
    {
        /// <summary>
        /// Current position size (positive for long, negative for short)
        /// </summary>
        public decimal Position { get; set; }

        /// <summary>
        /// Average entry price for current position
        /// </summary>
        public decimal AverageEntryPrice { get; set; }

        /// <summary>
        /// Unrealized PnL for current position
        /// </summary>
        public decimal UnrealizedPnL { get; set; }

        /// <summary>
        /// Realized PnL from closed trades
        /// </summary>
        public decimal RealizedPnL { get; set; }

        /// <summary>
        /// Active OCO bracket orders (stop-loss and take-profit pairs)
        /// </summary>
        public List<(OrderSpec StopLoss, OrderSpec TakeProfit)> ActiveBrackets { get; set; } = new();

        /// <summary>
        /// Last known market price for PnL calculation
        /// </summary>
        public decimal LastMarketPrice { get; set; }

        /// <summary>
        /// Commission paid on executed orders
        /// </summary>
        public decimal TotalCommissions { get; set; }

        /// <summary>
        /// Number of round-trip trades completed
        /// </summary>
        public int RoundTripTrades { get; set; }
    }

    /// <summary>
    /// Interface for simulating realistic order execution during backtesting
    /// Replaces random order execution with market impact simulation
    /// </summary>
    public interface IExecutionSimulator
    {
        /// <summary>
        /// Simulate execution of a market or limit order
        /// Applies realistic slippage based on spread width and market conditions
        /// </summary>
        /// <param name="order">Order specification to execute</param>
        /// <param name="currentQuote">Current market quote for execution simulation</param>
        /// <param name="state">Current simulation state</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Fill result with realistic execution details</returns>
        Task<FillResult?> SimulateOrderAsync(
            OrderSpec order,
            Quote currentQuote,
            SimState state,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Check if any stop-loss or take-profit orders should be triggered
        /// Processes OCO bracket orders based on market movement
        /// </summary>
        /// <param name="currentQuote">Current market quote</param>
        /// <param name="state">Current simulation state</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>List of triggered bracket orders</returns>
        Task<List<FillResult>> CheckBracketTriggersAsync(
            Quote currentQuote,
            SimState state,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Update position and PnL based on market movement
        /// Called for each new market quote to maintain accurate state
        /// </summary>
        /// <param name="currentQuote">Current market quote</param>
        /// <param name="state">Current simulation state to update</param>
        void UpdatePositionPnL(Quote currentQuote, SimState state);

        /// <summary>
        /// Reset simulation state for new backtest run
        /// Clears positions, brackets, and PnL tracking
        /// </summary>
        /// <param name="state">State to reset</param>
        void ResetState(SimState state);
    }
}