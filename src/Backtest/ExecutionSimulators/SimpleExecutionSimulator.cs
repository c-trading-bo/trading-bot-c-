using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.Backtest.ExecutionSimulators
{
    /// <summary>
    /// Simple execution simulator with realistic slippage and market impact
    /// Much more realistic than random fills - can be enhanced later with order book simulation
    /// </summary>
    public class SimpleExecutionSimulator : IExecutionSimulator
    {
        private readonly ILogger<SimpleExecutionSimulator> _logger;
        private readonly Random _random;
        
        public SimpleExecutionSimulator(ILogger<SimpleExecutionSimulator> logger)
        {
            _logger = logger;
            _random = new Random();
        }

        /// <summary>
        /// Simulate realistic order execution with market impact and slippage
        /// </summary>
        public async Task<FillResult?> SimulateOrderAsync(
            OrderSpec order, 
            Quote currentQuote, 
            SimState state, 
            CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask; // Satisfy async requirement

            var orderId = Guid.NewGuid().ToString("N")[..8];
            
            // Calculate realistic fill price based on order type
            var fillPrice = CalculateFillPrice(order, currentQuote);
            if (fillPrice <= 0)
            {
                _logger.LogWarning("Order {OrderId} could not be filled - invalid price", orderId);
                return null;
            }

            // Calculate slippage
            var expectedPrice = order.Type == OrderType.Market 
                ? (order.Side == OrderSide.Buy ? currentQuote.Ask : currentQuote.Bid)
                : order.LimitPrice ?? currentQuote.Last;
            
            var slippage = Math.Abs(fillPrice - expectedPrice);

            // Update simulation state
            UpdateStateWithFill(state, order, fillPrice);

            _logger.LogDebug("Filled order {OrderId}: {Side} {Quantity} {Symbol} at {Price:F4} (slippage: {Slippage:F4})",
                orderId, order.Side, order.Quantity, order.Symbol, fillPrice, slippage);

            return new FillResult(
                OrderId: orderId,
                FilledQuantity: order.Side == OrderSide.Buy ? order.Quantity : -order.Quantity,
                FillPrice: fillPrice,
                FillTime: order.PlacedAt,
                Slippage: slippage,
                Reason: $"{order.Type} order executed"
            );
        }

        /// <summary>
        /// Check for bracket order triggers (stop-loss and take-profit)
        /// </summary>
        public async Task<List<FillResult>> CheckBracketTriggersAsync(
            Quote currentQuote, 
            SimState state, 
            CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask; // Satisfy async requirement
            
            var triggeredFills = new List<FillResult>();
            var triggeredBrackets = new List<(OrderSpec StopLoss, OrderSpec TakeProfit)>();

            foreach (var bracket in state.ActiveBrackets)
            {
                var stopTriggered = CheckStopTrigger(bracket.StopLoss, currentQuote);
                var profitTriggered = CheckTakeProfitTrigger(bracket.TakeProfit, currentQuote);

                if (stopTriggered || profitTriggered)
                {
                    var triggerOrder = stopTriggered ? bracket.StopLoss : bracket.TakeProfit;
                    var fillResult = await SimulateOrderAsync(triggerOrder, currentQuote, state, cancellationToken);
                    
                    if (fillResult != null)
                    {
                        triggeredFills.Add(fillResult);
                        triggeredBrackets.Add(bracket);
                        
                        _logger.LogDebug("Bracket order triggered: {OrderType} at {Price:F4}", 
                            stopTriggered ? "Stop-Loss" : "Take-Profit", fillResult.FillPrice);
                    }
                }
            }

            // Remove triggered brackets
            foreach (var bracket in triggeredBrackets)
            {
                state.ActiveBrackets.Remove(bracket);
            }

            return triggeredFills;
        }

        /// <summary>
        /// Update position and PnL based on market movement
        /// </summary>
        public void UpdatePositionPnL(Quote currentQuote, SimState state)
        {
            if (state.Position == 0)
            {
                state.UnrealizedPnL = 0m;
                state.LastMarketPrice = currentQuote.Last;
                return;
            }

            var previousPrice = state.LastMarketPrice;
            var currentPrice = currentQuote.Last;
            
            if (previousPrice > 0)
            {
                // Calculate PnL change
                var priceChange = currentPrice - previousPrice;
                var pnlChange = state.Position * priceChange;
                
                // Update unrealized PnL
                state.UnrealizedPnL = state.Position * (currentPrice - state.AverageEntryPrice);
            }

            state.LastMarketPrice = currentPrice;
        }

        /// <summary>
        /// Reset simulation state for new backtest run
        /// </summary>
        public void ResetState(SimState state)
        {
            state.Position = 0m;
            state.AverageEntryPrice = 0m;
            state.UnrealizedPnL = 0m;
            state.RealizedPnL = 0m;
            state.TotalCommissions = 0m;
            state.RoundTripTrades = 0;
            state.LastMarketPrice = 0m;
            state.ActiveBrackets.Clear();
        }

        private decimal CalculateFillPrice(OrderSpec order, Quote currentQuote)
        {
            switch (order.Type)
            {
                case OrderType.Market:
                    // Market orders get filled at bid/ask with small slippage
                    var marketPrice = order.Side == OrderSide.Buy ? currentQuote.Ask : currentQuote.Bid;
                    var slippagePercent = (_random.NextDouble() * 0.002) - 0.001; // -0.1% to +0.1%
                    return marketPrice * (1m + (decimal)slippagePercent);

                case OrderType.Limit:
                    // Limit orders only fill if price is favorable
                    if (order.LimitPrice.HasValue)
                    {
                        if (order.Side == OrderSide.Buy && currentQuote.Ask <= order.LimitPrice.Value)
                            return Math.Min(order.LimitPrice.Value, currentQuote.Ask);
                        if (order.Side == OrderSide.Sell && currentQuote.Bid >= order.LimitPrice.Value)
                            return Math.Max(order.LimitPrice.Value, currentQuote.Bid);
                    }
                    return 0m; // No fill

                case OrderType.Stop:
                    // Stop orders become market orders when triggered
                    if (order.StopPrice.HasValue && IsStopTriggered(order, currentQuote))
                    {
                        return CalculateFillPrice(new OrderSpec(
                            order.Symbol, OrderType.Market, order.Side, order.Quantity,
                            null, null, order.TimeInForce, order.PlacedAt), currentQuote);
                    }
                    return 0m; // Not triggered

                default:
                    return currentQuote.Last;
            }
        }

        private bool IsStopTriggered(OrderSpec stopOrder, Quote currentQuote)
        {
            if (!stopOrder.StopPrice.HasValue) return false;

            return stopOrder.Side == OrderSide.Buy 
                ? currentQuote.Last >= stopOrder.StopPrice.Value
                : currentQuote.Last <= stopOrder.StopPrice.Value;
        }

        private bool CheckStopTrigger(OrderSpec stopOrder, Quote currentQuote)
        {
            return stopOrder.Type == OrderType.Stop && IsStopTriggered(stopOrder, currentQuote);
        }

        private bool CheckTakeProfitTrigger(OrderSpec profitOrder, Quote currentQuote)
        {
            return profitOrder.Type == OrderType.Limit && 
                   profitOrder.LimitPrice.HasValue &&
                   ((profitOrder.Side == OrderSide.Buy && currentQuote.Ask <= profitOrder.LimitPrice.Value) ||
                    (profitOrder.Side == OrderSide.Sell && currentQuote.Bid >= profitOrder.LimitPrice.Value));
        }

        private void UpdateStateWithFill(SimState state, OrderSpec order, decimal fillPrice)
        {
            var fillQuantity = order.Side == OrderSide.Buy ? order.Quantity : -order.Quantity;
            var oldPosition = state.Position;
            var newPosition = oldPosition + fillQuantity;

            // Update average entry price
            if (oldPosition == 0)
            {
                // Opening new position
                state.AverageEntryPrice = fillPrice;
                state.Position = newPosition;
            }
            else if (Math.Sign(oldPosition) == Math.Sign(newPosition))
            {
                // Adding to existing position
                var totalNotional = (oldPosition * state.AverageEntryPrice) + (fillQuantity * fillPrice);
                state.AverageEntryPrice = totalNotional / newPosition;
                state.Position = newPosition;
            }
            else
            {
                // Reducing or reversing position
                var reducedQuantity = Math.Min(Math.Abs(fillQuantity), Math.Abs(oldPosition));
                var realizedPnL = oldPosition > 0 
                    ? reducedQuantity * (fillPrice - state.AverageEntryPrice)
                    : -reducedQuantity * (fillPrice - state.AverageEntryPrice);

                state.RealizedPnL += realizedPnL;
                state.Position = newPosition;

                if (Math.Abs(oldPosition) == Math.Abs(fillQuantity))
                {
                    // Position closed
                    state.RoundTripTrades++;
                    state.AverageEntryPrice = 0m;
                }

                if (newPosition != 0 && Math.Sign(oldPosition) != Math.Sign(newPosition))
                {
                    // Position reversed
                    state.AverageEntryPrice = fillPrice;
                }
            }

            // Add commission
            state.TotalCommissions += 2.50m * Math.Abs(fillQuantity); // $2.50 per contract
        }
    }
}