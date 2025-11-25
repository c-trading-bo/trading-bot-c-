using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.Backtest.Services
{
    /// <summary>
    /// Mock implementation of IOrderService for backtesting
    /// Simulates order management (stop-loss modifications, trailing stops, breakeven) without a real broker
    /// Integrates with SimState and IExecutionSimulator to provide realistic backtest behavior
    /// </summary>
    public class BacktestOrderService : IOrderService
    {
        private readonly ILogger<BacktestOrderService> _logger;
        private readonly IExecutionSimulator _executionSimulator;
        
        // Shared state with backtest harness
        private SimState? _currentSimState;
        private Quote? _currentQuote;
        
        // Track simulated orders and positions
        private readonly Dictionary<string, BacktestOrder> _orders = new();
        private readonly Dictionary<string, BacktestPosition> _positions = new();
        private int _orderIdCounter = 1;
        
        public BacktestOrderService(
            ILogger<BacktestOrderService> logger,
            IExecutionSimulator executionSimulator)
        {
            _logger = logger;
            _executionSimulator = executionSimulator;
        }
        
        /// <summary>
        /// Set the current simulation state and quote - called by BacktestHarnessService
        /// This allows the order service to access real-time backtest state
        /// </summary>
        public void SetBacktestContext(SimState simState, Quote currentQuote)
        {
            _currentSimState = simState;
            _currentQuote = currentQuote;
        }
        
        /// <summary>
        /// Process pending orders and update positions based on current market price
        /// This should be called on each tick/bar during backtest
        /// </summary>
        public async Task ProcessMarketUpdateAsync()
        {
            if (_currentSimState == null || _currentQuote == null)
                return;
                
            // Check stop-loss and take-profit triggers
            var triggeredOrders = new List<BacktestOrder>();
            
            foreach (var order in _orders.Values.Where(o => o.Status == OrderStatus.Pending))
            {
                bool triggered = false;
                
                if (order.OrderType == "Stop" && order.StopPrice.HasValue)
                {
                    // Stop order triggers when price crosses stop level
                    if (order.Side == "Buy" && _currentQuote.Last >= order.StopPrice.Value)
                        triggered = true;
                    else if (order.Side == "Sell" && _currentQuote.Last <= order.StopPrice.Value)
                        triggered = true;
                }
                
                if (triggered)
                {
                    triggeredOrders.Add(order);
                }
            }
            
            // Execute triggered orders
            foreach (var order in triggeredOrders)
            {
                await ExecuteOrderAsync(order);
            }
            
            // Update position unrealized PnL
            foreach (var position in _positions.Values)
            {
                if (_currentQuote.Symbol == position.Symbol)
                {
                    var isLong = position.Side == "Long";
                    var currentPrice = _currentQuote.Last;
                    position.UnrealizedPnL = isLong
                        ? position.Quantity * (currentPrice - position.AveragePrice)
                        : position.Quantity * (position.AveragePrice - currentPrice);
                }
            }
            
            await Task.CompletedTask;
        }
        
        private async Task ExecuteOrderAsync(BacktestOrder order)
        {
            if (_currentQuote == null || _currentSimState == null)
                return;
                
            // Determine fill price based on order type
            decimal fillPrice = order.OrderType switch
            {
                "Market" => order.Side == "Buy" ? _currentQuote.Ask : _currentQuote.Bid,
                "Stop" => _currentQuote.Last, // Stop becomes market order
                "Limit" => order.Price ?? _currentQuote.Last,
                _ => _currentQuote.Last
            };
            
            // Update order status
            order.Status = OrderStatus.Filled;
            order.FilledQuantity = order.Quantity;
            order.UpdatedAt = DateTimeOffset.UtcNow;
            
            _logger.LogInformation("📋 [BACKTEST-ORDER] Filled {OrderType} order {OrderId}: {Side} {Qty} {Symbol} @ {Price:F2}",
                order.OrderType, order.Id, order.Side, order.Quantity, order.Symbol, fillPrice);
            
            await Task.CompletedTask;
        }

        #region IOrderService Implementation

        public Task<bool> IsHealthyAsync()
        {
            return Task.FromResult(true);
        }

        public Task<string> GetStatusAsync()
        {
            return Task.FromResult("Backtest OrderService Active");
        }

        public Task<string> PlaceMarketOrderAsync(string symbol, string side, int quantity, string? tag = null)
        {
            var orderId = $"BT-{_orderIdCounter++:D6}";
            var order = new BacktestOrder
            {
                Id = orderId,
                Symbol = symbol,
                Side = side,
                Quantity = quantity,
                OrderType = "Market",
                Status = OrderStatus.Pending,
                Tag = tag,
                ConfigSnapshotId = "backtest",
                CreatedAt = DateTimeOffset.UtcNow
            };
            
            _orders[orderId] = order;
            
            _logger.LogDebug("📋 [BACKTEST-ORDER] Placed market order {OrderId}: {Side} {Qty} {Symbol}",
                orderId, side, quantity, symbol);
            
            return Task.FromResult(orderId);
        }

        public Task<string> PlaceLimitOrderAsync(string symbol, string side, int quantity, decimal price, string? tag = null)
        {
            var orderId = $"BT-{_orderIdCounter++:D6}";
            var order = new BacktestOrder
            {
                Id = orderId,
                Symbol = symbol,
                Side = side,
                Quantity = quantity,
                Price = price,
                OrderType = "Limit",
                Status = OrderStatus.Pending,
                Tag = tag,
                ConfigSnapshotId = "backtest",
                CreatedAt = DateTimeOffset.UtcNow
            };
            
            _orders[orderId] = order;
            
            _logger.LogDebug("📋 [BACKTEST-ORDER] Placed limit order {OrderId}: {Side} {Qty} {Symbol} @ {Price:F2}",
                orderId, side, quantity, symbol, price);
            
            return Task.FromResult(orderId);
        }

        public Task<string> PlaceStopOrderAsync(string symbol, string side, int quantity, decimal stopPrice, string? tag = null)
        {
            var orderId = $"BT-{_orderIdCounter++:D6}";
            var order = new BacktestOrder
            {
                Id = orderId,
                Symbol = symbol,
                Side = side,
                Quantity = quantity,
                StopPrice = stopPrice,
                OrderType = "Stop",
                Status = OrderStatus.Pending,
                Tag = tag,
                ConfigSnapshotId = "backtest",
                CreatedAt = DateTimeOffset.UtcNow
            };
            
            _orders[orderId] = order;
            
            _logger.LogInformation("🛑 [BACKTEST-ORDER] Placed stop order {OrderId}: {Side} {Qty} {Symbol} @ {Stop:F2}",
                orderId, side, quantity, symbol, stopPrice);
            
            return Task.FromResult(orderId);
        }

        public Task<bool> CancelOrderAsync(string orderId)
        {
            if (_orders.TryGetValue(orderId, out var order))
            {
                if (order.Status == OrderStatus.Pending)
                {
                    order.Status = OrderStatus.Cancelled;
                    order.UpdatedAt = DateTimeOffset.UtcNow;
                    
                    _logger.LogInformation("❌ [BACKTEST-ORDER] Cancelled order {OrderId}: {Side} {Symbol} @ {Price}",
                        orderId, order.Side, order.Symbol, order.StopPrice ?? order.Price);
                    
                    return Task.FromResult(true);
                }
            }
            
            return Task.FromResult(false);
        }

        public Task<bool> ModifyOrderAsync(string orderId, int? quantity = null, decimal? price = null)
        {
            if (_orders.TryGetValue(orderId, out var order))
            {
                if (order.Status == OrderStatus.Pending)
                {
                    if (quantity.HasValue)
                        order.Quantity = quantity.Value;
                    if (price.HasValue)
                        order.Price = price.Value;
                    
                    order.UpdatedAt = DateTimeOffset.UtcNow;
                    
                    _logger.LogInformation("✏️ [BACKTEST-ORDER] Modified order {OrderId}: Qty={Qty}, Price={Price:F2}",
                        orderId, order.Quantity, order.Price);
                    
                    return Task.FromResult(true);
                }
            }
            
            return Task.FromResult(false);
        }

        public Task<OrderStatus> GetOrderStatusAsync(string orderId)
        {
            if (_orders.TryGetValue(orderId, out var order))
            {
                return Task.FromResult(order.Status);
            }
            
            return Task.FromResult(OrderStatus.Rejected);
        }

        public Task<bool> ClosePositionAsync(string positionId)
        {
            if (_positions.TryGetValue(positionId, out var position))
            {
                _logger.LogInformation("🔚 [BACKTEST-ORDER] Closing position {PositionId}: {Symbol} {Side} {Qty}",
                    positionId, position.Symbol, position.Side, position.Quantity);
                
                // Remove position
                _positions.Remove(positionId);
                
                return Task.FromResult(true);
            }
            
            _logger.LogWarning("⚠️ [BACKTEST-ORDER] Position {PositionId} not found for close", positionId);
            return Task.FromResult(false);
        }

        public Task<bool> ClosePositionAsync(string positionId, int quantity, CancellationToken cancellationToken = default)
        {
            if (_positions.TryGetValue(positionId, out var position))
            {
                _logger.LogInformation("🔚 [BACKTEST-ORDER] Partially closing position {PositionId}: {Symbol} {Qty}/{Total}",
                    positionId, position.Symbol, quantity, position.Quantity);
                
                if (quantity >= position.Quantity)
                {
                    _positions.Remove(positionId);
                }
                else
                {
                    position.Quantity -= quantity;
                }
                
                return Task.FromResult(true);
            }
            
            return Task.FromResult(false);
        }

        public Task<bool> ModifyStopLossAsync(string positionId, decimal stopPrice)
        {
            if (_positions.TryGetValue(positionId, out var position))
            {
                var oldStop = position.StopLoss;
                position.StopLoss = stopPrice;
                
                _logger.LogInformation("🛑 [BACKTEST-ORDER] Modified stop-loss for {PositionId}: {Old:F2} → {New:F2}",
                    positionId, oldStop, stopPrice);
                
                // Update or create stop order
                var stopOrder = _orders.Values.FirstOrDefault(o => 
                    o.Tag == $"SL-{positionId}" && o.Status == OrderStatus.Pending);
                
                if (stopOrder != null)
                {
                    // Cancel old stop
                    stopOrder.Status = OrderStatus.Cancelled;
                }
                
                // Place new stop order
                var side = position.Side == "Long" ? "Sell" : "Buy";
                PlaceStopOrderAsync(position.Symbol, side, position.Quantity, stopPrice, $"SL-{positionId}");
                
                return Task.FromResult(true);
            }
            
            _logger.LogWarning("⚠️ [BACKTEST-ORDER] Position {PositionId} not found for stop modification", positionId);
            return Task.FromResult(false);
        }

        public Task<bool> ModifyTakeProfitAsync(string positionId, decimal takeProfitPrice)
        {
            if (_positions.TryGetValue(positionId, out var position))
            {
                var oldTP = position.TakeProfit;
                position.TakeProfit = takeProfitPrice;
                
                _logger.LogInformation("🎯 [BACKTEST-ORDER] Modified take-profit for {PositionId}: {Old:F2} → {New:F2}",
                    positionId, oldTP, takeProfitPrice);
                
                // Update or create take-profit order
                var tpOrder = _orders.Values.FirstOrDefault(o => 
                    o.Tag == $"TP-{positionId}" && o.Status == OrderStatus.Pending);
                
                if (tpOrder != null)
                {
                    tpOrder.Price = takeProfitPrice;
                }
                else
                {
                    var side = position.Side == "Long" ? "Sell" : "Buy";
                    PlaceLimitOrderAsync(position.Symbol, side, position.Quantity, takeProfitPrice, $"TP-{positionId}");
                }
                
                return Task.FromResult(true);
            }
            
            return Task.FromResult(false);
        }

        public Task<List<Position>> GetPositionsAsync()
        {
            var positions = _positions.Values.Select(p => new Position
            {
                Id = p.Id,
                Symbol = p.Symbol,
                Side = p.Side,
                Quantity = p.Quantity,
                AveragePrice = p.AveragePrice,
                UnrealizedPnL = p.UnrealizedPnL,
                RealizedPnL = p.RealizedPnL,
                ConfigSnapshotId = p.ConfigSnapshotId,
                OpenTime = p.OpenTime,
                StopLoss = p.StopLoss,
                TakeProfit = p.TakeProfit,
                DecisionId = p.DecisionId,
                Strategy = p.Strategy,
                EntryTime = p.EntryTime
            }).ToList();
            
            return Task.FromResult(positions);
        }

        public Task<Position?> GetPositionAsync(string positionId)
        {
            if (_positions.TryGetValue(positionId, out var p))
            {
                var position = new Position
                {
                    Id = p.Id,
                    Symbol = p.Symbol,
                    Side = p.Side,
                    Quantity = p.Quantity,
                    AveragePrice = p.AveragePrice,
                    UnrealizedPnL = p.UnrealizedPnL,
                    RealizedPnL = p.RealizedPnL,
                    ConfigSnapshotId = p.ConfigSnapshotId,
                    OpenTime = p.OpenTime,
                    StopLoss = p.StopLoss,
                    TakeProfit = p.TakeProfit,
                    DecisionId = p.DecisionId,
                    Strategy = p.Strategy,
                    EntryTime = p.EntryTime
                };
                
                return Task.FromResult<Position?>(position);
            }
            
            return Task.FromResult<Position?>(null);
        }

        public Task<List<Order>> GetActiveOrdersAsync()
        {
            var orders = _orders.Values
                .Where(o => o.Status == OrderStatus.Pending || o.Status == OrderStatus.PartiallyFilled)
                .Select(o => new Order
                {
                    Id = o.Id,
                    Symbol = o.Symbol,
                    Side = o.Side,
                    Quantity = o.Quantity,
                    FilledQuantity = o.FilledQuantity,
                    Price = o.Price,
                    StopPrice = o.StopPrice,
                    OrderType = o.OrderType,
                    Status = o.Status,
                    Tag = o.Tag,
                    ConfigSnapshotId = o.ConfigSnapshotId,
                    CreatedAt = o.CreatedAt,
                    UpdatedAt = o.UpdatedAt
                })
                .ToList();
            
            return Task.FromResult(orders);
        }
        
        /// <summary>
        /// Register a new position (called when entry order is filled)
        /// </summary>
        public void RegisterPosition(string positionId, string symbol, string side, int quantity, decimal avgPrice, 
            decimal? stopLoss = null, decimal? takeProfit = null, string? strategy = null, string? decisionId = null)
        {
            var position = new BacktestPosition
            {
                Id = positionId,
                Symbol = symbol,
                Side = side,
                Quantity = quantity,
                AveragePrice = avgPrice,
                ConfigSnapshotId = "backtest",
                OpenTime = DateTimeOffset.UtcNow,
                EntryTime = DateTimeOffset.UtcNow,
                StopLoss = stopLoss,
                TakeProfit = takeProfit,
                Strategy = strategy,
                DecisionId = decisionId
            };
            
            _positions[positionId] = position;
            
            _logger.LogInformation("📍 [BACKTEST-ORDER] Registered position {PositionId}: {Symbol} {Side} {Qty} @ {Price:F2}",
                positionId, symbol, side, quantity, avgPrice);
        }
        
        /// <summary>
        /// Clear all orders and positions for new backtest run
        /// </summary>
        public void Reset()
        {
            _orders.Clear();
            _positions.Clear();
            _orderIdCounter = 1;
            _logger.LogInformation("🔄 [BACKTEST-ORDER] Reset - all orders and positions cleared");
        }

        #endregion
    }
    
    /// <summary>
    /// Internal model for tracking backtest orders
    /// </summary>
    internal class BacktestOrder
    {
        public required string Id { get; set; }
        public required string Symbol { get; set; }
        public required string Side { get; set; }
        public int Quantity { get; set; }
        public int FilledQuantity { get; set; }
        public decimal? Price { get; set; }
        public decimal? StopPrice { get; set; }
        public required string OrderType { get; set; }
        public OrderStatus Status { get; set; }
        public string? Tag { get; set; }
        public required string ConfigSnapshotId { get; set; }
        public DateTimeOffset CreatedAt { get; set; }
        public DateTimeOffset? UpdatedAt { get; set; }
    }
    
    /// <summary>
    /// Internal model for tracking backtest positions
    /// </summary>
    internal class BacktestPosition
    {
        public required string Id { get; set; }
        public required string Symbol { get; set; }
        public required string Side { get; set; }
        public int Quantity { get; set; }
        public decimal AveragePrice { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public decimal RealizedPnL { get; set; }
        public required string ConfigSnapshotId { get; set; }
        public DateTimeOffset OpenTime { get; set; }
        public decimal? StopLoss { get; set; }
        public decimal? TakeProfit { get; set; }
        public string? DecisionId { get; set; }
        public string? Strategy { get; set; }
        public DateTimeOffset EntryTime { get; set; }
    }
}
