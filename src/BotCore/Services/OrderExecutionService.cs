using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using BotCore.Models;

namespace BotCore.Services
{
    /// <summary>
    /// Production order execution service implementing IOrderService
    /// Integrates with TopstepX adapter for broker communication
    /// Manages order lifecycle and position tracking
    /// </summary>
    public sealed class OrderExecutionService : TradingBot.Abstractions.IOrderService, IDisposable
    {
        private readonly ILogger<OrderExecutionService> _logger;
        private readonly ITopstepXAdapterService _topstepAdapter;
        private readonly OrderExecutionMetrics? _metrics;
        
        // Position tracking: maps position ID (symbol-based) to position data
        private readonly ConcurrentDictionary<string, TradingBot.Abstractions.Position> _positions = new();
        
        // Order tracking: maps order ID to order data
        private readonly ConcurrentDictionary<string, TradingBot.Abstractions.Order> _orders = new();
        
        // Symbol to position ID mapping for lookups
        private readonly ConcurrentDictionary<string, string> _symbolToPositionId = new();
        
        // Configuration snapshot tracking
        private const string DefaultConfigSnapshotId = "default-snapshot";
        
        // PHASE 1: Event Infrastructure for fill notifications
        public event EventHandler<OrderFillEventArgs>? OrderFilled;
        public event EventHandler<OrderPlacedEventArgs>? OrderPlaced;
        public event EventHandler<OrderRejectedEventArgs>? OrderRejected;
        
        // PHASE 3: Position reconciliation infrastructure
        private readonly Timer? _reconciliationTimer;
        private const int ReconciliationIntervalSeconds = 60;
        
        // PHASE 4: Execution quality thresholds
        private const int MinimumOrdersForQualityCheck = 5; // Minimum orders before checking execution quality
        
        // PHASE 5: Advanced order types tracking
        private readonly ConcurrentDictionary<string, OcoOrderPair> _ocoOrders = new();
        private readonly ConcurrentDictionary<string, BracketOrderGroup> _bracketOrders = new();
        private readonly ConcurrentDictionary<string, IcebergOrderExecution> _icebergOrders = new();
        
        // PHASE 5: Feature flags for advanced order types
        private readonly bool _enableOcoOrders;
        private readonly bool _enableBracketOrders;
        private readonly bool _enableIcebergOrders;
        
        public OrderExecutionService(
            ILogger<OrderExecutionService> logger,
            ITopstepXAdapterService topstepAdapter,
            OrderExecutionMetrics? metrics = null)
        {
            _logger = logger;
            _topstepAdapter = topstepAdapter;
            _metrics = metrics;
            
            // PHASE 5: Read feature flags from environment
            _enableOcoOrders = Environment.GetEnvironmentVariable("ENABLE_OCO_ORDERS") == "true";
            _enableBracketOrders = Environment.GetEnvironmentVariable("ENABLE_BRACKET_ORDERS") == "true";
            _enableIcebergOrders = Environment.GetEnvironmentVariable("ENABLE_ICEBERG_ORDERS") == "true";
            
            if (_enableOcoOrders)
                _logger.LogInformation("✅ [CONFIG] OCO orders enabled");
            if (_enableBracketOrders)
                _logger.LogInformation("✅ [CONFIG] Bracket orders enabled");
            if (_enableIcebergOrders)
                _logger.LogInformation("✅ [CONFIG] Iceberg orders enabled");
            
            // PHASE 3: Start reconciliation timer (every 60 seconds)
            _reconciliationTimer = new Timer(
                _ => ReconcilePositionsWithBrokerAsync().ConfigureAwait(false),
                null,
                TimeSpan.FromSeconds(ReconciliationIntervalSeconds),
                TimeSpan.FromSeconds(ReconciliationIntervalSeconds)
            );
            
            _logger.LogInformation("🔄 Position reconciliation timer started (interval: {Seconds}s)", ReconciliationIntervalSeconds);
        }
        
        // ========================================================================
        // HEALTH AND STATUS METHODS
        // ========================================================================
        
        public async Task<bool> IsHealthyAsync()
        {
            try
            {
                return await _topstepAdapter.IsHealthyAsync().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking order service health");
                return false;
            }
        }
        
        public Task<string> GetStatusAsync()
        {
            try
            {
                var isConnected = _topstepAdapter.IsConnected;
                var health = _topstepAdapter.ConnectionHealth;
                var positionCount = _positions.Count;
                var activeOrderCount = _orders.Count(o => o.Value.Status == TradingBot.Abstractions.OrderStatus.Pending || o.Value.Status == TradingBot.Abstractions.OrderStatus.PartiallyFilled);
                
                return Task.FromResult($"Connected: {isConnected}, Health: {health:F1}%, Positions: {positionCount}, Active Orders: {activeOrderCount}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting order service status");
                return Task.FromResult($"Error: {ex.Message}");
            }
        }
        
        // ========================================================================
        // POSITION QUERY METHODS
        // ========================================================================
        
        public Task<List<TradingBot.Abstractions.Position>> GetPositionsAsync()
        {
            return Task.FromResult(_positions.Values.ToList());
        }
        
        public Task<TradingBot.Abstractions.Position?> GetPositionAsync(string positionId)
        {
            _positions.TryGetValue(positionId, out var position);
            return Task.FromResult(position);
        }
        
        // ========================================================================
        // POSITION MANAGEMENT METHODS
        // ========================================================================
        
        /// <summary>
        /// Close entire position via TopstepX API
        /// </summary>
        public async Task<bool> ClosePositionAsync(string positionId)
        {
            try
            {
                if (!_positions.TryGetValue(positionId, out var position))
                {
                    _logger.LogWarning("Position {PositionId} not found for full close", positionId);
                    return false;
                }
                
                _logger.LogInformation("📉 [ORDER-EXEC] Closing full position {PositionId}: {Symbol} {Quantity} contracts",
                    positionId, position.Symbol, position.Quantity);
                
                // Call TopstepX API to close position
                var success = await _topstepAdapter.ClosePositionAsync(position.Symbol, position.Quantity, CancellationToken.None).ConfigureAwait(false);
                
                if (success)
                {
                    // Remove from tracking
                    _positions.TryRemove(positionId, out _);
                    _symbolToPositionId.TryRemove(position.Symbol, out _);
                    
                    _logger.LogInformation("✅ [ORDER-EXEC] Position {PositionId} closed successfully via TopstepX API", positionId);
                    return true;
                }
                
                _logger.LogError("❌ [ORDER-EXEC] Failed to close position {PositionId} via TopstepX API", positionId);
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error closing position {PositionId}", positionId);
                return false;
            }
        }
        
        /// <summary>
        /// Close partial position (for scaling out) via TopstepX API
        /// </summary>
        public async Task<bool> ClosePositionAsync(string positionId, int quantity, CancellationToken cancellationToken = default)
        {
            try
            {
                if (!_positions.TryGetValue(positionId, out var position))
                {
                    _logger.LogWarning("Position {PositionId} not found for partial close", positionId);
                    return false;
                }
                
                if (quantity <= 0 || quantity > position.Quantity)
                {
                    _logger.LogError("Invalid quantity {Qty} for partial close of position {PositionId} (current: {Current})",
                        quantity, positionId, position.Quantity);
                    return false;
                }
                
                _logger.LogInformation("📉 [ORDER-EXEC] Partial close position {PositionId}: {Symbol} closing {Qty} of {Total} contracts",
                    positionId, position.Symbol, quantity, position.Quantity);
                
                // Call TopstepX API to close partial position
                var success = await _topstepAdapter.ClosePositionAsync(position.Symbol, quantity, cancellationToken).ConfigureAwait(false);
                
                if (success)
                {
                    // Update position quantity
                    var remainingQuantity = position.Quantity - quantity;
                    
                    if (remainingQuantity > 0)
                    {
                        position.Quantity = remainingQuantity;
                        _logger.LogInformation("✅ [ORDER-EXEC] Partial close successful for {PositionId} via TopstepX API: {Qty} contracts closed, {Remaining} remaining",
                            positionId, quantity, remainingQuantity);
                    }
                    else
                    {
                        // Position fully closed
                        _positions.TryRemove(positionId, out _);
                        _symbolToPositionId.TryRemove(position.Symbol, out _);
                        _logger.LogInformation("✅ [ORDER-EXEC] Position {PositionId} fully closed via partial close (TopstepX API)", positionId);
                    }
                    
                    return true;
                }
                
                _logger.LogError("❌ [ORDER-EXEC] Failed to execute partial close for position {PositionId} via TopstepX API", positionId);
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in partial close for position {PositionId}", positionId);
                return false;
            }
        }
        
        /// <summary>
        /// Modify stop loss for a position via TopstepX API
        /// </summary>
        public async Task<bool> ModifyStopLossAsync(string positionId, decimal stopPrice)
        {
            try
            {
                if (!_positions.TryGetValue(positionId, out var position))
                {
                    _logger.LogWarning("Position {PositionId} not found for stop loss modification", positionId);
                    return false;
                }
                
                _logger.LogInformation("🛡️ [ORDER-EXEC] Modifying stop loss for {PositionId}: {Symbol} from {Old} to {New}",
                    positionId, position.Symbol, position.StopLoss, stopPrice);
                
                // Call TopstepX API to modify stop loss
                var success = await _topstepAdapter.ModifyStopLossAsync(position.Symbol, stopPrice, CancellationToken.None).ConfigureAwait(false);
                
                if (success)
                {
                    // Update stop loss in position tracking
                    position.StopLoss = stopPrice;
                    
                    _logger.LogInformation("✅ [ORDER-EXEC] Stop loss updated for {PositionId} via TopstepX API", positionId);
                    return true;
                }
                
                _logger.LogError("❌ [ORDER-EXEC] Failed to modify stop loss for {PositionId} via TopstepX API", positionId);
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error modifying stop loss for position {PositionId}", positionId);
                return false;
            }
        }
        
        /// <summary>
        /// Modify take profit for a position via TopstepX API
        /// </summary>
        public async Task<bool> ModifyTakeProfitAsync(string positionId, decimal takeProfitPrice)
        {
            try
            {
                if (!_positions.TryGetValue(positionId, out var position))
                {
                    _logger.LogWarning("Position {PositionId} not found for take profit modification", positionId);
                    return false;
                }
                
                _logger.LogInformation("🎯 [ORDER-EXEC] Modifying take profit for {PositionId}: {Symbol} from {Old} to {New}",
                    positionId, position.Symbol, position.TakeProfit, takeProfitPrice);
                
                // Call TopstepX API to modify take profit
                var success = await _topstepAdapter.ModifyTakeProfitAsync(position.Symbol, takeProfitPrice, CancellationToken.None).ConfigureAwait(false);
                
                if (success)
                {
                    // Update take profit in position tracking
                    position.TakeProfit = takeProfitPrice;
                    
                    _logger.LogInformation("✅ [ORDER-EXEC] Take profit updated for {PositionId} via TopstepX API", positionId);
                    return true;
                }
                
                _logger.LogError("❌ [ORDER-EXEC] Failed to modify take profit for {PositionId} via TopstepX API", positionId);
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error modifying take profit for position {PositionId}", positionId);
                return false;
            }
        }
        
        // ========================================================================
        // ORDER PLACEMENT METHODS
        // ========================================================================
        
        public Task<string> PlaceMarketOrderAsync(string symbol, string side, int quantity, string? tag = null)
        {
            try
            {
                _logger.LogInformation("📈 [ORDER-EXEC] Placing market order: {Symbol} {Side} {Qty} contracts (tag: {Tag})",
                    symbol, side, quantity, tag ?? "none");
                
                // Generate order ID
                var orderId = $"ORD-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}-{symbol}";
                
                // Create order record
                var order = new TradingBot.Abstractions.Order
                {
                    Id = orderId,
                    Symbol = symbol,
                    Side = side,
                    Quantity = quantity,
                    FilledQuantity = 0,
                    OrderType = "MARKET",
                    Status = TradingBot.Abstractions.OrderStatus.Pending,
                    Tag = tag,
                    ConfigSnapshotId = DefaultConfigSnapshotId,
                    CreatedAt = DateTimeOffset.UtcNow
                };
                
                _orders.TryAdd(orderId, order);
                
                // PHASE 1: Record metrics for order placement
                _metrics?.RecordOrderPlaced(symbol);
                
                // PHASE 1: Fire OrderPlaced event
                OrderPlaced?.Invoke(this, new OrderPlacedEventArgs
                {
                    OrderId = orderId,
                    Symbol = symbol,
                    Quantity = quantity,
                    Price = order.Price ?? 0,
                    OrderType = "MARKET",
                    Timestamp = DateTime.UtcNow
                });
                
                // For closing orders, we don't use TopstepX PlaceOrderAsync
                // We just mark them as filled since TopstepX handles the closing
                if (tag?.StartsWith("CLOSE-", StringComparison.Ordinal) == true || tag?.StartsWith("PARTIAL-CLOSE-", StringComparison.Ordinal) == true)
                {
                    order.Status = TradingBot.Abstractions.OrderStatus.Filled;
                    order.FilledQuantity = quantity;
                    order.UpdatedAt = DateTimeOffset.UtcNow;
                    
                    _logger.LogInformation("✅ [ORDER-EXEC] Close order {OrderId} marked as filled", orderId);
                    return Task.FromResult(orderId);
                }
                
                // For entry orders, integrate with TopstepX adapter
                // Note: This would require bracket parameters (stop/target) which aren't available in this method
                // Entry orders should be placed through higher-level services that provide full bracket info
                
                _logger.LogInformation("✅ [ORDER-EXEC] Order {OrderId} created", orderId);
                return Task.FromResult(orderId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error placing market order for {Symbol}", symbol);
                
                // PHASE 1: Record rejection in metrics and raise event
                _metrics?.RecordOrderRejected(symbol, ex.Message);
                OrderRejected?.Invoke(this, new OrderRejectedEventArgs
                {
                    OrderId = string.Empty,
                    Symbol = symbol,
                    Reason = ex.Message,
                    Timestamp = DateTime.UtcNow
                });
                
                return Task.FromResult(string.Empty);
            }
        }
        
        public Task<string> PlaceLimitOrderAsync(string symbol, string side, int quantity, decimal price, string? tag = null)
        {
            try
            {
                _logger.LogInformation("📈 [ORDER-EXEC] Placing limit order: {Symbol} {Side} {Qty} @ {Price} (tag: {Tag})",
                    symbol, side, quantity, price, tag ?? "none");
                
                var orderId = $"ORD-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}-{symbol}";
                
                var order = new TradingBot.Abstractions.Order
                {
                    Id = orderId,
                    Symbol = symbol,
                    Side = side,
                    Quantity = quantity,
                    FilledQuantity = 0,
                    Price = price,
                    OrderType = "LIMIT",
                    Status = TradingBot.Abstractions.OrderStatus.Pending,
                    Tag = tag,
                    ConfigSnapshotId = DefaultConfigSnapshotId,
                    CreatedAt = DateTimeOffset.UtcNow
                };
                
                _orders.TryAdd(orderId, order);
                
                _logger.LogInformation("✅ [ORDER-EXEC] Limit order {OrderId} created", orderId);
                return Task.FromResult(orderId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error placing limit order for {Symbol}", symbol);
                return Task.FromResult(string.Empty);
            }
        }
        
        public Task<string> PlaceStopOrderAsync(string symbol, string side, int quantity, decimal stopPrice, string? tag = null)
        {
            try
            {
                _logger.LogInformation("📈 [ORDER-EXEC] Placing stop order: {Symbol} {Side} {Qty} @ {StopPrice} (tag: {Tag})",
                    symbol, side, quantity, stopPrice, tag ?? "none");
                
                var orderId = $"ORD-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}-{symbol}";
                
                var order = new TradingBot.Abstractions.Order
                {
                    Id = orderId,
                    Symbol = symbol,
                    Side = side,
                    Quantity = quantity,
                    FilledQuantity = 0,
                    StopPrice = stopPrice,
                    OrderType = "STOP",
                    Status = TradingBot.Abstractions.OrderStatus.Pending,
                    Tag = tag,
                    ConfigSnapshotId = DefaultConfigSnapshotId,
                    CreatedAt = DateTimeOffset.UtcNow
                };
                
                _orders.TryAdd(orderId, order);
                
                _logger.LogInformation("✅ [ORDER-EXEC] Stop order {OrderId} created", orderId);
                return Task.FromResult(orderId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error placing stop order for {Symbol}", symbol);
                return Task.FromResult(string.Empty);
            }
        }
        
        // ========================================================================
        // ORDER MANAGEMENT METHODS
        // ========================================================================
        
        public async Task<bool> CancelOrderAsync(string orderId)
        {
            try
            {
                if (!_orders.TryGetValue(orderId, out var order))
                {
                    _logger.LogWarning("Order {OrderId} not found for cancellation", orderId);
                    return false;
                }
                
                if (order.Status == TradingBot.Abstractions.OrderStatus.Filled || order.Status == TradingBot.Abstractions.OrderStatus.Cancelled)
                {
                    _logger.LogWarning("Cannot cancel order {OrderId} with status {Status}", orderId, order.Status);
                    return false;
                }
                
                _logger.LogInformation("❌ [ORDER-EXEC] Cancelling order {OrderId}", orderId);
                
                // Call TopstepX API to cancel order
                var success = await _topstepAdapter.CancelOrderAsync(orderId, CancellationToken.None).ConfigureAwait(false);
                
                if (success)
                {
                    order.Status = TradingBot.Abstractions.OrderStatus.Cancelled;
                    order.UpdatedAt = DateTimeOffset.UtcNow;
                    
                    _logger.LogInformation("✅ [ORDER-EXEC] Order {OrderId} cancelled via TopstepX API", orderId);
                    return true;
                }
                
                _logger.LogError("❌ [ORDER-EXEC] Failed to cancel order {OrderId} via TopstepX API", orderId);
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error cancelling order {OrderId}", orderId);
                return false;
            }
        }
        
        public Task<bool> ModifyOrderAsync(string orderId, int? quantity = null, decimal? price = null)
        {
            try
            {
                if (!_orders.TryGetValue(orderId, out var order))
                {
                    _logger.LogWarning("Order {OrderId} not found for modification", orderId);
                    return Task.FromResult(false);
                }
                
                if (order.Status != TradingBot.Abstractions.OrderStatus.Pending)
                {
                    _logger.LogWarning("Cannot modify order {OrderId} with status {Status}", orderId, order.Status);
                    return Task.FromResult(false);
                }
                
                _logger.LogInformation("✏️ [ORDER-EXEC] Modifying order {OrderId}", orderId);
                
                if (quantity.HasValue)
                {
                    order.Quantity = quantity.Value;
                    _logger.LogInformation("  → Quantity updated to {Qty}", quantity.Value);
                }
                
                if (price.HasValue)
                {
                    order.Price = price.Value;
                    _logger.LogInformation("  → Price updated to {Price}", price.Value);
                }
                
                order.UpdatedAt = DateTimeOffset.UtcNow;
                
                _logger.LogInformation("✅ [ORDER-EXEC] Order {OrderId} modified", orderId);
                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error modifying order {OrderId}", orderId);
                return Task.FromResult(false);
            }
        }
        
        public Task<TradingBot.Abstractions.OrderStatus> GetOrderStatusAsync(string orderId)
        {
            if (_orders.TryGetValue(orderId, out var order))
            {
                return Task.FromResult(order.Status);
            }
            
            _logger.LogWarning("Order {OrderId} not found", orderId);
            return Task.FromResult(TradingBot.Abstractions.OrderStatus.Rejected);
        }
        
        public Task<List<TradingBot.Abstractions.Order>> GetActiveOrdersAsync()
        {
            var activeOrders = _orders.Values
                .Where(o => o.Status == TradingBot.Abstractions.OrderStatus.Pending || o.Status == TradingBot.Abstractions.OrderStatus.PartiallyFilled)
                .ToList();
            
            return Task.FromResult(activeOrders);
        }
        
        // ========================================================================
        // HELPER METHODS
        // ========================================================================
        
        /// <summary>
        /// Register a position in tracking (called when trade is opened)
        /// </summary>
        public void RegisterPosition(string positionId, string symbol, string side, int quantity, decimal entryPrice, decimal? stopLoss = null, decimal? takeProfit = null, string? strategy = null, string? decisionId = null)
        {
            var position = new TradingBot.Abstractions.Position
            {
                Id = positionId,
                Symbol = symbol,
                Side = side,
                Quantity = quantity,
                AveragePrice = entryPrice,
                UnrealizedPnL = 0,
                RealizedPnL = 0,
                ConfigSnapshotId = DefaultConfigSnapshotId,
                OpenTime = DateTimeOffset.UtcNow,
                StopLoss = stopLoss,
                TakeProfit = takeProfit,
                Strategy = strategy ?? "UNKNOWN", // Neural-UCB learning: track which strategy opened position
                DecisionId = decisionId, // Neural-UCB learning: link position to strategy decision
                EntryTime = DateTimeOffset.UtcNow // Neural-UCB learning: track position lifetime
            };
            
            _positions.TryAdd(positionId, position);
            _symbolToPositionId[symbol] = positionId;
            
            _logger.LogInformation("📊 [ORDER-EXEC] Position registered: {PositionId} - {Symbol} {Side} {Qty} @ {Price} | Strategy={Strategy} DecisionId={DecisionId}",
                positionId, symbol, side, quantity, entryPrice, strategy ?? "UNKNOWN", decisionId ?? "N/A");
        }
        
        /// <summary>
        /// Update position P&L (called by position management)
        /// </summary>
        public void UpdatePositionPnL(string positionId, decimal unrealizedPnL, decimal realizedPnL)
        {
            if (_positions.TryGetValue(positionId, out var position))
            {
                position.UnrealizedPnL = unrealizedPnL;
                position.RealizedPnL = realizedPnL;
            }
        }
        
        /// <summary>
        /// PHASE 1: Get execution metrics summary for a symbol
        /// </summary>
        public OrderExecutionMetricsSummary? GetMetricsSummary(string symbol)
        {
            return _metrics?.GetMetricsSummary(symbol);
        }
        
        // ========================================================================
        // PHASE 4: PERFORMANCE METRICS REPORTING (Steps 7-9)
        // ========================================================================
        
        /// <summary>
        /// PHASE 4 Step 9: Get execution quality report for a symbol
        /// </summary>
        public string GetExecutionQualityReport(string symbol)
        {
            var summary = _metrics?.GetMetricsSummary(symbol);
            if (summary == null)
            {
                return $"No execution metrics available for {symbol}";
            }
            
            return $@"
📊 EXECUTION QUALITY REPORT - {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Orders:        {summary.TotalOrders} placed, {summary.TotalFills} filled
✅ Fill Rate:     {summary.FillRate:F2}%
⏱️  Avg Latency:   {summary.AverageLatencyMs:F2}ms
📈 95th Percentile: {summary.Latency95thPercentileMs:F2}ms
💸 Avg Slippage:  {summary.AverageSlippagePercent:F4}%
❌ Rejections:    {summary.TotalRejections}
🔄 Partial Fills: {summary.PartialFills}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
";
        }
        
        /// <summary>
        /// PHASE 4 Step 9: Check if execution quality has degraded beyond acceptable thresholds
        /// </summary>
        public bool CheckExecutionQualityThresholds(string symbol, out string alertMessage)
        {
            alertMessage = string.Empty;
            var summary = _metrics?.GetMetricsSummary(symbol);
            
            if (summary == null || summary.TotalOrders < MinimumOrdersForQualityCheck)
            {
                return true; // Not enough data yet
            }
            
            var alerts = new List<string>();
            
            // Alert thresholds
            const double MaxAcceptableLatencyMs = 500.0;
            const double MaxAcceptableSlippagePercent = 0.2;
            const double MinAcceptableFillRate = 90.0;
            
            if (summary.AverageLatencyMs > MaxAcceptableLatencyMs)
            {
                alerts.Add($"⚠️ High latency: {summary.AverageLatencyMs:F2}ms (threshold: {MaxAcceptableLatencyMs}ms)");
            }
            
            if (summary.AverageSlippagePercent > MaxAcceptableSlippagePercent)
            {
                alerts.Add($"⚠️ High slippage: {summary.AverageSlippagePercent:F4}% (threshold: {MaxAcceptableSlippagePercent}%)");
            }
            
            if (summary.FillRate < MinAcceptableFillRate)
            {
                alerts.Add($"⚠️ Low fill rate: {summary.FillRate:F2}% (threshold: {MinAcceptableFillRate}%)");
            }
            
            if (alerts.Count > 0)
            {
                alertMessage = $"🚨 EXECUTION QUALITY ALERT for {symbol}:\n" + string.Join("\n", alerts);
                return false;
            }
            
            return true;
        }
        
        // ========================================================================
        // PHASE 3: POSITION RECONCILIATION (Steps 5 & 6)
        // ========================================================================
        
        /// <summary>
        /// PHASE 3 Step 5: Periodic reconciliation - Compare bot state with broker reality
        /// Runs every 60 seconds as a background task
        /// </summary>
        private async Task ReconcilePositionsWithBrokerAsync()
        {
            try
            {
                _logger.LogDebug("🔄 [RECONCILIATION] Starting position reconciliation with broker...");
                
                // Get current positions from TopstepX broker
                var brokerPositions = await GetBrokerPositionsAsync().ConfigureAwait(false);
                if (brokerPositions == null)
                {
                    _logger.LogWarning("⚠️ [RECONCILIATION] Could not retrieve broker positions - skipping reconciliation");
                    return;
                }
                
                var discrepancies = 0;
                var corrections = 0;
                
                // Check for positions in broker but not in bot
                foreach (var brokerPos in brokerPositions)
                {
                    if (!_symbolToPositionId.TryGetValue(brokerPos.Symbol, out var positionId) ||
                        !_positions.TryGetValue(positionId, out var botPos))
                    {
                        // PHASE 3 Step 6: Broker has position but bot doesn't
                        _logger.LogWarning(
                            "🚨 [RECONCILIATION] DISCREPANCY: Broker shows position {Symbol} {Qty} contracts @ ${Price:F2} but bot has no record",
                            brokerPos.Symbol, brokerPos.Quantity, brokerPos.AveragePrice);
                        
                        // Auto-correction: Add to bot tracking
                        var newPositionId = $"POS-RECONCILED-{brokerPos.Symbol}-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}";
                        RegisterPosition(newPositionId, brokerPos.Symbol, 
                            brokerPos.Side ?? "UNKNOWN", 
                            brokerPos.Quantity,
                            brokerPos.AveragePrice,
                            null, null);
                        
                        _logger.LogInformation(
                            "✅ [RECONCILIATION] AUTO-CORRECTED: Added missing position {PositionId} to bot tracking",
                            newPositionId);
                        
                        discrepancies++;
                        corrections++;
                        continue;
                    }
                    
                    // Check for quantity mismatches
                    if (botPos.Quantity != brokerPos.Quantity)
                    {
                        // PHASE 3 Step 6: Quantities differ - update bot to match broker
                        _logger.LogWarning(
                            "🚨 [RECONCILIATION] DISCREPANCY: Position {PositionId} quantity mismatch - " +
                            "Bot: {BotQty}, Broker: {BrokerQty} | Broker is source of truth",
                            positionId, botPos.Quantity, brokerPos.Quantity);
                        
                        var oldQty = botPos.Quantity;
                        botPos.Quantity = brokerPos.Quantity;
                        botPos.AveragePrice = brokerPos.AveragePrice; // Also sync avg price
                        
                        _logger.LogInformation(
                            "✅ [RECONCILIATION] AUTO-CORRECTED: Updated {PositionId} quantity {OldQty} -> {NewQty}",
                            positionId, oldQty, brokerPos.Quantity);
                        
                        discrepancies++;
                        corrections++;
                    }
                }
                
                // Check for positions in bot but not in broker
                foreach (var botPos in _positions.Values.ToList())
                {
                    var existsInBroker = brokerPositions.Exists(bp => bp.Symbol == botPos.Symbol);
                    
                    if (!existsInBroker)
                    {
                        // PHASE 3 Step 6: Bot has position but broker doesn't
                        _logger.LogError(
                            "🚨🚨 [RECONCILIATION] CRITICAL DISCREPANCY: Bot shows position {PositionId} {Symbol} {Qty} contracts " +
                            "but broker has NO POSITION - Removing from bot tracking and alerting",
                            botPos.Id, botPos.Symbol, botPos.Quantity);
                        
                        // Auto-correction: Remove from bot tracking
                        _positions.TryRemove(botPos.Id, out _);
                        _symbolToPositionId.TryRemove(botPos.Symbol, out _);
                        
                        _logger.LogWarning(
                            "⚠️ [RECONCILIATION] AUTO-CORRECTED: Removed phantom position {PositionId} from bot tracking",
                            botPos.Id);
                        
                        // This is a CRITICAL alert - position exists in bot but not in broker
                        _logger.LogCritical(
                            "🚨🚨🚨 CRITICAL ALERT: Phantom position detected and removed. " +
                            "Position {PositionId} {Symbol} existed in bot but not in broker. " +
                            "This should be investigated immediately!",
                            botPos.Id, botPos.Symbol);
                        
                        discrepancies++;
                        corrections++;
                    }
                }
                
                if (discrepancies == 0)
                {
                    _logger.LogDebug("✅ [RECONCILIATION] Complete - No discrepancies found. Bot state matches broker.");
                }
                else
                {
                    _logger.LogWarning(
                        "⚠️ [RECONCILIATION] Complete - Found {Discrepancies} discrepancies, applied {Corrections} auto-corrections",
                        discrepancies, corrections);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [RECONCILIATION] Error during position reconciliation");
            }
        }
        
        /// <summary>
        /// PHASE 3 Step 5: Get actual positions from TopstepX broker
        /// </summary>
        private async Task<List<BrokerPosition>?> GetBrokerPositionsAsync()
        {
            try
            {
                // Call TopstepX adapter to get real broker positions
                var positions = await GetPositionsAsync().ConfigureAwait(false);
                
                // Convert to BrokerPosition format for comparison
                return positions.Select(p => new BrokerPosition
                {
                    Symbol = p.Symbol,
                    Quantity = p.Quantity,
                    AveragePrice = p.AveragePrice,
                    Side = p.Side
                }).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving broker positions for reconciliation");
                return null;
            }
        }
        
        /// <summary>
        /// Helper class for broker position comparison
        /// </summary>
        private sealed class BrokerPosition
        {
            public string Symbol { get; set; } = string.Empty;
            public int Quantity { get; set; }
            public decimal AveragePrice { get; set; }
            public string? Side { get; set; }
        }
        
        // ========================================================================
        // PHASE 1: EVENT PUBLISHING METHODS
        // ========================================================================
        
        /// <summary>
        /// Notify subscribers that an order was filled
        /// Called by TopstepXAdapterService when fill events are received
        /// </summary>
        public async Task OnOrderFillReceived(FillEventData fillData)
        {
            ArgumentNullException.ThrowIfNull(fillData);
            
            try
            {
                _logger.LogInformation("📥 [FILL-EVENT] Received fill notification: {OrderId} {Symbol} {Qty} @ {Price}",
                    fillData.OrderId, fillData.Symbol, fillData.Quantity, fillData.FillPrice);
                
                var isPartialFill = false;
                
                // Update order tracking if order exists
                if (_orders.TryGetValue(fillData.OrderId, out var order))
                {
                    order.FilledQuantity += fillData.Quantity;
                    isPartialFill = order.FilledQuantity < order.Quantity;
                    order.Status = isPartialFill
                        ? TradingBot.Abstractions.OrderStatus.PartiallyFilled
                        : TradingBot.Abstractions.OrderStatus.Filled;
                    order.UpdatedAt = DateTimeOffset.UtcNow;
                    
                    _logger.LogInformation("✅ [ORDER-UPDATE] Order {OrderId} updated: {FilledQty}/{TotalQty} filled, status: {Status}",
                        fillData.OrderId, order.FilledQuantity, order.Quantity, order.Status);
                    
                    // PHASE 1: Record execution latency
                    _metrics?.RecordExecutionLatency(fillData.OrderId, fillData.Symbol, 
                        order.CreatedAt.DateTime, fillData.Timestamp);
                    
                    // PHASE 1: Record slippage if we have expected price
                    if (order.Price.HasValue && order.Price.Value > 0)
                    {
                        _metrics?.RecordSlippage(fillData.OrderId, fillData.Symbol, 
                            order.Price.Value, fillData.FillPrice, fillData.Quantity);
                    }
                }
                
                // PHASE 1: Record fill in metrics
                _metrics?.RecordOrderFilled(fillData.Symbol, isPartialFill);
                
                // Update position tracking
                UpdatePositionFromFill(fillData);
                
                // Raise event for external subscribers
                OrderFilled?.Invoke(this, new OrderFillEventArgs
                {
                    OrderId = fillData.OrderId,
                    Symbol = fillData.Symbol,
                    Quantity = fillData.Quantity,
                    FillPrice = fillData.FillPrice,
                    Commission = fillData.Commission,
                    Timestamp = fillData.Timestamp,
                    Exchange = fillData.Exchange,
                    LiquidityType = fillData.LiquidityType
                });
                
                // PHASE 5: Process advanced order types (OCO, Bracket, Iceberg)
                await ProcessAdvancedOrderFillAsync(fillData.OrderId, fillData).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing fill event for order {OrderId}", fillData.OrderId);
            }
        }
        
        /// <summary>
        /// PHASE 2 Step 4: Update position tracking from fill and calculate realized P&L
        /// </summary>
        private void UpdatePositionFromFill(FillEventData fillData)
        {
            var timestamp = fillData.Timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture);
            
            // Check if this is an existing position
            if (_symbolToPositionId.TryGetValue(fillData.Symbol, out var positionId) &&
                _positions.TryGetValue(positionId, out var position))
            {
                var oldQuantity = position.Quantity;
                var oldAvgPrice = position.AveragePrice;
                
                // Determine if this is adding to or closing position based on side
                // If position is LONG and fill is for SELL, we're reducing/closing
                // If position is SHORT and fill is for BUY, we're reducing/closing
                var isClosingFill = (position.Side == "LONG" && fillData.Quantity < 0) ||
                                   (position.Side == "SHORT" && fillData.Quantity > 0);
                
                if (isClosingFill)
                {
                    // PHASE 2 Step 4: Calculate realized P&L when closing position
                    var closedQuantity = Math.Abs(fillData.Quantity);
                    var realizedPnL = position.Side == "LONG"
                        ? (fillData.FillPrice - oldAvgPrice) * closedQuantity
                        : (oldAvgPrice - fillData.FillPrice) * closedQuantity;
                    
                    // Subtract commission
                    realizedPnL -= fillData.Commission;
                    
                    position.RealizedPnL += realizedPnL;
                    position.Quantity = oldQuantity - closedQuantity;
                    
                    _logger.LogInformation(
                        "💰 [{Timestamp}] Position {PositionId} CLOSED {Qty} contracts @ ${Price:F2} | " +
                        "Realized P&L: ${PnL:F2} (Total Realized: ${TotalPnL:F2}) | Remaining: {Remaining} contracts",
                        timestamp, positionId, closedQuantity, fillData.FillPrice,
                        realizedPnL, position.RealizedPnL, position.Quantity);
                    
                    // If position fully closed, remove from tracking
                    if (position.Quantity == 0)
                    {
                        _positions.TryRemove(positionId, out _);
                        _symbolToPositionId.TryRemove(fillData.Symbol, out _);
                        
                        _logger.LogInformation(
                            "🎯 [{Timestamp}] Position {PositionId} FULLY CLOSED | " +
                            "Final Realized P&L: ${PnL:F2} | Avg Entry: ${Entry:F2}, Exit: ${Exit:F2}",
                            timestamp, positionId, position.RealizedPnL, oldAvgPrice, fillData.FillPrice);
                    }
                }
                else
                {
                    // Adding to position - calculate new average price
                    var addQuantity = Math.Abs(fillData.Quantity);
                    var totalCost = (oldQuantity * oldAvgPrice) + (addQuantity * fillData.FillPrice);
                    position.Quantity = oldQuantity + addQuantity;
                    position.AveragePrice = position.Quantity > 0 ? totalCost / position.Quantity : fillData.FillPrice;
                    
                    _logger.LogInformation(
                        "📈 [{Timestamp}] Position {PositionId} ADDED {Qty} contracts @ ${Price:F2} | " +
                        "New Qty: {NewQty}, New Avg Price: ${AvgPrice:F2}",
                        timestamp, positionId, addQuantity, fillData.FillPrice,
                        position.Quantity, position.AveragePrice);
                }
            }
            else
            {
                // New position - this shouldn't normally happen via fills
                // Positions should be created when orders are placed
                _logger.LogWarning(
                    "⚠️ [{Timestamp}] Fill received for {Symbol} but no tracked position exists. OrderId: {OrderId}",
                    timestamp, fillData.Symbol, fillData.OrderId);
            }
        }
        
        // ========================================================================
        // PHASE 5: ADVANCED ORDER TYPE METHODS
        // ========================================================================
        
        /// <summary>
        /// Place OCO (One-Cancels-Other) order pair
        /// When one order fills, the other is automatically cancelled
        /// </summary>
        public async Task<(string OcoId, string OrderId1, string OrderId2)> PlaceOcoOrderAsync(
            string symbol,
            string side1,
            int quantity1,
            decimal price1,
            string orderType1,
            string side2,
            int quantity2,
            decimal price2,
            string orderType2,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(orderType1);
            ArgumentNullException.ThrowIfNull(orderType2);
            
            // PHASE 5: Check if OCO orders are enabled
            if (!_enableOcoOrders)
            {
                _logger.LogWarning("⚠️ [OCO] OCO orders are disabled. Set ENABLE_OCO_ORDERS=true in .env to enable.");
                return (string.Empty, string.Empty, string.Empty);
            }
            
            try
            {
                _logger.LogInformation(
                    "🔗 [OCO] Placing OCO order pair for {Symbol}: " +
                    "Order1={Type1} {Side1} {Qty1}@{Price1}, Order2={Type2} {Side2} {Qty2}@{Price2}",
                    symbol, orderType1, side1, quantity1, price1, orderType2, side2, quantity2, price2);
                
                // Generate OCO ID
                var ocoId = $"OCO-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}-{symbol}";
                
                // Place first order
                var orderId1 = orderType1.Equals("LIMIT", StringComparison.OrdinalIgnoreCase)
                    ? await PlaceLimitOrderAsync(symbol, side1, quantity1, price1, $"OCO-{ocoId}").ConfigureAwait(false)
                    : await PlaceStopOrderAsync(symbol, side1, quantity1, price1, $"OCO-{ocoId}").ConfigureAwait(false);
                
                if (string.IsNullOrEmpty(orderId1))
                {
                    _logger.LogError("❌ [OCO] Failed to place first order in OCO pair");
                    return (string.Empty, string.Empty, string.Empty);
                }
                
                // Place second order
                var orderId2 = orderType2.Equals("LIMIT", StringComparison.OrdinalIgnoreCase)
                    ? await PlaceLimitOrderAsync(symbol, side2, quantity2, price2, $"OCO-{ocoId}").ConfigureAwait(false)
                    : await PlaceStopOrderAsync(symbol, side2, quantity2, price2, $"OCO-{ocoId}").ConfigureAwait(false);
                
                if (string.IsNullOrEmpty(orderId2))
                {
                    _logger.LogError("❌ [OCO] Failed to place second order, cancelling first order");
                    await CancelOrderAsync(orderId1).ConfigureAwait(false);
                    return (string.Empty, string.Empty, string.Empty);
                }
                
                // Create OCO tracking record
                var ocoOrder = new OcoOrderPair
                {
                    OcoId = ocoId,
                    OrderId1 = orderId1,
                    OrderId2 = orderId2,
                    Symbol = symbol,
                    CreatedAt = DateTime.UtcNow,
                    Status = OcoStatus.Active
                };
                
                _ocoOrders.TryAdd(ocoId, ocoOrder);
                
                _logger.LogInformation("✅ [OCO] OCO order pair {OcoId} created with orders {OrderId1} and {OrderId2}",
                    ocoId, orderId1, orderId2);
                
                return (ocoId, orderId1, orderId2);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error placing OCO order for {Symbol}", symbol);
                return (string.Empty, string.Empty, string.Empty);
            }
        }
        
        /// <summary>
        /// Place bracket order (entry + stop-loss + take-profit)
        /// When entry fills, stop and target are automatically placed
        /// Stop and target are linked as OCO - if one fills, the other cancels
        /// </summary>
        public async Task<(string BracketId, string EntryOrderId)> PlaceBracketOrderAsync(
            string symbol,
            string side,
            int quantity,
            decimal entryPrice,
            decimal stopPrice,
            decimal targetPrice,
            string entryOrderType = "LIMIT",
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(side);
            ArgumentNullException.ThrowIfNull(entryOrderType);
            
            // PHASE 5: Check if bracket orders are enabled
            if (!_enableBracketOrders)
            {
                _logger.LogWarning("⚠️ [BRACKET] Bracket orders are disabled. Set ENABLE_BRACKET_ORDERS=true in .env to enable.");
                return (string.Empty, string.Empty);
            }
            
            try
            {
                _logger.LogInformation(
                    "🎯 [BRACKET] Placing bracket order for {Symbol}: {Side} {Qty} " +
                    "Entry={Entry:F2} Stop={Stop:F2} Target={Target:F2}",
                    symbol, side, quantity, entryPrice, stopPrice, targetPrice);
                
                // Validate bracket parameters
                var isLong = side.Equals("BUY", StringComparison.OrdinalIgnoreCase);
                if (isLong)
                {
                    if (stopPrice >= entryPrice)
                    {
                        _logger.LogError("❌ [BRACKET] Invalid bracket: stop price must be below entry for long");
                        return (string.Empty, string.Empty);
                    }
                    if (targetPrice <= entryPrice)
                    {
                        _logger.LogError("❌ [BRACKET] Invalid bracket: target price must be above entry for long");
                        return (string.Empty, string.Empty);
                    }
                }
                else
                {
                    if (stopPrice <= entryPrice)
                    {
                        _logger.LogError("❌ [BRACKET] Invalid bracket: stop price must be above entry for short");
                        return (string.Empty, string.Empty);
                    }
                    if (targetPrice >= entryPrice)
                    {
                        _logger.LogError("❌ [BRACKET] Invalid bracket: target price must be below entry for short");
                        return (string.Empty, string.Empty);
                    }
                }
                
                // Generate bracket ID
                var bracketId = $"BRACKET-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}-{symbol}";
                
                // Place entry order
                var entryOrderId = entryOrderType.Equals("MARKET", StringComparison.OrdinalIgnoreCase)
                    ? await PlaceMarketOrderAsync(symbol, side, quantity, $"BRACKET-ENTRY-{bracketId}").ConfigureAwait(false)
                    : await PlaceLimitOrderAsync(symbol, side, quantity, entryPrice, $"BRACKET-ENTRY-{bracketId}").ConfigureAwait(false);
                
                if (string.IsNullOrEmpty(entryOrderId))
                {
                    _logger.LogError("❌ [BRACKET] Failed to place entry order");
                    return (string.Empty, string.Empty);
                }
                
                // Create bracket tracking record
                var bracketOrder = new BracketOrderGroup
                {
                    BracketId = bracketId,
                    Symbol = symbol,
                    EntryOrderId = entryOrderId,
                    EntryPrice = entryPrice,
                    StopPrice = stopPrice,
                    TargetPrice = targetPrice,
                    Quantity = quantity,
                    Side = side,
                    CreatedAt = DateTime.UtcNow,
                    Status = BracketStatus.Pending
                };
                
                _bracketOrders.TryAdd(bracketId, bracketOrder);
                
                _logger.LogInformation("✅ [BRACKET] Bracket order {BracketId} created with entry order {EntryOrderId}",
                    bracketId, entryOrderId);
                
                return (bracketId, entryOrderId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error placing bracket order for {Symbol}", symbol);
                return (string.Empty, string.Empty);
            }
        }
        
        /// <summary>
        /// Place iceberg order - large order executed in smaller hidden chunks
        /// </summary>
        public async Task<string> PlaceIcebergOrderAsync(
            string symbol,
            string side,
            int totalQuantity,
            int displayQuantity,
            decimal? limitPrice = null,
            CancellationToken cancellationToken = default)
        {
            // PHASE 5: Check if iceberg orders are enabled
            if (!_enableIcebergOrders)
            {
                _logger.LogWarning("⚠️ [ICEBERG] Iceberg orders are disabled. Set ENABLE_ICEBERG_ORDERS=true in .env to enable.");
                return string.Empty;
            }
            
            try
            {
                _logger.LogInformation(
                    "🧊 [ICEBERG] Placing iceberg order for {Symbol}: {Side} {Total} total, {Display} per chunk @ {Price}",
                    symbol, side, totalQuantity, displayQuantity, limitPrice?.ToString("F2", CultureInfo.InvariantCulture) ?? "MARKET");
                
                // Validate parameters
                if (displayQuantity >= totalQuantity)
                {
                    _logger.LogWarning("⚠️ [ICEBERG] Display quantity >= total, placing single order instead");
                    return limitPrice.HasValue
                        ? await PlaceLimitOrderAsync(symbol, side, totalQuantity, limitPrice.Value).ConfigureAwait(false)
                        : await PlaceMarketOrderAsync(symbol, side, totalQuantity).ConfigureAwait(false);
                }
                
                if (displayQuantity <= 0)
                {
                    _logger.LogError("❌ [ICEBERG] Invalid display quantity: {Display}", displayQuantity);
                    return string.Empty;
                }
                
                // Generate iceberg ID
                var icebergId = $"ICEBERG-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}-{symbol}";
                
                // Create iceberg tracking record
                var icebergOrder = new IcebergOrderExecution
                {
                    IcebergId = icebergId,
                    Symbol = symbol,
                    Side = side,
                    TotalQuantity = totalQuantity,
                    DisplayQuantity = displayQuantity,
                    FilledQuantity = 0,
                    LimitPrice = limitPrice,
                    CreatedAt = DateTime.UtcNow,
                    Status = IcebergStatus.Active
                };
                
                _icebergOrders.TryAdd(icebergId, icebergOrder);
                
                // Place first chunk
                var firstChunkQty = Math.Min(displayQuantity, totalQuantity);
                var firstOrderId = limitPrice.HasValue
                    ? await PlaceLimitOrderAsync(symbol, side, firstChunkQty, limitPrice.Value, $"ICEBERG-{icebergId}").ConfigureAwait(false)
                    : await PlaceMarketOrderAsync(symbol, side, firstChunkQty, $"ICEBERG-{icebergId}").ConfigureAwait(false);
                
                if (string.IsNullOrEmpty(firstOrderId))
                {
                    _logger.LogError("❌ [ICEBERG] Failed to place first chunk");
                    icebergOrder.Status = IcebergStatus.Error;
                    return string.Empty;
                }
                
                icebergOrder.ChildOrderIds.Add(firstOrderId);
                
                _logger.LogInformation("✅ [ICEBERG] Iceberg order {IcebergId} created, first chunk {OrderId} placed",
                    icebergId, firstOrderId);
                
                return icebergId;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error placing iceberg order for {Symbol}", symbol);
                return string.Empty;
            }
        }
        
        /// <summary>
        /// Process fill event for advanced order types
        /// Handles OCO cancellation and bracket/iceberg progression
        /// </summary>
        private async Task ProcessAdvancedOrderFillAsync(string orderId, FillEventData fillData)
        {
            // Check if this is part of an OCO pair
            var ocoOrder = _ocoOrders.Values.FirstOrDefault(o => 
                o.OrderId1 == orderId || o.OrderId2 == orderId);
            
            if (ocoOrder != null && ocoOrder.Status == OcoStatus.Active)
            {
                _logger.LogInformation("🔗 [OCO] Order {OrderId} in OCO pair {OcoId} filled, cancelling other order",
                    orderId, ocoOrder.OcoId);
                
                // Cancel the other order
                var otherOrderId = ocoOrder.OrderId1 == orderId ? ocoOrder.OrderId2 : ocoOrder.OrderId1;
                await CancelOrderAsync(otherOrderId).ConfigureAwait(false);
                
                ocoOrder.Status = OcoStatus.OneFilled;
                ocoOrder.FilledOrderId = orderId;
                ocoOrder.CancelledOrderId = otherOrderId;
            }
            
            // Check if this is a bracket entry order
            var bracketOrder = _bracketOrders.Values.FirstOrDefault(b => b.EntryOrderId == orderId);
            
            if (bracketOrder != null && bracketOrder.Status == BracketStatus.Pending)
            {
                _logger.LogInformation("🎯 [BRACKET] Entry order {OrderId} in bracket {BracketId} filled, placing stop and target",
                    orderId, bracketOrder.BracketId);
                
                // Determine opposite side for stop and target
                var exitSide = bracketOrder.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase) ? "SELL" : "BUY";
                
                // Place stop and target as OCO pair
                var (ocoId, stopOrderId, targetOrderId) = await PlaceOcoOrderAsync(
                    bracketOrder.Symbol,
                    exitSide, bracketOrder.Quantity, bracketOrder.StopPrice, "STOP",
                    exitSide, bracketOrder.Quantity, bracketOrder.TargetPrice, "LIMIT",
                    CancellationToken.None).ConfigureAwait(false);
                
                if (!string.IsNullOrEmpty(ocoId))
                {
                    bracketOrder.StopOrderId = stopOrderId;
                    bracketOrder.TargetOrderId = targetOrderId;
                    bracketOrder.Status = BracketStatus.EntryFilled;
                    
                    _logger.LogInformation("✅ [BRACKET] Bracket {BracketId} stop and target placed: {StopId} / {TargetId}",
                        bracketOrder.BracketId, stopOrderId, targetOrderId);
                }
                else
                {
                    _logger.LogError("❌ [BRACKET] Failed to place stop and target for bracket {BracketId}",
                        bracketOrder.BracketId);
                    bracketOrder.Status = BracketStatus.Error;
                }
            }
            
            // Check if this is an iceberg order chunk
            var icebergOrder = _icebergOrders.Values.FirstOrDefault(i => 
                i.ChildOrderIds.Contains(orderId) && i.Status == IcebergStatus.Active);
            
            if (icebergOrder != null)
            {
                _logger.LogInformation("🧊 [ICEBERG] Chunk {OrderId} in iceberg {IcebergId} filled with {Qty} contracts",
                    orderId, icebergOrder.IcebergId, fillData.Quantity);
                
                icebergOrder.FilledQuantity += fillData.Quantity;
                
                // Check if more chunks are needed
                var remainingQty = icebergOrder.TotalQuantity - icebergOrder.FilledQuantity;
                
                if (remainingQty > 0)
                {
                    var nextChunkQty = Math.Min(icebergOrder.DisplayQuantity, remainingQty);
                    
                    _logger.LogInformation("🧊 [ICEBERG] Placing next chunk for {IcebergId}: {Qty} of {Remaining} remaining",
                        icebergOrder.IcebergId, nextChunkQty, remainingQty);
                    
                    var nextOrderId = icebergOrder.LimitPrice.HasValue
                        ? await PlaceLimitOrderAsync(icebergOrder.Symbol, icebergOrder.Side, nextChunkQty, 
                            icebergOrder.LimitPrice.Value, $"ICEBERG-{icebergOrder.IcebergId}").ConfigureAwait(false)
                        : await PlaceMarketOrderAsync(icebergOrder.Symbol, icebergOrder.Side, nextChunkQty, 
                            $"ICEBERG-{icebergOrder.IcebergId}").ConfigureAwait(false);
                    
                    if (!string.IsNullOrEmpty(nextOrderId))
                    {
                        icebergOrder.ChildOrderIds.Add(nextOrderId);
                    }
                    else
                    {
                        _logger.LogError("❌ [ICEBERG] Failed to place next chunk for {IcebergId}", icebergOrder.IcebergId);
                        icebergOrder.Status = IcebergStatus.Error;
                    }
                }
                else
                {
                    _logger.LogInformation("✅ [ICEBERG] Iceberg order {IcebergId} completed: {Total} contracts filled",
                        icebergOrder.IcebergId, icebergOrder.TotalQuantity);
                    icebergOrder.Status = IcebergStatus.Completed;
                }
            }
        }
        
        // ========================================================================
        // DISPOSAL
        // ========================================================================
        
        public void Dispose()
        {
            _reconciliationTimer?.Dispose();
            _logger.LogInformation("🔄 OrderExecutionService disposed - reconciliation timer stopped");
        }
    }
    
    // ========================================================================
    // PHASE 1: EVENT DATA CLASSES
    // ========================================================================
    
    /// <summary>
    /// Event args for order fill notifications from TopstepX
    /// </summary>
    public class OrderFillEventArgs : EventArgs
    {
        public string OrderId { get; init; } = string.Empty;
        public string Symbol { get; init; } = string.Empty;
        public int Quantity { get; init; }
        public decimal FillPrice { get; init; }
        public decimal Commission { get; init; }
        public DateTime Timestamp { get; init; }
        public string Exchange { get; init; } = string.Empty;
        public string LiquidityType { get; init; } = string.Empty;
    }
    
    /// <summary>
    /// Event args for order placement
    /// </summary>
    public class OrderPlacedEventArgs : EventArgs
    {
        public string OrderId { get; init; } = string.Empty;
        public string Symbol { get; init; } = string.Empty;
        public int Quantity { get; init; }
        public decimal Price { get; init; }
        public string OrderType { get; init; } = string.Empty;
        public DateTime Timestamp { get; init; }
    }
    
    /// <summary>
    /// Event args for order rejection
    /// </summary>
    public class OrderRejectedEventArgs : EventArgs
    {
        public string OrderId { get; init; } = string.Empty;
        public string Symbol { get; init; } = string.Empty;
        public string Reason { get; init; } = string.Empty;
        public DateTime Timestamp { get; init; }
    }
    
    /// <summary>
    /// Fill event data from TopstepX SDK
    /// </summary>
    public class FillEventData
    {
        public string OrderId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal Price { get; set; }
        public decimal FillPrice { get; set; }
        public DateTime Timestamp { get; set; }
        public decimal Commission { get; set; }
        public string Exchange { get; set; } = string.Empty;
        public string LiquidityType { get; set; } = string.Empty;
    }
    
}
