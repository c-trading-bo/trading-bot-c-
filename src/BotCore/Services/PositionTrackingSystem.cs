using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using TopstepX.Bot.Abstractions;

namespace TopstepX.Bot.Core.Services
{
    /// <summary>
    /// Real-time Position Tracking and Risk Management System
    /// Critical component for live trading safety
    /// </summary>
    public class PositionTrackingSystem
    {
        private readonly ILogger<PositionTrackingSystem> _logger;
        private readonly ConcurrentDictionary<string, Position> _positions = new();
        private readonly ConcurrentDictionary<string, PendingOrder> _pendingOrders = new();
        private readonly RiskLimits _riskLimits;
        private readonly Timer _reconciliationTimer;
        private readonly object _lockObject = new();
        
        public event EventHandler<PositionUpdateEventArgs>? PositionUpdated;
        public event EventHandler<RiskViolationEventArgs>? RiskViolationDetected;
        
        public class Position
        {
            public string Symbol { get; set; } = string.Empty;
            public int NetQuantity { get; set; }
            public decimal AveragePrice { get; set; }
            public decimal UnrealizedPnL { get; set; }
            public decimal RealizedPnL { get; set; }
            public DateTime LastUpdate { get; set; }
            public List<Fill> Fills { get; set; } = new();
            public decimal MarketValue { get; set; }
            public decimal DailyPnL { get; set; }
        }
        
        public class PendingOrder
        {
            public string OrderId { get; set; } = string.Empty;
            public string ClientOrderId { get; set; } = string.Empty;
            public string Symbol { get; set; } = string.Empty;
            public int Quantity { get; set; }
            public decimal Price { get; set; }
            public string Side { get; set; } = string.Empty;
            public string Status { get; set; } = string.Empty;
            public DateTime SubmittedTime { get; set; }
            public string OrderType { get; set; } = string.Empty;
        }
        
        public class Fill
        {
            public string FillId { get; set; } = string.Empty;
            public string OrderId { get; set; } = string.Empty;
            public DateTime Time { get; set; }
            public decimal Price { get; set; }
            public int Quantity { get; set; }
            public decimal Commission { get; set; }
        }
        
        public class RiskLimits
        {
            public decimal MaxDailyLoss { get; set; } = -1000m; // $1000 max daily loss
            public decimal MaxPositionSize { get; set; } = 5; // 5 contracts max
            public decimal MaxDrawdown { get; set; } = -2000m; // $2000 max drawdown
            public int MaxOrdersPerMinute { get; set; } = 10;
            public decimal AccountBalance { get; set; } = 50000m; // Account size
            public decimal MaxRiskPerTrade { get; set; } = 200m; // $200 max per trade
        }
        
        public PositionTrackingSystem(ILogger<PositionTrackingSystem> logger, RiskLimits? riskLimits = null)
        {
            _logger = logger;
            _riskLimits = riskLimits ?? new RiskLimits();
            
            // Setup reconciliation timer - runs every 30 seconds
            _reconciliationTimer = new Timer(ReconcilePositions, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
            
            _logger.LogInformation("📊 Position Tracking System initialized with risk limits");
        }
        
        /// <summary>
        /// Process incoming order fill
        /// </summary>
        public async Task ProcessFillAsync(string orderId, string symbol, decimal fillPrice, int fillQuantity, decimal commission = 0)
        {
            try
            {
                Fill fill;
                Position? position;
                
                lock (_lockObject)
                {
                    fill = new Fill
                    {
                        FillId = Guid.NewGuid().ToString(),
                        OrderId = orderId,
                        Time = DateTime.UtcNow,
                        Price = fillPrice,
                        Quantity = fillQuantity,
                        Commission = commission
                    };
                    
                    // Update position
                    if (!_positions.TryGetValue(symbol, out position))
                    {
                        position = new Position
                        {
                            Symbol = symbol,
                            NetQuantity = 0,
                            AveragePrice = 0,
                            UnrealizedPnL = 0,
                            RealizedPnL = 0,
                            LastUpdate = DateTime.UtcNow,
                            Fills = new List<Fill>()
                        };
                        _positions[symbol] = position;
                    }
                    
                    // Calculate new average price and quantity
                    UpdatePositionFromFill(position, fill);
                }
                
                _logger.LogInformation("✅ Fill processed: {Symbol} {Quantity}@{Price}, Net: {NetQty}", 
                    symbol, fillQuantity, fillPrice, position.NetQuantity);
                
                // Check risk limits (outside lock)
                await CheckRiskLimitsAsync(position).ConfigureAwait(false);
                
                // Fire position update event
                PositionUpdated?.Invoke(this, new PositionUpdateEventArgs { Position = position });
                
                // Remove from pending orders
                var pendingToRemove = _pendingOrders.Values.Where(o => o.OrderId == orderId).ToList();
                foreach (var pending in pendingToRemove)
                {
                    _pendingOrders.TryRemove(pending.ClientOrderId, out _);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing fill for {Symbol}", symbol);
            }
        }
        
        private void UpdatePositionFromFill(Position position, Fill fill)
        {
            var oldQuantity = position.NetQuantity;
            var oldAvgPrice = position.AveragePrice;
            
            // Add fill to history
            position.Fills.Add(fill);
            
            // Update net quantity
            position.NetQuantity += fill.Quantity;
            
            // Calculate new average price
            if (position.NetQuantity != 0)
            {
                var totalCost = (oldQuantity * oldAvgPrice) + (fill.Quantity * fill.Price);
                position.AveragePrice = totalCost / position.NetQuantity;
            }
            else
            {
                // Position closed - realize P&L
                var realizedPnL = (fill.Price - oldAvgPrice) * Math.Abs(fill.Quantity);
                position.RealizedPnL += realizedPnL;
                position.AveragePrice;
            }
            
            position.LastUpdate = DateTime.UtcNow;
        }
        
        /// <summary>
        /// Add pending order for tracking
        /// </summary>
        public void AddPendingOrder(PendingOrder order)
        {
            _pendingOrders[order.ClientOrderId] = order;
            _logger.LogDebug("📝 Tracking pending order: {ClientOrderId} - {Symbol} {Quantity}@{Price}", 
                order.ClientOrderId, order.Symbol, order.Quantity, order.Price);
        }
        
        /// <summary>
        /// Update market prices for unrealized P&L calculation
        /// </summary>
        public Task UpdateMarketPricesAsync(Dictionary<string, decimal> marketPrices)
        {
            foreach (var position in _positions.Values)
            {
                if (marketPrices.TryGetValue(position.Symbol, out var marketPrice))
                {
                    if (position.NetQuantity != 0)
                    {
                        position.UnrealizedPnL = (marketPrice - position.AveragePrice) * position.NetQuantity;
                        position.MarketValue = marketPrice * Math.Abs(position.NetQuantity);
                        
                        // Calculate daily P&L
                        position.DailyPnL = position.RealizedPnL + position.UnrealizedPnL;
                    }
                }
            }

            // Check overall account risk
            return CheckAccountRiskAsync();
        }
        
        private Task CheckRiskLimitsAsync(Position position)
        {
            var violations = new List<string>();
            
            // Check position size limit
            if (Math.Abs(position.NetQuantity) > _riskLimits.MaxPositionSize)
            {
                violations.Add($"Position size violation: {position.NetQuantity} > {_riskLimits.MaxPositionSize}");
            }
            
            // Check daily P&L limit
            if (position.DailyPnL < _riskLimits.MaxDailyLoss)
            {
                violations.Add($"Daily loss limit exceeded: {position.DailyPnL:C} < {_riskLimits.MaxDailyLoss:C}");
            }
            
            if (violations.Any())
            {
                var eventArgs = new RiskViolationEventArgs
                {
                    Symbol = position.Symbol,
                    ViolationType = "Position Risk",
                    Position = position,
                    Timestamp = DateTime.UtcNow
                };
                
                foreach (var violation in violations)
                {
                    eventArgs.Violations.Add(violation);
                }
                
                _logger.LogCritical("🚨 RISK VIOLATION: {Symbol} - {Violations}", 
                    position.Symbol, string.Join(", ", violations));
                
                RiskViolationDetected?.Invoke(this, eventArgs);
            }
            
            return Task.CompletedTask;
        }
        
        private Task CheckAccountRiskAsync()
        {
            var totalDailyPnL = _positions.Values.Sum(p => p.DailyPnL);
            var totalUnrealizedPnL = _positions.Values.Sum(p => p.UnrealizedPnL);
            var totalRealizedPnL = _positions.Values.Sum(p => p.RealizedPnL);
            
            var violations = new List<string>();
            
            if (totalDailyPnL < _riskLimits.MaxDailyLoss)
            {
                violations.Add($"Account daily loss limit: {totalDailyPnL:C} < {_riskLimits.MaxDailyLoss:C}");
            }
            
            if (totalUnrealizedPnL < _riskLimits.MaxDrawdown)
            {
                violations.Add($"Drawdown limit exceeded: {totalUnrealizedPnL:C} < {_riskLimits.MaxDrawdown:C}");
            }
            
            if (violations.Any())
            {
                var eventArgs = new RiskViolationEventArgs
                {
                    Symbol = "ACCOUNT",
                    ViolationType = "Account Risk",
                    Position = null,
                    Timestamp = DateTime.UtcNow
                };
                
                foreach (var violation in violations)
                {
                    eventArgs.Violations.Add(violation);
                }
                
                _logger.LogCritical("🚨 ACCOUNT RISK VIOLATION: {Violations}", string.Join(", ", violations));
                RiskViolationDetected?.Invoke(this, eventArgs);
            }
            
            return Task.CompletedTask;
        }
        
        private void ReconcilePositions(object? state)
        {
            try
            {
                _logger.LogDebug("🔄 Reconciling positions...");
                
                // Clean up old pending orders (older than 1 hour)
                var cutoffTime = DateTime.UtcNow.AddHours(-1);
                var staleOrders = _pendingOrders.Values.Where(o => o.SubmittedTime < cutoffTime).ToList();
                
                foreach (var staleOrder in staleOrders)
                {
                    _pendingOrders.TryRemove(staleOrder.ClientOrderId, out _);
                    _logger.LogWarning("⚠️ Removed stale pending order: {ClientOrderId}", staleOrder.ClientOrderId);
                }
                
                // Log current positions
                foreach (var position in _positions.Values.Where(p => p.NetQuantity != 0))
                {
                    _logger.LogInformation("📊 Position: {Symbol} {NetQty}@{AvgPrice} PnL:{DailyPnL:C}", 
                        position.Symbol, position.NetQuantity, position.AveragePrice, position.DailyPnL);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error during position reconciliation");
            }
        }
        
        /// <summary>
        /// Get current positions summary
        /// </summary>
        public Dictionary<string, Position> GetAllPositions()
        {
            return new Dictionary<string, Position>(_positions);
        }
        
        /// <summary>
        /// Get pending orders
        /// </summary>
        public Dictionary<string, PendingOrder> GetPendingOrders()
        {
            return new Dictionary<string, PendingOrder>(_pendingOrders);
        }
        
        /// <summary>
        /// Get account summary
        /// </summary>
        public AccountSummary GetAccountSummary()
        {
            var totalDailyPnL = _positions.Values.Sum(p => p.DailyPnL);
            var totalUnrealizedPnL = _positions.Values.Sum(p => p.UnrealizedPnL);
            var totalRealizedPnL = _positions.Values.Sum(p => p.RealizedPnL);
            var totalMarketValue = _positions.Values.Sum(p => p.MarketValue);
            
            return new AccountSummary
            {
                AccountBalance = _riskLimits.AccountBalance,
                TotalDailyPnL = totalDailyPnL,
                TotalUnrealizedPnL = totalUnrealizedPnL,
                TotalRealizedPnL = totalRealizedPnL,
                TotalMarketValue = totalMarketValue,
                OpenPositions = _positions.Values.Count(p => p.NetQuantity != 0),
                PendingOrders = _pendingOrders.Count,
                LastUpdate = DateTime.UtcNow
            };
        }
        
        public void Dispose()
        {
            _reconciliationTimer?.Dispose();
        }
    }
    
    public class PositionUpdateEventArgs : EventArgs
    {
        public PositionTrackingSystem.Position Position { get; set; } = new();
    }
}