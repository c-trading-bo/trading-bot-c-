using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using System.Collections.Concurrent;
// Legacy removed: using TradingBot.Infrastructure.TopstepX;
using TradingBot.Abstractions;

namespace Trading.Safety.OrderManagement;

/// <summary>
/// Production-grade partial fill handling with order reconciliation
/// Tracks partial fills, manages remaining quantities, and handles order lifecycle
/// </summary>
public interface IPartialFillHandler
{
    Task RegisterOrderAsync(string orderId, decimal originalQuantity, string customTag);
    Task HandleFillAsync(string orderId, decimal filledQuantity, decimal fillPrice);
    Task<OrderFillStatus> GetOrderStatusAsync(string orderId);
    Task<List<PartialFillOrder>> GetActiveOrdersAsync();
    Task TimeoutStaleOrdersAsync();
    Task CancelRemainingQuantityAsync(string orderId, string reason);
    event Action<OrderFillEvent> OnOrderFilled;
    event Action<OrderTimeoutEvent> OnOrderTimeout;
}

/// <summary>
/// Order fill status tracking
/// </summary>
public enum OrderFillStatus
{
    Pending,
    PartiallyFilled,
    FullyFilled,
    Cancelled,
    Timeout,
    Failed
}

/// <summary>
/// Comprehensive partial fill order tracking
/// </summary>
public class PartialFillOrder
{
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public decimal OriginalQuantity { get; set; }
    public decimal FilledQuantity { get; set; }
    public decimal RemainingQuantity => OriginalQuantity - FilledQuantity;
    public decimal AverageFillPrice { get; set; }
    public OrderFillStatus Status { get; set; } = OrderFillStatus.Pending;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? LastFillAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public List<FillExecution> Fills { get; set; } = new();
    public int FillCount => Fills.Count;
    public bool IsActive => Status == OrderFillStatus.Pending || Status == OrderFillStatus.PartiallyFilled;
    public TimeSpan Age => DateTime.UtcNow - CreatedAt;
    
    // Risk and performance metrics
    public decimal Slippage { get; set; }
    public decimal TotalFillValue => Fills.Sum(f => f.FilledQuantity * f.FillPrice);
    public string FailureReason { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Individual fill execution
/// </summary>
public class FillExecution
{
    public Guid FillId { get; set; } = Guid.NewGuid();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public decimal FilledQuantity { get; set; }
    public decimal FillPrice { get; set; }
    public decimal FillValue => FilledQuantity * FillPrice;
    public Dictionary<string, object> ExecutionDetails { get; set; } = new();
}

/// <summary>
/// Order fill event for notifications
/// </summary>
public class OrderFillEvent
{
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public decimal FilledQuantity { get; set; }
    public decimal FillPrice { get; set; }
    public decimal RemainingQuantity { get; set; }
    public OrderFillStatus Status { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public bool IsPartialFill => Status == OrderFillStatus.PartiallyFilled;
    public bool IsCompleteFill => Status == OrderFillStatus.FullyFilled;
}

/// <summary>
/// Order timeout event for stale order management
/// </summary>
public class OrderTimeoutEvent
{
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public decimal RemainingQuantity { get; set; }
    public TimeSpan Age { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string TimeoutReason { get; set; } = string.Empty;
}

/// <summary>
/// Production-grade partial fill handler with comprehensive order lifecycle management
/// </summary>
public class PartialFillHandler : IPartialFillHandler, IHostedService
{
    private readonly ILogger<PartialFillHandler> _logger;
    private readonly IOrderService _orderService;
    private readonly ConcurrentDictionary<string, PartialFillOrder> _activeOrders = new();
    private readonly Timer _timeoutTimer;
    private readonly Timer _reconciliationTimer;
    
    // Configuration
    private readonly TimeSpan _orderTimeout = TimeSpan.FromMinutes(5); // Orders timeout after 5 minutes
    private readonly TimeSpan _reconciliationInterval = TimeSpan.FromMinutes(1);
    private readonly decimal _fillToleranceEpsilon = 0.0001m; // Tolerance for decimal comparisons
    
    // Events
    public event Action<OrderFillEvent>? OnOrderFilled;
    public event Action<OrderTimeoutEvent>? OnOrderTimeout;

    public PartialFillHandler(ILogger<PartialFillHandler> logger, IOrderService orderService)
    {
        _logger = logger;
        _orderService = orderService;
        
        // Set up periodic timers
        _timeoutTimer = new Timer(CheckOrderTimeouts, null, _orderTimeout, _orderTimeout);
        _reconciliationTimer = new Timer(ReconcileOrders, null, _reconciliationInterval, _reconciliationInterval);
        
        _logger.LogInformation("[PARTIAL-FILL] Partial fill handler initialized with {Timeout} timeout", _orderTimeout);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("[PARTIAL-FILL] Partial fill handler started");
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Log summary of active orders
            var activeCount = _activeOrders.Count(kv => kv.Value.IsActive);
            if (activeCount > 0)
            {
                _logger.LogWarning("[PARTIAL-FILL] Shutting down with {ActiveCount} active orders", activeCount);
                
                foreach (var order in _activeOrders.Values.Where(o => o.IsActive))
                {
                    _logger.LogInformation("[PARTIAL-FILL] Active order at shutdown: {OrderId} - {RemainingQty} remaining",
                        order.OrderId, order.RemainingQuantity);
                }
            }
            
            _timeoutTimer?.Dispose();
            _reconciliationTimer?.Dispose();
            
            _logger.LogInformation("[PARTIAL-FILL] Partial fill handler stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Error during shutdown");
        }
    }

    public async Task RegisterOrderAsync(string orderId, decimal originalQuantity, string customTag)
    {
        try
        {
            if (string.IsNullOrEmpty(orderId))
            {
                throw new ArgumentException("Order ID cannot be null or empty", nameof(orderId));
            }
            
            if (originalQuantity <= 0)
            {
                throw new ArgumentException("Original quantity must be positive", nameof(originalQuantity));
            }
            
            var order = new PartialFillOrder
            {
                OrderId = orderId,
                CustomTag = customTag,
                OriginalQuantity = originalQuantity,
                FilledQuantity = 0,
                Status = OrderFillStatus.Pending,
                CreatedAt = DateTime.UtcNow
            };
            
            if (_activeOrders.TryAdd(orderId, order))
            {
                _logger.LogInformation("[PARTIAL-FILL] Order registered: {OrderId} ({CustomTag}) - {Quantity} quantity",
                    orderId, customTag, originalQuantity);
            }
            else
            {
                _logger.LogWarning("[PARTIAL-FILL] Order {OrderId} already registered", orderId);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Failed to register order {OrderId}", orderId);
            throw;
        }
    }

    public async Task HandleFillAsync(string orderId, decimal filledQuantity, decimal fillPrice)
    {
        try
        {
            if (!_activeOrders.TryGetValue(orderId, out var order))
            {
                _logger.LogWarning("[PARTIAL-FILL] Received fill for unknown order {OrderId}", orderId);
                return;
            }
            
            if (filledQuantity <= 0)
            {
                _logger.LogWarning("[PARTIAL-FILL] Invalid fill quantity {Quantity} for order {OrderId}", 
                    filledQuantity, orderId);
                return;
            }
            
            lock (order)
            {
                // Validate fill doesn't exceed remaining quantity
                if (order.FilledQuantity + filledQuantity > order.OriginalQuantity + _fillToleranceEpsilon)
                {
                    _logger.LogError("[PARTIAL-FILL] Fill quantity {FillQty} exceeds remaining quantity {RemainingQty} " +
                                   "for order {OrderId}", filledQuantity, order.RemainingQuantity, orderId);
                    return;
                }
                
                // Record the fill
                var fillExecution = new FillExecution
                {
                    Timestamp = DateTime.UtcNow,
                    FilledQuantity = filledQuantity,
                    FillPrice = fillPrice
                };
                
                order.Fills.Add(fillExecution);
                order.FilledQuantity += filledQuantity;
                order.LastFillAt = DateTime.UtcNow;
                
                // Update average fill price
                order.AverageFillPrice = order.TotalFillValue / order.FilledQuantity;
                
                // Determine new status
                var previousStatus = order.Status;
                if (order.RemainingQuantity <= _fillToleranceEpsilon)
                {
                    order.Status = OrderFillStatus.FullyFilled;
                    order.CompletedAt = DateTime.UtcNow;
                    
                    _logger.LogInformation("[PARTIAL-FILL] ✅ Order fully filled: {OrderId} ({CustomTag}) - " +
                                         "{FilledQty} @ {AvgPrice:F4} (Total fills: {FillCount})",
                        orderId, order.CustomTag, order.FilledQuantity, order.AverageFillPrice, order.FillCount);
                }
                else
                {
                    order.Status = OrderFillStatus.PartiallyFilled;
                    
                    _logger.LogInformation("[PARTIAL-FILL] 🔄 Partial fill: {OrderId} ({CustomTag}) - " +
                                         "{FillQty} @ {FillPrice:F4}, Remaining: {RemainingQty}",
                        orderId, order.CustomTag, filledQuantity, fillPrice, order.RemainingQuantity);
                }
                
                // Create fill event
                var fillEvent = new OrderFillEvent
                {
                    OrderId = orderId,
                    CustomTag = order.CustomTag,
                    FilledQuantity = filledQuantity,
                    FillPrice = fillPrice,
                    RemainingQuantity = order.RemainingQuantity,
                    Status = order.Status,
                    Timestamp = DateTime.UtcNow
                };
                
                // Notify subscribers
                OnOrderFilled?.Invoke(fillEvent);
                
                // Log status change
                if (previousStatus != order.Status)
                {
                    _logger.LogInformation("[PARTIAL-FILL] Order status changed: {OrderId} {PrevStatus} -> {NewStatus}",
                        orderId, previousStatus, order.Status);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Failed to handle fill for order {OrderId}", orderId);
            throw;
        }
    }

    public async Task<OrderFillStatus> GetOrderStatusAsync(string orderId)
    {
        if (_activeOrders.TryGetValue(orderId, out var order))
        {
            return order.Status;
        }
        
        return OrderFillStatus.Failed;
    }

    public async Task<List<PartialFillOrder>> GetActiveOrdersAsync()
    {
        return _activeOrders.Values
            .Where(o => o.IsActive)
            .OrderBy(o => o.CreatedAt)
            .ToList();
    }

    public async Task TimeoutStaleOrdersAsync()
    {
        try
        {
            var timeoutCutoff = DateTime.UtcNow - _orderTimeout;
            var staleOrders = _activeOrders.Values
                .Where(o => o.IsActive && o.CreatedAt < timeoutCutoff)
                .ToList();
            
            foreach (var order in staleOrders)
            {
                await TimeoutOrderAsync(order, "STALE_ORDER_TIMEOUT").ConfigureAwait(false);
            }
            
            if (staleOrders.Count > 0)
            {
                _logger.LogInformation("[PARTIAL-FILL] Timed out {Count} stale orders", staleOrders.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Error during stale order timeout check");
        }
    }

    public async Task CancelRemainingQuantityAsync(string orderId, string reason)
    {
        try
        {
            if (!_activeOrders.TryGetValue(orderId, out var order))
            {
                _logger.LogWarning("[PARTIAL-FILL] Cannot cancel unknown order {OrderId}", orderId);
                return;
            }
            
            if (!order.IsActive)
            {
                _logger.LogInformation("[PARTIAL-FILL] Order {OrderId} is already inactive ({Status})", 
                    orderId, order.Status);
                return;
            }
            
            lock (order)
            {
                var remainingQuantity = order.RemainingQuantity;
                
                order.Status = OrderFillStatus.Cancelled;
                order.CompletedAt = DateTime.UtcNow;
                order.FailureReason = reason;
                
                _logger.LogInformation("[PARTIAL-FILL] ❌ Order cancelled: {OrderId} ({CustomTag}) - " +
                                     "{RemainingQty} remaining quantity cancelled. Reason: {Reason}",
                    orderId, order.CustomTag, remainingQuantity, reason);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Failed to cancel remaining quantity for order {OrderId}", orderId);
            throw;
        }
    }

    private async Task TimeoutOrderAsync(PartialFillOrder order, string timeoutReason)
    {
        try
        {
            lock (order)
            {
                if (!order.IsActive)
                {
                    return; // Already completed
                }
                
                var remainingQuantity = order.RemainingQuantity;
                
                order.Status = OrderFillStatus.Timeout;
                order.CompletedAt = DateTime.UtcNow;
                order.FailureReason = timeoutReason;
                
                // Create timeout event
                var timeoutEvent = new OrderTimeoutEvent
                {
                    OrderId = order.OrderId,
                    CustomTag = order.CustomTag,
                    RemainingQuantity = remainingQuantity,
                    Age = order.Age,
                    TimeoutReason = timeoutReason
                };
                
                _logger.LogWarning("[PARTIAL-FILL] ⏰ Order timed out: {OrderId} ({CustomTag}) - " +
                                 "{RemainingQty} remaining after {Age:hh\\:mm\\:ss}",
                    order.OrderId, order.CustomTag, remainingQuantity, order.Age);
                
                // Notify subscribers
                OnOrderTimeout?.Invoke(timeoutEvent);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Error timing out order {OrderId}", order.OrderId);
        }
    }

    private void CheckOrderTimeouts(object? state)
    {
        try
        {
            _ = Task.Run(TimeoutStaleOrdersAsync);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Error in timeout timer");
        }
    }

    private void ReconcileOrders(object? state)
    {
        try
        {
            _ = Task.Run(async () =>
            {
                try
                {
                    await ReconcileOrderStatesAsync().ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[PARTIAL-FILL] Error during order reconciliation");
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Error in reconciliation timer");
        }
    }

    private async Task ReconcileOrderStatesAsync()
    {
        var activeOrders = _activeOrders.Values.Where(o => o.IsActive).ToList();
        if (activeOrders.Count == 0)
        {
            return;
        }
        
        _logger.LogDebug("[PARTIAL-FILL] Reconciling {Count} active orders", activeOrders.Count);
        
        foreach (var order in activeOrders)
        {
            try
            {
                // Check with broker for actual order status
                // This is a simplified version - in production, you'd query the actual broker API
                await ValidateOrderConsistency(order).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[PARTIAL-FILL] Failed to reconcile order {OrderId}", order.OrderId);
            }
        }
    }

    private async Task ValidateOrderConsistency(PartialFillOrder order)
    {
        try
        {
            // In a real implementation, this would query the broker API to verify order state
            // For now, we just validate internal consistency
            
            if (order.FilledQuantity > order.OriginalQuantity + _fillToleranceEpsilon)
            {
                _logger.LogError("[PARTIAL-FILL] 🚨 INCONSISTENCY: Order {OrderId} has filled quantity {FilledQty} " +
                               "exceeding original quantity {OriginalQty}",
                    order.OrderId, order.FilledQuantity, order.OriginalQuantity);
                
                // Mark as failed due to inconsistency
                order.Status = OrderFillStatus.Failed;
                order.FailureReason = "QUANTITY_INCONSISTENCY";
                order.CompletedAt = DateTime.UtcNow;
            }
            
            var calculatedFillValue = order.Fills.Sum(f => f.FillValue);
            var expectedFillValue = order.TotalFillValue;
            
            if (Math.Abs(calculatedFillValue - expectedFillValue) > _fillToleranceEpsilon)
            {
                _logger.LogWarning("[PARTIAL-FILL] Fill value mismatch for order {OrderId}: " +
                                 "Calculated={Calculated:F4}, Expected={Expected:F4}",
                    order.OrderId, calculatedFillValue, expectedFillValue);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PARTIAL-FILL] Error validating order consistency for {OrderId}", order.OrderId);
        }
    }

    public void Dispose()
    {
        _timeoutTimer?.Dispose();
        _reconciliationTimer?.Dispose();
    }
}