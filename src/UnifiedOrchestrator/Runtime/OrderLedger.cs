using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Order ledger system for idempotent order tracking and evidence chain validation.
/// Generates unique client IDs and tracks order-to-fill relationships for production safety.
/// </summary>
public class OrderLedger
{
    private readonly ILogger<OrderLedger> _logger;
    private readonly ConcurrentDictionary<string, OrderRecord> _orderRecords;
    private readonly ConcurrentDictionary<string, string> _clientIdToGatewayId;
    private readonly object _idGenerationLock = new();
    private long _nextClientIdSequence = 1;

    public OrderLedger(ILogger<OrderLedger> logger)
    {
        _logger = logger;
        _orderRecords = new ConcurrentDictionary<string, OrderRecord>();
        _clientIdToGatewayId = new ConcurrentDictionary<string, string>();
        
        _logger.LogInformation("📋 [ORDER-LEDGER] Initialized order tracking system");
    }

    /// <summary>
    /// Generate a new unique client ID for order placement
    /// </summary>
    /// <param name="symbol">Trading symbol</param>
    /// <param name="strategy">Strategy name for traceability</param>
    /// <returns>Unique client ID</returns>
    public string NewClientId(string symbol, string strategy = "UNIFIED")
    {
        lock (_idGenerationLock)
        {
            var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var sequence = _nextClientIdSequence++;
            var clientId = $"CLT_{symbol}_{strategy}_{timestamp}_{sequence:D6}";
            
            _logger.LogDebug("🆔 [ORDER-LEDGER] Generated client ID: {ClientId}", clientId);
            return clientId;
        }
    }

    /// <summary>
    /// Check if a client ID has already been used (duplicate detection)
    /// </summary>
    /// <param name="clientId">Client ID to check</param>
    /// <returns>True if this is a duplicate order attempt</returns>
    public bool IsDuplicate(string clientId)
    {
        var isDuplicate = _orderRecords.ContainsKey(clientId);
        
        if (isDuplicate)
        {
            _logger.LogWarning("⚠️ [ORDER-LEDGER] Duplicate order attempt detected: {ClientId}", clientId);
        }
        
        return isDuplicate;
    }

    /// <summary>
    /// Record a new order in the ledger after successful placement
    /// </summary>
    /// <param name="clientId">Client-generated order ID</param>
    /// <param name="gatewayOrderId">Gateway-returned order ID</param>
    /// <param name="symbol">Trading symbol</param>
    /// <param name="quantity">Order quantity</param>
    /// <param name="price">Order price (if limit order)</param>
    /// <param name="orderType">Order type description</param>
    /// <returns>True if record was successfully created</returns>
    public bool TryRecord(string clientId, string gatewayOrderId, string symbol, decimal quantity, decimal? price = null, string orderType = "MARKET")
    {
        var record = new OrderRecord
        {
            ClientId = clientId,
            GatewayOrderId = gatewayOrderId,
            Symbol = symbol,
            Quantity = quantity,
            Price = price,
            OrderType = orderType,
            PlacementTime = DateTime.UtcNow,
            Status = OrderStatus.Placed
        };

        var added = _orderRecords.TryAdd(clientId, record);
        if (added)
        {
            _clientIdToGatewayId.TryAdd(clientId, gatewayOrderId);
            _logger.LogInformation("📝 [ORDER-LEDGER] Recorded order: {ClientId} → {GatewayId} ({Symbol} {Quantity})",
                clientId, gatewayOrderId, symbol, quantity);
        }
        else
        {
            _logger.LogError("❌ [ORDER-LEDGER] Failed to record order - ClientId already exists: {ClientId}", clientId);
        }

        return added;
    }

    /// <summary>
    /// Record a fill event and link it to the original order
    /// </summary>
    /// <param name="gatewayOrderId">Gateway order ID from fill event</param>
    /// <param name="fillId">Unique fill ID</param>
    /// <param name="fillQuantity">Quantity filled</param>
    /// <param name="fillPrice">Fill price</param>
    /// <param name="fillTime">Fill timestamp</param>
    /// <returns>True if fill was successfully linked to an order</returns>
    public bool RecordFill(string gatewayOrderId, string fillId, decimal fillQuantity, decimal fillPrice, DateTime fillTime)
    {
        // Find the order record by gateway ID
        var orderRecord = _orderRecords.Values.FirstOrDefault(r => r.GatewayOrderId == gatewayOrderId);
        if (orderRecord == null)
        {
            _logger.LogWarning("⚠️ [ORDER-LEDGER] Fill received for unknown order: {GatewayOrderId} Fill:{FillId}", 
                gatewayOrderId, fillId);
            return false;
        }

        // Create fill record
        var fillRecord = new FillRecord
        {
            FillId = fillId,
            GatewayOrderId = gatewayOrderId,
            ClientId = orderRecord.ClientId,
            Quantity = fillQuantity,
            Price = fillPrice,
            FillTime = fillTime
        };

        // Add to order's fill list
        orderRecord.Fills.Add(fillRecord);
        orderRecord.TotalFilled += fillQuantity;
        
        // Update order status
        if (Math.Abs(orderRecord.TotalFilled) >= Math.Abs(orderRecord.Quantity))
        {
            orderRecord.Status = OrderStatus.Filled;
        }
        else
        {
            orderRecord.Status = OrderStatus.PartiallyFilled;
        }

        _logger.LogInformation("💰 [ORDER-LEDGER] Fill recorded: {ClientId} → {GatewayOrderId} Fill:{FillId} Qty:{Quantity} Price:{Price}",
            orderRecord.ClientId, gatewayOrderId, fillId, fillQuantity, fillPrice);

        return true;
    }

    /// <summary>
    /// Verify the complete evidence chain: ClientId → GatewayId → Fill
    /// </summary>
    /// <param name="clientId">Original client ID</param>
    /// <param name="expectedFillQuantity">Expected fill quantity for validation</param>
    /// <returns>Evidence validation result</returns>
    public EvidenceValidationResult ValidateEvidence(string clientId, decimal expectedFillQuantity)
    {
        if (!_orderRecords.TryGetValue(clientId, out var record))
        {
            return new EvidenceValidationResult
            {
                IsValid = false,
                Reason = $"Order record not found for ClientId: {clientId}"
            };
        }

        if (string.IsNullOrEmpty(record.GatewayOrderId))
        {
            return new EvidenceValidationResult
            {
                IsValid = false,
                Reason = $"No gateway order ID linked to ClientId: {clientId}"
            };
        }

        if (record.Fills.Count == 0)
        {
            return new EvidenceValidationResult
            {
                IsValid = false,
                Reason = $"No fills recorded for order: {clientId} → {record.GatewayOrderId}"
            };
        }

        if (Math.Abs(record.TotalFilled - expectedFillQuantity) > 0.001m)
        {
            return new EvidenceValidationResult
            {
                IsValid = false,
                Reason = $"Fill quantity mismatch: expected {expectedFillQuantity}, got {record.TotalFilled}"
            };
        }

        return new EvidenceValidationResult
        {
            IsValid = true,
            ClientId = clientId,
            GatewayOrderId = record.GatewayOrderId,
            FillCount = record.Fills.Count,
            TotalFilled = record.TotalFilled,
            Reason = "Complete evidence chain validated"
        };
    }

    /// <summary>
    /// Get order record for monitoring/debugging
    /// </summary>
    public OrderRecord? GetOrderRecord(string clientId)
    {
        return _orderRecords.TryGetValue(clientId, out var record) ? record : null;
    }

    /// <summary>
    /// Get summary statistics for monitoring
    /// </summary>
    public OrderLedgerStats GetStats()
    {
        var records = _orderRecords.Values.ToList();
        return new OrderLedgerStats
        {
            TotalOrders = records.Count,
            FilledOrders = records.Count(r => r.Status == OrderStatus.Filled),
            PartiallyFilledOrders = records.Count(r => r.Status == OrderStatus.PartiallyFilled),
            PlacedOrders = records.Count(r => r.Status == OrderStatus.Placed),
            TotalFills = records.Sum(r => r.Fills.Count),
            OldestOrder = records.Any() ? records.Min(r => r.PlacementTime) : (DateTime?)null,
            NewestOrder = records.Any() ? records.Max(r => r.PlacementTime) : (DateTime?)null
        };
    }
}

/// <summary>
/// Order record tracking client ID to gateway ID relationship
/// </summary>
public class OrderRecord
{
    public string ClientId { get; set; } = string.Empty;
    public string GatewayOrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal? Price { get; set; }
    public string OrderType { get; set; } = string.Empty;
    public DateTime PlacementTime { get; set; }
    public OrderStatus Status { get; set; }
    public decimal TotalFilled { get; set; }
    public List<FillRecord> Fills { get; set; } = new();
}

/// <summary>
/// Fill record linked to an order
/// </summary>
public class FillRecord
{
    public string FillId { get; set; } = string.Empty;
    public string GatewayOrderId { get; set; } = string.Empty;
    public string ClientId { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime FillTime { get; set; }
}

/// <summary>
/// Order status enumeration
/// </summary>
public enum OrderStatus
{
    Placed,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected
}

/// <summary>
/// Result of evidence chain validation
/// </summary>
public class EvidenceValidationResult
{
    public bool IsValid { get; set; }
    public string Reason { get; set; } = string.Empty;
    public string ClientId { get; set; } = string.Empty;
    public string GatewayOrderId { get; set; } = string.Empty;
    public int FillCount { get; set; }
    public decimal TotalFilled { get; set; }
}

/// <summary>
/// Order ledger statistics for monitoring
/// </summary>
public class OrderLedgerStats
{
    public int TotalOrders { get; set; }
    public int FilledOrders { get; set; }
    public int PartiallyFilledOrders { get; set; }
    public int PlacedOrders { get; set; }
    public int TotalFills { get; set; }
    public DateTime? OldestOrder { get; set; }
    public DateTime? NewestOrder { get; set; }
}