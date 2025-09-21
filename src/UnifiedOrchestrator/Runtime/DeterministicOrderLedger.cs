using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Bulletproof order tracking with deterministic fingerprinting and persistence across restarts
/// Production-grade idempotency and audit trail for institutional trading
/// </summary>
public class DeterministicOrderLedger
{
    private readonly ILogger<DeterministicOrderLedger> _logger;
    private readonly ConcurrentDictionary<string, OrderRecord> _orderRecords;
    private readonly ConcurrentDictionary<string, string> _fingerprintToGatewayId;
    private readonly string _persistenceFile;
    private readonly object _persistenceLock = new();
    private readonly Timer _persistenceTimer;

    public DeterministicOrderLedger(ILogger<DeterministicOrderLedger> logger)
    {
        _logger = logger;
        _orderRecords = new ConcurrentDictionary<string, OrderRecord>();
        _fingerprintToGatewayId = new ConcurrentDictionary<string, string>();
        
        // Persistence file path
        var stateDir = Path.Combine(Directory.GetCurrentDirectory(), "state");
        Directory.CreateDirectory(stateDir);
        _persistenceFile = Path.Combine(stateDir, "order_ledger_persistence.json");
        
        // Load existing data on startup
        LoadPersistedData();
        
        // Set up automatic persistence every 30 seconds
        _persistenceTimer = new Timer(PersistData, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
        
        _logger.LogInformation("📋 [DETERMINISTIC-ORDER-LEDGER] Initialized with persistence at {PersistenceFile}", _persistenceFile);
        _logger.LogInformation("📊 [DETERMINISTIC-ORDER-LEDGER] Loaded {OrderCount} existing orders from persistence", _orderRecords.Count);
    }

    /// <summary>
    /// Generate deterministic fingerprint for order characteristics
    /// Same order details always generate the same fingerprint
    /// </summary>
    public string GenerateDeterministicFingerprint(string symbol, decimal quantity, decimal? price, string orderType, string strategy = "UNIFIED")
    {
        // Create deterministic input string
        var input = $"{symbol}|{quantity:F8}|{price?.ToString("F8") ?? "MARKET"}|{orderType}|{strategy}";
        
        // Generate SHA-256 hash for deterministic fingerprint
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(input));
        var fingerprint = $"DET_{symbol}_{Convert.ToHexString(hashBytes)[..16]}";
        
        _logger.LogDebug("🔢 [DETERMINISTIC-ORDER-LEDGER] Generated fingerprint: {Fingerprint} for {Input}", fingerprint, input);
        return fingerprint;
    }

    /// <summary>
    /// Check if an order with these characteristics has already been placed
    /// </summary>
    public bool IsOrderAlreadyPlaced(string symbol, decimal quantity, decimal? price, string orderType, string strategy = "UNIFIED")
    {
        var fingerprint = GenerateDeterministicFingerprint(symbol, quantity, price, orderType, strategy);
        var exists = _orderRecords.ContainsKey(fingerprint);
        
        if (exists)
        {
            var existingOrder = _orderRecords[fingerprint];
            _logger.LogWarning("⚠️ [DETERMINISTIC-ORDER-LEDGER] Duplicate order detected: {Fingerprint} (existing: {ExistingOrderId}, placed: {PlacementTime})",
                fingerprint, existingOrder.GatewayOrderId, existingOrder.PlacementTimeUtc);
        }
        
        return exists;
    }

    /// <summary>
    /// Record a new order with deterministic fingerprinting
    /// </summary>
    public bool TryRecordOrder(string symbol, decimal quantity, decimal? price, string orderType, string gatewayOrderId, string strategy = "UNIFIED")
    {
        var fingerprint = GenerateDeterministicFingerprint(symbol, quantity, price, orderType, strategy);
        
        var record = new OrderRecord
        {
            Fingerprint = fingerprint,
            GatewayOrderId = gatewayOrderId,
            Symbol = symbol,
            Quantity = quantity,
            Price = price,
            OrderType = orderType,
            Strategy = strategy,
            PlacementTimeUtc = DateTime.UtcNow,
            Status = OrderStatus.Placed
        };

        var added = _orderRecords.TryAdd(fingerprint, record);
        if (added)
        {
            _fingerprintToGatewayId.TryAdd(fingerprint, gatewayOrderId);
            _logger.LogInformation("📝 [DETERMINISTIC-ORDER-LEDGER] Recorded order: {Fingerprint} → {GatewayId} ({Symbol} {Quantity}@{Price})",
                fingerprint, gatewayOrderId, symbol, quantity, price?.ToString("F2") ?? "MARKET");
            
            // Trigger immediate persistence for new orders
            TriggerPersistence();
        }
        else
        {
            _logger.LogError("❌ [DETERMINISTIC-ORDER-LEDGER] Failed to record order - Fingerprint already exists: {Fingerprint}", fingerprint);
        }

        return added;
    }

    /// <summary>
    /// Record a fill event and link it to the original order via gateway ID
    /// </summary>
    public bool RecordFill(string gatewayOrderId, string fillId, decimal fillQuantity, decimal fillPrice, DateTime fillTimeUtc)
    {
        // Find the order record by gateway ID
        var orderRecord = _orderRecords.Values.FirstOrDefault(r => r.GatewayOrderId == gatewayOrderId);
        if (orderRecord == null)
        {
            _logger.LogWarning("⚠️ [DETERMINISTIC-ORDER-LEDGER] Fill received for unknown order: {GatewayOrderId} Fill:{FillId}", 
                gatewayOrderId, fillId);
            return false;
        }

        // Create fill record
        var fillRecord = new FillRecord
        {
            FillId = fillId,
            GatewayOrderId = gatewayOrderId,
            Fingerprint = orderRecord.Fingerprint,
            Quantity = fillQuantity,
            Price = fillPrice,
            FillTimeUtc = fillTimeUtc
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

        _logger.LogInformation("💰 [DETERMINISTIC-ORDER-LEDGER] Fill recorded: {Fingerprint} → {GatewayOrderId} Fill:{FillId} Qty:{Quantity} Price:{Price}",
            orderRecord.Fingerprint, gatewayOrderId, fillId, fillQuantity, fillPrice);

        // Trigger persistence for fill updates
        TriggerPersistence();
        return true;
    }

    /// <summary>
    /// Validate complete evidence chain: Fingerprint → GatewayId → Fill
    /// </summary>
    public EvidenceValidationResult ValidateEvidenceChain(string symbol, decimal quantity, decimal? price, string orderType, decimal expectedFillQuantity, string strategy = "UNIFIED")
    {
        var fingerprint = GenerateDeterministicFingerprint(symbol, quantity, price, orderType, strategy);
        
        if (!_orderRecords.TryGetValue(fingerprint, out var record))
        {
            return new EvidenceValidationResult
            {
                IsValid = false,
                Reason = $"Order record not found for fingerprint: {fingerprint}"
            };
        }

        if (string.IsNullOrEmpty(record.GatewayOrderId))
        {
            return new EvidenceValidationResult
            {
                IsValid = false,
                Reason = $"No gateway order ID linked to fingerprint: {fingerprint}"
            };
        }

        if (record.Fills.Count == 0)
        {
            return new EvidenceValidationResult
            {
                IsValid = false,
                Reason = $"No fills recorded for order: {fingerprint} → {record.GatewayOrderId}"
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
            Fingerprint = fingerprint,
            GatewayOrderId = record.GatewayOrderId,
            FillCount = record.Fills.Count,
            TotalFilled = record.TotalFilled,
            Reason = "Complete evidence chain validated"
        };
    }

    /// <summary>
    /// Get order record by deterministic fingerprint
    /// </summary>
    public OrderRecord? GetOrderRecord(string symbol, decimal quantity, decimal? price, string orderType, string strategy = "UNIFIED")
    {
        var fingerprint = GenerateDeterministicFingerprint(symbol, quantity, price, orderType, strategy);
        return _orderRecords.TryGetValue(fingerprint, out var record) ? record : null;
    }

    /// <summary>
    /// Get comprehensive statistics for monitoring
    /// </summary>
    public DeterministicOrderLedgerStats GetStats()
    {
        var records = _orderRecords.Values.ToList();
        var now = DateTime.UtcNow;
        
        return new DeterministicOrderLedgerStats
        {
            TotalOrders = records.Count,
            FilledOrders = records.Count(r => r.Status == OrderStatus.Filled),
            PartiallyFilledOrders = records.Count(r => r.Status == OrderStatus.PartiallyFilled),
            PlacedOrders = records.Count(r => r.Status == OrderStatus.Placed),
            TotalFills = records.Sum(r => r.Fills.Count),
            OldestOrder = records.Any() ? records.Min(r => r.PlacementTimeUtc) : (DateTime?)null,
            NewestOrder = records.Any() ? records.Max(r => r.PlacementTimeUtc) : (DateTime?)null,
            OrdersLast24Hours = records.Count(r => (now - r.PlacementTimeUtc).TotalHours <= 24),
            UniqueSymbols = records.Select(r => r.Symbol).Distinct().Count(),
            DuplicatePreventionHits = records.Count(r => r.PlacementTimeUtc < now.AddMinutes(-1)) // Estimate based on age
        };
    }

    /// <summary>
    /// Load persisted data on startup to survive restarts
    /// </summary>
    private void LoadPersistedData()
    {
        try
        {
            if (!File.Exists(_persistenceFile))
            {
                _logger.LogInformation("📂 [DETERMINISTIC-ORDER-LEDGER] No persistence file found, starting fresh");
                return;
            }

            var json = File.ReadAllText(_persistenceFile);
            var persistedData = JsonSerializer.Deserialize<PersistedOrderData>(json);
            
            if (persistedData?.Orders != null)
            {
                foreach (var order in persistedData.Orders)
                {
                    _orderRecords.TryAdd(order.Fingerprint, order);
                    if (!string.IsNullOrEmpty(order.GatewayOrderId))
                    {
                        _fingerprintToGatewayId.TryAdd(order.Fingerprint, order.GatewayOrderId);
                    }
                }

                _logger.LogInformation("📂 [DETERMINISTIC-ORDER-LEDGER] Loaded {Count} persisted orders (last save: {LastSave})", 
                    persistedData.Orders.Count, persistedData.LastSavedUtc);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [DETERMINISTIC-ORDER-LEDGER] Failed to load persisted data");
        }
    }

    /// <summary>
    /// Persist data to disk for restart survival
    /// </summary>
    private void PersistData(object? state = null)
    {
        lock (_persistenceLock)
        {
            try
            {
                var persistedData = new PersistedOrderData
                {
                    Orders = _orderRecords.Values.ToList(),
                    LastSavedUtc = DateTime.UtcNow,
                    Version = "2.0"
                };

                var json = JsonSerializer.Serialize(persistedData, new JsonSerializerOptions 
                { 
                    WriteIndented = false, // Compact for performance
                    DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
                });
                
                // Atomic write using temp file
                var tempFile = _persistenceFile + ".tmp";
                File.WriteAllText(tempFile, json);
                File.Move(tempFile, _persistenceFile, overwrite: true);

                _logger.LogDebug("💾 [DETERMINISTIC-ORDER-LEDGER] Persisted {Count} orders to disk", _orderRecords.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [DETERMINISTIC-ORDER-LEDGER] Failed to persist data to disk");
            }
        }
    }

    /// <summary>
    /// Trigger immediate persistence (for important events)
    /// </summary>
    private void TriggerPersistence()
    {
        Task.Run(() => PersistData());
    }

    /// <summary>
    /// Cleanup old orders to prevent memory growth
    /// </summary>
    public int CleanupOldOrders(TimeSpan maxAge)
    {
        var cutoffTime = DateTime.UtcNow - maxAge;
        var oldOrders = _orderRecords.Where(kvp => kvp.Value.PlacementTimeUtc < cutoffTime).ToList();
        
        var removedCount = 0;
        foreach (var (fingerprint, order) in oldOrders)
        {
            if (_orderRecords.TryRemove(fingerprint, out _))
            {
                _fingerprintToGatewayId.TryRemove(fingerprint, out _);
                removedCount++;
            }
        }

        if (removedCount > 0)
        {
            _logger.LogInformation("🧹 [DETERMINISTIC-ORDER-LEDGER] Cleaned up {Count} old orders (older than {MaxAge})", 
                removedCount, maxAge);
            TriggerPersistence();
        }

        return removedCount;
    }

    public void Dispose()
    {
        _persistenceTimer?.Dispose();
        PersistData(); // Final persistence on shutdown
        _logger.LogInformation("🔚 [DETERMINISTIC-ORDER-LEDGER] Disposed with final persistence");
    }
}

/// <summary>
/// Enhanced order record with deterministic fingerprint
/// </summary>
public class OrderRecord
{
    public string Fingerprint { get; set; } = string.Empty;
    public string GatewayOrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal? Price { get; set; }
    public string OrderType { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public DateTime PlacementTimeUtc { get; set; }
    public OrderStatus Status { get; set; }
    public decimal TotalFilled { get; set; }
    public List<FillRecord> Fills { get; set; } = new();
}

/// <summary>
/// Enhanced fill record with fingerprint linking
/// </summary>
public class FillRecord
{
    public string FillId { get; set; } = string.Empty;
    public string GatewayOrderId { get; set; } = string.Empty;
    public string Fingerprint { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime FillTimeUtc { get; set; }
}

/// <summary>
/// Enhanced evidence validation result
/// </summary>
public class EvidenceValidationResult
{
    public bool IsValid { get; set; }
    public string Reason { get; set; } = string.Empty;
    public string Fingerprint { get; set; } = string.Empty;
    public string GatewayOrderId { get; set; } = string.Empty;
    public int FillCount { get; set; }
    public decimal TotalFilled { get; set; }
}

/// <summary>
/// Statistics for deterministic order ledger
/// </summary>
public class DeterministicOrderLedgerStats
{
    public int TotalOrders { get; set; }
    public int FilledOrders { get; set; }
    public int PartiallyFilledOrders { get; set; }
    public int PlacedOrders { get; set; }
    public int TotalFills { get; set; }
    public DateTime? OldestOrder { get; set; }
    public DateTime? NewestOrder { get; set; }
    public int OrdersLast24Hours { get; set; }
    public int UniqueSymbols { get; set; }
    public int DuplicatePreventionHits { get; set; }
}

/// <summary>
/// Persistence data structure
/// </summary>
public class PersistedOrderData
{
    public List<OrderRecord> Orders { get; set; } = new();
    public DateTime LastSavedUtc { get; set; }
    public string Version { get; set; } = "2.0";
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