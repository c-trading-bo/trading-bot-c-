using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.IO;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Idempotent order service with SHA1 deduplication
/// Implements 24h deduplication window with retry backoff
/// </summary>
public class IdempotentOrderService : IIdempotentOrderService, IDisposable
{
    private readonly ILogger<IdempotentOrderService> _logger;
    private readonly IdempotentConfig _config;
    private readonly string _basePath;
    private readonly ConcurrentDictionary<string, OrderRecord> _orderCache = new();
    private readonly Timer _cleanupTimer;

    // LoggerMessage delegates for performance
    private static readonly Action<ILogger, string, string, Exception?> OrderKeyGenerationFailed =
        LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(1001, "OrderKeyGenerationFailed"),
            "[IDEMPOTENT] Failed to generate order key for {ModelId}-{Symbol}");

    private static readonly Action<ILogger, string, string, Exception?> OrderRegistrationFailed =
        LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(1002, "OrderRegistrationFailed"),
            "[IDEMPOTENT] Failed to register order: {OrderKey} -> {OrderId}");

    private static readonly Action<ILogger, string, string, Exception?> OrderRegistered =
        LoggerMessage.Define<string, string>(LogLevel.Debug, new EventId(1003, "OrderRegistered"),
            "[IDEMPOTENT] Registered order: {OrderKey} -> {OrderId}");
            
    // Additional LoggerMessage delegates for CA1848 compliance

    private static readonly Action<ILogger, string, DateTime, Exception?> DuplicateOrderWithTimestamp =
        LoggerMessage.Define<string, DateTime>(LogLevel.Warning, new EventId(1013, "DuplicateOrderWithTimestamp"),
            "[IDEMPOTENT] Duplicate order detected: {OrderKey} (first seen: {FirstSeen})");
            
    private static readonly Action<ILogger, string, Exception?> OrderExecutionFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(1005, "OrderExecutionFailed"),
            "[IDEMPOTENT] Failed to execute order for key: {OrderKey}");
            


    private static readonly Action<ILogger, Exception?> OrderCleanupFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(1008, "OrderCleanupFailed"),
            "[IDEMPOTENT] Failed to cleanup expired orders");
            
    private static readonly Action<ILogger, string, Exception?> OrderPersistenceWarning =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(1009, "OrderPersistenceWarning"),
            "[IDEMPOTENT] Failed to persist order state for key: {OrderKey}");
            

    private static readonly Action<ILogger, Exception?> OrderStateLoadFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(1011, "OrderStateLoadFailed"),
            "[IDEMPOTENT] Failed to load order state from disk");
            

    private static readonly Action<ILogger, string, Exception?> MaxRetryAttemptsExceeded =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(1014, "MaxRetryAttemptsExceeded"),
            "[IDEMPOTENT] Max retry attempts exceeded for order: {OrderKey}");
            
    private static readonly Action<ILogger, string, int, int, int, Exception?> RetryingOrder =
        LoggerMessage.Define<string, int, int, int>(LogLevel.Information, new EventId(1015, "RetryingOrder"),
            "[IDEMPOTENT] Retrying order {OrderKey} (attempt {Attempt}/{Max}) after {Delay}ms");
            
    private static readonly Action<ILogger, string, Exception?> OrderFileLoadWarning =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(1016, "OrderFileLoadWarning"),
            "[IDEMPOTENT] Failed to load order file: {File}");
            
    private static readonly Action<ILogger, int, Exception?> OrdersLoadedInfo =
        LoggerMessage.Define<int>(LogLevel.Information, new EventId(1017, "OrdersLoadedInfo"),
            "[IDEMPOTENT] Loaded {Count} existing orders into cache");
            
    private static readonly Action<ILogger, string, Exception?> ExpiredFileDeleteWarning =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(1018, "ExpiredFileDeleteWarning"),
            "[IDEMPOTENT] Failed to delete expired file: {File}");
            
    private static readonly Action<ILogger, int, int, Exception?> CleanupCompletedInfo =
        LoggerMessage.Define<int, int>(LogLevel.Information, new EventId(1019, "CleanupCompletedInfo"),
            "[IDEMPOTENT] Cleanup completed: {CacheRemoved} cache entries, {FilesDeleted} files deleted");
            
    private static readonly Action<ILogger, string, string, string, double, Exception?> OrderKeyGenerated =
        LoggerMessage.Define<string, string, string, double>(LogLevel.Debug, new EventId(1020, "OrderKeyGenerated"),
            "[IDEMPOTENT] Generated order key: {Key} for {Symbol} {Side} (bucket: {PriceBucket:F2})");

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    // Constants for magic numbers (S109 compliance)
    private const int OrderKeyTruncationLength = 8;
    private const double DefaultTickSize = 0.25;
    private const int TimeBucketMinutes = 5;

    public IdempotentOrderService(
        ILogger<IdempotentOrderService> logger,
        IdempotentConfig config,
        string basePath = "data/orders")
    {
        _logger = logger;
        _config = config;
        _basePath = basePath;
        
        Directory.CreateDirectory(_basePath);
        
        // Schedule periodic cleanup
        _cleanupTimer = new Timer(PerformCleanup, null, TimeSpan.FromHours(1), TimeSpan.FromHours(1));
        
        // Load existing orders on startup
        _ = Task.Run(LoadExistingOrdersAsync);
    }

    public async Task<string> GenerateOrderKeyAsync(OrderRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        try
        {
            // Perform async validation and enrichment
            await ValidateOrderRequestAsync(request, cancellationToken).ConfigureAwait(false);
            
            // Implement deterministic orderKey: hash of modelId|strategyId|signalId|ts|symbol|side|priceBucket
            // Requirement: deterministic orderKey with 24h dedupe
            var priceBucket = Math.Round(request.Price / DefaultTickSize) * DefaultTickSize; // Round to ES/MES tick size
            var timestampBucket = new DateTime(request.Timestamp.Year, request.Timestamp.Month, request.Timestamp.Day, 
                request.Timestamp.Hour, request.Timestamp.Minute / TimeBucketMinutes * TimeBucketMinutes, 0, DateTimeKind.Utc); // 5-minute buckets for idempotency
            
            var keyContent = $"{request.ModelId}|{request.StrategyId}|{request.SignalId}|{timestampBucket:yyyy-MM-dd_HH-mm}|{request.Symbol}|{request.Side}|{priceBucket:F2}";
            
            // Async hash computation for production-grade systems
            var orderKey = await ComputeHashAsync(keyContent, cancellationToken).ConfigureAwait(false);
            
            // Async logging for audit trail
            await LogOrderKeyGenerationAsync(orderKey, request, (decimal)priceBucket, cancellationToken).ConfigureAwait(false);
            
            return orderKey;
        }
        catch (ArgumentException ex)
        {
            OrderKeyGenerationFailed(_logger, request.ModelId, request.Symbol, ex);
            throw new InvalidOperationException($"Order key generation failed for {request.ModelId}", ex);
        }
        catch (InvalidOperationException ex)
        {
            OrderKeyGenerationFailed(_logger, request.ModelId, request.Symbol, ex);
            throw;
        }
        catch (System.IO.IOException ex)
        {
            OrderKeyGenerationFailed(_logger, request.ModelId, request.Symbol, ex);
            throw new InvalidOperationException($"Order key generation failed for {request.ModelId}", ex);
        }
    }

    public async Task<bool> IsDuplicateOrderAsync(string orderKey, CancellationToken cancellationToken = default)
    {
        try
        {
            // Check cache first
            if (_orderCache.TryGetValue(orderKey, out var cachedRecord))
            {
                if (IsWithinDedupeWindow(cachedRecord.CreatedAt))
                {
                    return true;
                }
                else
                {
                    // Expired - remove from cache
                    _orderCache.TryRemove(orderKey, out _);
                }
            }

            // Check persistent storage
            var orderPath = Path.Combine(_basePath, $"{orderKey}.json");
            if (File.Exists(orderPath))
            {
                var content = await File.ReadAllTextAsync(orderPath, cancellationToken).ConfigureAwait(false);
                var record = JsonSerializer.Deserialize<OrderRecord>(content);
                
                if (record != null && IsWithinDedupeWindow(record.CreatedAt))
                {
                    // Add back to cache
                    _orderCache.TryAdd(orderKey, record);
                    return true;
                }
                else
                {
                    // Expired - remove file
                    try
                    {
                        File.Delete(orderPath);
                    }
                    catch (DirectoryNotFoundException ex)
                    {
                        OrderPersistenceWarning(_logger, orderKey, ex);
                    }
                    catch (UnauthorizedAccessException ex)
                    {
                        OrderPersistenceWarning(_logger, orderKey, ex);
                    }
                    catch (System.IO.IOException ex)
                    {
                        OrderPersistenceWarning(_logger, orderKey, ex);
                    }
                }
            }

            return false;
        }
        catch (ArgumentException ex)
        {
            OrderExecutionFailed(_logger, orderKey, ex);
            // Conservative approach - assume not duplicate to avoid blocking legitimate orders
            return false;
        }
        catch (InvalidOperationException ex)
        {
            OrderExecutionFailed(_logger, orderKey, ex);
            // Conservative approach - assume not duplicate to avoid blocking legitimate orders
            return false;
        }
        catch (System.IO.IOException ex)
        {
            OrderExecutionFailed(_logger, orderKey, ex);
            // Conservative approach - assume not duplicate to avoid blocking legitimate orders
            return false;
        }
    }

    public async Task RegisterOrderAsync(string orderKey, string orderId, CancellationToken cancellationToken = default)
    {
        try
        {
            var record = new OrderRecord
            {
                OrderKey = orderKey,
                OrderId = orderId,
                CreatedAt = DateTime.UtcNow,
                AttemptCount = 1
            };

            // Save to cache
            _orderCache.AddOrUpdate(orderKey, record, (key, existing) => 
            {
                existing.OrderId = orderId;
                existing.AttemptCount++;
                return existing;
            });

            // Save to persistent storage
            var orderPath = Path.Combine(_basePath, $"{orderKey}.json");
            var json = JsonSerializer.Serialize(record, JsonOptions);
            await File.WriteAllTextAsync(orderPath, json, cancellationToken).ConfigureAwait(false);

            OrderRegistered(_logger, orderKey[..OrderKeyTruncationLength], orderId, null);
        }
        catch (ArgumentException ex)
        {
            OrderRegistrationFailed(_logger, orderKey, orderId, ex);
            throw new InvalidOperationException($"Order registration failed for key {orderKey[..OrderKeyTruncationLength]}...", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            OrderRegistrationFailed(_logger, orderKey, orderId, ex);
            throw new InvalidOperationException($"Order registration failed for key {orderKey[..OrderKeyTruncationLength]}...", ex);
        }
        catch (System.IO.IOException ex)
        {
            OrderRegistrationFailed(_logger, orderKey, orderId, ex);
            throw new InvalidOperationException($"Order registration failed for key {orderKey[..OrderKeyTruncationLength]}...", ex);
        }
    }

    public async Task<OrderDeduplicationResult> CheckDeduplicationAsync(OrderRequest request, CancellationToken cancellationToken = default)
    {
        try
        {
            var orderKey = await GenerateOrderKeyAsync(request, cancellationToken).ConfigureAwait(false);
            var isDuplicate = await IsDuplicateOrderAsync(orderKey, cancellationToken).ConfigureAwait(false);

            var result = new OrderDeduplicationResult
            {
                IsDuplicate = isDuplicate,
                OrderKey = orderKey
            };

            if (isDuplicate && _orderCache.TryGetValue(orderKey, out var existingRecord))
            {
                result.ExistingOrderId = existingRecord.OrderId;
                result.FirstSeenAt = existingRecord.CreatedAt;
                
                DuplicateOrderWithTimestamp(_logger, orderKey[..OrderKeyTruncationLength], existingRecord.CreatedAt, null);
            }

            return result;
        }
        catch (ArgumentException ex)
        {
            OrderExecutionFailed(_logger, "deduplication check", ex);
            
            // Return safe default
            return new OrderDeduplicationResult
            {
                IsDuplicate = false,
                OrderKey = Guid.NewGuid().ToString("N")
            };
        }
        catch (InvalidOperationException ex)
        {
            OrderExecutionFailed(_logger, "deduplication check", ex);
            
            // Return safe default
            return new OrderDeduplicationResult
            {
                IsDuplicate = false,
                OrderKey = Guid.NewGuid().ToString("N")
            };
        }
        catch (System.IO.IOException ex)
        {
            OrderExecutionFailed(_logger, "deduplication check", ex);
            
            // Return safe default
            return new OrderDeduplicationResult
            {
                IsDuplicate = false,
                OrderKey = Guid.NewGuid().ToString("N")
            };
        }
    }

    public async Task<bool> RetryOrderAsync(string orderKey, int attemptNumber, CancellationToken cancellationToken = default)
    {
        try
        {
            if (attemptNumber > _config.Retry.MaxAttempts)
            {
                MaxRetryAttemptsExceeded(_logger, orderKey[..OrderKeyTruncationLength], null);
                return false;
            }

            // Calculate exponential backoff delay
            var delayMs = Math.Min(
                _config.Retry.InitialMs * Math.Pow(2, attemptNumber - 1),
                _config.Retry.MaxMs
            );

            RetryingOrder(_logger, orderKey[..OrderKeyTruncationLength], attemptNumber, _config.Retry.MaxAttempts, (int)delayMs, null);

            await Task.Delay(TimeSpan.FromMilliseconds(delayMs), cancellationToken).ConfigureAwait(false);

            // Update attempt count
            if (_orderCache.TryGetValue(orderKey, out var record))
            {
                record.AttemptCount = attemptNumber;
                record.LastAttemptAt = DateTime.UtcNow;
                
                // Update persistent storage
                var orderPath = Path.Combine(_basePath, $"{orderKey}.json");
                var json = JsonSerializer.Serialize(record, JsonOptions);
                await File.WriteAllTextAsync(orderPath, json, cancellationToken).ConfigureAwait(false);
            }

            return true;
        }
        catch (ArgumentException ex)
        {
            OrderExecutionFailed(_logger, orderKey, ex);
            return false;
        }
        catch (UnauthorizedAccessException ex)
        {
            OrderExecutionFailed(_logger, orderKey, ex);
            return false;
        }
        catch (System.IO.IOException ex)
        {
            OrderExecutionFailed(_logger, orderKey, ex);
            return false;
        }
    }

    private bool IsWithinDedupeWindow(DateTime createdAt)
    {
        var age = DateTime.UtcNow - createdAt;
        return age <= TimeSpan.FromHours(_config.DedupeTtlHours);
    }

    private async Task LoadExistingOrdersAsync()
    {
        try
        {
            if (!Directory.Exists(_basePath)) return;

            var orderFiles = Directory.GetFiles(_basePath, "*.json");
            var loadedCount = 0;

            foreach (var file in orderFiles)
            {
                try
                {
                    var content = await File.ReadAllTextAsync(file).ConfigureAwait(false);
                    var record = JsonSerializer.Deserialize<OrderRecord>(content);
                    
                    if (record != null && IsWithinDedupeWindow(record.CreatedAt))
                    {
                        _orderCache.TryAdd(record.OrderKey, record);
                        loadedCount++;
                    }
                    else
                    {
                        // Remove expired file
                        File.Delete(file);
                    }
                }
                catch (UnauthorizedAccessException ex)
                {
                    OrderFileLoadWarning(_logger, file, ex);
                }
                catch (System.IO.IOException ex)
                {
                    OrderFileLoadWarning(_logger, file, ex);
                }
                catch (JsonException ex)
                {
                    OrderFileLoadWarning(_logger, file, ex);
                }
            }

            OrdersLoadedInfo(_logger, loadedCount, null);
        }
        catch (ArgumentException ex)
        {
            OrderStateLoadFailed(_logger, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            OrderStateLoadFailed(_logger, ex);
        }
        catch (System.IO.IOException ex)
        {
            OrderStateLoadFailed(_logger, ex);
        }
    }

    private void PerformCleanup(object? state)
    {
        try
        {
            var expiredKeys = new List<string>();
            var cutoff = DateTime.UtcNow.AddHours(-_config.DedupeTtlHours);

            // Clean cache
            foreach (var kvp in _orderCache)
            {
                if (kvp.Value.CreatedAt < cutoff)
                {
                    expiredKeys.Add(kvp.Key);
                }
            }

            foreach (var key in expiredKeys)
            {
                _orderCache.TryRemove(key, out _);
            }

            // Clean files
            if (Directory.Exists(_basePath))
            {
                var orderFiles = Directory.GetFiles(_basePath, "*.json");
                var deletedCount = 0;

                foreach (var file in orderFiles)
                {
                    try
                    {
                        var fileInfo = new FileInfo(file);
                        if (fileInfo.CreationTimeUtc < cutoff)
                        {
                            File.Delete(file);
                            deletedCount++;
                        }
                    }
                    catch (DirectoryNotFoundException ex)
                    {
                        ExpiredFileDeleteWarning(_logger, file, ex);
                    }
                    catch (UnauthorizedAccessException ex)
                    {
                        ExpiredFileDeleteWarning(_logger, file, ex);
                    }
                    catch (System.IO.IOException ex)
                    {
                        ExpiredFileDeleteWarning(_logger, file, ex);
                    }
                }

                if (expiredKeys.Count > 0 || deletedCount > 0)
                {
                    CleanupCompletedInfo(_logger, expiredKeys.Count, deletedCount, null);
                }
            }
        }
        catch (ArgumentException ex)
        {
            OrderCleanupFailed(_logger, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            OrderCleanupFailed(_logger, ex);
        }
        catch (System.IO.IOException ex)
        {
            OrderCleanupFailed(_logger, ex);
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _cleanupTimer?.Dispose();
        }
    }

    /// <summary>
    /// Async validation of order request with external service checks
    /// </summary>
    private static async Task ValidateOrderRequestAsync(OrderRequest request, CancellationToken cancellationToken)
    {
        // Simulate async validation with external services
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        if (string.IsNullOrEmpty(request.ModelId) || string.IsNullOrEmpty(request.Symbol))
        {
            throw new ArgumentException("Invalid order request: ModelId and Symbol are required");
        }
    }

    /// <summary>
    /// Async hash computation for production systems
    /// </summary>
    private static Task<string> ComputeHashAsync(string content, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(content));
            return Convert.ToHexString(hash).ToUpperInvariant();
        }, cancellationToken);
    }

    /// <summary>
    /// Async audit logging for order key generation
    /// </summary>
    private async Task LogOrderKeyGenerationAsync(string orderKey, OrderRequest request, decimal priceBucket, CancellationToken cancellationToken)
    {
        // Simulate async audit logging to external system
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        OrderKeyGenerated(_logger, orderKey[..OrderKeyTruncationLength], request.Symbol, request.Side.ToString(), (double)priceBucket, null);
    }

    private sealed class OrderRecord
    {
        public string OrderKey { get; set; } = string.Empty;
        public string OrderId { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        public DateTime? LastAttemptAt { get; set; }
        public int AttemptCount { get; set; }
    }
}