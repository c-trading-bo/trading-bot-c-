using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Hosting;
using System.Collections.Concurrent;
using TradingBot.Abstractions;
using Trading.Safety.Journaling;

namespace Trading.Safety.OrderLifecycle;

/// <summary>
/// Production-grade order lifecycle management with evidence consolidation
/// Correlates orderId, fills, and decision metadata in comprehensive tracking
/// Handles stale order timeout/cancel and idempotent replay protection
/// </summary>
public interface IOrderLifecycleManager
{
    Task<string> CreateOrderEvidenceAsync(OrderCreationRequest request);
    Task UpdateOrderEvidenceAsync(string evidenceId, OrderUpdateEvent updateEvent);
    Task<OrderEvidence?> GetOrderEvidenceAsync(string evidenceId);
    Task<List<OrderEvidence>> GetActiveOrdersAsync();
    Task<bool> IsCustomTagUsedAsync(string customTag);
    Task CancelStaleOrdersAsync();
    Task ForceTimeoutOrderAsync(string evidenceId, string reason);
    event Action<OrderEvidence> OnOrderEvidenceUpdated;
    event Action<OrderEvidence> OnStaleOrderDetected;
}

public class OrderLifecycleManager : IOrderLifecycleManager, IHostedService
{
    private readonly ILogger<OrderLifecycleManager> _logger;
    private readonly OrderLifecycleConfig _config;
    private readonly ITradeJournal _tradeJournal;
    private readonly ConcurrentDictionary<string, OrderEvidence> _orderEvidence = new();
    private readonly ConcurrentDictionary<string, bool> _usedCustomTags = new();
    private readonly Timer _staleOrderTimer;
    private readonly object _lock = new object();

    public event Action<OrderEvidence> OnOrderEvidenceUpdated = delegate { };
    public event Action<OrderEvidence> OnStaleOrderDetected = delegate { };

    public OrderLifecycleManager(
        ILogger<OrderLifecycleManager> logger,
        IOptions<OrderLifecycleConfig> config,
        ITradeJournal tradeJournal)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _config = config?.Value ?? throw new ArgumentNullException(nameof(config));
        _tradeJournal = tradeJournal ?? throw new ArgumentNullException(nameof(tradeJournal));
        
        _staleOrderTimer = new Timer(CheckStaleOrdersCallback, null, Timeout.Infinite, Timeout.Infinite);
    }

    public async Task<string> CreateOrderEvidenceAsync(OrderCreationRequest request)
    {
        // Check for duplicate customTag (idempotent replay guard)
        if (await IsCustomTagUsedAsync(request.CustomTag))
        {
            _logger.LogWarning("[ORDER_LIFECYCLE] Duplicate customTag detected: {CustomTag} - blocking replay", 
                request.CustomTag).ConfigureAwait(false);
            throw new InvalidOperationException($"CustomTag {request.CustomTag} already used - preventing duplicate order");
        }

        var evidenceId = Guid.NewGuid().ToString("N");
        var correlationId = request.CorrelationId ?? Guid.NewGuid().ToString("N")[..8];

        var evidence = new OrderEvidence
        {
            EvidenceId = evidenceId,
            TradeId = request.TradeId,
            CorrelationId = correlationId,
            CustomTag = request.CustomTag,
            CreatedAt = DateTime.UtcNow,
            Status = OrderEvidenceStatus.Created,
            Symbol = request.Symbol,
            Side = request.Side,
            Quantity = request.Quantity,
            OrderType = request.OrderType,
            LimitPrice = request.LimitPrice,
            StopPrice = request.StopPrice,
            Strategy = request.Strategy,
            DecisionContext = request.DecisionContext ?? new(),
            TimeoutAt = DateTime.UtcNow.Add(_config.OrderTimeout),
            Metadata = new Dictionary<string, object>
            {
                ["created_by"] = "OrderLifecycleManager",
                ["request_timestamp"] = request.Timestamp,
                ["timeout_seconds"] = _config.OrderTimeout.TotalSeconds
            }
        };

        _orderEvidence.TryAdd(evidenceId, evidence);
        _usedCustomTags.TryAdd(request.CustomTag, true);

        // Log to trade journal
        await _tradeJournal.LogDecisionAsync(new TradingDecisionEvent
        {
            TradeId = request.TradeId,
            CorrelationId = correlationId,
            Timestamp = DateTime.UtcNow,
            Symbol = request.Symbol,
            Side = request.Side,
            Quantity = request.Quantity,
            LimitPrice = request.LimitPrice,
            StopPrice = request.StopPrice,
            Strategy = request.Strategy,
            Context = request.DecisionContext ?? new()
        }).ConfigureAwait(false);

        _logger.LogInformation("[ORDER_LIFECYCLE] Order evidence created: {EvidenceId} {Symbol} {Side} {Quantity} tag={CustomTag} [CorrelationId: {CorrelationId}]",
            evidenceId, request.Symbol, request.Side, request.Quantity, request.CustomTag, correlationId);

        OnOrderEvidenceUpdated.Invoke(evidence);
        return evidenceId;
    }

    public async Task UpdateOrderEvidenceAsync(string evidenceId, OrderUpdateEvent updateEvent)
    {
        if (!_orderEvidence.TryGetValue(evidenceId, out var evidence))
        {
            _logger.LogWarning("[ORDER_LIFECYCLE] Update for unknown evidence ID: {EvidenceId}", evidenceId);
            return;
        }

        lock (_lock)
        {
            evidence.UpdatedAt = DateTime.UtcNow;

            switch (updateEvent.EventType)
            {
                case OrderEventType.OrderPlaced:
                    evidence.OrderId = updateEvent.OrderId;
                    evidence.Status = OrderEvidenceStatus.Placed;
                    evidence.PlacedAt = updateEvent.Timestamp;
                    break;

                case OrderEventType.OrderAccepted:
                    evidence.Status = OrderEvidenceStatus.Open;
                    evidence.AcceptedAt = updateEvent.Timestamp;
                    break;

                case OrderEventType.PartialFill:
                    evidence.Fills ??= new List<FillEvidence>();
                    evidence.Fills.Add(new FillEvidence
                    {
                        FillId = updateEvent.FillId!,
                        Timestamp = updateEvent.Timestamp,
                        Quantity = updateEvent.FilledQuantity,
                        Price = updateEvent.FillPrice,
                        Commission = updateEvent.Commission ?? 0m
                    });
                    evidence.TotalFilled += updateEvent.FilledQuantity;
                    evidence.RemainingQuantity = evidence.Quantity - evidence.TotalFilled;
                    evidence.Status = OrderEvidenceStatus.PartiallyFilled;
                    evidence.LastFillAt = updateEvent.Timestamp;
                    break;

                case OrderEventType.FullFill:
                    evidence.Fills ??= new List<FillEvidence>();
                    evidence.Fills.Add(new FillEvidence
                    {
                        FillId = updateEvent.FillId!,
                        Timestamp = updateEvent.Timestamp,
                        Quantity = updateEvent.FilledQuantity,
                        Price = updateEvent.FillPrice,
                        Commission = updateEvent.Commission ?? 0m
                    });
                    evidence.TotalFilled += updateEvent.FilledQuantity;
                    evidence.RemainingQuantity;
                    evidence.Status = OrderEvidenceStatus.FullyFilled;
                    evidence.CompletedAt = updateEvent.Timestamp;
                    evidence.LastFillAt = updateEvent.Timestamp;
                    break;

                case OrderEventType.Cancelled:
                    evidence.Status = OrderEvidenceStatus.Cancelled;
                    evidence.CompletedAt = updateEvent.Timestamp;
                    evidence.CancelReason = updateEvent.Reason;
                    break;

                case OrderEventType.Rejected:
                    evidence.Status = OrderEvidenceStatus.Rejected;
                    evidence.CompletedAt = updateEvent.Timestamp;
                    evidence.RejectReason = updateEvent.Reason;
                    break;

                case OrderEventType.Timeout:
                    evidence.Status = OrderEvidenceStatus.TimedOut;
                    evidence.CompletedAt = updateEvent.Timestamp;
                    evidence.TimeoutReason = updateEvent.Reason;
                    break;
            }

            // Update metadata
            if (updateEvent.Metadata != null)
            {
                foreach (var kvp in updateEvent.Metadata)
                {
                    evidence.Metadata[$"event_{updateEvent.EventType}_{kvp.Key}"] = kvp.Value;
                }
            }
        }

        // Log to trade journal
        await LogEventToJournalAsync(evidence, updateEvent).ConfigureAwait(false);

        _logger.LogInformation("[ORDER_LIFECYCLE] Evidence updated: {EvidenceId} {EventType} {Status} [CorrelationId: {CorrelationId}]",
            evidenceId, updateEvent.EventType, evidence.Status, evidence.CorrelationId);

        OnOrderEvidenceUpdated.Invoke(evidence);

        // Remove from active tracking if completed
        if (evidence.Status.IsCompleted())
        {
            _ = Task.Delay(TimeSpan.FromMinutes(5)).ContinueWith(t =>
            {
                _orderEvidence.TryRemove(evidenceId, out _);
            });
        }
    }

    public async Task<OrderEvidence?> GetOrderEvidenceAsync(string evidenceId)
    {
        return await Task.FromResult(_orderEvidence.GetValueOrDefault(evidenceId)).ConfigureAwait(false);
    }

    public async Task<List<OrderEvidence>> GetActiveOrdersAsync()
    {
        var activeOrders = _orderEvidence.Values
            .Where(e => !e.Status.IsCompleted())
            .OrderBy(e => e.CreatedAt)
            .ToList();

        return await Task.FromResult(activeOrders).ConfigureAwait(false);
    }

    public async Task<bool> IsCustomTagUsedAsync(string customTag)
    {
        return await Task.FromResult(_usedCustomTags.ContainsKey(customTag)).ConfigureAwait(false);
    }

    public async Task CancelStaleOrdersAsync()
    {
        var now = DateTime.UtcNow;
        var staleOrders = _orderEvidence.Values
            .Where(e => !e.Status.IsCompleted() && now > e.TimeoutAt)
            .ToList();

        foreach (var staleOrder in staleOrders)
        {
            await ForceTimeoutOrderAsync(staleOrder.EvidenceId, "Stale order timeout").ConfigureAwait(false);
            OnStaleOrderDetected.Invoke(staleOrder);
        }
    }

    public async Task ForceTimeoutOrderAsync(string evidenceId, string reason)
    {
        await UpdateOrderEvidenceAsync(evidenceId, new OrderUpdateEvent
        {
            EventType = OrderEventType.Timeout,
            Timestamp = DateTime.UtcNow,
            Reason = reason,
            Metadata = new Dictionary<string, object>
            {
                ["forced_timeout"] = true,
                ["timeout_reason"] = reason
            }
        }).ConfigureAwait(false);

        _logger.LogWarning("[ORDER_LIFECYCLE] Order forcibly timed out: {EvidenceId} - {Reason}", 
            evidenceId, reason);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _staleOrderTimer.Change(TimeSpan.Zero, _config.StaleOrderCheckInterval);
        _logger.LogInformation("[ORDER_LIFECYCLE] Started with timeout: {Timeout}, check interval: {Interval}",
            _config.OrderTimeout, _config.StaleOrderCheckInterval);
        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _staleOrderTimer.Change(Timeout.Infinite, Timeout.Infinite);
        _logger.LogInformation("[ORDER_LIFECYCLE] Stopped");
        await Task.CompletedTask.ConfigureAwait(false);
    }

    private void CheckStaleOrdersCallback(object? state)
    {
        try
        {
            _ = Task.Run(async () => await CancelStaleOrdersAsync()).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ORDER_LIFECYCLE] Error checking stale orders");
        }
    }

    private async Task LogEventToJournalAsync(OrderEvidence evidence, OrderUpdateEvent updateEvent)
    {
        switch (updateEvent.EventType)
        {
            case OrderEventType.OrderPlaced:
                await _tradeJournal.LogOrderAsync(new OrderEvent
                {
                    TradeId = evidence.TradeId,
                    OrderId = evidence.OrderId!,
                    CustomTag = evidence.CustomTag,
                    Timestamp = updateEvent.Timestamp,
                    Status = "PLACED",
                    Metadata = updateEvent.Metadata ?? new()
                }).ConfigureAwait(false);
                break;

            case OrderEventType.PartialFill:
            case OrderEventType.FullFill:
                await _tradeJournal.LogFillAsync(new FillEvent
                {
                    TradeId = evidence.TradeId,
                    FillId = updateEvent.FillId!,
                    OrderId = evidence.OrderId!,
                    Timestamp = updateEvent.Timestamp,
                    FilledQuantity = updateEvent.FilledQuantity,
                    FillPrice = updateEvent.FillPrice,
                    Commission = updateEvent.Commission ?? 0m,
                    Metadata = updateEvent.Metadata ?? new()
                }).ConfigureAwait(false);
                break;


        }
    }

    public void Dispose()
    {
        _staleOrderTimer?.Dispose();
    }
}

// Data models
public class OrderEvidence
{
    public string EvidenceId { get; set; } = string.Empty;
    public string TradeId { get; set; } = string.Empty;
    public string CorrelationId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public string? OrderId { get; set; }
    
    public DateTime CreatedAt { get; set; }
    public DateTime? UpdatedAt { get; set; }
    public DateTime? PlacedAt { get; set; }
    public DateTime? AcceptedAt { get; set; }
    public DateTime? LastFillAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public DateTime TimeoutAt { get; set; }
    
    public OrderEvidenceStatus Status { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal TotalFilled { get; set; }
    public decimal RemainingQuantity { get; set; }
    public string OrderType { get; set; } = string.Empty;
    public decimal? LimitPrice { get; set; }
    public decimal? StopPrice { get; set; }
    public string Strategy { get; set; } = string.Empty;
    
    public List<FillEvidence>? Fills { get; set; }
    public Dictionary<string, object> DecisionContext { get; } = new();
    public Dictionary<string, object> Metadata { get; } = new();
    
    public string? RejectReason { get; set; }
    public string? CancelReason { get; set; }
    public string? TimeoutReason { get; set; }
}

public class FillEvidence
{
    public string FillId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public decimal Commission { get; set; }
}

public enum OrderEvidenceStatus
{
    Created,
    Placed,
    Open,
    PartiallyFilled,
    FullyFilled,
    Cancelled,
    Rejected,
    TimedOut
}

public enum OrderEventType
{
    OrderPlaced,
    OrderAccepted,
    PartialFill,
    FullFill,
    Cancelled,
    Rejected,
    Timeout
}

public class OrderCreationRequest
{
    public string TradeId { get; set; } = string.Empty;
    public string? CorrelationId { get; set; }
    public string CustomTag { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public string OrderType { get; set; } = string.Empty;
    public decimal? LimitPrice { get; set; }
    public decimal? StopPrice { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public Dictionary<string, object>? DecisionContext { get; set; }
}

public class OrderUpdateEvent
{
    public OrderEventType EventType { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string? OrderId { get; set; }
    public string? FillId { get; set; }
    public decimal FilledQuantity { get; set; }
    public decimal FillPrice { get; set; }
    public decimal? Commission { get; set; }
    public string? Reason { get; set; }
    public Dictionary<string, object>? Metadata { get; set; }
}

public class OrderLifecycleConfig
{
    public TimeSpan OrderTimeout { get; set; } = TimeSpan.FromMinutes(5);
    public TimeSpan StaleOrderCheckInterval { get; set; } = TimeSpan.FromMinutes(1);
    public bool EnableIdempotencyCheck { get; set; } = true;
    public bool EnableJournaling { get; set; } = true;
}

public static class OrderEvidenceStatusExtensions
{
    public static bool IsCompleted(this OrderEvidenceStatus status)
    {
        return status is OrderEvidenceStatus.FullyFilled 
                    or OrderEvidenceStatus.Cancelled 
                    or OrderEvidenceStatus.Rejected 
                    or OrderEvidenceStatus.TimedOut;
    }
}

public static class EnumerableExtensions
{
    public static decimal WeightedAverage<T>(this IEnumerable<T> source, 
        Func<T, decimal> valueSelector, 
        Func<T, decimal> weightSelector)
    {
        var weightedSum = source.Sum(x => valueSelector(x) * weightSelector(x));
        var totalWeight = source.Sum(weightSelector);
        return totalWeight == 0 ? 0 : weightedSum / totalWeight;
    }
}