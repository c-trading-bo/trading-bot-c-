using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace BotCore.Services;

/// <summary>
/// Production-ready order evidence service following guardrails:
/// "No fills without proof. Before saying 'filled', require at least one of:
/// - orderId returned by the place-order call and
/// - a fill event from the User Hub or
/// - Trade search shows an execution."
/// </summary>
public class ProductionOrderEvidenceService
{
    private readonly ILogger<ProductionOrderEvidenceService> _logger;

    public ProductionOrderEvidenceService(ILogger<ProductionOrderEvidenceService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Verify order fill evidence according to production guardrails
    /// </summary>
    public async Task<OrderEvidenceResult> VerifyOrderFillEvidenceAsync(
        string? orderId,
        GatewayUserTrade? fillEvent,
        string customTag,
        CancellationToken cancellationToken = default)
    {
        var result = new OrderEvidenceResult
        {
            OrderId = orderId,
            CustomTag = customTag,
            Timestamp = DateTime.UtcNow
        };

        _logger.LogInformation("🔍 [ORDER-EVIDENCE] Verifying fill evidence for tag: {CustomTag}, orderId: {OrderId}", 
            customTag, orderId ?? "NULL");

        // Requirement 1: orderId returned by place-order call
        bool hasOrderId = !string.IsNullOrWhiteSpace(orderId);
        
        // Requirement 2: Fill event from User Hub
        bool hasFillEvent = fillEvent != null;
        
        // Log evidence findings
        if (hasOrderId)
        {
            _logger.LogInformation("✅ [ORDER-EVIDENCE] Evidence 1: OrderId present - {OrderId}", orderId);
            result.EvidenceTypes.Add("OrderId");
        }
        else
        {
            _logger.LogWarning("❌ [ORDER-EVIDENCE] Evidence 1: Missing OrderId");
        }

        if (hasFillEvent)
        {
            _logger.LogInformation("✅ [ORDER-EVIDENCE] Evidence 2: Fill event present - OrderId: {EventOrderId}, FillPrice: {FillPrice}, Qty: {Qty}", 
                fillEvent.OrderId, fillEvent.FillPrice, fillEvent.Quantity);
            result.EvidenceTypes.Add("FillEvent");
            result.FillPrice = fillEvent.FillPrice;
            result.Quantity = fillEvent.Quantity;
        }
        else
        {
            _logger.LogWarning("❌ [ORDER-EVIDENCE] Evidence 2: Missing fill event");
        }

        // TODO: Requirement 3: Trade search verification (would require trade search service)
        // For now, we'll mark this as not implemented
        _logger.LogInformation("⏳ [ORDER-EVIDENCE] Evidence 3: Trade search verification not implemented yet");

        // Determine if we have sufficient evidence
        result.HasSufficientEvidence = hasOrderId && hasFillEvent;
        result.TotalEvidenceCount = result.EvidenceTypes.Count;

        if (result.HasSufficientEvidence)
        {
            _logger.LogInformation("✅ [ORDER-EVIDENCE] SUFFICIENT EVIDENCE for fill - CustomTag: {CustomTag}, Evidence: [{Evidence}]", 
                customTag, string.Join(", ", result.EvidenceTypes));
        }
        else
        {
            _logger.LogCritical("🔴 [ORDER-EVIDENCE] INSUFFICIENT EVIDENCE for fill - CustomTag: {CustomTag}, Evidence: [{Evidence}]", 
                customTag, string.Join(", ", result.EvidenceTypes));
            _logger.LogCritical("🔴 [ORDER-EVIDENCE] GUARDRAIL VIOLATION: Cannot claim fill without proper evidence");
        }

        await Task.CompletedTask; // Ensure async pattern
        return result;
    }

    /// <summary>
    /// Log structured order information following guardrails format
    /// </summary>
    public void LogOrderStructured(string signal, string side, string symbol, int quantity, 
        decimal entry, decimal stop, decimal target, decimal rMultiple, string customTag, string? orderId = null)
    {
        // Following guardrails format:
        // [{sig}] side={BUY|SELL} symbol={ES} qty={n} entry={0.00} stop={0.00} t1={0.00} R~{0.00} tag={customTag} orderId={guid?}
        _logger.LogInformation("[{Signal}] side={Side} symbol={Symbol} qty={Quantity} entry={Entry:0.00} stop={Stop:0.00} t1={Target:0.00} R~{RMultiple:0.00} tag={CustomTag} orderId={OrderId}", 
            signal, side, symbol, quantity, entry, stop, target, rMultiple, customTag, orderId ?? "NULL");
    }

    /// <summary>
    /// Log order status following guardrails format
    /// </summary>
    public void LogOrderStatus(string accountId, string status, string? orderId, string? reason = null)
    {
        // Following guardrails format:
        // ORDER account={id} status={New|Open|Filled|Cancelled|Rejected} orderId={id} reason={...}
        _logger.LogInformation("ORDER account={AccountId} status={Status} orderId={OrderId} reason={Reason}", 
            accountId, status, orderId ?? "NULL", reason ?? "N/A");
    }

    /// <summary>
    /// Log trade execution following guardrails format
    /// </summary>
    public void LogTrade(string accountId, string? orderId, decimal fillPrice, int quantity, DateTime time)
    {
        // Following guardrails format:
        // TRADE account={id} orderId={id} fillPrice={0.00} qty={n} time={iso}
        _logger.LogInformation("TRADE account={AccountId} orderId={OrderId} fillPrice={FillPrice:0.00} qty={Quantity} time={Time:yyyy-MM-ddTHH:mm:ss.fffZ}", 
            accountId, orderId ?? "NULL", fillPrice, quantity, time);
    }
}

/// <summary>
/// Result of order evidence verification
/// </summary>
public class OrderEvidenceResult
{
    public string? OrderId { get; set; }
    public string CustomTag { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public List<string> EvidenceTypes { get; set; } = new();
    public int TotalEvidenceCount { get; set; }
    public bool HasSufficientEvidence { get; set; }
    public decimal? FillPrice { get; set; }
    public int? Quantity { get; set; }
}