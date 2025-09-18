using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System.Globalization;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Enhanced trading activity logger with comprehensive signal, order, and position tracking
/// Integrates with all trading services to provide complete audit trail
/// </summary>
public class TradingActivityLogger
{
    private readonly ILogger<TradingActivityLogger> _logger;
    private readonly ITradingLogger _tradingLogger;
    
    public TradingActivityLogger(
        ILogger<TradingActivityLogger> logger,
        ITradingLogger tradingLogger)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
    }

    /// <summary>
    /// Log signal generation with comprehensive details
    /// </summary>
    public async Task LogSignalGenerationAsync(string strategy, string symbol, string direction, 
        decimal confidence, decimal entry, decimal stop, decimal target, decimal rMultiple, 
        string customTag, object? context = null)
    {
        // Format as specified in the requirements
        var signalMessage = $"[{customTag}] side={direction} symbol={symbol} qty=1 entry={entry:0.00} stop={stop:0.00} t1={target:0.00} R~{rMultiple:0.00} tag={customTag}";
        
        var signalData = new
        {
            strategy,
            symbol,
            direction,
            quantity = 1,
            entry = entry.ToString("0.00", CultureInfo.InvariantCulture),
            stop = stop.ToString("0.00", CultureInfo.InvariantCulture),
            target = target.ToString("0.00", CultureInfo.InvariantCulture),
            rMultiple = rMultiple.ToString("0.00", CultureInfo.InvariantCulture),
            confidence = confidence.ToString("0.00", CultureInfo.InvariantCulture),
            customTag,
            context
        };

        await _tradingLogger.LogEventAsync(TradingLogCategory.SIGNAL, TradingLogLevel.INFO, 
            "SIGNAL_GENERATED", signalData, customTag).ConfigureAwait(false);

        // Also log to console with structured format
        _logger.LogInformation("📡 SIGNAL: {SignalMessage}", signalMessage);
    }

    /// <summary>
    /// Log order placement with comprehensive details
    /// </summary>
    public async Task LogOrderPlacementAsync(string accountId, string orderId, string symbol, 
        string side, decimal quantity, decimal price, string customTag, object? orderDetails = null)
    {
        var orderMessage = $"ORDER account={accountId} status=New orderId={orderId} symbol={symbol} side={side} qty={quantity} price={price:0.00} tag={customTag}";
        
        var orderData = new
        {
            accountId,
            orderId,
            symbol,
            side,
            quantity,
            price = price.ToString("0.00", CultureInfo.InvariantCulture),
            customTag,
            status = "New",
            orderDetails
        };

        await _tradingLogger.LogEventAsync(TradingLogCategory.ORDER, TradingLogLevel.INFO, 
            "ORDER_PLACED", orderData, customTag).ConfigureAwait(false);

        _logger.LogInformation("📋 ORDER: {OrderMessage}", orderMessage);
    }

    /// <summary>
    /// Log order status changes
    /// </summary>
    public async Task LogOrderStatusChangeAsync(string accountId, string orderId, string status, 
        string? reason = null, object? statusDetails = null)
    {
        var statusMessage = $"ORDER account={accountId} status={status} orderId={orderId}" + 
                           (reason != null ? $" reason={reason}" : "");
        
        var statusData = new
        {
            accountId,
            orderId,
            status,
            reason,
            statusDetails
        };

        await _tradingLogger.LogEventAsync(TradingLogCategory.ORDER, TradingLogLevel.INFO, 
            "ORDER_STATUS_CHANGE", statusData, orderId).ConfigureAwait(false);

        var logLevel = status == "Rejected" || status == "Cancelled" ? LogLevel.Warning : LogLevel.Information;
        _logger.Log(logLevel, "📋 ORDER STATUS: {StatusMessage}", statusMessage);
    }

    /// <summary>
    /// Log trade fills with comprehensive details
    /// </summary>
    public async Task LogTradeFilledAsync(string accountId, string orderId, decimal fillPrice, 
        decimal quantity, DateTime fillTime, object? fillDetails = null)
    {
        var tradeMessage = $"TRADE account={accountId} orderId={orderId} fillPrice={fillPrice:0.00} qty={quantity} time={fillTime:yyyy-MM-ddTHH:mm:ss.fffZ}";
        
        var tradeData = new
        {
            accountId,
            orderId,
            fillPrice = fillPrice.ToString("0.00", CultureInfo.InvariantCulture),
            quantity,
            fillTime = fillTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture),
            fillDetails
        };

        await _tradingLogger.LogEventAsync(TradingLogCategory.FILL, TradingLogLevel.INFO, 
            "TRADE_FILLED", tradeData, orderId).ConfigureAwait(false);

        _logger.LogInformation("💰 TRADE: {TradeMessage}", tradeMessage);
    }

    /// <summary>
    /// Log position changes with P&L tracking
    /// </summary>
    public async Task LogPositionChangeAsync(string accountId, string symbol, decimal positionSize, 
        decimal averagePrice, decimal unrealizedPnL, object? positionDetails = null)
    {
        var positionData = new
        {
            accountId,
            symbol,
            positionSize,
            averagePrice = averagePrice.ToString("0.00", CultureInfo.InvariantCulture),
            unrealizedPnL = unrealizedPnL.ToString("0.00", CultureInfo.InvariantCulture),
            positionDetails
        };

        await _tradingLogger.LogEventAsync(TradingLogCategory.RISK, TradingLogLevel.INFO, 
            "POSITION_CHANGE", positionData).ConfigureAwait(false);

        var pnlIcon = unrealizedPnL >= 0 ? "📈" : "📉";
        _logger.LogInformation("{Icon} POSITION: {Symbol} size={PositionSize} avgPrice={AvgPrice:0.00} PnL={PnL:0.00}", 
            pnlIcon, symbol, positionSize, averagePrice, unrealizedPnL);
    }

    /// <summary>
    /// Log risk limit checks and violations
    /// </summary>
    public async Task LogRiskCheckAsync(string riskType, bool passed, decimal currentValue, 
        decimal limit, object? riskDetails = null)
    {
        var riskData = new
        {
            riskType,
            passed,
            currentValue = currentValue.ToString("0.00", CultureInfo.InvariantCulture),
            limit = limit.ToString("0.00", CultureInfo.InvariantCulture),
            utilizationPercent = Math.Round((currentValue / limit) * 100, 2),
            riskDetails
        };

        var level = passed ? TradingLogLevel.DEBUG : TradingLogLevel.WARN;
        await _tradingLogger.LogEventAsync(TradingLogCategory.RISK, level, 
            "RISK_CHECK", riskData).ConfigureAwait(false);

        if (!passed)
        {
            _logger.LogWarning("⚠️ RISK VIOLATION: {RiskType} - Current: {Current:0.00}, Limit: {Limit:0.00}", 
                riskType, currentValue, limit);
        }
    }

    /// <summary>
    /// Log kill switch activation events
    /// </summary>
    public async Task LogKillSwitchAsync(string reason, string triggeredBy, object? details = null)
    {
        var killSwitchData = new
        {
            reason,
            triggeredBy,
            timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC", CultureInfo.InvariantCulture),
            details
        };

        await _tradingLogger.LogEventAsync(TradingLogCategory.ERROR, TradingLogLevel.ERROR, 
            "KILL_SWITCH_ACTIVATED", killSwitchData).ConfigureAwait(false);

        _logger.LogError("🛑 KILL SWITCH ACTIVATED: {Reason} (triggered by {TriggeredBy})", reason, triggeredBy);
    }
}