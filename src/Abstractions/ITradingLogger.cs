using System;
using System.Threading.Tasks;

namespace TradingBot.Abstractions;

/// <summary>
/// High-performance trading logger interface for production-ready logging
/// Supports structured logging with categories, async operations, and correlation tracking
/// </summary>
public interface ITradingLogger
{
    /// <summary>
    /// Log trading events with structured data
    /// </summary>
    Task LogEventAsync(TradingLogCategory category, TradingLogLevel level, string eventType, object data, string? correlationId = null);
    
    /// <summary>
    /// Log order events specifically
    /// </summary>
    Task LogOrderAsync(string side, string symbol, decimal quantity, decimal entry, decimal stopPrice, decimal target, decimal rMultiple, string customTag, string? orderId = null);
    
    /// <summary>
    /// Log trade fills
    /// </summary>
    Task LogTradeAsync(string accountId, string orderId, decimal fillPrice, decimal quantity, DateTime fillTime);
    
    /// <summary>
    /// Log order status changes
    /// </summary>
    Task LogOrderStatusAsync(string accountId, string orderId, string status, string? reason = null);
    
    /// <summary>
    /// Log system events (auth, hub connections, etc)
    /// </summary>
    Task LogSystemAsync(TradingLogLevel level, string component, string message, object? context = null);
    
    /// <summary>
    /// Log market data events
    /// </summary>
    Task LogMarketDataAsync(string symbol, string dataType, object data);
    
    /// <summary>
    /// Log ML/Risk events
    /// </summary>
    Task LogMLAsync(string model, string action, object data, string? correlationId = null);
    
    /// <summary>
    /// Get recent log entries for debugging
    /// </summary>
    Task<TradingLogEntry[]> GetRecentEntriesAsync(int count = 1000, TradingLogCategory? category = null);
    
    /// <summary>
    /// Force flush all pending log entries
    /// </summary>
    Task FlushAsync();
}

/// <summary>
/// Trading log categories for structured logging
/// </summary>
public enum TradingLogCategory
{
    ORDER,
    FILL,
    SIGNAL,
    AUTH,
    HUB,
    ERROR,
    ML,
    RISK,
    SYSTEM,
    MARKET
}

/// <summary>
/// Trading log levels
/// </summary>
public enum TradingLogLevel
{
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE
}

/// <summary>
/// Structured trading log entry
/// </summary>
public class TradingLogEntry
{
    public DateTime Timestamp { get; set; }
    public TradingLogCategory Category { get; set; }
    public TradingLogLevel Level { get; set; }
    public string EventType { get; set; } = string.Empty;
    public object? Data { get; set; }
    public object? Context { get; set; }
    public string? CorrelationId { get; set; }
    public string? AccountId { get; set; }
    public string? SessionId { get; set; }
}