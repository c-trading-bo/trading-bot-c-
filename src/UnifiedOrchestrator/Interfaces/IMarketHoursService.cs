using System;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for managing futures market hours and safe promotion windows
/// </summary>
public interface IMarketHoursService
{
    /// <summary>
    /// Check if current time is within a safe promotion window
    /// </summary>
    Task<bool> IsInSafePromotionWindowAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get the next safe promotion window
    /// </summary>
    Task<DateTime?> GetNextSafeWindowAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if market is currently open for trading
    /// </summary>
    Task<bool> IsMarketOpenAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current market session
    /// </summary>
    Task<string> GetCurrentMarketSessionAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if it's a trading day (not weekend/holiday)
    /// </summary>
    Task<bool> IsTradingDayAsync(DateTime date, CancellationToken cancellationToken = default);
}

/// <summary>
/// Safe promotion window information
/// </summary>
public class SafePromotionWindow
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public string WindowType { get; set; } = string.Empty; // OVERNIGHT, POST_CLOSE, WEEKEND
    public bool IsFlat { get; set; }
    public string Description { get; set; } = string.Empty;
}