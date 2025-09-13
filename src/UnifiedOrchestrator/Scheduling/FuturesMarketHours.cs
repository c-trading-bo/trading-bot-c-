using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;

namespace TradingBot.UnifiedOrchestrator.Scheduling;

/// <summary>
/// Futures market hours service with CME session behavior
/// Implements safe promotion windows to avoid disrupting live trading
/// </summary>
public class FuturesMarketHours : IMarketHoursService
{
    private readonly ILogger<FuturesMarketHours> _logger;
    
    // CME Futures trading hours (Eastern Time)
    private readonly TimeSpan MarketOpenTime = new(18, 0, 0);  // 6:00 PM ET (Sunday-Thursday)
    private readonly TimeSpan MarketCloseTime = new(17, 0, 0); // 5:00 PM ET (Monday-Friday)
    
    // Safe promotion windows (Eastern Time)
    private readonly TimeSpan OvernightWindowStart = new(4, 30, 0);  // 4:30 AM ET
    private readonly TimeSpan OvernightWindowEnd = new(5, 0, 0);     // 5:00 AM ET
    private readonly TimeSpan PostCloseWindowStart = new(16, 0, 0);  // 4:00 PM ET
    private readonly TimeSpan PostCloseWindowEnd = new(16, 15, 0);   // 4:15 PM ET

    public FuturesMarketHours(ILogger<FuturesMarketHours> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Check if current time is within a safe promotion window
    /// Safe windows: overnight 4:30-5:00 ET, post-close 16:00-16:15 ET, weekends
    /// </summary>
    public async Task<bool> IsInSafePromotionWindowAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        var etNow = GetEasternTime();
        var dayOfWeek = etNow.DayOfWeek;
        var timeOfDay = etNow.TimeOfDay;

        // Weekend is always safe (Saturday and Sunday)
        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            _logger.LogDebug("In safe promotion window: Weekend");
            return true;
        }

        // Overnight window: 4:30-5:00 AM ET (Monday-Friday)
        if (timeOfDay >= OvernightWindowStart && timeOfDay <= OvernightWindowEnd)
        {
            _logger.LogDebug("In safe promotion window: Overnight ({Start}-{End} ET)", 
                OvernightWindowStart, OvernightWindowEnd);
            return true;
        }

        // Post-close window: 4:00-4:15 PM ET (Monday-Friday)
        if (timeOfDay >= PostCloseWindowStart && timeOfDay <= PostCloseWindowEnd)
        {
            _logger.LogDebug("In safe promotion window: Post-close ({Start}-{End} ET)", 
                PostCloseWindowStart, PostCloseWindowEnd);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Get the next safe promotion window
    /// </summary>
    public async Task<DateTime?> GetNextSafeWindowAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        var etNow = GetEasternTime();
        var currentDate = etNow.Date;
        var timeOfDay = etNow.TimeOfDay;

        // Check if we're already in a safe window
        if (await IsInSafePromotionWindowAsync(cancellationToken))
        {
            return etNow; // Already in safe window
        }

        // Try to find next safe window today
        var nextWindows = new[]
        {
            // Post-close window today (if not passed)
            timeOfDay < PostCloseWindowStart ? 
                currentDate.Add(PostCloseWindowStart) : (DateTime?)null,
            
            // Overnight window tomorrow morning
            currentDate.AddDays(1).Add(OvernightWindowStart),
            
            // Post-close window tomorrow
            currentDate.AddDays(1).Add(PostCloseWindowStart)
        };

        foreach (var window in nextWindows)
        {
            if (window.HasValue)
            {
                var windowDate = window.Value;
                
                // Skip if it's a weekend for overnight/post-close windows
                if (windowDate.DayOfWeek != DayOfWeek.Saturday && 
                    windowDate.DayOfWeek != DayOfWeek.Sunday)
                {
                    return ConvertFromEasternTime(windowDate);
                }
            }
        }

        // If all else fails, return next Saturday (weekend)
        var nextSaturday = GetNextWeekend(etNow);
        return ConvertFromEasternTime(nextSaturday);
    }

    /// <summary>
    /// Check if market is currently open for trading
    /// </summary>
    public async Task<bool> IsMarketOpenAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        var etNow = GetEasternTime();
        var dayOfWeek = etNow.DayOfWeek;
        var timeOfDay = etNow.TimeOfDay;

        // Market closed on weekends
        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return false;
        }

        // CME futures trade nearly 24/5, with a brief closure from 5:00-6:00 PM ET
        var isInDailyCloseWindow = timeOfDay >= MarketCloseTime && timeOfDay < MarketOpenTime;
        
        return !isInDailyCloseWindow;
    }

    /// <summary>
    /// Get current market session
    /// </summary>
    public async Task<string> GetCurrentMarketSessionAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        var etNow = GetEasternTime();
        var dayOfWeek = etNow.DayOfWeek;
        var timeOfDay = etNow.TimeOfDay;

        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return "WEEKEND";
        }

        if (timeOfDay >= MarketCloseTime && timeOfDay < MarketOpenTime)
        {
            return "CLOSED";
        }

        // More granular sessions during trading hours
        return timeOfDay switch
        {
            var t when t >= new TimeSpan(6, 0, 0) && t < new TimeSpan(9, 30, 0) => "PRE_MARKET",
            var t when t >= new TimeSpan(9, 30, 0) && t < new TimeSpan(16, 0, 0) => "OPEN",
            var t when t >= new TimeSpan(16, 0, 0) && t < new TimeSpan(17, 0, 0) => "CLOSE",
            _ => "OVERNIGHT"
        };
    }

    /// <summary>
    /// Check if it's a trading day (not weekend/holiday)
    /// </summary>
    public async Task<bool> IsTradingDayAsync(DateTime date, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        var dayOfWeek = date.DayOfWeek;
        
        // Basic check - not weekend
        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return false;
        }

        // TODO: Add holiday calendar checking
        // For now, assume all weekdays are trading days
        return true;
    }

    #region Private Methods

    private DateTime GetEasternTime()
    {
        // Convert UTC to Eastern Time
        // Note: This is a simplified implementation
        // In production, use TimeZoneInfo for proper DST handling
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, easternZone);
        }
        catch
        {
            // Fallback to UTC-5 (EST) if timezone not found
            return DateTime.UtcNow.AddHours(-5);
        }
    }

    private DateTime ConvertFromEasternTime(DateTime easternTime)
    {
        // Convert Eastern Time back to UTC
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeToUtc(easternTime, easternZone);
        }
        catch
        {
            // Fallback to UTC+5 if timezone not found
            return easternTime.AddHours(5);
        }
    }

    private DateTime GetNextWeekend(DateTime currentEt)
    {
        var daysUntilSaturday = ((int)DayOfWeek.Saturday - (int)currentEt.DayOfWeek + 7) % 7;
        if (daysUntilSaturday == 0 && currentEt.DayOfWeek != DayOfWeek.Saturday)
        {
            daysUntilSaturday = 7;
        }
        
        return currentEt.Date.AddDays(daysUntilSaturday);
    }

    #endregion
}