using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Scheduling;

/// <summary>
/// Futures market hours service with CME session behavior
/// Implements safe promotion windows and 24/7 operation scheduling to avoid disrupting live trading
/// Supports intensive training during market downtime and light background training during trading hours
/// </summary>
public class FuturesMarketHours : IMarketHoursService
{
    private readonly ILogger<FuturesMarketHours> _logger;
    
    // CME E-mini S&P 500 (ES) Futures trading hours (Eastern Time)
    // Nearly 24/5 trading: Sunday 6:00 PM ET to Friday 5:00 PM ET
    // Daily maintenance break: 5:00 PM - 6:00 PM ET Monday-Thursday
    private readonly TimeSpan MarketOpenTime = new(18, 0, 0);  // 6:00 PM ET (Sunday-Thursday)
    private readonly TimeSpan MarketCloseTime = new(17, 0, 0); // 5:00 PM ET (Monday-Friday)
    
    // Safe promotion windows (Eastern Time) - Low volume/volatility periods
    private readonly TimeSpan EarlyMorningWindowStart = new(3, 0, 0);   // 3:00 AM ET
    private readonly TimeSpan EarlyMorningWindowEnd = new(4, 0, 0);     // 4:00 AM ET  
    private readonly TimeSpan LunchWindowStart = new(11, 0, 0);         // 11:00 AM ET
    private readonly TimeSpan LunchWindowEnd = new(13, 0, 0);           // 1:00 PM ET
    private readonly TimeSpan MaintenanceWindowStart = new(17, 0, 0);   // 5:00 PM ET (daily break)
    private readonly TimeSpan MaintenanceWindowEnd = new(18, 0, 0);     // 6:00 PM ET

    // 24/7 Training schedule windows
    private readonly TimeSpan IntensiveTrainingStart = new(21, 0, 0);   // 9:00 PM ET (Friday evening)
    private readonly TimeSpan IntensiveTrainingEnd = new(17, 0, 0);     // 5:00 PM ET (Sunday)
    private readonly TimeSpan BackgroundTrainingStart = new(1, 0, 0);   // 1:00 AM ET (overnight)
    private readonly TimeSpan BackgroundTrainingEnd = new(5, 0, 0);     // 5:00 AM ET

    public FuturesMarketHours(ILogger<FuturesMarketHours> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Check if current time is within a safe promotion window
    /// Safe windows: early morning 3:00-4:00 AM ET, lunch 11:00 AM-1:00 PM ET, daily maintenance 5:00-6:00 PM ET, weekends
    /// These periods typically have lower volatility and volume
    /// </summary>
    public async Task<bool> IsInSafePromotionWindowAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var etNow = GetEasternTime();
        var dayOfWeek = etNow.DayOfWeek;
        var timeOfDay = etNow.TimeOfDay;

        // Weekend is always safe (Friday after 5 PM to Sunday before 6 PM)
        if (dayOfWeek == DayOfWeek.Saturday || 
            (dayOfWeek == DayOfWeek.Sunday && timeOfDay < MarketOpenTime) ||
            (dayOfWeek == DayOfWeek.Friday && timeOfDay >= MarketCloseTime))
        {
            _logger.LogDebug("In safe promotion window: Weekend/Market Closed");
            return true;
        }

        // Early morning low-volume window: 3:00-4:00 AM ET
        if (timeOfDay >= EarlyMorningWindowStart && timeOfDay <= EarlyMorningWindowEnd)
        {
            _logger.LogDebug("In safe promotion window: Early morning ({Start}-{End} ET)", 
                EarlyMorningWindowStart, EarlyMorningWindowEnd);
            return true;
        }

        // Lunch time low-volatility window: 11:00 AM-1:00 PM ET
        if (timeOfDay >= LunchWindowStart && timeOfDay <= LunchWindowEnd)
        {
            _logger.LogDebug("In safe promotion window: Lunch period ({Start}-{End} ET)", 
                LunchWindowStart, LunchWindowEnd);
            return true;
        }

        // Daily maintenance window: 5:00-6:00 PM ET (when market is closed for maintenance)
        if (timeOfDay >= MaintenanceWindowStart && timeOfDay <= MaintenanceWindowEnd)
        {
            _logger.LogDebug("In safe promotion window: Daily maintenance ({Start}-{End} ET)", 
                MaintenanceWindowStart, MaintenanceWindowEnd);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Check if current time is suitable for intensive training (weekends, market downtime)
    /// </summary>
    public async Task<bool> IsInIntensiveTrainingWindowAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var etNow = GetEasternTime();
        var dayOfWeek = etNow.DayOfWeek;
        var timeOfDay = etNow.TimeOfDay;

        // Weekend intensive training: Friday 9:00 PM to Sunday 5:00 PM ET
        if (dayOfWeek == DayOfWeek.Saturday)
        {
            return true; // Full Saturday available
        }
        
        if (dayOfWeek == DayOfWeek.Sunday && timeOfDay < MarketCloseTime)
        {
            return true; // Sunday until market opens
        }
        
        if (dayOfWeek == DayOfWeek.Friday && timeOfDay >= IntensiveTrainingStart)
        {
            return true; // Friday evening after markets calm down
        }

        // Daily maintenance window for shorter intensive training sessions
        if (timeOfDay >= MaintenanceWindowStart && timeOfDay <= MaintenanceWindowEnd)
        {
            return true;
        }

        return false;
    }

    /// <summary>
    /// Check if current time is suitable for background training (low-priority, resource-limited)
    /// </summary>
    public async Task<bool> IsInBackgroundTrainingWindowAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var etNow = GetEasternTime();
        var dayOfWeek = etNow.DayOfWeek;
        var timeOfDay = etNow.TimeOfDay;

        // Don't run background training during weekends (use for intensive training instead)
        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return false;
        }

        // Overnight background training: 1:00-5:00 AM ET on weekdays
        if (timeOfDay >= BackgroundTrainingStart && timeOfDay <= BackgroundTrainingEnd)
        {
            return true;
        }

        // Light training during low-volatility periods (but not promotion windows)
        if (timeOfDay >= LunchWindowStart && timeOfDay <= LunchWindowEnd)
        {
            return true;
        }

        return false;
    }

    /// <summary>
    /// Get recommended training intensity based on current market conditions
    /// </summary>
    public async Task<TrainingIntensity> GetRecommendedTrainingIntensityAsync(CancellationToken cancellationToken = default)
    {
        var isIntensive = await IsInIntensiveTrainingWindowAsync(cancellationToken).ConfigureAwait(false);
        var isBackground = await IsInBackgroundTrainingWindowAsync(cancellationToken).ConfigureAwait(false);
        var isMarketOpen = await IsMarketOpenAsync(cancellationToken).ConfigureAwait(false);

        if (isIntensive)
        {
            return new TrainingIntensity
            {
                Level = TrainingIntensityLevel.Intensive,
                ParallelJobs = 4
            };
        }
        
        if (isBackground)
        {
            return new TrainingIntensity
            {
                Level = TrainingIntensityLevel.Light,
                ParallelJobs = 1
            };
        }
        
        if (!isMarketOpen)
        {
            return new TrainingIntensity
            {
                Level = TrainingIntensityLevel.High,
                ParallelJobs = 2
            };
        }

        return new TrainingIntensity
        {
            Level = TrainingIntensityLevel.Light,
            ParallelJobs = 0
        };
    }

    /// <summary>
    /// Get the next scheduled training window
    /// </summary>
    public async Task<DateTime?> GetNextTrainingWindowAsync(string trainingType = "INTENSIVE", CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var etNow = GetEasternTime();
        
        return trainingType.ToUpperInvariant() switch
        {
            "INTENSIVE" => await GetNextIntensiveTrainingWindowAsync(etNow, cancellationToken),
            "BACKGROUND" => await GetNextBackgroundTrainingWindowAsync(etNow, cancellationToken),
            _ => await GetNextSafeWindowAsync(cancellationToken)
        }.ConfigureAwait(false);
    }

    /// <summary>
    /// Get the next safe promotion window
    /// </summary>
    public async Task<DateTime?> GetNextSafeWindowAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var etNow = GetEasternTime();
        var currentDate = etNow.Date;
        var timeOfDay = etNow.TimeOfDay;

        // Check if we're already in a safe window
        if (await IsInSafePromotionWindowAsync(cancellationToken))
        {
            return etNow.ConfigureAwait(false); // Already in safe window
        }

        // Try to find next safe window today
        var nextWindows = new[]
        {
            // Lunch window today (if not passed)
            timeOfDay < LunchWindowStart ? 
                currentDate.Add(LunchWindowStart) : (DateTime?)null,
            
            // Daily maintenance window today (if not passed)  
            timeOfDay < MaintenanceWindowStart ?
                currentDate.Add(MaintenanceWindowStart) : (DateTime?)null,
            
            // Early morning window tomorrow
            currentDate.AddDays(1).Add(EarlyMorningWindowStart),
            
            // Lunch window tomorrow
            currentDate.AddDays(1).Add(LunchWindowStart)
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
    /// Check if market is currently open for trading (synchronous version)
    /// </summary>
    public bool IsMarketOpen(DateTime? time = null, string symbol = "ES")
    {
        var checkTime = time ?? DateTime.UtcNow;
        var etTime = GetEasternTimeFromUtc(checkTime);
        var dayOfWeek = etTime.DayOfWeek;
        var timeOfDay = etTime.TimeOfDay;

        // Market closed on weekends
        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return false;
        }

        // CME futures trade nearly 24/5, with a brief closure from 5:00-6:00 PM ET
        var isInDailyCloseWindow = timeOfDay >= MarketCloseTime && timeOfDay < MarketOpenTime;
        
        return !isInDailyCloseWindow;
    }

    private DateTime GetEasternTimeFromUtc(DateTime utcTime)
    {
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeFromUtc(utcTime, easternZone);
        }
        catch
        {
            // Fallback to UTC-5 (EST) if timezone not found
            return utcTime.AddHours(-5);
        }
    }

    /// <summary>
    /// Check if market is currently open for trading
    /// </summary>
    public async Task<bool> IsMarketOpenAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
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
        await Task.CompletedTask.ConfigureAwait(false);
        
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
            var t when t >= new TimeSpan(9, 30, 0) && t < new TimeSpan(11, 0, 0) => "MORNING_SESSION", 
            var t when t >= new TimeSpan(11, 0, 0) && t < new TimeSpan(13, 0, 0) => "LUNCH_SESSION",
            var t when t >= new TimeSpan(13, 0, 0) && t < new TimeSpan(16, 0, 0) => "AFTERNOON_SESSION",
            var t when t >= new TimeSpan(16, 0, 0) && t < new TimeSpan(17, 0, 0) => "CLOSE_SESSION",
            _ => "OVERNIGHT"
        };
    }

    /// <summary>
    /// Check if it's a trading day (not weekend/holiday) with comprehensive holiday calendar support
    /// Includes major US holidays that affect CME futures trading
    /// </summary>
    public async Task<bool> IsTradingDayAsync(DateTime date, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var dayOfWeek = date.DayOfWeek;
        
        // Basic check - not weekend
        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return false;
        }

        // Check against comprehensive holiday calendar
        return !IsHoliday(date);
    }
    
    /// <summary>
    /// Comprehensive US holiday calendar for CME futures trading
    /// Includes all major holidays that close futures markets
    /// </summary>
    private bool IsHoliday(DateTime date)
    {
        var year = date.Year;
        
        // Fixed holidays
        if (IsFixedHoliday(date)) return true;
        
        // Variable holidays (calculated based on date rules)
        var holidays = new[]
        {
            GetMartinLutherKingDay(year),      // 3rd Monday in January
            GetPresidentsDay(year),            // 3rd Monday in February  
            GetMemorialDay(year),              // Last Monday in May
            GetLaborDay(year),                 // 1st Monday in September
            GetColumbusDay(year),              // 2nd Monday in October (observed by some exchanges)
            GetThanksgiving(year),             // 4th Thursday in November
            GetThanksgivingFriday(year)        // Friday after Thanksgiving
        };
        
        return holidays.Any(holiday => holiday.Date == date.Date);
    }
    
    private bool IsFixedHoliday(DateTime date)
    {
        var month = date.Month;
        var day = date.Day;
        
        return (month, day) switch
        {
            (1, 1) => true,     // New Year's Day
            (7, 4) => true,     // Independence Day
            (11, 11) => true,   // Veterans Day (observed by some markets)
            (12, 25) => true,   // Christmas Day
            (12, 24) when date.DayOfWeek == DayOfWeek.Friday => true, // Christmas Eve (early close)
            (12, 31) when date.DayOfWeek == DayOfWeek.Friday => true, // New Year's Eve (early close)
            _ => false
        };
    }
    
    private DateTime GetMartinLutherKingDay(int year)
    {
        // 3rd Monday in January
        var firstMonday = GetFirstMondayOfMonth(year, 1);
        return firstMonday.AddDays(14);
    }
    
    private DateTime GetPresidentsDay(int year)
    {
        // 3rd Monday in February
        var firstMonday = GetFirstMondayOfMonth(year, 2);
        return firstMonday.AddDays(14);
    }
    
    private DateTime GetMemorialDay(int year)
    {
        // Last Monday in May
        var lastDayOfMay = new DateTime(year, 5, 31);
        while (lastDayOfMay.DayOfWeek != DayOfWeek.Monday)
        {
            lastDayOfMay = lastDayOfMay.AddDays(-1);
        }
        return lastDayOfMay;
    }
    
    private DateTime GetLaborDay(int year)
    {
        // 1st Monday in September
        return GetFirstMondayOfMonth(year, 9);
    }
    
    private DateTime GetColumbusDay(int year)
    {
        // 2nd Monday in October
        var firstMonday = GetFirstMondayOfMonth(year, 10);
        return firstMonday.AddDays(7);
    }
    
    private DateTime GetThanksgiving(int year)
    {
        // 4th Thursday in November
        var firstThursday = GetFirstThursdayOfMonth(year, 11);
        return firstThursday.AddDays(21);
    }
    
    private DateTime GetThanksgivingFriday(int year)
    {
        // Friday after Thanksgiving
        return GetThanksgiving(year).AddDays(1);
    }
    
    private DateTime GetFirstMondayOfMonth(int year, int month)
    {
        var firstDayOfMonth = new DateTime(year, month, 1);
        while (firstDayOfMonth.DayOfWeek != DayOfWeek.Monday)
        {
            firstDayOfMonth = firstDayOfMonth.AddDays(1);
        }
        return firstDayOfMonth;
    }
    
    private DateTime GetFirstThursdayOfMonth(int year, int month)
    {
        var firstDayOfMonth = new DateTime(year, month, 1);
        while (firstDayOfMonth.DayOfWeek != DayOfWeek.Thursday)
        {
            firstDayOfMonth = firstDayOfMonth.AddDays(1);
        }
        return firstDayOfMonth;
    }
    
    /// <summary>
    /// Get a list of all holidays for a given year for reporting/validation
    /// </summary>
    public List<(DateTime Date, string Name)> GetHolidaysForYear(int year)
    {
        return new List<(DateTime, string)>
        {
            (new DateTime(year, 1, 1), "New Year's Day"),
            (GetMartinLutherKingDay(year), "Martin Luther King Jr. Day"),
            (GetPresidentsDay(year), "Presidents Day"),
            (GetMemorialDay(year), "Memorial Day"),
            (new DateTime(year, 7, 4), "Independence Day"),
            (GetLaborDay(year), "Labor Day"),
            (GetColumbusDay(year), "Columbus Day"),
            (new DateTime(year, 11, 11), "Veterans Day"),
            (GetThanksgiving(year), "Thanksgiving"),
            (GetThanksgivingFriday(year), "Thanksgiving Friday"),
            (new DateTime(year, 12, 25), "Christmas Day")
        };
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

    private async Task<DateTime?> GetNextIntensiveTrainingWindowAsync(DateTime etNow, CancellationToken cancellationToken)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var currentDate = etNow.Date;
        var timeOfDay = etNow.TimeOfDay;

        // If currently Friday evening, next window is now
        if (etNow.DayOfWeek == DayOfWeek.Friday && timeOfDay >= IntensiveTrainingStart)
        {
            return ConvertFromEasternTime(etNow);
        }

        // If weekend, next window is now
        if (etNow.DayOfWeek == DayOfWeek.Saturday || 
            (etNow.DayOfWeek == DayOfWeek.Sunday && timeOfDay < MarketCloseTime))
        {
            return ConvertFromEasternTime(etNow);
        }

        // Otherwise, next Friday evening
        var daysUntilFriday = ((int)DayOfWeek.Friday - (int)etNow.DayOfWeek + 7) % 7;
        if (daysUntilFriday == 0) daysUntilFriday = 7; // Next Friday, not today

        var nextFriday = currentDate.AddDays(daysUntilFriday).Add(IntensiveTrainingStart);
        return ConvertFromEasternTime(nextFriday);
    }

    private async Task<DateTime?> GetNextBackgroundTrainingWindowAsync(DateTime etNow, CancellationToken cancellationToken)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var currentDate = etNow.Date;
        var timeOfDay = etNow.TimeOfDay;

        // If currently in background window, return now
        if (await IsInBackgroundTrainingWindowAsync(cancellationToken))
        {
            return ConvertFromEasternTime(etNow).ConfigureAwait(false);
        }

        // Check if background window is later today
        if (timeOfDay < BackgroundTrainingStart)
        {
            var todayBackgroundStart = currentDate.Add(BackgroundTrainingStart);
            return ConvertFromEasternTime(todayBackgroundStart);
        }

        // Check if lunch window is later today
        if (timeOfDay < LunchWindowStart)
        {
            var todayLunchStart = currentDate.Add(LunchWindowStart);
            return ConvertFromEasternTime(todayLunchStart);
        }

        // Otherwise, tomorrow's background window
        var tomorrowBackgroundStart = currentDate.AddDays(1).Add(BackgroundTrainingStart);
        return ConvertFromEasternTime(tomorrowBackgroundStart);
    }

    #endregion
}