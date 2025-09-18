using System;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Services;

/// <summary>
/// Market hours service interface for autonomous trading
/// </summary>
public interface IMarketHours
{
    Task<bool> IsMarketOpenAsync(CancellationToken cancellationToken = default);
    Task<string> GetCurrentMarketSessionAsync(CancellationToken cancellationToken = default);
    bool IsMarketOpen(DateTime? time = null, string symbol = "ES");
}

/// <summary>
/// Basic market hours implementation
/// </summary>
public class BasicMarketHours : IMarketHours
{
    public async Task<bool> IsMarketOpenAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        return IsMarketOpen();
    }
    
    public async Task<string> GetCurrentMarketSessionAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var easternTime = GetEasternTime();
        var timeOfDay = easternTime.TimeOfDay;
        
        return timeOfDay switch
        {
            var t when t >= new TimeSpan(9, 30, 0) && t < new TimeSpan(11, 0, 0) => "MORNING_SESSION",
            var t when t >= new TimeSpan(11, 0, 0) && t < new TimeSpan(13, 0, 0) => "LUNCH_SESSION",
            var t when t >= new TimeSpan(13, 0, 0) && t < new TimeSpan(16, 0, 0) => "AFTERNOON_SESSION",
            var t when t >= new TimeSpan(16, 0, 0) && t < new TimeSpan(17, 0, 0) => "CLOSE_SESSION",
            _ => "OVERNIGHT"
        };
    }
    
    public bool IsMarketOpen(DateTime? time = null, string symbol = "ES")
    {
        var checkTime = time ?? DateTime.UtcNow;
        var easternTime = GetEasternTimeFromUtc(checkTime);
        var dayOfWeek = easternTime.DayOfWeek;
        var timeOfDay = easternTime.TimeOfDay;
        
        // Market closed on weekends
        if (dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday)
        {
            return false;
        }
        
        // Basic trading hours: 9:30 AM - 4:00 PM ET
        return timeOfDay >= new TimeSpan(9, 30, 0) && timeOfDay < new TimeSpan(16, 0, 0);
    }
    
    private DateTime GetEasternTime()
    {
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, easternZone);
        }
        catch
        {
            return DateTime.UtcNow.AddHours(-5); // Fallback to EST
        }
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
            return utcTime.AddHours(-5); // Fallback to EST
        }
    }
}