using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Market hours service with ET maintenance, Sunday curb, and holiday enforcement
/// </summary>
internal sealed class MarketHoursService : IMarketHoursService
{
    private readonly ILogger<MarketHoursService> _logger;
    private readonly IConfiguration _configuration;
    private readonly HashSet<DateTime> _holidays = new();
    
    public MarketHoursService(ILogger<MarketHoursService> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        LoadHolidays();
    }
    
    private void LoadHolidays()
    {
        try
        {
            var holidayPath = _configuration.GetValue("Paths:HolidayCme", "config/calendar/holiday-cme.json");
            if (File.Exists(holidayPath))
            {
                var lines = File.ReadAllLines(holidayPath);
                foreach (var line in lines)
                {
                    if (DateTime.TryParse(line.Trim(), out var holiday))
                    {
                        _holidays.Add(holiday.Date);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to load holiday calendar");
        }
    }
    
    public async Task<bool> IsInSafePromotionWindowAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        var et = GetEasternTime();
        
        // Safe promotion during maintenance break (5:00-6:00 PM ET)
        if (et.TimeOfDay >= new TimeSpan(17, 0, 0) && et.TimeOfDay < new TimeSpan(18, 0, 0))
            return true;
            
        // Safe promotion on weekends
        if (et.DayOfWeek == DayOfWeek.Saturday || et.DayOfWeek == DayOfWeek.Sunday)
            return true;
            
        return false;
    }
    
    public async Task<DateTime?> GetNextSafeWindowAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        var et = GetEasternTime();
        
        // Next maintenance break
        var nextMaintenanceStart = et.Date.AddHours(17);
        if (et.TimeOfDay >= new TimeSpan(17, 0, 0))
        {
            nextMaintenanceStart = nextMaintenanceStart.AddDays(1);
        }
        
        return nextMaintenanceStart;
    }
    
    public async Task<bool> IsMarketOpenAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        return IsMarketOpen(DateTime.UtcNow);
    }
    
    private bool IsMarketOpen(DateTime utc)
    {
        var et = GetEasternTimeFromUtc(utc);
        
        // Check for holidays
        if (_holidays.Contains(et.Date))
            return false;
            
        // Check for maintenance break (5:00-6:00 PM ET)
        if (et.TimeOfDay >= new TimeSpan(17, 0, 0) && et.TimeOfDay < new TimeSpan(18, 0, 0))
            return false;
        
        // Sunday before 6:00 PM ET is closed
        if (et.DayOfWeek == DayOfWeek.Sunday && et.TimeOfDay < new TimeSpan(18, 0, 0))
            return false;
        
        // Saturday is closed
        if (et.DayOfWeek == DayOfWeek.Saturday)
            return false;
            
        return true;
    }
    
    public async Task<string> GetCurrentMarketSessionAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        var et = GetEasternTime();
        var timeOfDay = et.TimeOfDay;
        
        return timeOfDay switch
        {
            var t when t >= new TimeSpan(17, 0, 0) && t < new TimeSpan(18, 0, 0) => "MAINTENANCE",
            var t when t >= new TimeSpan(18, 0, 0) || t < new TimeSpan(9, 30, 0) => "OVERNIGHT",
            var t when t >= new TimeSpan(9, 30, 0) && t < new TimeSpan(16, 0, 0) => "RTH",
            var t when t >= new TimeSpan(16, 0, 0) && t < new TimeSpan(17, 0, 0) => "POST_MARKET",
            _ => "CLOSED"
        };
    }
    
    public async Task<bool> IsTradingDayAsync(DateTime date, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        // Check for holidays
        if (_holidays.Contains(date.Date))
            return false;
            
        // Weekends are not trading days
        if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday)
            return false;
            
        return true;
    }
    
    public async Task<TrainingIntensity> GetRecommendedTrainingIntensityAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        if (await IsMarketOpenAsync(cancellationToken).ConfigureAwait(false))
        {
            return TrainingIntensity.Light; // Light learning during market hours
        }
        
        return TrainingIntensity.Intensive; // Intensive learning when market closed
    }
    
    private DateTime GetEasternTime()
    {
        return GetEasternTimeFromUtc(DateTime.UtcNow);
    }
    
    private DateTime GetEasternTimeFromUtc(DateTime utc)
    {
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeFromUtc(utc, easternZone);
        }
        catch
        {
            return utc.AddHours(-5); // Fallback to EST
        }
    }
}