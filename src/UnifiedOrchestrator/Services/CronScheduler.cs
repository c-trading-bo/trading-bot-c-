using System;
using System.Collections.Generic;
using System.Globalization;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Cron expression scheduler for workflow execution timing with CME futures session support
/// Supports basic cron format: second minute hour day month day-of-week
/// </summary>
internal static class CronScheduler
{
    /// <summary>
    /// Calculate next execution time based on cron expression with time zone support
    /// Format: second minute hour day month day-of-week
    /// Examples: 
    /// - "*/15 * 18-23,0-16 * * SUN-FRI" = every 15 seconds during CME futures hours
    /// - "0 0 18 ? * SUN" = Sunday 6:00 PM ET (session open)
    /// - "0 0 17 ? * FRI" = Friday 5:00 PM ET (session close)
    /// </summary>
    public static DateTime? GetNextExecution(string cronExpression, DateTime currentTime, string timeZone = "America/New_York")
    {
        if (string.IsNullOrWhiteSpace(cronExpression))
            return null;

        try
        {
            var parts = cronExpression.Split(' ');
            if (parts.Length != 6)
            {
                // Fallback to simple interval parsing
                return ParseSimpleInterval(cronExpression, currentTime);
            }

            var (second, minute, hour, day, month, dayOfWeek) = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]);

            // Convert current time to target timezone
            var targetTimeZoneInfo = TimeZoneInfo.FindSystemTimeZoneById(timeZone);
            var localTime = TimeZoneInfo.ConvertTimeFromUtc(currentTime, targetTimeZoneInfo);
            
            // Start from the next minute to avoid immediate execution
            var next = new DateTime(localTime.Year, localTime.Month, localTime.Day, localTime.Hour, localTime.Minute, 0, DateTimeKind.Unspecified).AddMinutes(1);

            // Find next valid time within the next 7 days (to handle weekly cycles)
            for (int i; i < 10080; i++) // 7 days * 24 hours * 60 minutes
            {
                if (MatchesCronExpression(next, second, minute, hour, day, month, dayOfWeek))
                {
                    // Convert back to UTC for return
                    return TimeZoneInfo.ConvertTimeToUtc(next, targetTimeZoneInfo);
                }
                next = next.AddMinutes(1);
            }

            return currentTime.AddHours(1); // Fallback
        }
        catch
        {
            return ParseSimpleInterval(cronExpression, currentTime);
        }
    }

    /// <summary>
    /// Check if the given time is during CME daily maintenance break (5-6 PM ET Monday-Thursday)
    /// </summary>
    public static bool IsMaintenanceBreak(DateTime utcTime, string timeZone = "America/New_York")
    {
        try
        {
            var targetTimeZoneInfo = TimeZoneInfo.FindSystemTimeZoneById(timeZone);
            var et = TimeZoneInfo.ConvertTimeFromUtc(utcTime, targetTimeZoneInfo);
            
            // Monday-Thursday 5-6 PM ET
            if (et.DayOfWeek >= DayOfWeek.Monday && et.DayOfWeek <= DayOfWeek.Thursday)
            {
                return et.Hour == 17; // 5 PM ET (17:00-17:59)
            }
            
            return false;
        }
        catch
        {
            return false;
        }
    }

    private static DateTime ParseSimpleInterval(string expression, DateTime currentTime)
    {
        // Handle simple formats like "every 15 minutes", "hourly", etc.
        if (expression.Contains("*/"))
        {
            var match = System.Text.RegularExpressions.Regex.Match(expression, @"\*/(\d+)");
            if (match.Success && int.TryParse(match.Groups[1].Value, out int interval))
            {
                if (expression.Contains("minute"))
                    return currentTime.AddMinutes(interval);
                if (expression.Contains("hour"))
                    return currentTime.AddHours(interval);
                if (expression.Contains("second"))
                    return currentTime.AddSeconds(interval);
                    
                // Default to minutes for */X format
                return currentTime.AddMinutes(interval);
            }
        }

        // Default fallback
        return currentTime.AddHours(1);
    }

    private static bool MatchesCronExpression(DateTime dateTime, string second, string minute, string hour, string day, string month, string dayOfWeek)
    {
        return MatchesField(dateTime.Second, second) &&
               MatchesField(dateTime.Minute, minute) &&
               MatchesField(dateTime.Hour, hour) &&
               MatchesField(dateTime.Day, day) &&
               MatchesField(dateTime.Month, month) &&
               MatchesDayOfWeek(dateTime.DayOfWeek, dayOfWeek);
    }

    private static bool MatchesField(int value, string field)
    {
        if (field == "*" || field == "?") return true;

        // Handle */X format
        if (field.StartsWith("*/"))
        {
            if (int.TryParse(field.Substring(2), out int interval))
            {
                return value % interval == 0;
            }
        }

        // Handle ranges like 9-16 or 18-23,0-16 (multiple ranges)
        if (field.Contains("-"))
        {
            var ranges = field.Split(',');
            foreach (var range in ranges)
            {
                var parts = range.Trim().Split('-');
                if (parts.Length == 2 && int.TryParse(parts[0], out int start) && int.TryParse(parts[1], out int end))
                {
                    if (value >= start && value <= end)
                        return true;
                }
            }
            return false;
        }

        // Handle comma-separated values
        if (field.Contains(","))
        {
            var values = field.Split(',');
            foreach (var v in values)
            {
                var trimmed = v.Trim();
                // Check if it's a range
                if (trimmed.Contains("-"))
                {
                    var parts = trimmed.Split('-');
                    if (parts.Length == 2 && int.TryParse(parts[0], out int start) && int.TryParse(parts[1], out int end))
                    {
                        if (value >= start && value <= end)
                            return true;
                    }
                }
                else if (int.TryParse(trimmed, out int val) && val == value)
                {
                    return true;
                }
            }
            return false;
        }

        // Single value
        if (int.TryParse(field, out int singleValue))
        {
            return value == singleValue;
        }

        return false;
    }

    private static bool MatchesDayOfWeek(DayOfWeek dayOfWeek, string field)
    {
        if (field == "*" || field == "?") return true;

        // Handle day names
        if (field.Contains("SUN-FRI"))
        {
            return dayOfWeek >= DayOfWeek.Sunday && dayOfWeek <= DayOfWeek.Friday;
        }
        
        if (field.Contains("MON-FRI"))
        {
            return dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Friday;
        }
        
        if (field.Contains("MON-THU"))
        {
            return dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Thursday;
        }

        if (field.Contains("SAT-SUN"))
        {
            return dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday;
        }

        // Handle individual day names
        var dayNames = new Dictionary<string, DayOfWeek>
        {
            {"SUN", DayOfWeek.Sunday}, {"MON", DayOfWeek.Monday}, {"TUE", DayOfWeek.Tuesday},
            {"WED", DayOfWeek.Wednesday}, {"THU", DayOfWeek.Thursday}, {"FRI", DayOfWeek.Friday}, {"SAT", DayOfWeek.Saturday}
        };

        if (field.Contains(","))
        {
            var days = field.Split(',');
            foreach (var day in days)
            {
                var trimmed = day.Trim();
                if (dayNames.ContainsKey(trimmed) && dayNames[trimmed] == dayOfWeek)
                    return true;
            }
        }
        else if (dayNames.ContainsKey(field) && dayNames[field] == dayOfWeek)
        {
            return true;
        }

        // Handle numeric day of week (0=Sunday)
        var dayNum = (int)dayOfWeek;
        return MatchesField(dayNum, field);
    }

    /// <summary>
    /// Check if current time is during market holidays
    /// </summary>
    public static bool IsMarketHoliday(DateTime date, List<string> holidays)
    {
        var dateStr = date.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        return holidays.Contains(dateStr);
    }
}