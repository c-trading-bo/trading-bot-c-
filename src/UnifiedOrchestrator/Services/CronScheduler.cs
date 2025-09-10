using System;
using System.Globalization;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Simple cron expression scheduler for workflow execution timing
/// Supports basic cron format: second minute hour day month day-of-week
/// </summary>
public static class CronScheduler
{
    /// <summary>
    /// Calculate next execution time based on cron expression
    /// Format: second minute hour day month day-of-week
    /// Examples: 
    /// - "*/15 * 9-16 * * MON-FRI" = every 15 seconds during market hours Mon-Fri
    /// - "0 */30 * * * *" = every 30 minutes
    /// </summary>
    public static DateTime? GetNextExecution(string cronExpression, DateTime currentTime)
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

            // Start from the next minute to avoid immediate execution
            var next = new DateTime(currentTime.Year, currentTime.Month, currentTime.Day, currentTime.Hour, currentTime.Minute, 0, DateTimeKind.Utc).AddMinutes(1);

            // Simple implementation - find next valid time within 24 hours
            for (int i = 0; i < 1440; i++) // 24 hours * 60 minutes
            {
                if (MatchesCronExpression(next, second, minute, hour, day, month, dayOfWeek))
                {
                    return next;
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
        if (field == "*") return true;

        // Handle */X format
        if (field.StartsWith("*/"))
        {
            if (int.TryParse(field.Substring(2), out int interval))
            {
                return value % interval == 0;
            }
        }

        // Handle ranges like 9-16
        if (field.Contains("-"))
        {
            var parts = field.Split('-');
            if (parts.Length == 2 && int.TryParse(parts[0], out int start) && int.TryParse(parts[1], out int end))
            {
                return value >= start && value <= end;
            }
        }

        // Handle comma-separated values
        if (field.Contains(","))
        {
            var values = field.Split(',');
            foreach (var v in values)
            {
                if (int.TryParse(v.Trim(), out int val) && val == value)
                    return true;
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
        if (field == "*") return true;

        // Handle day names
        if (field.Contains("MON-FRI"))
        {
            return dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Friday;
        }

        if (field.Contains("SAT-SUN"))
        {
            return dayOfWeek == DayOfWeek.Saturday || dayOfWeek == DayOfWeek.Sunday;
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
        var dateStr = date.ToString("yyyy-MM-dd");
        return holidays.Contains(dateStr);
    }
}