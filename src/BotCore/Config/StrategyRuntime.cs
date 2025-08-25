using System.Globalization;
using System.Text.RegularExpressions;

namespace BotCore.Config;

// Deprecated: Use BotCore.Models.MarketSnapshot instead
// public sealed class MarketSnapshot { /* removed */ }

public static class TimeWindows
{
    private static readonly TimeZoneInfo Et = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
    private static readonly Regex Range = new(@"^(?<s>\d{2}:\d{2})-(?<e>\d{2}:\d{2})$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    public static bool IsNowWithinEt(string windowEt, DateTime utcNow)
    {
        var m = Range.Match(windowEt);
        if (!m.Success) return true;
        var start = TimeSpan.ParseExact(m.Groups["s"].Value, @"hh\:mm", CultureInfo.InvariantCulture);
        var end = TimeSpan.ParseExact(m.Groups["e"].Value, @"hh\:mm", CultureInfo.InvariantCulture);
        var local = TimeZoneInfo.ConvertTimeFromUtc(utcNow, Et).TimeOfDay;
        return start <= end ? (local >= start && local <= end)
                            : (local >= start || local <= end);
    }
}
