using System.Globalization;
using System.Text.RegularExpressions;

namespace BotCore.Config;

public sealed class MarketSnapshot
{
    public string Symbol { get; init; } = "ES";
    public int SpreadTicks { get; init; }
    public int VolumePct5m { get; init; }
    public decimal AggIAbs { get; init; }
    public decimal SignalBarAtrMult { get; init; }
    public int Adx5m { get; init; }
    public bool Ema9Over21_5m { get; init; }
    public decimal Z5mReturnDiff { get; init; }
    public DateTime UtcNow { get; init; }
}

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
        var end   = TimeSpan.ParseExact(m.Groups["e"].Value, @"hh\:mm", CultureInfo.InvariantCulture);
        var local = TimeZoneInfo.ConvertTimeFromUtc(utcNow, Et).TimeOfDay;
        return start <= end ? (local >= start && local <= end)
                            : (local >= start || local <= end);
    }
}
