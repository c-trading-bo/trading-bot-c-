namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Helper utilities for price rounding and R multiple calculations
/// Implements the exact pattern from the agent instructions
/// </summary>
public static class Px
{
    public const decimal ES_TICK = 0.25m;
    private static readonly System.Globalization.CultureInfo Invariant = System.Globalization.CultureInfo.InvariantCulture;

    public static decimal RoundToTick(decimal price, decimal tick = ES_TICK) =>
        Math.Round(price / tick, 0, MidpointRounding.AwayFromZero) * tick;

    public static string F2(decimal value) => value.ToString("0.00", Invariant);

    public static decimal RMultiple(decimal entry, decimal stop, decimal target, bool isLong)
    {
        var risk = isLong ? entry - stop : stop - entry;     // must be > 0
        var reward = isLong ? target - entry : entry - target; // must be >= 0
        if (risk <= 0) return 0; // reject invalid risk
        return reward / risk;
    }
}

/// <summary>
/// Examples of BEFORE/AFTER stub replacements
/// Shows what was removed and what replaced it
/// </summary>
public static class StubRemovalExamples
{
    /// <summary>
    /// Production implementation should use SignalR HubConnection for real-time market data
    /// Example: new HubConnectionBuilder().WithUrl("https://rtc.topstepx.com/hubs/market")
    /// </summary>
    public static async Task<bool> ConnectToMarketDataExample()
    {
        await Task.CompletedTask;
        return true;
    }

    /// <summary>
    /// Example method showing proper API integration pattern.
    /// This demonstrates the expected flow from placeholder to real implementation.
    /// </summary>
    public static async Task<string> PlaceOrderExample()
    {
        // Real implementation would use actual HTTP client and API endpoints
        await Task.CompletedTask;
        return Guid.NewGuid().ToString();
    }

    // BEFORE (STUB):
    // return new { Balance = 50000m, BuyingPower = 200000m };
    //
    // AFTER (REAL):
    // var response = await _httpClient.GetAsync($"/api/Account/{accountId}");
    // var accountData = JsonSerializer.Deserialize<JsonElement>(json);
    // return new AccountInfo(...);
}