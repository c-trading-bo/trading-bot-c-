using System.Text.Json.Serialization;

namespace TopstepX;

public sealed class TopstepProjectXOptions
{
    // REAL TopstepX endpoints (not demo)
    public string ApiBaseUrl { get; init; } = "https://api.topstepx.com";
    public string UserHubUrl { get; init; } = "https://rtc.topstepx.com/hubs/user";
    public string MarketHubUrl { get; init; } = "https://rtc.topstepx.com/hubs/market";

    public string UserName { get; init; } = default!;
    public string ApiKey { get; init; } = default!;

    public string[] DefaultSymbols { get; init; } = new[] { "F.US.EP", "F.US.ENQ" }; // ES, NQ
    public bool IsEvaluationAccount { get; init; } = true; // eval/Express Funded → SIM data
}
