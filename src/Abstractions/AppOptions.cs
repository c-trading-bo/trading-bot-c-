namespace TradingBot.Abstractions;

public sealed class AppOptions
{
    public string ApiBase { get; init; } = "https://api.topstepx.com";
    public string AuthToken { get; init; } = "";
    public string AccountId { get; init; } = "";
    public bool EnableDryRunMode { get; init; } = true;
    public bool EnableAutoExecution { get; init; } = false;
    public decimal MaxDailyLoss { get; init; } = -1000m;
    public int MaxPositionSize { get; init; } = 5;
    public decimal DrawdownLimit { get; init; } = -2000m;
    public string KillFile { get; init; } = "kill.txt";
}