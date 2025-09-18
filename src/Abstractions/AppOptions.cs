namespace TradingBot.Abstractions;

public sealed class AppOptions
{
    // Default risk management constants
    private const decimal DefaultMaxDailyLoss = -1000m;
    private const int DefaultMaxPositionSize = 5;
    private const decimal DefaultDrawdownLimit = -2000m;
    
    public string ApiBase { get; init; } = "https://api.topstepx.com";
    public string AuthToken { get; init; } = "";
    public string AccountId { get; init; } = "";
    public bool EnableDryRunMode { get; init; } = true;
    public bool EnableAutoExecution { get; init; }
    public decimal MaxDailyLoss { get; init; } = DefaultMaxDailyLoss;
    public int MaxPositionSize { get; init; } = DefaultMaxPositionSize;
    public decimal DrawdownLimit { get; init; } = DefaultDrawdownLimit;
    public string KillFile { get; init; } = "kill.txt";
}