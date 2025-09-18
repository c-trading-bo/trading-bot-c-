using System;

namespace TradingBot.Abstractions;

public sealed class AppOptions
{
    // Default risk management constants (configurable via environment)
    private static readonly decimal DefaultMaxDailyLoss = 
        decimal.TryParse(Environment.GetEnvironmentVariable("DEFAULT_MAX_DAILY_LOSS"), out var maxLoss) ? maxLoss : -1000m;
    private static readonly int DefaultMaxPositionSize = 
        int.TryParse(Environment.GetEnvironmentVariable("DEFAULT_MAX_POSITION_SIZE"), out var maxPos) ? maxPos : 5;
    private static readonly decimal DefaultDrawdownLimit = 
        decimal.TryParse(Environment.GetEnvironmentVariable("DEFAULT_DRAWDOWN_LIMIT"), out var drawdown) ? drawdown : -2000m;
    
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