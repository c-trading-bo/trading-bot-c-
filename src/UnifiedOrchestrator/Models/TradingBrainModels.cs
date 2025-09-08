using System.Text.Json.Serialization;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Models;

// Type aliases to Abstractions models for backward compatibility
using TradingBrainState = TradingBot.Abstractions.TradingBrainState;
using MLSystemState = TradingBot.Abstractions.MLSystemState;
using RiskSystemState = TradingBot.Abstractions.RiskSystemState;
using TradingSystemState = TradingBot.Abstractions.TradingSystemState;
using DataSystemState = TradingBot.Abstractions.DataSystemState;
using TradingMessage = TradingBot.Abstractions.TradingMessage;
using TradingSignal = TradingBot.Abstractions.TradingSignal;
using TradingDecision = TradingBot.Abstractions.TradingDecision;
using TradingAction = TradingBot.Abstractions.TradingAction;
using RiskAssessment = TradingBot.Abstractions.RiskAssessment;
using MarketRegime = TradingBot.Abstractions.MarketRegime;

// Additional models that are only used locally can remain here
/// <summary>
/// ML Recommendation
/// </summary>
public class MLRecommendation
{
    public string RecommendedStrategy { get; set; } = string.Empty;
    public decimal Confidence { get; set; } = 0m;
    public Dictionary<string, decimal> StrategyScores { get; set; } = new();
    public string[] Features { get; set; } = Array.Empty<string>();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Strategy Performance
/// </summary>
public class StrategyPerformance
{
    public string StrategyName { get; set; } = string.Empty;
    public decimal Returns { get; set; } = 0m;
    public decimal Sharpe { get; set; } = 0m;
    public decimal MaxDrawdown { get; set; } = 0m;
    public int TotalTrades { get; set; } = 0;
    public decimal WinRate { get; set; } = 0m;
    public bool IsActive { get; set; } = false;
    public DateTime LastTrade { get; set; } = DateTime.MinValue;
}

/// <summary>
/// Component Health Status
/// </summary>
public class ComponentHealth
{
    public string ComponentName { get; set; } = string.Empty;
    public bool IsHealthy { get; set; } = false;
    public string Status { get; set; } = "UNKNOWN";
    public Dictionary<string, object> Metrics { get; set; } = new();
    public List<string> Errors { get; set; } = new();
    public DateTime LastCheck { get; set; } = DateTime.UtcNow;
    public TimeSpan Uptime { get; set; } = TimeSpan.Zero;
}