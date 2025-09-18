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
    public decimal Confidence { get; set; }
    public Dictionary<string, decimal> StrategyScores { get; } = new();
    public string[] Features { get; set; } = Array.Empty<string>();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Strategy Performance
/// </summary>
public class StrategyPerformance
{
    public string StrategyName { get; set; } = string.Empty;
    public decimal Returns { get; set; }
    public decimal Sharpe { get; set; }
    public decimal MaxDrawdown { get; set; }
    public int TotalTrades { get; set; }
    public decimal WinRate { get; set; }
    public bool IsActive { get; set; }
    public DateTime LastTrade { get; set; } = DateTime.MinValue;
}

/// <summary>
/// Component Health Status
/// </summary>
public class ComponentHealth
{
    public string ComponentName { get; set; } = string.Empty;
    public bool IsHealthy { get; set; }
    public string Status { get; set; } = "UNKNOWN";
    public Dictionary<string, object> Metrics { get; } = new();
    public List<string> Errors { get; } = new();
    public DateTime LastCheck { get; set; } = DateTime.UtcNow;
    public TimeSpan Uptime { get; set; } = TimeSpan.Zero;
}

/// <summary>
/// Validation Report with comprehensive statistical analysis
/// </summary>
public class ValidationReport
{
    public string ValidationId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string ChampionAlgorithm { get; set; } = string.Empty;
    public string ChallengerAlgorithm { get; set; } = string.Empty;
    public TimeSpan TestPeriod { get; set; }
    public int SampleSize { get; set; }
    public StatisticalSignificance StatisticalSignificance { get; set; } = null!;
    public PerformanceComparison PerformanceMetrics { get; set; } = null!;
    public RiskComparison RiskMetrics { get; set; } = null!;
    public BehaviorAlignment BehaviorAlignment { get; set; } = null!;
    public double ValidationDurationMs { get; set; }
    public bool Passed { get; set; }
    public string? ErrorMessage { get; set; }
    
    // Additional properties for compatibility
    public double ChallengerSharpe => PerformanceMetrics?.SharpeChallenger ?? 0;
    public double ChampionSharpe => PerformanceMetrics?.SharpeChampion ?? 0;
    public double PValue => StatisticalSignificance?.PValue ?? 1.0;
}

/// <summary>
/// Statistical significance analysis
/// </summary>
public class StatisticalSignificance
{
    public double PValue { get; set; }
    public double TStatistic { get; set; }
    public double KSStatistic { get; set; }
    public double KSPValue { get; set; }
    public double WilcoxonPValue { get; set; }
    public bool IsSignificant { get; set; }
    public double ConfidenceLevel { get; set; }
    public double EffectSize { get; set; }
}

/// <summary>
/// Performance comparison between champion and challenger
/// </summary>
public class PerformanceComparison
{
    public double SharpeChampion { get; set; }
    public double SharpeChallenger { get; set; }
    public double SharpeImprovement { get; set; }
    public double SortinoChampion { get; set; }
    public double SortinoChallenger { get; set; }
    public double SortinoImprovement { get; set; }
    public double TotalReturnChampion { get; set; }
    public double TotalReturnChallenger { get; set; }
    public double ReturnImprovement { get; set; }
    public double WinRateChampion { get; set; }
    public double WinRateChallenger { get; set; }
    public double WinRateImprovement { get; set; }
}

/// <summary>
/// Risk comparison including CVaR and drawdown analysis
/// </summary>
public class RiskComparison
{
    public double CVaRChampion { get; set; }
    public double CVaRChallenger { get; set; }
    public double CVaRImprovement { get; set; }
    public double MaxDrawdownChampion { get; set; }
    public double MaxDrawdownChallenger { get; set; }
    public double DrawdownImprovement { get; set; }
    public double VolatilityChampion { get; set; }
    public double VolatilityChallenger { get; set; }
    public double VolatilityChange { get; set; }
    public double VaRChampion { get; set; }
    public double VaRChallenger { get; set; }
}

/// <summary>
/// Behavior alignment analysis
/// </summary>
public class BehaviorAlignment
{
    public double AlignmentPercentage { get; set; }
    public double MajorDisagreementRate { get; set; }
    public double AverageConfidenceDelta { get; set; }
    public double MaxConfidenceDelta { get; set; }
    public int TotalDecisionsCompared { get; set; }
    public double BehaviorSimilarityScore { get; set; }
}

/// <summary>
/// Shadow test result for validation
/// </summary>
public class ShadowTestResult
{
    public string TestId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string Algorithm { get; set; } = string.Empty;
    public string ChampionVersionId { get; set; } = string.Empty;
    public string ChallengerVersionId { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public int DecisionCount { get; set; }
    public double AgreementRate { get; set; }
    public double PerformanceScore { get; set; }
    public TradingAction Decision { get; set; }
    public double Confidence { get; set; }
    public double Return { get; set; }
    public bool Success { get; set; }
}

/// <summary>
/// Trading context for decision making - unified definition
/// </summary>
public class TradingContext
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    // Market data (from simple version)
    public decimal CurrentPrice { get; set; }
    public decimal Price { get; set; } // Alias for CurrentPrice for compatibility
    public long Volume { get; set; }
    public decimal Spread { get; set; }
    public bool IsMarketOpen { get; set; }
    
    // Required OHLC properties per production specification
    public decimal High { get; set; }
    public decimal Low { get; set; }
    public decimal Open { get; set; }
    public decimal Close { get; set; }
    
    // Technical indicators and market data (from comprehensive version)
    public decimal Volatility { get; set; }
    public Dictionary<string, decimal> TechnicalIndicators { get; } = new();
    public Dictionary<string, decimal> MarketData { get; } = new();
    public Dictionary<string, object> Metadata { get; } = new();
    
    // Position context (from comprehensive version)
    public decimal CurrentPosition { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public decimal DailyPnL { get; set; }
    public decimal AccountBalance { get; set; }
    
    // Required properties per production specification
    public decimal RealizedPnL { get; set; }
    public decimal MaxPositionSize { get; set; } = 1000000m;
    public bool IsBacktest { get; set; }
    
    // Risk context (from both versions)
    public decimal MaxDrawdown { get; set; }
    public decimal DailyLossLimit { get; set; }
    public bool IsEmergencyStop { get; set; }
    public Dictionary<string, object> RiskParameters { get; } = new();
    
    // Shadow testing source identifier
    public string Source { get; set; } = string.Empty;
}