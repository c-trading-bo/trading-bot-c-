using System;
using System.Collections.Generic;

namespace TradingBot.UnifiedOrchestrator.Models;

/// <summary>
/// Unified backtest configuration for enhanced learning service
/// </summary>
public class UnifiedBacktestConfig
{
    public string Symbol { get; set; } = "ES";
    public string Strategy { get; set; } = string.Empty;
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public decimal InitialCapital { get; set; } = 100000m;
    public int MinTradesRequired { get; set; } = 10;
    public bool EnableUnifiedBrainLearning { get; set; } = true;
    public bool EnableContinuousLearning { get; set; } = true;
    public Dictionary<string, object> Parameters { get; set; } = new();
}

/// <summary>
/// Basic backtest configuration for legacy compatibility
/// </summary>
public class BacktestConfig
{
    public string Symbol { get; set; } = "ES";
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public decimal InitialCapital { get; set; } = 100000m;
    public string DataSource { get; set; } = "historical";
    public TrainingIntensity TrainingIntensity { get; set; } = TrainingIntensity.Medium;
    public Dictionary<string, object> Parameters { get; set; } = new();
}

/// <summary>
/// Unified backtest result with enhanced metrics
/// </summary>
public class UnifiedBacktestResult
{
    public string BacktestId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    
    // Performance metrics
    public decimal TotalReturn { get; set; }
    public decimal NetPnL { get; set; }
    public decimal SharpeRatio { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal WinRate { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    
    // Enhanced metrics
    public decimal CalmarRatio { get; set; }
    public decimal SortinoRatio { get; set; }
    public decimal VaR95 { get; set; }
    public decimal CVaR { get; set; }
    public List<UnifiedHistoricalDecision> Decisions { get; set; } = new();
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Basic backtest result for legacy compatibility
/// </summary>
public class BacktestResult
{
    public string BacktestId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public decimal TotalReturn { get; set; }
    public decimal SharpeRatio { get; set; }
    public decimal MaxDrawdown { get; set; }
    public int TotalTrades { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// Unified backtest state tracking
/// </summary>
public class UnifiedBacktestState
{
    public decimal StartingCapital { get; set; }
    public decimal CurrentCapital { get; set; }
    public decimal Position { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public decimal RealizedPnL { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal AverageEntryPrice { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public List<UnifiedHistoricalDecision> UnifiedDecisions { get; set; } = new();
}

/// <summary>
/// Historical replay context for unified backtesting
/// </summary>
public class UnifiedHistoricalReplayContext
{
    public string ReplayId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public int TotalBars { get; set; }
    public int ProcessedBars { get; set; }
    public bool IsActive { get; set; }
    public Dictionary<string, object> Context { get; set; } = new();
}

/// <summary>
/// Unified historical decision for enhanced tracking
/// </summary>
public class UnifiedHistoricalDecision
{
    public DateTime Timestamp { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public decimal Size { get; set; }
    public decimal Price { get; set; }
    public decimal Confidence { get; set; }
    public string Reasoning { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Training intensity enumeration
/// </summary>
public enum TrainingIntensity
{
    Light,
    Medium, 
    High,
    Intensive
}

/// <summary>
/// Adapter statistics for brain adapter monitoring
/// </summary>
public class AdapterStatistics
{
    public int TotalDecisions { get; set; }
    public int AgreementCount { get; set; }
    public int DisagreementCount { get; set; }
    public double AgreementRate { get; set; }
    public string CurrentPrimary { get; set; } = string.Empty;
    public DateTime LastDecisionTime { get; set; }
}

/// <summary>
/// Comparison between champion and challenger decisions
/// </summary>
public class DecisionComparison
{
    public DateTime Timestamp { get; set; }
    public TradingContext Context { get; set; } = null!;
    public TradingBot.UnifiedOrchestrator.Interfaces.TradingDecision ChampionDecision { get; set; } = null!;
    public TradingBot.UnifiedOrchestrator.Interfaces.TradingDecision ChallengerDecision { get; set; } = null!;
    public bool Agreement { get; set; }
    public double ConfidenceDelta { get; set; }
}

/// <summary>
/// Validation result class for consistency with interfaces
/// </summary>
public class ValidationResult
{
    public ValidationReport Report { get; set; } = null!;
    public DateTime Timestamp { get; set; }
    public ValidationOutcome Outcome { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// Validation outcome enumeration
/// </summary>
public enum ValidationOutcome
{
    Passed,
    Failed,
    InsufficientData,
    Error
}