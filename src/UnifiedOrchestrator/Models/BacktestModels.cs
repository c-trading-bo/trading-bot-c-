using System;
using System.Collections.Generic;
using AbstractionsTradingDecision = TradingBot.Abstractions.TradingDecision;

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
    public bool UseUnifiedBrain { get; set; } = true;
    public bool LearningMode { get; set; } = true;
    public string ConfigId { get; set; } = Guid.NewGuid().ToString();
    public Dictionary<string, object> Parameters { get; } = new();
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
    public Dictionary<string, object> Parameters { get; } = new();
    
    // Required properties per production specification
    public decimal MaxDrawdown { get; set; } = 0.10m; // 10% max drawdown default
    public decimal MaxPositionSize { get; set; } = 1000000m; // $1M default max position
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
    public List<UnifiedHistoricalDecision> Decisions { get; } = new();
    public Dictionary<string, object> Metadata { get; } = new();
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
    
    // Required properties per production specification
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public decimal InitialCapital { get; set; } = 100000m;
    public decimal FinalCapital { get; set; }
    public decimal SortinoRatio { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public DateTime CompletedAt { get; set; } = DateTime.UtcNow;
    public int BrainDecisionCount { get; set; }
    public double AverageProcessingTimeMs { get; set; }
    public int RiskCheckFailures { get; set; }
    public Dictionary<string, object> AlgorithmUsage { get; } = new();
    
    // Additional performance metrics added for production readiness
    public decimal WinRate { get; set; }
    public decimal ProfitFactor { get; set; }
    public decimal AverageWin { get; set; }
    public decimal AverageLoss { get; set; }
    public decimal AnnualizedReturn { get; set; }
    public decimal AnnualizedVolatility { get; set; }
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
    public List<UnifiedHistoricalDecision> UnifiedDecisions { get; } = new();
}

/// <summary>
/// Historical replay context for unified backtesting
/// </summary>
public class UnifiedHistoricalReplayContext
{
    public string ReplayId { get; set; } = string.Empty;
    public string BacktestId { get; set; } = string.Empty;
    public UnifiedBacktestConfig Config { get; set; } = new();
    public DateTime CurrentTime { get; set; } = DateTime.UtcNow;
    public string Strategy { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public int TotalBars { get; set; }
    public int ProcessedBars { get; set; }
    public bool IsActive { get; set; }
    public Dictionary<string, object> Context { get; } = new();
}

/// <summary>
/// Unified historical decision for enhanced tracking
/// </summary>
public class UnifiedHistoricalDecision
{
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public decimal Size { get; set; }
    public decimal Price { get; set; }
    public decimal Confidence { get; set; }
    public string Reasoning { get; set; } = string.Empty;
    public AbstractionsTradingDecision Decision { get; set; } = new();
    public TradingBot.UnifiedOrchestrator.Models.TradingContext MarketContext { get; set; } = new();
    public Dictionary<string, object> Metadata { get; } = new();
}

/// <summary>
/// Training intensity configuration
/// </summary>
public class TrainingIntensity
{
    public TrainingIntensityLevel Level { get; set; } = TrainingIntensityLevel.Medium;
    public int ParallelJobs { get; set; } = 2;
    
    // Predefined intensity levels for compatibility
    public static TrainingIntensity Light => new() { Level = TrainingIntensityLevel.Light, ParallelJobs = 1 };
    public static TrainingIntensity Medium => new() { Level = TrainingIntensityLevel.Medium, ParallelJobs = 2 };
    public static TrainingIntensity High => new() { Level = TrainingIntensityLevel.High, ParallelJobs = 4 };
    public static TrainingIntensity Intensive => new() { Level = TrainingIntensityLevel.Intensive, ParallelJobs = 8 };
}

/// <summary>
/// Training intensity level enumeration
/// </summary>
public enum TrainingIntensityLevel
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
    public AbstractionsTradingDecision ChampionDecision { get; set; } = null!;
    public AbstractionsTradingDecision ChallengerDecision { get; set; } = null!;
    public bool Agreement { get; set; }
    public double ConfidenceDelta { get; set; }
}

/// <summary>
/// Validation result class for consistency with interfaces
/// </summary>
public class ValidationResult
{
    public string ValidationId { get; set; } = string.Empty;
    public string? ChallengerVersionId { get; set; }
    public string? ChampionAlgorithm { get; set; }
    public string? ChallengerAlgorithm { get; set; }
    public ValidationReport Report { get; set; } = null!;
    public DateTime Timestamp { get; set; }
    public ValidationOutcome Outcome { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public double PerformanceScore { get; set; }
    public double RiskScore { get; set; }
    public double BehaviorScore { get; set; }
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

// ====================================================================
// ADDITIONAL REQUIRED MODELS PER PRODUCTION SPECIFICATION
// ====================================================================

/// <summary>
/// Enhanced backtest config with unified brain settings
/// </summary>
public class UnifiedBacktestConfigEnhanced
{
    public bool UseUnifiedBrain { get; set; } = true;
    public LearningMode LearningMode { get; set; } = LearningMode.Active;
    public string ConfigId { get; set; } = Guid.NewGuid().ToString();
    public string Symbol { get; set; } = "ES";
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public decimal InitialCapital { get; set; } = 100000m;
    public Dictionary<string, object> Parameters { get; } = new();
}

/// <summary>
/// Learning mode enumeration
/// </summary>
public enum LearningMode
{
    Passive,
    Active,
    Aggressive
}

/// <summary>
/// Enhanced historical replay context with unified brain support
/// </summary>
public class UnifiedHistoricalReplayContextEnhanced
{
    public string BacktestId { get; set; } = string.Empty;
    public UnifiedBacktestConfigEnhanced Config { get; set; } = new();
    public DateTime CurrentTime { get; set; } = DateTime.UtcNow;
    public string Strategy { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public int TotalBars { get; set; }
    public int ProcessedBars { get; set; }
    public bool IsActive { get; set; }
    public Dictionary<string, object> Context { get; } = new();
}

/// <summary>
/// Enhanced historical decision with market context
/// </summary>
public class UnifiedHistoricalDecisionEnhanced
{
    public string Symbol { get; set; } = string.Empty;
    public AbstractionsTradingDecision Decision { get; set; } = new();
    public TradingBot.UnifiedOrchestrator.Models.TradingContext MarketContext { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Strategy { get; set; } = string.Empty;
    public decimal Confidence { get; set; }
    public string Reasoning { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; } = new();
}