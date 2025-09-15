using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Analytics;

/// <summary>
/// Position direction
/// </summary>
public enum PositionDirection
{
    Long,
    Short
}

/// <summary>
/// Interface for post-trade attribution analysis by regime and context
/// </summary>
public interface IPostTradeAttributionManager
{
    /// <summary>
    /// Record a completed trade with context for attribution analysis
    /// </summary>
    Task RecordTradeAsync(TradeRecord trade, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Generate attribution report for specified period and filters
    /// </summary>
    Task<AttributionReport> GenerateAttributionReportAsync(AttributionAnalysisRequest request, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Analyze performance by market regime
    /// </summary>
    Task<RegimePerformanceAnalysis> AnalyzeRegimePerformanceAsync(TimeSpan lookbackPeriod, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get trading context impact analysis
    /// </summary>
    Task<ContextImpactAnalysis> AnalyzeContextImpactAsync(List<string> contextFilters, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Calculate strategy-specific attribution metrics
    /// </summary>
    Task<StrategyAttributionReport> AnalyzeStrategyAttributionAsync(string strategyId, TimeSpan period, CancellationToken cancellationToken = default);
}

/// <summary>
/// Trade record with comprehensive context for attribution
/// </summary>
public record TradeRecord(
    string TradeId,
    string Symbol,
    string Strategy,
    DateTime EntryTime,
    DateTime ExitTime,
    decimal EntryPrice,
    decimal ExitPrice,
    decimal Quantity,
    PositionDirection Direction,
    decimal PnL,
    decimal Commission,
    MarketRegime EntryRegime,
    MarketRegime ExitRegime,
    TradingContext Context,
    Dictionary<string, object>? Metadata = null
);

/// <summary>
/// Market regime classification
/// </summary>
public enum MarketRegime
{
    LowVolatility,
    HighVolatility,
    Trending,
    Ranging,
    Crisis,
    Recovery,
    Unknown
}

/// <summary>
/// Trading context information
/// </summary>
public record TradingContext(
    TimeOfDay TimeSlot,
    DayOfWeek TradingDay,
    bool IsHoliday,
    bool IsEarningsDay,
    bool IsFOMCDay,
    VolumeRegime VolumeCondition,
    VolatilityRegime VolatilityCondition,
    Dictionary<string, object>? CustomContexts = null
);

/// <summary>
/// Time of day classification
/// </summary>
public enum TimeOfDay
{
    PreMarket,
    MarketOpen,
    MidMorning,
    Lunch,
    Afternoon,
    MarketClose,
    AfterHours
}

/// <summary>
/// Volume regime classification
/// </summary>
public enum VolumeRegime
{
    Low,
    Normal,
    High,
    Extreme
}

/// <summary>
/// Volatility regime classification
/// </summary>
public enum VolatilityRegime
{
    Low,
    Normal,
    High,
    Extreme
}

/// <summary>
/// Request for attribution analysis
/// </summary>
public record AttributionAnalysisRequest(
    DateTime StartDate,
    DateTime EndDate,
    List<string>? Strategies = null,
    List<string>? Symbols = null,
    List<MarketRegime>? Regimes = null,
    AttributionDimension PrimaryDimension = AttributionDimension.Strategy
);

/// <summary>
/// Attribution analysis dimensions
/// </summary>
public enum AttributionDimension
{
    Strategy,
    Symbol,
    Regime,
    TimeOfDay,
    VolatilityRegime,
    Custom
}

/// <summary>
/// Comprehensive attribution report
/// </summary>
public record AttributionReport(
    DateTime GeneratedAt,
    TimeSpan AnalysisPeriod,
    AttributionSummary Summary,
    List<AttributionBreakdown> Breakdowns,
    AttributionInsights Insights,
    List<AttributionRecommendation> Recommendations
);

/// <summary>
/// Summary of attribution analysis
/// </summary>
public record AttributionSummary(
    decimal TotalPnL,
    decimal TotalCommissions,
    decimal NetPnL,
    int TotalTrades,
    double WinRate,
    double ProfitFactor,
    double SharpeRatio,
    decimal MaxDrawdown
);

/// <summary>
/// Attribution breakdown by dimension
/// </summary>
public record AttributionBreakdown(
    AttributionDimension Dimension,
    string DimensionValue,
    decimal PnL,
    int TradeCount,
    double WinRate,
    double AverageReturn,
    double Volatility,
    double ContributionToTotal
);

/// <summary>
/// Insights from attribution analysis
/// </summary>
public record AttributionInsights(
    List<string> BestPerformingCategories,
    List<string> WorstPerformingCategories,
    List<string> MostConsistentCategories,
    List<AttributionPattern> Patterns,
    List<string> OutlierTrades
);

/// <summary>
/// Pattern identified in attribution analysis
/// </summary>
public record AttributionPattern(
    string PatternType,
    string Description,
    double Confidence,
    decimal Impact,
    List<string> SupportingEvidence
);

/// <summary>
/// Recommendation from attribution analysis
/// </summary>
public record AttributionRecommendation(
    RecommendationType Type,
    string Description,
    decimal PotentialImpact,
    List<string> ActionItems,
    RecommendationPriority Priority
);

/// <summary>
/// Recommendation types
/// </summary>
public enum RecommendationType
{
    IncreaseAllocation,
    DecreaseAllocation,
    OptimizeTiming,
    AvoidConditions,
    ImproveStrategy,
    InvestigateAnomaly
}

/// <summary>
/// Recommendation priority levels
/// </summary>
public enum RecommendationPriority
{
    Low,
    Medium,
    High,
    Critical
}

/// <summary>
/// Performance analysis by market regime
/// </summary>
public record RegimePerformanceAnalysis(
    DateTime GeneratedAt,
    TimeSpan AnalysisPeriod,
    List<RegimePerformance> RegimePerformances,
    RegimeTransitionAnalysis TransitionAnalysis,
    List<string> KeyFindings
);

/// <summary>
/// Performance in specific market regime
/// </summary>
public record RegimePerformance(
    MarketRegime Regime,
    TimeSpan TimeInRegime,
    decimal PnL,
    int TradeCount,
    double WinRate,
    double SharpeRatio,
    decimal MaxDrawdown,
    double PerformanceRank
);

/// <summary>
/// Analysis of regime transitions
/// </summary>
public record RegimeTransitionAnalysis(
    List<RegimeTransition> Transitions,
    double AverageTransitionImpact,
    List<string> TransitionPatterns
);

/// <summary>
/// Individual regime transition
/// </summary>
public record RegimeTransition(
    DateTime TransitionTime,
    MarketRegime FromRegime,
    MarketRegime ToRegime,
    decimal ImpactOnPnL,
    int TradesAffected
);

/// <summary>
/// Analysis of trading context impact
/// </summary>
public record ContextImpactAnalysis(
    DateTime GeneratedAt,
    List<ContextPerformance> ContextPerformances,
    List<ContextCorrelation> Correlations,
    ContextOptimizationSuggestions Suggestions
);

/// <summary>
/// Performance in specific context
/// </summary>
public record ContextPerformance(
    string ContextName,
    string ContextValue,
    decimal AvgPnLPerTrade,
    double WinRate,
    int SampleSize,
    double StatisticalSignificance
);

/// <summary>
/// Correlation between contexts
/// </summary>
public record ContextCorrelation(
    string Context1,
    string Context2,
    double CorrelationCoefficient,
    double CombinedImpact
);

/// <summary>
/// Optimization suggestions based on context
/// </summary>
public record ContextOptimizationSuggestions(
    List<string> ContextsToAvoid,
    List<string> ContextsToSeek,
    List<string> TimingOptimizations,
    List<string> StrategyAdjustments
);

/// <summary>
/// Strategy performance metrics
/// </summary>
public record StrategyPerformanceMetrics(
    double SharpeRatio,
    double CalmarRatio,
    double WinRate,
    double AverageReturn,
    double MaxDrawdown,
    int TotalTrades,
    TimeSpan TrackRecord
);

/// <summary>
/// Strategy-specific attribution report
/// </summary>
public record StrategyAttributionReport(
    string StrategyId,
    DateTime GeneratedAt,
    TimeSpan AnalysisPeriod,
    StrategyPerformanceMetrics OverallMetrics,
    List<SymbolAttributionBreakdown> SymbolBreakdowns,
    List<RegimeAttributionBreakdown> RegimeBreakdowns,
    StrategyEfficiencyMetrics Efficiency,
    List<string> ImprovementOpportunities
);

/// <summary>
/// Symbol-specific attribution for strategy
/// </summary>
public record SymbolAttributionBreakdown(
    string Symbol,
    decimal PnL,
    int TradeCount,
    double WinRate,
    double ContributionToStrategy,
    double RiskAdjustedReturn
);

/// <summary>
/// Regime-specific attribution for strategy
/// </summary>
public record RegimeAttributionBreakdown(
    MarketRegime Regime,
    decimal PnL,
    int TradeCount,
    double PerformanceInRegime,
    double RegimeOptimizationScore
);

/// <summary>
/// Strategy efficiency metrics
/// </summary>
public record StrategyEfficiencyMetrics(
    double CapitalEfficiency,
    double TimeEfficiency,
    double RiskEfficiency,
    double OpportunityCapture,
    double ConsistencyScore
);