using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Portfolio;

/// <summary>
/// Interface for dynamic capital allocation based on volatility and risk
/// </summary>
public interface ICapitalAllocationManager
{
    /// <summary>
    /// Calculate optimal position sizes based on risk and volatility
    /// </summary>
    Task<PositionSizingResult> CalculatePositionSizeAsync(PositionSizingRequest request, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Allocate capital across multiple strategies
    /// </summary>
    Task<CapitalAllocationResult> AllocateCapitalAsync(CapitalAllocationRequest request, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Rebalance existing positions based on updated risk metrics
    /// </summary>
    Task<RebalancingResult> RebalancePortfolioAsync(RebalancingConfiguration config, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current capital allocation breakdown
    /// </summary>
    Task<CapitalAllocationBreakdown> GetAllocationBreakdownAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Update allocation parameters and risk models
    /// </summary>
    Task UpdateAllocationParametersAsync(AllocationParameters parameters, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Monitor allocation performance and efficiency
    /// </summary>
    Task<AllocationPerformanceReport> GetPerformanceReportAsync(TimeSpan lookbackPeriod, CancellationToken cancellationToken = default);
}

/// <summary>
/// Request for position sizing calculation
/// </summary>
public record PositionSizingRequest(
    string Symbol,
    decimal EntryPrice,
    decimal StopPrice,
    PositionDirection Direction,
    string Strategy,
    double? ConfidenceLevel = null,
    decimal? MaxRiskAmount = null
);

/// <summary>
/// Position direction
/// </summary>
public enum PositionDirection
{
    Long,
    Short
}

/// <summary>
/// Result of position sizing calculation
/// </summary>
public record PositionSizingResult(
    string Symbol,
    decimal RecommendedQuantity,
    decimal DollarAmount,
    decimal RiskAmount,
    double RiskPercentage,
    PositionSizingMethod Method,
    AllocationRationale Rationale
);

/// <summary>
/// Position sizing methods
/// </summary>
public enum PositionSizingMethod
{
    FixedDollar,
    FixedRisk,
    VolatilityAdjusted,
    KellyOptimal,
    RiskParity
}

/// <summary>
/// Rationale for allocation decision
/// </summary>
public record AllocationRationale(
    PositionSizingMethod Method,
    Dictionary<string, double> Inputs,
    Dictionary<string, double> Outputs,
    List<string> Considerations,
    double ConfidenceScore
);

/// <summary>
/// Request for capital allocation across strategies
/// </summary>
public record CapitalAllocationRequest(
    decimal TotalCapital,
    List<StrategyAllocationRequest> StrategyRequests,
    AllocationObjective Objective,
    AllocationConstraints Constraints
);

/// <summary>
/// Individual strategy allocation request
/// </summary>
public record StrategyAllocationRequest(
    string StrategyId,
    decimal RequestedAmount,
    StrategyPerformanceMetrics Performance,
    StrategyRiskMetrics Risk,
    int Priority = 1
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
/// Strategy risk metrics
/// </summary>
public record StrategyRiskMetrics(
    double Volatility,
    double VaR95,
    double Beta,
    double MaxDailyLoss,
    CorrelationToMarket CorrelationMetrics
);

/// <summary>
/// Correlation to market metrics
/// </summary>
public record CorrelationToMarket(
    double CorrelationCoefficient,
    double RSquared,
    double TrackingError
);

/// <summary>
/// Allocation objectives
/// </summary>
public enum AllocationObjective
{
    MaximizeReturn,
    MinimizeRisk,
    MaximizeSharpe,
    RiskParity,
    EqualWeight
}

/// <summary>
/// Allocation constraints
/// </summary>
public record AllocationConstraints(
    decimal MinAllocationPerStrategy,
    decimal MaxAllocationPerStrategy,
    double MaxCorrelationThreshold,
    bool RequireRiskBudgeting = true,
    Dictionary<string, decimal>? StrategyLimits = null
);

/// <summary>
/// Result of capital allocation
/// </summary>
public record CapitalAllocationResult(
    decimal TotalAllocated,
    List<StrategyAllocation> StrategyAllocations,
    AllocationMetrics Metrics,
    List<string> Warnings
);

/// <summary>
/// Individual strategy allocation
/// </summary>
public record StrategyAllocation(
    string StrategyId,
    decimal AllocatedAmount,
    double AllocationPercentage,
    decimal RiskContribution,
    AllocationRationale Rationale
);

/// <summary>
/// Allocation metrics
/// </summary>
public record AllocationMetrics(
    double ExpectedReturn,
    double ExpectedVolatility,
    double ExpectedSharpe,
    double DiversificationRatio,
    double RiskBudgetUtilization
);

/// <summary>
/// Configuration for portfolio rebalancing
/// </summary>
public record RebalancingConfiguration(
    RebalancingTrigger Trigger,
    double RebalanceThreshold,
    bool AllowPartialRebalancing = true,
    List<string>? StrategiesToRebalance = null,
    Dictionary<string, object>? CustomParameters = null
);

/// <summary>
/// Rebalancing triggers
/// </summary>
public enum RebalancingTrigger
{
    Scheduled,
    Threshold,
    RiskBreach,
    PerformanceDrift,
    Manual
}

/// <summary>
/// Result of rebalancing operation
/// </summary>
public record RebalancingResult(
    bool WasRebalanced,
    List<RebalancingAction> Actions,
    AllocationMetrics PreRebalanceMetrics,
    AllocationMetrics PostRebalanceMetrics,
    decimal TotalTradingCosts
);

/// <summary>
/// Individual rebalancing action
/// </summary>
public record RebalancingAction(
    string StrategyId,
    decimal CurrentAllocation,
    decimal TargetAllocation,
    decimal AllocationChange,
    string Reason
);

/// <summary>
/// Current capital allocation breakdown
/// </summary>
public record CapitalAllocationBreakdown(
    DateTime AsOfDate,
    decimal TotalCapital,
    decimal AllocatedCapital,
    decimal UnallocatedCapital,
    List<StrategyAllocation> StrategyAllocations,
    AllocationMetrics CurrentMetrics,
    List<AllocationAlert> Alerts
);

/// <summary>
/// Allocation alert
/// </summary>
public record AllocationAlert(
    AllocationAlertType Type,
    string Description,
    AllocationAlertSeverity Severity,
    DateTime AlertedAt,
    string? RecommendedAction = null
);

/// <summary>
/// Types of allocation alerts
/// </summary>
public enum AllocationAlertType
{
    OverAllocation,
    UnderAllocation,
    RiskBreach,
    PerformanceDegradation,
    CorrelationIncrease
}

/// <summary>
/// Allocation alert severity
/// </summary>
public enum AllocationAlertSeverity
{
    Info,
    Warning,
    Critical
}

/// <summary>
/// Allocation parameters configuration
/// </summary>
public record AllocationParameters(
    AllocationObjective DefaultObjective,
    AllocationConstraints DefaultConstraints,
    Dictionary<PositionSizingMethod, double> SizingMethodWeights,
    double RiskFreeRate,
    TimeSpan LookbackPeriodForMetrics,
    bool EnableDynamicAdjustment = true
);

/// <summary>
/// Performance report for allocation decisions
/// </summary>
public record AllocationPerformanceReport(
    TimeSpan ReportPeriod,
    DateTime GeneratedAt,
    double ActualReturn,
    double BenchmarkReturn,
    double Alpha,
    double InformationRatio,
    List<StrategyPerformanceContribution> StrategyContributions,
    AllocationEfficiencyMetrics Efficiency
);

/// <summary>
/// Strategy contribution to performance
/// </summary>
public record StrategyPerformanceContribution(
    string StrategyId,
    double ReturnContribution,
    double RiskContribution,
    double AllocationWeight,
    double PerformanceAttribution
);

/// <summary>
/// Allocation efficiency metrics
/// </summary>
public record AllocationEfficiencyMetrics(
    double CapitalUtilization,
    double RiskAdjustedReturn,
    double TurnoverRate,
    double RebalancingCosts,
    double OpportunityScore
);