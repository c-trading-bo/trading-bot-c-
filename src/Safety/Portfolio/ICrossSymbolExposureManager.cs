using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Portfolio;

/// <summary>
/// Interface for cross-symbol exposure management and risk aggregation
/// </summary>
public interface ICrossSymbolExposureManager
{
    /// <summary>
    /// Calculate aggregate risk across all symbols
    /// </summary>
    Task<AggregateRiskReport> CalculateAggregateRiskAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if proposed position would violate cross-symbol limits
    /// </summary>
    Task<ExposureCheckResult> CheckExposureLimitsAsync(ProposedPosition position, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current exposure by sector, asset class, and correlation groups
    /// </summary>
    Task<ExposureBreakdown> GetExposureBreakdownAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Update position and recalculate exposures
    /// </summary>
    Task UpdatePositionAsync(PositionUpdate update, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Set exposure limits and risk parameters
    /// </summary>
    Task UpdateExposureLimitsAsync(ExposureLimits limits, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get correlation matrix for all active symbols
    /// </summary>
    Task<CorrelationMatrix> GetSymbolCorrelationsAsync(TimeSpan lookbackPeriod, CancellationToken cancellationToken = default);
}

/// <summary>
/// Proposed position for exposure checking
/// </summary>
public record ProposedPosition(
    string Symbol,
    decimal Quantity,
    decimal Price,
    PositionSide Side,
    string? Strategy = null,
    DateTime ProposedAt = default
);

/// <summary>
/// Position side enumeration
/// </summary>
public enum PositionSide
{
    Long,
    Short,
    Flat
}

/// <summary>
/// Result of exposure limit checking
/// </summary>
public record ExposureCheckResult(
    bool IsAllowed,
    List<ExposureViolation> Violations,
    AggregateRiskProjection RiskProjection,
    List<string> Warnings
);

/// <summary>
/// Exposure limit violation
/// </summary>
public record ExposureViolation(
    ExposureViolationType Type,
    string Description,
    decimal CurrentValue,
    decimal LimitValue,
    decimal ProposedValue,
    ExposureViolationSeverity Severity
);

/// <summary>
/// Types of exposure violations
/// </summary>
public enum ExposureViolationType
{
    SingleSymbolExposure,
    SectorExposure,
    AssetClassExposure,
    CorrelationGroupExposure,
    TotalPortfolioExposure,
    ConcentrationLimit
}

/// <summary>
/// Severity of exposure violations
/// </summary>
public enum ExposureViolationSeverity
{
    Warning,
    Soft,
    Hard
}

/// <summary>
/// Position update record
/// </summary>
public record PositionUpdate(
    string Symbol,
    decimal NewQuantity,
    decimal NewPrice,
    DateTime UpdatedAt,
    string Source,
    string? Notes = null
);

/// <summary>
/// Comprehensive aggregate risk report
/// </summary>
public record AggregateRiskReport(
    DateTime GeneratedAt,
    decimal TotalPortfolioValue,
    decimal TotalExposure,
    decimal MaxPortfolioRisk,
    decimal CurrentPortfolioRisk,
    decimal BetaToMarket,
    Dictionary<string, SectorExposure> SectorExposures,
    Dictionary<string, AssetClassExposure> AssetClassExposures,
    List<ConcentrationRisk> ConcentrationRisks,
    CorrelationRiskMetrics CorrelationRisk
);

/// <summary>
/// Sector exposure details
/// </summary>
public record SectorExposure(
    string Sector,
    decimal TotalValue,
    decimal PercentageOfPortfolio,
    decimal NetExposure,
    decimal GrossExposure,
    List<string> Symbols
);

/// <summary>
/// Asset class exposure details
/// </summary>
public record AssetClassExposure(
    string AssetClass,
    decimal TotalValue,
    decimal PercentageOfPortfolio,
    decimal AverageVolatility,
    decimal VaR95,
    List<string> Symbols
);

/// <summary>
/// Concentration risk item
/// </summary>
public record ConcentrationRisk(
    string Symbol,
    decimal PositionValue,
    decimal PercentageOfPortfolio,
    ConcentrationRiskLevel RiskLevel,
    string Recommendation
);

/// <summary>
/// Concentration risk levels
/// </summary>
public enum ConcentrationRiskLevel
{
    Low,
    Medium,
    High,
    Critical
}

/// <summary>
/// Correlation risk metrics
/// </summary>
public record CorrelationRiskMetrics(
    double AverageCorrelation,
    double MaxCorrelation,
    int HighlyCorrelatedPairs,
    List<CorrelatedPair> TopCorrelations
);

/// <summary>
/// Highly correlated symbol pair
/// </summary>
public record CorrelatedPair(
    string Symbol1,
    string Symbol2,
    double Correlation,
    decimal CombinedExposure,
    bool ExceedsThreshold
);

/// <summary>
/// Exposure breakdown by various dimensions
/// </summary>
public record ExposureBreakdown(
    DateTime GeneratedAt,
    Dictionary<string, SymbolExposure> SymbolExposures,
    Dictionary<string, SectorExposure> SectorExposures,
    Dictionary<string, AssetClassExposure> AssetClassExposures,
    Dictionary<string, decimal> StrategyExposures,
    ExposureSummary Summary
);

/// <summary>
/// Individual symbol exposure
/// </summary>
public record SymbolExposure(
    string Symbol,
    decimal Quantity,
    decimal MarketValue,
    decimal UnrealizedPnL,
    decimal DollarRisk,
    double Beta,
    string Sector,
    string AssetClass
);

/// <summary>
/// Summary of total exposures
/// </summary>
public record ExposureSummary(
    decimal TotalLongExposure,
    decimal TotalShortExposure,
    decimal NetExposure,
    decimal GrossExposure,
    decimal TotalRisk
);

/// <summary>
/// Exposure limits configuration
/// </summary>
public record ExposureLimits(
    decimal MaxSingleSymbolExposure,
    decimal MaxSectorExposure,
    decimal MaxAssetClassExposure,
    decimal MaxTotalExposure,
    decimal MaxConcentrationPercent,
    double MaxCorrelationThreshold,
    Dictionary<string, decimal>? CustomSectorLimits = null,
    Dictionary<string, decimal>? CustomSymbolLimits = null
);

/// <summary>
/// Correlation matrix for symbols
/// </summary>
public record CorrelationMatrix(
    List<string> Symbols,
    DateTime CalculatedAt,
    TimeSpan LookbackPeriod,
    double[,] CorrelationValues,
    Dictionary<string, double> Volatilities
);

/// <summary>
/// Risk projection with proposed position
/// </summary>
public record AggregateRiskProjection(
    decimal CurrentRisk,
    decimal ProjectedRisk,
    decimal RiskIncrease,
    decimal ProjectedVaR95,
    double ProjectedSharpeRatio,
    List<string> RiskFactors
);