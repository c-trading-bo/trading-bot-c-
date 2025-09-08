using System;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.MetaLabeler;

/// <summary>
/// Meta-labeler that estimates win probability p(win) for potential trades
/// using supervised ML models trained on triple-barrier labeled historical data.
/// </summary>
public interface IMetaLabeler
{
    /// <summary>
    /// Estimates win probability for a potential trade signal.
    /// </summary>
    /// <param name="signal">Trade signal with entry, stop, target prices</param>
    /// <param name="marketContext">Current market conditions (spread, volatility, etc.)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Estimated probability of winning (0.0 to 1.0)</returns>
    Task<decimal> EstimateWinProbabilityAsync(
        TradeSignalContext signal,
        MarketContext marketContext,
        CancellationToken ct = default);

    /// <summary>
    /// Gets the minimum win probability threshold for accepting trades.
    /// This can be dynamic based on market regime.
    /// </summary>
    decimal GetMinWinProbabilityThreshold();

    /// <summary>
    /// Updates model calibration metrics after trade outcomes are known.
    /// </summary>
    /// <param name="predictedProb">Originally predicted win probability</param>
    /// <param name="actualOutcome">True if trade was profitable, false otherwise</param>
    /// <param name="ct">Cancellation token</param>
    Task UpdateCalibrationAsync(decimal predictedProb, bool actualOutcome, CancellationToken ct = default);

    /// <summary>
    /// Gets current model calibration metrics (Brier score, reliability, etc.)
    /// </summary>
    Task<CalibrationMetrics> GetCalibrationMetricsAsync(CancellationToken ct = default);
}

/// <summary>
/// Context for a potential trade signal used by meta-labeler
/// </summary>
public record TradeSignalContext(
    string Symbol,
    string Strategy,
    string SignalId,
    DateTime Timestamp,
    decimal EntryPrice,
    decimal StopPrice,
    decimal TargetPrice,
    bool IsLong,
    decimal Confidence,
    string Regime,
    decimal AtrMultiple,
    decimal RMultiple
);

/// <summary>
/// Current market conditions for meta-labeler features
/// </summary>
public record MarketContext(
    decimal BidAskSpread,
    decimal SpreadBps,
    decimal Volatility,
    decimal Volume,
    decimal AtrZScore,
    decimal TimeOfDay,
    int DayOfWeek,
    decimal RecentMomentum,
    decimal RegimeProbability,
    bool IsSessionStart,
    bool IsSessionEnd
);

/// <summary>
/// Model calibration metrics for monitoring prediction quality
/// </summary>
public record CalibrationMetrics(
    decimal BrierScore,
    decimal LogLoss,
    decimal ReliabilityScore,
    decimal ResolutionScore,
    int TotalPredictions,
    DateTime LastUpdated,
    bool IsWellCalibrated
);
