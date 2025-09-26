using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.MetaLabeler;

/// <summary>
/// Generates supervised learning labels using triple-barrier method:
/// - Profit target hit (label = 1)
/// - Stop loss hit (label = 0)  
/// - Time exit based on holding period (label = 0.5 or based on final PnL)
/// </summary>
public class TripleBarrierLabeler
{
    private readonly IHistoricalDataProvider _dataProvider;
    private readonly decimal _defaultProfitRatio;
    private readonly decimal _defaultStopRatio;
    private readonly TimeSpan _maxHoldingPeriod;

    public TripleBarrierLabeler(
        IHistoricalDataProvider dataProvider,
        decimal defaultProfitRatio = 2.0m,
        decimal defaultStopRatio = 1.0m,
        TimeSpan? maxHoldingPeriod = null)
    {
        _dataProvider = dataProvider;
        _defaultProfitRatio = defaultProfitRatio;
        _defaultStopRatio = defaultStopRatio;
        _maxHoldingPeriod = maxHoldingPeriod ?? TimeSpan.FromHours(4);
    }

    /// <summary>
    /// Labels historical trade signals using triple-barrier method.
    /// </summary>
    /// <param name="signals">Historical trade signals to label</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Labeled training data for supervised learning</returns>
    public async Task<List<LabeledTradeData>> LabelSignalsAsync(
        List<HistoricalTradeSignal> signals,
        CancellationToken ct = default)
    {
        if (signals is null) throw new ArgumentNullException(nameof(signals));
        
        var labeled = new List<LabeledTradeData>();

        foreach (var signal in signals)
        {
            var label = await ComputeTripleBarrierLabelAsync(signal, ct).ConfigureAwait(false);
            if (label != null)
            {
                labeled.Add(label);
            }
        }

        return labeled;
    }

    private async Task<LabeledTradeData?> ComputeTripleBarrierLabelAsync(
        HistoricalTradeSignal signal,
        CancellationToken ct)
    {
        try
        {
            // Get price data after signal timestamp
            var endTime = signal.Timestamp.Add(_maxHoldingPeriod);
            var priceData = await _dataProvider.GetPriceDataAsync(
                signal.Symbol,
                signal.Timestamp,
                endTime,
                ct).ConfigureAwait(false);

            if (!priceData.Any())
                return null;

            // Calculate barriers
            var profitTarget = signal.IsLong
                ? signal.EntryPrice * (1 + _defaultProfitRatio * 0.0025m) // Assuming 0.25% risk per R
                : signal.EntryPrice * (1 - _defaultProfitRatio * 0.0025m);

            var stopLoss = signal.IsLong
                ? signal.EntryPrice * (1 - _defaultStopRatio * 0.0025m)
                : signal.EntryPrice * (1 + _defaultStopRatio * 0.0025m);

            // Find first barrier hit
            var result = FindFirstBarrierHit(signal, priceData, profitTarget, stopLoss);

            // Create labeled data with features
            return new LabeledTradeData
            {
                SignalId = signal.SignalId,
                Symbol = signal.Symbol,
                Strategy = signal.Strategy,
                EntryTime = signal.Timestamp,
                ExitTime = result.ExitTime,
                ExitReason = result.ExitReason,
                Label = result.Label,

                // Features for ML model
                Features = ExtractFeatures(signal),

                // Outcome metrics
                HoldingPeriod = result.ExitTime - signal.Timestamp,
                RealizedR = CalculateRealizedR(signal, result.ExitPrice),
                ExitPrice = result.ExitPrice
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LABELER] Error labeling signal {signal.SignalId}: {ex.Message}");
            return null;
        }
    }

    private static BarrierResult FindFirstBarrierHit(
        HistoricalTradeSignal signal,
        IEnumerable<PriceBar> priceData,
        decimal profitTarget,
        decimal stopLoss)
    {
        foreach (var bar in priceData)
        {
            if (signal.IsLong)
            {
                // Long position: check high for profit, low for stop
                if (bar.High >= profitTarget)
                {
                    return new BarrierResult
                    {
                        ExitTime = bar.Timestamp,
                        ExitPrice = profitTarget,
                        ExitReason = "PROFIT_TARGET",
                        Label = 1.0m // Win
                    };
                }
                if (bar.Low <= stopLoss)
                {
                    return new BarrierResult
                    {
                        ExitTime = bar.Timestamp,
                        ExitPrice = stopLoss,
                        ExitReason = "STOP_LOSS",
                        Label = 0.0m // Loss
                    };
                }
            }
            else
            {
                // Short position: check low for profit, high for stop
                if (bar.Low <= profitTarget)
                {
                    return new BarrierResult
                    {
                        ExitTime = bar.Timestamp,
                        ExitPrice = profitTarget,
                        ExitReason = "PROFIT_TARGET",
                        Label = 1.0m // Win
                    };
                }
                if (bar.High >= stopLoss)
                {
                    return new BarrierResult
                    {
                        ExitTime = bar.Timestamp,
                        ExitPrice = stopLoss,
                        ExitReason = "STOP_LOSS",
                        Label = 0.0m // Loss
                    };
                }
            }
        }

        // Time exit - use final price and calculate label based on PnL
        var lastBar = priceData.Last();
        var finalPnL = signal.IsLong
            ? (lastBar.Close - signal.EntryPrice) / signal.EntryPrice
            : (signal.EntryPrice - lastBar.Close) / signal.EntryPrice;

        return new BarrierResult
        {
            ExitTime = lastBar.Timestamp,
            ExitPrice = lastBar.Close,
            ExitReason = "TIME_EXIT",
            Label = finalPnL > 0 ? 0.7m : 0.3m // Partial win/loss for time exits
        };
    }

    private Dictionary<string, decimal> ExtractFeatures(
        HistoricalTradeSignal signal
        )
    {
        return new Dictionary<string, decimal>
        {
            ["atr_multiple"] = signal.AtrMultiple,
            ["r_multiple"] = signal.RMultiple,
            ["confidence"] = signal.Confidence,
            ["spread_bps"] = signal.SpreadBps,
            ["volatility"] = signal.Volatility,
            ["volume_ratio"] = signal.VolumeRatio,
            ["time_of_day"] = (decimal)signal.Timestamp.TimeOfDay.TotalHours,
            ["day_of_week"] = (decimal)signal.Timestamp.DayOfWeek,
            ["regime_prob"] = signal.RegimeProbability,
            ["momentum"] = signal.RecentMomentum,
            ["is_session_start"] = signal.IsSessionStart ? 1m : 0m,
            ["is_session_end"] = signal.IsSessionEnd ? 1m : 0m
        };
    }

    private decimal CalculateRealizedR(HistoricalTradeSignal signal, decimal exitPrice)
    {
        var risk = Math.Abs(signal.EntryPrice - signal.StopPrice);
        var reward = signal.IsLong
            ? exitPrice - signal.EntryPrice
            : signal.EntryPrice - exitPrice;

        return risk > 0 ? reward / risk : 0m;
    }

    private sealed record BarrierResult
    {
        public DateTime ExitTime { get; init; }
        public decimal ExitPrice { get; init; }
        public string ExitReason { get; init; } = "";
        public decimal Label { get; init; }
    }
}

/// <summary>
/// Historical trade signal for labeling
/// </summary>
public record HistoricalTradeSignal
{
    public string SignalId { get; init; } = "";
    public string Symbol { get; init; } = "";
    public string Strategy { get; init; } = "";
    public DateTime Timestamp { get; init; }
    public decimal EntryPrice { get; init; }
    public decimal StopPrice { get; init; }
    public decimal TargetPrice { get; init; }
    public bool IsLong { get; init; }
    public decimal Confidence { get; init; }
    public decimal AtrMultiple { get; init; }
    public decimal RMultiple { get; init; }
    public decimal SpreadBps { get; init; }
    public decimal Volatility { get; init; }
    public decimal VolumeRatio { get; init; }
    public decimal RegimeProbability { get; init; }
    public decimal RecentMomentum { get; init; }
    public bool IsSessionStart { get; init; }
    public bool IsSessionEnd { get; init; }
}

/// <summary>
/// Labeled training data for supervised learning
/// </summary>
public record LabeledTradeData
{
    public string SignalId { get; init; } = "";
    public string Symbol { get; init; } = "";
    public string Strategy { get; init; } = "";
    public DateTime EntryTime { get; init; }
    public DateTime ExitTime { get; init; }
    public string ExitReason { get; init; } = "";
    public decimal Label { get; init; }
    public Dictionary<string, decimal> Features { get; init; } = new();
    public TimeSpan HoldingPeriod { get; init; }
    public decimal RealizedR { get; init; }
    public decimal ExitPrice { get; init; }
}

/// <summary>
/// Price bar data for barrier calculations
/// </summary>
public record PriceBar
{
    public DateTime Timestamp { get; init; }
    public decimal Open { get; init; }
    public decimal High { get; init; }
    public decimal Low { get; init; }
    public decimal Close { get; init; }
    public long Volume { get; init; }
}

/// <summary>
/// Provider for historical price data
/// </summary>
public interface IHistoricalDataProvider
{
    Task<List<PriceBar>> GetPriceDataAsync(
        string symbol,
        DateTime start,
        DateTime end,
        CancellationToken ct = default);
}
