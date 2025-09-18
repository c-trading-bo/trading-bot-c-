using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Execution;

/// <summary>
/// Expected value based execution router that chooses between limit and market orders
/// to maximize trade profitability after accounting for execution costs.
/// </summary>
public class EvExecutionRouter : IExecutionRouter
{
    private readonly IMicrostructureAnalyzer _microstructureAnalyzer;
    private readonly IExecutionCostTracker _costTracker;

    public EvExecutionRouter(
        IMicrostructureAnalyzer microstructureAnalyzer,
        IExecutionCostTracker costTracker)
    {
        _microstructureAnalyzer = microstructureAnalyzer;
        _costTracker = costTracker;
    }

    /// <summary>
    /// Routes order using expected value optimization.
    /// Calculates EV = p(win) × avgWin - (1-p) × avgLoss - predictedSlippage
    /// </summary>
    public async Task<ExecutionDecision> RouteOrderAsync(
        TradeSignal signal,
        MarketContext marketContext,
        CancellationToken ct = default)
    {
        try
        {
            var intent = CreateTradeIntent(signal, marketContext);
            var recommendation = await _microstructureAnalyzer.GetExecutionRecommendationAsync(intent, null, ct).ConfigureAwait(false);

            var decision = new ExecutionDecision
            {
                OrderType = recommendation.RecommendedOrderType,
                LimitPrice = recommendation.LimitPrice,
                ExpectedSlippageBps = recommendation.PredictedSlippageBps,
                FillProbability = recommendation.FillProbability,
                ExpectedValue = recommendation.ExpectedValue,
                Reasoning = CreateReasoningText(signal, recommendation),
                RiskLevel = recommendation.RiskAssessment,
                EstimatedFillTime = recommendation.EstimatedFillTime,
                ShouldExecute = ShouldExecuteTrade(signal, recommendation)
            };

            // Log the decision
            Console.WriteLine($"[EV-ROUTER] {signal.SignalId} {decision.OrderType} " +
                            $"EV={decision.ExpectedValue:F3} slippage={decision.ExpectedSlippageBps:F1}bps " +
                            $"fill_prob={decision.FillProbability:P0} execute={decision.ShouldExecute}");

            return decision;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[EV-ROUTER] Error routing order for {signal.SignalId}: {ex.Message}");

            // Fallback to market order
            return new ExecutionDecision
            {
                OrderType = OrderType.Market,
                ExpectedSlippageBps = 5m, // Conservative estimate
                FillProbability = 1.0m,
                ExpectedValue = 0m,
                Reasoning = "Fallback to market order due to routing error",
                RiskLevel = ExecutionRisk.Medium,
                EstimatedFillTime = TimeSpan.FromSeconds(5),
                ShouldExecute = true
            };
        }
    }

    /// <summary>
    /// Updates execution costs after trade completion for learning.
    /// </summary>
    public async Task UpdateExecutionResultAsync(
        string signalId,
        ExecutionDecision originalDecision,
        ExecutionResult actualResult,
        CancellationToken ct = default)
    {
        await _costTracker.RecordExecutionAsync(signalId, originalDecision, actualResult, ct).ConfigureAwait(false);

        var predictionError = Math.Abs(originalDecision.ExpectedSlippageBps - actualResult.ActualSlippageBps);

        Console.WriteLine($"[EV-ROUTER] Execution update {signalId}: " +
                        $"predicted={originalDecision.ExpectedSlippageBps:F1}bps " +
                        $"actual={actualResult.ActualSlippageBps:F1}bps " +
                        $"error={predictionError:F1}bps");

        // Learn from significant prediction errors
        if (predictionError > 3m)
        {
            await _costTracker.AnalyzePredictionErrorAsync(signalId, originalDecision, actualResult, ct).ConfigureAwait(false);
        }
    }

    private static TradeIntent CreateTradeIntent(TradeSignal signal, MarketContext marketContext)
    {
        return new TradeIntent
        {
            Symbol = signal.Symbol,
            Quantity = signal.Quantity,
            IsBuy = signal.Side == TradeSide.Buy,
            MaxSlippageBps = 10m, // Configurable per strategy
            MaxWaitTime = TimeSpan.FromMinutes(3),
            Urgency = DetermineUrgency(signal, marketContext),
            ExpectedWinRate = signal.MetaWinProbability ?? 0.5m, // From meta-labeler
            RMultiple = signal.RMultiple
        };
    }

    private static ExecutionUrgency DetermineUrgency(TradeSignal signal, MarketContext marketContext)
    {
        // High urgency for strong signals in volatile markets
        if (signal.Confidence > 0.8m && marketContext.IsVolatile)
            return ExecutionUrgency.High;

        // Low urgency for weak signals
        if (signal.Confidence < 0.4m)
            return ExecutionUrgency.Low;

        return ExecutionUrgency.Normal;
    }

    private static string CreateReasoningText(TradeSignal signal, ExecutionRecommendation recommendation)
    {
        var confidence = signal.Confidence;
        var winProb = signal.MetaWinProbability ?? 0.5m;

        return $"{recommendation.Reasoning}. " +
               $"Signal confidence={confidence:P0}, meta p(win)={winProb:P0}, " +
               $"R={signal.RMultiple:F1}, risk={recommendation.RiskAssessment}";
    }

    private static bool ShouldExecuteTrade(TradeSignal signal, ExecutionRecommendation recommendation)
    {
        // Don't execute if EV is negative
        if (recommendation.ExpectedValue < 0)
            return false;

        // Don't execute if slippage too high relative to expected R
        var maxAcceptableSlippage = signal.RMultiple * 2500; // 25% of R multiple in bps
        if (recommendation.PredictedSlippageBps > maxAcceptableSlippage)
            return false;

        // Don't execute if execution risk is extreme
        if (recommendation.RiskAssessment == ExecutionRisk.Extreme)
            return false;

        return true;
    }
}

/// <summary>
/// Execution router interface
/// </summary>
public interface IExecutionRouter
{
    Task<ExecutionDecision> RouteOrderAsync(
        TradeSignal signal,
        MarketContext marketContext,
        CancellationToken ct = default);

    Task UpdateExecutionResultAsync(
        string signalId,
        ExecutionDecision originalDecision,
        ExecutionResult actualResult,
        CancellationToken ct = default);
}

/// <summary>
/// Trade signal for execution routing
/// </summary>
public record TradeSignal
{
    public string SignalId { get; init; } = "";
    public string Symbol { get; init; } = "";
    public TradeSide Side { get; init; }
    public decimal Quantity { get; init; }
    public decimal EntryPrice { get; init; }
    public decimal StopPrice { get; init; }
    public decimal TargetPrice { get; init; }
    public decimal RMultiple { get; init; }
    public decimal Confidence { get; init; }
    public decimal? MetaWinProbability { get; init; } // From meta-labeler
    public DateTime Timestamp { get; init; }
}

/// <summary>
/// Market context for execution decisions
/// </summary>
public record MarketContext
{
    public string Symbol { get; init; } = "";
    public decimal BidAskSpread { get; init; }
    public decimal SpreadBps { get; init; }
    public decimal Volatility { get; init; }
    public decimal Volume { get; init; }
    public bool IsVolatile { get; init; }
    public bool IsLiquid { get; init; }
    public MarketSession Session { get; init; }
}

/// <summary>
/// Execution decision from router
/// </summary>
public record ExecutionDecision
{
    public OrderType OrderType { get; init; }
    public decimal? LimitPrice { get; init; }
    public decimal ExpectedSlippageBps { get; init; }
    public decimal FillProbability { get; init; }
    public decimal ExpectedValue { get; init; }
    public string Reasoning { get; init; } = "";
    public ExecutionRisk RiskLevel { get; init; }
    public TimeSpan EstimatedFillTime { get; init; }
    public bool ShouldExecute { get; init; }
}

/// <summary>
/// Actual execution result for learning
/// </summary>
public record ExecutionResult
{
    public string SignalId { get; init; } = "";
    public DateTime ExecutionTime { get; init; }
    public decimal FillPrice { get; init; }
    public decimal ActualSlippageBps { get; init; }
    public bool WasFilled { get; init; }
    public TimeSpan FillTime { get; init; }
    public string ExecutionVenue { get; init; } = "";
}

/// <summary>
/// Trade side enumeration
/// </summary>
public enum TradeSide
{
    Buy,
    Sell
}

/// <summary>
/// Tracks execution costs for learning and improvement
/// </summary>
public interface IExecutionCostTracker
{
    Task RecordExecutionAsync(
        string signalId,
        ExecutionDecision decision,
        ExecutionResult result,
        CancellationToken ct = default);

    Task AnalyzePredictionErrorAsync(
        string signalId,
        ExecutionDecision decision,
        ExecutionResult result,
        CancellationToken ct = default);

    Task<ExecutionStatistics> GetExecutionStatsAsync(
        string? symbol = null,
        TimeSpan? lookback = null,
        CancellationToken ct = default);
}

/// <summary>
/// Execution performance statistics
/// </summary>
public record ExecutionStatistics
{
    public int TotalExecutions { get; init; }
    public decimal AverageSlippageBps { get; init; }
    public decimal SlippagePredictionError { get; init; }
    public decimal FillRate { get; init; }
    public decimal CostSavingsBps { get; init; } // vs always using market orders
    public Dictionary<OrderType, decimal> CostByOrderType { get; init; } = new();
}
