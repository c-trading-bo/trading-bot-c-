using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.Models;
using BotCore.Services;

namespace BotCore.Integrations
{
    /// <summary>
    /// Enhanced strategy integration that automatically collects training data from all strategy executions.
    /// Bridges the existing strategy system with the new RL training data collection.
    /// </summary>
    public static class EnhancedStrategyIntegration
    {
        /// <summary>
        /// Enhance a strategy signal with comprehensive data collection for RL training.
        /// Call this whenever a strategy generates a signal.
        /// </summary>
        public static async Task<string?> CollectSignalDataAsync(
            ILogger logger,
            IEnhancedTrainingDataService trainingService,
            StrategySignal signal,
            Bar currentBar,
            MarketSnapshot snapshot)
        {
            try
            {
                // Convert strategy signal to training data format
                var signalData = ConvertToTrainingSignalData(signal, currentBar, snapshot);

                // Record the trade data for RL training
                var tradeId = await trainingService.RecordTradeAsync(signalData);

                logger.LogDebug("[EnhancedIntegration] Collected signal data for {Strategy}: {TradeId}",
                    signal.Strategy, tradeId);

                return tradeId;
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "[EnhancedIntegration] Failed to collect signal data for {Strategy}",
                    signal.Strategy);
                return null;
            }
        }

        /// <summary>
        /// Record trade outcome for RL training when a trade closes.
        /// Call this whenever a trade is closed/exited.
        /// </summary>
        public static async Task RecordTradeOutcomeAsync(
            ILogger logger,
            IEnhancedTrainingDataService trainingService,
            string tradeId,
            decimal entryPrice,
            decimal exitPrice,
            decimal pnl,
            bool isWin,
            DateTime exitTime,
            TimeSpan holdingTime)
        {
            try
            {
                var outcomeData = new TradeOutcomeData
                {
                    IsWin = isWin,
                    ActualPnl = pnl,
                    ExitPrice = exitPrice,
                    ExitTime = exitTime,
                    HoldingTimeMinutes = (decimal)holdingTime.TotalMinutes,
                    ActualRMultiple = CalculateRMultiple(entryPrice, exitPrice, pnl, isWin),
                    MaxDrawdown = Math.Min(0, pnl) // Simplified - could be enhanced with real-time tracking
                };

                await trainingService.RecordTradeResultAsync(tradeId, outcomeData);

                logger.LogDebug("[EnhancedIntegration] Recorded trade outcome for {TradeId}: {Result} (R={RMultiple:F2})",
                    tradeId, isWin ? "WIN" : "LOSS", outcomeData.ActualRMultiple);
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "[EnhancedIntegration] Failed to record trade outcome for {TradeId}", tradeId);
            }
        }

        /// <summary>
        /// Enhanced signal processing that integrates with both existing collectors and new training system.
        /// Call this as part of the strategy execution pipeline.
        /// </summary>
        public static async Task<EnhancedSignalResult> ProcessSignalWithDataCollectionAsync(
            ILogger logger,
            IEnhancedTrainingDataService trainingService,
            StrategySignal signal,
            Bar currentBar,
            MarketSnapshot snapshot)
        {
            var result = new EnhancedSignalResult
            {
                OriginalSignal = signal,
                TradeId = null,
                Success = false
            };

            try
            {
                // Process with enhanced training data collection
                result.TradeId = await CollectSignalDataAsync(logger, trainingService, signal, currentBar, snapshot);
                result.Success = !string.IsNullOrEmpty(result.TradeId);

                logger.LogInformation("[EnhancedIntegration] Processed signal for {Strategy} - TradeId: {TradeId}",
                    signal.Strategy, result.TradeId ?? "N/A");

                return result;
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "[EnhancedIntegration] Failed to process signal with data collection");
                return result;
            }
        }

        private static TradeSignalData ConvertToTrainingSignalData(StrategySignal signal, Bar currentBar, MarketSnapshot snapshot)
        {
            return new TradeSignalData
            {
                Id = signal.ClientOrderId ?? Guid.NewGuid().ToString(),
                Symbol = signal.Symbol,
                Direction = DetermineDirection(signal),
                Entry = signal.LimitPrice ?? currentBar.Close,
                Size = signal.Size,
                Strategy = signal.Strategy,
                StopLoss = currentBar.Close * 0.99m, // Simplified stop loss
                TakeProfit = currentBar.Close * 1.02m, // Simplified take profit
                Regime = DetermineMarketRegime(currentBar, snapshot),
                Atr = CalculateAtr(currentBar),
                Rsi = CalculateRsi(currentBar),
                Ema20 = currentBar.Close, // Simplified - could use actual EMA
                Ema50 = currentBar.Close, // Simplified - could use actual EMA
                Momentum = CalculateMomentum(currentBar),
                TrendStrength = CalculateTrendStrength(currentBar),
                VixLevel = 20m // Default VIX level - could be fetched from market data
            };
        }

        private static string DetermineDirection(StrategySignal signal)
        {
            if (signal.Side == SignalSide.Long) return "BUY";
            if (signal.Side == SignalSide.Short) return "SELL";
            return "HOLD";
        }

        private static string DetermineMarketRegime(Bar currentBar, MarketSnapshot snapshot)
        {
            var range = currentBar.High - currentBar.Low;
            var price = currentBar.Close;

            if (range / price > 0.01m) return "HighVol";
            if (range / price < 0.005m) return "LowVol";
            return "Range";
        }

        private static decimal CalculateRMultiple(decimal entryPrice, decimal exitPrice, decimal pnl, bool isWin)
        {
            if (entryPrice == 0) return 0;

            var priceMove = Math.Abs(exitPrice - entryPrice);
            var percentMove = priceMove / entryPrice;

            // Estimate R-multiple based on typical risk parameters
            var estimatedRisk = entryPrice * 0.01m; // 1% risk assumption
            var rMultiple = pnl / estimatedRisk;

            return rMultiple;
        }

        // Helper calculation methods (simplified - could be enhanced with actual technical analysis)
        private static decimal CalculateAtr(Bar bar) => (bar.High - bar.Low);
        private static decimal CalculateRsi(Bar bar) => 50m; // Neutral RSI - could be calculated from price history
        private static decimal CalculateMomentum(Bar bar) => (bar.Close - bar.Open) / bar.Open;
        private static decimal CalculateTrendStrength(Bar bar) => Math.Abs(bar.Close - bar.Open) / bar.Open;
        private static decimal CalculateVolume(Bar bar) => (decimal)bar.Volume;
    }

    /// <summary>
    /// Result of enhanced signal processing with data collection
    /// </summary>
    public class EnhancedSignalResult
    {
        public StrategySignal OriginalSignal { get; set; } = null!;
        public string? TradeId { get; set; }
        public bool Success { get; set; }
    }
}