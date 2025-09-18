using Microsoft.Extensions.Logging;
using BotCore.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Strategy
{
    /// <summary>
    /// Integration layer that connects S1-S14 strategies with ML/RL data collection.
    /// Maps strategy outcomes to the 4 ML strategy types and logs comprehensive features.
    /// </summary>
    public static class StrategyMlIntegration
    {
        /// <summary>
        /// Maps S1-S14 strategy IDs to the 4 ML strategy types for data collection
        /// </summary>
        private static readonly Dictionary<string, MultiStrategyRlCollector.StrategyType> StrategyTypeMapping = new()
        {
            // EMA Cross strategies
            ["S1"] = MultiStrategyRlCollector.StrategyType.EmaCross,
            ["S2"] = MultiStrategyRlCollector.StrategyType.EmaCross,
            ["S3"] = MultiStrategyRlCollector.StrategyType.EmaCross,

            // Mean Reversion strategies  
            ["S4"] = MultiStrategyRlCollector.StrategyType.MeanReversion,
            ["S5"] = MultiStrategyRlCollector.StrategyType.MeanReversion,
            ["S6"] = MultiStrategyRlCollector.StrategyType.MeanReversion,
            ["S7"] = MultiStrategyRlCollector.StrategyType.MeanReversion,

            // Breakout strategies
            ["S8"] = MultiStrategyRlCollector.StrategyType.Breakout,
            ["S9"] = MultiStrategyRlCollector.StrategyType.Breakout,
            ["S10"] = MultiStrategyRlCollector.StrategyType.Breakout,
            ["S11"] = MultiStrategyRlCollector.StrategyType.Breakout,

            // Momentum strategies
            ["S12"] = MultiStrategyRlCollector.StrategyType.Momentum,
            ["S13"] = MultiStrategyRlCollector.StrategyType.Momentum,
            ["S14"] = MultiStrategyRlCollector.StrategyType.Momentum
        };

        /// <summary>
        /// Log comprehensive features for a strategy signal to enable ML training
        /// </summary>
        public static void LogStrategySignal(
            ILogger logger,
            string strategyId,
            string symbol,
            Side side,
            decimal entry,
            decimal stop,
            decimal target,
            decimal score,
            decimal qScore,
            IList<Bar> bars,
            string? customSignalId = null)
        {
            try
            {
                if (!StrategyTypeMapping.TryGetValue(strategyId, out var strategyType))
                {
                    // Default to breakout if strategy not mapped
                    strategyType = MultiStrategyRlCollector.StrategyType.Breakout;
                }

                var signalId = customSignalId ?? $"{strategyId}-{symbol}-{DateTime.UtcNow:yyyyMMddHHmmss}";

                // Create comprehensive features based on strategy type
                var features = strategyType switch
                {
                    MultiStrategyRlCollector.StrategyType.EmaCross => CreateEmaCrossFeatures(signalId, symbol, entry, bars, score, qScore),
                    MultiStrategyRlCollector.StrategyType.MeanReversion => CreateMeanReversionFeatures(signalId, symbol, entry, bars, score, qScore),
                    MultiStrategyRlCollector.StrategyType.Breakout => CreateBreakoutFeatures(signalId, symbol, entry, bars, score, qScore),
                    MultiStrategyRlCollector.StrategyType.Momentum => CreateMomentumFeatures(signalId, symbol, entry, bars, score, qScore),
                    _ => CreateDefaultFeatures(signalId, symbol, entry, bars, score, qScore, strategyType)
                };

                // Calculate risk metrics
                features.RiskPerShare = Math.Abs(entry - stop);
                features.RewardRisk = Math.Abs(target - entry) / Math.Max(features.RiskPerShare, 0.01m);
                features.Side = side.ToString();
                features.StrategyId = strategyId;

                // Log to RL collector
                MultiStrategyRlCollector.LogComprehensiveFeatures(logger, features);

                logger.LogDebug("[ML-Integration] Logged {StrategyType} features for {StrategyId} signal {SignalId}",
                    strategyType, strategyId, signalId);
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "[ML-Integration] Failed to log strategy signal for {StrategyId}", strategyId);
            }
        }

        /// <summary>
        /// Log trade outcome for ML feedback loop
        /// </summary>
        public static void LogTradeOutcome(
            ILogger logger,
            string strategyId,
            string signalId,
            decimal actualPnl,
            decimal actualRMultiple,
            bool isWin,
            DateTime exitTime,
            string exitReason)
        {
            try
            {
                if (!StrategyTypeMapping.TryGetValue(strategyId, out var strategyType))
                {
                    strategyType = MultiStrategyRlCollector.StrategyType.Breakout;
                }

                var outcome = new MultiStrategyRlCollector.EnhancedTradeOutcome
                {
                    SignalId = signalId,
                    Strategy = strategyType,
                    ExitTime = exitTime,
                    ActualPnl = actualPnl,
                    ActualRMultiple = actualRMultiple,
                    IsWin = isWin,
                    ExitReason = exitReason,
                    HoldingTimeMinutes = (int)(exitTime - DateTime.UtcNow).TotalMinutes // Approximate
                };

                MultiStrategyRlCollector.LogTradeOutcome(logger, outcome);

                logger.LogDebug("[ML-Integration] Logged trade outcome for {StrategyId} signal {SignalId}: {Result}",
                    strategyId, signalId, isWin ? "WIN" : "LOSS");
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "[ML-Integration] Failed to log trade outcome for {StrategyId}", strategyId);
            }
        }

        private static MultiStrategyRlCollector.ComprehensiveFeatures CreateEmaCrossFeatures(
            string signalId, string symbol, decimal price, IList<Bar> bars, decimal score, decimal qScore)
        {
            // Calculate EMA values if we have enough bars
            decimal ema9 = bars.Count >= 9 ? CalculateEma(bars, 9) : price;
            decimal ema20 = bars.Count >= 20 ? CalculateEma(bars, 20) : price;
            decimal ema50 = bars.Count >= 50 ? CalculateEma(bars, 50) : price;
            decimal volume = bars.Any() ? bars.Last().Volume : 1000m;
            decimal atr = bars.Count >= 14 ? CalculateAtr(bars, 14) : 1m;

            var features = MultiStrategyRlCollector.CreateEmaCrossFeatures(signalId, symbol, price, ema9, ema20, ema50, volume, atr);

            if (bars.Any())
            {
                var latest = bars.Last();
                features.Volume = latest.Volume;
                features.Price = latest.Close;
                features.DailyRange = latest.High - latest.Low;

                // Additional calculations
                features.Ema8 = CalculateEma(bars, 8);
                features.Ema21 = CalculateEma(bars, 21);
                features.MaAlignment = (features.Ema8 - features.Ema21) / Math.Max(features.Ema21, 0.01m);
            }

            features.Score = score;
            features.QScore = qScore;
            return features;
        }

        private static MultiStrategyRlCollector.ComprehensiveFeatures CreateMeanReversionFeatures(
            string signalId, string symbol, decimal price, IList<Bar> bars, decimal score, decimal qScore)
        {
            decimal rsi = bars.Count >= 14 ? CalculateRsi(bars, 14) : 50m;
            decimal bbUpper = 0m, bbLower = 0m, vwap = price;

            if (bars.Count >= 20)
            {
                var closes = bars.TakeLast(20).Select(b => b.Close).ToList();
                var sma = closes.Average();
                var stdDev = (decimal)Math.Sqrt((double)closes.Select(c => (c - sma) * (c - sma)).Average());
                bbUpper = sma + (2 * stdDev);
                bbLower = sma - (2 * stdDev);
                vwap = bars.Sum(b => b.Close * b.Volume) / bars.Sum(b => b.Volume);
            }

            decimal volume = bars.Any() ? bars.Last().Volume : 1000m;

            var features = MultiStrategyRlCollector.CreateMeanReversionFeatures(signalId, symbol, price, rsi, bbUpper, bbLower, vwap, volume);

            if (bars.Any())
            {
                var latest = bars.Last();
                features.Volume = latest.Volume;
                features.Price = latest.Close;
                features.DistanceFromVwap = (price - vwap) / Math.Max(vwap, 0.01m);
            }

            features.Score = score;
            features.QScore = qScore;
            return features;
        }

        private static MultiStrategyRlCollector.ComprehensiveFeatures CreateBreakoutFeatures(
            string signalId, string symbol, decimal price, IList<Bar> bars, decimal score, decimal qScore)
        {
            decimal highBreakout = 0m, lowBreakout = 0m, volume = 1000m, avgVolume = 1000m, consolidationTime;

            if (bars.Count >= 20)
            {
                var recent = bars.TakeLast(20).ToList();
                highBreakout = recent.Max(b => b.High);
                lowBreakout = recent.Min(b => b.Low);
                volume = bars.Last().Volume;
                avgVolume = (decimal)recent.Average(b => b.Volume);
                consolidationTime = 20m; // Simplified
            }

            var features = MultiStrategyRlCollector.CreateBreakoutFeatures(signalId, symbol, price, highBreakout, lowBreakout, volume, avgVolume, consolidationTime);

            if (bars.Any())
            {
                var latest = bars.Last();
                features.Volume = latest.Volume;
                features.Price = latest.Close;
            }

            features.Score = score;
            features.QScore = qScore;
            return features;
        }

        private static MultiStrategyRlCollector.ComprehensiveFeatures CreateMomentumFeatures(
            string signalId, string symbol, decimal price, IList<Bar> bars, decimal score, decimal qScore)
        {
            decimal priorPrice = bars.Count >= 2 ? bars[bars.Count - 2].Close : price;
            decimal rsi = bars.Count >= 14 ? CalculateRsi(bars, 14) : 50m;
            decimal volume = bars.Any() ? bars.Last().Volume : 1000m;
            decimal atr = bars.Count >= 14 ? CalculateAtr(bars, 14) : 1m;

            var features = MultiStrategyRlCollector.CreateMomentumFeatures(signalId, symbol, price, priorPrice, rsi, volume, atr);

            if (bars.Any())
            {
                var latest = bars.Last();
                features.Volume = latest.Volume;
                features.Price = latest.Close;

                // Additional momentum calculations
                features.MomentumStrength = CalculateMomentumStrength(bars);
            }

            features.Score = score;
            features.QScore = qScore;
            return features;
        }

        private static MultiStrategyRlCollector.ComprehensiveFeatures CreateDefaultFeatures(
            string signalId, string symbol, decimal price, IList<Bar> bars, decimal score, decimal qScore,
            MultiStrategyRlCollector.StrategyType strategyType)
        {
            var features = MultiStrategyRlCollector.CreateBaseFeatures(signalId, symbol, strategyType, price);

            if (bars.Any())
            {
                var latest = bars.Last();
                features.Volume = latest.Volume;
                features.Price = latest.Close;
                features.DailyRange = latest.High - latest.Low;
            }

            features.Score = score;
            features.QScore = qScore;
            return features;
        }

        #region Technical Indicator Calculations

        private static decimal CalculateEma(IList<Bar> bars, int period)
        {
            if (bars.Count < period) return bars.Last().Close;

            var multiplier = 2m / (period + 1);
            var ema = bars[0].Close;

            for (int i = 1; i < bars.Count; i++)
            {
                ema = (bars[i].Close * multiplier) + (ema * (1 - multiplier));
            }

            return ema;
        }

        private static decimal CalculateRsi(IList<Bar> bars, int period)
        {
            if (bars.Count < period + 1) return 50m;

            var gains;
            var losses;

            for (int i = bars.Count - period; i < bars.Count; i++)
            {
                var change = bars[i].Close - bars[i - 1].Close;
                if (change > 0) gains += change;
                else losses -= change;
            }

            var avgGain = gains / period;
            var avgLoss = losses / period;

            if (avgLoss == 0) return 100m;

            var rs = avgGain / avgLoss;
            return 100m - (100m / (1 + rs));
        }

        private static decimal CalculateAtr(IList<Bar> bars, int period)
        {
            if (bars.Count < period + 1) return 1m;

            var trs = new List<decimal>();
            for (int i = 1; i < bars.Count; i++)
            {
                var tr = Math.Max(bars[i].High - bars[i].Low,
                    Math.Max(Math.Abs(bars[i].High - bars[i - 1].Close),
                        Math.Abs(bars[i].Low - bars[i - 1].Close)));
                trs.Add(tr);
            }

            return trs.TakeLast(period).Average();
        }

        private static decimal CalculateVolumeRatio(IList<Bar> bars, int period)
        {
            if (bars.Count < period) return 1m;

            var latest = bars.Last().Volume;
            var average = (decimal)bars.TakeLast(period).Average(b => b.Volume);

            return average > 0 ? latest / average : 1m;
        }

        private static decimal CalculateBollingerPosition(IList<Bar> bars, int period)
        {
            if (bars.Count < period) return 0.5m;

            var closes = bars.TakeLast(period).Select(b => b.Close).ToList();
            var sma = closes.Average();
            var stdDev = (decimal)Math.Sqrt((double)closes.Select(c => (c - sma) * (c - sma)).Average());

            var latest = bars.Last().Close;
            var upperBand = sma + (2 * stdDev);
            var lowerBand = sma - (2 * stdDev);

            if (upperBand == lowerBand) return 0.5m;

            return (latest - lowerBand) / (upperBand - lowerBand);
        }

        private static decimal CalculateVwapDistance(IList<Bar> bars)
        {
            if (bars.Count < 2) return 0m;

            var vwap = bars.Sum(b => b.Close * b.Volume) / bars.Sum(b => b.Volume);
            var latest = bars.Last().Close;

            return vwap > 0 ? (latest - vwap) / vwap : 0m;
        }

        private static decimal CalculateBreakoutStrength(IList<Bar> bars)
        {
            if (bars.Count < 20) return 0m;

            var recent = bars.TakeLast(20).ToList();
            var highest = recent.Max(b => b.High);
            var lowest = recent.Min(b => b.Low);
            var latest = bars.Last().Close;

            if (highest == lowest) return 0m;

            return (latest - lowest) / (highest - lowest);
        }

        private static decimal CalculateRateOfChange(IList<Bar> bars, int period)
        {
            if (bars.Count < period + 1) return 0m;

            var current = bars.Last().Close;
            var previous = bars[bars.Count - period - 1].Close;

            return previous > 0 ? (current - previous) / previous : 0m;
        }

        private static decimal CalculateMomentumStrength(IList<Bar> bars)
        {
            if (bars.Count < 10) return 0m;

            var recent = bars.TakeLast(10).ToList();
            var upBars = recent.Count(b => b.Close > b.Open);

            return upBars / 10m;
        }

        /// <summary>
        /// Get the ML strategy type for a given strategy ID (S1-S14)
        /// </summary>
        public static MultiStrategyRlCollector.StrategyType GetStrategyType(string strategyId)
        {
            return StrategyTypeMapping.TryGetValue(strategyId, out var strategyType)
                ? strategyType
                : MultiStrategyRlCollector.StrategyType.Breakout; // Default fallback
        }

        #endregion
    }
}