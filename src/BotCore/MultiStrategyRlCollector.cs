using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Linq;
using Microsoft.Extensions.Logging;
using BotCore.Services;

namespace BotCore
{
    /// <summary>
    /// Enhanced training data collector supporting all 4 active strategies.
    /// Captures comprehensive features from EmaCross, MeanReversion, Breakout, and Momentum.
    /// </summary>
    public static class MultiStrategyRlCollector
    {
        public enum StrategyType
        {
            EmaCross,
            MeanReversion,
            Breakout,
            Momentum
        }

        public enum MarketRegime
        {
            Range,
            Trend,
            HighVol,
            LowVol,
            Choppy,
            Squeeze
        }

        public class ComprehensiveFeatures
        {
            public DateTime Timestamp { get; set; }
            public string Symbol { get; set; } = "";
            public StrategyType Strategy { get; set; }
            public string SignalId { get; set; } = "";
            public string Session { get; set; } = ""; // RTH, ETH
            public MarketRegime Regime { get; set; }

            // === Core Price Action ===
            public decimal Price { get; set; }
            public decimal Atr14 { get; set; }
            public decimal Atr50 { get; set; }
            public decimal DailyRange { get; set; }
            public decimal Gap { get; set; }

            // === Moving Averages ===
            public decimal Ema9 { get; set; }
            public decimal Ema20 { get; set; }
            public decimal Ema50 { get; set; }
            public decimal Sma200 { get; set; }
            public decimal MaAlignment { get; set; } // -1 to 1 (bearish to bullish)

            // === Technical Indicators ===
            public decimal Rsi14 { get; set; }
            public decimal Rsi2 { get; set; }
            public decimal StochK { get; set; }
            public decimal StochD { get; set; }
            public decimal Williams { get; set; }
            public decimal Cci { get; set; }

            // === Volatility & Bollinger ===
            public decimal BbUpper { get; set; }
            public decimal BbLower { get; set; }
            public decimal BbPosition { get; set; } // 0-1 within bands
            public decimal BbSqueeze { get; set; } // Keltner vs BB
            public decimal HistVol20 { get; set; }
            public decimal VolumeProfile { get; set; }

            // === Market Microstructure ===
            public decimal Volume { get; set; }
            public decimal VolumeRatio { get; set; } // vs 20-day avg
            public decimal Spread { get; set; }
            public decimal BidAskImbalance { get; set; }
            public decimal OrderBookImbalance { get; set; }
            public decimal TickDirection { get; set; }
            public decimal VwapDistance { get; set; }

            // === Strategy-Specific Signals ===
            // EmaCross
            public decimal EmaSpread920 { get; set; }
            public decimal EmaSlope9 { get; set; }
            public decimal EmaSlope20 { get; set; }
            public decimal CrossStrength { get; set; }

            // MeanReversion
            public decimal MeanReversionZ { get; set; }
            public decimal BounceQuality { get; set; }
            public decimal SupportResistance { get; set; }
            public decimal OversoldDuration { get; set; }

            // Breakout
            public decimal BreakoutStrength { get; set; }
            public decimal ConsolidationTime { get; set; }
            public decimal VolumeConfirmation { get; set; }
            public decimal FalseBreakoutRisk { get; set; }

            // Momentum
            public decimal MomentumScore { get; set; }
            public decimal AccelerationDivergence { get; set; }
            public decimal TrendStrength { get; set; }
            public decimal MomentumSustainability { get; set; }

            // === Risk & Performance Factors ===
            public decimal SignalQuality { get; set; } // 0-1 composite score
            public decimal PriorWinRate { get; set; } // last 20 trades this strategy
            public decimal AvgRMultiple { get; set; } // recent performance
            public decimal DrawdownRisk { get; set; }
            public decimal CorrelationRisk { get; set; } // with other positions
            public decimal LiquidityScore { get; set; }
            public decimal NewsImpact { get; set; } // 0-1 upcoming events

            // === Time-Based Features ===
            public int DayOfWeek { get; set; }
            public int HourOfDay { get; set; }
            public int MinutesFromOpen { get; set; }
            public int MinutesToClose { get; set; }
            public bool IsOpex { get; set; }
            public bool IsFomc { get; set; }
            public bool IsEarnings { get; set; }

            // === Position Sizing Targets ===
            public decimal BaselineMultiplier { get; set; } = 1.0m;
            public decimal OptimalMultiplier { get; set; } = 1.0m; // ML predicted
            public decimal ActualMultiplier { get; set; } = 1.0m; // What was used
            public decimal ConfidenceScore { get; set; } = 0.5m;

            // === Additional Integration Properties ===
            public decimal RiskPerShare { get; set; }
            public decimal RewardRisk { get; set; }
            public string Side { get; set; } = "";
            public string StrategyId { get; set; } = "";
            public decimal Score { get; set; }
            public decimal QScore { get; set; }
            public decimal Ema8 { get; set; }
            public decimal Ema21 { get; set; }
            public decimal MomentumStrength { get; set; }
            public decimal DistanceFromVwap { get; set; }
        }

        public class EnhancedTradeOutcome
        {
            public string SignalId { get; set; } = "";
            public StrategyType Strategy { get; set; }
            public DateTime EntryTime { get; set; }
            public DateTime? ExitTime { get; set; }
            public decimal EntryPrice { get; set; }
            public decimal? ExitPrice { get; set; }
            public decimal StopPrice { get; set; }
            public decimal TargetPrice { get; set; }
            public decimal ActualRMultiple { get; set; }
            public decimal MaxFavorableExcursion { get; set; }
            public decimal MaxAdverseExcursion { get; set; }
            public decimal SlippageTicks { get; set; }
            public decimal CommissionCost { get; set; }
            public int HoldingTimeBars { get; set; }
            public bool IsWin { get; set; }
            public bool IsCompleted { get; set; }
            public string ExitReason { get; set; } = ""; // Stop, Target, Manual, Timeout, RiskReduction
            public decimal PostTradeDrawdown { get; set; }
            public decimal PositionSizeUsed { get; set; }
            public decimal VolatilityAdjustment { get; set; }

            // === Additional Integration Properties ===
            public decimal ActualPnl { get; set; }
            public int HoldingTimeMinutes { get; set; }
        }

        private static readonly string DataPath = Path.Combine("data", "rl_training");
        private static readonly object FileLock = new();
        private static readonly Dictionary<StrategyType, List<EnhancedTradeOutcome>> RecentTrades = new();

        static MultiStrategyRlCollector()
        {
            Directory.CreateDirectory(DataPath);
            foreach (StrategyType strategy in Enum.GetValues<StrategyType>())
            {
                RecentTrades[strategy] = new List<EnhancedTradeOutcome>();
            }
        }

        /// <summary>
        /// Collect comprehensive features for any of the 4 strategies
        /// </summary>
        public static void LogComprehensiveFeatures(ILogger log, ComprehensiveFeatures features)
        {
            if (features is null) throw new ArgumentNullException(nameof(features));
            
            try
            {
                // Enrich with calculated fields
                EnrichFeatures(features);

                var json = JsonSerializer.Serialize(features, new JsonSerializerOptions
                {
                    WriteIndented = false,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });

                var fileName = $"features_{features.Strategy.ToString().ToLower()}_{DateTime.UtcNow:yyyyMMdd}.jsonl";
                var filePath = Path.Combine(DataPath, fileName);

                lock (FileLock)
                {
                    File.AppendAllText(filePath, json + Environment.NewLine);
                }

                log.LogDebug("[RL-{Strategy}] Logged comprehensive features for signal {SignalId}",
                    features.Strategy, features.SignalId);
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[RL-{Strategy}] Failed to log comprehensive features", features.Strategy);
            }
        }

        /// <summary>
        /// Log enhanced trade outcome with full performance metrics
        /// </summary>
        public static void LogEnhancedOutcome(ILogger log, EnhancedTradeOutcome outcome)
        {
            if (outcome is null) throw new ArgumentNullException(nameof(outcome));
            
            try
            {
                var json = JsonSerializer.Serialize(outcome, new JsonSerializerOptions
                {
                    WriteIndented = false,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });

                var fileName = $"outcomes_{outcome.Strategy.ToString().ToLower()}_{DateTime.UtcNow:yyyyMMdd}.jsonl";
                var filePath = Path.Combine(DataPath, fileName);

                lock (FileLock)
                {
                    File.AppendAllText(filePath, json + Environment.NewLine);

                    // Update recent trades cache for performance metrics
                    UpdateRecentTradesCache(outcome);
                }

                log.LogDebug("[RL-{Strategy}] Logged enhanced outcome for {SignalId}: R={R:F2} Exit={Exit}",
                    outcome.Strategy, outcome.SignalId, outcome.ActualRMultiple, outcome.ExitReason);
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[RL-{Strategy}] Failed to log enhanced outcome", outcome.Strategy);
            }
        }

        /// <summary>
        /// Create features specific to EmaCross strategy
        /// </summary>
        public static ComprehensiveFeatures CreateEmaCrossFeatures(
            string signalId,
            string symbol,
            decimal price,
            decimal ema9,
            decimal ema20,
            decimal ema50,
            decimal volume,
            decimal atr)
        {
            var features = CreateBaseFeatures(signalId, symbol, StrategyType.EmaCross, price);

            features.Ema9 = ema9;
            features.Ema20 = ema20;
            features.Ema50 = ema50;
            features.Volume = volume;
            features.Atr14 = atr;

            // EmaCross specific calculations
            features.EmaSpread920 = (ema9 - ema20) / price * 100m; // % spread
            features.EmaSlope9 = CalculateSlope(ema9, price); // Simplified slope
            features.EmaSlope20 = CalculateSlope(ema20, price);
            features.CrossStrength = Math.Abs(features.EmaSpread920) * (volume / 1000m);
            features.MaAlignment = CalculateMaAlignment(ema9, ema20, ema50);

            return features;
        }

        /// <summary>
        /// Create features specific to MeanReversion strategy
        /// </summary>
        public static ComprehensiveFeatures CreateMeanReversionFeatures(
            string signalId,
            string symbol,
            decimal price,
            decimal rsi,
            decimal bbUpper,
            decimal bbLower,
            decimal vwap,
            decimal volume)
        {
            var features = CreateBaseFeatures(signalId, symbol, StrategyType.MeanReversion, price);

            features.Rsi14 = rsi;
            features.BbUpper = bbUpper;
            features.BbLower = bbLower;
            features.VwapDistance = (price - vwap) / price * 100m;
            features.Volume = volume;

            // MeanReversion specific
            features.BbPosition = (price - bbLower) / (bbUpper - bbLower);
            features.MeanReversionZ = (50m - rsi) / 10m; // Z-score from neutral
            features.BounceQuality = CalculateBounceQuality(price, bbLower, bbUpper);
            features.OversoldDuration = rsi < 30 ? 1m : 0m; // Simplified

            return features;
        }

        /// <summary>
        /// Create features specific to Breakout strategy
        /// </summary>
        public static ComprehensiveFeatures CreateBreakoutFeatures(
            string signalId,
            string symbol,
            decimal price,
            decimal highBreakout,
            decimal lowBreakout,
            decimal volume,
            decimal avgVolume,
            decimal consolidationTime)
        {
            var features = CreateBaseFeatures(signalId, symbol, StrategyType.Breakout, price);

            features.Volume = volume;
            features.VolumeRatio = volume / Math.Max(avgVolume, 1m);
            features.ConsolidationTime = consolidationTime;

            // Breakout specific
            features.BreakoutStrength = Math.Max(
                (price - highBreakout) / price * 100m,
                (lowBreakout - price) / price * 100m
            );
            features.VolumeConfirmation = Math.Min(features.VolumeRatio / 2m, 1m);
            features.FalseBreakoutRisk = CalculateFalseBreakoutRisk(price, highBreakout, lowBreakout);

            return features;
        }

        /// <summary>
        /// Create features specific to Momentum strategy
        /// </summary>
        public static ComprehensiveFeatures CreateMomentumFeatures(
            string signalId,
            string symbol,
            decimal price,
            decimal priorPrice,
            decimal rsi,
            decimal volume,
            decimal atr)
        {
            var features = CreateBaseFeatures(signalId, symbol, StrategyType.Momentum, price);

            features.Rsi14 = rsi;
            features.Volume = volume;
            features.Atr14 = atr;

            // Momentum specific
            var priceChange = (price - priorPrice) / priorPrice * 100m;
            features.MomentumScore = Math.Abs(priceChange) * (volume / 1000m);
            features.TrendStrength = Math.Min(Math.Abs(rsi - 50m) / 25m, 1m);
            features.AccelerationDivergence = 0; // Would need more bars to calculate
            features.MomentumSustainability = CalculateMomentumSustainability(priceChange, atr);

            return features;
        }

        /// <summary>
        /// Export training data optimized for specific strategy type
        /// </summary>
        public static async Task<string> ExportStrategyData(ILogger log, StrategyType strategy, DateTime? startDate = null)
        {
            var start = startDate ?? DateTime.UtcNow.AddDays(-30);
            var outputPath = Path.Combine(DataPath, $"{strategy.ToString().ToLower()}_training_{start:yyyyMMdd}.parquet");

            try
            {
                // Implement strategy-specific feature selection and export
                await ExportStrategySpecificFeaturesAsync(strategy, start, outputPath, log).ConfigureAwait(false);

                log.LogInformation("[RL-{Strategy}] Training data exported to {Path}", strategy, outputPath);
                return outputPath;
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[RL-{Strategy}] Failed to export training data", strategy);
                throw;
            }
        }

        private static async Task ExportStrategySpecificFeaturesAsync(StrategyType strategy, DateTime start, string outputPath, ILogger log)
        {
            try
            {
                // Read strategy-specific .jsonl files and optimize features for RL training
                var inputPattern = Path.Combine(DataPath, $"{strategy.ToString().ToLower()}_*.jsonl");
                var inputFiles = Directory.GetFiles(Path.GetDirectoryName(inputPattern) ?? DataPath, 
                    Path.GetFileName(inputPattern));

                var allFeatures = new List<ComprehensiveFeatures>();
                
                foreach (var file in inputFiles)
                {
                    var lines = await File.ReadAllLinesAsync(file).ConfigureAwait(false);
                    foreach (var line in lines)
                    {
                        if (string.IsNullOrWhiteSpace(line)) continue;
                        
                        try
                        {
                            var features = JsonSerializer.Deserialize<ComprehensiveFeatures>(line);
                            if (features?.Timestamp >= start)
                            {
                                allFeatures.Add(features);
                            }
                        }
                        catch (JsonException)
                        {
                            // Skip malformed JSON lines
                            continue;
                        }
                    }
                }

                // Write optimized features to parquet-style JSON for training
                if (allFeatures.Any())
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                    var json = JsonSerializer.Serialize(allFeatures, new JsonSerializerOptions { WriteIndented = true });
                    await File.WriteAllTextAsync(outputPath.Replace(".parquet", ".json"), json).ConfigureAwait(false);
                }
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[RL-{Strategy}] Failed to export strategy-specific features", strategy);
                throw;
            }
        }

        #region Private Helper Methods

        public static ComprehensiveFeatures CreateBaseFeatures(string signalId, string symbol, StrategyType strategy, decimal price)
        {
            return new ComprehensiveFeatures
            {
                Timestamp = DateTime.UtcNow,
                Symbol = symbol,
                Strategy = strategy,
                SignalId = signalId,
                Session = GetSessionType(),
                Price = price,
                DayOfWeek = (int)DateTime.UtcNow.DayOfWeek,
                HourOfDay = DateTime.UtcNow.Hour,
                MinutesFromOpen = CalculateMinutesFromOpen(),
                MinutesToClose = CalculateMinutesToClose()
            };
        }

        private static void EnrichFeatures(ComprehensiveFeatures features)
        {
            // Calculate performance metrics from recent trades
            if (RecentTrades.TryGetValue(features.Strategy, out var trades) && trades.Any())
            {
                var recent = trades.TakeLast(20).ToList();
                features.PriorWinRate = recent.Count(t => t.IsWin) / (decimal)recent.Count;
                features.AvgRMultiple = recent.Average(t => t.ActualRMultiple);
            }

            // Calculate signal quality composite score
            features.SignalQuality = CalculateSignalQuality(features);

            // Market regime detection
            features.Regime = DetectMarketRegime(features);
        }

        private static void UpdateRecentTradesCache(EnhancedTradeOutcome outcome)
        {
            if (RecentTrades.TryGetValue(outcome.Strategy, out var trades))
            {
                trades.Add(outcome);
                if (trades.Count > 50) // Keep last 50 trades
                {
                    trades.RemoveAt(0);
                }
            }
        }

        private static decimal CalculateSlope(decimal ma, decimal price)
        {
            return (ma - price) / price * 100m; // Simplified slope calculation
        }

        private static decimal CalculateMaAlignment(decimal ema9, decimal ema20, decimal ema50)
        {
            if (ema9 > ema20 && ema20 > ema50) return 1.0m; // Bullish alignment
            if (ema9 < ema20 && ema20 < ema50) return -1.0m; // Bearish alignment
            return 0m; // Mixed
        }

        private static decimal CalculateBounceQuality(decimal price, decimal bbLower, decimal bbUpper)
        {
            var range = bbUpper - bbLower;
            if (range == 0) return 0m;

            var distanceFromLower = price - bbLower;
            return Math.Min(distanceFromLower / (range * 0.2m), 1m); // Quality based on distance from lower band
        }

        private static decimal CalculateFalseBreakoutRisk(decimal price, decimal high, decimal low)
        {
            var range = high - low;
            if (range == 0) return 1m; // High risk if no range

            // Risk higher if breakout is marginal
            var breakoutDistance = Math.Max(price - high, low - price);
            return Math.Max(0m, 1m - (breakoutDistance / (range * 0.1m)));
        }

        private static decimal CalculateMomentumSustainability(decimal priceChange, decimal atr)
        {
            if (atr == 0) return 0m;
            return Math.Min(Math.Abs(priceChange) / (atr * 100m), 2m); // Momentum relative to volatility
        }

        private static decimal CalculateSignalQuality(ComprehensiveFeatures features)
        {
            // Composite score based on multiple factors
            var qualityFactors = new List<decimal>();

            // Volume confirmation
            if (features.Volume > 0)
                qualityFactors.Add(Math.Min(features.VolumeRatio / 1.5m, 1m));

            // Technical alignment
            qualityFactors.Add((Math.Abs(features.MaAlignment) + 1m) / 2m);

            // Prior performance
            qualityFactors.Add(features.PriorWinRate);

            return qualityFactors.Any() ? qualityFactors.Average() : 0.5m;
        }

        private static MarketRegime DetectMarketRegime(ComprehensiveFeatures features)
        {
            // Simple regime detection based on available indicators
            if (features.HistVol20 > 0.02m) return MarketRegime.HighVol;
            if (features.HistVol20 < 0.005m) return MarketRegime.LowVol;
            if (Math.Abs(features.MaAlignment) > 0.5m) return MarketRegime.Trend;
            if (features.BbSqueeze > 0.8m) return MarketRegime.Squeeze;
            if (features.Rsi14 > 40 && features.Rsi14 < 60) return MarketRegime.Range;

            return MarketRegime.Choppy;
        }

        private static string GetSessionType()
        {
            var et = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
                TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            var hour = et.Hour;

            return (hour >= 9 && hour < 16) ? "RTH" : "ETH";
        }

        private static int CalculateMinutesFromOpen()
        {
            var et = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
                TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));

            var marketOpen = new TimeSpan(9, 30, 0);
            var currentTime = et.TimeOfDay;

            if (currentTime >= marketOpen)
                return (int)(currentTime - marketOpen).TotalMinutes;

            return -1; // Before market open
        }

        private static int CalculateMinutesToClose()
        {
            var et = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
                TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));

            var marketClose = new TimeSpan(16, 0, 0);
            var currentTime = et.TimeOfDay;

            if (currentTime < marketClose)
                return (int)(marketClose - currentTime).TotalMinutes;

            return -1; // After market close
        }

        /// <summary>
        /// Log trade outcome for feedback loop and performance tracking
        /// </summary>
        public static void LogTradeOutcome(ILogger log, EnhancedTradeOutcome outcome)
        {
            try
            {
                lock (FileLock)
                {
                    // Add to recent trades for strategy performance tracking
                    if (RecentTrades.TryGetValue(outcome.Strategy, out var trades))
                    {
                        trades.Add(outcome);

                        // Keep only last 100 trades per strategy
                        if (trades.Count > 100)
                        {
                            trades.RemoveAt(0);
                        }
                    }

                    // Log outcome to file
                    var fileName = $"trade_outcomes_{DateTime.UtcNow:yyyyMMdd}.jsonl";
                    var filePath = Path.Combine(DataPath, fileName);

                    var outcomeRecord = new
                    {
                        timestamp = DateTime.UtcNow,
                        signalId = outcome.SignalId,
                        strategy = outcome.Strategy.ToString(),
                        exitTime = outcome.ExitTime,
                        actualPnl = outcome.ActualPnl,
                        actualRMultiple = outcome.ActualRMultiple,
                        isWin = outcome.IsWin,
                        exitReason = outcome.ExitReason,
                        holdingTimeMinutes = outcome.HoldingTimeMinutes
                    };

                    var json = JsonSerializer.Serialize(outcomeRecord);
                    File.AppendAllText(filePath, json + Environment.NewLine);
                }

                log.LogDebug("[RL-{Strategy}] Logged trade outcome for signal {SignalId}: {Result}",
                    outcome.Strategy, outcome.SignalId, outcome.IsWin ? "WIN" : "LOSS");
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[RL-{Strategy}] Failed to log trade outcome", outcome.Strategy);
            }
        }

        /// <summary>
        /// Convert ComprehensiveFeatures to TradeSignalData for enhanced training
        /// </summary>
        public static TradeSignalData ConvertToTradeSignalData(ComprehensiveFeatures features, string signalId)
        {
            return new TradeSignalData
            {
                Id = signalId,
                Symbol = features.Symbol,
                Direction = DetermineDirection(features),
                Entry = features.Price,
                Size = 1.0m, // Base size, will be adjusted by position sizing
                Strategy = features.Strategy.ToString(),
                StopLoss = features.Price * 0.99m, // Approximate stop loss
                TakeProfit = features.Price * 1.02m, // Approximate take profit
                Regime = features.Regime.ToString(),
                Atr = features.Atr14,
                Rsi = features.Rsi14,
                Ema20 = features.Ema20,
                Ema50 = features.Ema50,
                BbUpper = features.BbUpper,
                BbLower = features.BbLower,
                Momentum = features.MomentumStrength,
                TrendStrength = features.TrendStrength,
                VixLevel = 20m // Default VIX level
            };
        }

        /// <summary>
        /// Convert EnhancedTradeOutcome to TradeOutcomeData for enhanced training
        /// </summary>
        public static TradeOutcomeData ConvertToTradeOutcomeData(EnhancedTradeOutcome outcome)
        {
            return new TradeOutcomeData
            {
                IsWin = outcome.IsWin,
                ActualPnl = outcome.ActualPnl,
                ExitPrice = outcome.ExitPrice ?? 0m,
                ExitTime = outcome.ExitTime ?? DateTime.UtcNow,
                HoldingTimeMinutes = outcome.HoldingTimeMinutes,
                ActualRMultiple = outcome.ActualRMultiple,
                MaxDrawdown = 0m // Simplified - could be enhanced
            };
        }

        /// <summary>
        /// Get training sample count across all collection methods
        /// </summary>
        public static int GetTotalTrainingSampleCount()
        {
            var totalCount = 0;

            try
            {
                // Count samples from traditional collection
                var files = Directory.GetFiles(DataPath, "*.jsonl");
                foreach (var file in files)
                {
                    totalCount += File.ReadAllLines(file).Length;
                }

                // Count samples from emergency data generation
                var emergencyDataPath = Path.Combine(AppContext.BaseDirectory, "data", "rl_training");
                if (Directory.Exists(emergencyDataPath))
                {
                    var emergencyFiles = Directory.GetFiles(emergencyDataPath, "*.jsonl");
                    foreach (var file in emergencyFiles)
                    {
                        totalCount += File.ReadAllLines(file).Length;
                    }
                }
            }
            catch (Exception)
            {
                // Return 0 if unable to count
            }

            return totalCount;
        }

        private static string DetermineDirection(ComprehensiveFeatures features)
        {
            // Simple logic to determine trade direction based on features
            if (features.Rsi14 < 30 && features.Price < features.BbLower)
                return "BUY"; // Oversold
            else if (features.Rsi14 > 70 && features.Price > features.BbUpper)
                return "SELL"; // Overbought
            else if (features.MomentumStrength > 0.5m)
                return "BUY"; // Positive momentum
            else if (features.MomentumStrength < -0.5m)
                return "SELL"; // Negative momentum
            else
                return "HOLD"; // Neutral
        }

        #endregion
    }
}
