using System;
using System.Collections.Generic;
using System.Linq;

namespace OrchestratorAgent.Execution
{
    /// <summary>
    /// Purged, embargoed walk-forward validation system.
    /// Prevents look-ahead bias and provides credible confidence intervals.
    /// Essential for realistic performance estimation and bandit optimization.
    /// </summary>
    public class PurgedWalkForwardValidator
    {
        private readonly List<TradeRecord> _tradeHistory = new();
        private readonly TimeSpan _purgeWindow = TimeSpan.FromHours(4); // Purge 4 hours around test period
        private readonly TimeSpan _embargoWindow = TimeSpan.FromHours(12); // Embargo 12 hours after test period

        // Configuration
        public TimeSpan TrainingWindow { get; set; } = TimeSpan.FromDays(30); // 30 days of training data
        public TimeSpan TestWindow { get; set; } = TimeSpan.FromDays(1); // 1 day test period
        public TimeSpan WalkForwardStep { get; set; } = TimeSpan.FromDays(1); // Step forward 1 day each time
        public int MinTrainingTrades { get; set; } = 50; // Minimum trades for valid training
        public double ConfidenceLevel { get; set; } = 0.95; // 95% confidence intervals

        /// <summary>
        /// Runs purged walk-forward validation on strategy parameters
        /// </summary>
        public WalkForwardResults RunValidation(StrategyParameters parameters, DateTime startDate, DateTime endDate)
        {
            var results = new WalkForwardResults
            {
                Parameters = parameters,
                StartDate = startDate,
                EndDate = endDate,
                FoldResults = new List<ValidationFold>()
            };

            try
            {
                Console.WriteLine($"[PURGED-WF] Starting validation from {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");

                var currentDate = startDate.Add(TrainingWindow);
                var foldNumber = 1;

                while (currentDate.Add(TestWindow) <= endDate)
                {
                    var fold = RunSingleFold(parameters, currentDate, foldNumber);
                    results.FoldResults.Add(fold);

                    Console.WriteLine($"[PURGED-WF] Fold {foldNumber}: winRate={fold.WinRate:F3}, avgR={fold.AverageR:F2}, trades={fold.TradeCount}");

                    currentDate = currentDate.Add(WalkForwardStep);
                    foldNumber++;
                }

                // Calculate aggregate statistics
                results.CalculateAggregateStats();

                // Calculate confidence intervals
                results.CalculateConfidenceIntervals(ConfidenceLevel);

                Console.WriteLine($"[PURGED-WF] Validation complete: {results.FoldResults.Count} folds, avgWinRate={results.OverallWinRate:F3} [{results.WinRateCI.lower:F3}, {results.WinRateCI.upper:F3}]");

                return results;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PURGED-WF] Validation error: {ex.Message}");
                results.HasError = true;
                results.ErrorMessage = ex.Message;
                return results;
            }
        }

        /// <summary>
        /// Adds a trade record for future validation
        /// </summary>
        public void AddTradeRecord(TradeRecord trade)
        {
            _tradeHistory.Add(trade);

            // Keep history manageable (last 6 months)
            var cutoff = DateTime.Now.AddMonths(-6);
            _tradeHistory.RemoveAll(t => t.EntryTime < cutoff);
        }

        /// <summary>
        /// Gets strategy performance metrics with confidence intervals
        /// </summary>
        public PerformanceMetrics GetPerformanceMetrics(StrategyParameters parameters, TimeSpan lookbackPeriod)
        {
            var cutoff = DateTime.Now.Subtract(lookbackPeriod);
            var relevantTrades = _tradeHistory
                .Where(t => t.EntryTime >= cutoff && t.StrategyParameters.Equals(parameters))
                .ToList();

            if (relevantTrades.Count < 10)
            {
                return new PerformanceMetrics
                {
                    HasSufficientData = false,
                    Message = $"Insufficient data: {relevantTrades.Count} trades (need ≥10)"
                };
            }

            return CalculateMetricsWithCI(relevantTrades);
        }

        private ValidationFold RunSingleFold(StrategyParameters parameters, DateTime testStart, int foldNumber)
        {
            var testEnd = testStart.Add(TestWindow);
            var trainingStart = testStart.Subtract(TrainingWindow);
            var trainingEnd = testStart;

            // Get training data with purging
            var trainingTrades = GetPurgedTrainingData(trainingStart, trainingEnd, testStart, testEnd);

            // Get test data
            var testTrades = _tradeHistory
                .Where(t => t.EntryTime >= testStart && t.EntryTime < testEnd)
                .Where(t => t.StrategyParameters.Equals(parameters))
                .ToList();

            var fold = new ValidationFold
            {
                FoldNumber = foldNumber,
                TrainingStart = trainingStart,
                TrainingEnd = trainingEnd,
                TestStart = testStart,
                TestEnd = testEnd,
                TrainingTradeCount = trainingTrades.Count,
                TestTradeCount = testTrades.Count
            };

            if (trainingTrades.Count < MinTrainingTrades)
            {
                fold.IsValid = false;
                fold.ValidationMessage = $"Insufficient training data: {trainingTrades.Count} < {MinTrainingTrades}";
                return fold;
            }

            if (testTrades.Count == 0)
            {
                fold.IsValid = false;
                fold.ValidationMessage = "No test trades found";
                return fold;
            }

            // Calculate performance metrics
            fold.WinRate = testTrades.Count(t => t.IsWin) / (double)testTrades.Count;
            fold.AverageR = testTrades.Average(t => t.RMultiple);
            fold.TotalR = testTrades.Sum(t => t.RMultiple);
            fold.MaxDrawdown = CalculateMaxDrawdown(testTrades);
            fold.SharpeRatio = CalculateSharpeRatio(testTrades);
            fold.TradeCount = testTrades.Count;
            fold.IsValid = true;

            return fold;
        }

        private List<TradeRecord> GetPurgedTrainingData(DateTime trainingStart, DateTime trainingEnd,
            DateTime testStart, DateTime testEnd)
        {
            // Get base training data
            var trainingTrades = _tradeHistory
                .Where(t => t.EntryTime >= trainingStart && t.EntryTime < trainingEnd)
                .ToList();

            // Apply purging: remove trades that could have overlap with test period
            var purgeStart = testStart.Subtract(_purgeWindow);
            var purgeEnd = testEnd.Add(_embargoWindow);

            var purgedTrades = trainingTrades
                .Where(t => !(t.EntryTime >= purgeStart && t.EntryTime <= purgeEnd))
                .Where(t => !(t.ExitTime.HasValue && t.ExitTime >= purgeStart && t.ExitTime <= purgeEnd))
                .ToList();

            var purgedCount = trainingTrades.Count - purgedTrades.Count;
            if (purgedCount > 0)
            {
                Console.WriteLine($"[PURGED-WF] Purged {purgedCount} trades to prevent look-ahead bias");
            }

            return purgedTrades;
        }

        private PerformanceMetrics CalculateMetricsWithCI(List<TradeRecord> trades)
        {
            var metrics = new PerformanceMetrics
            {
                HasSufficientData = true,
                TradeCount = trades.Count,
                WinRate = trades.Count(t => t.IsWin) / (double)trades.Count,
                AverageR = trades.Average(t => t.RMultiple),
                TotalR = trades.Sum(t => t.RMultiple),
                MaxDrawdown = CalculateMaxDrawdown(trades),
                SharpeRatio = CalculateSharpeRatio(trades)
            };

            // Calculate confidence intervals
            metrics.WinRateCI = CalculateWinRateCI(trades, ConfidenceLevel);
            metrics.AverageRCI = CalculateAverageRCI(trades, ConfidenceLevel);

            return metrics;
        }

        private (double lower, double upper) CalculateWinRateCI(List<TradeRecord> trades, double confidence)
        {
            var n = trades.Count;
            var wins = trades.Count(t => t.IsWin);
            var p = wins / (double)n;

            // Wilson score interval (better than normal approximation for small samples)
            var z = confidence == 0.95 ? 1.96 : (confidence == 0.99 ? 2.576 : 1.645);
            var denominator = 1 + z * z / n;
            var center = (p + z * z / (2 * n)) / denominator;
            var delta = z * Math.Sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator;

            return (Math.Max(0, center - delta), Math.Min(1, center + delta));
        }

        private (double lower, double upper) CalculateAverageRCI(List<TradeRecord> trades, double confidence)
        {
            var rValues = trades.Select(t => t.RMultiple).ToArray();
            var mean = rValues.Average();
            var stdev = CalculateStandardDeviation(rValues);
            var n = rValues.Length;

            // t-distribution for small samples
            var tValue = confidence == 0.95 ? 1.96 : (confidence == 0.99 ? 2.576 : 1.645); // Approximation
            var margin = tValue * stdev / Math.Sqrt(n);

            return (mean - margin, mean + margin);
        }

        private double CalculateMaxDrawdown(List<TradeRecord> trades)
        {
            if (trades.Count == 0) return 0.0;

            var cumulative = 0.0;
            var peak = 0.0;
            var maxDrawdown = 0.0;

            foreach (var trade in trades.OrderBy(t => t.EntryTime))
            {
                cumulative += trade.RMultiple;
                peak = Math.Max(peak, cumulative);
                var drawdown = peak - cumulative;
                maxDrawdown = Math.Max(maxDrawdown, drawdown);
            }

            return maxDrawdown;
        }

        private double CalculateSharpeRatio(List<TradeRecord> trades)
        {
            if (trades.Count < 2) return 0.0;

            var returns = trades.Select(t => t.RMultiple).ToArray();
            var avgReturn = returns.Average();
            var stdev = CalculateStandardDeviation(returns);

            return stdev > 0 ? avgReturn / stdev : 0.0;
        }

        private double CalculateStandardDeviation(double[] values)
        {
            if (values.Length < 2) return 0.0;

            var mean = values.Average();
            var sumSquaredDiffs = values.Sum(v => Math.Pow(v - mean, 2));
            return Math.Sqrt(sumSquaredDiffs / (values.Length - 1));
        }
    }

    public class TradeRecord
    {
        public DateTime EntryTime { get; set; }
        public DateTime? ExitTime { get; set; }
        public string Symbol { get; set; } = "";
        public string Side { get; set; } = "";
        public double EntryPrice { get; set; }
        public double ExitPrice { get; set; }
        public double RMultiple { get; set; }
        public bool IsWin { get; set; }
        public string Strategy { get; set; } = "";
        public StrategyParameters StrategyParameters { get; set; } = new();
        public string ExitReason { get; set; } = "";
    }

    public class StrategyParameters
    {
        public string StrategyName { get; set; } = "";
        public Dictionary<string, object> Parameters { get; } = new();

        public override bool Equals(object? obj)
        {
            if (obj is not StrategyParameters other) return false;
            return StrategyName == other.StrategyName &&
                   Parameters.Count == other.Parameters.Count &&
                   Parameters.All(kvp => other.Parameters.ContainsKey(kvp.Key) &&
                                        Equals(kvp.Value, other.Parameters[kvp.Key]));
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(StrategyName, Parameters.Count);
        }
    }

    public class WalkForwardResults
    {
        public StrategyParameters Parameters { get; set; } = new();
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public List<ValidationFold> FoldResults { get; } = new();

        // Aggregate statistics
        public double OverallWinRate { get; set; }
        public double OverallAverageR { get; set; }
        public double OverallSharpeRatio { get; set; }
        public int TotalTrades { get; set; }

        // Confidence intervals
        public (double lower, double upper) WinRateCI { get; set; }
        public (double lower, double upper) AverageRCI { get; set; }

        // Validation info
        public bool HasError { get; set; }
        public string ErrorMessage { get; set; } = "";

        public void CalculateAggregateStats()
        {
            var validFolds = FoldResults.Where(f => f.IsValid).ToList();
            if (validFolds.Count == 0) return;

            TotalTrades = validFolds.Sum(f => f.TradeCount);
            OverallWinRate = validFolds.Average(f => f.WinRate);
            OverallAverageR = validFolds.Average(f => f.AverageR);
            OverallSharpeRatio = validFolds.Average(f => f.SharpeRatio);
        }

        public void CalculateConfidenceIntervals(double confidence)
        {
            var validFolds = FoldResults.Where(f => f.IsValid).ToList();
            if (validFolds.Count < 2) return;

            var winRates = validFolds.Select(f => f.WinRate).ToArray();
            var avgRs = validFolds.Select(f => f.AverageR).ToArray();

            WinRateCI = CalculateCI(winRates, confidence);
            AverageRCI = CalculateCI(avgRs, confidence);
        }

        private (double lower, double upper) CalculateCI(double[] values, double confidence)
        {
            if (values.Length < 2) return (0, 0);

            var mean = values.Average();
            var stdev = Math.Sqrt(values.Sum(v => Math.Pow(v - mean, 2)) / (values.Length - 1));
            var tValue = confidence == 0.95 ? 1.96 : (confidence == 0.99 ? 2.576 : 1.645);
            var margin = tValue * stdev / Math.Sqrt(values.Length);

            return (mean - margin, mean + margin);
        }
    }

    public class ValidationFold
    {
        public int FoldNumber { get; set; }
        public DateTime TrainingStart { get; set; }
        public DateTime TrainingEnd { get; set; }
        public DateTime TestStart { get; set; }
        public DateTime TestEnd { get; set; }

        public int TrainingTradeCount { get; set; }
        public int TestTradeCount { get; set; }

        public bool IsValid { get; set; }
        public string ValidationMessage { get; set; } = "";

        public double WinRate { get; set; }
        public double AverageR { get; set; }
        public double TotalR { get; set; }
        public double MaxDrawdown { get; set; }
        public double SharpeRatio { get; set; }
        public int TradeCount { get; set; }
    }

    public class PerformanceMetrics
    {
        public bool HasSufficientData { get; set; }
        public string Message { get; set; } = "";

        public int TradeCount { get; set; }
        public double WinRate { get; set; }
        public double AverageR { get; set; }
        public double TotalR { get; set; }
        public double MaxDrawdown { get; set; }
        public double SharpeRatio { get; set; }

        public (double lower, double upper) WinRateCI { get; set; }
        public (double lower, double upper) AverageRCI { get; set; }
    }
}
