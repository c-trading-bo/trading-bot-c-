using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    /// <summary>
    /// Rolling-window backtest harness with purge/embargo logic and auto-retrain
    /// Implements walk-forward analysis with performance decay detection
    /// </summary>
    public class BacktestHarnessService
    {
        private readonly ILogger<BacktestHarnessService> _logger;
        private readonly BacktestOptions _options;
        private readonly JsonSerializerOptions _jsonOptions;
        private readonly ModelRegistryService _modelRegistry;
        private readonly Dictionary<string, PerformanceTracker> _performanceTrackers = new();

        public BacktestHarnessService(
            ILogger<BacktestHarnessService> logger,
            IOptions<BacktestOptions> options,
            ModelRegistryService modelRegistry)
        {
            _logger = logger;
            _options = options.Value;
            _modelRegistry = modelRegistry;
            
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true
            };

            _logger.LogInformation("📊 Backtest Harness initialized with {WindowDays} day windows", _options.TrainingWindowDays);
        }

        /// <summary>
        /// Run walk-forward backtest with rolling windows
        /// </summary>
        public async Task<BacktestResult> RunWalkForwardBacktestAsync(
            string modelName,
            DateTime startDate,
            DateTime endDate,
            CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("🚀 Starting walk-forward backtest: {ModelName} from {StartDate} to {EndDate}",
                    modelName, startDate.ToString("yyyy-MM-dd"), endDate.ToString("yyyy-MM-dd"));

                var result = new BacktestResult
                {
                    ModelName = modelName,
                    StartDate = startDate,
                    EndDate = endDate,
                    WindowResults = new List<WindowResult>()
                };

                var currentDate = startDate;
                var windowNumber = 0;

                while (currentDate.AddDays(_options.TestWindowDays) <= endDate)
                {
                    var trainingStart = currentDate.AddDays(-_options.TrainingWindowDays);
                    var trainingEnd = currentDate.AddDays(-_options.PurgeDays);
                    var testStart = currentDate.AddDays(_options.EmbargoDays);
                    var testEnd = testStart.AddDays(_options.TestWindowDays);

                    var windowResult = await RunWindowBacktestAsync(
                        modelName, 
                        windowNumber,
                        trainingStart, 
                        trainingEnd, 
                        testStart, 
                        testEnd, 
                        cancellationToken);

                    result.WindowResults.Add(windowResult);

                    // Check for performance decay
                    var shouldRetrain = await CheckPerformanceDecayAsync(modelName, windowResult, cancellationToken);
                    if (shouldRetrain)
                    {
                        _logger.LogWarning("📉 Performance decay detected for {ModelName} at window {WindowNumber}",
                            modelName, windowNumber);
                        
                        if (_options.AutoRetrain)
                        {
                            await TriggerRetrainingAsync(modelName, trainingStart, trainingEnd, cancellationToken);
                        }
                    }

                    currentDate = currentDate.AddDays(_options.StepSizeDays);
                    windowNumber++;
                }

                // Calculate final metrics
                result.OverallMetrics = CalculateOverallMetrics(result.WindowResults);
                
                // Save backtest report
                await SaveBacktestReportAsync(result, cancellationToken);

                _logger.LogInformation("✅ Walk-forward backtest completed: {ModelName}, {WindowCount} windows, Overall Sharpe: {Sharpe:F3}",
                    modelName, result.WindowResults.Count, result.OverallMetrics.SharpeRatio);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Walk-forward backtest failed: {ModelName}", modelName);
                throw;
            }
        }

        /// <summary>
        /// Run backtest for a single window
        /// </summary>
        private async Task<WindowResult> RunWindowBacktestAsync(
            string modelName,
            int windowNumber,
            DateTime trainingStart,
            DateTime trainingEnd,
            DateTime testStart,
            DateTime testEnd,
            CancellationToken cancellationToken)
        {
            var windowResult = new WindowResult
            {
                WindowNumber = windowNumber,
                TrainingStart = trainingStart,
                TrainingEnd = trainingEnd,
                TestStart = testStart,
                TestEnd = testEnd,
                Trades = new List<BacktestTrade>()
            };

            try
            {
                // Load market data for the test period
                var marketData = await LoadMarketDataAsync(testStart, testEnd, cancellationToken);
                if (!marketData.Any())
                {
                    _logger.LogWarning("⚠️ No market data found for window {WindowNumber}", windowNumber);
                    return windowResult;
                }

                // Load or train model for this window
                var modelPath = await PrepareModelForWindowAsync(modelName, trainingStart, trainingEnd, cancellationToken);
                if (string.IsNullOrEmpty(modelPath))
                {
                    _logger.LogWarning("⚠️ No model available for window {WindowNumber}", windowNumber);
                    return windowResult;
                }

                // Run simulation
                var trades = await SimulateTradesAsync(modelPath, marketData, cancellationToken);
                windowResult.Trades = trades;

                // Calculate window metrics
                windowResult.Metrics = CalculateWindowMetrics(trades);

                _logger.LogDebug("📊 Window {WindowNumber} completed: {TradeCount} trades, PnL: ${PnL:F2}, Sharpe: {Sharpe:F3}",
                    windowNumber, trades.Count, windowResult.Metrics.TotalPnL, windowResult.Metrics.SharpeRatio);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Window backtest failed: {WindowNumber}", windowNumber);
                windowResult.HasError = true;
                windowResult.ErrorMessage = ex.Message;
            }

            return windowResult;
        }

        /// <summary>
        /// Check if model performance has decayed below threshold
        /// </summary>
        private async Task<bool> CheckPerformanceDecayAsync(
            string modelName,
            WindowResult currentWindow,
            CancellationToken cancellationToken)
        {
            if (!_performanceTrackers.TryGetValue(modelName, out var tracker))
            {
                tracker = new PerformanceTracker(modelName, _options.PerformanceWindowSize);
                _performanceTrackers[modelName] = tracker;
            }

            tracker.AddPerformance(currentWindow.Metrics);

            // Check for decay only if we have enough data
            if (tracker.GetWindowCount() < _options.MinWindowsForDecayCheck)
            {
                return false;
            }

            var recentPerformance = tracker.GetRecentPerformance();
            var historicalPerformance = tracker.GetHistoricalPerformance();

            // Check multiple decay indicators
            var sharpeDecay = (historicalPerformance.SharpeRatio - recentPerformance.SharpeRatio) / Math.Max(Math.Abs(historicalPerformance.SharpeRatio), 0.1);
            var winRateDecay = historicalPerformance.WinRate - recentPerformance.WinRate;
            var drawdownIncrease = recentPerformance.MaxDrawdown - historicalPerformance.MaxDrawdown;

            var isDecayed = sharpeDecay > _options.SharpeDecayThreshold ||
                           winRateDecay > _options.WinRateDecayThreshold ||
                           drawdownIncrease > _options.DrawdownIncreaseThreshold;

            if (isDecayed)
            {
                _logger.LogWarning("📉 Performance decay detected: Sharpe decay: {SharpeDecay:F3}, Win rate decay: {WinRateDecay:F3}, Drawdown increase: {DrawdownIncrease:F3}",
                    sharpeDecay, winRateDecay, drawdownIncrease);
            }

            return isDecayed;
        }

        /// <summary>
        /// Trigger model retraining
        /// </summary>
        private async Task TriggerRetrainingAsync(
            string modelName,
            DateTime trainingStart,
            DateTime trainingEnd,
            CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation("🔄 Triggering retraining for {ModelName} from {StartDate} to {EndDate}",
                    modelName, trainingStart.ToString("yyyy-MM-dd"), trainingEnd.ToString("yyyy-MM-dd"));

                // Create retraining task
                var retrainTask = new RetrainingTask
                {
                    ModelName = modelName,
                    TrainingStart = trainingStart,
                    TrainingEnd = trainingEnd,
                    RequestedAt = DateTime.UtcNow,
                    Status = RetrainingStatus.Pending
                };

                // Save retraining request
                await SaveRetrainingTaskAsync(retrainTask, cancellationToken);

                // In a real implementation, this would trigger the actual ML training pipeline
                // For now, we'll just log the request
                _logger.LogInformation("📝 Retraining task created for {ModelName}", modelName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to trigger retraining for {ModelName}", modelName);
            }
        }

        /// <summary>
        /// Load market data for backtesting
        /// </summary>
        private async Task<List<MarketDataPoint>> LoadMarketDataAsync(
            DateTime startDate,
            DateTime endDate,
            CancellationToken cancellationToken)
        {
            // Placeholder implementation - would load from actual data source
            var dataPoints = new List<MarketDataPoint>();
            
            var current = startDate;
            var random = new Random(42); // Deterministic for testing

            while (current <= endDate)
            {
                // Generate synthetic market data
                dataPoints.Add(new MarketDataPoint
                {
                    Timestamp = current,
                    Open = 4000 + (float)(random.NextDouble() * 100),
                    High = 4000 + (float)(random.NextDouble() * 150),
                    Low = 4000 + (float)(random.NextDouble() * 50),
                    Close = 4000 + (float)(random.NextDouble() * 100),
                    Volume = random.Next(1000, 10000)
                });

                current = current.AddMinutes(1);
            }

            await Task.Delay(1, cancellationToken); // Make it async
            return dataPoints;
        }

        /// <summary>
        /// Prepare model for a specific window
        /// </summary>
        private async Task<string> PrepareModelForWindowAsync(
            string modelName,
            DateTime trainingStart,
            DateTime trainingEnd,
            CancellationToken cancellationToken)
        {
            try
            {
                // Check if we have a model for this time period
                var latestModel = await _modelRegistry.GetLatestModelAsync(modelName, cancellationToken);
                if (latestModel != null)
                {
                    return latestModel.RegistryPath;
                }

                // If no model exists, return empty string to indicate we need training
                return string.Empty;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to prepare model for window: {ModelName}", modelName);
                return string.Empty;
            }
        }

        /// <summary>
        /// Simulate trades using the model
        /// </summary>
        private async Task<List<BacktestTrade>> SimulateTradesAsync(
            string modelPath,
            List<MarketDataPoint> marketData,
            CancellationToken cancellationToken)
        {
            var trades = new List<BacktestTrade>();
            
            // Placeholder implementation - would use actual model for predictions
            var random = new Random(42);
            
            for (int i = 100; i < marketData.Count - 10; i++) // Leave buffer for features and exit
            {
                // Simple synthetic trading logic
                if (random.NextDouble() < 0.05) // 5% chance of trade
                {
                    var entryPrice = marketData[i].Close;
                    var exitIndex = i + random.Next(1, 10);
                    var exitPrice = marketData[Math.Min(exitIndex, marketData.Count - 1)].Close;
                    
                    var isLong = random.NextDouble() > 0.5;
                    var pnl = isLong ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
                    
                    trades.Add(new BacktestTrade
                    {
                        EntryTime = marketData[i].Timestamp,
                        ExitTime = marketData[exitIndex].Timestamp,
                        EntryPrice = entryPrice,
                        ExitPrice = exitPrice,
                        Quantity = 1,
                        Side = isLong ? "Long" : "Short",
                        PnL = pnl,
                        Commission = 2.5f // $2.50 per contract
                    });
                }
            }

            await Task.Delay(1, cancellationToken); // Make it async
            return trades;
        }

        /// <summary>
        /// Calculate metrics for a window
        /// </summary>
        private WindowMetrics CalculateWindowMetrics(List<BacktestTrade> trades)
        {
            if (!trades.Any())
            {
                return new WindowMetrics();
            }

            var totalPnL = trades.Sum(t => t.PnL - t.Commission);
            var winningTrades = trades.Where(t => t.PnL > 0).ToList();
            var losingTrades = trades.Where(t => t.PnL <= 0).ToList();

            var dailyReturns = CalculateDailyReturns(trades);
            var sharpeRatio = CalculateSharpeRatio(dailyReturns);
            var maxDrawdown = CalculateMaxDrawdown(trades);

            return new WindowMetrics
            {
                TotalPnL = totalPnL,
                TotalTrades = trades.Count,
                WinningTrades = winningTrades.Count,
                LosingTrades = losingTrades.Count,
                WinRate = (double)winningTrades.Count / trades.Count,
                AverageWin = winningTrades.Any() ? winningTrades.Average(t => t.PnL) : 0,
                AverageLoss = losingTrades.Any() ? losingTrades.Average(t => t.PnL) : 0,
                SharpeRatio = sharpeRatio,
                MaxDrawdown = maxDrawdown,
                ProfitFactor = winningTrades.Sum(t => t.PnL) / Math.Max(Math.Abs(losingTrades.Sum(t => t.PnL)), 1)
            };
        }

        /// <summary>
        /// Calculate overall metrics across all windows
        /// </summary>
        private WindowMetrics CalculateOverallMetrics(List<WindowResult> windowResults)
        {
            var allTrades = windowResults.SelectMany(w => w.Trades).ToList();
            return CalculateWindowMetrics(allTrades);
        }

        private List<double> CalculateDailyReturns(List<BacktestTrade> trades)
        {
            var dailyPnL = trades
                .GroupBy(t => t.EntryTime.Date)
                .Select(g => g.Sum(t => t.PnL - t.Commission))
                .ToList();

            return dailyPnL;
        }

        private double CalculateSharpeRatio(List<double> dailyReturns)
        {
            if (dailyReturns.Count < 2)
                return 0;

            var averageReturn = dailyReturns.Average();
            var standardDeviation = Math.Sqrt(dailyReturns.Average(r => Math.Pow(r - averageReturn, 2)));
            
            return standardDeviation > 0 ? averageReturn / standardDeviation * Math.Sqrt(252) : 0;
        }

        private double CalculateMaxDrawdown(List<BacktestTrade> trades)
        {
            if (!trades.Any())
                return 0;

            var cumulativePnL = 0.0;
            var peak = 0.0;
            var maxDrawdown = 0.0;

            foreach (var trade in trades.OrderBy(t => t.EntryTime))
            {
                cumulativePnL += trade.PnL - trade.Commission;
                peak = Math.Max(peak, cumulativePnL);
                var drawdown = peak - cumulativePnL;
                maxDrawdown = Math.Max(maxDrawdown, drawdown);
            }

            return maxDrawdown;
        }

        private async Task SaveBacktestReportAsync(BacktestResult result, CancellationToken cancellationToken)
        {
            try
            {
                var reportsDir = "reports/backtests";
                Directory.CreateDirectory(reportsDir);
                
                var fileName = $"backtest_{result.ModelName}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
                var filePath = Path.Combine(reportsDir, fileName);
                
                var json = JsonSerializer.Serialize(result, _jsonOptions);
                await File.WriteAllTextAsync(filePath, json, cancellationToken);
                
                _logger.LogInformation("📄 Backtest report saved: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to save backtest report");
            }
        }

        private async Task SaveRetrainingTaskAsync(RetrainingTask task, CancellationToken cancellationToken)
        {
            try
            {
                var tasksDir = "data/retrain_tasks";
                Directory.CreateDirectory(tasksDir);
                
                var fileName = $"retrain_{task.ModelName}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
                var filePath = Path.Combine(tasksDir, fileName);
                
                var json = JsonSerializer.Serialize(task, _jsonOptions);
                await File.WriteAllTextAsync(filePath, json, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to save retraining task");
            }
        }
    }

    /// <summary>
    /// Configuration options for backtest harness
    /// </summary>
    public class BacktestOptions
    {
        public int TrainingWindowDays { get; set; } = 365;
        public int TestWindowDays { get; set; } = 30;
        public int StepSizeDays { get; set; } = 7;
        public int PurgeDays { get; set; } = 1;
        public int EmbargoDays { get; set; } = 1;
        public bool AutoRetrain { get; set; } = true;
        public double SharpeDecayThreshold { get; set; } = 0.3;
        public double WinRateDecayThreshold { get; set; } = 0.1;
        public double DrawdownIncreaseThreshold { get; set; } = 100.0;
        public int PerformanceWindowSize { get; set; } = 10;
        public int MinWindowsForDecayCheck { get; set; } = 5;
    }

    /// <summary>
    /// Backtest result containing all windows
    /// </summary>
    public class BacktestResult
    {
        public string ModelName { get; set; } = string.Empty;
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public List<WindowResult> WindowResults { get; set; } = new();
        public WindowMetrics OverallMetrics { get; set; } = new();
    }

    /// <summary>
    /// Result for a single backtest window
    /// </summary>
    public class WindowResult
    {
        public int WindowNumber { get; set; }
        public DateTime TrainingStart { get; set; }
        public DateTime TrainingEnd { get; set; }
        public DateTime TestStart { get; set; }
        public DateTime TestEnd { get; set; }
        public List<BacktestTrade> Trades { get; set; } = new();
        public WindowMetrics Metrics { get; set; } = new();
        public bool HasError { get; set; }
        public string ErrorMessage { get; set; } = string.Empty;
    }

    /// <summary>
    /// Performance metrics for a window or overall backtest
    /// </summary>
    public class WindowMetrics
    {
        public double TotalPnL { get; set; }
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public int LosingTrades { get; set; }
        public double WinRate { get; set; }
        public double AverageWin { get; set; }
        public double AverageLoss { get; set; }
        public double SharpeRatio { get; set; }
        public double MaxDrawdown { get; set; }
        public double ProfitFactor { get; set; }
    }

    /// <summary>
    /// Individual trade in backtest
    /// </summary>
    public class BacktestTrade
    {
        public DateTime EntryTime { get; set; }
        public DateTime ExitTime { get; set; }
        public float EntryPrice { get; set; }
        public float ExitPrice { get; set; }
        public int Quantity { get; set; }
        public string Side { get; set; } = string.Empty;
        public double PnL { get; set; }
        public double Commission { get; set; }
    }

    /// <summary>
    /// Market data point for backtesting
    /// </summary>
    public class MarketDataPoint
    {
        public DateTime Timestamp { get; set; }
        public float Open { get; set; }
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
        public int Volume { get; set; }
    }

    /// <summary>
    /// Performance tracker for decay detection
    /// </summary>
    public class PerformanceTracker
    {
        private readonly string _modelName;
        private readonly int _windowSize;
        private readonly Queue<WindowMetrics> _performanceWindow = new();

        public PerformanceTracker(string modelName, int windowSize)
        {
            _modelName = modelName;
            _windowSize = windowSize;
        }

        public void AddPerformance(WindowMetrics metrics)
        {
            _performanceWindow.Enqueue(metrics);
            if (_performanceWindow.Count > _windowSize)
            {
                _performanceWindow.Dequeue();
            }
        }

        public int GetWindowCount() => _performanceWindow.Count;

        public WindowMetrics GetRecentPerformance()
        {
            var recent = _performanceWindow.TakeLast(_windowSize / 2).ToList();
            return AggregateMetrics(recent);
        }

        public WindowMetrics GetHistoricalPerformance()
        {
            var historical = _performanceWindow.Take(_windowSize / 2).ToList();
            return AggregateMetrics(historical);
        }

        private WindowMetrics AggregateMetrics(List<WindowMetrics> metrics)
        {
            if (!metrics.Any())
                return new WindowMetrics();

            return new WindowMetrics
            {
                SharpeRatio = metrics.Average(m => m.SharpeRatio),
                WinRate = metrics.Average(m => m.WinRate),
                MaxDrawdown = metrics.Average(m => m.MaxDrawdown),
                TotalPnL = metrics.Sum(m => m.TotalPnL),
                TotalTrades = metrics.Sum(m => m.TotalTrades)
            };
        }
    }

    /// <summary>
    /// Retraining task
    /// </summary>
    public class RetrainingTask
    {
        public string ModelName { get; set; } = string.Empty;
        public DateTime TrainingStart { get; set; }
        public DateTime TrainingEnd { get; set; }
        public DateTime RequestedAt { get; set; }
        public RetrainingStatus Status { get; set; }
        public string ErrorMessage { get; set; } = string.Empty;
    }

    /// <summary>
    /// Retraining status enumeration
    /// </summary>
    public enum RetrainingStatus
    {
        Pending,
        InProgress,
        Completed,
        Failed
    }
}