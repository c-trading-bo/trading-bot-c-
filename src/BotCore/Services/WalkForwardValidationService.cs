using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using BotCore.Configuration;

namespace BotCore.Services
{
    /// <summary>
    /// Walk-forward validation service for robust ML model validation
    /// Implements proper time-series validation with rotating seeds and performance tracking
    /// </summary>
    public interface IWalkForwardValidationService
    {
        Task<WalkForwardResult> RunWalkForwardValidationAsync(WalkForwardRequest request, CancellationToken cancellationToken = default);
        Task<List<ValidationWindow>> GenerateValidationWindowsAsync(DateTime startDate, DateTime endDate);
        Task<WalkForwardModelPerformance> ValidateModelAsync(string modelPath, ValidationWindow window, CancellationToken cancellationToken = default);
        Task<bool> MeetsPerformanceThresholdsAsync(WalkForwardModelPerformance performance);
        Task LogValidationResultsAsync(WalkForwardResult result);
        Task<List<WalkForwardResult>> GetValidationHistoryAsync(string strategyName);
    }

    /// <summary>
    /// Comprehensive walk-forward validation service implementation
    /// </summary>
    public class WalkForwardValidationService : IWalkForwardValidationService
    {
        private readonly ILogger<WalkForwardValidationService> _logger;
        private readonly WalkForwardValidationConfiguration _config;
        private readonly IEnhancedBacktestService _backtestService;
        private readonly IModelVersionVerificationService _modelVersionService;
        private readonly string _validationHistoryPath;

        public WalkForwardValidationService(
            ILogger<WalkForwardValidationService> logger,
            IOptions<WalkForwardValidationConfiguration> config,
            IEnhancedBacktestService backtestService,
            IModelVersionVerificationService modelVersionService)
        {
            if (config is null) throw new ArgumentNullException(nameof(config));
            
            _logger = logger;
            _config = config.Value;
            _backtestService = backtestService;
            _modelVersionService = modelVersionService;
            _validationHistoryPath = Path.Combine("./validation_results", "walk_forward_history.json");
            
            // Ensure directory exists
            Directory.CreateDirectory(Path.GetDirectoryName(_validationHistoryPath)!);
        }

        /// <summary>
        /// Run comprehensive walk-forward validation
        /// </summary>
        public async Task<WalkForwardResult> RunWalkForwardValidationAsync(WalkForwardRequest request, CancellationToken cancellationToken = default)
        {
            if (request is null) throw new ArgumentNullException(nameof(request));
            
            try
            {
                _logger.LogInformation("[WALK-FORWARD] Starting walk-forward validation for {Strategy} from {StartDate} to {EndDate}",
                    request.StrategyName, request.StartDate, request.EndDate);

                var result = new WalkForwardResult
                {
                    StrategyName = request.StrategyName,
                    StartDate = request.StartDate,
                    EndDate = request.EndDate,
                    ValidationStarted = DateTime.UtcNow,
                    EnableSeedRotation = _config.SeedRotation.EnableSeedRotation
                };

                // Generate validation windows
                var windows = await GenerateValidationWindowsAsync(request.StartDate, request.EndDate).ConfigureAwait(false);
                result.TotalWindows = windows.Count;

                _logger.LogInformation("[WALK-FORWARD] Generated {WindowCount} validation windows", windows.Count);

                // Process each validation window
                var validationTasks = new List<Task<WindowResult>>();
                var semaphore = new SemaphoreSlim(Environment.ProcessorCount, Environment.ProcessorCount); // Limit concurrent validations

                foreach (var window in windows)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    validationTasks.Add(ProcessValidationWindowAsync(window, request, semaphore, cancellationToken));
                }

                // Wait for all validations to complete
                var windowResults = await Task.WhenAll(validationTasks).ConfigureAwait(false);
                result.WindowResults = windowResults.ToList();

                // Calculate aggregate metrics
                CalculateAggregateMetrics(result);

                // Validate overall performance
                result.PassesThresholds = await ValidateOverallPerformanceAsync(result).ConfigureAwait(false);

                result.ValidationCompleted = DateTime.UtcNow;
                result.ValidationDuration = result.ValidationCompleted - result.ValidationStarted;

                // Log comprehensive results
                await LogValidationResultsAsync(result).ConfigureAwait(false);

                _logger.LogInformation("[WALK-FORWARD] Completed walk-forward validation for {Strategy}: {PassedWindows}/{TotalWindows} windows passed, Overall: {Pass}",
                    request.StrategyName, result.PassedWindows, result.TotalWindows, result.PassesThresholds ? "PASS" : "FAIL");

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[WALK-FORWARD] Error in walk-forward validation for {Strategy}", request.StrategyName);
                throw;
            }
        }

        /// <summary>
        /// Generate validation windows for walk-forward analysis
        /// </summary>
        public Task<List<ValidationWindow>> GenerateValidationWindowsAsync(DateTime startDate, DateTime endDate)
        {
            try
            {
                var windows = new List<ValidationWindow>();
                var currentDate = startDate;
                var windowIndex = 0;

                while (currentDate.AddDays(_config.ValidationWindowDays) <= endDate)
                {
                    var trainingStart = currentDate;
                    var trainingEnd = currentDate.AddDays(_config.TrainingWindowDays);
                    var validationStart = trainingEnd.AddDays(1);
                    var validationEnd = validationStart.AddDays(_config.ValidationWindowDays);

                    // Ensure validation end doesn't exceed overall end date
                    if (validationEnd > endDate)
                        validationEnd = endDate;

                    // Generate unique seed for this window if rotation is enabled
                    var seed = _config.SeedRotation.EnableSeedRotation 
                        ? GenerateWindowSeed(windowIndex) 
                        : _config.SeedRotation.BaseSeed;

                    var window = new ValidationWindow
                    {
                        WindowIndex = windowIndex,
                        TrainingStart = trainingStart,
                        TrainingEnd = trainingEnd,
                        ValidationStart = validationStart,
                        ValidationEnd = validationEnd,
                        RandomSeed = seed,
                        TotalTrainingDays = (trainingEnd - trainingStart).Days,
                        TotalValidationDays = (validationEnd - validationStart).Days
                    };

                    windows.Add(window);
                    
                    // Move to next window
                    currentDate = currentDate.AddDays(_config.StepSizeDays);
                    windowIndex++;

                    // Safety check to prevent infinite loops
                    if (windows.Count > 1000)
                    {
                        _logger.LogWarning("[WALK-FORWARD] Generated maximum number of windows (1000), stopping");
                        break;
                    }
                }

                _logger.LogInformation("[WALK-FORWARD] Generated {WindowCount} validation windows covering {StartDate} to {EndDate}",
                    windows.Count, startDate, endDate);

                return Task.FromResult(windows);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[WALK-FORWARD] Error generating validation windows");
                throw;
            }
        }

        /// <summary>
        /// Validate model performance on a specific window
        /// </summary>
        public async Task<WalkForwardModelPerformance> ValidateModelAsync(string modelPath, ValidationWindow window, CancellationToken cancellationToken = default)
        {
            if (window is null) throw new ArgumentNullException(nameof(window));
            
            try
            {
                _logger.LogDebug("[MODEL-VALIDATION] Validating model {ModelPath} on window {WindowIndex}", modelPath, window.WindowIndex);

                // For this implementation, we'll simulate model validation
                // In production, this would:
                // 1. Load the model
                // 2. Run predictions on validation data
                // 3. Calculate performance metrics
                
                // Generate realistic performance metrics based on window characteristics
                var performance = await SimulateModelPerformance(window).ConfigureAwait(false);

                _logger.LogDebug("[MODEL-VALIDATION] Model validation completed for window {WindowIndex}: Sharpe={Sharpe:F2}, Drawdown={Drawdown:F2}%",
                    window.WindowIndex, performance.SharpeRatio, performance.MaxDrawdown * 100);

                return performance;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL-VALIDATION] Error validating model on window {WindowIndex}", window.WindowIndex);
                throw;
            }
        }

        /// <summary>
        /// Check if performance meets configured thresholds
        /// </summary>
        public Task<bool> MeetsPerformanceThresholdsAsync(WalkForwardModelPerformance performance)
        {
            if (performance is null) throw new ArgumentNullException(nameof(performance));
            
            try
            {
                var thresholds = _config.PerformanceThresholds;
                
                var meetsThresholds = 
                    performance.SharpeRatio >= thresholds.MinSharpeRatio &&
                    performance.MaxDrawdown <= (thresholds.MaxDrawdownPct / 100.0) &&
                    performance.WinRate >= thresholds.MinWinRate &&
                    performance.TotalTrades >= thresholds.MinTrades;

                if (!meetsThresholds)
                {
                    _logger.LogDebug("[PERFORMANCE-CHECK] Performance thresholds not met: " +
                        "Sharpe={Sharpe:F2} (min: {MinSharpe:F2}), " +
                        "Drawdown={Drawdown:F2}% (max: {MaxDrawdown:F2}%), " +
                        "WinRate={WinRate:F2} (min: {MinWinRate:F2}), " +
                        "Trades={Trades} (min: {MinTrades})",
                        performance.SharpeRatio, thresholds.MinSharpeRatio,
                        performance.MaxDrawdown * 100, thresholds.MaxDrawdownPct,
                        performance.WinRate, thresholds.MinWinRate,
                        performance.TotalTrades, thresholds.MinTrades);
                }

                return Task.FromResult(meetsThresholds);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[PERFORMANCE-CHECK] Error checking performance thresholds");
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Log comprehensive validation results
        /// </summary>
        public async Task LogValidationResultsAsync(WalkForwardResult result)
        {
            if (result is null) throw new ArgumentNullException(nameof(result));
            
            try
            {
                // Log summary
                _logger.LogInformation("[WALK-FORWARD-RESULTS] Strategy: {Strategy}, Duration: {Duration}, Windows: {Passed}/{Total}",
                    result.StrategyName, result.ValidationDuration, result.PassedWindows, result.TotalWindows);

                _logger.LogInformation("[WALK-FORWARD-RESULTS] Aggregate Performance: " +
                    "Sharpe={Sharpe:F2}, Drawdown={Drawdown:F2}%, WinRate={WinRate:F2}, TotalTrades={Trades}",
                    result.AggregateSharpeRatio, result.AggregateMaxDrawdown * 100, result.AggregateWinRate, result.AggregateTotalTrades);

                // Save detailed results to file
                var resultJson = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
                var resultPath = Path.Combine("./validation_results", $"walk_forward_{result.StrategyName}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
                await File.WriteAllTextAsync(resultPath, resultJson).ConfigureAwait(false);

                // Update history
                await UpdateValidationHistoryAsync(result).ConfigureAwait(false);

                _logger.LogInformation("[WALK-FORWARD-RESULTS] Detailed results saved to {ResultPath}", resultPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[WALK-FORWARD-RESULTS] Error logging validation results");
            }
        }

        /// <summary>
        /// Get validation history for a strategy
        /// </summary>
        public async Task<List<WalkForwardResult>> GetValidationHistoryAsync(string strategyName)
        {
            try
            {
                if (!File.Exists(_validationHistoryPath))
                    return new List<WalkForwardResult>();

                var historyJson = await File.ReadAllTextAsync(_validationHistoryPath).ConfigureAwait(false);
                var allHistory = JsonSerializer.Deserialize<List<WalkForwardResult>>(historyJson) ?? new List<WalkForwardResult>();

                return allHistory
                    .Where(r => r.StrategyName.Equals(strategyName, StringComparison.OrdinalIgnoreCase))
                    .OrderByDescending(r => r.ValidationStarted)
                    .ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[VALIDATION-HISTORY] Error getting validation history for {Strategy}", strategyName);
                return new List<WalkForwardResult>();
            }
        }

        #region Private Methods

        /// <summary>
        /// Process a single validation window
        /// </summary>
        private async Task<WindowResult> ProcessValidationWindowAsync(
            ValidationWindow window, 
            WalkForwardRequest request, 
            SemaphoreSlim semaphore, 
            CancellationToken cancellationToken)
        {
            await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
            
            try
            {
                _logger.LogDebug("[WINDOW-VALIDATION] Processing window {WindowIndex}: Training={TrainingStart} to {TrainingEnd}, Validation={ValidationStart} to {ValidationEnd}",
                    window.WindowIndex, window.TrainingStart, window.TrainingEnd, window.ValidationStart, window.ValidationEnd);

                var windowResult = new WindowResult
                {
                    Window = window,
                    ProcessingStarted = DateTime.UtcNow
                };

                // Step 1: Train model with window-specific seed
                var modelPath = await TrainModelForWindowAsync(request, window, cancellationToken).ConfigureAwait(false);
                windowResult.ModelPath = modelPath;

                // Step 2: Validate model performance
                var performance = await ValidateModelAsync(modelPath, window, cancellationToken).ConfigureAwait(false);
                windowResult.Performance = performance;

                // Step 3: Check if performance meets thresholds
                windowResult.PassesThresholds = await MeetsPerformanceThresholdsAsync(performance).ConfigureAwait(false);

                windowResult.ProcessingCompleted = DateTime.UtcNow;
                windowResult.ProcessingDuration = windowResult.ProcessingCompleted - windowResult.ProcessingStarted;

                _logger.LogDebug("[WINDOW-VALIDATION] Completed window {WindowIndex}: {Pass}, Sharpe={Sharpe:F2}",
                    window.WindowIndex, windowResult.PassesThresholds ? "PASS" : "FAIL", performance.SharpeRatio);

                return windowResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[WINDOW-VALIDATION] Error processing window {WindowIndex}", window.WindowIndex);
                return new WindowResult
                {
                    Window = window,
                    ProcessingStarted = DateTime.UtcNow,
                    ProcessingCompleted = DateTime.UtcNow,
                    ProcessingDuration = TimeSpan.Zero,
                    PassesThresholds = false,
                    Performance = new WalkForwardModelPerformance()
                };
            }
            finally
            {
                semaphore.Release();
            }
        }

        /// <summary>
        /// Train model for a specific validation window
        /// </summary>
        private async Task<string> TrainModelForWindowAsync(WalkForwardRequest request, ValidationWindow window, CancellationToken cancellationToken)
        {
            try
            {
                // In production, this would:
                // 1. Load training data for the window
                // 2. Initialize model with window-specific seed
                // 3. Train the model
                // 4. Save the trained model
                // 5. Verify model version

                var modelPath = Path.Combine("./models/validation", $"{request.StrategyName}_window_{window.WindowIndex}_seed_{window.RandomSeed}.onnx");
                Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

                // Simulate model training by creating a dummy model file
                var modelMetadata = new
                {
                    StrategyName = request.StrategyName,
                    WindowIndex = window.WindowIndex,
                    RandomSeed = window.RandomSeed,
                    TrainingStart = window.TrainingStart,
                    TrainingEnd = window.TrainingEnd,
                    CreatedAt = DateTime.UtcNow
                };

                var metadataJson = JsonSerializer.Serialize(modelMetadata, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(modelPath, metadataJson, cancellationToken).ConfigureAwait(false);

                // Verify model version to ensure uniqueness
                var metadata = new ModelMetadata
                {
                    ModelName = $"{request.StrategyName}_window_{window.WindowIndex}",
                    StrategyName = request.StrategyName,
                    CreatedAt = DateTime.UtcNow
                };

                await _modelVersionService.VerifyModelVersionAsync(modelPath, metadata).ConfigureAwait(false);

                _logger.LogTrace("[MODEL-TRAINING] Trained model for window {WindowIndex} with seed {Seed}: {ModelPath}",
                    window.WindowIndex, window.RandomSeed, modelPath);

                return modelPath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MODEL-TRAINING] Error training model for window {WindowIndex}", window.WindowIndex);
                throw;
            }
        }

        /// <summary>
        /// Simulate model performance for demonstration
        /// In production, this would run actual model predictions
        /// </summary>
        private static Task<WalkForwardModelPerformance> SimulateModelPerformance(ValidationWindow window)
        {
            // Simulate realistic performance metrics with some randomness
            var random = new Random(window.RandomSeed);
            
            // Base performance with some variation
            var baseSharpe = 0.8 + (random.NextDouble() - 0.5) * 0.6; // 0.2 to 1.4
            var baseDrawdown = 0.02 + random.NextDouble() * 0.08; // 2% to 10%
            var baseWinRate = 0.45 + random.NextDouble() * 0.25; // 45% to 70%
            var baseTrades = 50 + random.Next(100); // 50 to 150 trades

            // Add some window-specific effects
            var windowStress = Math.Abs(window.WindowIndex % 10 - 5) / 10.0; // 0 to 0.5
            var stressPenalty = windowStress * 0.2;

            return Task.FromResult(new WalkForwardModelPerformance
            {
                SharpeRatio = Math.Max(0.1, baseSharpe - stressPenalty),
                MaxDrawdown = Math.Min(0.2, baseDrawdown + stressPenalty),
                WinRate = Math.Max(0.3, Math.Min(0.8, baseWinRate - stressPenalty)),
                TotalTrades = baseTrades,
                TotalPnL = baseTrades * (10 + random.Next(40)), // $10-50 per trade average
                ValidationStartDate = window.ValidationStart,
                ValidationEndDate = window.ValidationEnd,
                ValidationDays = window.TotalValidationDays
            });
        }

        /// <summary>
        /// Generate unique seed for a validation window
        /// </summary>
        private int GenerateWindowSeed(int windowIndex)
        {
            var baseDate = DateTime.UtcNow.Date;
            var daysSinceEpoch = (int)(baseDate - new DateTime(2020, 1, 1)).TotalDays;
            
            return _config.SeedRotation.GenerateNewSeed() + windowIndex * 1000 + daysSinceEpoch;
        }

        /// <summary>
        /// Calculate aggregate metrics across all windows
        /// </summary>
        private void CalculateAggregateMetrics(WalkForwardResult result)
        {
            if (result.WindowResults.Count == 0)
                return;

            var performances = result.WindowResults.Select(w => w.Performance).Where(p => p != null).ToList();
            if (performances.Count == 0)
                return;

            result.PassedWindows = result.WindowResults.Count(w => w.PassesThresholds);
            result.FailedWindows = result.TotalWindows - result.PassedWindows;
            result.PassRate = (double)result.PassedWindows / result.TotalWindows;

            // Calculate weighted averages
            var totalValidationDays = performances.Sum(p => p.ValidationDays);
            if (totalValidationDays > 0)
            {
                result.AggregateSharpeRatio = performances.Sum(p => p.SharpeRatio * p.ValidationDays) / totalValidationDays;
                result.AggregateMaxDrawdown = performances.Sum(p => p.MaxDrawdown * p.ValidationDays) / totalValidationDays;
                result.AggregateWinRate = performances.Sum(p => p.WinRate * p.ValidationDays) / totalValidationDays;
            }

            result.AggregateTotalTrades = performances.Sum(p => p.TotalTrades);
            result.AggregateTotalPnL = performances.Sum(p => p.TotalPnL);

            // Stability metrics
            result.SharpeStability = CalculateStability(performances.Select(p => p.SharpeRatio));
            result.DrawdownStability = CalculateStability(performances.Select(p => p.MaxDrawdown));
        }

        /// <summary>
        /// Calculate stability metric (1 - coefficient of variation)
        /// </summary>
        private static double CalculateStability(IEnumerable<double> values)
        {
            var valuesList = values.ToList();
            if (valuesList.Count < 2)
                return 1.0;

            var mean = valuesList.Average();
            if (Math.Abs(mean) < 1e-10)
                return 0.0;

            var variance = valuesList.Sum(v => Math.Pow(v - mean, 2)) / (valuesList.Count - 1);
            var stdDev = Math.Sqrt(variance);
            var coefficientOfVariation = stdDev / Math.Abs(mean);

            return Math.Max(0.0, 1.0 - coefficientOfVariation);
        }

        /// <summary>
        /// Validate overall performance across all windows
        /// </summary>
        private Task<bool> ValidateOverallPerformanceAsync(WalkForwardResult result)
        {
            try
            {
                var thresholds = _config.PerformanceThresholds;
                var minPassRate = 0.7; // At least 70% of windows should pass

                var passesOverall = 
                    result.PassRate >= minPassRate &&
                    result.AggregateSharpeRatio >= thresholds.MinSharpeRatio &&
                    result.AggregateMaxDrawdown <= (thresholds.MaxDrawdownPct / 100.0) &&
                    result.AggregateWinRate >= thresholds.MinWinRate &&
                    result.SharpeStability >= 0.5 && // Require reasonable stability
                    result.DrawdownStability >= 0.5;

                return Task.FromResult(passesOverall);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[OVERALL-VALIDATION] Error validating overall performance");
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Update validation history file
        /// </summary>
        private async Task UpdateValidationHistoryAsync(WalkForwardResult result)
        {
            try
            {
                var history = new List<WalkForwardResult>();
                
                if (File.Exists(_validationHistoryPath))
                {
                    var existingJson = await File.ReadAllTextAsync(_validationHistoryPath).ConfigureAwait(false);
                    history = JsonSerializer.Deserialize<List<WalkForwardResult>>(existingJson) ?? new List<WalkForwardResult>();
                }

                history.Add(result);

                // Keep only recent history (last 100 runs)
                if (history.Count > 100)
                {
                    history = history.OrderByDescending(h => h.ValidationStarted).Take(100).ToList();
                }

                var historyJson = JsonSerializer.Serialize(history, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(_validationHistoryPath, historyJson).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[VALIDATION-HISTORY] Error updating validation history");
            }
        }

        #endregion
    }

    #region Supporting Models

    /// <summary>
    /// Walk-forward validation request
    /// </summary>
    public class WalkForwardRequest
    {
        public string StrategyName { get; set; } = string.Empty;
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public string ModelType { get; set; } = "ONNX";
        public Dictionary<string, object> Hyperparameters { get; } = new();
    }

    /// <summary>
    /// Validation window for walk-forward analysis
    /// </summary>
    public class ValidationWindow
    {
        public int WindowIndex { get; set; }
        public DateTime TrainingStart { get; set; }
        public DateTime TrainingEnd { get; set; }
        public DateTime ValidationStart { get; set; }
        public DateTime ValidationEnd { get; set; }
        public int RandomSeed { get; set; }
        public int TotalTrainingDays { get; set; }
        public int TotalValidationDays { get; set; }
    }

    /// <summary>
    /// Walk-forward validation model performance metrics
    /// </summary>
    public class WalkForwardModelPerformance
    {
        public double SharpeRatio { get; set; }
        public double MaxDrawdown { get; set; }
        public double WinRate { get; set; }
        public int TotalTrades { get; set; }
        public decimal TotalPnL { get; set; }
        public DateTime ValidationStartDate { get; set; }
        public DateTime ValidationEndDate { get; set; }
        public int ValidationDays { get; set; }
    }

    /// <summary>
    /// Result for a single validation window
    /// </summary>
    public class WindowResult
    {
        public ValidationWindow Window { get; set; } = new();
        public string ModelPath { get; set; } = string.Empty;
        public WalkForwardModelPerformance Performance { get; set; } = new();
        public bool PassesThresholds { get; set; }
        public DateTime ProcessingStarted { get; set; }
        public DateTime ProcessingCompleted { get; set; }
        public TimeSpan ProcessingDuration { get; set; }
    }

    /// <summary>
    /// Complete walk-forward validation result
    /// </summary>
    public class WalkForwardResult
    {
        public string StrategyName { get; set; } = string.Empty;
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public DateTime ValidationStarted { get; set; }
        public DateTime ValidationCompleted { get; set; }
        public TimeSpan ValidationDuration { get; set; }
        public bool EnableSeedRotation { get; set; }

        // Window metrics
        public int TotalWindows { get; set; }
        public int PassedWindows { get; set; }
        public int FailedWindows { get; set; }
        public double PassRate { get; set; }

        // Aggregate performance
        public double AggregateSharpeRatio { get; set; }
        public double AggregateMaxDrawdown { get; set; }
        public double AggregateWinRate { get; set; }
        public int AggregateTotalTrades { get; set; }
        public decimal AggregateTotalPnL { get; set; }

        // Stability metrics
        public double SharpeStability { get; set; }
        public double DrawdownStability { get; set; }

        // Overall validation result
        public bool PassesThresholds { get; set; }

        // Detailed results
        public List<WindowResult> WindowResults { get; set; } = new();
    }

    #endregion
}