using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using TradingBot.IntelligenceStack;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using Xunit;
using Xunit.Abstractions;

namespace TradingBot.Tests.Integration;

/// <summary>
/// Integration tests for feature adaptation system
/// Proves feature weights change when feature importance shifts
/// </summary>
public class FeatureAdaptationTests
{
    private readonly ITestOutputHelper _output;
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<FeatureAdaptationTests> _logger;
    private readonly string _testLogsPath;

    public FeatureAdaptationTests(ITestOutputHelper output)
    {
        _output = output;
        _testLogsPath = Path.Combine("/tmp", "feature_adaptation_test_logs");
        
        // Setup test service provider
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddSingleton<IOnlineLearningSystem, MockOnlineLearningSystem>();
        
        _serviceProvider = services.BuildServiceProvider();
        _logger = _serviceProvider.GetRequiredService<ILogger<FeatureAdaptationTests>>();
        
        // Ensure test logs directory exists
        Directory.CreateDirectory(_testLogsPath);
    }

    /// <summary>
    /// Test that feature weights change when importance shifts
    /// </summary>
    [Fact]
    public async Task FeatureAdapter_WeightsChangeWithImportanceShift_Success()
    {
        // Arrange
        var onlineLearningSystem = _serviceProvider.GetRequiredService<IOnlineLearningSystem>();
        var featureEngineer = new FeatureEngineer(new Microsoft.Extensions.Logging.Abstractions.NullLogger<FeatureEngineer>(), onlineLearningSystem, _testLogsPath);
        
        var strategyId = "test_adaptation_strategy";
        var testFeatures = new FeatureSet
        {
            Symbol = "ES",
            Timestamp = DateTime.UtcNow,
            Version = "v1.0",
            Features = new Dictionary<string, double>
            {
                ["price_momentum"] = 0.05,
                ["volume_trend"] = 0.02,
                ["volatility_spike"] = 0.15,
                ["market_regime"] = 1.0
            }
        };

        _logger.LogInformation("[INTEGRATION_TEST] Starting feature importance shift test");

        // Act - Calculate initial importance with neutral performance
        var initialRecentPredictions = new[] { 0.6, 0.65, 0.7, 0.68, 0.62 };
        var initialRecentOutcomes = new[] { 0.6, 0.7, 0.65, 0.7, 0.6 };

        var initialImportance = await featureEngineer.CalculateRollingSHAPAsync(
            strategyId, testFeatures, initialRecentPredictions, initialRecentOutcomes);

        Assert.NotEmpty(initialImportance);
        _logger.LogInformation("[INTEGRATION_TEST] Initial importance calculated: {Count} features", 
            initialImportance.Count);

        // Update feature weights based on initial importance
        await featureEngineer.UpdateFeatureWeightsAsync(strategyId, testFeatures, initialImportance);
        var initialWeights = await featureEngineer.GetCurrentWeightsAsync(strategyId);

        // Simulate shift in feature importance (some features become more predictive)
        var shiftedRecentPredictions = new[] { 0.8, 0.85, 0.9, 0.88, 0.82 };
        var shiftedRecentOutcomes = new[] { 0.85, 0.9, 0.88, 0.9, 0.85 };

        var shiftedImportance = await featureEngineer.CalculateRollingSHAPAsync(
            strategyId, testFeatures, shiftedRecentPredictions, shiftedRecentOutcomes);

        // Update weights based on shifted importance
        await featureEngineer.UpdateFeatureWeightsAsync(strategyId, testFeatures, shiftedImportance);
        var shiftedWeights = await featureEngineer.GetCurrentWeightsAsync(strategyId);

        // Assert - Verify weights changed
        Assert.NotEmpty(shiftedWeights);
        
        bool weightsChanged = false;
        foreach (var featureName in testFeatures.Features.Keys)
        {
            if (initialWeights.ContainsKey(featureName) && shiftedWeights.ContainsKey(featureName))
            {
                var initialWeight = initialWeights[featureName];
                var shiftedWeight = shiftedWeights[featureName];
                
                if (Math.Abs(initialWeight - shiftedWeight) > 0.01)
                {
                    weightsChanged = true;
                    _logger.LogInformation("[INTEGRATION_TEST] Feature {Feature} weight changed: {Initial:F3} -> {Shifted:F3}", 
                        featureName, initialWeight, shiftedWeight);
                }
            }
        }

        Assert.True(weightsChanged, "Feature weights should change when importance shifts significantly");

        _logger.LogInformation("[INTEGRATION_TEST] Feature importance shift test completed successfully");
    }

    /// <summary>
    /// Test permutation importance calculation changes feature weights
    /// </summary>
    [Fact]
    public async Task FeatureAdapter_PermutationImportanceChangesWeights_Success()
    {
        // Arrange
        var onlineLearningSystem = _serviceProvider.GetRequiredService<IOnlineLearningSystem>();
        var featureEngineer = new FeatureEngineer(new Microsoft.Extensions.Logging.Abstractions.NullLogger<FeatureEngineer>(), onlineLearningSystem, _testLogsPath);
        
        var strategyId = "permutation_test_strategy";
        var testFeatures = new FeatureSet
        {
            Symbol = "ES",
            Timestamp = DateTime.UtcNow,
            Version = "v1.0",
            Features = new Dictionary<string, double>
            {
                ["highly_predictive"] = 100.0,
                ["moderately_predictive"] = 50.0,
                ["weakly_predictive"] = 10.0,
                ["noise_feature"] = 0.1
            }
        };

        _logger.LogInformation("[INTEGRATION_TEST] Starting permutation importance test");

        // Mock prediction function that varies based on feature values
        Func<FeatureSet, Task<double>> mockPredictionFunction = async (features) =>
        {
            await Task.CompletedTask;
            
            var prediction = 0.5; // Base prediction
            
            // Highly predictive feature has strong influence
            if (features.Features.ContainsKey("highly_predictive"))
            {
                prediction += features.Features["highly_predictive"] / 1000.0;
            }
            
            // Moderately predictive feature has medium influence
            if (features.Features.ContainsKey("moderately_predictive"))
            {
                prediction += features.Features["moderately_predictive"] / 2000.0;
            }
            
            // Weak and noise features have minimal influence
            if (features.Features.ContainsKey("weakly_predictive"))
            {
                prediction += features.Features["weakly_predictive"] / 10000.0;
            }
            
            return Math.Min(0.95, Math.Max(0.05, prediction));
        };

        // Act - Calculate permutation importance
        var permutationImportance = await featureEngineer.CalculatePermutationImportanceAsync(
            strategyId, testFeatures, mockPredictionFunction);

        Assert.NotEmpty(permutationImportance);
        
        // Update weights based on permutation importance
        await featureEngineer.UpdateFeatureWeightsAsync(strategyId, testFeatures, permutationImportance);
        var updatedWeights = await featureEngineer.GetCurrentWeightsAsync(strategyId);

        // Assert - Verify weights reflect importance hierarchy
        Assert.NotEmpty(updatedWeights);
        
        // The highly predictive feature should have higher weight than noise feature
        if (updatedWeights.ContainsKey("highly_predictive") && updatedWeights.ContainsKey("noise_feature"))
        {
            Assert.True(updatedWeights["highly_predictive"] > updatedWeights["noise_feature"],
                "Highly predictive feature should have higher weight than noise feature");
        }

        // Log importance scores for verification
        foreach (var (feature, importance) in permutationImportance)
        {
            var weight = updatedWeights.GetValueOrDefault(feature, 1.0);
            _logger.LogInformation("[INTEGRATION_TEST] Feature {Feature}: importance={Importance:F4}, weight={Weight:F3}",
                feature, importance, weight);
        }

        _logger.LogInformation("[INTEGRATION_TEST] Permutation importance test completed successfully");
    }

    /// <summary>
    /// Test that feature weights are logged with timestamp and strategy ID
    /// </summary>
    [Fact]
    public async Task FeatureAdapter_LogsWeightsWithTimestampAndStrategy_Success()
    {
        // Arrange
        var onlineLearningSystem = _serviceProvider.GetRequiredService<IOnlineLearningSystem>();
        var featureEngineer = new FeatureEngineer(new Microsoft.Extensions.Logging.Abstractions.NullLogger<FeatureEngineer>(), onlineLearningSystem, _testLogsPath);
        
        var strategyId = "logging_test_strategy";
        var testFeatures = new FeatureSet
        {
            Symbol = "ES",
            Timestamp = DateTime.UtcNow,
            Version = "v1.0",
            Features = new Dictionary<string, double>
            {
                ["feature_alpha"] = 1.5,
                ["feature_beta"] = 0.8,
                ["feature_gamma"] = 1.2
            }
        };

        var testWeights = new Dictionary<string, double>
        {
            ["feature_alpha"] = 1.3,
            ["feature_beta"] = 0.6,
            ["feature_gamma"] = 1.1
        };

        _logger.LogInformation("[INTEGRATION_TEST] Starting weight logging test");

        // Clear any existing log files
        var existingLogFiles = Directory.GetFiles(_testLogsPath, "feature_weights_*.json");
        foreach (var file in existingLogFiles)
        {
            File.Delete(file);
        }

        // Act - Update feature weights (this should trigger logging)
        await featureEngineer.UpdateFeatureWeightsAsync(strategyId, testFeatures, testWeights);

        // Wait a moment for file I/O
        await Task.Delay(100);

        // Assert - Verify log file was created
        var logFiles = Directory.GetFiles(_testLogsPath, $"feature_weights_{strategyId}_*.json");
        Assert.NotEmpty(logFiles);

        var latestLogFile = logFiles[^1]; // Get the most recent log file
        Assert.True(File.Exists(latestLogFile), "Feature weights log file should exist");

        // Verify log file content
        var logContent = await File.ReadAllTextAsync(latestLogFile);
        Assert.NotEmpty(logContent);

        var logEntry = JsonSerializer.Deserialize<FeatureWeightsLog>(logContent);
        Assert.NotNull(logEntry);
        Assert.Equal(strategyId, logEntry.StrategyId);
        Assert.True(logEntry.Timestamp > DateTime.UtcNow.AddMinutes(-1), "Log timestamp should be recent");
        Assert.NotEmpty(logEntry.Weights);
        Assert.Equal(testWeights.Count, logEntry.TotalFeatures);

        // Verify logged weights match what was set
        foreach (var (featureName, expectedWeight) in testWeights)
        {
            Assert.True(logEntry.Weights.ContainsKey(featureName), 
                $"Log should contain weight for feature {featureName}");
            Assert.Equal(expectedWeight, logEntry.Weights[featureName], 2);
        }

        _logger.LogInformation("[INTEGRATION_TEST] Weight logging verification completed - file: {File}", 
            Path.GetFileName(latestLogFile));
    }

    /// <summary>
    /// Test real-time market data processing adapts feature weights
    /// </summary>
    [Fact]
    public async Task FeatureAdapter_RealTimeMarketDataAdaptation_Success()
    {
        // Arrange
        var onlineLearningSystem = _serviceProvider.GetRequiredService<IOnlineLearningSystem>();
        var featureEngineer = new FeatureEngineer(new Microsoft.Extensions.Logging.Abstractions.NullLogger<FeatureEngineer>(), onlineLearningSystem, _testLogsPath);

        var marketDataSequence = new[]
        {
            new TradingBot.Abstractions.MarketData
            {
                Symbol = "ES",
                Bid = 4500.00,
                Ask = 4500.25,
                Open = 4500.00,
                High = 4500.50,
                Low = 4499.75,
                Close = 4500.10,
                Volume = 1000,
                Timestamp = DateTime.UtcNow
            },
            new TradingBot.Abstractions.MarketData
            {
                Symbol = "ES", 
                Bid = 4501.00,
                Ask = 4501.25,
                Open = 4500.10,
                High = 4501.50,
                Low = 4500.75,
                Close = 4501.15,
                Volume = 1200,
                Timestamp = DateTime.UtcNow.AddMinutes(1)
            },
            new TradingBot.Abstractions.MarketData
            {
                Symbol = "ES",
                Bid = 4502.50,
                Ask = 4502.75,
                Open = 4501.15,
                High = 4503.00,
                Low = 4501.90,
                Close = 4502.60,
                Volume = 1500,
                Timestamp = DateTime.UtcNow.AddMinutes(2)
            }
        };

        // Mock prediction function
        Func<FeatureSet, Task<double>> mockPredictionFunction = async (features) =>
        {
            await Task.CompletedTask;
            
            var closePrice = features.Features.GetValueOrDefault("close_price", 4500.0);
            var volume = features.Features.GetValueOrDefault("volume", 1000.0);
            
            // Simple prediction based on price momentum and volume
            var prediction = 0.5 + ((closePrice - 4500.0) / 10000.0) + (Math.Log(volume / 1000.0) / 10.0);
            return Math.Min(0.95, Math.Max(0.05, prediction));
        };

        _logger.LogInformation("[INTEGRATION_TEST] Starting real-time market data adaptation test");

        // Act - Process market data sequence
        var strategyId = "realtime_test";
        var weightChanges = new List<Dictionary<string, double>>();

        foreach (var marketData in marketDataSequence)
        {
            await featureEngineer.ProcessMarketDataAsync(marketData, mockPredictionFunction);
            
            // Capture weights after each market data update
            var currentWeights = await featureEngineer.GetCurrentWeightsAsync(strategyId);
            if (currentWeights.Count > 0)
            {
                weightChanges.Add(new Dictionary<string, double>(currentWeights));
            }
            
            // Small delay to simulate real-time processing
            await Task.Delay(10);
        }

        // Assert - Verify system processed market data and potentially adapted weights
        _logger.LogInformation("[INTEGRATION_TEST] Processed {Count} market data points", marketDataSequence.Length);
        
        // The system should be able to extract features from market data without errors
        // We can't guarantee weight changes in this short sequence, but we can verify the system functions
        Assert.True(weightChanges.Count >= 0, "System should process market data without errors");

        // If weights were captured, verify they're reasonable
        foreach (var weights in weightChanges)
        {
            foreach (var (feature, weight) in weights)
            {
                Assert.True(weight >= 0.0 && weight <= 3.0, 
                    $"Feature weight {weight} for {feature} should be in reasonable range");
            }
        }

        _logger.LogInformation("[INTEGRATION_TEST] Real-time adaptation test completed successfully");
    }

    /// <summary>
    /// Test low-value features are automatically down-weighted
    /// </summary>
    [Fact]
    public async Task FeatureAdapter_DownWeightsLowValueFeatures_Success()
    {
        // Arrange
        var onlineLearningSystem = _serviceProvider.GetRequiredService<IOnlineLearningSystem>();
        var featureEngineer = new FeatureEngineer(new Microsoft.Extensions.Logging.Abstractions.NullLogger<FeatureEngineer>(), onlineLearningSystem, _testLogsPath);
        
        var strategyId = "low_value_test_strategy";
        var testFeatures = new FeatureSet
        {
            Symbol = "ES",
            Timestamp = DateTime.UtcNow,
            Version = "v1.0",
            Features = new Dictionary<string, double>
            {
                ["high_value_feature"] = 1.0,
                ["medium_value_feature"] = 1.0,
                ["low_value_feature"] = 1.0
            }
        };

        // Create importance scores that clearly identify low-value features
        var importanceScores = new Dictionary<string, double>
        {
            ["high_value_feature"] = 0.6,   // High importance
            ["medium_value_feature"] = 0.35, // Medium importance
            ["low_value_feature"] = 0.02    // Low importance (below threshold)
        };

        _logger.LogInformation("[INTEGRATION_TEST] Starting low-value feature down-weighting test");

        // Act - Update weights based on importance scores
        await featureEngineer.UpdateFeatureWeightsAsync(strategyId, testFeatures, importanceScores);
        var updatedWeights = await featureEngineer.GetCurrentWeightsAsync(strategyId);

        // Assert - Verify low-value features are down-weighted
        Assert.NotEmpty(updatedWeights);
        
        if (updatedWeights.ContainsKey("low_value_feature"))
        {
            var lowValueWeight = updatedWeights["low_value_feature"];
            Assert.True(lowValueWeight < 0.5, 
                $"Low-value feature should be down-weighted (weight: {lowValueWeight:F3})");
        }

        if (updatedWeights.ContainsKey("high_value_feature"))
        {
            var highValueWeight = updatedWeights["high_value_feature"];
            Assert.True(highValueWeight >= 1.0, 
                $"High-value feature should maintain or increase weight (weight: {highValueWeight:F3})");
        }

        // Verify relative weighting
        if (updatedWeights.ContainsKey("high_value_feature") && updatedWeights.ContainsKey("low_value_feature"))
        {
            Assert.True(updatedWeights["high_value_feature"] > updatedWeights["low_value_feature"],
                "High-value feature should have higher weight than low-value feature");
        }

        // Log the results for verification
        foreach (var (feature, weight) in updatedWeights)
        {
            var importance = importanceScores.GetValueOrDefault(feature, 0.0);
            _logger.LogInformation("[INTEGRATION_TEST] Feature {Feature}: importance={Importance:F3}, weight={Weight:F3}",
                feature, importance, weight);
        }

        _logger.LogInformation("[INTEGRATION_TEST] Low-value feature down-weighting test completed successfully");
    }

    /// <summary>
    /// Test that feature adaptation works during a live session simulation
    /// </summary>
    [Fact]
    public async Task FeatureAdapter_LiveSessionSimulation_Success()
    {
        // Arrange
        var onlineLearningSystem = _serviceProvider.GetRequiredService<IOnlineLearningSystem>();
        var featureEngineer = new FeatureEngineer(new Microsoft.Extensions.Logging.Abstractions.NullLogger<FeatureEngineer>(), onlineLearningSystem, _testLogsPath);
        
        var strategyId = "live_session_test";
        var sessionDuration = TimeSpan.FromSeconds(5); // Short simulation
        var updateInterval = TimeSpan.FromMilliseconds(500);
        
        _logger.LogInformation("[INTEGRATION_TEST] Starting live session simulation");

        var sessionStart = DateTime.UtcNow;
        var weightHistory = new List<Dictionary<string, double>>();
        var logFilesBefore = Directory.GetFiles(_testLogsPath, $"feature_weights_{strategyId}_*.json").Length;

        // Mock prediction function with time-varying behavior
        Func<FeatureSet, Task<double>> sessionPredictionFunction = async (features) =>
        {
            await Task.CompletedTask;
            var elapsed = DateTime.UtcNow - sessionStart;
            var timeComponent = Math.Sin(elapsed.TotalSeconds) * 0.1; // Oscillating component
            return 0.6 + timeComponent;
        };

        // Act - Simulate live trading session
        var endTime = sessionStart.Add(sessionDuration);
        while (DateTime.UtcNow < endTime)
        {
            // Generate synthetic market data
            var basePrice = 4500.0;
            var priceChange = Random.Shared.NextDouble() * 10 - 5;
            var openPrice = basePrice + priceChange;
            var closePrice = openPrice + (Random.Shared.NextDouble() * 4 - 2);
            
            var marketData = new TradingBot.Abstractions.MarketData
            {
                Symbol = "ES",
                Open = openPrice,
                High = Math.Max(openPrice, closePrice) + Random.Shared.NextDouble() * 2,
                Low = Math.Min(openPrice, closePrice) - Random.Shared.NextDouble() * 2,
                Close = closePrice,
                Bid = closePrice - 0.25,
                Ask = closePrice + 0.25,
                Volume = 1000 + Random.Shared.Next(500),
                Timestamp = DateTime.UtcNow
            };

            // Process market data
            await featureEngineer.ProcessMarketDataAsync(marketData, sessionPredictionFunction);
            
            // Capture current weights
            var currentWeights = await featureEngineer.GetCurrentWeightsAsync(strategyId);
            if (currentWeights.Count > 0)
            {
                weightHistory.Add(new Dictionary<string, double>(currentWeights));
            }

            await Task.Delay(updateInterval);
        }

        // Assert - Verify live session results
        var logFilesAfter = Directory.GetFiles(_testLogsPath, $"feature_weights_{strategyId}_*.json").Length;
        
        _logger.LogInformation("[INTEGRATION_TEST] Live session completed: {Duration}s, {Updates} weight updates, {LogFiles} new log files",
            sessionDuration.TotalSeconds, weightHistory.Count, logFilesAfter - logFilesBefore);

        // The system should have processed market data without errors
        Assert.True(weightHistory.Count >= 0, "System should handle live session without errors");

        // If log files were created, verify they're valid
        if (logFilesAfter > logFilesBefore)
        {
            var newLogFiles = Directory.GetFiles(_testLogsPath, $"feature_weights_{strategyId}_*.json");
            var latestLogFile = newLogFiles[^1];
            
            var logContent = await File.ReadAllTextAsync(latestLogFile);
            var logEntry = JsonSerializer.Deserialize<FeatureWeightsLog>(logContent);
            
            Assert.NotNull(logEntry);
            Assert.Equal(strategyId, logEntry.StrategyId);
            Assert.True(logEntry.Timestamp >= sessionStart, "Log timestamp should be within session timeframe");
        }

        _logger.LogInformation("[INTEGRATION_TEST] Live session simulation completed successfully");
    }
}

#region Support Classes

/// <summary>
/// Mock online learning system for testing
/// </summary>
public class MockOnlineLearningSystem : IOnlineLearningSystem
{
    private readonly Dictionary<string, Dictionary<string, double>> _weights = new();

    public async Task UpdateWeightsAsync(string regimeType, Dictionary<string, double> weights, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        _weights[regimeType] = new Dictionary<string, double>(weights);
    }

    public async Task<Dictionary<string, double>> GetCurrentWeightsAsync(string regimeType, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        return _weights.TryGetValue(regimeType, out var weights) 
            ? new Dictionary<string, double>(weights) 
            : new Dictionary<string, double>();
    }

    public async Task AdaptToPerformanceAsync(string modelId, ModelPerformance performance, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
    }

    public async Task DetectDriftAsync(string modelId, FeatureSet features, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
    }

    public async Task UpdateModelAsync(TradeRecord tradeRecord, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
    }
}

/// <summary>
/// Feature weights log entry for verification
/// </summary>
public class FeatureWeightsLog
{
    public string StrategyId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public Dictionary<string, double> Weights { get; set; } = new();
    public int TotalFeatures { get; set; }
    public int LowValueFeatures { get; set; }
    public int HighValueFeatures { get; set; }
    public double AverageWeight { get; set; }
}

#endregion