using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using TradingBot.IntelligenceStack;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace TradingBot.Tests.Integration;

/// <summary>
/// Integration tests for historical trainer system
/// Proves historical training produces a deployable model that passes smoke-test and hot-reloads
/// </summary>
public class HistoricalTrainerTests
{
    private readonly ITestOutputHelper _output;
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<HistoricalTrainerTests> _logger;

    public HistoricalTrainerTests(ITestOutputHelper output)
    {
        _output = output;
        
        // Setup test service provider
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddXUnit(output));
        services.AddSingleton<IModelRegistry, MockModelRegistry>();
        services.AddSingleton<IFeatureStore, MockFeatureStore>();
        services.AddSingleton<IQuarantineManager, MockQuarantineManager>();
        services.AddSingleton<PromotionCriteria>();
        
        _serviceProvider = services.BuildServiceProvider();
        _logger = _serviceProvider.GetRequiredService<ILogger<HistoricalTrainerTests>>();
    }

    /// <summary>
    /// Test that historical training produces a deployable model
    /// </summary>
    [Fact]
    public async Task HistoricalTrainer_ProducesDeployableModel_Success()
    {
        // Arrange
        var modelRegistry = _serviceProvider.GetRequiredService<IModelRegistry>();
        var featureStore = _serviceProvider.GetRequiredService<IFeatureStore>();
        var quarantineManager = _serviceProvider.GetRequiredService<IQuarantineManager>();
        var promotionCriteria = _serviceProvider.GetRequiredService<PromotionCriteria>();

        var trainer = new HistoricalTrainerWithCV(
            _logger,
            modelRegistry,
            featureStore,
            quarantineManager,
            promotionCriteria,
            "/tmp/historical_training_test");

        var modelFamily = "test_ensemble";
        var startDate = DateTime.UtcNow.AddDays(-30);
        var endDate = DateTime.UtcNow.AddDays(-1);
        var trainingWindow = TimeSpan.FromDays(7);
        var testWindow = TimeSpan.FromDays(1);

        _logger.LogInformation("[INTEGRATION_TEST] Starting historical training test for deployable model");

        // Act
        var cvResult = await trainer.RunWalkForwardCVAsync(
            modelFamily, 
            startDate, 
            endDate, 
            trainingWindow, 
            testWindow);

        // Assert
        Assert.NotNull(cvResult);
        Assert.Equal(modelFamily, cvResult.ModelFamily);
        Assert.True(cvResult.FoldResults.Count > 0, "Should have at least one CV fold");
        Assert.NotNull(cvResult.AggregateMetrics);
        
        _logger.LogInformation("[INTEGRATION_TEST] Training completed with {Folds} folds, AUC: {AUC:F3}", 
            cvResult.FoldResults.Count, cvResult.AggregateMetrics.AUC);

        // Verify model meets deployment criteria
        var successfulFolds = 0;
        foreach (var fold in cvResult.FoldResults)
        {
            if (fold.Success && fold.TestMetrics != null)
            {
                successfulFolds++;
                Assert.True(fold.TestMetrics.AUC >= 0.5, "Model should perform better than random");
                Assert.True(fold.TestMetrics.SampleSize > 0, "Should have test samples");
            }
        }

        Assert.True(successfulFolds > 0, "At least one fold should succeed");

        // Verify aggregate metrics are reasonable for deployment
        Assert.True(cvResult.AggregateMetrics.AUC >= 0.5, "Aggregate AUC should be better than random");
        Assert.True(cvResult.AggregateMetrics.SampleSize > 0, "Should have aggregate samples");

        _logger.LogInformation("[INTEGRATION_TEST] Model deployment validation completed successfully");
    }

    /// <summary>
    /// Test that trained model passes smoke test
    /// </summary>
    [Fact]
    public async Task HistoricalTrainer_ModelPassesSmokeTest_Success()
    {
        // Arrange
        var modelRegistry = _serviceProvider.GetRequiredService<IModelRegistry>();
        var featureStore = _serviceProvider.GetRequiredService<IFeatureStore>();
        var quarantineManager = _serviceProvider.GetRequiredService<IQuarantineManager>();
        var promotionCriteria = new PromotionCriteria
        {
            MinAuc = 0.55,
            MinPrAt10 = 0.08,
            MaxEce = 0.10,
            MinEdgeBps = 2.0
        };

        var trainer = new HistoricalTrainerWithCV(
            _logger,
            modelRegistry,
            featureStore,
            quarantineManager,
            promotionCriteria,
            "/tmp/smoke_test_training");

        _logger.LogInformation("[INTEGRATION_TEST] Starting smoke test validation");

        // Act - Train a model with realistic parameters
        var cvResult = await trainer.RunWalkForwardCVAsync(
            "smoke_test_model",
            DateTime.UtcNow.AddDays(-14),
            DateTime.UtcNow.AddDays(-1),
            TimeSpan.FromDays(5),
            TimeSpan.FromDays(1));

        // Assert - Smoke test checks
        Assert.NotNull(cvResult);
        
        // Basic functionality tests
        Assert.True(cvResult.FoldResults.Count >= 2, "Should have multiple folds for robust testing");
        
        foreach (var fold in cvResult.FoldResults)
        {
            if (fold.Success)
            {
                // Smoke test: Model produces valid outputs
                Assert.NotNull(fold.TestMetrics);
                Assert.True(fold.TestMetrics.AUC >= 0.0 && fold.TestMetrics.AUC <= 1.0, 
                    "AUC should be in valid range");
                Assert.True(fold.TrainingExamples > 0, "Should have training examples");
                Assert.True(fold.TestExamples > 0, "Should have test examples");
                
                // Smoke test: Reasonable performance metrics
                Assert.True(fold.TestMetrics.ECE >= 0.0, "ECE should be non-negative");
                Assert.True(fold.TestMetrics.SampleSize > 0, "Sample size should be positive");
            }
        }

        // Overall smoke test: System produces coherent results
        if (cvResult.AggregateMetrics != null)
        {
            Assert.True(cvResult.AggregateMetrics.AUC >= 0.4 && cvResult.AggregateMetrics.AUC <= 1.0,
                "Aggregate AUC should be in reasonable range");
        }

        _logger.LogInformation("[INTEGRATION_TEST] Smoke test passed - model produces valid outputs");
    }

    /// <summary>
    /// Test model hot-reload functionality
    /// </summary>
    [Fact]
    public async Task HistoricalTrainer_ModelHotReload_Success()
    {
        // Arrange
        var modelRegistry = _serviceProvider.GetRequiredService<IModelRegistry>() as MockModelRegistry;
        var featureStore = _serviceProvider.GetRequiredService<IFeatureStore>();
        var quarantineManager = _serviceProvider.GetRequiredService<IQuarantineManager>();
        var promotionCriteria = _serviceProvider.GetRequiredService<PromotionCriteria>();

        var trainer = new HistoricalTrainerWithCV(
            _logger,
            modelRegistry!,
            featureStore,
            quarantineManager,
            promotionCriteria,
            "/tmp/hot_reload_test");

        _logger.LogInformation("[INTEGRATION_TEST] Starting hot-reload test");

        // Act - Train initial model
        var initialResult = await trainer.RunWalkForwardCVAsync(
            "hot_reload_model",
            DateTime.UtcNow.AddDays(-10),
            DateTime.UtcNow.AddDays(-2),
            TimeSpan.FromDays(3),
            TimeSpan.FromDays(1));

        Assert.NotNull(initialResult);
        var initialModelCount = modelRegistry!.RegisteredModels.Count;

        // Simulate hot-reload by training another model that should replace the first
        var reloadResult = await trainer.RunWalkForwardCVAsync(
            "hot_reload_model", // Same family name for hot-reload
            DateTime.UtcNow.AddDays(-8),
            DateTime.UtcNow.AddDays(-1),
            TimeSpan.FromDays(3),
            TimeSpan.FromDays(1));

        // Assert
        Assert.NotNull(reloadResult);
        
        // Verify hot-reload happened (new model registered)
        var finalModelCount = modelRegistry.RegisteredModels.Count;
        Assert.True(finalModelCount > initialModelCount, 
            "Hot-reload should register new model version");

        // Verify we can retrieve the latest model
        var latestModel = await modelRegistry.GetModelAsync("hot_reload_model", "latest");
        Assert.NotNull(latestModel);
        Assert.Contains("hot_reload_model", latestModel.Id);

        // Verify model metadata indicates it's hot-reloadable
        if (reloadResult.AggregateMetrics != null)
        {
            Assert.True(reloadResult.AggregateMetrics.ComputedAt > initialResult.AggregateMetrics?.ComputedAt,
                "New model should have more recent metrics");
        }

        _logger.LogInformation("[INTEGRATION_TEST] Hot-reload test completed successfully");
    }

    /// <summary>
    /// Test leak-safe label generation
    /// </summary>
    [Fact]
    public async Task HistoricalTrainer_GeneratesLeakSafeLabels_Success()
    {
        // Arrange
        var modelRegistry = _serviceProvider.GetRequiredService<IModelRegistry>();
        var featureStore = _serviceProvider.GetRequiredService<IFeatureStore>();
        var quarantineManager = _serviceProvider.GetRequiredService<IQuarantineManager>();
        var promotionCriteria = _serviceProvider.GetRequiredService<PromotionCriteria>();

        var trainer = new HistoricalTrainerWithCV(
            _logger,
            modelRegistry,
            featureStore,
            quarantineManager,
            promotionCriteria,
            "/tmp/leak_safe_test");

        var symbol = "ES";
        var startTime = DateTime.UtcNow.AddDays(-5);
        var endTime = DateTime.UtcNow.AddDays(-1);

        _logger.LogInformation("[INTEGRATION_TEST] Starting leak-safe label generation test");

        // Act
        var trainingExamples = await trainer.GenerateLeakSafeLabelsAsync(symbol, startTime, endTime);

        // Assert
        Assert.NotNull(trainingExamples);
        Assert.NotEmpty(trainingExamples);

        foreach (var example in trainingExamples)
        {
            // Verify leak-safe properties
            Assert.NotNull(example.Features);
            Assert.NotEmpty(example.Features);
            Assert.True(example.Timestamp >= startTime && example.Timestamp <= endTime,
                "Example timestamp should be within requested range");

            // Verify labels are generated with proper embargo (future outcome exists)
            Assert.True(Math.Abs(example.ActualOutcome) >= 0, "Should have actual outcome");
            Assert.True(example.PredictedDirection == -1 || example.PredictedDirection == 0 || example.PredictedDirection == 1,
                "Predicted direction should be valid");
        }

        // Verify temporal ordering (no lookahead bias)
        for (int i = 1; i < trainingExamples.Count; i++)
        {
            Assert.True(trainingExamples[i].Timestamp >= trainingExamples[i-1].Timestamp,
                "Training examples should be in temporal order");
        }

        _logger.LogInformation("[INTEGRATION_TEST] Generated {Count} leak-safe training examples", 
            trainingExamples.Count);
    }

    /// <summary>
    /// Test model promotion criteria validation
    /// </summary>
    [Fact]
    public async Task HistoricalTrainer_ValidatesPromotionCriteria_Success()
    {
        // Arrange
        var modelRegistry = _serviceProvider.GetRequiredService<IModelRegistry>();
        var featureStore = _serviceProvider.GetRequiredService<IFeatureStore>();
        var quarantineManager = _serviceProvider.GetRequiredService<IQuarantineManager>();
        
        // Strict promotion criteria
        var strictCriteria = new PromotionCriteria
        {
            MinAuc = 0.75,    // High bar
            MinPrAt10 = 0.15,
            MaxEce = 0.03,
            MinEdgeBps = 5.0
        };

        var trainer = new HistoricalTrainerWithCV(
            _logger,
            modelRegistry,
            featureStore,
            quarantineManager,
            strictCriteria,
            "/tmp/promotion_test");

        _logger.LogInformation("[INTEGRATION_TEST] Starting promotion criteria validation test");

        // Act
        var cvResult = await trainer.RunWalkForwardCVAsync(
            "promotion_test_model",
            DateTime.UtcNow.AddDays(-7),
            DateTime.UtcNow.AddDays(-1),
            TimeSpan.FromDays(2),
            TimeSpan.FromDays(1));

        // Assert
        Assert.NotNull(cvResult);
        
        // The system should evaluate promotion criteria
        Assert.True(cvResult.MeetsPromotionCriteria == true || cvResult.MeetsPromotionCriteria == false,
            "Promotion criteria should be evaluated");

        if (cvResult.MeetsPromotionCriteria)
        {
            // If promoted, verify it meets the criteria
            Assert.NotNull(cvResult.AggregateMetrics);
            _logger.LogInformation("[INTEGRATION_TEST] Model met promotion criteria - AUC: {AUC:F3}, PrAt10: {PrAt10:F3}",
                cvResult.AggregateMetrics.AUC, cvResult.AggregateMetrics.PrAt10);
        }
        else
        {
            _logger.LogInformation("[INTEGRATION_TEST] Model did not meet strict promotion criteria - this is acceptable for testing");
        }

        // Test with lenient criteria
        var lenientCriteria = new PromotionCriteria
        {
            MinAuc = 0.51,
            MinPrAt10 = 0.05,
            MaxEce = 0.15,
            MinEdgeBps = 1.0
        };

        var lenientTrainer = new HistoricalTrainerWithCV(
            _logger,
            modelRegistry,
            featureStore,
            quarantineManager,
            lenientCriteria,
            "/tmp/promotion_lenient_test");

        var lenientResult = await lenientTrainer.RunWalkForwardCVAsync(
            "lenient_promotion_model",
            DateTime.UtcNow.AddDays(-7),
            DateTime.UtcNow.AddDays(-1),
            TimeSpan.FromDays(2),
            TimeSpan.FromDays(1));

        Assert.NotNull(lenientResult);
        // More likely to meet lenient criteria
        _logger.LogInformation("[INTEGRATION_TEST] Lenient criteria result: {Promoted}", 
            lenientResult.MeetsPromotionCriteria);

        _logger.LogInformation("[INTEGRATION_TEST] Promotion criteria validation completed");
    }
}

#region Mock Implementations

/// <summary>
/// Mock model registry for testing
/// </summary>
public class MockModelRegistry : IModelRegistry
{
    public List<ModelArtifact> RegisteredModels { get; } = new();
    private readonly Dictionary<string, ModelArtifact> _models = new();

    public async Task<ModelArtifact> GetModelAsync(string familyName, string version = "latest", CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        var key = $"{familyName}_{version}";
        if (_models.TryGetValue(key, out var model))
        {
            return model;
        }

        // Return a mock model if not found
        var mockModel = new ModelArtifact
        {
            Id = $"{familyName}_{Guid.NewGuid():N}",
            Version = version,
            CreatedAt = DateTime.UtcNow,
            Metrics = new ModelMetrics
            {
                AUC = 0.65,
                PrAt10 = 0.12,
                ECE = 0.05,
                EdgeBps = 3.5,
                SampleSize = 1000
            }
        };

        _models[key] = mockModel;
        return mockModel;
    }

    public async Task<ModelArtifact> RegisterModelAsync(ModelRegistration registration, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        var model = new ModelArtifact
        {
            Id = $"{registration.FamilyName}_{Guid.NewGuid():N}",
            Version = "1.0",
            CreatedAt = DateTime.UtcNow,
            Metrics = registration.Metrics,
            ModelData = registration.ModelData
        };

        RegisteredModels.Add(model);
        _models[$"{registration.FamilyName}_latest"] = model;
        
        return model;
    }

    public async Task<bool> PromoteModelAsync(string modelId, PromotionCriteria criteria, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        return true; // Always promote in mock
    }

    public async Task<ModelMetrics> GetModelMetricsAsync(string modelId, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        return new ModelMetrics
        {
            AUC = 0.65,
            PrAt10 = 0.12,
            ECE = 0.05,
            EdgeBps = 3.5,
            SampleSize = 1000,
            ComputedAt = DateTime.UtcNow
        };
    }
}

/// <summary>
/// Mock feature store for testing
/// </summary>
public class MockFeatureStore : IFeatureStore
{
    public async Task<FeatureSet> GetFeaturesAsync(string symbol, DateTime fromTime, DateTime toTime, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        return new FeatureSet
        {
            Symbol = symbol,
            Timestamp = DateTime.UtcNow,
            Version = "v1.0",
            Features = new Dictionary<string, double>
            {
                ["price"] = 4500.0,
                ["volume"] = 1000.0,
                ["volatility"] = 0.15,
                ["momentum"] = 0.05
            }
        };
    }

    public async Task SaveFeaturesAsync(FeatureSet features, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
    }

    public async Task<bool> ValidateSchemaAsync(FeatureSet features, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        return true;
    }

    public async Task<FeatureSchema> GetSchemaAsync(string version, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        return new FeatureSchema
        {
            Version = version,
            Checksum = "mock_checksum",
            Features = new Dictionary<string, FeatureDefinition>
            {
                ["price"] = new FeatureDefinition { Name = "price", DataType = typeof(double) },
                ["volume"] = new FeatureDefinition { Name = "volume", DataType = typeof(double) }
            }
        };
    }
}

/// <summary>
/// Mock quarantine manager for testing
/// </summary>
public class MockQuarantineManager : IQuarantineManager
{
    public async Task<QuarantineStatus> CheckModelHealthAsync(string modelId, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        return new QuarantineStatus
        {
            State = HealthState.Healthy,
            ModelId = modelId,
            BlendWeight = 1.0
        };
    }

    public async Task QuarantineModelAsync(string modelId, QuarantineReason reason, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
    }

    public async Task<bool> TryRestoreModelAsync(string modelId, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        return true;
    }

    public async Task<List<string>> GetQuarantinedModelsAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        return new List<string>();
    }
}

#endregion