using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Extension methods for registering the complete intelligence stack
/// Ensures all services are properly configured and wired
/// </summary>
public static class IntelligenceStackServiceExtensions
{
    /// <summary>
    /// Register the complete intelligence stack with dependency injection
    /// </summary>
    public static IServiceCollection AddIntelligenceStack(
        this IServiceCollection services, 
        IConfiguration configuration)
    {
        // Load configuration
        var intelligenceConfig = configuration.GetSection("IntelligenceStack").Get<IntelligenceStackConfig>() 
            ?? new IntelligenceStackConfig();
        
        services.AddSingleton(intelligenceConfig);
        services.AddSingleton(intelligenceConfig.ML);
        services.AddSingleton(intelligenceConfig.ML.Regime.Hysteresis);
        services.AddSingleton(intelligenceConfig.Online.MetaLearning);
        services.AddSingleton(intelligenceConfig.Orchestrator.LeaderElection);
        services.AddSingleton(intelligenceConfig.ML.Quarantine);
        services.AddSingleton(intelligenceConfig.SLO);
        services.AddSingleton(intelligenceConfig.Observability);
        services.AddSingleton(intelligenceConfig.Promotions);

        // Register core intelligence services
        services.AddSingleton<IRegimeDetector, RegimeDetectorWithHysteresis>();
        services.AddSingleton<IFeatureStore, FeatureStore>();
        services.AddSingleton<IModelRegistry, ModelRegistry>();
        services.AddSingleton<ICalibrationManager, CalibrationManager>();
        services.AddSingleton<IOnlineLearningSystem, OnlineLearningSystem>();
        services.AddSingleton<IQuarantineManager, ModelQuarantineManager>();
        services.AddSingleton<IDecisionLogger, DecisionLogger>();
        services.AddSingleton<IIdempotentOrderService, IdempotentOrderService>();
        services.AddSingleton<ILeaderElectionService, LeaderElectionService>();
        services.AddSingleton<IStartupValidator, StartupValidator>();

        // Register advanced intelligence services
        services.AddSingleton<EnsembleMetaLearner>();
        services.AddSingleton<ModelQuarantineManager>();
        services.AddSingleton<HistoricalTrainerWithCV>();
        services.AddSingleton<MAMLLiveIntegration>();
        services.AddSingleton<NightlyParameterTuner>();
        services.AddSingleton<RLAdvisorSystem>();
        services.AddSingleton<ObservabilityDashboard>();
        services.AddSingleton<LineageTrackingSystem>();

        // Register main orchestrator
        services.AddSingleton<IntelligenceOrchestrator>();
        services.AddSingleton<IIntelligenceOrchestrator>(provider => 
            provider.GetRequiredService<IntelligenceOrchestrator>());

        // Register monitoring services
        services.AddSingleton<SLOMonitor>();

        return services;
    }

    /// <summary>
    /// Register mock implementations for testing
    /// </summary>
    public static IServiceCollection AddMockIntelligenceStack(this IServiceCollection services)
    {
        var config = new IntelligenceStackConfig();
        services.AddSingleton(config);
        
        // Register mock implementations
        services.AddSingleton<IRegimeDetector, MockRegimeDetector>();
        services.AddSingleton<IFeatureStore, MockFeatureStore>();
        services.AddSingleton<IModelRegistry, MockModelRegistry>();
        services.AddSingleton<ICalibrationManager, MockCalibrationManager>();
        services.AddSingleton<IOnlineLearningSystem, MockOnlineLearningSystem>();
        services.AddSingleton<IQuarantineManager, MockQuarantineManager>();
        services.AddSingleton<IDecisionLogger, MockDecisionLogger>();
        services.AddSingleton<IIdempotentOrderService, MockIdempotentOrderService>();
        services.AddSingleton<ILeaderElectionService, MockLeaderElectionService>();
        services.AddSingleton<IStartupValidator, MockStartupValidator>();

        return services;
    }
}

#region Mock Implementations for Testing

public class MockRegimeDetector : IRegimeDetector
{
    public Task<RegimeState> DetectCurrentRegimeAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new RegimeState
        {
            Type = RegimeType.Range,
            Confidence = 0.75,
            DetectedAt = DateTime.UtcNow,
            Indicators = new Dictionary<string, double> { ["test"] = 1.0 }
        });
    }

    public Task<RegimeTransition> CheckTransitionAsync(RegimeState currentState, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new RegimeTransition
        {
            ShouldTransition = false,
            FromRegime = currentState.Type,
            ToRegime = currentState.Type,
            TransitionConfidence = 0.8
        });
    }

    public bool IsInDwellPeriod(RegimeState state) => false;
}

public class MockFeatureStore : IFeatureStore
{
    public Task<FeatureSet> GetFeaturesAsync(string symbol, DateTime fromTime, DateTime toTime, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new FeatureSet
        {
            Symbol = symbol,
            Version = "mock_v1",
            Features = new Dictionary<string, double> { ["test_feature"] = 1.0 }
        });
    }

    public Task SaveFeaturesAsync(FeatureSet features, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Task<bool> ValidateSchemaAsync(FeatureSet features, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true);
    }

    public Task<FeatureSchema> GetSchemaAsync(string version, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new FeatureSchema
        {
            Version = version,
            Features = new Dictionary<string, FeatureDefinition>
            {
                ["test_feature"] = new() { Name = "test_feature", DataType = typeof(double) }
            }
        });
    }
}

public class MockModelRegistry : IModelRegistry
{
    public Task<ModelArtifact> GetModelAsync(string familyName, string version = "latest", CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new ModelArtifact
        {
            Id = $"{familyName}_mock",
            Version = version,
            Metrics = new ModelMetrics { AUC = 0.65, PrAt10 = 0.15 }
        });
    }

    public Task<ModelArtifact> RegisterModelAsync(ModelRegistration registration, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new ModelArtifact
        {
            Id = $"{registration.FamilyName}_mock",
            Metrics = registration.Metrics
        });
    }

    public Task<bool> PromoteModelAsync(string modelId, PromotionCriteria criteria, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true);
    }

    public Task<ModelMetrics> GetModelMetricsAsync(string modelId, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new ModelMetrics { AUC = 0.65, PrAt10 = 0.15 });
    }
}

public class MockCalibrationManager : ICalibrationManager
{
    public Task<CalibrationMap> LoadCalibrationMapAsync(string modelId, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CalibrationMap
        {
            ModelId = modelId,
            Method = CalibrationMethod.Platt,
            Parameters = new Dictionary<string, double> { ["slope"] = 1.0, ["intercept"] = 0.0 }
        });
    }

    public Task<CalibrationMap> FitCalibrationAsync(string modelId, IEnumerable<CalibrationPoint> points, CancellationToken cancellationToken = default)
    {
        return LoadCalibrationMapAsync(modelId, cancellationToken);
    }

    public Task<double> CalibrateConfidenceAsync(string modelId, double rawConfidence, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(rawConfidence); // No calibration in mock
    }

    public Task PerformNightlyCalibrationAsync(CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }
}

public class MockOnlineLearningSystem : IOnlineLearningSystem
{
    public Task UpdateWeightsAsync(string regimeType, Dictionary<string, double> weights, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Task<Dictionary<string, double>> GetCurrentWeightsAsync(string regimeType, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new Dictionary<string, double> { ["default"] = 1.0 });
    }

    public Task AdaptToPerformanceAsync(string modelId, ModelPerformance performance, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Task DetectDriftAsync(string modelId, FeatureSet features, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Task UpdateModelAsync(TradeRecord tradeRecord, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }
}

public class MockQuarantineManager : IQuarantineManager
{
    public Task<QuarantineStatus> CheckModelHealthAsync(string modelId, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new QuarantineStatus
        {
            State = HealthState.Healthy,
            ModelId = modelId,
            BlendWeight = 1.0
        });
    }

    public Task QuarantineModelAsync(string modelId, QuarantineReason reason, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Task<bool> TryRestoreModelAsync(string modelId, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true);
    }

    public Task<List<string>> GetQuarantinedModelsAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new List<string>());
    }
}

public class MockDecisionLogger : IDecisionLogger
{
    public Task LogDecisionAsync(IntelligenceDecision decision, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Task<List<IntelligenceDecision>> GetDecisionHistoryAsync(DateTime fromTime, DateTime toTime, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new List<IntelligenceDecision>());
    }
}

public class MockIdempotentOrderService : IIdempotentOrderService
{
    private readonly HashSet<string> _seenOrders = new();

    public Task<string> GenerateOrderKeyAsync(OrderRequest request, CancellationToken cancellationToken = default)
    {
        return Task.FromResult($"mock_{request.Symbol}_{request.Side}_{DateTime.UtcNow.Ticks}");
    }

    public Task<bool> IsDuplicateOrderAsync(string orderKey, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_seenOrders.Contains(orderKey));
    }

    public Task RegisterOrderAsync(string orderKey, string orderId, CancellationToken cancellationToken = default)
    {
        _seenOrders.Add(orderKey);
        return Task.CompletedTask;
    }

    public Task<OrderDeduplicationResult> CheckDeduplicationAsync(OrderRequest request, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new OrderDeduplicationResult
        {
            IsDuplicate = false,
            OrderKey = $"mock_{request.Symbol}_{DateTime.UtcNow.Ticks}"
        });
    }
}

public class MockLeaderElectionService : ILeaderElectionService
{
    public event EventHandler<LeadershipChangedEventArgs>? LeadershipChanged;

    public Task<bool> TryAcquireLeadershipAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true);
    }

    public Task ReleaseLeadershipAsync(CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public Task<bool> IsLeaderAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true);
    }

    public Task<bool> RenewLeadershipAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true);
    }
}

public class MockStartupValidator : IStartupValidator
{
    public Task<StartupValidationResult> ValidateSystemAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new StartupValidationResult
        {
            AllTestsPassed = true,
            TestResults = new Dictionary<string, TestResult>
            {
                ["Mock_Test"] = new TestResult { Passed = true, TestName = "Mock_Test", Duration = TimeSpan.FromMilliseconds(100) }
            }
        });
    }

    public Task<bool> ValidateDIGraphAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);
    public Task<bool> ValidateFeatureStoreAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);
    public Task<bool> ValidateModelRegistryAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);
    public Task<bool> ValidateCalibrationAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);
    public Task<bool> ValidateIdempotencyAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);
    public Task<bool> ValidateKillSwitchAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);
    public Task<bool> ValidateLeaderElectionAsync(CancellationToken cancellationToken = default) => Task.FromResult(true);
}

#endregion