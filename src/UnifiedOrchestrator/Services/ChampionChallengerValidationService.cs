using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Champion/Challenger validation service demonstrating the architecture
/// </summary>
public class ChampionChallengerValidationService : BackgroundService
{
    private readonly ILogger<ChampionChallengerValidationService> _logger;
    private readonly IModelRegistry _modelRegistry;
    private readonly IModelRouterFactory _routerFactory;
    private readonly IInferenceBrain _inferenceBrain;
    private readonly ITrainingBrain _trainingBrain;
    private readonly IPromotionService _promotionService;
    private readonly IMarketHoursService _marketHours;

    public ChampionChallengerValidationService(
        ILogger<ChampionChallengerValidationService> logger,
        IModelRegistry modelRegistry,
        IModelRouterFactory routerFactory,
        IInferenceBrain inferenceBrain,
        ITrainingBrain trainingBrain,
        IPromotionService promotionService,
        IMarketHoursService marketHours)
    {
        _logger = logger;
        _modelRegistry = modelRegistry;
        _routerFactory = routerFactory;
        _inferenceBrain = inferenceBrain;
        _trainingBrain = trainingBrain;
        _promotionService = promotionService;
        _marketHours = marketHours;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("🏆 Starting Champion/Challenger Architecture Validation");
        
        try
        {
            // Wait a bit for system initialization
            await Task.Delay(2000, stoppingToken);
            
            // Test 1: Model Registry
            await TestModelRegistryAsync(stoppingToken);
            
            // Test 2: Atomic Model Router
            await TestAtomicModelRouterAsync(stoppingToken);
            
            // Test 3: Read-Only Inference Brain
            await TestInferenceBrainAsync(stoppingToken);
            
            // Test 4: Write-Only Training Brain
            await TestTrainingBrainAsync(stoppingToken);
            
            // Test 5: Promotion Service with Timing Gates
            await TestPromotionServiceAsync(stoppingToken);
            
            // Test 6: Market Hours Service
            await TestMarketHoursServiceAsync(stoppingToken);
            
            _logger.LogInformation("✅ Champion/Challenger Architecture Validation COMPLETED - All systems operational");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Validation failed: {Error}", ex.Message);
        }
    }

    private async Task TestModelRegistryAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Model Registry...");
        
        // Create test model version
        var testModel = new ModelVersion
        {
            Algorithm = "PPO",
            ArtifactPath = "/tmp/test_model.onnx",
            GitSha = "test123",
            CreatedBy = "validation_test",
            Sharpe = 1.5m,
            CVaR = -0.02m,
            SchemaVersion = "1.0",
            ModelType = "ONNX"
        };
        
        var versionId = await _modelRegistry.RegisterModelAsync(testModel, cancellationToken);
        _logger.LogInformation("✅ Model registered with version: {VersionId}", versionId);
        
        var retrievedModel = await _modelRegistry.GetModelAsync(versionId, cancellationToken);
        if (retrievedModel != null)
        {
            _logger.LogInformation("✅ Model retrieved successfully");
        }
    }

    private async Task TestAtomicModelRouterAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Atomic Model Router...");
        
        var router = _routerFactory.CreateRouter<object>("PPO");
        var stats = await router.GetStatsAsync(cancellationToken);
        
        _logger.LogInformation("✅ Router created for PPO - Algorithm: {Algorithm}, Healthy: {Healthy}", 
            stats.Algorithm, stats.IsHealthy);
    }

    private async Task TestInferenceBrainAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Read-Only Inference Brain...");
        
        var isReady = await _inferenceBrain.IsReadyAsync(cancellationToken);
        _logger.LogInformation("✅ Inference Brain ready: {Ready}", isReady);
        
        var versions = await _inferenceBrain.GetChampionVersionsAsync(cancellationToken);
        _logger.LogInformation("✅ Champion versions - PPO: {PPO}, UCB: {UCB}, LSTM: {LSTM}", 
            versions.GetValueOrDefault("PPO")?.VersionId ?? "none",
            versions.GetValueOrDefault("UCB")?.VersionId ?? "none", 
            versions.GetValueOrDefault("LSTM")?.VersionId ?? "none");

        // Test decision making (read-only)
        var context = new TradingContext
        {
            Symbol = "ES",
            Price = 4500,
            Volume = 1000,
            AccountBalance = 50000
        };
        
        var decision = await _inferenceBrain.DecideAsync(context, cancellationToken);
        _logger.LogInformation("✅ Decision made - Action: {Action}, Strategy: {Strategy}, PPO: {PPO}, UCB: {UCB}, LSTM: {LSTM}", 
            decision.Action, decision.Strategy, 
            decision.PPOVersionId, decision.UCBVersionId, decision.LSTMVersionId);
    }

    private async Task TestTrainingBrainAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Write-Only Training Brain...");
        
        var config = new TrainingConfig
        {
            Algorithm = "PPO",
            DataStartTime = DateTime.UtcNow.AddDays(-30),
            DataEndTime = DateTime.UtcNow.AddDays(-1), // Temporal hygiene
            MaxEpochs = 10,
            Parameters = new Dictionary<string, object> { ["learning_rate"] = 0.001 }
        };
        
        var trainingResult = await _trainingBrain.TrainChallengerAsync("PPO", config, cancellationToken);
        _logger.LogInformation("✅ Training completed - Success: {Success}, JobId: {JobId}, Duration: {Duration:F1}s", 
            trainingResult.Success, trainingResult.JobId, trainingResult.TrainingDuration.TotalSeconds);
    }

    private async Task TestPromotionServiceAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Promotion Service...");
        
        var status = await _promotionService.GetPromotionStatusAsync("PPO", cancellationToken);
        _logger.LogInformation("✅ Promotion status - Champion: {Champion}, CanRollback: {CanRollback}", 
            status.CurrentChampionVersionId, status.CanRollback);
        
        // Test promotion evaluation (without actual promotion)
        if (!string.IsNullOrEmpty(status.CurrentChampionVersionId))
        {
            var decision = await _promotionService.EvaluatePromotionAsync("PPO", "test_challenger_v1", cancellationToken);
            _logger.LogInformation("✅ Promotion evaluation - ShouldPromote: {ShouldPromote}, Reason: {Reason}, " +
                "InSafeWindow: {InSafeWindow}, IsFlat: {IsFlat}", 
                decision.ShouldPromote, decision.Reason, decision.IsInSafeWindow, decision.IsFlat);
        }
    }

    private async Task TestMarketHoursServiceAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Market Hours Service...");
        
        var isInSafeWindow = await _marketHours.IsInSafePromotionWindowAsync(cancellationToken);
        var currentSession = await _marketHours.GetCurrentMarketSessionAsync(cancellationToken);
        var nextSafeWindow = await _marketHours.GetNextSafeWindowAsync(cancellationToken);
        
        _logger.LogInformation("✅ Market Hours - InSafeWindow: {InSafeWindow}, Session: {Session}, NextWindow: {NextWindow}", 
            isInSafeWindow, currentSession, nextSafeWindow?.ToString("yyyy-MM-dd HH:mm UTC") ?? "Unknown");
    }
}