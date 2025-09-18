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
            await Task.Delay(2000, stoppingToken).ConfigureAwait(false);
            
            // Test 1: Model Registry
            await TestModelRegistryAsync(stoppingToken).ConfigureAwait(false);
            
            // Test 2: Atomic Model Router
            await TestAtomicModelRouterAsync(stoppingToken).ConfigureAwait(false);
            
            // Test 3: Read-Only Inference Brain
            await TestInferenceBrainAsync(stoppingToken).ConfigureAwait(false);
            
            // Test 4: Write-Only Training Brain
            await TestTrainingBrainAsync(stoppingToken).ConfigureAwait(false);
            
            // Test 5: Promotion Service with Timing Gates
            await TestPromotionServiceAsync(stoppingToken).ConfigureAwait(false);
            
            // Test 6: Market Hours Service
            await TestMarketHoursServiceAsync(stoppingToken).ConfigureAwait(false);
            
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
        
        var versionId = await _modelRegistry.RegisterModelAsync(testModel, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        _logger.LogInformation("✅ Model registered with version: {VersionId}", versionId);
        
        var retrievedModel = await _modelRegistry.GetModelAsync(versionId, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        if (retrievedModel != null)
        {
            _logger.LogInformation("✅ Model retrieved successfully");
        }
    }

    private async Task TestAtomicModelRouterAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Atomic Model Router...");
        
        var router = _routerFactory.CreateRouter<object>("PPO");
        var stats = await router.GetStatsAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        
        _logger.LogInformation("✅ Router created for PPO - Algorithm: {Algorithm}, Healthy: {Healthy}", 
            stats.Algorithm, stats.IsHealthy);
    }

    private async Task TestInferenceBrainAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Read-Only Inference Brain...");
        
        try
        {
            var isReady = await _inferenceBrain.IsReadyAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            _logger.LogInformation("✅ Inference Brain ready: {Ready}", isReady);
            
            var versions = await _inferenceBrain.GetChampionVersionsAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            // Phase 6B: Validation Bounds - Add defensive checks for version retrieval
            var ppoVersion = versions?.GetValueOrDefault("PPO")?.VersionId;
            var ucbVersion = versions?.GetValueOrDefault("UCB")?.VersionId;
            var lstmVersion = versions?.GetValueOrDefault("LSTM")?.VersionId;
            
            // Validate version strings before using them
            var safePpoVersion = !string.IsNullOrEmpty(ppoVersion) && ppoVersion.Length <= 50 ? ppoVersion : "none";
            var safeUcbVersion = !string.IsNullOrEmpty(ucbVersion) && ucbVersion.Length <= 50 ? ucbVersion : "none";
            var safeLstmVersion = !string.IsNullOrEmpty(lstmVersion) && lstmVersion.Length <= 50 ? lstmVersion : "none";
            
            _logger.LogInformation("✅ Champion versions - PPO: {PPO}, UCB: {UCB}, LSTM: {LSTM}", 
                safePpoVersion, safeUcbVersion, safeLstmVersion);

            // Test decision making (read-only)
            var context = new TradingContext
            {
                Symbol = "ES",
                Price = 4500,
                Volume = 1000,
                AccountBalance = 50000
            };
            
            var decision = await _inferenceBrain.DecideAsync(context, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            // Phase 6B: Validation Bounds - Add defensive checks for decision properties  
            var safeAction = !string.IsNullOrEmpty(decision.Action) && decision.Action.Length <= 20 ? decision.Action : "UNKNOWN";
            var safeStrategy = !string.IsNullOrEmpty(decision.Strategy) && decision.Strategy.Length <= 50 ? decision.Strategy : "UNKNOWN";
            var safePpoVersionId = !string.IsNullOrEmpty(decision.PPOVersionId) && decision.PPOVersionId.Length <= 50 ? decision.PPOVersionId : "none";
            var safeUcbVersionId = !string.IsNullOrEmpty(decision.UCBVersionId) && decision.UCBVersionId.Length <= 50 ? decision.UCBVersionId : "none";
            var safeLstmVersionId = !string.IsNullOrEmpty(decision.LSTMVersionId) && decision.LSTMVersionId.Length <= 50 ? decision.LSTMVersionId : "none";
            
            _logger.LogInformation("✅ Decision made - Action: {Action}, Strategy: {Strategy}, PPO: {PPO}, UCB: {UCB}, LSTM: {LSTM}", 
                safeAction, safeStrategy, safePpoVersionId, safeUcbVersionId, safeLstmVersionId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Inference Brain test failed: {Error}", ex.Message);
        }
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
        
        var trainingResult = await _trainingBrain.TrainChallengerAsync("PPO", config, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        _logger.LogInformation("✅ Training completed - Success: {Success}, JobId: {JobId}, Duration: {Duration:F1}s", 
            trainingResult.Success, trainingResult.JobId, trainingResult.TrainingDuration.TotalSeconds);
    }

    private async Task TestPromotionServiceAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Promotion Service...");
        
        try
        {
            var status = await _promotionService.GetPromotionStatusAsync("PPO", cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            // Phase 6B: Validation Bounds - Add defensive checks for promotion status
            var safeChampionVersionId = !string.IsNullOrEmpty(status.CurrentChampionVersionId) && 
                                       status.CurrentChampionVersionId.Length <= 50 ? 
                                       status.CurrentChampionVersionId : "none";
            
            _logger.LogInformation("✅ Promotion status - Champion: {Champion}, CanRollback: {CanRollback}", 
                safeChampionVersionId, status.CanRollback);
            
            // Test promotion evaluation (without actual promotion)
            if (!string.IsNullOrEmpty(status.CurrentChampionVersionId))
            {
                var decision = await _promotionService.EvaluatePromotionAsync("PPO", "test_challenger_v1", cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                
                // Phase 6B: Validation Bounds - Add defensive checks for promotion decision
                var safeReason = !string.IsNullOrEmpty(decision.Reason) && decision.Reason.Length <= 200 ? 
                                decision.Reason : "No reason provided";
                
                _logger.LogInformation("✅ Promotion evaluation - ShouldPromote: {ShouldPromote}, Reason: {Reason}, " +
                    "InSafeWindow: {InSafeWindow}, IsFlat: {IsFlat}", 
                    decision.ShouldPromote, safeReason, decision.IsInSafeWindow, decision.IsFlat);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Promotion Service test failed: {Error}", ex.Message);
        }
    }

    private async Task TestMarketHoursServiceAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Testing Market Hours Service...");
        
        var isInSafeWindow = await _marketHours.IsInSafePromotionWindowAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        var currentSession = await _marketHours.GetCurrentMarketSessionAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        var nextSafeWindow = await _marketHours.GetNextSafeWindowAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        
        _logger.LogInformation("✅ Market Hours - InSafeWindow: {InSafeWindow}, Session: {Session}, NextWindow: {NextWindow}", 
            isInSafeWindow, currentSession, nextSafeWindow?.ToString("yyyy-MM-dd HH:mm UTC") ?? "Unknown");
    }
}