using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Automatically bootstraps the model registry with initial champions on first startup.
/// Only runs once - skips if champions already registered.
/// 
/// ARCHITECTURE NOTE - ML/RL Component Training Status:
/// 
/// ✅ TRAINABLE COMPONENTS (2/9):
/// - CVaR-PPO: Has CVaRPPOTrainer.cs, trains in HistoricalTrainingOrchestrator
/// - Neural-UCB: Has NeuralUcbBanditTrainer.cs, trains in HistoricalTrainingOrchestrator
/// 
/// ⚠️ INTEGRATED COMPONENTS (7/9) - No dedicated trainers:
/// - Regime-Detector: Placeholder bootstrap model
/// - Model-Ensemble: Placeholder bootstrap model
/// - Online-Learning-System: Uses IncrementalBayesianOptimizer (optimization, not training)
/// - Slippage-Latency-Model: Placeholder bootstrap model
/// - S15-RL-Policy: Validated but not trained (uses external S15 system)
/// - Pattern-Recognition: Placeholder bootstrap model
/// - PM-Optimizer: Uses PositionManagementOptimizer (optimization, not training)
/// 
/// All 9 components get bootstrap champion pointers created to prevent re-bootstrap on each startup.
/// Only CVaR-PPO and Neural-UCB produce new trained ONNX models during Sunday Lab Mode sessions.
/// </summary>
internal sealed class ModelRegistryBootstrapService : IHostedService
{
    private readonly IModelRegistry _modelRegistry;
    private readonly ILogger<ModelRegistryBootstrapService> _logger;

    public ModelRegistryBootstrapService(
        IModelRegistry modelRegistry,
        ILogger<ModelRegistryBootstrapService> logger)
    {
        _modelRegistry = modelRegistry;
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Check if bootstrap needed
            var cvarChampion = await _modelRegistry.GetChampionAsync("CVaR-PPO", cancellationToken).ConfigureAwait(false);
            if (cvarChampion != null)
            {
                _logger.LogInformation("🌱 [MODEL-BOOTSTRAP] Registry already initialized - skipping bootstrap");
                return;
            }

            _logger.LogWarning("🌱 [MODEL-BOOTSTRAP] Empty registry detected - registering initial champions...");
            
            // Register all 9 learning components
            await RegisterComponent("CVaR-PPO", "models/rl/cvar_ppo_agent.onnx", cancellationToken);
            await RegisterComponent("Neural-UCB", "models/rl/neural_ucb.onnx", cancellationToken);
            await RegisterComponent("Regime-Detector", "models/regime/detector.onnx", cancellationToken);
            await RegisterComponent("Model-Ensemble", "models/ensemble/meta_learner.onnx", cancellationToken);
            await RegisterComponent("Online-Learning-System", "models/online/incremental.onnx", cancellationToken);
            await RegisterComponent("Slippage-Latency-Model", "models/execution/slippage.onnx", cancellationToken);
            await RegisterComponent("S15-RL-Policy", "artifacts/current/rl_policy.onnx", cancellationToken);
            await RegisterComponent("Pattern-Recognition", "models/patterns/recognition.onnx", cancellationToken);
            await RegisterComponent("PM-Optimizer", "models/pm/optimizer.onnx", cancellationToken);

            _logger.LogWarning("✅ [MODEL-BOOTSTRAP] Registered 9 ML/RL components as initial champions");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [MODEL-BOOTSTRAP] Bootstrap failed - registry may be empty");
        }
    }

    private async Task RegisterComponent(string algorithm, string modelPath, CancellationToken cancellationToken)
    {
        try
        {
            // Check if model already exists
            var existingModel = await _modelRegistry.GetModelAsync("v1.0.0-bootstrap", cancellationToken).ConfigureAwait(false);
            if (existingModel != null && existingModel.Algorithm == algorithm)
            {
                _logger.LogDebug("  ⏭️ Skipping {Algorithm} - already registered", algorithm);
                return;
            }

            var model = new ModelVersion
            {
                Algorithm = algorithm,
                VersionId = "v1.0.0-bootstrap",
                CreatedAt = DateTime.UtcNow,
                ArtifactPath = File.Exists(modelPath) ? modelPath : string.Empty,
                Sharpe = 0.0m,
                WinRate = 0.0m,
                MaxDrawdown = 0.0m,
                IsPromoted = true,
                PromotedAt = DateTime.UtcNow
            };

            await _modelRegistry.RegisterModelAsync(model, cancellationToken).ConfigureAwait(false);
            await _modelRegistry.PromoteToChampionAsync(algorithm, model.VersionId, new PromotionRecord
            {
                Id = Guid.NewGuid().ToString(),
                PromotedAt = DateTime.UtcNow,
                Reason = "Bootstrap: Initial champion registration"
            }, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("  ✅ Registered {Algorithm} champion", algorithm);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "  ⚠️ Failed to register {Algorithm} champion (may already exist)", algorithm);
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
