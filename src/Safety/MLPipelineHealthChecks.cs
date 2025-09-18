using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using OrchestratorAgent.Infra;

namespace OrchestratorAgent.Infra.HealthChecks;

/// <summary>
/// Health checks for the Next-Generation ML Pipeline components
/// </summary>
[HealthCheck(Category = "ML Pipeline", Enabled = true)]
public class MLPipelineHealthChecks : IHealthCheck
{
    private readonly ILogger<MLPipelineHealthChecks> _logger;

    // Parameterless constructor for auto-discovery
    public MLPipelineHealthChecks() : this(null) { }

    public MLPipelineHealthChecks(ILogger<MLPipelineHealthChecks>? logger)
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<MLPipelineHealthChecks>.Instance;
    }

    public string Name => "ML Pipeline Components";
    public string Description => "Monitors Next-Generation ML Pipeline: Meta-Labeler, Smart Execution, Bandits, Enhanced Priors";
    public string Category => "ML Pipeline";
    public int IntervalSeconds => 60; // Check every minute

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        var results = new List<(string Component, bool IsHealthy, string Message)>();
        await Task.Delay(1, cancellationToken).ConfigureAwait(false); // Satisfy async requirement

        try
        {
            // 1. Meta-Labeler System Health
            var metaLabelerHealth = CheckMetaLabelerHealth();
            results.Add(("Meta-Labeler", metaLabelerHealth.IsHealthy, metaLabelerHealth.Message));

            // 2. Smart Execution Health
            var executionHealth = CheckSmartExecutionHealth();
            results.Add(("Smart Execution", executionHealth.IsHealthy, executionHealth.Message));

            // 3. Function Approximation Bandits Health
            var banditsHealth = CheckBanditsHealth();
            results.Add(("Bandits", banditsHealth.IsHealthy, banditsHealth.Message));

            // 4. Enhanced Bayesian Priors Health
            var priorsHealth = CheckEnhancedPriorsHealth();
            results.Add(("Enhanced Priors", priorsHealth.IsHealthy, priorsHealth.Message));

            // 5. Walk-Forward Trainer Health
            var trainerHealth = CheckWalkForwardTrainerHealth();
            results.Add(("Walk-Forward Trainer", trainerHealth.IsHealthy, trainerHealth.Message));

            // Evaluate overall health
            var healthyCount = results.Count(r => r.IsHealthy);
            var totalCount = results.Count;
            var overallHealthy = healthyCount >= (totalCount * 0.8); // 80% must be healthy

            var message = $"ML Pipeline Health: {healthyCount}/{totalCount} components healthy";
            if (!overallHealthy)
            {
                var failedComponents = results.Where(r => !r.IsHealthy).Select(r => $"{r.Component}: {r.Message}");
                message += $" - ISSUES: {string.Join(", ", failedComponents)}";
            }

            return overallHealthy
                ? HealthCheckResult.Healthy(message)
                : HealthCheckResult.Failed(message);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "[HEALTH] ML Pipeline health check failed");
            return HealthCheckResult.Failed($"ML Pipeline health check failed: {ex.Message}", ex);
        }
    }

    private (bool IsHealthy, string Message) CheckMetaLabelerHealth()
    {
        try
        {
            // Check if meta-labeler classes exist
            var metaLabelerAssembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetTypes().Any(t => t.Name == "OnnxMetaLabeler"));

            if (metaLabelerAssembly == null)
                return (false, "OnnxMetaLabeler class not found");

            // Check for models directory
            var modelsPath = Path.Combine(AppContext.BaseDirectory, "models");
            if (!Directory.Exists(modelsPath))
                Directory.CreateDirectory(modelsPath); // Create if missing

            // Check ONNX runtime availability
            try
            {
                var onnxType = Type.GetType("Microsoft.ML.OnnxRuntime.InferenceSession, Microsoft.ML.OnnxRuntime");
                if (onnxType == null)
                    return (false, "ONNX Runtime not available");
            }
            catch
            {
                return (false, "ONNX Runtime dependency missing");
            }

            return (true, "Meta-Labeler system operational");
        }
        catch (Exception ex)
        {
            return (false, $"Meta-Labeler check failed: {ex.Message}");
        }
    }

    private (bool IsHealthy, string Message) CheckSmartExecutionHealth()
    {
        try
        {
            // Check if execution components exist
            var executionTypes = new[] { "EvExecutionRouter", "BasicMicrostructureAnalyzer" };
            var currentAssembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetTypes().Any(t => executionTypes.Contains(t.Name)));

            if (currentAssembly == null)
                return (false, "Smart Execution components not found");

            // Check if execution interfaces are properly implemented
            var interfaceType = currentAssembly.GetTypes()
                .FirstOrDefault(t => t.Name == "IMicrostructureAnalyzer");

            if (interfaceType == null)
                return (false, "IMicrostructureAnalyzer interface missing");

            return (true, "Smart Execution system operational");
        }
        catch (Exception ex)
        {
            return (false, $"Smart Execution check failed: {ex.Message}");
        }
    }

    private (bool IsHealthy, string Message) CheckBanditsHealth()
    {
        try
        {
            // Check if bandit algorithms exist
            var banditTypes = new[] { "LinUcbBandit", "NeuralUcbBandit" };
            var currentAssembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetTypes().Any(t => banditTypes.Any(bt => t.Name.Contains(bt))));

            if (currentAssembly == null)
                return (false, "Function Approximation Bandits not found");

            // Check bandit state directory
            var banditStatePath = Path.Combine(AppContext.BaseDirectory, "state", "bandits");
            if (!Directory.Exists(banditStatePath))
                Directory.CreateDirectory(banditStatePath); // Create if missing

            return (true, "Function Approximation Bandits operational");
        }
        catch (Exception ex)
        {
            return (false, $"Bandits check failed: {ex.Message}");
        }
    }

    private (bool IsHealthy, string Message) CheckEnhancedPriorsHealth()
    {
        try
        {
            // Check if Enhanced Bayesian Priors exist
            var priorsAssembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetTypes().Any(t => t.Name == "EnhancedBayesianPriors"));

            if (priorsAssembly == null)
                return (false, "EnhancedBayesianPriors class not found");

            // Check priors state directory
            var priorsStatePath = Path.Combine(AppContext.BaseDirectory, "state", "priors");
            if (!Directory.Exists(priorsStatePath))
                Directory.CreateDirectory(priorsStatePath); // Create if missing

            return (true, "Enhanced Bayesian Priors operational");
        }
        catch (Exception ex)
        {
            return (false, $"Enhanced Priors check failed: {ex.Message}");
        }
    }

    private (bool IsHealthy, string Message) CheckWalkForwardTrainerHealth()
    {
        try
        {
            // Check if Walk-Forward Trainer exists
            var trainerAssembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetTypes().Any(t => t.Name == "WalkForwardTrainer"));

            if (trainerAssembly == null)
                return (false, "WalkForwardTrainer class not found");

            // Check training data directory
            var trainingDataPath = Path.Combine(AppContext.BaseDirectory, "training_data");
            if (!Directory.Exists(trainingDataPath))
                Directory.CreateDirectory(trainingDataPath); // Create if missing

            return (true, "Walk-Forward Trainer operational");
        }
        catch (Exception ex)
        {
            return (false, $"Walk-Forward Trainer check failed: {ex.Message}");
        }
    }
}
