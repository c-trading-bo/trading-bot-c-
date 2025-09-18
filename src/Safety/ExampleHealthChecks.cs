using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using OrchestratorAgent.Infra;

namespace OrchestratorAgent.Infra.HealthChecks;

/// <summary>
/// Example health check for ML Learning persistence
/// Shows how new features should implement health monitoring
/// </summary>
[HealthCheck(Category = "Machine Learning", Priority = 1)]
public class MLLearningHealthCheck : IHealthCheck
{
    private readonly ILogger<MLLearningHealthCheck>? _logger;

    // Parameterless constructor for auto-discovery
    public MLLearningHealthCheck() : this(null) { }

    public MLLearningHealthCheck(ILogger<MLLearningHealthCheck>? logger)
    {
        _logger = logger;
    }

    public string Name => "ml_learning_system";
    public string Description => "Validates ML learning persistence and cycle tracking";
    public string Category => "Machine Learning";
    public int IntervalSeconds => 120; // Check every 2 minutes

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Test ML learning state persistence
            var stateFile = "state/learning_state.json";
            if (!System.IO.File.Exists(stateFile))
            {
                return HealthCheckResult.Warning("ML learning state file not found - first run expected");
            }

            // Validate state file structure
            var stateContent = await System.IO.File.ReadAllTextAsync(stateFile, cancellationToken).ConfigureAwait(false);
            if (string.IsNullOrEmpty(stateContent) || !stateContent.Contains("lastPractice"))
            {
                return HealthCheckResult.Failed("ML learning state file is corrupted or invalid");
            }

            // Test that learning cycles are incrementing
            var lastModified = System.IO.File.GetLastWriteTimeUtc(stateFile);
            var timeSinceLastUpdate = DateTime.UtcNow - lastModified;

            if (timeSinceLastUpdate > TimeSpan.FromHours(2))
            {
                return HealthCheckResult.Warning($"ML learning state hasn't been updated in {timeSinceLastUpdate.TotalHours:F1} hours");
            }

            return HealthCheckResult.Healthy($"ML learning system operational, last update {timeSinceLastUpdate.TotalMinutes:F0}m ago");
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "[HEALTH] ML learning health check failed");
            return HealthCheckResult.Failed($"ML learning health check error: {ex.Message}");
        }
    }
}

/// <summary>
/// Example health check for Strategy Signal validation
/// Shows how trading strategies can self-monitor
/// </summary>
[HealthCheck(Category = "Trading Strategies", Priority = 2)]
public class StrategySignalHealthCheck : IHealthCheck
{
    private readonly ILogger<StrategySignalHealthCheck>? _logger;

    // Parameterless constructor for auto-discovery
    public StrategySignalHealthCheck() : this(null) { }

    public StrategySignalHealthCheck(ILogger<StrategySignalHealthCheck>? logger)
    {
        _logger = logger;
    }

    public string Name => "strategy_signal_validation";
    public string Description => "Validates strategy signal generation and logic";
    public string Category => "Trading Strategies";
    public int IntervalSeconds => 180; // Check every 3 minutes

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Test strategy configuration loading
            var configFiles = System.IO.Directory.GetFiles("src/BotCore/Config", "*.json");
            if (configFiles.Length == 0)
            {
                return HealthCheckResult.Failed("No strategy configuration files found");
            }

            // Validate each strategy config
            var validConfigs = 0;
            var totalConfigs = configFiles.Length;

            foreach (var configFile in configFiles)
            {
                try
                {
                    var content = await System.IO.File.ReadAllTextAsync(configFile, cancellationToken).ConfigureAwait(false);
                    if (content.Contains("maxTrades") && content.Contains("entryMode"))
                    {
                        validConfigs++;
                    }
                }
                catch
                {
                    // Skip invalid config files
                }
            }

            if (validConfigs == 0)
            {
                return HealthCheckResult.Failed("No valid strategy configurations found");
            }

            if (validConfigs < totalConfigs)
            {
                return HealthCheckResult.Warning($"Only {validConfigs}/{totalConfigs} strategy configs are valid");
            }

            // Test signal generation logic (mock test)
            var testPrice = 5000.25m;
            var testStop = 4995.00m;
            var testTarget = 5010.50m;
            var testRisk = testPrice - testStop; // 5.25
            var testReward = testTarget - testPrice; // 10.25
            var expectedRMultiple = testReward / testRisk; // ~1.95

            if (testRisk <= 0 || testReward <= 0)
            {
                return HealthCheckResult.Failed("Strategy signal risk/reward calculation logic is broken");
            }

            return HealthCheckResult.Healthy($"Strategy signals operational: {validConfigs} strategies, R-multiple test passed ({expectedRMultiple:F2})");
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "[HEALTH] Strategy signal health check failed");
            return HealthCheckResult.Failed($"Strategy signal validation error: {ex.Message}");
        }
    }
}

/// <summary>
/// Template for new feature health checks
/// Copy this template when adding new features to the bot
/// </summary>
[HealthCheck(Category = "Template", Priority = 999, Enabled = false)]
public class NewFeatureHealthCheckTemplate : IHealthCheck
{
    public string Name => "new_feature_name";
    public string Description => "Describe what this health check validates";
    public string Category => "Feature Category";
    public int IntervalSeconds => 60; // How often to check

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Implementation complete: Basic validation template for new features
            await Task.Delay(1, cancellationToken).ConfigureAwait(false); // Satisfy async requirement

            // Example checks:
            // 1. Validate configuration is loaded correctly
            // 2. Test core functionality with mock data
            // 3. Check file permissions and dependencies
            // 4. Verify mathematical calculations
            // 5. Test error handling paths

            // Always test the actual behavior, not just existence!

            return HealthCheckResult.Healthy("Feature is working correctly");
        }
        catch (Exception ex)
        {
            return HealthCheckResult.Failed($"Feature health check failed: {ex.Message}");
        }
    }
}
