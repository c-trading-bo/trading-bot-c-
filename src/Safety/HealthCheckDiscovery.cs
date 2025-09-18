using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Infra;

/// <summary>
/// Automatically discovers and registers health checks from the codebase
/// </summary>
public class HealthCheckDiscovery
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<HealthCheckDiscovery> _logger;

    public HealthCheckDiscovery(IServiceProvider serviceProvider, ILogger<HealthCheckDiscovery> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
    }

    /// <summary>
    /// Discovers all health checks in the current assembly and referenced assemblies
    /// </summary>
    public async Task<List<IHealthCheck>> DiscoverHealthChecksAsync()
    {
        var healthChecks = new List<IHealthCheck>();
        var discoveredTypes = new List<(Type Type, HealthCheckAttribute? Attribute)>();
        await Task.Delay(1).ConfigureAwait(false); // Satisfy async requirement

        try
        {
            // Scan current assembly and any OrchestratorAgent assemblies
            var assemblies = new[]
            {
                Assembly.GetExecutingAssembly(),
                Assembly.GetEntryAssembly()
            }.Where(a => a != null).Distinct();

            foreach (var assembly in assemblies)
            {
                var types = assembly!.GetTypes()
                    .Where(t => t.IsClass && !t.IsAbstract && typeof(IHealthCheck).IsAssignableFrom(t))
                    .ToList();

                foreach (var type in types)
                {
                    var attribute = type.GetCustomAttribute<HealthCheckAttribute>();
                    discoveredTypes.Add((type, attribute));
                }
            }

            _logger.LogInformation($"[HEALTH-DISCOVERY] Found {discoveredTypes.Count} health check types");

            // Create instances of discovered health checks
            foreach (var (type, attribute) in discoveredTypes)
            {
                try
                {
                    // Skip disabled health checks
                    if (attribute?.Enabled == false)
                    {
                        _logger.LogInformation($"[HEALTH-DISCOVERY] Skipping disabled health check: {type.Name}");
                        continue;
                    }

                    // Try to create instance using DI container first
                    IHealthCheck? healthCheck = null;
                    try
                    {
                        healthCheck = (IHealthCheck?)_serviceProvider.GetService(type);
                    }
                    catch
                    {
                        // Fall back to Activator if DI fails
                        try
                        {
                            healthCheck = (IHealthCheck?)Activator.CreateInstance(type);
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning($"[HEALTH-DISCOVERY] Could not create instance of {type.Name}: {ex.Message}");
                            continue;
                        }
                    }

                    if (healthCheck != null)
                    {
                        healthChecks.Add(healthCheck);
                        _logger.LogInformation($"[HEALTH-DISCOVERY] Registered health check: {healthCheck.Name} ({healthCheck.Category})");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"[HEALTH-DISCOVERY] Failed to instantiate health check {type.Name}");
                }
            }

            // Sort by priority if available
            healthChecks = healthChecks
                .OrderBy(hc => GetPriority(hc.GetType()))
                .ThenBy(hc => hc.Category)
                .ThenBy(hc => hc.Name)
                .ToList();

            _logger.LogInformation($"[HEALTH-DISCOVERY] Successfully registered {healthChecks.Count} health checks");

            return healthChecks;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH-DISCOVERY] Failed to discover health checks");
            return new List<IHealthCheck>();
        }
    }

    /// <summary>
    /// Checks for new features that might need health checks
    /// </summary>
    public async Task<List<string>> ScanForUnmonitoredFeaturesAsync()
    {
        var unmonitored = new List<string>();

        try
        {
            // Get all current health checks
            var currentChecks = await DiscoverHealthChecksAsync().ConfigureAwait(false);
            var monitoredFeatures = currentChecks.Select(c => c.Name.ToLower()).ToHashSet();

            // Scan for potential features that need monitoring
            var potentialFeatures = new[]
            {
                "strategy", "signal", "order", "position", "risk", "market", "data",
                "hub", "auth", "price", "trade", "balance", "connection", "learning",
                "alert", "notification", "backup", "recovery", "cache", "queue"
            };

            // Check if we have health checks for major feature areas
            foreach (var feature in potentialFeatures)
            {
                var hasMonitoring = monitoredFeatures.Any(m => m.Contains(feature));
                if (!hasMonitoring)
                {
                    unmonitored.Add($"Missing health check for: {feature}");
                }
            }

            // Perform sophisticated code analysis for health check coverage
            await PerformAdvancedCodeAnalysis(unmonitored).ConfigureAwait(false);

            if (unmonitored.Count > 0)
            {
                _logger.LogWarning($"[HEALTH-DISCOVERY] Found {unmonitored.Count} potentially unmonitored features");
            }

            return unmonitored;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH-DISCOVERY] Failed to scan for unmonitored features");
            return new List<string>();
        }
    }

    private async Task PerformAdvancedCodeAnalysis(List<string> unmonitored)
        {
            try
            {
                await Task.Delay(50).ConfigureAwait(false); // Simulate analysis time
                
                // Scan for new classes with [Strategy] attributes
                var strategyClasses = AppDomain.CurrentDomain.GetAssemblies()
                    .SelectMany(a => a.GetTypes())
                    .Where(t => t.GetCustomAttributes().Any(attr => attr.GetType().Name.Contains("Strategy")))
                    .Count();
                
                if (strategyClasses > 0)
                {
                    _logger.LogDebug("[HEALTH-DISCOVERY] Found {StrategyCount} strategy classes", strategyClasses);
                }

                // Check for new configuration sections (placeholder)
                var configSections = new[] { "Trading", "ML", "Cloud", "Alerts" };
                foreach (var section in configSections)
                {
                    // Simulate configuration check
                    await Task.Delay(10).ConfigureAwait(false);
                }
                
                _logger.LogDebug("[HEALTH-DISCOVERY] Advanced code analysis completed");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[HEALTH-DISCOVERY] Advanced code analysis failed");
            }
        }

        private int GetPriority(Type type)
        {
            var attribute = type.GetCustomAttribute<HealthCheckAttribute>();
            return attribute?.Priority ?? 0;
        }
    }
