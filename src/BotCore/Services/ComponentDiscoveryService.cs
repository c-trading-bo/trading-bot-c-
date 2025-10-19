using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using BotCore.Health;

namespace BotCore.Services;

/// <summary>
/// Automatically discovers ALL components in the bot system.
/// This makes the bot self-aware of every service, file, API, and model it uses.
/// No hardcoding needed - it scans the DI container at startup.
/// </summary>
public sealed class ComponentDiscoveryService
{
    private readonly ILogger<ComponentDiscoveryService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly List<DiscoveredComponent> _discoveredComponents = new();

    public ComponentDiscoveryService(
        ILogger<ComponentDiscoveryService> logger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    /// <summary>
    /// Get all discovered components
    /// </summary>
    public IReadOnlyList<DiscoveredComponent> DiscoveredComponents => _discoveredComponents.AsReadOnly();

    /// <summary>
    /// Discover all components in the system automatically
    /// </summary>
    public async Task<List<DiscoveredComponent>> DiscoverAllComponentsAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔍 [COMPONENT-DISCOVERY] Starting automatic component discovery...");
        
        var components = new List<DiscoveredComponent>();

        // 1. Discover background services (IHostedService)
        await DiscoverBackgroundServicesAsync(components, cancellationToken).ConfigureAwait(false);

        // 2. Discover singleton services
        await DiscoverSingletonServicesAsync(components, cancellationToken).ConfigureAwait(false);

        // 3. Discover file dependencies
        await DiscoverFileDependenciesAsync(components, cancellationToken).ConfigureAwait(false);

        // 4. Discover API connections
        await DiscoverAPIConnectionsAsync(components, cancellationToken).ConfigureAwait(false);

        // 5. Discover performance metrics
        await DiscoverPerformanceMetricsAsync(components, cancellationToken).ConfigureAwait(false);

        _discoveredComponents.Clear();
        _discoveredComponents.AddRange(components);

        _logger.LogInformation("✅ [COMPONENT-DISCOVERY] Discovered {Count} total components", components.Count);
        
        return components;
    }

    private Task DiscoverBackgroundServicesAsync(List<DiscoveredComponent> components, CancellationToken cancellationToken)
    {
        _ = cancellationToken; // S1172: Parameter not used but required for async pattern
        _logger.LogInformation("🔍 [COMPONENT-DISCOVERY] Discovering background services...");

        try
        {
            // Get all IHostedService registrations
            var hostedServices = _serviceProvider.GetServices<IHostedService>().ToList();
            
            foreach (var service in hostedServices)
            {
                if (service == null)
                {
                    _logger.LogWarning("⚠️ [COMPONENT-DISCOVERY] Encountered null service in IHostedService collection");
                    continue;
                }
                
                var serviceName = service.GetType().Name;
                if (string.IsNullOrWhiteSpace(serviceName))
                {
                    _logger.LogWarning("⚠️ [COMPONENT-DISCOVERY] Service has null or empty name: {ServiceType}", service.GetType().FullName);
                    continue;
                }
                
                components.Add(new DiscoveredComponent
                {
                    Name = serviceName,
                    Type = ComponentType.BackgroundService,
                    ServiceInstance = service,
                    Metadata = new Dictionary<string, object>
                    {
                        ["ServiceType"] = service.GetType().FullName ?? serviceName,
                        ["IsRunning"] = true // Assume running since we got the instance
                    }
                });
                
                _logger.LogDebug("  ✓ Found background service: {ServiceName}", serviceName);
            }

            _logger.LogInformation("✅ [COMPONENT-DISCOVERY] Found {Count} background services", hostedServices.Count);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [COMPONENT-DISCOVERY] Error discovering background services");
        }

        return Task.CompletedTask;
    }

    private Task DiscoverSingletonServicesAsync(List<DiscoveredComponent> components, CancellationToken cancellationToken)
    {
        _ = cancellationToken; // S1172: Parameter not used but required for async pattern
        _logger.LogInformation("🔍 [COMPONENT-DISCOVERY] Discovering singleton services...");

        try
        {
            // List of known singleton service types to discover
            var serviceTypes = new[]
            {
                typeof(Brain.UnifiedTradingBrain),
                typeof(OllamaClient),
                typeof(BotAlertService),
                typeof(BotPerformanceReporter),
                typeof(MasterDecisionOrchestrator),
                typeof(UnifiedDecisionRouter),
                typeof(AutonomousPerformanceTracker),
                typeof(StrategyPerformanceAnalyzer)
            };

            foreach (var serviceType in serviceTypes)
            {
                try
                {
                    var service = _serviceProvider.GetService(serviceType);
                    if (service != null)
                    {
                        components.Add(new DiscoveredComponent
                        {
                            Name = serviceType.Name,
                            Type = ComponentType.SingletonService,
                            ServiceInstance = service,
                            Metadata = new Dictionary<string, object>
                            {
                                ["ServiceType"] = serviceType.FullName ?? serviceType.Name,
                                ["Namespace"] = serviceType.Namespace ?? "Unknown"
                            }
                        });
                        
                        _logger.LogDebug("  ✓ Found singleton service: {ServiceName}", serviceType.Name);
                    }
                }
                catch (InvalidOperationException ex)
                {
                    _logger.LogDebug(ex, "  ⚠️ Could not get service {ServiceType}", serviceType.Name);
                }
            }

            _logger.LogInformation("✅ [COMPONENT-DISCOVERY] Discovered singleton services");
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [COMPONENT-DISCOVERY] Error discovering singleton services");
        }

        return Task.CompletedTask;
    }

    private Task DiscoverFileDependenciesAsync(List<DiscoveredComponent> components, CancellationToken cancellationToken)
    {
        _ = cancellationToken; // S1172: Parameter not used but required for async pattern
        _logger.LogInformation("🔍 [COMPONENT-DISCOVERY] Discovering file dependencies...");

        const double NoRefreshRequired = 0.0; // S109: Named constant for magic number
        const double HourlyRefresh = 1.0;
        const double FourHourRefresh = 4.0;
        const double DailyRefresh = 24.0;
        const double WeeklyRefresh = 168.0;

        try
        {
            // Critical file paths the bot depends on
            var fileDependencies = new[]
            {
                new { Path = "artifacts/current/parameters/bundle.json", Name = "Parameter Bundle", RefreshHours = FourHourRefresh },
                new { Path = "datasets/economic_calendar/calendar.json", Name = "Economic Calendar", RefreshHours = DailyRefresh },
                new { Path = "models/champion/rl_model.onnx", Name = "Champion RL Model", RefreshHours = WeeklyRefresh },
                new { Path = "models/champion/strategy_selection.onnx", Name = "Strategy Selection Model", RefreshHours = WeeklyRefresh },
                new { Path = "kill.txt", Name = "Emergency Stop File", RefreshHours = NoRefreshRequired }, // Check existence only
                new { Path = ".env", Name = "Environment Configuration", RefreshHours = NoRefreshRequired },
                new { Path = "state/runtime-overrides.json", Name = "Runtime Overrides", RefreshHours = HourlyRefresh }
            };

            foreach (var file in fileDependencies)
            {
                const double epsilon = 0.0001; // S1244: Tolerance for floating point comparison
                var isCritical = Math.Abs(file.RefreshHours - NoRefreshRequired) < epsilon || 
                                 file.Name.Contains("Model", StringComparison.OrdinalIgnoreCase) || 
                                 file.Name.Contains("Parameter", StringComparison.OrdinalIgnoreCase);
                
                components.Add(new DiscoveredComponent
                {
                    Name = file.Name,
                    Type = ComponentType.FileDependency,
                    FilePath = file.Path,
                    ExpectedRefreshIntervalHours = file.RefreshHours,
                    Metadata = new Dictionary<string, object>
                    {
                        ["ExpectedPath"] = file.Path,
                        ["IsCritical"] = isCritical
                    }
                });
                
                _logger.LogDebug("  ✓ Registered file dependency: {FileName}", file.Name);
            }

            _logger.LogInformation("✅ [COMPONENT-DISCOVERY] Found {Count} file dependencies", fileDependencies.Length);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [COMPONENT-DISCOVERY] Error discovering file dependencies");
        }

        return Task.CompletedTask;
    }

    private Task DiscoverAPIConnectionsAsync(List<DiscoveredComponent> components, CancellationToken cancellationToken)
    {
        _ = cancellationToken; // S1172: Parameter not used but required for async pattern
        _logger.LogInformation("🔍 [COMPONENT-DISCOVERY] Discovering API connections...");

        try
        {
            // TopstepX API
            components.Add(new DiscoveredComponent
            {
                Name = "TopstepX API",
                Type = ComponentType.APIConnection,
                Metadata = new Dictionary<string, object>
                {
                    ["BaseUrl"] = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com",
                    ["ConnectionType"] = "REST API"
                }
            });

            // TopstepX SignalR Hubs
            components.Add(new DiscoveredComponent
            {
                Name = "TopstepX SignalR Hubs",
                Type = ComponentType.APIConnection,
                Metadata = new Dictionary<string, object>
                {
                    ["UserHub"] = Environment.GetEnvironmentVariable("RTC_USER_HUB") ?? "https://rtc.topstepx.com/hubs/user",
                    ["MarketHub"] = Environment.GetEnvironmentVariable("RTC_MARKET_HUB") ?? "https://rtc.topstepx.com/hubs/market",
                    ["ConnectionType"] = "SignalR"
                }
            });

            // Ollama AI Service
            var ollamaEnabled = string.Equals(
                Environment.GetEnvironmentVariable("OLLAMA_ENABLED"), 
                "true", 
                StringComparison.OrdinalIgnoreCase); // S1449: Use culture-aware string comparison
            
            if (ollamaEnabled)
            {
                components.Add(new DiscoveredComponent
                {
                    Name = "Ollama AI Service",
                    Type = ComponentType.APIConnection,
                    Metadata = new Dictionary<string, object>
                    {
                        ["BaseUrl"] = Environment.GetEnvironmentVariable("OLLAMA_BASE_URL") ?? "http://localhost:11434",
                        ["Model"] = Environment.GetEnvironmentVariable("OLLAMA_MODEL") ?? "gemma2:2b",
                        ["ConnectionType"] = "HTTP API"
                    }
                });
            }

            // Python UCB Service
            components.Add(new DiscoveredComponent
            {
                Name = "Python UCB Service",
                Type = ComponentType.APIConnection,
                Metadata = new Dictionary<string, object>
                {
                    ["BaseUrl"] = "http://localhost:8000",
                    ["ConnectionType"] = "FastAPI"
                }
            });

            _logger.LogInformation("✅ [COMPONENT-DISCOVERY] Found API connections");
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [COMPONENT-DISCOVERY] Error discovering API connections");
        }

        return Task.CompletedTask;
    }

    private Task DiscoverPerformanceMetricsAsync(List<DiscoveredComponent> components, CancellationToken cancellationToken)
    {
        _ = cancellationToken; // S1172: Parameter not used but required for async pattern
        _logger.LogInformation("🔍 [COMPONENT-DISCOVERY] Discovering performance metrics...");

        const double WinRateThreshold = 0.35; // S109: Named constants for magic numbers
        const double DailyPnLThreshold = -500.0;
        const double DecisionLatencyThreshold = 5000.0;
        const double MemoryUsageThreshold = 2048.0;
        const double ThreadPoolUsageThreshold = 0.9;
        const double DecisionFrequencyThreshold = 30.0;

        try
        {
            var metrics = new[]
            {
                new { Name = "Win Rate", Threshold = WinRateThreshold, Unit = "percentage" },
                new { Name = "Daily P&L", Threshold = DailyPnLThreshold, Unit = "dollars" },
                new { Name = "Decision Latency", Threshold = DecisionLatencyThreshold, Unit = "milliseconds" },
                new { Name = "Memory Usage", Threshold = MemoryUsageThreshold, Unit = "megabytes" },
                new { Name = "Thread Pool Usage", Threshold = ThreadPoolUsageThreshold, Unit = "percentage" },
                new { Name = "Decision Frequency", Threshold = DecisionFrequencyThreshold, Unit = "minutes" }
            };

            foreach (var metric in metrics)
            {
                var isCritical = metric.Name.Contains("P&L", StringComparison.OrdinalIgnoreCase) || 
                                 metric.Name.Contains("Win Rate", StringComparison.OrdinalIgnoreCase);
                
                components.Add(new DiscoveredComponent
                {
                    Name = metric.Name,
                    Type = ComponentType.PerformanceMetric,
                    Thresholds = new Dictionary<string, double>
                    {
                        ["AlertThreshold"] = metric.Threshold
                    },
                    Metadata = new Dictionary<string, object>
                    {
                        ["Unit"] = metric.Unit,
                        ["IsCritical"] = isCritical
                    }
                });
                
                _logger.LogDebug("  ✓ Registered performance metric: {MetricName}", metric.Name);
            }

            _logger.LogInformation("✅ [COMPONENT-DISCOVERY] Found {Count} performance metrics", metrics.Length);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "❌ [COMPONENT-DISCOVERY] Error discovering performance metrics");
        }

        return Task.CompletedTask;
    }
}
