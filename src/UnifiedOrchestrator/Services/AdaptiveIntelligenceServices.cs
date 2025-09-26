using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Adaptive intelligence coordinator interface
/// </summary>
internal interface IAdaptiveIntelligenceCoordinator
{
    Task<Dictionary<string, object>> GetAdaptiveParametersAsync(string context, CancellationToken cancellationToken = default);
    Task UpdatePerformanceMetricsAsync(string context, Dictionary<string, double> metrics, CancellationToken cancellationToken = default);
    Task TriggerAdaptationAsync(string reason, CancellationToken cancellationToken = default);
}

/// <summary>
/// Adaptive intelligence coordinator implementation
/// Manages dynamic parameter adaptation based on performance feedback
/// </summary>
internal sealed class AdaptiveIntelligenceCoordinator : BackgroundService, IAdaptiveIntelligenceCoordinator
{
    private readonly ILogger<AdaptiveIntelligenceCoordinator> _logger;
    private readonly ConcurrentDictionary<string, Dictionary<string, object>> _adaptiveParameters = new();
    private readonly ConcurrentDictionary<string, Dictionary<string, double>> _performanceMetrics = new();
    
    public AdaptiveIntelligenceCoordinator(ILogger<AdaptiveIntelligenceCoordinator> logger)
    {
        _logger = logger;
        InitializeDefaultParameters();
    }
    
    private void InitializeDefaultParameters()
    {
        _adaptiveParameters["ucb"] = new Dictionary<string, object>
        {
            { "exploration_rate", 0.1 },
            { "confidence_threshold", 0.75 },
            { "min_decisions", 400 }
        };
        
        _adaptiveParameters["bracket"] = new Dictionary<string, object>
        {
            { "mode", "Auto" },
            { "frozen_per_position", true }
        };
        
        _adaptiveParameters["decision"] = new Dictionary<string, object>
        {
            { "hold_rate", 0.0 }, // Never hold
            { "switch_frequency", 1.0 }
        };
    }
    
    public async Task<Dictionary<string, object>> GetAdaptiveParametersAsync(string context, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        if (_adaptiveParameters.TryGetValue(context, out var parameters))
        {
            return new Dictionary<string, object>(parameters);
        }
        
        return new Dictionary<string, object>();
    }
    
    public async Task UpdatePerformanceMetricsAsync(string context, Dictionary<string, double> metrics, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        _performanceMetrics.AddOrUpdate(context, metrics, (key, existing) =>
        {
            foreach (var kvp in metrics)
            {
                existing[kvp.Key] = kvp.Value;
            }
            return existing;
        });
        
        _logger.LogDebug("Updated performance metrics for context {Context}: {@Metrics}", context, metrics);
    }
    
    public async Task TriggerAdaptationAsync(string reason, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        _logger.LogInformation("Triggered adaptation: {Reason}", reason);
        
        // Adapt UCB parameters based on performance
        if (_performanceMetrics.TryGetValue("ucb", out var ucbMetrics))
        {
            if (ucbMetrics.TryGetValue("confidence", out var confidence) && confidence < 0.7)
            {
                // Lower confidence threshold if struggling
                if (_adaptiveParameters.TryGetValue("ucb", out var ucbParams))
                {
                    ucbParams["confidence_threshold"] = Math.Max(0.6, confidence - 0.05);
                    _logger.LogInformation("Adapted UCB confidence threshold to {Threshold}", ucbParams["confidence_threshold"]);
                }
            }
        }
        
        // Ensure validation mode settings
        if (_adaptiveParameters.TryGetValue("ucb", out var ucbParameters))
        {
            ucbParameters["exploration_rate"] = 0.0; // UCB exploration OFF in validation
            _logger.LogDebug("Set UCB exploration OFF for validation mode");
        }
        
        if (_adaptiveParameters.TryGetValue("bracket", out var bracketParameters))
        {
            bracketParameters["frozen_per_position"] = true; // Bracket UCB frozen per position epoch
            _logger.LogDebug("Set bracket UCB frozen per position epoch");
        }
    }
    
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Periodic adaptation check
                await TriggerAdaptationAsync("periodic_check", stoppingToken).ConfigureAwait(false);
                
                // Wait 5 minutes before next adaptation cycle
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in adaptive intelligence coordinator");
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken).ConfigureAwait(false);
            }
        }
    }
}

/// <summary>
/// Adaptive parameter service interface
/// </summary>
internal interface IAdaptiveParameterService
{
    T GetParameter<T>(string name, T defaultValue = default!);
    void SetParameter<T>(string name, T value);
    void ResetToDefaults();
}

/// <summary>
/// Adaptive parameter service implementation
/// Manages runtime parameter adjustments
/// </summary>
internal sealed class AdaptiveParameterService : IAdaptiveParameterService
{
    private readonly ILogger<AdaptiveParameterService> _logger;
    private readonly ConcurrentDictionary<string, object> _parameters = new();
    
    public AdaptiveParameterService(ILogger<AdaptiveParameterService> logger)
    {
        _logger = logger;
        InitializeDefaults();
    }
    
    private void InitializeDefaults()
    {
        _parameters["decision.hold_rate"] = 0.0;
        _parameters["trade.rate_per_min"] = 5.0;
        _parameters["controller.switches_per_hour"] = 4.0;
        _parameters["latency.ms.p50"] = 50.0;
        _parameters["latency.ms.p95"] = 150.0;
        _parameters["ucb.exploration_off"] = true; // Validation mode
        _parameters["bracket.frozen_per_position"] = true; // Validation mode
    }
    
    public T GetParameter<T>(string name, T defaultValue = default!)
    {
        if (_parameters.TryGetValue(name, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        
        return defaultValue;
    }
    
    public void SetParameter<T>(string name, T value)
    {
        if (value == null)
        {
            _parameters.TryRemove(name, out _);
            return;
        }
        
        _parameters.AddOrUpdate(name, value, (key, existing) => value);
        _logger.LogDebug("Updated parameter {Name} = {Value}", name, value);
    }
    
    public void ResetToDefaults()
    {
        _parameters.Clear();
        InitializeDefaults();
        _logger.LogInformation("Reset parameters to defaults");
    }
}

/// <summary>
/// Runtime configuration bus interface
/// </summary>
internal interface IRuntimeConfigBus
{
    Task PublishConfigUpdateAsync(string configPath, object value, CancellationToken cancellationToken = default);
    Task<T> GetConfigValueAsync<T>(string configPath, T defaultValue = default!, CancellationToken cancellationToken = default);
    void Subscribe(string configPath, Action<object> handler);
}

/// <summary>
/// Runtime configuration bus implementation
/// Handles real-time configuration updates and notifications
/// </summary>
internal sealed class RuntimeConfigBus : IRuntimeConfigBus
{
    private readonly ILogger<RuntimeConfigBus> _logger;
    private readonly ConcurrentDictionary<string, object> _configValues = new();
    private readonly ConcurrentDictionary<string, List<Action<object>>> _subscribers = new();
    
    public RuntimeConfigBus(ILogger<RuntimeConfigBus> logger)
    {
        _logger = logger;
    }
    
    public async Task PublishConfigUpdateAsync(string configPath, object value, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        _configValues.AddOrUpdate(configPath, value, (key, existing) => value);
        
        if (_subscribers.TryGetValue(configPath, out var handlers))
        {
            foreach (var handler in handlers)
            {
                try
                {
                    handler(value);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in config update handler for {ConfigPath}", configPath);
                }
            }
        }
        
        _logger.LogDebug("Published config update: {ConfigPath} = {Value}", configPath, value);
    }
    
    public async Task<T> GetConfigValueAsync<T>(string configPath, T defaultValue = default!, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        if (_configValues.TryGetValue(configPath, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        
        return defaultValue;
    }
    
    public void Subscribe(string configPath, Action<object> handler)
    {
        _subscribers.AddOrUpdate(configPath, new List<Action<object>> { handler }, (key, existing) =>
        {
            existing.Add(handler);
            return existing;
        });
        
        _logger.LogDebug("Added subscriber for config path {ConfigPath}", configPath);
    }
}