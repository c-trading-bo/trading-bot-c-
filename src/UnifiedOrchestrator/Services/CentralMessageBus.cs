using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Central message bus implementation - The "ONE BRAIN" communication system
/// </summary>
public class CentralMessageBus : ICentralMessageBus
{
    private readonly ILogger<CentralMessageBus> _logger;
    private readonly ConcurrentDictionary<string, object> _sharedState = new();
    private readonly ConcurrentDictionary<string, List<Delegate>> _subscribers = new();
    private readonly TradingBrainState _brainState = new();
    private bool _isStarted = false;

    public CentralMessageBus(ILogger<CentralMessageBus> logger)
    {
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Starting Central Message Bus");
        _isStarted = true;
        _brainState.IsActive = true;
        _brainState.LastUpdate = DateTime.UtcNow;
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Stopping Central Message Bus");
        _isStarted = false;
        _brainState.IsActive = false;
        return Task.CompletedTask;
    }

    public async Task PublishAsync<T>(string topic, T message, CancellationToken cancellationToken = default)
    {
        if (!_isStarted)
        {
            _logger.LogWarning("Message bus not started, ignoring message for topic: {Topic}", topic);
            return;
        }

        if (_subscribers.TryGetValue(topic, out var handlers))
        {
            var tasks = handlers.Cast<Func<T, Task>>().Select(handler => 
            {
                try
                {
                    return handler(message);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in message handler for topic: {Topic}", topic);
                    return Task.CompletedTask;
                }
            });

            await Task.WhenAll(tasks);
        }

        _logger.LogDebug("Published message to topic: {Topic}", topic);
    }

    public void Subscribe<T>(string topic, Func<T, Task> handler)
    {
        _subscribers.AddOrUpdate(topic, 
            [handler], 
            (key, existing) => existing.Concat([handler]).ToList());
        
        _logger.LogDebug("Subscribed to topic: {Topic}", topic);
    }

    public void UpdateSharedState(string key, object value)
    {
        _sharedState.AddOrUpdate(key, value, (k, v) => value);
        _brainState.LastUpdate = DateTime.UtcNow;
        _logger.LogDebug("Updated shared state: {Key}", key);
    }

    public T? GetSharedState<T>(string key)
    {
        if (_sharedState.TryGetValue(key, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        return default;
    }

    public TradingBrainState GetBrainState()
    {
        return _brainState.Clone();
    }

    public async Task<TradingDecision> RequestTradingDecisionAsync(TradingSignal signal, CancellationToken cancellationToken = default)
    {
        // Simulate decision making process
        await Task.Delay(50, cancellationToken);
        
        return new TradingDecision
        {
            DecisionId = Guid.NewGuid().ToString(),
            Symbol = signal.Symbol,
            Signal = signal,
            Action = TradingAction.Hold,
            Confidence = 0.5m,
            Timestamp = DateTime.UtcNow,
            Reasoning = new Dictionary<string, object>
            {
                ["source"] = "CentralMessageBus",
                ["signal_strength"] = signal.Strength,
                ["market_regime"] = _brainState.CurrentMarketRegime
            }
        };
    }
}