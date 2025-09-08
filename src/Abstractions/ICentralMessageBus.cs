namespace TradingBot.Abstractions;

/// <summary>
/// Interface for the central message bus - The "ONE BRAIN" communication system
/// Enables all components to communicate in real-time for unified trading decisions
/// </summary>
public interface ICentralMessageBus
{
    Task StartAsync(CancellationToken cancellationToken = default);
    Task StopAsync(CancellationToken cancellationToken = default);
    Task PublishAsync<T>(string topic, T message, CancellationToken cancellationToken = default);
    void Subscribe<T>(string topic, Func<T, Task> handler);
    void UpdateSharedState(string key, object value);
    T? GetSharedState<T>(string key);
    TradingBrainState GetBrainState();
    Task<TradingDecision> RequestTradingDecisionAsync(TradingSignal signal, CancellationToken cancellationToken = default);
}

/// <summary>
/// Message handler interface
/// </summary>
public interface IMessageHandler
{
    Task HandleAsync(TradingMessage message, CancellationToken cancellationToken);
}