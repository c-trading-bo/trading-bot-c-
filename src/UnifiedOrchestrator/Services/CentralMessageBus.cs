using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Text.Json;
using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Central Message Bus - The "ONE BRAIN" communication system
/// Enables all components to communicate in real-time for unified trading decisions
/// </summary>
public class CentralMessageBus : ICentralMessageBus, IDisposable
{
    private readonly ILogger<CentralMessageBus> _logger;
    private readonly ConcurrentDictionary<string, List<IMessageHandler>> _handlers = new();
    private readonly ConcurrentDictionary<string, object> _sharedState = new();
    private readonly ConcurrentQueue<TradingMessage> _messageQueue = new();
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    
    // Real-time state for all trading components
    private readonly TradingBrainState _brainState = new();
    private readonly object _stateLock = new();
    
    private Task? _processingTask;
    private bool _isRunning = false;

    public CentralMessageBus(ILogger<CentralMessageBus> logger)
    {
        _logger = logger;
    }

    #region ICentralMessageBus Implementation

    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        if (_isRunning) return;

        _logger.LogInformation("🧠 Starting Central Message Bus - ONE BRAIN communication system");
        
        _isRunning = true;
        _processingTask = ProcessMessagesAsync(_cancellationTokenSource.Token);
        
        // Initialize shared state
        await InitializeSharedStateAsync();
        
        _logger.LogInformation("✅ Central Message Bus started - All components can now communicate");
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        if (!_isRunning) return;

        _logger.LogInformation("🛑 Stopping Central Message Bus");
        
        _isRunning = false;
        _cancellationTokenSource.Cancel();
        
        if (_processingTask != null)
        {
            await _processingTask;
        }
        
        _logger.LogInformation("✅ Central Message Bus stopped");
    }

    public async Task PublishAsync<T>(string topic, T message, CancellationToken cancellationToken = default)
    {
        var tradingMessage = new TradingMessage
        {
            Id = Guid.NewGuid().ToString(),
            Topic = topic,
            Payload = JsonSerializer.Serialize(message),
            Timestamp = DateTime.UtcNow,
            MessageType = typeof(T).Name
        };

        _messageQueue.Enqueue(tradingMessage);
        
        _logger.LogDebug("📤 Published message to topic: {Topic} ({MessageType})", topic, typeof(T).Name);
        
        await Task.CompletedTask;
    }

    public void Subscribe<T>(string topic, Func<T, Task> handler)
    {
        var wrappedHandler = new MessageHandler<T>(handler);
        
        _handlers.AddOrUpdate(topic, 
            new List<IMessageHandler> { wrappedHandler },
            (key, existing) => 
            {
                existing.Add(wrappedHandler);
                return existing;
            });
        
        _logger.LogInformation("📥 Subscribed to topic: {Topic} ({MessageType})", topic, typeof(T).Name);
    }

    public void UpdateSharedState(string key, object value)
    {
        lock (_stateLock)
        {
            _sharedState.AddOrUpdate(key, value, (k, existing) => value);
            
            // Update brain state for key components
            UpdateBrainState(key, value);
        }
        
        _logger.LogDebug("🧠 Updated shared state: {Key}", key);
    }

    public T? GetSharedState<T>(string key)
    {
        if (_sharedState.TryGetValue(key, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        return default(T);
    }

    public TradingBrainState GetBrainState()
    {
        lock (_stateLock)
        {
            return _brainState.Clone();
        }
    }

    public async Task<TradingDecision> RequestTradingDecisionAsync(TradingSignal signal, CancellationToken cancellationToken = default)
    {
        // Publish signal to all components
        await PublishAsync("trading.signal", signal, cancellationToken);
        
        // Collect inputs from all systems
        var decision = new TradingDecision
        {
            Signal = signal,
            Timestamp = DateTime.UtcNow,
            DecisionId = Guid.NewGuid().ToString()
        };

        // Get ML/RL recommendations
        var mlRecommendation = GetSharedState<MLRecommendation>("ml.latest_recommendation");
        if (mlRecommendation != null)
        {
            decision.MLConfidence = mlRecommendation.Confidence;
            decision.MLStrategy = mlRecommendation.RecommendedStrategy;
        }

        // Get risk assessment
        var riskAssessment = GetSharedState<RiskAssessment>("risk.current_assessment");
        if (riskAssessment != null)
        {
            decision.RiskScore = riskAssessment.RiskScore;
            decision.MaxPositionSize = riskAssessment.MaxPositionSize;
        }

        // Get market regime
        var marketRegime = GetSharedState<MarketRegime>("intelligence.market_regime");
        if (marketRegime != null)
        {
            decision.MarketRegime = marketRegime.CurrentRegime;
            decision.RegimeConfidence = marketRegime.Confidence;
        }

        // Combine all inputs for final decision
        decision.Action = DetermineOptimalAction(decision);
        decision.Confidence = CalculateOverallConfidence(decision);
        
        // Publish decision back to all components
        await PublishAsync("trading.decision", decision, cancellationToken);
        
        _logger.LogInformation("🎯 Trading decision made: {Action} for {Symbol} (Confidence: {Confidence:P})", 
            decision.Action, signal.Symbol, decision.Confidence);
        
        return decision;
    }

    #endregion

    #region Private Methods

    private async Task ProcessMessagesAsync(CancellationToken cancellationToken)
    {
        _logger.LogDebug("🔄 Message processing loop started");
        
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                while (_messageQueue.TryDequeue(out var message))
                {
                    await ProcessMessageAsync(message, cancellationToken);
                }
                
                await Task.Delay(10, cancellationToken); // Small delay to prevent busy waiting
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing messages");
            }
        }
        
        _logger.LogDebug("🔄 Message processing loop stopped");
    }

    private async Task ProcessMessageAsync(TradingMessage message, CancellationToken cancellationToken)
    {
        if (_handlers.TryGetValue(message.Topic, out var handlers))
        {
            var tasks = handlers.Select(async handler =>
            {
                try
                {
                    await handler.HandleAsync(message, cancellationToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ Error handling message {MessageId} on topic {Topic}", 
                        message.Id, message.Topic);
                }
            });
            
            await Task.WhenAll(tasks);
        }
    }

    private async Task InitializeSharedStateAsync()
    {
        // Initialize with default state
        UpdateSharedState("system.status", "initializing");
        UpdateSharedState("trading.active_positions", new Dictionary<string, decimal>());
        UpdateSharedState("risk.daily_pnl", 0m);
        UpdateSharedState("intelligence.last_update", DateTime.UtcNow);
        
        // Initialize brain state
        lock (_stateLock)
        {
            _brainState.IsActive = true;
            _brainState.LastUpdate = DateTime.UtcNow;
            _brainState.ConnectedComponents = new List<string>();
            _brainState.ActiveStrategies = new List<string> { "S2", "S3", "S6", "S11" }; // Primary strategies
        }
        
        await Task.CompletedTask;
    }

    private void UpdateBrainState(string key, object value)
    {
        // Update the central brain state based on key updates
        switch (key)
        {
            case "trading.active_positions":
                if (value is Dictionary<string, decimal> positions)
                {
                    _brainState.ActivePositions = positions;
                }
                break;
                
            case "risk.daily_pnl":
                if (value is decimal pnl)
                {
                    _brainState.DailyPnL = pnl;
                }
                break;
                
            case "intelligence.market_regime":
                if (value is MarketRegime regime)
                {
                    _brainState.MarketRegime = regime.CurrentRegime;
                }
                break;
                
            case "ml.strategy_selection":
                if (value is string[] strategies)
                {
                    _brainState.ActiveStrategies = strategies.ToList();
                }
                break;
        }
        
        _brainState.LastUpdate = DateTime.UtcNow;
    }

    private TradingAction DetermineOptimalAction(TradingDecision decision)
    {
        // Combine all intelligence for optimal action
        var signal = decision.Signal;
        
        // Risk-first approach
        if (decision.RiskScore > 0.8m)
        {
            return TradingAction.Hold; // Too risky
        }
        
        // ML confidence check
        if (decision.MLConfidence < 0.6m)
        {
            return TradingAction.Hold; // Not confident enough
        }
        
        // Market regime check
        if (decision.MarketRegime == "CHOPPY" && decision.RegimeConfidence > 0.7m)
        {
            return TradingAction.Hold; // Bad regime for directional trades
        }
        
        // Signal strength
        return signal.Strength switch
        {
            > 0.8m => signal.Direction == "LONG" ? TradingAction.Buy : TradingAction.Sell,
            > 0.6m => signal.Direction == "LONG" ? TradingAction.BuySmall : TradingAction.SellSmall,
            _ => TradingAction.Hold
        };
    }

    private decimal CalculateOverallConfidence(TradingDecision decision)
    {
        var weights = new Dictionary<string, decimal>
        {
            ["signal"] = 0.3m,
            ["ml"] = 0.3m,
            ["risk"] = 0.2m,
            ["regime"] = 0.2m
        };
        
        decimal totalConfidence = 0m;
        
        totalConfidence += decision.Signal.Strength * weights["signal"];
        totalConfidence += decision.MLConfidence * weights["ml"];
        totalConfidence += (1 - decision.RiskScore) * weights["risk"]; // Lower risk = higher confidence
        totalConfidence += decision.RegimeConfidence * weights["regime"];
        
        return Math.Max(0m, Math.Min(1m, totalConfidence));
    }

    #endregion

    public void Dispose()
    {
        _cancellationTokenSource?.Cancel();
        _cancellationTokenSource?.Dispose();
        _processingTask?.Dispose();
    }
}

/// <summary>
/// Generic message handler
/// </summary>
public class MessageHandler<T> : IMessageHandler
{
    private readonly Func<T, Task> _handler;

    public MessageHandler(Func<T, Task> handler)
    {
        _handler = handler;
    }

    public async Task HandleAsync(TradingMessage message, CancellationToken cancellationToken)
    {
        if (JsonSerializer.Deserialize<T>(message.Payload) is T typedMessage)
        {
            await _handler(typedMessage);
        }
    }
}