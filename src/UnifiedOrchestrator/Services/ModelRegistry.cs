using System;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Model registry interface for hot-reload notifications
/// Enables brain to reload ONNX models without restart
/// </summary>
internal interface IModelRegistry 
{ 
    event Action<string>? OnModelsUpdated; 
    void NotifyUpdated(string sha); 
}

/// <summary>
/// Simple model registry implementation that notifies brain of model updates
/// </summary>
internal sealed class ModelRegistry : IModelRegistry 
{
    public event Action<string>? OnModelsUpdated;
    
    public void NotifyUpdated(string sha) => OnModelsUpdated?.Invoke(sha);
}