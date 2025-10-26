using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using TradingBot.UnifiedOrchestrator.Services;
using global::BotCore.ML;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Brain hot-reload service that subscribes to model registry updates
/// Implements double-buffered ONNX session swapping for zero-downtime model updates
/// Converted to event-driven IHostedService - no polling loops (Phase 4 refactoring)
/// </summary>
internal sealed class BrainHotReloadService : IHostedService, IDisposable
{
    private readonly ILogger<BrainHotReloadService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly OnnxModelLoader _modelLoader;
    private volatile bool _reloadInProgress;
    private readonly SemaphoreSlim _reloadSemaphore = new(1, 1);
    private IOnnxModelRegistry? _modelRegistry;
    private bool _disposed;

    public BrainHotReloadService(
        ILogger<BrainHotReloadService> logger,
        IServiceProvider serviceProvider,
        OnnxModelLoader modelLoader)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _modelLoader = modelLoader;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🧠 Brain hot-reload service starting (event-driven mode)...");

        // Subscribe to model registry updates
        _modelRegistry = _serviceProvider.GetService<IOnnxModelRegistry>();
        if (_modelRegistry != null)
        {
            _modelRegistry.OnModelsUpdated += HandleModelUpdate;
            _logger.LogInformation("✅ Subscribed to model registry updates - will reload on model change events");
        }
        else
        {
            _logger.LogWarning("⚠️ ModelRegistry not available - hot-reload disabled");
        }

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Brain hot-reload service stopping...");
        Dispose();
        return Task.CompletedTask;
    }

    private async void HandleModelUpdate(string sha)
    {
        if (_reloadInProgress)
        {
            _logger.LogDebug("🔄 Model reload already in progress, skipping update for SHA: {Sha}", sha);
            return;
        }

        await _reloadSemaphore.WaitAsync().ConfigureAwait(false);
        try
        {
            _reloadInProgress = true;
            _logger.LogInformation("🔄 Starting brain hot-reload for SHA: {Sha}", sha);

            await PerformDoubleBufferedReloadAsync(sha).ConfigureAwait(false);

            _logger.LogInformation("✅ Brain hot-reload completed successfully for SHA: {Sha}", sha);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to hot-reload brain for SHA: {Sha}", sha);
        }
        finally
        {
            _reloadInProgress = false;
            _reloadSemaphore.Release();
        }
    }

    /// <summary>
    /// Perform double-buffered model reload with atomic session swap
    /// Ensures zero downtime during model updates
    /// </summary>
    private async Task PerformDoubleBufferedReloadAsync(string sha)
    {
        var modelsPath = "artifacts/current";
        if (!System.IO.Directory.Exists(modelsPath))
        {
            _logger.LogWarning("⚠️ Models directory not found: {ModelsPath}", modelsPath);
            return;
        }

        var modelFiles = System.IO.Directory.GetFiles(modelsPath, "*.onnx", System.IO.SearchOption.AllDirectories);
        var reloadedCount = 0;

        foreach (var modelFile in modelFiles)
        {
            try
            {
                _logger.LogDebug("🔄 Reloading model: {ModelFile}", modelFile);

                // Load new model session (this creates a new session)
                var newSession = await _modelLoader.LoadModelAsync(modelFile, validateInference: true).ConfigureAwait(false);
                
                if (newSession != null)
                {
                    // The OnnxModelLoader handles the atomic swap internally
                    reloadedCount++;
                    _logger.LogInformation("✅ Successfully reloaded model: {ModelFile}", System.IO.Path.GetFileName(modelFile));
                }
                else
                {
                    _logger.LogWarning("⚠️ Failed to reload model: {ModelFile}", System.IO.Path.GetFileName(modelFile));
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error reloading model: {ModelFile}", System.IO.Path.GetFileName(modelFile));
            }
        }

        _logger.LogInformation("🎯 Hot-reload summary: {ReloadedCount}/{TotalCount} models reloaded", 
            reloadedCount, modelFiles.Length);

        // Emit telemetry for monitoring
        EmitHotReloadTelemetry(sha, reloadedCount, modelFiles.Length);
    }

    private void EmitHotReloadTelemetry(string sha, int reloadedCount, int totalCount)
    {
        try
        {
            // Simple telemetry emission - integrate with your monitoring system
            _logger.LogInformation("📊 [TELEMETRY] brain.hot_reload.completed sha={Sha} reloaded={Reloaded} total={Total} timestamp={Timestamp}",
                sha, reloadedCount, totalCount, DateTimeOffset.UtcNow.ToUnixTimeSeconds());
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to emit hot-reload telemetry");
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            // Unsubscribe from model registry updates
            try
            {
                if (_modelRegistry != null)
                {
                    _modelRegistry.OnModelsUpdated -= HandleModelUpdate;
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error unsubscribing from model registry");
            }

            _reloadSemaphore?.Dispose();
            _disposed = true;
        }
    }
}