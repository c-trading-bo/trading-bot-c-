using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Helper methods for IntelligenceOrchestrator extracted to reduce file size
/// Contains private helper methods for model management, maintenance, and workflow execution
/// </summary>
public partial class IntelligenceOrchestratorHelpers
{
    private readonly ILogger<IntelligenceOrchestrator> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IntelligenceStackConfig _config;
    private readonly IRegimeDetector _regimeDetector;
    private readonly IModelRegistry _modelRegistry;
    private readonly ICalibrationManager _calibrationManager;
    private readonly IDecisionLogger _decisionLogger;
    private readonly FeatureEngineer _featureEngineer;
    private readonly CloudFlowService _cloudFlowService;
    
    // State tracking (shared with main orchestrator)
    private readonly Dictionary<string, ModelArtifact> _activeModels;
    private readonly object _lock = new();
    private bool _isInitialized;
    private bool _isTradingEnabled;
    private DateTime _lastNightlyMaintenance = DateTime.MinValue;
    
    public IntelligenceOrchestratorHelpers(
        ILogger<IntelligenceOrchestrator> logger,
        IServiceProvider serviceProvider,
        IntelligenceStackConfig config,
        IRegimeDetector regimeDetector,
        IModelRegistry modelRegistry,
        ICalibrationManager calibrationManager,
        IDecisionLogger decisionLogger,
        FeatureEngineer featureEngineer,
        CloudFlowService cloudFlowService,
        Dictionary<string, ModelArtifact> activeModels)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _config = config;
        _regimeDetector = regimeDetector;
        _modelRegistry = modelRegistry;
        _calibrationManager = calibrationManager;
        _decisionLogger = decisionLogger;
        _featureEngineer = featureEngineer;
        _cloudFlowService = cloudFlowService;
        _activeModels = activeModels;
    }

    public async Task LoadActiveModelsAsync(CancellationToken cancellationToken)
    {
        try
        {
            lock (_lock)
            {
                _activeModels.Clear();
            }

            var models = await _modelRegistry.GetActiveModelsAsync(cancellationToken).ConfigureAwait(false);
            
            lock (_lock)
            {
                foreach (var model in models)
                {
                    _activeModels[model.Id] = model;
                }
            }
            
            ModelsLoaded(_logger, models.Count(), null);
        }
        catch (ArgumentException ex)
        {
            ModelLoadError(_logger, ex.Message, ex);
        }
        catch (InvalidOperationException ex)
        {
            ModelLoadError(_logger, ex.Message, ex);
        }
    }

    public async Task PerformNightlyMaintenanceAsync(CancellationToken cancellationToken)
    {
        try
        {
            MaintenanceStarted(_logger, null);
            
            // Model cleanup and optimization
            await _modelRegistry.CleanupExpiredModelsAsync(cancellationToken).ConfigureAwait(false);
            
            // Feature store maintenance
            var featureStore = _serviceProvider.GetService<IFeatureStore>();
            if (featureStore != null)
            {
                await featureStore.OptimizeStorageAsync(cancellationToken).ConfigureAwait(false);
            }
            
            // Calibration updates
            await _calibrationManager.PerformNightlyCalibrationAsync(cancellationToken).ConfigureAwait(false);
            
            lock (_lock)
            {
                _lastNightlyMaintenance = DateTime.UtcNow;
            }
            
            MaintenanceCompleted(_logger, null);
        }
        catch (TimeoutException ex)
        {
            MaintenanceError(_logger, ex.Message, ex);
        }
        catch (InvalidOperationException ex)
        {
            MaintenanceError(_logger, ex.Message, ex);
        }
    }

    // Workflow execution helper methods
    public async Task<WorkflowExecutionResult> LoadModelsWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        await LoadActiveModelsAsync(cancellationToken).ConfigureAwait(false);
        return new WorkflowExecutionResult { Success = true };
    }

    public async Task<WorkflowExecutionResult> PerformMaintenanceWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        await PerformNightlyMaintenanceAsync(cancellationToken).ConfigureAwait(false);
        return new WorkflowExecutionResult { Success = true };
    }

    public async Task<WorkflowExecutionResult> AnalyzeCorrelationsWrapperAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        await AnalyzeCorrelationsAsync(context, cancellationToken).ConfigureAwait(false);
        return new WorkflowExecutionResult { Success = true };
    }

    private async Task AnalyzeCorrelationsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        
        // Extract market data from context
        var marketData = WorkflowHelpers.ExtractMarketDataFromWorkflow(context);
        
        // Perform correlation analysis using feature engineer
        var correlations = await _featureEngineer.AnalyzeCorrelationsAsync(cancellationToken).ConfigureAwait(false);
    }

    #region LoggerMessage Delegates

    [LoggerMessage(EventId = 3001, Level = LogLevel.Information, Message = "[INTELLIGENCE] Loaded {ModelCount} active models")]
    private static partial void ModelsLoaded(ILogger logger, int modelCount, Exception? ex);

    [LoggerMessage(EventId = 3002, Level = LogLevel.Error, Message = "[INTELLIGENCE] Model load error: {ErrorMessage}")]
    private static partial void ModelLoadError(ILogger logger, string errorMessage, Exception ex);

    [LoggerMessage(EventId = 3003, Level = LogLevel.Information, Message = "[INTELLIGENCE] Nightly maintenance started")]
    private static partial void MaintenanceStarted(ILogger logger, Exception? ex);

    [LoggerMessage(EventId = 3004, Level = LogLevel.Information, Message = "[INTELLIGENCE] Nightly maintenance completed")]
    private static partial void MaintenanceCompleted(ILogger logger, Exception? ex);

    [LoggerMessage(EventId = 3005, Level = LogLevel.Error, Message = "[INTELLIGENCE] Maintenance error: {ErrorMessage}")]
    private static partial void MaintenanceError(ILogger logger, string errorMessage, Exception ex);

    #endregion
}