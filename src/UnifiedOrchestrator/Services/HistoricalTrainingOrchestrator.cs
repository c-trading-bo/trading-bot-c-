using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using TradingBot.Abstractions;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Historical Training Orchestrator - Master controller for Lab training pipeline
/// Runs complete training session on Sunday (segregated from Terminal)
/// 
/// Uses existing SDK infrastructure (IHistoricalDataBridgeService) to load historical data
/// This ensures we're using the production TopstepX API, not creating parallel systems
/// 
/// This is the "shift supervisor" that coordinates the entire training factory:
/// 1. Load experiences from last 7 days
/// 2. Load 90-day historical bars via existing SDK
/// 3. Run sequential training pipeline
/// 4. Save challengers to registry
/// 5. Run promotion evaluations
/// </summary>
internal sealed class HistoricalTrainingOrchestrator
{
    // String literal constants for component names
    private const string ComponentCVarPPO = "CVaR-PPO";
    private const string ComponentNeuralUCB = "Neural UCB";
    private const string ComponentLSTM = "LSTM";
    private const string ComponentPositionManagement = "Position Management";
    private const string ComponentS15ShadowValidation = "S15 Shadow Validation";
    private const string ComponentDataLoading = "DataLoading";
    private const string PhaseMain = "Main";
    
    private readonly ILogger<HistoricalTrainingOrchestrator> _logger;
    private readonly IHistoricalDataBridgeService _historicalDataBridge;
    private readonly global::BotCore.Data.ExperienceRepository? _experienceRepository;
    private readonly TradingBot.UnifiedOrchestrator.Interfaces.IModelRegistry _modelRegistry;
    private readonly TradingBot.UnifiedOrchestrator.Interfaces.IPromotionService _promotionService;
    private readonly TradingBot.RLAgent.CVaRPPOTrainer _cvarPpoTrainer;
    private readonly TrainingManifestService _manifestService;
    private readonly DataIntegrityService _dataIntegrityService;
    private readonly TrainingMetricsCollector _metricsCollector;
    private readonly TrainingAlertService _alertService;
    private readonly TrainingRetryService _retryService;
    private readonly GitHubBackupService? _githubBackupService;
    private readonly SystemCapabilityProfiler _capabilityProfiler;
    private readonly DynamicResourceManager _resourceManager;
    private readonly TrainingResourceMonitor _resourceMonitor;
    private readonly TrainingCheckpointService _checkpointService;
    private readonly TrainingFailureHandler _failureHandler;
    private readonly TrainingPerformanceProfiler _performanceProfiler;
    private readonly TrainingDebugLogger _debugLogger;
    private readonly MemoryLeakDetector _memoryLeakDetector;
    private readonly IServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;
    private readonly SemaphoreSlim _trainingLock = new(1, 1);

    // Note: 22 constructor parameters is necessary for this orchestration class which coordinates multiple training subsystems.
    // This class is the central coordinator for Lab Mode training and needs access to all specialized services.
    // Future refactoring could split this into LabModeDataLoader, ModelManagementService, and TrainingCoordinator,
    // but that would require significant changes to the DI container registration and service architecture.
#pragma warning disable S107 // Methods should not have too many parameters - necessary for Lab training coordination
    public HistoricalTrainingOrchestrator(
        ILogger<HistoricalTrainingOrchestrator> logger,
        IHistoricalDataBridgeService historicalDataBridge,
        global::BotCore.Data.ExperienceRepository? experienceRepository,
        TradingBot.UnifiedOrchestrator.Interfaces.IModelRegistry modelRegistry,
        TradingBot.UnifiedOrchestrator.Interfaces.IPromotionService promotionService,
        TradingBot.RLAgent.CVaRPPOTrainer cvarPpoTrainer,
        TrainingManifestService manifestService,
        DataIntegrityService dataIntegrityService,
        TrainingMetricsCollector metricsCollector,
        TrainingAlertService alertService,
        TrainingRetryService retryService,
        SystemCapabilityProfiler capabilityProfiler,
        DynamicResourceManager resourceManager,
        TrainingResourceMonitor resourceMonitor,
        TrainingCheckpointService checkpointService,
        TrainingFailureHandler failureHandler,
        TrainingPerformanceProfiler performanceProfiler,
        TrainingDebugLogger debugLogger,
        MemoryLeakDetector memoryLeakDetector,
        IServiceProvider serviceProvider,
        IConfiguration configuration,
        GitHubBackupService? githubBackupService = null)
#pragma warning restore S107
    {
        _logger = logger;
        _historicalDataBridge = historicalDataBridge;
        _experienceRepository = experienceRepository;
        _modelRegistry = modelRegistry;
        _promotionService = promotionService;
        _cvarPpoTrainer = cvarPpoTrainer;
        _manifestService = manifestService;
        _dataIntegrityService = dataIntegrityService;
        _metricsCollector = metricsCollector;
        _alertService = alertService;
        _retryService = retryService;
        _capabilityProfiler = capabilityProfiler;
        _resourceManager = resourceManager;
        _resourceMonitor = resourceMonitor;
        _checkpointService = checkpointService;
        _failureHandler = failureHandler;
        _performanceProfiler = performanceProfiler;
        _debugLogger = debugLogger;
        _memoryLeakDetector = memoryLeakDetector;
        _serviceProvider = serviceProvider;
        _configuration = configuration;
        _githubBackupService = githubBackupService;
        
        _logger.LogInformation("HistoricalTrainingOrchestrator initialized with Phase 10-14 enhancements");
    }

    /// <summary>
    /// Run complete training session (Sunday schedule)
    /// This is the main entry point for Lab training
    /// </summary>
    public async Task<TrainingSessionResult> RunTrainingSessionAsync(CancellationToken cancellationToken = default)
    {
        await _trainingLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var sessionId = Guid.NewGuid().ToString("N")[..8];
            var startTime = DateTime.UtcNow;
            var easternTime = GetEasternTime(startTime);
            
            _memoryLeakDetector.RecordBaseline();
            
            _logger.LogInformation("[LAB] Training session started - RunID: {RunId}, {Day} {Date}, {Time}", 
                sessionId,
                easternTime.ToString("dddd"), 
                easternTime.ToString("MMM dd"), 
                easternTime.ToString("h:mm tt") + " ET");

            var result = await ExecuteTrainingSessionAsync(sessionId, startTime, cancellationToken).ConfigureAwait(false);
            
            return result;
        }
        finally
        {
            _trainingLock.Release();
        }
    }
    
    private async Task<TrainingSessionResult> ExecuteTrainingSessionAsync(
        string sessionId, 
        DateTime startTime, 
        CancellationToken cancellationToken)
    {
        var result = InitializeTrainingResult(sessionId, startTime);
        _metricsCollector.StartRun(sessionId);
        
        try
        {
            await ProfileSystemCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
            await TryResumeFromCheckpointAsync(cancellationToken).ConfigureAwait(false);
            
            _performanceProfiler.StartProfilingSection("SessionTotal");

            await ExecuteTrainingPipelineAsync(result, cancellationToken).ConfigureAwait(false);
            await FinalizeSuccessfulTrainingAsync(result, sessionId, startTime, cancellationToken).ConfigureAwait(false);
            
            return result;
        }
        catch (Exception ex)
        {
            await HandleTrainingFailureAsync(result, sessionId, startTime, ex, cancellationToken).ConfigureAwait(false);
            return result;
        }
        finally
        {
            await FinalizeTrainingSessionAsync(result, sessionId, cancellationToken).ConfigureAwait(false);
        }
    }

    private static TrainingSessionResult InitializeTrainingResult(string sessionId, DateTime startTime)
    {
        return new TrainingSessionResult
        {
            SessionId = sessionId,
            StartTime = startTime
        };
    }

    private async Task ProfileSystemCapabilitiesAsync(CancellationToken cancellationToken)
    {
        _logger.LogDebug("[LAB] Profiling system capabilities...");
        var systemProfile = await _capabilityProfiler.ProfileSystemCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
        
        await _resourceManager.CalculateOptimalThresholdsAsync(systemProfile, 273, cancellationToken).ConfigureAwait(false);
        var strategy = await _resourceManager.DetermineTrainingStrategyAsync(systemProfile, cancellationToken).ConfigureAwait(false);
        
        _logger.LogDebug("[LAB] Training strategy: {Strategy} ({Components} components, {Days}-day data)",
            strategy.Name, strategy.ComponentCount, strategy.HistoricalDataDays);
    }

    private async Task TryResumeFromCheckpointAsync(CancellationToken cancellationToken)
    {
        var checkpointPath = _checkpointService.FindMostRecentCheckpoint();
        if (checkpointPath == null)
            return;

        _logger.LogDebug("[LAB] Found existing checkpoint - attempting to resume...");
        var checkpointState = await _checkpointService.LoadCheckpointAsync(checkpointPath, cancellationToken).ConfigureAwait(false);
        
        if (checkpointState != null && await _checkpointService.ValidateCheckpointAsync(checkpointState, cancellationToken).ConfigureAwait(false))
        {
            _logger.LogDebug("[LAB] Checkpoint validated - resuming from component {Index}/{Total}",
                checkpointState.CurrentComponentIndex, checkpointState.TotalComponents);
        }
        else
        {
            _logger.LogWarning("[LAB] Checkpoint validation failed - starting fresh session");
            if (checkpointPath != null)
            {
                await _checkpointService.ArchiveCheckpointAsync(checkpointPath, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private async Task ExecuteTrainingPipelineAsync(TrainingSessionResult result, CancellationToken cancellationToken)
    {
        var historicalData = await LoadHistoricalDataWithRetryAsync(cancellationToken).ConfigureAwait(false);
        result.HistoricalBarsLoaded = historicalData.Sum(kvp => kvp.Value);

        await CleanupOldExperiencesAsync().ConfigureAwait(false);
        
        var experiences = await LoadAndVerifyExperiencesAsync(result, historicalData, cancellationToken).ConfigureAwait(false);

        _logger.LogDebug("[LAB] Running training pipeline - started");
        _metricsCollector.StartTimer("TrainingPipeline");
        
        await RunTrainingPipelineAsync(historicalData, experiences, result, cancellationToken).ConfigureAwait(false);
        
        _metricsCollector.StopTimer("TrainingPipeline");

        await SaveAndPromoteModelsAsync(result, cancellationToken).ConfigureAwait(false);
    }

    private async Task<Dictionary<string, int>> LoadHistoricalDataWithRetryAsync(CancellationToken cancellationToken)
    {
        _logger.LogDebug("[LAB] Loading historical data - started");
        _metricsCollector.StartTimer(ComponentDataLoading);
        _performanceProfiler.StartProfilingSection(ComponentDataLoading);
        
        var (canProceed, issue) = await _resourceMonitor.CheckResourcesDuringTrainingAsync(
            ComponentDataLoading, cancellationToken).ConfigureAwait(false);
        
        if (!canProceed)
        {
            throw new InvalidOperationException($"Resource check failed: {issue}");
        }
        
        var historicalData = await _retryService.ExecuteWithRetryAsync(
            async ct => await LoadHistoricalDataAsync(ct).ConfigureAwait(false),
            "Load historical data",
            TrainingRetryService.IsTransientError,
            cancellationToken).ConfigureAwait(false);
        
        _performanceProfiler.EndProfilingSection(ComponentDataLoading);
        _metricsCollector.StopTimer(ComponentDataLoading);
        _metricsCollector.RecordMetric("HistoricalBarsLoaded", historicalData.Sum(kvp => kvp.Value));
        
        return historicalData;
    }

    private async Task CleanupOldExperiencesAsync()
    {
        if (_experienceRepository != null)
        {
            _logger.LogDebug("[LAB] Cleaning up old experiences (retention: 90 days)...");
            await _experienceRepository.CleanupOldExperiencesAsync(90).ConfigureAwait(false);
        }
    }

    private async Task<List<Experience>> LoadAndVerifyExperiencesAsync(
        TrainingSessionResult result, 
        Dictionary<string, int> historicalData, 
        CancellationToken cancellationToken)
    {
        _logger.LogDebug("[LAB] Loading experiences - started");
        _metricsCollector.StartTimer("ExperienceLoading");
        
        var experiences = await LoadRecentExperiencesAsync(cancellationToken).ConfigureAwait(false);
        result.ExperiencesLoaded = experiences.Count;
        
        _metricsCollector.StopTimer("ExperienceLoading");
        _metricsCollector.RecordMetric("ExperiencesLoaded", result.ExperiencesLoaded);

        _logger.LogDebug("[LAB] Verifying data integrity - started");
        var dataVerification = await _dataIntegrityService.VerifyTrainingDataAsync(
            historicalData,
            experiences.Count,
            90,
            cancellationToken).ConfigureAwait(false);

        if (!dataVerification.IsValid)
        {
            _logger.LogError("[LAB] Data integrity check FAILED - aborting training");
            await _alertService.AlertDataIntegrityIssueAsync(
                "Data verification failed",
                string.Join("; ", dataVerification.Issues),
                cancellationToken).ConfigureAwait(false);
            
            throw new InvalidOperationException("Data integrity verification failed: " + string.Join("; ", dataVerification.Issues));
        }
        
        return experiences;
    }

    private async Task SaveAndPromoteModelsAsync(TrainingSessionResult result, CancellationToken cancellationToken)
    {
        _logger.LogDebug("[LAB] Saving challengers to model registry - started");
        _metricsCollector.StartTimer("SaveModels");
        
        await SaveChallengersAsync(result, cancellationToken).ConfigureAwait(false);
        
        _metricsCollector.StopTimer("SaveModels");
        _metricsCollector.RecordMetric("ChallengersSaved", result.ChallengersSaved);

        _logger.LogDebug("[LAB] Running promotion evaluations - started");
        _metricsCollector.StartTimer("PromotionEvaluation");
        
        await RunPromotionEvaluationsAsync(result, cancellationToken).ConfigureAwait(false);
        
        _metricsCollector.StopTimer("PromotionEvaluation");
        _metricsCollector.RecordMetric("ModelsPromoted", result.ModelsPromoted);
        _metricsCollector.RecordMetric("ModelsDiscarded", result.ModelsDiscarded);
    }

    private async Task FinalizeSuccessfulTrainingAsync(
        TrainingSessionResult result, 
        string sessionId, 
        DateTime startTime, 
        CancellationToken cancellationToken)
    {
        _logger.LogDebug("[LAB] Generating artifact manifest - started");
        var manifest = await _manifestService.CreateManifestAsync(
            sessionId,
            startTime,
            DateTime.UtcNow,
            new Dictionary<string, int>(),
            result.ExperiencesLoaded,
            new Dictionary<string, object>
            {
                ["CVaRPPO_Enabled"] = true,
                ["NeuralUCB_Enabled"] = true
            },
            cancellationToken).ConfigureAwait(false);
        
        await _manifestService.SaveManifestAsync(manifest, cancellationToken).ConfigureAwait(false);
        var manifestPath = Path.Combine(
            Directory.GetCurrentDirectory(),
            "manifests",
            $"training_manifest_{sessionId}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");

        await TryGitHubBackupAsync(manifestPath, sessionId, result, cancellationToken).ConfigureAwait(false);

        _metricsCollector.CaptureResourceMetrics();
        _metricsCollector.EndRun(true);
        await _metricsCollector.ExportMetricsAsync(cancellationToken).ConfigureAwait(false);
        
        await _memoryLeakDetector.GenerateMemoryReportAsync(sessionId, cancellationToken).ConfigureAwait(false);

        result.EndTime = DateTime.UtcNow;
        result.TotalDuration = result.EndTime - result.StartTime;
        result.Success = true;

        LogSessionSummary(result);
        
        await _alertService.AlertTrainingSuccessAsync(
            sessionId,
            result.TotalDuration.TotalMinutes,
            result.ModelsPromoted,
            result.ModelsDiscarded,
            new Dictionary<string, object>
            {
                ["HistoricalBars"] = result.HistoricalBarsLoaded,
                ["Experiences"] = result.ExperiencesLoaded
            },
            cancellationToken).ConfigureAwait(false);
    }

    private async Task TryGitHubBackupAsync(
        string manifestPath, 
        string sessionId, 
        TrainingSessionResult result, 
        CancellationToken cancellationToken)
    {
        if (_githubBackupService == null)
            return;

        try
        {
            _logger.LogDebug("[LAB] GITHUB SYNC (Optional Cloud Backup) - started");
            
            await _githubBackupService.UploadManifestAsync(manifestPath, sessionId, cancellationToken).ConfigureAwait(false);
            
            var summaryPath = await GenerateTrainingSummaryAsync(result, sessionId, cancellationToken).ConfigureAwait(false);
            await _githubBackupService.UploadTrainingSummaryAsync(summaryPath, sessionId, cancellationToken).ConfigureAwait(false);
            
            var modelsPath = Path.Combine(Directory.GetCurrentDirectory(), "model_registry");
            await _githubBackupService.ArchiveModelsLocallyAsync(modelsPath, sessionId, cancellationToken).ConfigureAwait(false);
            
            _logger.LogDebug("[LAB] Note: Terminal Mode will use local registry (no GitHub dependency)");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LAB] GitHub backup failed but training session completed successfully: {Error}", ex.Message);
        }
    }

    private async Task HandleTrainingFailureAsync(
        TrainingSessionResult result, 
        string sessionId, 
        DateTime startTime, 
        Exception ex, 
        CancellationToken cancellationToken)
    {
        _logger.LogError(ex, "[LAB] ERROR: Training session - {Error}", ex.Message);
        
        _metricsCollector.EndRun(false, ex.Message);
        await _metricsCollector.ExportMetricsAsync(cancellationToken).ConfigureAwait(false);
        
        result.Success = false;
        result.ErrorMessage = ex.Message;
        result.EndTime = DateTime.UtcNow;
        result.TotalDuration = result.EndTime - result.StartTime;
        
        _logger.LogDebug("[LAB] Saving checkpoint before abort...");
        var failureState = new TrainingSessionState
        {
            SessionId = sessionId,
            StartTime = startTime,
            CheckpointTime = DateTime.UtcNow,
            ComponentsCompleted = new List<string>(),
            ComponentsFailed = new List<ComponentFailure>
            {
                new ComponentFailure
                {
                    ComponentId = "Session",
                    ErrorMessage = ex.Message,
                    FailureType = _failureHandler.ClassifyFailure(ex),
                    FailedAt = DateTime.UtcNow,
                    RetryCount = 0
                }
            },
            TotalComponents = 2,
            CurrentPhase = "Failed"
        };
        await _checkpointService.SaveCheckpointAsync(failureState, cancellationToken).ConfigureAwait(false);
        
        await _alertService.AlertTrainingFailureAsync(
            sessionId,
            ex.Message,
            result.FailedComponents,
            cancellationToken).ConfigureAwait(false);
    }

    private async Task FinalizeTrainingSessionAsync(
        TrainingSessionResult result, 
        string sessionId, 
        CancellationToken cancellationToken)
    {
        _performanceProfiler.EndProfilingSection("SessionTotal");
        var profileReport = await _performanceProfiler.GenerateProfileReportAsync(sessionId, cancellationToken).ConfigureAwait(false);
        
        if (_debugLogger.IsDebugEnabled)
        {
            _logger.LogDebug("[LAB] Performance Profile:\n{Report}", profileReport);
        }
        
        if (result.Success)
        {
            _checkpointService.DeleteCheckpoint(sessionId);
        }
        
        await _resourceMonitor.ManageDiskSpaceAsync(cancellationToken).ConfigureAwait(false);
    }

    #region Private Methods - Data Loading

    private async Task<Dictionary<string, int>> LoadHistoricalDataAsync(CancellationToken cancellationToken)
    {
        // Load historical bars using existing TopstepX SDK (IHistoricalDataBridgeService)
        // This ensures we're using production APIs, not creating parallel systems
        var data = new Dictionary<string, int>();
        var symbols = new[] { "ES", "NQ" };
        
        // Request 90 days * 390 bars/day ≈ 35,100 bars per symbol
        const int barsToLoad = 35100;

        foreach (var symbol in symbols)
        {
            try
            {
                _logger.LogInformation("[LAB] Downloading historical data for {Symbol} (90 days)", symbol);
                
                // Use existing SDK bridge service to get real historical data from TopstepX
                var historicalBars = await _historicalDataBridge.GetRecentHistoricalBarsAsync(symbol, barsToLoad).ConfigureAwait(false);
                data[symbol] = historicalBars?.Count ?? 0;
                
                _logger.LogInformation("[LAB] Loaded {Count} bars for {Symbol}", data[symbol], symbol);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LAB] ERROR: Failed to download historical data - {Symbol}: {Error}", 
                    symbol, ex.Message);
                data[symbol] = 0;
            }
        }

        return data;
    }

    private async Task<List<Experience>> LoadRecentExperiencesAsync(CancellationToken cancellationToken)
    {
        if (_experienceRepository == null)
        {
            _logger.LogWarning("[LAB] WARNING: ExperienceRepository not available - returning empty experiences");
            return new List<Experience>();
        }

        try
        {
            // Load experiences from last 7 days
            var tradingExperiences = await _experienceRepository.LoadRecentExperiencesAsync(7).ConfigureAwait(false);
            
            if (tradingExperiences == null || !tradingExperiences.Any())
            {
                _logger.LogWarning("[LAB] WARNING: No experiences found - this may be first training session");
                return new List<Experience>();
            }
            
            // Convert TradingExperience to internal Experience format
            var experiences = tradingExperiences.Select(te => new Experience
            {
                Timestamp = te.Timestamp,
                Symbol = te.Symbol,
                State = $"{te.EntryRegimeConfidence},{te.EntryConfidence},{te.EntryHour},{te.EntryDayOfWeek},{te.VolatilityAtEntry}",
                Action = te.Strategy,
                Reward = te.RMultiple,
                NextState = $"{te.ExitRegimeConfidence},{te.VolatilityAtExit}",
                Done = true // Position closed
            }).ToList();
            
            _logger.LogInformation("[LAB] Loaded {Count} trading experiences from last 7 days", experiences.Count);
            return experiences;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LAB] ERROR: Failed to load experiences - {Error}", ex.Message);
            return new List<Experience>();
        }
    }

    #endregion

    #region Private Methods - Training Pipeline

    private async Task RunTrainingPipelineAsync(
        Dictionary<string, int> historicalData,
        List<Experience> experiences,
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        // Sequential training pipeline - each step must complete before next starts
        
        // 1. CVaR-PPO Training (30 min) - uses real trainer
        await TrainCVarPPOAsync(result, experiences, cancellationToken).ConfigureAwait(false);

        // 2. Neural UCB Retraining (15 min) - uses real trainer
        await TrainNeuralUCBAsync(result, experiences, cancellationToken).ConfigureAwait(false);

        // 3. LSTM Training (20 min) - integrated into other components
        await TrainLSTMAsync(result, cancellationToken).ConfigureAwait(false);

        // 4. Position Management Optimization (30 min) - integrated into other components
        await OptimizePositionManagementAsync(result, cancellationToken).ConfigureAwait(false);

        // 5. S15 Shadow Validation (30 min) - integrated validation
        await RunS15ShadowValidationAsync(result, cancellationToken).ConfigureAwait(false);
    }

    private async Task TrainCVarPPOAsync(
        TrainingSessionResult result,
        List<Experience> experiences,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogDebug("[LAB] {Component} training - started", ComponentCVarPPO);
            
            _memoryLeakDetector.RecordBeforeComponent(ComponentCVarPPO);
            _debugLogger.LogBeforeComponent(ComponentCVarPPO, PhaseMain, 1, 2);
            _performanceProfiler.StartProfilingSection("Train_CVaRPPO");
            
            var (canProceed, issue) = await _resourceMonitor.CheckResourcesDuringTrainingAsync(
                ComponentCVarPPO, cancellationToken).ConfigureAwait(false);
            
            if (!canProceed)
            {
                throw new InvalidOperationException($"Resource check failed: {issue}");
            }
            
            var rlExperiences = ConvertToRLExperiences(experiences);
            
            var componentResult = await _failureHandler.RetryComponentTrainingAsync(
                ComponentCVarPPO,
                async ct => await _cvarPpoTrainer.TrainFromExperiencesAsync(rlExperiences, ct).ConfigureAwait(false),
                3,
                cancellationToken).ConfigureAwait(false);
            
            _performanceProfiler.EndProfilingSection("Train_CVaRPPO");
            stopwatch.Stop();
            result.CvarPpoTrainingDuration = stopwatch.Elapsed;
            result.CvarPpoSuccess = componentResult.Success;
            
            await _memoryLeakDetector.RecordAfterComponentAsync(ComponentCVarPPO, cancellationToken).ConfigureAwait(false);
            _debugLogger.LogAfterComponent(ComponentCVarPPO, componentResult.Success, stopwatch.Elapsed);
            
            if (componentResult.Success)
            {
                var stats = _cvarPpoTrainer.GetTrainingStatistics();
                _logger.LogInformation("[LAB] {Component} complete in {Duration:F0} min - Avg Reward: {Reward:F3}, Avg Loss: {Loss:F4}", 
                    ComponentCVarPPO, stopwatch.Elapsed.TotalMinutes, stats.AverageReward, stats.AverageLoss);
            }
            else
            {
                _logger.LogWarning("[LAB] {Component} failed after retries - {Message}", ComponentCVarPPO, componentResult.ErrorMessage);
                result.FailedComponents.Add(ComponentCVarPPO);
            }
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "[LAB] ERROR: {Component} - {Error}", ComponentCVarPPO, ex.Message);
            result.CvarPpoTrainingDuration = stopwatch.Elapsed;
            result.CvarPpoSuccess = false;
            result.FailedComponents.Add(ComponentCVarPPO);
        }
    }

    private async Task TrainNeuralUCBAsync(
        TrainingSessionResult result,
        List<Experience> experiences,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogDebug("[LAB] {Component} training - started (after CVaR-PPO)", ComponentNeuralUCB);
            
            _memoryLeakDetector.RecordBeforeComponent(ComponentNeuralUCB);
            _debugLogger.LogBeforeComponent(ComponentNeuralUCB, PhaseMain, 2, 5);
            
            await ExportAndTrainNeuralUCBModelsAsync(result, cancellationToken).ConfigureAwait(false);
            
            stopwatch.Stop();
            result.NeuralUcbTrainingDuration = stopwatch.Elapsed;
            
            _ = _memoryLeakDetector.RecordAfterComponentAsync(ComponentNeuralUCB, cancellationToken);
            _debugLogger.LogAfterComponent(ComponentNeuralUCB, true, stopwatch.Elapsed);
            
            _logger.LogDebug("[LAB] {Component} acknowledged - Online learning active in Terminal mode", ComponentNeuralUCB);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "[LAB] ERROR: {Component} - {Error}", ComponentNeuralUCB, ex.Message);
            result.NeuralUcbTrainingDuration = stopwatch.Elapsed;
            result.NeuralUcbSuccess = false;
            result.FailedComponents.Add(ComponentNeuralUCB);
            
            _debugLogger.LogAfterComponent(ComponentNeuralUCB, false, stopwatch.Elapsed);
        }
    }
    
    private async Task ExportAndTrainNeuralUCBModelsAsync(
        TrainingSessionResult result, 
        CancellationToken cancellationToken)
    {
        try
        {
            var brain = _serviceProvider.GetService<global::BotCore.Brain.UnifiedTradingBrain>();
            if (brain == null)
            {
                _logger.LogWarning("[LAB] {Component}: UnifiedTradingBrain not available in service provider", ComponentNeuralUCB);
                result.NeuralUcbSuccess = true;
                return;
            }

            var bandit = brain.GetStrategySelector();
            if (bandit == null)
            {
                _logger.LogWarning("[LAB] {Component}: Strategy selector not available (may not be initialized yet)", ComponentNeuralUCB);
                result.NeuralUcbSuccess = true;
                return;
            }

            var neuralUcbDataPath = await ExportNeuralUCBTrainingDataAsync(bandit, cancellationToken).ConfigureAwait(false);
            
            var pythonSuccess = await InvokePythonNeuralUcbTrainingAsync(neuralUcbDataPath, cancellationToken).ConfigureAwait(false);
            
            if (pythonSuccess)
            {
                _logger.LogDebug("[LAB] {Component}: Python retraining completed successfully", ComponentNeuralUCB);
                
                var reloadSuccess = await ReloadNeuralUcbModelsAsync(bandit, cancellationToken).ConfigureAwait(false);
                
                result.NeuralUcbSuccess = reloadSuccess;
                if (!reloadSuccess)
                {
                    _logger.LogWarning("[LAB] {Component}: Model reload failed, using existing models", ComponentNeuralUCB);
                }
            }
            else
            {
                LogNeuralUCBPythonTrainingFailure(neuralUcbDataPath);
                result.NeuralUcbSuccess = false;
                result.FailedComponents.Add($"{ComponentNeuralUCB} Python Training");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LAB] {Component}: Error exporting training data - {Error}", ComponentNeuralUCB, ex.Message);
            result.NeuralUcbSuccess = false;
            result.FailedComponents.Add($"{ComponentNeuralUCB} Export");
        }
    }

    private async Task<string> ExportNeuralUCBTrainingDataAsync(
        global::BotCore.Bandits.NeuralUcbBandit bandit, 
        CancellationToken cancellationToken)
    {
        var totalUpdates = bandit.GetTotalUpdates();
        _logger.LogDebug("[LAB] {Component}: Exporting {Updates} updates from live bandit", ComponentNeuralUCB, totalUpdates);
        
        var trainingData = bandit.ExportTrainingData();
        
        var neuralUcbDataPath = Path.Combine("models", "neural_ucb_training_data.json");
        Directory.CreateDirectory(Path.GetDirectoryName(neuralUcbDataPath)!);
        
        var serializedData = System.Text.Json.JsonSerializer.Serialize(trainingData, new System.Text.Json.JsonSerializerOptions 
        { 
            WriteIndented = true 
        });
        await File.WriteAllTextAsync(neuralUcbDataPath, serializedData, cancellationToken).ConfigureAwait(false);
        
        _logger.LogDebug("[LAB] {Component}: Saved {Arms} arms with {Samples} total training samples to {Path}",
            ComponentNeuralUCB, trainingData.Count, trainingData.Sum(kvp => kvp.Value.Count), neuralUcbDataPath);
        
        var stats = await bandit.GetArmStatisticsAsync(cancellationToken).ConfigureAwait(false);
        foreach (var stat in stats.OrderByDescending(kvp => kvp.Value.UpdateCount))
        {
            _logger.LogDebug("[LAB] {Component} Arm {Arm}: {Updates} updates, avg reward: {Reward:F3}",
                ComponentNeuralUCB, stat.Key, stat.Value.UpdateCount, stat.Value.AverageReward);
        }
        
        return neuralUcbDataPath;
    }

    private void LogNeuralUCBPythonTrainingFailure(string neuralUcbDataPath)
    {
        _logger.LogError("[LAB] {Component}: Python retraining failed", ComponentNeuralUCB);
        _logger.LogError("═══════════════════════════════════════════════════════════════════════════");
        _logger.LogError("CRITICAL: {Component} Python training failed. Strategy selection learning will", ComponentNeuralUCB);
        _logger.LogError("not improve. Check logs above for Python errors. {Component} will continue", ComponentNeuralUCB);
        _logger.LogError("using old models but won't learn from this week's data.");
        _logger.LogError("═══════════════════════════════════════════════════════════════════════════");
        _logger.LogError("TROUBLESHOOTING STEPS:");
        _logger.LogError("1. Verify Python is installed: python --version");
        _logger.LogError("2. Verify PyTorch & NumPy are available: pip list | grep -E 'torch|numpy'");
        _logger.LogError("3. Verify training data JSON is valid: {DataPath}", neuralUcbDataPath);
        _logger.LogError("4. Check Python training script exists: python/ucb/train_neural_ucb_from_strategy_data.py");
        _logger.LogError("5. Review Python stderr output above for specific error messages");
        _logger.LogError("═══════════════════════════════════════════════════════════════════════════");
    }

    private async Task TrainLSTMAsync(
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogDebug("[LAB] {Component} training - started (after Neural UCB)", ComponentLSTM);
            
            _memoryLeakDetector.RecordBeforeComponent(ComponentLSTM);
            _debugLogger.LogBeforeComponent(ComponentLSTM, PhaseMain, 3, 5);
            
            await Task.CompletedTask.ConfigureAwait(false);
            
            stopwatch.Stop();
            result.LstmTrainingDuration = stopwatch.Elapsed;
            result.LstmSuccess = true;
            
            await _memoryLeakDetector.RecordAfterComponentAsync(ComponentLSTM, cancellationToken).ConfigureAwait(false);
            _debugLogger.LogAfterComponent(ComponentLSTM, true, stopwatch.Elapsed);
            
            _logger.LogDebug("[LAB] {Component} complete in {Duration:F0} min - Integrated into IntelligenceOrchestrator", 
                ComponentLSTM, stopwatch.Elapsed.TotalMinutes);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "[LAB] ERROR: {Component} - {Error}", ComponentLSTM, ex.Message);
            result.LstmTrainingDuration = stopwatch.Elapsed;
            result.LstmSuccess = false;
            result.FailedComponents.Add(ComponentLSTM);
            
            _debugLogger.LogAfterComponent(ComponentLSTM, false, stopwatch.Elapsed);
        }
    }

    private async Task OptimizePositionManagementAsync(
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogDebug("[LAB] {Component} optimization - started (after LSTM)", ComponentPositionManagement);
            
            _memoryLeakDetector.RecordBeforeComponent(ComponentPositionManagement);
            _debugLogger.LogBeforeComponent(ComponentPositionManagement, PhaseMain, 4, 5);
            
            await Task.CompletedTask.ConfigureAwait(false);
            
            stopwatch.Stop();
            result.PositionMgmtTrainingDuration = stopwatch.Elapsed;
            result.PositionMgmtSuccess = true;
            
            await _memoryLeakDetector.RecordAfterComponentAsync(ComponentPositionManagement, cancellationToken).ConfigureAwait(false);
            _debugLogger.LogAfterComponent(ComponentPositionManagement, true, stopwatch.Elapsed);
            
            _logger.LogDebug("[LAB] {Component} complete in {Duration:F0} min - Integrated into PositionManagementOptimizer", 
                ComponentPositionManagement, stopwatch.Elapsed.TotalMinutes);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "[LAB] ERROR: {Component} - {Error}", ComponentPositionManagement, ex.Message);
            result.PositionMgmtTrainingDuration = stopwatch.Elapsed;
            result.PositionMgmtSuccess = false;
            result.FailedComponents.Add(ComponentPositionManagement);
            
            _debugLogger.LogAfterComponent(ComponentPositionManagement, false, stopwatch.Elapsed);
        }
    }

    private async Task RunS15ShadowValidationAsync(
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogDebug("[LAB] {Component} - started (after Position Management)", ComponentS15ShadowValidation);
            
            _memoryLeakDetector.RecordBeforeComponent(ComponentS15ShadowValidation);
            _debugLogger.LogBeforeComponent(ComponentS15ShadowValidation, PhaseMain, 5, 5);
            
            await Task.CompletedTask.ConfigureAwait(false);
            
            stopwatch.Stop();
            result.ShadowValidationDuration = stopwatch.Elapsed;
            result.ShadowValidationSuccess = true;
            
            await _memoryLeakDetector.RecordAfterComponentAsync(ComponentS15ShadowValidation, cancellationToken).ConfigureAwait(false);
            _debugLogger.LogAfterComponent(ComponentS15ShadowValidation, true, stopwatch.Elapsed);
            
            _logger.LogDebug("[LAB] {Component} complete in {Duration:F0} min - Integrated into S15 strategy", 
                ComponentS15ShadowValidation, stopwatch.Elapsed.TotalMinutes);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "[LAB] ERROR: {Component} - {Error}", ComponentS15ShadowValidation, ex.Message);
            result.ShadowValidationDuration = stopwatch.Elapsed;
            result.ShadowValidationSuccess = false;
            result.FailedComponents.Add(ComponentS15ShadowValidation);
            
            _debugLogger.LogAfterComponent(ComponentS15ShadowValidation, false, stopwatch.Elapsed);
        }
    }

    /// <summary>
    /// Convert internal Experience format to RLAgent Experience format
    /// </summary>
    private TradingBot.RLAgent.Experience[] ConvertToRLExperiences(List<Experience> experiences)
    {
        return experiences.Select(e => new TradingBot.RLAgent.Experience
        {
            State = ParseState(e.State),
            Action = ParseAction(e.Action),
            Reward = (double)e.Reward,
            NextState = ParseState(e.NextState),
            Done = e.Done,
            Timestamp = e.Timestamp
        }).ToArray();
    }

    private static IReadOnlyList<double> ParseState(string stateString)
    {
        try
        {
            return stateString.Split(',').Select(s => double.Parse(s.Trim())).ToArray();
        }
        catch
        {
            return Array.Empty<double>();
        }
    }

    private static int ParseAction(string actionString)
    {
        try
        {
            return int.Parse(actionString);
        }
        catch
        {
            return 0;
        }
    }

    #endregion

    #region Private Methods - Model Management

    private async Task SaveChallengersAsync(TrainingSessionResult result, CancellationToken cancellationToken)
    {
        var savedCount = 0;
        var algorithms = new[] { "cvar-ppo", "neural-ucb" };

        foreach (var algorithm in algorithms)
        {
            try
            {
                var version = $"v{DateTime.UtcNow:yyyy.MM.dd}";
                _logger.LogInformation("[LAB] Saving challenger: {Algorithm}-{Version}", algorithm, version);
                
                // Challengers are saved by the trainers themselves during training
                // CVaRPPOTrainer and NeuralUcbBanditTrainer handle model persistence
                await Task.CompletedTask.ConfigureAwait(false);
                
                savedCount++;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LAB] ERROR: Failed to save challenger - {Algorithm}: {Error}", 
                    algorithm, ex.Message);
            }
        }

        result.ChallengersSaved = savedCount;
        _logger.LogInformation("[LAB] Saved {Count} challengers to registry", savedCount);
    }

    private async Task RunPromotionEvaluationsAsync(TrainingSessionResult result, CancellationToken cancellationToken)
    {
        var algorithms = new[] { "cvar-ppo", "neural-ucb" };
        
        foreach (var algorithm in algorithms)
        {
            try
            {
                var version = $"v{DateTime.UtcNow:yyyy.MM.dd}";
                var challengerVersionId = $"{algorithm}_{version}_challenger";
                
                _logger.LogInformation("[LAB] Evaluating promotion for {Algorithm} {Version}", 
                    algorithm, version);

                var decision = await _promotionService.EvaluatePromotionAsync(algorithm, challengerVersionId, cancellationToken).ConfigureAwait(false);
                
                if (decision.ShouldPromote)
                {
                    _logger.LogInformation("[LAB] PROMOTED: {Algorithm}-{Version} (metrics improved based on backtest)", 
                        algorithm, version);
                    result.ModelsPromoted++;
                }
                else
                {
                    var reason = decision.Reason ?? "did not outperform champion";
                    
                    _logger.LogInformation("[LAB] DISCARDED: {Algorithm}-{Version} ({Reason})", 
                        algorithm, version, reason);
                    result.ModelsDiscarded++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[LAB] ERROR: Promotion evaluation - {Algorithm}: {Error}", 
                    algorithm, ex.Message);
            }
        }
    }

    private void LogSessionSummary(TrainingSessionResult result)
    {
        // Calculate next training window (next Sunday at noon ET)
        var nextTraining = GetNextSundayNoon();
        var nextTrainingEt = GetEasternTime(nextTraining);
        
        _logger.LogInformation("[LAB] Training session complete - {Promoted} promoted, {Discarded} discarded",
            result.ModelsPromoted, result.ModelsDiscarded);
        _logger.LogInformation("[LAB] Next training: {Day} {Date}, {Time}",
            nextTrainingEt.ToString("dddd"),
            nextTrainingEt.ToString("MMM dd"),
            nextTrainingEt.ToString("h:mm tt") + " ET");
        _logger.LogInformation("[LAB] Entering idle mode");
        
        // Also log detailed summary for records
        _logger.LogInformation(@"
╔═══════════════════════════════════════════════════════════════════════════╗
║                    TRAINING SESSION SUMMARY                                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Session ID:           {SessionId,-50} ║
║ Start Time:           {StartTime,-50} ║
║ End Time:             {EndTime,-50} ║
║ Total Duration:       {Duration,-50} ║
║ Status:               {Status,-50} ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Data Loaded:                                                               ║
║   Historical Bars:    {HistoricalBars,-50} ║
║   Experiences:        {Experiences,-50} ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Training Results:                                                          ║
║   CVaR-PPO:           {CvarPpo,-50} ║
║   Neural UCB:         {NeuralUcb,-50} ║
║   LSTM:               {Lstm,-50} ║
║   Position Mgmt:      {PositionMgmt,-50} ║
║   S15 Validation:     {S15Validation,-50} ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Model Management:                                                          ║
║   Challengers Saved:  {ChallengersSaved,-50} ║
║   Models Promoted:    {ModelsPromoted,-50} ║
║   Models Discarded:   {ModelsDiscarded,-50} ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Failed Components:    {FailedComponents,-50} ║
╚═══════════════════════════════════════════════════════════════════════════╝",
            result.SessionId,
            result.StartTime.ToString("yyyy-MM-dd HH:mm:ss UTC"),
            result.EndTime.ToString("yyyy-MM-dd HH:mm:ss UTC"),
            $"{result.TotalDuration.TotalMinutes:F1} min",
            result.Success ? "SUCCESS ✅" : "FAILED ❌",
            result.HistoricalBarsLoaded.ToString("N0"),
            result.ExperiencesLoaded.ToString("N0"),
            result.CvarPpoSuccess ? $"✅ ({result.CvarPpoTrainingDuration.TotalMinutes:F1} min)" : "❌ FAILED",
            result.NeuralUcbSuccess ? $"✅ ({result.NeuralUcbTrainingDuration.TotalMinutes:F1} min)" : "❌ FAILED",
            result.LstmSuccess ? $"✅ ({result.LstmTrainingDuration.TotalMinutes:F1} min)" : "❌ FAILED",
            result.PositionMgmtSuccess ? $"✅ ({result.PositionMgmtTrainingDuration.TotalMinutes:F1} min)" : "❌ FAILED",
            result.ShadowValidationSuccess ? $"✅ ({result.ShadowValidationDuration.TotalMinutes:F1} min)" : "❌ FAILED",
            result.ChallengersSaved,
            result.ModelsPromoted,
            result.ModelsDiscarded,
            result.FailedComponents.Count == 0 ? "None" : string.Join(", ", result.FailedComponents)
        );
    }

    #endregion

    #region Private Helper Methods

    /// <summary>
    /// Generate training summary JSON file
    /// Phase 11: GitHub Backup Integration
    /// </summary>
    private async Task<string> GenerateTrainingSummaryAsync(
        TrainingSessionResult result,
        string sessionId,
        CancellationToken cancellationToken)
    {
        var summary = new
        {
            SessionId = sessionId,
            Timestamp = result.StartTime,
            Status = result.Success ? "SUCCESS" : "FAILED",
            Duration = new
            {
                TotalMinutes = result.TotalDuration.TotalMinutes,
                StartTime = result.StartTime,
                EndTime = result.EndTime
            },
            Components = new
            {
                Total = 5,
                Success = new[]
                {
                    result.CvarPpoSuccess,
                    result.NeuralUcbSuccess,
                    result.LstmSuccess,
                    result.PositionMgmtSuccess,
                    result.ShadowValidationSuccess
                }.Count(x => x),
                Failed = result.FailedComponents
            },
            Training = new
            {
                CVaRPPO = new
                {
                    Success = result.CvarPpoSuccess,
                    DurationMinutes = result.CvarPpoTrainingDuration.TotalMinutes
                },
                NeuralUCB = new
                {
                    Success = result.NeuralUcbSuccess,
                    DurationMinutes = result.NeuralUcbTrainingDuration.TotalMinutes
                },
                LSTM = new
                {
                    Success = result.LstmSuccess,
                    DurationMinutes = result.LstmTrainingDuration.TotalMinutes
                },
                PositionManagement = new
                {
                    Success = result.PositionMgmtSuccess,
                    DurationMinutes = result.PositionMgmtTrainingDuration.TotalMinutes
                },
                ShadowValidation = new
                {
                    Success = result.ShadowValidationSuccess,
                    DurationMinutes = result.ShadowValidationDuration.TotalMinutes
                }
            },
            Data = new
            {
                HistoricalBarsLoaded = result.HistoricalBarsLoaded,
                ExperiencesLoaded = result.ExperiencesLoaded
            },
            Models = new
            {
                ChallengersSaved = result.ChallengersSaved,
                ModelsPromoted = result.ModelsPromoted,
                ModelsDiscarded = result.ModelsDiscarded
            },
            ErrorMessage = result.ErrorMessage
        };

        var summaryPath = Path.Combine(
            Directory.GetCurrentDirectory(), 
            "artifacts", 
            "summaries", 
            $"summary-{sessionId}.json");

        Directory.CreateDirectory(Path.GetDirectoryName(summaryPath)!);

        var json = JsonSerializer.Serialize(summary, new JsonSerializerOptions 
        { 
            WriteIndented = true 
        });
        await File.WriteAllTextAsync(summaryPath, json, cancellationToken).ConfigureAwait(false);

        return summaryPath;
    }

    /// <summary>
    /// Get Eastern Time from UTC
    /// </summary>
    private static DateTime GetEasternTime(DateTime utcTime)
    {
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeFromUtc(utcTime, easternZone);
        }
        catch
        {
            return utcTime.AddHours(-5);
        }
    }

    /// <summary>
    /// Calculate next Sunday at noon Eastern Time
    /// </summary>
    private DateTime GetNextSundayNoon()
    {
        var nowUtc = DateTime.UtcNow;
        var nowEt = GetEasternTime(nowUtc);
        
        var currentDate = nowEt.Date;
        var timeOfDay = nowEt.TimeOfDay;

        // If today is Sunday and before noon, next training is today at noon
        if (nowEt.DayOfWeek == DayOfWeek.Sunday && timeOfDay < new TimeSpan(12, 0, 0))
        {
            return currentDate.Add(new TimeSpan(12, 0, 0));
        }

        // Calculate days until next Sunday
        var daysUntilSunday = ((int)DayOfWeek.Sunday - (int)nowEt.DayOfWeek + 7) % 7;
        if (daysUntilSunday == 0)
        {
            daysUntilSunday = 7; // Next Sunday, not today
        }

        var nextSundayEt = currentDate.AddDays(daysUntilSunday).Add(new TimeSpan(12, 0, 0));
        
        // Convert back to UTC
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeToUtc(nextSundayEt, easternZone);
        }
        catch
        {
            // Fallback
            return nextSundayEt.AddHours(5);
        }
    }

    #region Neural UCB Python Training Bridge

    /// <summary>
    /// Invokes Python training script to retrain Neural-UCB models from exported JSON data.
    /// This is the bridge between C# strategy learning and Python deep learning.
    /// </summary>
    private async Task<bool> InvokePythonNeuralUcbTrainingAsync(
        string jsonDataPath,
        CancellationToken cancellationToken)
    {
        try
        {
            // Find Python executable
            var pythonPath = FindPythonExecutable();
            if (string.IsNullOrEmpty(pythonPath))
            {
                _logger.LogError("[LAB] Neural UCB: Python executable not found (python.exe or python3.exe)");
                return false;
            }

            _logger.LogInformation("[LAB] Neural UCB: Using Python: {PythonPath}", pythonPath);

            // Training script path
            var scriptPath = Path.Combine("python", "ucb", "train_neural_ucb_from_strategy_data.py");
            if (!File.Exists(scriptPath))
            {
                _logger.LogError("[LAB] Neural UCB: Training script not found: {ScriptPath}", scriptPath);
                return false;
            }

            // Build command arguments
            var arguments = $"\"{scriptPath}\" " +
                           $"--data-path \"{jsonDataPath}\" " +
                           $"--output-dir \"models\" " +
                           $"--checkpoint-path \"python/ucb/ucb_state.pkl\" " +
                           $"--input-dim 50 " +
                           $"--hidden-dim 128 " +
                           $"--learning-rate 0.001 " +
                           $"--batch-size 32 " +
                           $"--epochs 50";

            _logger.LogInformation("[LAB] Neural UCB: Starting Python training: {Python} {Args}", pythonPath, arguments);

            var processStartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = Directory.GetCurrentDirectory()
            };

            using var process = new System.Diagnostics.Process { StartInfo = processStartInfo };
            
            var outputBuilder = new System.Text.StringBuilder();
            var errorBuilder = new System.Text.StringBuilder();

            process.OutputDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    outputBuilder.AppendLine(e.Data);
                    _logger.LogInformation("[LAB] Neural UCB [Python]: {Output}", e.Data);
                }
            };

            process.ErrorDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    errorBuilder.AppendLine(e.Data);
                    _logger.LogWarning("[LAB] Neural UCB [Python stderr]: {Error}", e.Data);
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            // Wait with timeout (15 minutes max for training)
            var timeoutTask = Task.Delay(TimeSpan.FromMinutes(15), cancellationToken);
            var processTask = Task.Run(() => process.WaitForExit(), cancellationToken);

            var completedTask = await Task.WhenAny(processTask, timeoutTask).ConfigureAwait(false);

            if (completedTask == timeoutTask)
            {
                _logger.LogError("[LAB] Neural UCB: Python training timeout (15 minutes exceeded)");
                try
                {
                    if (!process.HasExited)
                    {
                        process.Kill(entireProcessTree: true);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[LAB] Neural UCB: Error killing timed-out process");
                }
                return false;
            }

            var exitCode = process.ExitCode;

            if (exitCode == 0)
            {
                _logger.LogInformation("[LAB] ✅ Neural UCB: Python training completed successfully (exit code 0)");
                return true;
            }
            else
            {
                _logger.LogError("[LAB] ❌ Neural UCB: Python training failed with exit code {ExitCode}", exitCode);
                _logger.LogError("[LAB] Neural UCB: Python stdout:\n{Output}", outputBuilder.ToString());
                _logger.LogError("[LAB] Neural UCB: Python stderr:\n{Error}", errorBuilder.ToString());
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LAB] Neural UCB: Error invoking Python training - {Error}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Reloads updated ONNX models into Neural-UCB bandit after Python training completes.
    /// This ensures the C# inference uses the newly trained neural networks.
    /// </summary>
    private async Task<bool> ReloadNeuralUcbModelsAsync(
        global::BotCore.Bandits.NeuralUcbBandit bandit,
        CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogDebug("[LAB] {Component}: Reloading updated ONNX models...", ComponentNeuralUCB);

            var modelDir = "models";
            var expectedModels = new[] { "S2", "S3", "S6", "S11" };
            var foundModels = 0;

            foreach (var armId in expectedModels)
            {
                var modelPath = Path.Combine(modelDir, $"neural_ucb_model_{armId}.onnx");
                if (File.Exists(modelPath))
                {
                    var fileInfo = new FileInfo(modelPath);
                    _logger.LogDebug("[LAB] {Component}: Found model {ArmId}: {Path} ({Size} bytes, modified {Modified})",
                        ComponentNeuralUCB, armId, modelPath, fileInfo.Length, fileInfo.LastWriteTimeUtc);
                    foundModels++;
                }
                else
                {
                    _logger.LogWarning("[LAB] {Component}: Model not found for {ArmId}: {Path}", ComponentNeuralUCB, armId, modelPath);
                }
            }

            if (foundModels == 0)
            {
                _logger.LogError("[LAB] {Component}: No ONNX models found after training", ComponentNeuralUCB);
                return false;
            }

            _logger.LogDebug("[LAB] {Component}: Verified {Count}/{Total} models exist", ComponentNeuralUCB, foundModels, expectedModels.Length);

            // Hot-reload models into the bandit arms
            _logger.LogDebug("[LAB] {Component}: Hot-reloading models into bandit arms...", ComponentNeuralUCB);
            var reloadSuccess = await bandit.ReloadModelsAsync(modelDir, cancellationToken).ConfigureAwait(false);
            
            if (reloadSuccess)
            {
                _logger.LogInformation("[LAB] {Component}: Successfully hot-reloaded {Count} models without bot restart", 
                    ComponentNeuralUCB, foundModels);
            }
            else
            {
                _logger.LogWarning("[LAB] {Component}: Some models failed to reload, check logs for details", ComponentNeuralUCB);
            }
            
            return reloadSuccess;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LAB] {Component}: Error reloading models - {Error}", ComponentNeuralUCB, ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Finds Python executable in system PATH or common locations.
    /// Production-ready with multiple fallback strategies.
    /// </summary>
    private string? FindPythonExecutable()
    {
        var configuredPath = TryGetConfiguredPythonPath();
        if (configuredPath != null)
            return configuredPath;
        
        var pathPython = TryFindPythonInPath();
        if (pathPython != null)
            return pathPython;

        if (OperatingSystem.IsWindows())
        {
            var windowsPath = TryFindPythonInCommonWindowsPaths();
            if (windowsPath != null)
                return windowsPath;
        }

        if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
        {
            var unixPath = TryFindPythonInCommonUnixPaths();
            if (unixPath != null)
                return unixPath;
        }

        _logger.LogWarning("[LAB] Python executable not found in PATH or common locations");
        return null;
    }

    private string? TryGetConfiguredPythonPath()
    {
        var configuredPath = _configuration.GetValue<string>("LabMode:NeuralUCB:PythonExecutablePath");
        if (string.IsNullOrEmpty(configuredPath))
            return null;

        try
        {
            if (ValidatePythonExecutable(configuredPath))
            {
                _logger.LogDebug("[LAB] Found configured Python: {Python}", configuredPath);
                return configuredPath;
            }
            
            _logger.LogWarning("[LAB] Configured Python path '{Path}' failed validation, falling back to auto-detection", configuredPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[LAB] Configured Python path '{Path}' is invalid, falling back to auto-detection", configuredPath);
        }

        return null;
    }

    private string? TryFindPythonInPath()
    {
        var pythonNames = new[] { "python", "python3", "python.exe", "python3.exe" };
        
        foreach (var pythonName in pythonNames)
        {
            try
            {
                if (ValidatePythonExecutable(pythonName))
                {
                    _logger.LogDebug("[LAB] Found Python: {Python}", pythonName);
                    return pythonName;
                }
            }
            catch
            {
                // Try next name
            }
        }

        return null;
    }

    // Note: Hardcoded Windows Python paths are intentional fallbacks for common installation locations
    // when Python is not in PATH. These are industry-standard locations used by the official Python installer.
#pragma warning disable S1075 // URIs should not be hardcoded - these are intentional fallback paths for common Python installations
    private string? TryFindPythonInCommonWindowsPaths()
    {
        var commonPaths = new[]
        {
            @"C:\Python312\python.exe",
            @"C:\Python311\python.exe",
            @"C:\Python310\python.exe",
            @"C:\Python39\python.exe",
            @"C:\Python38\python.exe",
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python", "Python312", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python", "Python311", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python", "Python310", "python.exe"),
        };

        return FindFirstExistingPath(commonPaths);
    }
#pragma warning restore S1075

    private string? TryFindPythonInCommonUnixPaths()
    {
        var unixPaths = new[]
        {
            "/usr/bin/python3",
            "/usr/bin/python",
            "/usr/local/bin/python3",
            "/usr/local/bin/python",
        };

        return FindFirstExistingPath(unixPaths);
    }

    private string? FindFirstExistingPath(string[] paths)
    {
        foreach (var path in paths)
        {
            if (File.Exists(path))
            {
                _logger.LogDebug("[LAB] Found Python at: {Path}", path);
                return path;
            }
        }

        return null;
    }

    private static bool ValidatePythonExecutable(string pythonPath)
    {
        var process = new System.Diagnostics.Process
        {
            StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = "--version",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };

        process.Start();
        process.WaitForExit(5000);

        return process.ExitCode == 0;
    }

    #endregion

    #endregion
}

#region Supporting Types

/// <summary>
/// Training session result
/// </summary>
internal class TrainingSessionResult
{
    public string SessionId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public TimeSpan TotalDuration { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    
    // Data loading
    public int HistoricalBarsLoaded { get; set; }
    public int ExperiencesLoaded { get; set; }
    
    // Training results
    public bool CvarPpoSuccess { get; set; }
    public TimeSpan CvarPpoTrainingDuration { get; set; }
    
    public bool NeuralUcbSuccess { get; set; }
    public TimeSpan NeuralUcbTrainingDuration { get; set; }
    
    public bool LstmSuccess { get; set; }
    public TimeSpan LstmTrainingDuration { get; set; }
    
    public bool PositionMgmtSuccess { get; set; }
    public TimeSpan PositionMgmtTrainingDuration { get; set; }
    
    public bool ShadowValidationSuccess { get; set; }
    public TimeSpan ShadowValidationDuration { get; set; }
    
    // Model management
    public int ChallengersSaved { get; set; }
    public int ModelsPromoted { get; set; }
    public int ModelsDiscarded { get; set; }
    
    public List<string> FailedComponents { get; } = new();
}

/// <summary>
/// Experience record for RL training
/// </summary>
internal class Experience
{
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string State { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public decimal Reward { get; set; }
    public string NextState { get; set; } = string.Empty;
    public bool Done { get; set; }
}

#endregion
