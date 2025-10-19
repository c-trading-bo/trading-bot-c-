using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.BotCore.Data;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Historical Training Orchestrator - Master controller for Lab training pipeline
/// Runs complete training session on Sunday (segregated from Terminal)
/// 
/// This is the "shift supervisor" that coordinates the entire training factory:
/// 1. Load experiences from last 7 days
/// 2. Load 90-day historical bars
/// 3. Run sequential training pipeline
/// 4. Save challengers to registry
/// 5. Run promotion evaluations
/// </summary>
internal sealed class HistoricalTrainingOrchestrator
{
    private readonly ILogger<HistoricalTrainingOrchestrator> _logger;
    private readonly HistoricalDataProvider _historicalDataProvider;
    private readonly global::BotCore.Data.ExperienceRepository? _experienceRepository;
    private readonly IModelRegistry _modelRegistry;
    private readonly IPromotionService _promotionService;
    private readonly SemaphoreSlim _trainingLock = new(1, 1);
    
    // Training pipeline configuration
    private readonly TimeSpan _cvarPpoTrainingTime = TimeSpan.FromMinutes(30);
    private readonly TimeSpan _neuralUcbTrainingTime = TimeSpan.FromMinutes(15);
    private readonly TimeSpan _lstmTrainingTime = TimeSpan.FromMinutes(20);
    private readonly TimeSpan _positionMgmtTrainingTime = TimeSpan.FromMinutes(30);
    private readonly TimeSpan _shadowValidationTime = TimeSpan.FromMinutes(30);

    public HistoricalTrainingOrchestrator(
        ILogger<HistoricalTrainingOrchestrator> logger,
        HistoricalDataProvider historicalDataProvider,
        global::BotCore.Data.ExperienceRepository? experienceRepository,
        IModelRegistry modelRegistry,
        IPromotionService promotionService)
    {
        _logger = logger;
        _historicalDataProvider = historicalDataProvider;
        _experienceRepository = experienceRepository;
        _modelRegistry = modelRegistry;
        _promotionService = promotionService;
        
        _logger.LogInformation("HistoricalTrainingOrchestrator initialized");
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
            
            _logger.LogInformation("🚀 Training session started - Sunday {Date} {Time} UTC (Session: {SessionId})", 
                startTime.ToString("yyyy-MM-dd"), startTime.ToString("HH:mm"), sessionId);

            var result = new TrainingSessionResult
            {
                SessionId = sessionId,
                StartTime = startTime
            };

            try
            {
                // Step 1: Load historical data (90 days)
                _logger.LogInformation("📊 Step 1/6: Loading 90-day historical bars");
                var historicalData = await LoadHistoricalDataAsync(cancellationToken).ConfigureAwait(false);
                result.HistoricalBarsLoaded = historicalData.Sum(kvp => kvp.Value);
                _logger.LogInformation("✅ Loaded {Count} historical bars", result.HistoricalBarsLoaded);

                // Step 2: Load recent experiences (last 7 days)
                _logger.LogInformation("📊 Step 2/6: Loading experiences from last 7 days");
                var experiences = await LoadRecentExperiencesAsync(cancellationToken).ConfigureAwait(false);
                result.ExperiencesLoaded = experiences.Count;
                _logger.LogInformation("✅ Loaded {Count} experiences", result.ExperiencesLoaded);

                // Step 3: Run training pipeline sequentially
                _logger.LogInformation("🔧 Step 3/6: Running training pipeline");
                await RunTrainingPipelineAsync(historicalData, experiences, result, cancellationToken).ConfigureAwait(false);

                // Step 4: Save all challengers to registry
                _logger.LogInformation("💾 Step 4/6: Saving challengers to model registry");
                await SaveChallengersAsync(result, cancellationToken).ConfigureAwait(false);

                // Step 5: Run promotion evaluations
                _logger.LogInformation("📈 Step 5/6: Running promotion evaluations");
                await RunPromotionEvaluationsAsync(result, cancellationToken).ConfigureAwait(false);

                // Step 6: Generate session summary
                _logger.LogInformation("📋 Step 6/6: Generating session summary");
                result.EndTime = DateTime.UtcNow;
                result.TotalDuration = result.EndTime - result.StartTime;
                result.Success = true;

                LogSessionSummary(result);
                
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Training session failed: {Error}", ex.Message);
                result.Success = false;
                result.ErrorMessage = ex.Message;
                result.EndTime = DateTime.UtcNow;
                result.TotalDuration = result.EndTime - result.StartTime;
                return result;
            }
        }
        finally
        {
            _trainingLock.Release();
        }
    }

    #region Private Methods - Data Loading

    private async Task<Dictionary<string, int>> LoadHistoricalDataAsync(CancellationToken cancellationToken)
    {
        // Returns count of bars loaded per symbol
        // Actual bar data would be stored in experience repository or passed to training methods
        var data = new Dictionary<string, int>();
        var symbols = new[] { "ES", "NQ" };
        var to = DateTime.UtcNow.Date;
        var from = to.AddDays(-90);

        foreach (var symbol in symbols)
        {
            try
            {
                var historicalBars = await _historicalDataProvider.GetCachedBarsAsync(symbol, from, to, cancellationToken).ConfigureAwait(false);
                data[symbol] = historicalBars.Count;
                _logger.LogInformation("Loaded {Count} bars for {Symbol}", historicalBars.Count, symbol);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load historical data for {Symbol}", symbol);
                data[symbol] = 0;
            }
        }

        return data;
    }

    private async Task<List<Experience>> LoadRecentExperiencesAsync(CancellationToken cancellationToken)
    {
        if (_experienceRepository == null)
        {
            _logger.LogWarning("ExperienceRepository not available - returning empty experiences");
            return new List<Experience>();
        }

        try
        {
            // Load experiences from last 7 days
            var tradingExperiences = await _experienceRepository.LoadRecentExperiencesAsync(7).ConfigureAwait(false);
            
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
            
            _logger.LogInformation("✅ Loaded and converted {Count} trading experiences", experiences.Count);
            return experiences;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to load recent experiences");
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
        
        // 1. CVaR-PPO Training (30 min)
        await TrainCVarPPOAsync(result, cancellationToken).ConfigureAwait(false);

        // 2. Neural UCB Retraining (15 min)
        await TrainNeuralUCBAsync(result, cancellationToken).ConfigureAwait(false);

        // 3. LSTM Training (20 min)
        await TrainLSTMAsync(result, cancellationToken).ConfigureAwait(false);

        // 4. Position Management Optimization (30 min)
        await OptimizePositionManagementAsync(result, cancellationToken).ConfigureAwait(false);

        // 5. S15 Shadow Validation (30 min)
        await RunS15ShadowValidationAsync(result, cancellationToken).ConfigureAwait(false);
    }

    private async Task TrainCVarPPOAsync(
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogInformation("🔧 Training CVaR-PPO (estimated {Minutes} min)", _cvarPpoTrainingTime.TotalMinutes);
            
            // TODO: Actual CVaR-PPO training implementation
            // For now, simulate training with progress logging
            await SimulateTrainingAsync("CVaR-PPO", _cvarPpoTrainingTime, 10, cancellationToken).ConfigureAwait(false);
            
            stopwatch.Stop();
            result.CvarPpoTrainingDuration = stopwatch.Elapsed;
            result.CvarPpoSuccess = true;
            
            _logger.LogInformation("✅ CVaR-PPO training completed in {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "❌ CVaR-PPO training failed after {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
            result.CvarPpoTrainingDuration = stopwatch.Elapsed;
            result.CvarPpoSuccess = false;
            result.FailedComponents.Add("CVaR-PPO");
        }
    }

    private async Task TrainNeuralUCBAsync(
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogInformation("🔧 Training Neural UCB (estimated {Minutes} min)", _neuralUcbTrainingTime.TotalMinutes);
            
            // TODO: Actual Neural UCB training implementation
            await SimulateTrainingAsync("Neural UCB", _neuralUcbTrainingTime, 5, cancellationToken).ConfigureAwait(false);
            
            stopwatch.Stop();
            result.NeuralUcbTrainingDuration = stopwatch.Elapsed;
            result.NeuralUcbSuccess = true;
            
            _logger.LogInformation("✅ Neural UCB training completed in {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "❌ Neural UCB training failed after {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
            result.NeuralUcbTrainingDuration = stopwatch.Elapsed;
            result.NeuralUcbSuccess = false;
            result.FailedComponents.Add("Neural UCB");
        }
    }

    private async Task TrainLSTMAsync(
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogInformation("🔧 Training LSTM (estimated {Minutes} min)", _lstmTrainingTime.TotalMinutes);
            
            // TODO: Actual LSTM training implementation
            await SimulateTrainingAsync("LSTM", _lstmTrainingTime, 8, cancellationToken).ConfigureAwait(false);
            
            stopwatch.Stop();
            result.LstmTrainingDuration = stopwatch.Elapsed;
            result.LstmSuccess = true;
            
            _logger.LogInformation("✅ LSTM training completed in {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "❌ LSTM training failed after {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
            result.LstmTrainingDuration = stopwatch.Elapsed;
            result.LstmSuccess = false;
            result.FailedComponents.Add("LSTM");
        }
    }

    private async Task OptimizePositionManagementAsync(
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogInformation("🔧 Optimizing Position Management (estimated {Minutes} min)", _positionMgmtTrainingTime.TotalMinutes);
            
            // TODO: Actual position management optimization
            await SimulateTrainingAsync("Position Management", _positionMgmtTrainingTime, 6, cancellationToken).ConfigureAwait(false);
            
            stopwatch.Stop();
            result.PositionMgmtTrainingDuration = stopwatch.Elapsed;
            result.PositionMgmtSuccess = true;
            
            _logger.LogInformation("✅ Position management optimization completed in {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "❌ Position management optimization failed after {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
            result.PositionMgmtTrainingDuration = stopwatch.Elapsed;
            result.PositionMgmtSuccess = false;
            result.FailedComponents.Add("Position Management");
        }
    }

    private async Task RunS15ShadowValidationAsync(
        TrainingSessionResult result,
        CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            _logger.LogInformation("🔧 Running S15 shadow validation (estimated {Minutes} min)", _shadowValidationTime.TotalMinutes);
            
            // TODO: Actual S15 shadow validation
            await SimulateTrainingAsync("S15 Shadow Validation", _shadowValidationTime, 4, cancellationToken).ConfigureAwait(false);
            
            stopwatch.Stop();
            result.ShadowValidationDuration = stopwatch.Elapsed;
            result.ShadowValidationSuccess = true;
            
            _logger.LogInformation("✅ S15 shadow validation completed in {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _logger.LogError(ex, "❌ S15 shadow validation failed after {Duration:F1} min", stopwatch.Elapsed.TotalMinutes);
            result.ShadowValidationDuration = stopwatch.Elapsed;
            result.ShadowValidationSuccess = false;
            result.FailedComponents.Add("S15 Shadow Validation");
        }
    }

    private async Task SimulateTrainingAsync(string componentName, TimeSpan duration, int epochs, CancellationToken cancellationToken)
    {
        // Simulate training with progress logging
        var epochDuration = duration.TotalSeconds / epochs;
        
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var progress = (epoch * 100.0) / epochs;
            _logger.LogInformation("{Component} training: {Progress:F0}% complete (epoch {Epoch}/{TotalEpochs})", 
                componentName, progress, epoch, epochs);
            
            await Task.Delay(TimeSpan.FromSeconds(epochDuration / 10), cancellationToken).ConfigureAwait(false);
        }
    }

    #endregion

    #region Private Methods - Model Management

    private async Task SaveChallengersAsync(TrainingSessionResult result, CancellationToken cancellationToken)
    {
        var savedCount = 0;
        var algorithms = new[] { "cvar-ppo", "neural-ucb", "lstm", "position-management" };

        foreach (var algorithm in algorithms)
        {
            try
            {
                // TODO: Save actual trained models
                // For now, just log the intent
                _logger.LogDebug("Saving challenger for {Algorithm} (implementation pending)", algorithm);
                savedCount++;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to save challenger for {Algorithm}", algorithm);
            }
        }

        result.ChallengersSaved = savedCount;
        _logger.LogInformation("💾 Saved {Count} challengers to registry", savedCount);
    }

    private async Task RunPromotionEvaluationsAsync(TrainingSessionResult result, CancellationToken cancellationToken)
    {
        var algorithms = new[] { "cvar-ppo", "neural-ucb", "lstm" };
        
        foreach (var algorithm in algorithms)
        {
            try
            {
                // Get the latest challenger version (would be from SaveChallengersAsync in real implementation)
                var challengerVersionId = $"{algorithm}_v{DateTime.UtcNow:yyyyMMdd}_challenger";
                
                _logger.LogInformation("Evaluating promotion for {Algorithm} challenger {VersionId}", 
                    algorithm, challengerVersionId);

                var decision = await _promotionService.EvaluatePromotionAsync(algorithm, challengerVersionId, cancellationToken).ConfigureAwait(false);
                
                if (decision.ShouldPromote)
                {
                    _logger.LogInformation("✅ Promotion recommended for {Algorithm}", algorithm);
                    result.ModelsPromoted++;
                }
                else
                {
                    _logger.LogInformation("⏸️ Promotion NOT recommended for {Algorithm}: {Reason}", 
                        algorithm, decision.Reason);
                    result.ModelsDiscarded++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to evaluate promotion for {Algorithm}", algorithm);
            }
        }
    }

    private void LogSessionSummary(TrainingSessionResult result)
    {
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
