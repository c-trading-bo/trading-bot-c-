using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.IO;
using System.Linq;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Ensemble meta-learner with per-regime blend heads
/// Implements smooth transitions and regime-specific model blending
/// </summary>
public class EnsembleMetaLearner
{
    private readonly ILogger<EnsembleMetaLearner> _logger;
    private readonly EnsembleConfig _config;
    private readonly IRegimeDetector _regimeDetector;
    private readonly IModelRegistry _modelRegistry;
    private readonly IOnlineLearningSystem _onlineLearning;
    private readonly string _statePath;
    
    private readonly Dictionary<RegimeType, RegimeBlendHead> _regimeHeads = new();
    private readonly Dictionary<string, ModelArtifact> _activeModels = new();
    private readonly object _lock = new();
    
    private RegimeType _currentRegime = RegimeType.Range;
    private RegimeType _previousRegime = RegimeType.Range;
    private DateTime _lastTransitionTime = DateTime.MinValue;
    private bool _inTransition = false;

    public EnsembleMetaLearner(
        ILogger<EnsembleMetaLearner> logger,
        EnsembleConfig config,
        IRegimeDetector regimeDetector,
        IModelRegistry modelRegistry,
        IOnlineLearningSystem onlineLearning,
        string statePath = "data/ensemble")
    {
        _logger = logger;
        _config = config;
        _regimeDetector = regimeDetector;
        _modelRegistry = modelRegistry;
        _onlineLearning = onlineLearning;
        _statePath = statePath;
        
        Directory.CreateDirectory(_statePath);
        InitializeRegimeHeads();
        _ = Task.Run(() => LoadStateAsync());
    }

    /// <summary>
    /// Get blended prediction from ensemble with regime-specific weighting
    /// </summary>
    public async Task<EnsemblePrediction> GetBlendedPredictionAsync(
        MarketContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Detect current regime
            var regimeState = await _regimeDetector.DetectCurrentRegimeAsync(cancellationToken);
            
            // Check for regime transition
            var transition = await CheckRegimeTransitionAsync(regimeState, cancellationToken);
            
            // Get predictions from all active models
            var modelPredictions = await GetModelPredictionsAsync(context, cancellationToken);
            
            // Blend predictions based on current regime
            var prediction = await BlendPredictionsAsync(
                modelPredictions, 
                regimeState.Type, 
                transition,
                cancellationToken);

            // Update online learning with prediction feedback
            await UpdateOnlineLearningAsync(regimeState.Type, prediction, cancellationToken);

            return prediction;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENSEMBLE] Failed to get blended prediction");
            return CreateFallbackPrediction();
        }
    }

    /// <summary>
    /// Train regime-specific blend heads
    /// </summary>
    public async Task TrainRegimeHeadsAsync(
        RegimeType regime,
        IEnumerable<TrainingExample> examples,
        CancellationToken cancellationToken = default)
    {
        try
        {
            if (!_regimeHeads.TryGetValue(regime, out var head))
            {
                _logger.LogWarning("[ENSEMBLE] No blend head found for regime: {Regime}", regime);
                return;
            }

            _logger.LogInformation("[ENSEMBLE] Training blend head for regime: {Regime} with {Count} examples", 
                regime, examples.Count());

            await head.TrainAsync(examples, cancellationToken);
            
            // Save updated state
            await SaveStateAsync(cancellationToken);

            _logger.LogInformation("[ENSEMBLE] Completed training for regime: {Regime}", regime);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENSEMBLE] Failed to train regime head: {Regime}", regime);
        }
    }

    /// <summary>
    /// Update ensemble with new model performance feedback
    /// </summary>
    public async Task UpdateWithFeedbackAsync(
        string modelId,
        ModelPerformance performance,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Update online learning system
            await _onlineLearning.AdaptToPerformanceAsync(modelId, performance, cancellationToken);

            // Update regime-specific weights based on performance
            var regimeTypeStr = _currentRegime.ToString();
            var currentWeights = await _onlineLearning.GetCurrentWeightsAsync(regimeTypeStr, cancellationToken);
            
            // Adjust model weight based on performance
            var performanceScore = CalculatePerformanceScore(performance);
            if (currentWeights.ContainsKey(modelId))
            {
                var currentWeight = currentWeights[modelId];
                var adjustment = (performanceScore - 0.5) * 0.1; // ±10% adjustment
                currentWeights[modelId] = Math.Max(0.1, Math.Min(2.0, currentWeight + adjustment));
            }

            await _onlineLearning.UpdateWeightsAsync(regimeTypeStr, currentWeights, cancellationToken);

            _logger.LogDebug("[ENSEMBLE] Updated feedback for model: {ModelId} in regime: {Regime}", 
                modelId, _currentRegime);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENSEMBLE] Failed to update feedback for model: {ModelId}", modelId);
        }
    }

    /// <summary>
    /// Get current ensemble status and weights
    /// </summary>
    public EnsembleStatus GetCurrentStatus()
    {
        lock (_lock)
        {
            var status = new EnsembleStatus
            {
                CurrentRegime = _currentRegime,
                PreviousRegime = _previousRegime,
                InTransition = _inTransition,
                TransitionStartTime = _lastTransitionTime,
                ActiveModels = new Dictionary<string, double>(),
                RegimeHeadStatus = new Dictionary<RegimeType, RegimeHeadStatus>()
            };

            // Get current weights for active regime
            var regimeTypeStr = _currentRegime.ToString();
            var task = _onlineLearning.GetCurrentWeightsAsync(regimeTypeStr, CancellationToken.None);
            task.Wait(TimeSpan.FromSeconds(1));
            
            if (task.IsCompletedSuccessfully)
            {
                status.ActiveModels = task.Result;
            }

            // Get regime head status
            foreach (var (regime, head) in _regimeHeads)
            {
                status.RegimeHeadStatus[regime] = new RegimeHeadStatus
                {
                    IsActive = regime == _currentRegime,
                    LastTrainingTime = head.LastTrainingTime,
                    TrainingExamples = head.TrainingExampleCount,
                    ValidationScore = head.LastValidationScore
                };
            }

            return status;
        }
    }

    private void InitializeRegimeHeads()
    {
        foreach (RegimeType regime in Enum.GetValues<RegimeType>())
        {
            _regimeHeads[regime] = new RegimeBlendHead(_logger, regime);
        }
    }

    private async Task<RegimeTransition?> CheckRegimeTransitionAsync(
        RegimeState currentState,
        CancellationToken cancellationToken)
    {
        if (currentState.Type != _currentRegime)
        {
            _logger.LogInformation("[ENSEMBLE] Regime transition detected: {From} -> {To}", 
                _currentRegime, currentState.Type);

            _previousRegime = _currentRegime;
            _currentRegime = currentState.Type;
            _lastTransitionTime = DateTime.UtcNow;
            _inTransition = true;

            return new RegimeTransition
            {
                ShouldTransition = true,
                FromRegime = _previousRegime,
                ToRegime = _currentRegime,
                TransitionConfidence = currentState.Confidence,
                BlendDuration = TimeSpan.FromSeconds(_config.MetaPerRegime.TransitionBlendSeconds)
            };
        }

        // Check if we're still in transition
        if (_inTransition)
        {
            var transitionDuration = DateTime.UtcNow - _lastTransitionTime;
            var maxDuration = TimeSpan.FromSeconds(_config.MetaPerRegime.TransitionBlendSeconds);
            
            if (transitionDuration >= maxDuration)
            {
                _inTransition = false;
                _logger.LogDebug("[ENSEMBLE] Transition completed for regime: {Regime}", _currentRegime);
            }
        }

        return null;
    }

    private async Task<Dictionary<string, ModelPrediction>> GetModelPredictionsAsync(
        MarketContext context,
        CancellationToken cancellationToken)
    {
        var predictions = new Dictionary<string, ModelPrediction>();

        // Get predictions from all active models
        foreach (var (modelId, model) in _activeModels)
        {
            try
            {
                var prediction = await GetModelPredictionAsync(model, context, cancellationToken);
                predictions[modelId] = prediction;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[ENSEMBLE] Failed to get prediction from model: {ModelId}", modelId);
            }
        }

        return predictions;
    }

    private async Task<ModelPrediction> GetModelPredictionAsync(
        ModelArtifact model,
        MarketContext context,
        CancellationToken cancellationToken)
    {
        // Simplified model prediction - in production would use actual model inference
        var baseConfidence = 0.5 + (new Random().NextDouble() - 0.5) * 0.4;
        var direction = context.Price > context.TechnicalIndicators.GetValueOrDefault("sma_20", context.Price) ? 1.0 : -1.0;
        
        return new ModelPrediction
        {
            ModelId = model.Id,
            ModelVersion = model.Version,
            Confidence = baseConfidence,
            Direction = direction,
            Strength = Math.Abs(direction) * baseConfidence,
            Features = context.TechnicalIndicators,
            Timestamp = DateTime.UtcNow
        };
    }

    private async Task<EnsemblePrediction> BlendPredictionsAsync(
        Dictionary<string, ModelPrediction> modelPredictions,
        RegimeType currentRegime,
        RegimeTransition? transition,
        CancellationToken cancellationToken)
    {
        var weights = await _onlineLearning.GetCurrentWeightsAsync(currentRegime.ToString(), cancellationToken);
        
        // Handle transition blending
        if (transition != null && _inTransition)
        {
            var transitionWeight = CalculateTransitionWeight(transition);
            var previousWeights = await _onlineLearning.GetCurrentWeightsAsync(_previousRegime.ToString(), cancellationToken);
            
            weights = BlendWeights(previousWeights, weights, transitionWeight);
        }

        // Calculate weighted ensemble prediction
        double totalWeight = 0;
        double weightedDirection = 0;
        double weightedConfidence = 0;
        var blendedFeatures = new Dictionary<string, double>();

        foreach (var (modelId, prediction) in modelPredictions)
        {
            var weight = weights.GetValueOrDefault(modelId, 1.0);
            totalWeight += weight;
            weightedDirection += prediction.Direction * weight;
            weightedConfidence += prediction.Confidence * weight;

            // Blend features
            foreach (var (feature, value) in prediction.Features)
            {
                blendedFeatures[feature] = blendedFeatures.GetValueOrDefault(feature, 0.0) + value * weight;
            }
        }

        if (totalWeight > 0)
        {
            weightedDirection /= totalWeight;
            weightedConfidence /= totalWeight;
            
            foreach (var feature in blendedFeatures.Keys.ToList())
            {
                blendedFeatures[feature] /= totalWeight;
            }
        }

        return new EnsemblePrediction
        {
            Direction = weightedDirection,
            Confidence = weightedConfidence,
            Strength = Math.Abs(weightedDirection) * weightedConfidence,
            CurrentRegime = currentRegime,
            InTransition = _inTransition,
            ModelCount = modelPredictions.Count,
            BlendedFeatures = blendedFeatures,
            Weights = weights,
            Timestamp = DateTime.UtcNow
        };
    }

    private double CalculateTransitionWeight(RegimeTransition transition)
    {
        var elapsed = DateTime.UtcNow - _lastTransitionTime;
        var totalDuration = transition.BlendDuration;
        
        if (elapsed >= totalDuration)
        {
            return 1.0; // Full transition to new regime
        }
        
        // Smooth sigmoid transition
        var progress = elapsed.TotalSeconds / totalDuration.TotalSeconds;
        return 1.0 / (1.0 + Math.Exp(-6 * (progress - 0.5))); // Sigmoid centered at 0.5
    }

    private Dictionary<string, double> BlendWeights(
        Dictionary<string, double> fromWeights,
        Dictionary<string, double> toWeights,
        double transitionWeight)
    {
        var blended = new Dictionary<string, double>();
        var allKeys = fromWeights.Keys.Union(toWeights.Keys);
        
        foreach (var key in allKeys)
        {
            var fromWeight = fromWeights.GetValueOrDefault(key, 1.0);
            var toWeight = toWeights.GetValueOrDefault(key, 1.0);
            blended[key] = fromWeight * (1.0 - transitionWeight) + toWeight * transitionWeight;
        }
        
        return blended;
    }

    private async Task UpdateOnlineLearningAsync(
        RegimeType regime,
        EnsemblePrediction prediction,
        CancellationToken cancellationToken)
    {
        // Update weights based on prediction confidence
        var regimeStr = regime.ToString();
        var currentWeights = await _onlineLearning.GetCurrentWeightsAsync(regimeStr, cancellationToken);
        
        // Boost weights for high-confidence predictions
        if (prediction.Confidence > 0.7)
        {
            foreach (var key in currentWeights.Keys.ToList())
            {
                currentWeights[key] *= 1.01; // Small boost
            }
            
            await _onlineLearning.UpdateWeightsAsync(regimeStr, currentWeights, cancellationToken);
        }
    }

    private double CalculatePerformanceScore(ModelPerformance performance)
    {
        // Combine multiple metrics into a single performance score
        var brierScore = Math.Max(0, 0.25 - performance.BrierScore) / 0.25; // 0-1 scale
        var hitRate = performance.HitRate; // Already 0-1 scale
        var latencyScore = Math.Max(0, 1.0 - performance.Latency / 1000.0); // Penalize high latency
        
        return (brierScore * 0.5) + (hitRate * 0.4) + (latencyScore * 0.1);
    }

    private EnsemblePrediction CreateFallbackPrediction()
    {
        return new EnsemblePrediction
        {
            Direction = 0.0,
            Confidence = 0.1,
            Strength = 0.0,
            CurrentRegime = _currentRegime,
            InTransition = false,
            ModelCount = 0,
            BlendedFeatures = new Dictionary<string, double>(),
            Weights = new Dictionary<string, double>(),
            Timestamp = DateTime.UtcNow
        };
    }

    private async Task SaveStateAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var state = new EnsembleState
            {
                CurrentRegime = _currentRegime,
                PreviousRegime = _previousRegime,
                LastTransitionTime = _lastTransitionTime,
                InTransition = _inTransition,
                RegimeHeadData = _regimeHeads.ToDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value.GetState()
                )
            };

            var stateFile = Path.Combine(_statePath, "ensemble_state.json");
            var json = JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(stateFile, json, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ENSEMBLE] Failed to save state");
        }
    }

    private async Task LoadStateAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var stateFile = Path.Combine(_statePath, "ensemble_state.json");
            if (!File.Exists(stateFile))
            {
                return;
            }

            var content = await File.ReadAllTextAsync(stateFile, cancellationToken);
            var state = JsonSerializer.Deserialize<EnsembleState>(content);
            
            if (state != null)
            {
                _currentRegime = state.CurrentRegime;
                _previousRegime = state.PreviousRegime;
                _lastTransitionTime = state.LastTransitionTime;
                _inTransition = state.InTransition;

                foreach (var (regime, headData) in state.RegimeHeadData)
                {
                    if (_regimeHeads.TryGetValue(regime, out var head))
                    {
                        head.LoadState(headData);
                    }
                }

                _logger.LogInformation("[ENSEMBLE] Loaded ensemble state - current regime: {Regime}", _currentRegime);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ENSEMBLE] Failed to load state");
        }
    }
}

/// <summary>
/// Regime-specific blend head for ensemble learning
/// </summary>
public class RegimeBlendHead
{
    private readonly ILogger _logger;
    private readonly RegimeType _regime;
    
    public DateTime LastTrainingTime { get; private set; } = DateTime.MinValue;
    public int TrainingExampleCount { get; private set; } = 0;
    public double LastValidationScore { get; private set; } = 0.0;

    public RegimeBlendHead(ILogger logger, RegimeType regime)
    {
        _logger = logger;
        _regime = regime;
    }

    public async Task TrainAsync(IEnumerable<TrainingExample> examples, CancellationToken cancellationToken)
    {
        var exampleList = examples.ToList();
        TrainingExampleCount = exampleList.Count;
        LastTrainingTime = DateTime.UtcNow;
        
        // Simplified training - in production would use actual ML training
        var validationExamples = exampleList.TakeLast(Math.Min(100, exampleList.Count / 5));
        LastValidationScore = CalculateValidationScore(validationExamples);
        
        _logger.LogDebug("[REGIME_HEAD] Trained {Regime} head with {Count} examples (validation: {Score:F3})", 
            _regime, TrainingExampleCount, LastValidationScore);
    }

    public RegimeHeadState GetState()
    {
        return new RegimeHeadState
        {
            Regime = _regime,
            LastTrainingTime = LastTrainingTime,
            TrainingExampleCount = TrainingExampleCount,
            LastValidationScore = LastValidationScore
        };
    }

    public void LoadState(RegimeHeadState state)
    {
        LastTrainingTime = state.LastTrainingTime;
        TrainingExampleCount = state.TrainingExampleCount;
        LastValidationScore = state.LastValidationScore;
    }

    private double CalculateValidationScore(IEnumerable<TrainingExample> examples)
    {
        // Simplified validation score calculation
        var exampleList = examples.ToList();
        if (exampleList.Count == 0) return 0.0;
        
        var correctPredictions = exampleList.Count(e => 
            (e.PredictedDirection > 0 && e.ActualOutcome > 0) ||
            (e.PredictedDirection < 0 && e.ActualOutcome < 0));
            
        return (double)correctPredictions / exampleList.Count;
    }
}

#region Supporting Classes

public class EnsemblePrediction
{
    public double Direction { get; set; }
    public double Confidence { get; set; }
    public double Strength { get; set; }
    public RegimeType CurrentRegime { get; set; }
    public bool InTransition { get; set; }
    public int ModelCount { get; set; }
    public Dictionary<string, double> BlendedFeatures { get; set; } = new();
    public Dictionary<string, double> Weights { get; set; } = new();
    public DateTime Timestamp { get; set; }
}

public class ModelPrediction
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelVersion { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public double Direction { get; set; }
    public double Strength { get; set; }
    public Dictionary<string, double> Features { get; set; } = new();
    public DateTime Timestamp { get; set; }
}

public class TrainingExample
{
    public Dictionary<string, double> Features { get; set; } = new();
    public double PredictedDirection { get; set; }
    public double ActualOutcome { get; set; }
    public DateTime Timestamp { get; set; }
    public RegimeType Regime { get; set; }
}

public class EnsembleStatus
{
    public RegimeType CurrentRegime { get; set; }
    public RegimeType PreviousRegime { get; set; }
    public bool InTransition { get; set; }
    public DateTime TransitionStartTime { get; set; }
    public Dictionary<string, double> ActiveModels { get; set; } = new();
    public Dictionary<RegimeType, RegimeHeadStatus> RegimeHeadStatus { get; set; } = new();
}

public class RegimeHeadStatus
{
    public bool IsActive { get; set; }
    public DateTime LastTrainingTime { get; set; }
    public int TrainingExamples { get; set; }
    public double ValidationScore { get; set; }
}

public class EnsembleState
{
    public RegimeType CurrentRegime { get; set; }
    public RegimeType PreviousRegime { get; set; }
    public DateTime LastTransitionTime { get; set; }
    public bool InTransition { get; set; }
    public Dictionary<RegimeType, RegimeHeadState> RegimeHeadData { get; set; } = new();
}

public class RegimeHeadState
{
    public RegimeType Regime { get; set; }
    public DateTime LastTrainingTime { get; set; }
    public int TrainingExampleCount { get; set; }
    public double LastValidationScore { get; set; }
}

#endregion