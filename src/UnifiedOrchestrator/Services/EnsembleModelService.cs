using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using TradingBot.UnifiedOrchestrator.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Ensemble wrapper with voting/weights by confidence and input anomaly detection
/// Implements requirement: Ensemble wrapper with voting/weights by confidence and input anomaly detection
/// </summary>
public class EnsembleModelService
{
    private readonly ILogger<EnsembleModelService> _logger;
    private readonly Dictionary<string, ModelInfo> _models = new();
    private readonly object _lock = new();

    public EnsembleModelService(ILogger<EnsembleModelService> logger)
    {
        _logger = logger;
    }

    public void RegisterModel(string modelId, ModelInfo modelInfo)
    {
        lock (_lock)
        {
            _models[modelId] = modelInfo;
            _logger.LogInformation("Registered ensemble model: {ModelId} with weight {Weight}", 
                modelId, modelInfo.Weight);
        }
    }

    public async Task<EnsemblePrediction> GetEnsemblePredictionAsync(
        Dictionary<string, object> features, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Detect input anomalies first
            var anomalyScore = DetectInputAnomalies(features);
            if (anomalyScore > 0.8) // Configurable threshold
            {
                _logger.LogWarning("High anomaly score detected: {AnomalyScore:F3} - reducing ensemble confidence", anomalyScore);
            }

            var predictions = new List<ModelPrediction>();
            var totalWeight = 0.0;

            // Get predictions from all registered models
            foreach (var (modelId, modelInfo) in _models.ToList())
            {
                try
                {
                    var prediction = await GetModelPredictionAsync(modelId, modelInfo, features, cancellationToken);
                    predictions.Add(prediction);
                    totalWeight += modelInfo.Weight;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to get prediction from model {ModelId}", modelId);
                }
            }

            if (predictions.Count == 0)
            {
                throw new InvalidOperationException("No model predictions available");
            }

            // Calculate weighted ensemble prediction
            var ensembleResult = CalculateWeightedEnsemble(predictions, totalWeight, anomalyScore);

            _logger.LogDebug("Ensemble prediction: {Prediction:F3} confidence: {Confidence:F3} from {ModelCount} models", 
                ensembleResult.Prediction, ensembleResult.Confidence, predictions.Count);

            return ensembleResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get ensemble prediction");
            throw;
        }
    }

    private double DetectInputAnomalies(Dictionary<string, object> features)
    {
        try
        {
            // Simple anomaly detection based on feature value ranges
            // In production, this would use more sophisticated methods like Isolation Forest
            
            var anomalyScore = 0.0;
            var checkedFeatures = 0;

            foreach (var (key, value) in features)
            {
                if (value is double doubleValue)
                {
                    checkedFeatures++;
                    
                    // Check for extreme values (configurable thresholds)
                    if (double.IsNaN(doubleValue) || double.IsInfinity(doubleValue))
                    {
                        anomalyScore += 1.0; // Maximum penalty for invalid values
                    }
                    else if (Math.Abs(doubleValue) > 1000) // Configurable threshold
                    {
                        anomalyScore += 0.5; // Penalty for extreme values
                    }
                    else if (Math.Abs(doubleValue) > 100) // Configurable threshold
                    {
                        anomalyScore += 0.2; // Minor penalty for large values
                    }
                }
            }

            // Normalize by number of checked features
            return checkedFeatures > 0 ? anomalyScore / checkedFeatures : 0.0;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to detect input anomalies");
            return 0.0; // Return safe default
        }
    }

    private async Task<ModelPrediction> GetModelPredictionAsync(
        string modelId, 
        ModelInfo modelInfo, 
        Dictionary<string, object> features, 
        CancellationToken cancellationToken)
    {
        // Placeholder implementation - in real system would call actual model inference
        // This would integrate with ONNXModelLoader or other inference engines
        
        await Task.Delay(10, cancellationToken); // Simulate inference time
        
        // Simulate model prediction with some randomness based on model type
        var random = new Random(modelId.GetHashCode());
        var basePrediction = random.NextDouble();
        var confidence = 0.5 + (random.NextDouble() * 0.4); // 0.5 to 0.9 range

        return new ModelPrediction
        {
            ModelId = modelId,
            Prediction = basePrediction,
            Confidence = confidence,
            InferenceTime = TimeSpan.FromMilliseconds(random.Next(5, 50))
        };
    }

    private EnsemblePrediction CalculateWeightedEnsemble(
        List<ModelPrediction> predictions, 
        double totalWeight, 
        double anomalyScore)
    {
        // Calculate confidence-weighted prediction
        var weightedSum = 0.0;
        var confidenceWeightSum = 0.0;

        foreach (var prediction in predictions)
        {
            var modelWeight = _models[prediction.ModelId].Weight;
            var confidenceWeight = modelWeight * prediction.Confidence;
            
            weightedSum += prediction.Prediction * confidenceWeight;
            confidenceWeightSum += confidenceWeight;
        }

        var ensemblePrediction = confidenceWeightSum > 0 ? weightedSum / confidenceWeightSum : 0.0;

        // Calculate ensemble confidence considering individual confidences and anomaly score
        var avgConfidence = predictions.Average(p => p.Confidence);
        var confidenceVariance = predictions.Select(p => Math.Pow(p.Confidence - avgConfidence, 2)).Average();
        var consensusBonus = Math.Exp(-confidenceVariance * 10); // Higher consensus = higher confidence
        var anomalyPenalty = Math.Max(0.1, 1.0 - anomalyScore); // Reduce confidence for anomalies

        var ensembleConfidence = avgConfidence * consensusBonus * anomalyPenalty;

        // Calculate prediction metrics
        var predictionVariance = predictions.Select(p => Math.Pow(p.Prediction - ensemblePrediction, 2)).Average();
        var avgInferenceTime = TimeSpan.FromMilliseconds(predictions.Average(p => p.InferenceTime.TotalMilliseconds));

        return new EnsemblePrediction
        {
            Prediction = ensemblePrediction,
            Confidence = Math.Min(1.0, Math.Max(0.0, ensembleConfidence)),
            AnomalyScore = anomalyScore,
            ModelCount = predictions.Count,
            PredictionVariance = predictionVariance,
            AvgInferenceTime = avgInferenceTime,
            ModelPredictions = predictions
        };
    }
}

/// <summary>
/// Information about a model in the ensemble
/// </summary>
public class ModelInfo
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelType { get; set; } = string.Empty;
    public double Weight { get; set; } = 1.0;
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
    public double HistoricalAccuracy { get; set; } = 0.6;
}

/// <summary>
/// Individual model prediction
/// </summary>
public class ModelPrediction
{
    public string ModelId { get; set; } = string.Empty;
    public double Prediction { get; set; }
    public double Confidence { get; set; }
    public TimeSpan InferenceTime { get; set; }
}

/// <summary>
/// Ensemble prediction result
/// </summary>
public class EnsemblePrediction
{
    public double Prediction { get; set; }
    public double Confidence { get; set; }
    public double AnomalyScore { get; set; }
    public int ModelCount { get; set; }
    public double PredictionVariance { get; set; }
    public TimeSpan AvgInferenceTime { get; set; }
    public List<ModelPrediction> ModelPredictions { get; set; } = new();
}