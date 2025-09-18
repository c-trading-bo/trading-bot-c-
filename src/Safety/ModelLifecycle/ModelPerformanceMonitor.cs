using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace TradingBot.Safety.ModelLifecycle;

/// <summary>
/// Model performance monitor with drift detection and automated rollback capabilities
/// </summary>
public class ModelPerformanceMonitor : IModelPerformanceMonitor
{
    private readonly ILogger<ModelPerformanceMonitor> _logger;
    private readonly IModelVersionManager _versionManager;
    private readonly ModelPerformanceOptions _options;
    private readonly ConcurrentDictionary<string, List<ModelPrediction>> _predictionHistory;
    private readonly SemaphoreSlim _semaphore;
    
    public ModelPerformanceMonitor(
        ILogger<ModelPerformanceMonitor> logger,
        IModelVersionManager versionManager,
        IOptions<ModelPerformanceOptions> options)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _versionManager = versionManager ?? throw new ArgumentNullException(nameof(versionManager));
        _options = options?.Value ?? new ModelPerformanceOptions();
        _predictionHistory = new ConcurrentDictionary<string, List<ModelPrediction>>();
        _semaphore = new SemaphoreSlim(1, 1);
    }

    public async Task RecordPredictionAsync(string modelHash, ModelPrediction prediction, CancellationToken cancellationToken = default)
    {
        _predictionHistory.AddOrUpdate(modelHash, 
            new List<ModelPrediction> { prediction },
            (key, existing) =>
            {
                existing.Add(prediction);
                
                // Keep only recent predictions to manage memory
                if (existing.Count > _options.MaxPredictionHistory)
                {
                    existing.RemoveRange(0, existing.Count - _options.MaxPredictionHistory);
                }
                
                return existing;
            });
            
        _logger.LogDebug("Recorded prediction for model {ModelHash} at {Timestamp}", 
            modelHash, prediction.Timestamp);
            
        // Check for degradation if we have enough data
        if (_predictionHistory[modelHash].Count >= _options.MinPredictionsForDegradationCheck)
        {
            var alert = await CheckForDegradationAsync(modelHash, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            if (alert != null)
            {
                _logger.LogWarning("Model degradation detected: {Description}", alert.Description);
                
                if (alert.SeverityScore >= _options.AutoRollbackThreshold)
                {
                    await TriggerSafeRollbackIfNeededAsync(modelHash, cancellationToken).ConfigureAwait(false);
                }
            }
        }
    }

    public async Task<ModelPerformanceReport> GetPerformanceReportAsync(string modelHash, TimeSpan lookbackPeriod, CancellationToken cancellationToken = default)
    {
        if (!_predictionHistory.TryGetValue(modelHash, out var predictions))
        {
            return new ModelPerformanceReport(
                modelHash, lookbackPeriod, 0, 0, 0, 0, 0, 0, 0, 
                new Dictionary<string, double>(), new List<PerformanceAlert>(), DateTime.UtcNow);
        }
        
        var cutoffTime = DateTime.UtcNow - lookbackPeriod;
        var recentPredictions = predictions
            .Where(p => p.Timestamp >= cutoffTime && p.ActualOutcome != null)
            .ToList();
            
        if (recentPredictions.Count == 0)
        {
            return new ModelPerformanceReport(
                modelHash, lookbackPeriod, 0, 0, 0, 0, 0, 0, 0, 
                new Dictionary<string, double>(), new List<PerformanceAlert>(), DateTime.UtcNow);
        }
        
        var metrics = CalculatePerformanceMetrics(recentPredictions);
        var regimePerformance = CalculateRegimePerformance(recentPredictions);
        var alerts = GeneratePerformanceAlerts(metrics);
        
        _logger.LogInformation("Generated performance report for model {ModelHash}: Accuracy={Accuracy:F3}, Sharpe={SharpeRatio:F2}", 
            modelHash, metrics.Accuracy, metrics.SharpeRatio);
        
        return new ModelPerformanceReport(
            modelHash, lookbackPeriod, recentPredictions.Count,
            metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1Score,
            metrics.SharpeRatio, metrics.MaxDrawdown, regimePerformance, alerts, DateTime.UtcNow);
    }

    public async Task<ModelDegradationAlert?> CheckForDegradationAsync(string modelHash, CancellationToken cancellationToken = default)
    {
        var currentReport = await GetPerformanceReportAsync(modelHash, _options.DegradationLookbackPeriod, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        
        if (currentReport.TotalPredictions < _options.MinPredictionsForDegradationCheck)
        {
            return null;
        }
        
        // Compare against baseline performance from model metadata
        var modelMetadata = await _versionManager.LoadModelAsync(modelHash, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        if (modelMetadata?.Performance == null)
        {
            return null;
        }
        
        var baseline = modelMetadata.Performance;
        
        // Check for significant degradation in key metrics
        var accuracyDrop = (baseline.Accuracy - currentReport.Accuracy) / baseline.Accuracy;
        var sharpeDrop = (baseline.SharpeRatio - currentReport.SharpeRatio) / Math.Abs(baseline.SharpeRatio);
        var drawdownIncrease = (currentReport.MaxDrawdown - baseline.MaxDrawdown) / baseline.MaxDrawdown;
        
        if (accuracyDrop > _options.AccuracyDegradationThreshold)
        {
            return new ModelDegradationAlert(
                modelHash, DegradationType.AccuracyDrop, currentReport.Accuracy, baseline.Accuracy,
                accuracyDrop, DateTime.UtcNow, 
                $"Accuracy dropped {accuracyDrop:P2} from baseline {baseline.Accuracy:F3} to {currentReport.Accuracy:F3}");
        }
        
        if (sharpeDrop > _options.SharpeRatioDegradationThreshold)
        {
            return new ModelDegradationAlert(
                modelHash, DegradationType.SharpeRatioDrop, currentReport.SharpeRatio, baseline.SharpeRatio,
                sharpeDrop, DateTime.UtcNow,
                $"Sharpe ratio dropped {sharpeDrop:P2} from baseline {baseline.SharpeRatio:F2} to {currentReport.SharpeRatio:F2}");
        }
        
        if (drawdownIncrease > _options.MaxDrawdownIncreaseThreshold)
        {
            return new ModelDegradationAlert(
                modelHash, DegradationType.DrawdownIncrease, currentReport.MaxDrawdown, baseline.MaxDrawdown,
                drawdownIncrease, DateTime.UtcNow,
                $"Max drawdown increased {drawdownIncrease:P2} from baseline {baseline.MaxDrawdown:F2} to {currentReport.MaxDrawdown:F2}");
        }
        
        return null;
    }

    public async Task<bool> TriggerSafeRollbackIfNeededAsync(string currentModelHash, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Find the previous rollback candidate
            var models = await _versionManager.ListModelsAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            var rollbackCandidate = models
                .Where(m => m.Status == "rollback")
                .OrderByDescending(m => m.CreatedAt)
                .FirstOrDefault();
                
            if (rollbackCandidate == null)
            {
                _logger.LogWarning("No rollback candidate available for model {ModelHash}", currentModelHash);
                return false;
            }
            
            // Compare performance
            var comparison = await CompareModelsAsync(rollbackCandidate.Hash, currentModelHash, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            
            if (comparison.ShouldRollback)
            {
                await _versionManager.SetActiveModelAsync(rollbackCandidate.Hash, false, cancellationToken).ConfigureAwait(false);
                
                _logger.LogWarning("Triggered automatic rollback from {CurrentModel} to {RollbackModel} due to performance degradation", 
                    currentModelHash, rollbackCandidate.Hash);
                    
                return true;
            }
            
            return false;
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async Task<ModelComparisonReport> CompareModelsAsync(string baselineHash, string candidateHash, CancellationToken cancellationToken = default)
    {
        var baselineReport = await GetPerformanceReportAsync(baselineHash, _options.ComparisonPeriod, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        var candidateReport = await GetPerformanceReportAsync(candidateHash, _options.ComparisonPeriod, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
        
        var comparisons = new Dictionary<string, ModelMetricComparison>
        {
            ["Accuracy"] = CompareMetric("Accuracy", baselineReport.Accuracy, candidateReport.Accuracy),
            ["Precision"] = CompareMetric("Precision", baselineReport.Precision, candidateReport.Precision),
            ["Recall"] = CompareMetric("Recall", baselineReport.Recall, candidateReport.Recall),
            ["F1Score"] = CompareMetric("F1Score", baselineReport.F1Score, candidateReport.F1Score),
            ["SharpeRatio"] = CompareMetric("SharpeRatio", baselineReport.SharpeRatio, candidateReport.SharpeRatio),
            ["MaxDrawdown"] = CompareMetric("MaxDrawdown", baselineReport.MaxDrawdown, candidateReport.MaxDrawdown, lowerIsBetter: true)
        };
        
        // Determine if rollback is recommended
        var shouldRollback = ShouldRecommendRollback(comparisons);
        var recommendation = GenerateRecommendation(comparisons, shouldRollback);
        
        return new ModelComparisonReport(
            baselineHash, candidateHash, _options.ComparisonPeriod, comparisons,
            shouldRollback, recommendation, DateTime.UtcNow);
    }
    
    private ModelMetricComparison CompareMetric(string name, double baseline, double candidate, bool lowerIsBetter = false)
    {
        var percentageChange = baseline == 0 ? 0 : (candidate - baseline) / Math.Abs(baseline);
        var isSignificant = Math.Abs(percentageChange) > _options.SignificanceThreshold;
        
        // Simplified statistical significance (in practice, would use proper statistical tests)
        var pValue = isSignificant ? 0.01 : 0.5;
        
        return new ModelMetricComparison(name, baseline, candidate, percentageChange, isSignificant, pValue);
    }
    
    private bool ShouldRecommendRollback(Dictionary<string, ModelMetricComparison> comparisons)
    {
        var significantDegradations = 0;
        var significantImprovements = 0;
        
        foreach (var comparison in comparisons.Values)
        {
            if (!comparison.IsSignificantDifference) continue;
            
            if (comparison.MetricName == "MaxDrawdown")
            {
                // For drawdown, lower is better
                if (comparison.PercentageChange > 0) significantDegradations++;
                else significantImprovements++;
            }
            else
            {
                // For other metrics, higher is better
                if (comparison.PercentageChange < 0) significantDegradations++;
                else significantImprovements++;
            }
        }
        
        return significantDegradations > significantImprovements;
    }
    
    private string GenerateRecommendation(Dictionary<string, ModelMetricComparison> comparisons, bool shouldRollback)
    {
        if (shouldRollback)
        {
            return "Recommend rollback: candidate model shows significant performance degradation";
        }
        
        var improvements = comparisons.Values.Count(c => c.IsSignificantDifference && 
            (c.MetricName == "MaxDrawdown" ? c.PercentageChange < 0 : c.PercentageChange > 0));
            
        return improvements > 0 
            ? $"Candidate model shows {improvements} significant improvements, recommend keeping"
            : "No significant performance differences detected";
    }
    
    private PerformanceMetrics CalculatePerformanceMetrics(List<ModelPrediction> predictions)
    {
        // Simplified metrics calculation - in practice would use proper statistical methods
        var correctPredictions = predictions.Count(p => PredictionIsCorrect(p));
        var accuracy = (double)correctPredictions / predictions.Count;
        
        return new PerformanceMetrics
        {
            Accuracy = accuracy,
            Precision = accuracy, // Simplified
            Recall = accuracy,    // Simplified
            F1Score = accuracy,   // Simplified
            SharpeRatio = CalculateSharpeRatio(predictions),
            MaxDrawdown = CalculateMaxDrawdown(predictions)
        };
    }
    
    private Dictionary<string, double> CalculateRegimePerformance(List<ModelPrediction> predictions)
    {
        return predictions
            .Where(p => !string.IsNullOrEmpty(p.MarketRegime))
            .GroupBy(p => p.MarketRegime!)
            .ToDictionary(
                g => g.Key,
                g => (double)g.Count(p => PredictionIsCorrect(p)) / g.Count()
            );
    }
    
    private List<PerformanceAlert> GeneratePerformanceAlerts(PerformanceMetrics metrics)
    {
        var alerts = new List<PerformanceAlert>();
        
        if (metrics.Accuracy < _options.AccuracyWarningThreshold)
        {
            alerts.Add(new PerformanceAlert(AlertSeverity.Warning, "Accuracy", 
                metrics.Accuracy, _options.AccuracyWarningThreshold, 
                $"Accuracy {metrics.Accuracy:F3} below warning threshold"));
        }
        
        if (metrics.SharpeRatio < _options.SharpeRatioWarningThreshold)
        {
            alerts.Add(new PerformanceAlert(AlertSeverity.Warning, "SharpeRatio", 
                metrics.SharpeRatio, _options.SharpeRatioWarningThreshold, 
                $"Sharpe ratio {metrics.SharpeRatio:F2} below warning threshold"));
        }
        
        return alerts;
    }
    
    private bool PredictionIsCorrect(ModelPrediction prediction)
    {
        // Simplified correctness check - in practice would depend on prediction type
        return prediction.Prediction?.ToString() == prediction.ActualOutcome?.ToString();
    }
    
    private double CalculateSharpeRatio(List<ModelPrediction> predictions)
    {
        // Simplified Sharpe ratio calculation
        if (predictions.Count < 2) return 0;
        
        var returns = predictions.Select(p => GetPredictionReturn(p)).Where(r => r != 0).ToList();
        if (returns.Count == 0) return 0;
        
        var avgReturn = returns.Average();
        var stdDev = Math.Sqrt(returns.Sum(r => Math.Pow(r - avgReturn, 2)) / returns.Count);
        
        return stdDev == 0 ? 0 : avgReturn / stdDev;
    }
    
    private double CalculateMaxDrawdown(List<ModelPrediction> predictions)
    {
        // Simplified max drawdown calculation
        var cumulativeReturns = new List<double> { 0 };
        var runningReturn = 0.0;
        
        foreach (var prediction in predictions)
        {
            runningReturn += GetPredictionReturn(prediction);
            cumulativeReturns.Add(runningReturn);
        }
        
        var maxDrawdown = 0.0;
        var peak = cumulativeReturns[0];
        
        foreach (var value in cumulativeReturns)
        {
            if (value > peak) peak = value;
            var drawdown = peak - value;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        }
        
        return maxDrawdown;
    }
    
    private double GetPredictionReturn(ModelPrediction prediction)
    {
        // Simplified return calculation - would depend on actual prediction/outcome format
        if (prediction.ActualOutcome is double outcome && prediction.Prediction is double pred)
        {
            return Math.Sign(pred) == Math.Sign(outcome) ? Math.Abs(outcome) : -Math.Abs(outcome);
        }
        return 0;
    }
    
    private class PerformanceMetrics
    {
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double F1Score { get; set; }
        public double SharpeRatio { get; set; }
        public double MaxDrawdown { get; set; }
    }
}

/// <summary>
/// Configuration options for model performance monitoring
/// </summary>
public class ModelPerformanceOptions
{
    public int MaxPredictionHistory { get; set; } = 10000;
    public int MinPredictionsForDegradationCheck { get; set; } = 100;
    public TimeSpan DegradationLookbackPeriod { get; set; } = TimeSpan.FromDays(7);
    public TimeSpan ComparisonPeriod { get; set; } = TimeSpan.FromDays(14);
    public double AccuracyDegradationThreshold { get; set; } = 0.1; // 10%
    public double SharpeRatioDegradationThreshold { get; set; } = 0.2; // 20%
    public double MaxDrawdownIncreaseThreshold { get; set; } = 0.5; // 50%
    public double AutoRollbackThreshold { get; set; } = 0.3; // 30% degradation triggers rollback
    public double AccuracyWarningThreshold { get; set; } = 0.6;
    public double SharpeRatioWarningThreshold { get; set; } = 0.5;
    public double SignificanceThreshold { get; set; } = 0.05; // 5% change considered significant
}