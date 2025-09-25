using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using System.Text.Json;
using System.Collections.Concurrent;
using System.Globalization;

namespace BotCore.Services;

/// <summary>
/// Production-grade feedback service that creates automated learning loops
/// Monitors trading performance and triggers model retraining when needed
/// </summary>
public class TradingFeedbackService : BackgroundService
{
    private readonly ILogger<TradingFeedbackService> _logger;
    private readonly CloudModelSynchronizationService _cloudSync;
    private readonly ModelEnsembleService _ensemble;
    private readonly ConcurrentQueue<TradingOutcome> _feedbackQueue = new();
    private readonly ConcurrentDictionary<string, PerformanceMetrics> _performanceMetrics = new();
    
    // Configuration
    private readonly TimeSpan _processingInterval = TimeSpan.FromMinutes(5);
    private readonly int _minFeedbackSamples = 10;
    private readonly double _performanceThreshold = 0.6; // 60% accuracy threshold
    private readonly double _volatilityThreshold = 0.3; // 30% volatility threshold
    
    private readonly string _feedbackDataPath;
    private DateTime _lastRetrainingTrigger = DateTime.MinValue;
    
    public TradingFeedbackService(
        ILogger<TradingFeedbackService> logger,
        CloudModelSynchronizationService cloudSync,
        ModelEnsembleService ensemble)
    {
        _logger = logger;
        _cloudSync = cloudSync;
        _ensemble = ensemble;
        
        // Create feedback data directory
        _feedbackDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "feedback");
        Directory.CreateDirectory(_feedbackDataPath);
        
        _logger.LogInformation("🔄 [FEEDBACK] Service initialized - Threshold: {Threshold:P0}, Min samples: {MinSamples}", 
            _performanceThreshold, _minFeedbackSamples);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("🔄 [FEEDBACK] Background service started");
        
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await ProcessFeedbackQueue(stoppingToken).ConfigureAwait(false);
                await AnalyzePerformance(stoppingToken).ConfigureAwait(false);
                await CheckRetrainingTriggers(stoppingToken).ConfigureAwait(false);
                
                await Task.Delay(_processingInterval, stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🔄 [FEEDBACK] Error in background processing");
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken).ConfigureAwait(false);
            }
        }
        
        _logger.LogInformation("🔄 [FEEDBACK] Background service stopped");
    }

    /// <summary>
    /// Submit trading outcome for learning feedback
    /// Called by trading strategies after trade completion
    /// </summary>
    public void SubmitTradingOutcome(TradingOutcome outcome)
    {
        try
        {
            _feedbackQueue.Enqueue(outcome);
            
            _logger.LogDebug("🔄 [FEEDBACK] Outcome submitted: {Strategy} {Action} P&L: {PnL:C2} (accuracy: {Accuracy:P1})", 
                outcome.Strategy, outcome.Action, outcome.RealizedPnL, outcome.PredictionAccuracy);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🔄 [FEEDBACK] Error submitting trading outcome");
        }
    }

    /// <summary>
    /// Submit model prediction feedback
    /// Called when we can evaluate prediction accuracy
    /// </summary>
    public void SubmitPredictionFeedback(PredictionFeedback feedback)
    {
        try
        {
            var outcome = new TradingOutcome
            {
                Timestamp = feedback.Timestamp,
                Strategy = feedback.ModelName,
                Action = feedback.PredictedAction,
                Symbol = feedback.Symbol,
                PredictionAccuracy = feedback.ActualAccuracy,
                RealizedPnL = feedback.ImpactOnPnL,
                MarketConditions = feedback.MarketContext,
                ModelConfidence = feedback.OriginalConfidence,
                ActualOutcome = feedback.ActualOutcome
            };
            
            // Populate the readonly dictionary
            foreach (var kvp in feedback.TradingContext)
            {
                outcome.TradingContext[kvp.Key] = kvp.Value;
            }
            
            SubmitTradingOutcome(outcome);
            
            // Update model performance in ensemble
            _ensemble.UpdateModelPerformance(feedback.ModelName, feedback.ActualAccuracy, "prediction_feedback");
            
            _logger.LogDebug("🔄 [FEEDBACK] Prediction feedback submitted for {ModelName}: {Accuracy:P1}", 
                feedback.ModelName, feedback.ActualAccuracy);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🔄 [FEEDBACK] Error submitting prediction feedback");
        }
    }

    /// <summary>
    /// Process queued feedback and update performance metrics
    /// </summary>
    private async Task ProcessFeedbackQueue(CancellationToken cancellationToken)
    {
        var processedCount = 0;
        var outcomes = new List<TradingOutcome>();
        
        // Dequeue all pending outcomes
        while (_feedbackQueue.TryDequeue(out var outcome))
        {
            outcomes.Add(outcome);
            processedCount++;
            
            // Update performance metrics
            UpdatePerformanceMetrics(outcome);
        }
        
        if (processedCount > 0)
        {
            _logger.LogDebug("🔄 [FEEDBACK] Processed {Count} feedback items", processedCount);
            
            // Save feedback data to disk for analysis
            await SaveFeedbackDataAsync(outcomes, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Update performance metrics for strategies and models
    /// </summary>
    private void UpdatePerformanceMetrics(TradingOutcome outcome)
    {
        var key = $"{outcome.Strategy}_{outcome.Symbol}";
        
        if (!_performanceMetrics.TryGetValue(key, out var metrics))
        {
            metrics = new PerformanceMetrics
            {
                Strategy = outcome.Strategy,
                Symbol = outcome.Symbol,
                FirstTrade = outcome.Timestamp,
                LastUpdate = outcome.Timestamp
            };
            _performanceMetrics[key] = metrics;
        }
        
        // Update metrics
        metrics.TotalTrades++;
        metrics.LastUpdate = outcome.Timestamp;
        metrics.TotalPnL += outcome.RealizedPnL;
        
        if (outcome.PredictionAccuracy >= 0.5)
        {
            metrics.SuccessfulTrades++;
        }
        
        if (outcome.RealizedPnL > 0)
        {
            metrics.ProfitableTrades++;
        }
        
        // Calculate rolling accuracy
        metrics.AccuracyHistory.Add(outcome.PredictionAccuracy);
        if (metrics.AccuracyHistory.Count > 100) // Keep last 100 trades
        {
            metrics.AccuracyHistory.RemoveAt(0);
        }
        
        // Phase 6A: Exception Guards - Add .Any() check before .Average()
        metrics.AverageAccuracy = metrics.AccuracyHistory.Any() ? metrics.AccuracyHistory.Average() : 0.0;
        metrics.WinRate = (double)metrics.ProfitableTrades / metrics.TotalTrades;
        
        // Calculate volatility of predictions
        if (metrics.AccuracyHistory.Count >= 10)
        {
            // Phase 6A: Exception Guards - Add .Any() check before .Average()
            var variance = metrics.AccuracyHistory.Any() ? 
                metrics.AccuracyHistory.Select(x => Math.Pow(x - metrics.AverageAccuracy, 2)).Average() : 0.0;
            metrics.AccuracyVolatility = Math.Sqrt(variance);
        }
    }

    /// <summary>
    /// Analyze overall performance and identify issues
    /// </summary>
    private async Task AnalyzePerformance(CancellationToken cancellationToken)
    {
        var performanceIssues = new List<PerformanceIssue>();
        
        foreach (var kvp in _performanceMetrics)
        {
            var metrics = kvp.Value;
            
            // Check if we have enough samples
            if (metrics.TotalTrades < _minFeedbackSamples)
                continue;
            
            // Check for performance degradation
            if (metrics.AverageAccuracy < _performanceThreshold)
            {
                performanceIssues.Add(new PerformanceIssue
                {
                    Type = "low_accuracy",
                    Strategy = metrics.Strategy,
                    Symbol = metrics.Symbol,
                    Severity = CalculateSeverity(metrics.AverageAccuracy, _performanceThreshold),
                    Description = $"Accuracy {metrics.AverageAccuracy:P1} below threshold {_performanceThreshold:P1}",
                    Metrics = metrics
                });
            }
            
            // Check for high volatility (unstable predictions)
            if (metrics.AccuracyVolatility > _volatilityThreshold)
            {
                performanceIssues.Add(new PerformanceIssue
                {
                    Type = "high_volatility",
                    Strategy = metrics.Strategy,
                    Symbol = metrics.Symbol,
                    Severity = CalculateSeverity(metrics.AccuracyVolatility, _volatilityThreshold, true),
                    Description = $"Prediction volatility {metrics.AccuracyVolatility:P1} above threshold {_volatilityThreshold:P1}",
                    Metrics = metrics
                });
            }
            
            // Check for recent performance drops
            if (metrics.AccuracyHistory.Count >= 20)
            {
                // Phase 6A: Exception Guards - Add .Any() checks before .Average()
                var recentHistory = metrics.AccuracyHistory.TakeLast(10).ToList();
                var previousHistory = metrics.AccuracyHistory.Take(10).ToList();
                
                var recent = recentHistory.Any() ? recentHistory.Average() : 0.0;
                var previous = previousHistory.Any() ? previousHistory.Average() : 0.0;
                
                if (recent < previous - 0.1) // 10% drop
                {
                    performanceIssues.Add(new PerformanceIssue
                    {
                        Type = "performance_drop",
                        Strategy = metrics.Strategy,
                        Symbol = metrics.Symbol,
                        Severity = CalculateSeverity(recent, previous),
                        Description = $"Recent accuracy {recent:P1} vs previous {previous:P1}",
                        Metrics = metrics
                    });
                }
            }
        }
        
        if (performanceIssues.Any())
        {
            _logger.LogWarning("🔄 [FEEDBACK] Detected {IssueCount} performance issues", performanceIssues.Count);
            
            foreach (var issue in performanceIssues)
            {
                _logger.LogWarning("🔄 [FEEDBACK] Issue: {Type} for {Strategy}:{Symbol} - {Description} (severity: {Severity})", 
                    issue.Type, issue.Strategy, issue.Symbol, issue.Description, issue.Severity);
            }
            
            // Save issues for analysis
            await SavePerformanceIssuesAsync(performanceIssues, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Check if retraining should be triggered
    /// </summary>
    private async Task CheckRetrainingTriggers(CancellationToken cancellationToken)
    {
        // Don't retrain too frequently
        if (DateTime.UtcNow - _lastRetrainingTrigger < TimeSpan.FromHours(2))
            return;
        
        // Check if any strategy consistently underperforms
        var underperformingStrategies = _performanceMetrics.Values
            .Where(m => m.TotalTrades >= _minFeedbackSamples && 
                       m.AverageAccuracy < _performanceThreshold)
            .GroupBy(m => m.Strategy)
            .Where(g => g.Count() >= 2) // Multiple symbols affected
            .Select(g => g.Key)
            .ToList();
        
        if (underperformingStrategies.Any())
        {
            _logger.LogWarning("🔄 [FEEDBACK] Triggering retraining for underperforming strategies: {Strategies}", 
                string.Join(", ", underperformingStrategies));
            
            await TriggerModelRetraining(underperformingStrategies, cancellationToken).ConfigureAwait(false);
            _lastRetrainingTrigger = DateTime.UtcNow;
        }
        
        // Check for overall model ensemble degradation
        var ensembleStats = _ensemble.GetModelPerformanceStats();
        
        // Phase 6A: Exception Guards - Add .Any() check before .Average()
        var validStats = ensembleStats.Values.Where(s => s.PredictionCount >= 10).ToList();
        var overallAccuracy = validStats.Any() ? validStats.Average(s => s.AccuracyScore) : 0.0;
        
        if (overallAccuracy < _performanceThreshold && ensembleStats.Count >= 3)
        {
            _logger.LogWarning("🔄 [FEEDBACK] Triggering ensemble retraining - Overall accuracy: {Accuracy:P1}", overallAccuracy);
            
            await TriggerEnsembleRetraining(cancellationToken).ConfigureAwait(false);
            _lastRetrainingTrigger = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Trigger model retraining via cloud synchronization
    /// </summary>
    private async Task TriggerModelRetraining(List<string> strategies, CancellationToken cancellationToken)
    {
        try
        {
            var retrainingRequest = new ModelRetrainingRequest
            {
                Timestamp = DateTime.UtcNow,
                Reason = "performance_degradation",
                TriggerThreshold = _performanceThreshold,
                MinSamples = _minFeedbackSamples
            };
            
            // Populate readonly collections
            foreach (var strategy in strategies)
            {
                retrainingRequest.Strategies.Add(strategy);
            }
            
            var filteredMetrics = _performanceMetrics.Values
                .Where(m => strategies.Contains(m.Strategy))
                .ToList();
            foreach (var metric in filteredMetrics)
            {
                retrainingRequest.PerformanceMetrics.Add(metric);
            }
            
            // Save retraining request
            var requestPath = Path.Combine(_feedbackDataPath, $"retraining_request_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            var requestJson = JsonSerializer.Serialize(retrainingRequest, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(requestPath, requestJson, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("🔄 [FEEDBACK] Retraining request saved: {Path}", requestPath);
            
            // Trigger cloud synchronization to check for new models
            // This will eventually trigger GitHub Actions for retraining
            await _cloudSync.SynchronizeModelsAsync(cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("🔄 [FEEDBACK] Cloud synchronization triggered for retraining");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🔄 [FEEDBACK] Error triggering model retraining");
        }
    }

    /// <summary>
    /// Trigger ensemble retraining
    /// </summary>
    private async Task TriggerEnsembleRetraining(CancellationToken cancellationToken)
    {
        try
        {
            var ensembleRequest = new EnsembleRetrainingRequest
            {
                Timestamp = DateTime.UtcNow,
                Reason = "ensemble_degradation",
                TriggerThreshold = _performanceThreshold
            };
            
            // Populate readonly collections
            var modelPerformanceStats = _ensemble.GetModelPerformanceStats();
            foreach (var kvp in modelPerformanceStats)
            {
                ensembleRequest.ModelPerformance[kvp.Key] = kvp.Value;
            }
            
            var overallPerformanceList = _performanceMetrics.Values.ToList();
            foreach (var metric in overallPerformanceList)
            {
                ensembleRequest.OverallPerformance.Add(metric);
            }
            
            // Save ensemble retraining request
            var requestPath = Path.Combine(_feedbackDataPath, $"ensemble_retraining_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            var requestJson = JsonSerializer.Serialize(ensembleRequest, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(requestPath, requestJson, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("🔄 [FEEDBACK] Ensemble retraining request saved: {Path}", requestPath);
            
            // Force cloud model synchronization
            await _cloudSync.SynchronizeModelsAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🔄 [FEEDBACK] Error triggering ensemble retraining");
        }
    }

    /// <summary>
    /// Save feedback data for analysis
    /// </summary>
    private async Task SaveFeedbackDataAsync(List<TradingOutcome> outcomes, CancellationToken cancellationToken)
    {
        try
        {
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            var filePath = Path.Combine(_feedbackDataPath, $"feedback_{timestamp}.json");
            
            var data = new
            {
                timestamp = DateTime.UtcNow,
                count = outcomes.Count,
                outcomes = outcomes
            };
            
            var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(filePath, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogDebug("🔄 [FEEDBACK] Saved {Count} outcomes to {Path}", outcomes.Count, filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🔄 [FEEDBACK] Error saving feedback data");
        }
    }

    /// <summary>
    /// Save performance issues for analysis
    /// </summary>
    private async Task SavePerformanceIssuesAsync(List<PerformanceIssue> issues, CancellationToken cancellationToken)
    {
        try
        {
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            var filePath = Path.Combine(_feedbackDataPath, $"performance_issues_{timestamp}.json");
            
            var data = new
            {
                timestamp = DateTime.UtcNow,
                count = issues.Count,
                issues = issues
            };
            
            var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(filePath, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogDebug("🔄 [FEEDBACK] Saved {Count} performance issues to {Path}", issues.Count, filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🔄 [FEEDBACK] Error saving performance issues");
        }
    }

    /// <summary>
    /// Calculate severity based on deviation from threshold
    /// </summary>
    private string CalculateSeverity(double actual, double threshold, bool higherIsBad = false)
    {
        var deviation = higherIsBad ? (actual - threshold) / threshold : (threshold - actual) / threshold;
        
        return deviation switch
        {
            > 0.5 => "critical",
            > 0.3 => "high",
            > 0.1 => "medium",
            _ => "low"
        };
    }

    /// <summary>
    /// Get current performance summary
    /// </summary>
    public PerformanceSummary GetPerformanceSummary()
    {
        var summary = new PerformanceSummary
        {
            Timestamp = DateTime.UtcNow,
            TotalStrategies = _performanceMetrics.Values.Select(m => m.Strategy).Distinct().Count(),
            TotalTrades = _performanceMetrics.Values.Sum(m => m.TotalTrades),
            // Phase 6A: Exception Guards - Add .Any() check before .Average()
            OverallAccuracy = _performanceMetrics.Values
                .Where(m => m.TotalTrades >= _minFeedbackSamples)
                .Any() ? _performanceMetrics.Values
                .Where(m => m.TotalTrades >= _minFeedbackSamples)
                .Average(m => m.AverageAccuracy) : 0.0,
            OverallPnL = _performanceMetrics.Values.Sum(m => m.TotalPnL),
            LastRetrainingTrigger = _lastRetrainingTrigger
        };
        
        // Populate readonly collections
        var strategyMetrics = _performanceMetrics.Values.ToList();
        foreach (var metric in strategyMetrics)
        {
            summary.StrategyMetrics.Add(metric);
        }
        
        var modelMetrics = _ensemble.GetModelPerformanceStats();
        foreach (var kvp in modelMetrics)
        {
            summary.ModelMetrics[kvp.Key] = kvp.Value;
        }
        
        return summary;
    }
}

#region Data Models

public class TradingOutcome
{
    public DateTime Timestamp { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public double PredictionAccuracy { get; set; }
    public decimal RealizedPnL { get; set; }
    public string MarketConditions { get; set; } = string.Empty;
    public double ModelConfidence { get; set; }
    public string ActualOutcome { get; set; } = string.Empty;
    public Dictionary<string, object> TradingContext { get; } = new();
}

public class PredictionFeedback
{
    public DateTime Timestamp { get; set; }
    public string ModelName { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string PredictedAction { get; set; } = string.Empty;
    public string ActualOutcome { get; set; } = string.Empty;
    public double ActualAccuracy { get; set; }
    public decimal ImpactOnPnL { get; set; }
    public double OriginalConfidence { get; set; }
    public string MarketContext { get; set; } = string.Empty;
    public Dictionary<string, object> TradingContext { get; } = new();
}

public class PerformanceMetrics
{
    public string Strategy { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public int TotalTrades { get; set; }
    public int SuccessfulTrades { get; set; }
    public int ProfitableTrades { get; set; }
    public double AverageAccuracy { get; set; }
    public double WinRate { get; set; }
    public double AccuracyVolatility { get; set; }
    public decimal TotalPnL { get; set; }
    public List<double> AccuracyHistory { get; } = new();
    public DateTime FirstTrade { get; set; }
    public DateTime LastUpdate { get; set; }
}

public class PerformanceIssue
{
    public string Type { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Severity { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public PerformanceMetrics Metrics { get; set; } = null!;
}

public class ModelRetrainingRequest
{
    public DateTime Timestamp { get; set; }
    public List<string> Strategies { get; } = new();
    public string Reason { get; set; } = string.Empty;
    public List<PerformanceMetrics> PerformanceMetrics { get; } = new();
    public double TriggerThreshold { get; set; }
    public int MinSamples { get; set; }
}

public class EnsembleRetrainingRequest
{
    public DateTime Timestamp { get; set; }
    public string Reason { get; set; } = string.Empty;
    public Dictionary<string, ModelPerformance> ModelPerformance { get; } = new();
    public List<PerformanceMetrics> OverallPerformance { get; } = new();
    public double TriggerThreshold { get; set; }
}

public class PerformanceSummary
{
    public DateTime Timestamp { get; set; }
    public int TotalStrategies { get; set; }
    public int TotalTrades { get; set; }
    public double OverallAccuracy { get; set; }
    public decimal OverallPnL { get; set; }
    public List<PerformanceMetrics> StrategyMetrics { get; } = new();
    public Dictionary<string, ModelPerformance> ModelMetrics { get; } = new();
    public DateTime LastRetrainingTrigger { get; set; }
}

#endregion