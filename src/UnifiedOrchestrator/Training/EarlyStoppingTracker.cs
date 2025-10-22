using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Early Stopping Tracker - Monitors validation performance during training and stops
/// when validation metrics plateau. Prevents overfitting by detecting when model
/// performance on validation set stops improving.
/// 
/// Features:
/// - Tracks validation metric (Sharpe ratio or win rate) after each epoch
/// - Saves checkpoint when validation metric improves
/// - Counts epochs without improvement (patience counter)
/// - Stops training early when patience exhausted
/// - Loads best checkpoint (not final epoch) when stopping
/// </summary>
public sealed class EarlyStoppingTracker
{
    private readonly ILogger<EarlyStoppingTracker> _logger;
    private readonly int _patience;
    private readonly string _checkpointDirectory;
    
    // Default patience: 10 epochs without improvement
    private const int DefaultPatience = 10;
    
    // Tracking state
    private double _bestValidationMetric = double.MinValue;
    private int _epochsWithoutImprovement = 0;
    private int _bestEpoch = 0;
    private string? _bestCheckpointPath;

    public EarlyStoppingTracker(
        ILogger<EarlyStoppingTracker> logger,
        string checkpointDirectory,
        int patience = DefaultPatience)
    {
        _logger = logger;
        _patience = patience;
        _checkpointDirectory = checkpointDirectory;
        
        Directory.CreateDirectory(_checkpointDirectory);
    }

    /// <summary>
    /// Check if training should stop based on validation performance
    /// </summary>
    /// <param name="validationMetric">Validation metric (e.g., Sharpe ratio, win rate)</param>
    /// <param name="currentEpoch">Current training epoch</param>
    /// <param name="componentName">Name of component being trained</param>
    /// <param name="checkpointCallback">Callback to save checkpoint when metric improves</param>
    /// <returns>True if training should stop, false to continue</returns>
    public bool ShouldStop(
        double validationMetric,
        int currentEpoch,
        string componentName,
        Func<string, Task>? checkpointCallback = null)
    {
        var improved = validationMetric > _bestValidationMetric;
        
        if (improved)
        {
            // Validation improved - save checkpoint
            var improvement = validationMetric - _bestValidationMetric;
            _bestValidationMetric = validationMetric;
            _bestEpoch = currentEpoch;
            _epochsWithoutImprovement = 0;
            
            // Save checkpoint if callback provided
            if (checkpointCallback != null)
            {
                _bestCheckpointPath = Path.Combine(_checkpointDirectory, $"{componentName}_epoch_{currentEpoch}.ckpt");
                _ = checkpointCallback(_bestCheckpointPath); // Fire and forget
            }
            
            _logger.LogInformation(
                "[EARLY-STOP] {Component}: Validation improved by {Improvement:F3} at epoch {Epoch} (new best: {Metric:F3})",
                componentName, improvement, currentEpoch, validationMetric);
            
            return false; // Continue training
        }
        else
        {
            // No improvement
            _epochsWithoutImprovement++;
            
            if (_epochsWithoutImprovement >= _patience)
            {
                // Patience exhausted - stop training
                _logger.LogInformation(
                    "[EARLY-STOP] {Component}: STOPPING at epoch {CurrentEpoch}, best was epoch {BestEpoch} with metric {BestMetric:F3}",
                    componentName, currentEpoch, _bestEpoch, _bestValidationMetric);
                
                _logger.LogInformation(
                    "[EARLY-STOP] {Component}: Saved {SavedEpochs} epochs worth of training time, avoided overfitting",
                    componentName, currentEpoch - _bestEpoch);
                
                return true; // Stop training
            }
            
            _logger.LogDebug(
                "[EARLY-STOP] {Component}: No improvement at epoch {Epoch} ({Count}/{Patience} patience)",
                componentName, currentEpoch, _epochsWithoutImprovement, _patience);
            
            return false; // Continue training
        }
    }

    /// <summary>
    /// Get the best checkpoint path (to load best model after early stopping)
    /// </summary>
    public string? GetBestCheckpointPath()
    {
        return _bestCheckpointPath;
    }

    /// <summary>
    /// Get the best epoch number
    /// </summary>
    public int GetBestEpoch()
    {
        return _bestEpoch;
    }

    /// <summary>
    /// Get the best validation metric
    /// </summary>
    public double GetBestValidationMetric()
    {
        return _bestValidationMetric;
    }

    /// <summary>
    /// Reset tracker for new component training
    /// </summary>
    public void Reset()
    {
        _bestValidationMetric = double.MinValue;
        _epochsWithoutImprovement = 0;
        _bestEpoch = 0;
        _bestCheckpointPath = null;
    }

    /// <summary>
    /// Get early stopping statistics for logging
    /// </summary>
    public EarlyStoppingStats GetStats()
    {
        return new EarlyStoppingStats
        {
            BestEpoch = _bestEpoch,
            BestValidationMetric = _bestValidationMetric,
            EpochsWithoutImprovement = _epochsWithoutImprovement,
            BestCheckpointPath = _bestCheckpointPath
        };
    }
}

/// <summary>
/// Early stopping statistics
/// </summary>
public sealed class EarlyStoppingStats
{
    public int BestEpoch { get; init; }
    public double BestValidationMetric { get; init; }
    public int EpochsWithoutImprovement { get; init; }
    public string? BestCheckpointPath { get; init; }
}
