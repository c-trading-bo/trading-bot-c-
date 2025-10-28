using System;
using System.Collections.Generic;
using System.Linq;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Tracks training progress and calculates ETAs
/// Provides centralized progress state for rendering and monitoring
/// </summary>
public sealed class ProgressTracker
{
    private readonly object _lock = new();
    private readonly List<ComponentTiming> _componentTimings = new();
    
    /// <summary>
    /// Total number of components to train (25: Heavy=11, Medium=7, Light=7)
    /// </summary>
    public int TotalComponents { get; set; }

    /// <summary>
    /// Number of components completed so far
    /// </summary>
    public int CompletedComponents { get; private set; }

    /// <summary>
    /// Current training phase
    /// </summary>
    public string CurrentPhase { get; private set; } = "NotStarted";

    /// <summary>
    /// Name of component currently training
    /// </summary>
    public string? CurrentComponent { get; private set; }

    /// <summary>
    /// Current component progress (0.0 to 1.0)
    /// </summary>
    public double CurrentComponentProgress { get; private set; }

    /// <summary>
    /// Current epoch being processed
    /// </summary>
    public int CurrentEpoch { get; private set; }

    /// <summary>
    /// Total epochs for current component
    /// </summary>
    public int TotalEpochs { get; private set; }

    /// <summary>
    /// Current training loss value
    /// </summary>
    public double CurrentLoss { get; private set; }

    /// <summary>
    /// When training session started
    /// </summary>
    public DateTimeOffset StartTime { get; set; }

    /// <summary>
    /// Estimated completion time (calculated)
    /// </summary>
    public DateTimeOffset? EstimatedEndTime { get; private set; }

    /// <summary>
    /// Update progress for current component
    /// </summary>
    public void UpdateComponentProgress(
        string componentName,
        double progress,
        int currentEpoch = 0,
        int totalEpochs = 0,
        double currentLoss = 0.0)
    {
        lock (_lock)
        {
            CurrentComponent = componentName;
            CurrentComponentProgress = Math.Clamp(progress, 0.0, 1.0);
            CurrentEpoch = currentEpoch;
            TotalEpochs = totalEpochs;
            CurrentLoss = currentLoss;

            CalculateETA();
        }
    }

    /// <summary>
    /// Mark component as completed
    /// </summary>
    public void CompleteComponent(string componentName, TimeSpan timeTaken)
    {
        lock (_lock)
        {
            CompletedComponents++;
            CurrentComponentProgress = 1.0;
            
            // Record timing for ETA calculation
            _componentTimings.Add(new ComponentTiming
            {
                ComponentName = componentName,
                Phase = CurrentPhase,
                Duration = timeTaken
            });

            // Clear current component state
            CurrentComponent = null;
            CurrentComponentProgress = 0.0;
            CurrentEpoch = 0;
            TotalEpochs = 0;

            CalculateETA();
        }
    }

    /// <summary>
    /// Set current training phase
    /// </summary>
    public void SetPhase(string phase)
    {
        lock (_lock)
        {
            CurrentPhase = phase;
        }
    }

    /// <summary>
    /// Calculate estimated time remaining
    /// </summary>
    public void CalculateETA()
    {
        lock (_lock)
        {
            if (CompletedComponents == 0 || TotalComponents == 0)
            {
                EstimatedEndTime = null;
                return;
            }

            // Calculate average time per component based on completed components
            var avgTimePerComponent = TimeSpan.FromTicks(
                _componentTimings.Sum(t => t.Duration.Ticks) / _componentTimings.Count);

            // Adjust based on phase (heavy components take longer)
            var phaseMultiplier = CurrentPhase switch
            {
                "Heavy" => 1.5,
                "Medium" => 1.0,
                "Light" => 0.5,
                _ => 1.0
            };

            var adjustedAvgTime = TimeSpan.FromTicks((long)(avgTimePerComponent.Ticks * phaseMultiplier));

            // Calculate remaining time
            var remainingComponents = TotalComponents - CompletedComponents;
            var estimatedRemainingTime = TimeSpan.FromTicks(adjustedAvgTime.Ticks * remainingComponents);

            // Add 20% buffer for safety
            var bufferedTime = TimeSpan.FromTicks((long)(estimatedRemainingTime.Ticks * 1.2));

            // Account for current component progress
            if (CurrentComponentProgress > 0 && CurrentComponentProgress < 1.0)
            {
                var currentComponentRemaining = TimeSpan.FromTicks(
                    (long)(adjustedAvgTime.Ticks * (1.0 - CurrentComponentProgress)));
                bufferedTime = bufferedTime.Add(currentComponentRemaining);
            }

            EstimatedEndTime = DateTimeOffset.UtcNow.Add(bufferedTime);
        }
    }

    /// <summary>
    /// Get overall progress percentage (0-100)
    /// </summary>
    public double GetProgressPercentage()
    {
        lock (_lock)
        {
            if (TotalComponents == 0)
                return 0.0;

            var baseProgress = (double)CompletedComponents / TotalComponents;
            var currentComponentContribution = CurrentComponentProgress / TotalComponents;
            
            return Math.Clamp((baseProgress + currentComponentContribution) * 100.0, 0.0, 100.0);
        }
    }

    /// <summary>
    /// Generate ASCII progress bar string
    /// </summary>
    public string GetProgressBar(int width = 50)
    {
        lock (_lock)
        {
            var percentage = GetProgressPercentage();
            var filledWidth = (int)(width * percentage / 100.0);
            var emptyWidth = width - filledWidth;

            var filled = new string('█', filledWidth);
            var empty = new string('░', emptyWidth);

            return $"[{filled}{empty}]";
        }
    }

    /// <summary>
    /// Get human-readable ETA string (e.g., "2h 15m")
    /// </summary>
    public string GetFormattedETA()
    {
        lock (_lock)
        {
            if (!EstimatedEndTime.HasValue)
                return "Calculating...";

            var remaining = EstimatedEndTime.Value - DateTimeOffset.UtcNow;
            
            if (remaining.TotalSeconds < 0)
                return "Completing...";

            if (remaining.TotalHours >= 1)
            {
                var hours = (int)remaining.TotalHours;
                var minutes = remaining.Minutes;
                return $"{hours}h {minutes}m";
            }
            else if (remaining.TotalMinutes >= 1)
            {
                var minutes = (int)remaining.TotalMinutes;
                var seconds = remaining.Seconds;
                return $"{minutes}m {seconds}s";
            }
            else
            {
                var seconds = (int)remaining.TotalSeconds;
                return $"{seconds}s";
            }
        }
    }

    /// <summary>
    /// Get elapsed time since start
    /// </summary>
    public TimeSpan GetElapsedTime()
    {
        return DateTimeOffset.UtcNow - StartTime;
    }

    /// <summary>
    /// Get formatted elapsed time string
    /// </summary>
    public string GetFormattedElapsedTime()
    {
        var elapsed = GetElapsedTime();
        
        if (elapsed.TotalHours >= 1)
        {
            var hours = (int)elapsed.TotalHours;
            var minutes = elapsed.Minutes;
            return $"{hours}h {minutes}m";
        }
        else if (elapsed.TotalMinutes >= 1)
        {
            var minutes = (int)elapsed.TotalMinutes;
            var seconds = elapsed.Seconds;
            return $"{minutes}m {seconds}s";
        }
        else
        {
            var seconds = (int)elapsed.TotalSeconds;
            return $"{seconds}s";
        }
    }

    /// <summary>
    /// Get current status summary
    /// </summary>
    public ProgressSummary GetSummary()
    {
        lock (_lock)
        {
            return new ProgressSummary
            {
                TotalComponents = TotalComponents,
                CompletedComponents = CompletedComponents,
                RemainingComponents = TotalComponents - CompletedComponents,
                CurrentPhase = CurrentPhase,
                CurrentComponent = CurrentComponent,
                ProgressPercentage = GetProgressPercentage(),
                ElapsedTime = GetElapsedTime(),
                EstimatedTimeRemaining = EstimatedEndTime.HasValue 
                    ? EstimatedEndTime.Value - DateTimeOffset.UtcNow 
                    : null,
                CurrentEpoch = CurrentEpoch,
                TotalEpochs = TotalEpochs,
                CurrentLoss = CurrentLoss
            };
        }
    }
}

/// <summary>
/// Component timing record for ETA calculation
/// </summary>
internal sealed class ComponentTiming
{
    public string ComponentName { get; set; } = string.Empty;
    public string Phase { get; set; } = string.Empty;
    public TimeSpan Duration { get; set; }
}

/// <summary>
/// Progress summary snapshot
/// </summary>
public sealed class ProgressSummary
{
    public int TotalComponents { get; set; }
    public int CompletedComponents { get; set; }
    public int RemainingComponents { get; set; }
    public string CurrentPhase { get; set; } = string.Empty;
    public string? CurrentComponent { get; set; }
    public double ProgressPercentage { get; set; }
    public TimeSpan ElapsedTime { get; set; }
    public TimeSpan? EstimatedTimeRemaining { get; set; }
    public int CurrentEpoch { get; set; }
    public int TotalEpochs { get; set; }
    public double CurrentLoss { get; set; }
}
