using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Lab Mode Dashboard State Manager
/// Centralizes collection and management of dashboard metrics during training
/// Provides real-time updates for strategy performance, phase progress, and system resources
/// </summary>
public sealed class LabModeDashboardStateManager
{
    private readonly ILogger<LabModeDashboardStateManager> _logger;
    private readonly ConcurrentDictionary<string, StrategyTrainingMetrics> _strategyMetrics = new();
    private readonly ConcurrentQueue<ActivityLogEntry> _activityLog = new();
    private readonly object _stateLock = new();
    
    private LabModeDashboardState _currentState;
    private Process? _currentProcess;
    
    private const int MaxActivityLogEntries = 50;

    public LabModeDashboardStateManager(ILogger<LabModeDashboardStateManager> logger)
    {
        _logger = logger;
        _currentState = InitializeState();
        _currentProcess = Process.GetCurrentProcess();
    }

    /// <summary>
    /// Initialize a new training session
    /// </summary>
    public void InitializeSession(string sessionId, int totalComponents)
    {
        lock (_stateLock)
        {
            _currentState = InitializeState();
            _currentState.SessionId = sessionId;
            _currentState.SessionStartTime = DateTimeOffset.UtcNow;
            _currentState.TotalComponents = totalComponents;
            _currentState.ComponentsRemaining = totalComponents;
            
            // Initialize strategies (S2, S3, S6, S11)
            var strategies = new[] { "S2", "S3", "S6", "S11" };
            foreach (var strategy in strategies)
            {
                var metrics = new StrategyTrainingMetrics
                {
                    StrategyName = strategy,
                    Status = TrainingPhaseStatus.Pending
                };
                _strategyMetrics[strategy] = metrics;
                _currentState.StrategyMetrics.Add(metrics);
            }
            
            _logger.LogInformation("[DASHBOARD] Session initialized: {SessionId}, Components: {Total}", 
                sessionId, totalComponents);
        }
    }

    /// <summary>
    /// Update phase information
    /// </summary>
    public void UpdatePhase(string phaseName, int totalComponentsInPhase)
    {
        lock (_stateLock)
        {
            _currentState.CurrentPhase = phaseName;
            
            var phaseDetails = GetPhaseDetails(phaseName);
            if (phaseDetails != null)
            {
                phaseDetails.PhaseName = phaseName;
                phaseDetails.Status = TrainingPhaseStatus.InProgress;
                phaseDetails.TotalComponents = totalComponentsInPhase;
            }
            
            _logger.LogInformation("[DASHBOARD] Phase updated: {Phase}, Components: {Total}", 
                phaseName, totalComponentsInPhase);
        }
    }

    /// <summary>
    /// Mark phase as complete
    /// </summary>
    public void CompletePhase(string phaseName, TimeSpan duration, int succeeded, int failed)
    {
        lock (_stateLock)
        {
            var phaseDetails = GetPhaseDetails(phaseName);
            if (phaseDetails != null)
            {
                phaseDetails.Status = TrainingPhaseStatus.Complete;
                phaseDetails.Duration = duration;
                phaseDetails.CompletedComponents = succeeded;
                phaseDetails.FailedComponents = failed;
            }
            
            LogActivity("info", $"{phaseName}Phase", 
                $"✓ {phaseName} phase complete - {succeeded}/{succeeded + failed} succeeded");
        }
    }

    /// <summary>
    /// Update component training progress
    /// </summary>
    public void UpdateComponentProgress(string componentName, string phase, int currentEpoch, int totalEpochs, double currentLoss, double progress)
    {
        lock (_stateLock)
        {
            _currentState.CurrentComponent = new ComponentTrainingDetails
            {
                ComponentName = componentName,
                Phase = phase,
                EpochsCompleted = currentEpoch,
                TotalEpochs = totalEpochs,
                CurrentLoss = currentLoss,
                ProgressPercentage = progress * 100.0
            };
            
            // Update phase-specific component list
            var phaseDetails = GetPhaseDetails(phase);
            if (phaseDetails != null)
            {
                var component = phaseDetails.Components.FirstOrDefault(c => c.ComponentName == componentName);
                if (component == null)
                {
                    component = new ComponentSummary { ComponentName = componentName };
                    phaseDetails.Components.Add(component);
                }
                
                component.EpochsCompleted = currentEpoch;
                component.TotalEpochs = totalEpochs;
                component.FinalLoss = currentLoss;
                component.ProgressPercentage = progress * 100.0;
                component.Status = "InProgress";
            }
        }
    }

    /// <summary>
    /// Mark component as complete
    /// </summary>
    public void CompleteComponent(string componentName, string phase, int epochsCompleted, double finalLoss, Dictionary<string, string>? metrics = null)
    {
        lock (_stateLock)
        {
            _currentState.ComponentsCompleted++;
            _currentState.ComponentsRemaining = _currentState.TotalComponents - _currentState.ComponentsCompleted;
            _currentState.OverallProgress = (_currentState.ComponentsCompleted / (double)_currentState.TotalComponents) * 100.0;
            
            // Update phase-specific component
            var phaseDetails = GetPhaseDetails(phase);
            if (phaseDetails != null)
            {
                var component = phaseDetails.Components.FirstOrDefault(c => c.ComponentName == componentName);
                if (component == null)
                {
                    component = new ComponentSummary { ComponentName = componentName };
                    phaseDetails.Components.Add(component);
                }
                
                component.EpochsCompleted = epochsCompleted;
                component.TotalEpochs = epochsCompleted;
                component.FinalLoss = finalLoss;
                component.ProgressPercentage = 100.0;
                component.Status = "Complete";
                
                if (metrics != null)
                {
                    component.Metrics = metrics;
                }
                
                phaseDetails.CompletedComponents++;
            }
            
            _currentState.CurrentComponent = null;
            
            LogActivity("info", componentName, 
                $"✓ Training complete - {epochsCompleted} epochs, loss: {finalLoss:F4}");
        }
    }

    /// <summary>
    /// Update strategy training metrics
    /// </summary>
    public void UpdateStrategyMetrics(string strategyName, decimal winRate, decimal totalPnL, decimal totalWon, decimal totalLost, int winningTrades, int losingTrades)
    {
        if (_strategyMetrics.TryGetValue(strategyName, out var metrics))
        {
            lock (_stateLock)
            {
                metrics.WinRate = winRate;
                metrics.TotalPnL = totalPnL;
                metrics.TotalWon = totalWon;
                metrics.TotalLost = totalLost;
                metrics.WinningTrades = winningTrades;
                metrics.LosingTrades = losingTrades;
                metrics.TotalTrades = winningTrades + losingTrades;
                metrics.Status = TrainingPhaseStatus.InProgress;
                
                _logger.LogDebug("[DASHBOARD] Strategy {Strategy} updated: WR={WinRate:F1}%, PnL=${PnL:F2}", 
                    strategyName, winRate, totalPnL);
            }
        }
    }

    /// <summary>
    /// Mark strategy training as complete
    /// </summary>
    public void CompleteStrategyTraining(string strategyName, string modelVersion)
    {
        if (_strategyMetrics.TryGetValue(strategyName, out var metrics))
        {
            lock (_stateLock)
            {
                metrics.Status = TrainingPhaseStatus.Complete;
                metrics.ModelVersion = modelVersion;
                
                LogActivity("info", $"{strategyName}Training", 
                    $"✓ Strategy training complete - WR: {metrics.WinRate:F1}%, PnL: ${metrics.TotalPnL:F2}");
            }
        }
    }

    /// <summary>
    /// Update system resource metrics
    /// </summary>
    public void UpdateResources()
    {
        try
        {
            if (_currentProcess == null)
                return;

            _currentProcess.Refresh();
            
            lock (_stateLock)
            {
                // CPU usage (approximate)
                _currentState.Resources.CpuUsagePercent = GetCpuUsage();
                
                // Memory usage
                _currentState.Resources.MemoryUsedMb = _currentProcess.WorkingSet64 / (1024 * 1024);
                _currentState.Resources.MemoryTotalMb = GetTotalMemoryMb();
                
                // Process count
                _currentState.Resources.ActiveProcesses = Process.GetProcesses().Count(p => 
                    p.ProcessName.Contains("dotnet", StringComparison.OrdinalIgnoreCase));
                
                // Disk I/O tracking is platform-specific, currently using default values
                _currentState.Resources.DiskReadMbPerSec = 0;
                _currentState.Resources.DiskWriteMbPerSec = 0;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[DASHBOARD] Failed to update resource metrics");
        }
    }

    /// <summary>
    /// Update time tracking
    /// </summary>
    public void UpdateTiming(TimeSpan estimatedTimeRemaining)
    {
        lock (_stateLock)
        {
            _currentState.Elapsed = DateTimeOffset.UtcNow - _currentState.SessionStartTime;
            _currentState.EstimatedTimeRemaining = estimatedTimeRemaining;
        }
    }

    /// <summary>
    /// Log activity entry
    /// </summary>
    public void LogActivity(string logLevel, string source, string message)
    {
        var entry = new ActivityLogEntry
        {
            Timestamp = DateTimeOffset.UtcNow,
            LogLevel = logLevel,
            Source = source,
            Message = message
        };
        
        _activityLog.Enqueue(entry);
        
        // Keep only recent entries
        while (_activityLog.Count > MaxActivityLogEntries)
        {
            _activityLog.TryDequeue(out _);
        }
        
        lock (_stateLock)
        {
            _currentState.RecentActivity = _activityLog.ToList();
        }
    }

    /// <summary>
    /// Get current dashboard state (thread-safe copy)
    /// </summary>
    public LabModeDashboardState GetCurrentState()
    {
        lock (_stateLock)
        {
            // Return a copy to avoid threading issues
            return new LabModeDashboardState
            {
                SessionId = _currentState.SessionId,
                SessionStartTime = _currentState.SessionStartTime,
                Elapsed = _currentState.Elapsed,
                EstimatedTimeRemaining = _currentState.EstimatedTimeRemaining,
                CurrentPhase = _currentState.CurrentPhase,
                OverallProgress = _currentState.OverallProgress,
                ComponentsCompleted = _currentState.ComponentsCompleted,
                TotalComponents = _currentState.TotalComponents,
                ComponentsRemaining = _currentState.ComponentsRemaining,
                HeavyPhase = ClonePhaseDetails(_currentState.HeavyPhase),
                MediumPhase = ClonePhaseDetails(_currentState.MediumPhase),
                LightPhase = ClonePhaseDetails(_currentState.LightPhase),
                StrategyMetrics = _currentState.StrategyMetrics.Select(CloneStrategyMetrics).ToList(),
                CurrentComponent = _currentState.CurrentComponent,
                Resources = CloneResourceMetrics(_currentState.Resources),
                RecentActivity = _currentState.RecentActivity.ToList()
            };
        }
    }

    private static LabModeDashboardState InitializeState()
    {
        return new LabModeDashboardState
        {
            HeavyPhase = new PhaseDetails { PhaseName = "Heavy" },
            MediumPhase = new PhaseDetails { PhaseName = "Medium" },
            LightPhase = new PhaseDetails { PhaseName = "Light" },
            Resources = new ResourceMetrics(),
            StrategyMetrics = new List<StrategyTrainingMetrics>(),
            RecentActivity = new List<ActivityLogEntry>()
        };
    }

    private PhaseDetails? GetPhaseDetails(string phaseName)
    {
        return phaseName switch
        {
            "Heavy" => _currentState.HeavyPhase,
            "Medium" => _currentState.MediumPhase,
            "Light" => _currentState.LightPhase,
            _ => null
        };
    }

    private static double GetCpuUsage()
    {
        // Simplified CPU usage - would need PerformanceCounter for accurate measurement
        return Math.Min(100.0, Environment.ProcessorCount * 20.0);
    }

    private static long GetTotalMemoryMb()
    {
        // Default to 16GB if unable to determine
        return 16 * 1024;
    }

    private static PhaseDetails ClonePhaseDetails(PhaseDetails source)
    {
        return new PhaseDetails
        {
            PhaseName = source.PhaseName,
            Status = source.Status,
            Duration = source.Duration,
            TotalComponents = source.TotalComponents,
            CompletedComponents = source.CompletedComponents,
            FailedComponents = source.FailedComponents,
            Components = source.Components.Select(c => new ComponentSummary
            {
                ComponentName = c.ComponentName,
                ProgressPercentage = c.ProgressPercentage,
                EpochsCompleted = c.EpochsCompleted,
                TotalEpochs = c.TotalEpochs,
                FinalLoss = c.FinalLoss,
                Status = c.Status,
                Metrics = new Dictionary<string, string>(c.Metrics)
            }).ToList()
        };
    }

    private static StrategyTrainingMetrics CloneStrategyMetrics(StrategyTrainingMetrics source)
    {
        return new StrategyTrainingMetrics
        {
            StrategyName = source.StrategyName,
            WinRate = source.WinRate,
            TotalPnL = source.TotalPnL,
            TotalWon = source.TotalWon,
            TotalLost = source.TotalLost,
            WinningTrades = source.WinningTrades,
            LosingTrades = source.LosingTrades,
            TotalTrades = source.TotalTrades,
            EpochsCompleted = source.EpochsCompleted,
            TotalEpochs = source.TotalEpochs,
            CurrentLoss = source.CurrentLoss,
            ModelVersion = source.ModelVersion,
            Status = source.Status
        };
    }

    private static ResourceMetrics CloneResourceMetrics(ResourceMetrics source)
    {
        return new ResourceMetrics
        {
            CpuUsagePercent = source.CpuUsagePercent,
            MemoryUsedMb = source.MemoryUsedMb,
            MemoryTotalMb = source.MemoryTotalMb,
            DiskReadMbPerSec = source.DiskReadMbPerSec,
            DiskWriteMbPerSec = source.DiskWriteMbPerSec,
            ActiveProcesses = source.ActiveProcesses
        };
    }
}
