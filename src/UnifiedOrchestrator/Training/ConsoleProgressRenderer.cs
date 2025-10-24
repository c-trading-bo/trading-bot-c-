using System;
using System.Text;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Renders progress bars and training status to console
/// Provides real-time visual feedback during training sessions
/// Supports both legacy progress display and new Lab Mode dashboard
/// </summary>
public sealed class ConsoleProgressRenderer
{
    private readonly ILogger<ConsoleProgressRenderer> _logger;
    private readonly ProgressTracker _progressTracker;
    private readonly LabModeDashboardRenderer? _dashboardRenderer;
    private readonly LabModeDashboardStateManager? _dashboardStateManager;
    private int _lastRenderedLineCount = 0;
    private readonly bool _useLabModeDashboard;

    public ConsoleProgressRenderer(
        ILogger<ConsoleProgressRenderer> logger,
        ProgressTracker progressTracker,
        LabModeDashboardRenderer? dashboardRenderer = null,
        LabModeDashboardStateManager? dashboardStateManager = null)
    {
        _logger = logger;
        _progressTracker = progressTracker;
        _dashboardRenderer = dashboardRenderer;
        _dashboardStateManager = dashboardStateManager;
        
        // Use Lab Mode dashboard if enabled and components are available
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        _useLabModeDashboard = labMode == "1" && _dashboardRenderer != null && _dashboardStateManager != null;
    }

    /// <summary>
    /// Render full progress display to console
    /// Uses Lab Mode dashboard if enabled, otherwise uses legacy display
    /// </summary>
    public void RenderProgress()
    {
        // Use Lab Mode dashboard if enabled
        if (_useLabModeDashboard && _dashboardRenderer != null && _dashboardStateManager != null)
        {
            var dashboardState = _dashboardStateManager.GetCurrentState();
            _dashboardRenderer.RenderDashboard(dashboardState);
            return;
        }
        
        // Fallback to legacy display
        var summary = _progressTracker.GetSummary();
        var output = new StringBuilder();

        // Header
        output.AppendLine("╔══════════════════════════════════════════════════════════════════════╗");
        output.AppendLine("║                      TRAINING SESSION PROGRESS                       ║");
        output.AppendLine("╚══════════════════════════════════════════════════════════════════════╝");
        output.AppendLine();

        // Overall Progress Bar
        var progressBar = _progressTracker.GetProgressBar(60);
        var percentage = summary.ProgressPercentage;
        output.AppendLine($"Overall Progress: {progressBar} {percentage:F1}%");
        output.AppendLine();

        // Components Status
        output.AppendLine($"Components: {summary.CompletedComponents}/{summary.TotalComponents} completed " +
                         $"({summary.RemainingComponents} remaining)");
        output.AppendLine();

        // Current Phase
        output.AppendLine($"Phase: {GetPhaseDisplay(summary.CurrentPhase)}");
        output.AppendLine();

        // Current Component
        if (!string.IsNullOrEmpty(summary.CurrentComponent))
        {
            output.AppendLine($"Training: {summary.CurrentComponent}");
            
            // Component progress bar
            var componentProgress = _progressTracker.CurrentComponentProgress;
            var componentBar = GenerateProgressBar((int)(componentProgress * 60), 60);
            output.AppendLine($"  {componentBar} {componentProgress * 100:F1}%");
            
            // Epoch information
            if (summary.TotalEpochs > 0)
            {
                output.AppendLine($"  Epoch: {summary.CurrentEpoch}/{summary.TotalEpochs}");
                output.AppendLine($"  Loss: {summary.CurrentLoss:F4}");
            }
            
            output.AppendLine();
        }

        // Timing Information
        var elapsedTime = _progressTracker.GetFormattedElapsedTime();
        var eta = _progressTracker.GetFormattedETA();
        output.AppendLine($"Elapsed: {elapsedTime} | ETA: {eta}");
        output.AppendLine();

        // Footer
        output.AppendLine("─────────────────────────────────────────────────────────────────────────");

        // Log the output
        _logger.LogInformation("{Progress}", output.ToString());
        _lastRenderedLineCount = CountLines(output.ToString());
    }

    /// <summary>
    /// Render compact progress line (single line)
    /// </summary>
    public void RenderCompactProgress()
    {
        var summary = _progressTracker.GetSummary();
        var percentage = summary.ProgressPercentage;
        var eta = _progressTracker.GetFormattedETA();
        var elapsed = _progressTracker.GetFormattedElapsedTime();
        
        var compactBar = _progressTracker.GetProgressBar(30);
        var status = $"{compactBar} {percentage:F1}% | {summary.CompletedComponents}/{summary.TotalComponents} | " +
                    $"Elapsed: {elapsed} | ETA: {eta}";

        if (!string.IsNullOrEmpty(summary.CurrentComponent))
        {
            status += $" | Training: {summary.CurrentComponent}";
        }

        _logger.LogInformation("[PROGRESS] {Status}", status);
    }

    /// <summary>
    /// Render phase start banner
    /// </summary>
    public void RenderPhaseStart(string phase, int componentCount)
    {
        var output = new StringBuilder();
        output.AppendLine();
        output.AppendLine("╔══════════════════════════════════════════════════════════════════════╗");
        output.AppendLine($"║  {GetPhaseDisplay(phase).PadRight(68)}║");
        output.AppendLine($"║  Components: {componentCount.ToString().PadRight(57)}║");
        output.AppendLine("╚══════════════════════════════════════════════════════════════════════╝");
        output.AppendLine();

        _logger.LogInformation("{PhaseStart}", output.ToString());
    }

    /// <summary>
    /// Render phase completion banner
    /// </summary>
    public void RenderPhaseComplete(string phase, int successful, int failed, TimeSpan duration)
    {
        var output = new StringBuilder();
        output.AppendLine();
        output.AppendLine("╔══════════════════════════════════════════════════════════════════════╗");
        output.AppendLine($"║  {GetPhaseDisplay(phase)} - COMPLETE".PadRight(70) + "║");
        output.AppendLine($"║  ✓ Successful: {successful}  ✗ Failed: {failed}  Duration: {FormatDuration(duration)}".PadRight(70) + "║");
        output.AppendLine("╚══════════════════════════════════════════════════════════════════════╝");
        output.AppendLine();

        _logger.LogInformation("{PhaseComplete}", output.ToString());
    }

    /// <summary>
    /// Render component start notification
    /// </summary>
    public void RenderComponentStart(string componentName, int componentNumber, int totalComponents)
    {
        _logger.LogInformation("[{Current}/{Total}] Starting: {Component}", 
            componentNumber, totalComponents, componentName);
    }

    /// <summary>
    /// Render component completion notification
    /// </summary>
    public void RenderComponentComplete(string componentName, bool success, TimeSpan duration, string? error = null)
    {
        if (success)
        {
            _logger.LogInformation("  ✓ Completed: {Component} ({Duration})", 
                componentName, FormatDuration(duration));
        }
        else
        {
            _logger.LogError("  ✗ Failed: {Component} - {Error}", componentName, error ?? "Unknown error");
        }
    }

    /// <summary>
    /// Render training session summary
    /// </summary>
    public void RenderSessionSummary(TrainingSessionSummary summary)
    {
        var output = new StringBuilder();
        output.AppendLine();
        output.AppendLine("╔══════════════════════════════════════════════════════════════════════╗");
        output.AppendLine("║                    TRAINING SESSION SUMMARY                          ║");
        output.AppendLine("╚══════════════════════════════════════════════════════════════════════╝");
        output.AppendLine();
        output.AppendLine($"Session ID:       {summary.SessionId}");
        output.AppendLine($"Status:           {summary.Status}");
        output.AppendLine($"Duration:         {FormatDuration(summary.Duration)}");
        output.AppendLine();
        output.AppendLine($"Components:");
        output.AppendLine($"  Total:          {summary.ComponentsTotal}");
        output.AppendLine($"  Completed:      {summary.ComponentsCompleted}");
        output.AppendLine($"  Failed:         {summary.ComponentsFailed}");
        output.AppendLine($"  Success Rate:   {summary.SuccessRate:P1}");
        output.AppendLine();
        output.AppendLine($"Promotion:        {(summary.PromotionSuccess ? "✓ Success" : "✗ Failed")}");
        
        if (summary.FailedComponents.Count > 0)
        {
            output.AppendLine();
            output.AppendLine("Failed Components:");
            foreach (var failed in summary.FailedComponents)
            {
                output.AppendLine($"  ✗ {failed}");
            }
        }
        
        output.AppendLine();
        output.AppendLine("╚══════════════════════════════════════════════════════════════════════╝");

        _logger.LogInformation("{Summary}", output.ToString());
    }

    /// <summary>
    /// Generate ASCII progress bar
    /// </summary>
    private static string GenerateProgressBar(int filled, int total)
    {
        var filledStr = new string('█', filled);
        var emptyStr = new string('░', total - filled);
        return $"[{filledStr}{emptyStr}]";
    }

    /// <summary>
    /// Get phase display name with icon
    /// </summary>
    private static string GetPhaseDisplay(string phase)
    {
        return phase switch
        {
            "Heavy" => "🔴 HEAVY PHASE (Large Neural Networks)",
            "Medium" => "🟡 MEDIUM PHASE (Calibration & Optimization)",
            "Light" => "🟢 LIGHT PHASE (Online Learning)",
            "NotStarted" => "⚪ Not Started",
            _ => phase
        };
    }

    /// <summary>
    /// Format duration to human-readable string
    /// </summary>
    private static string FormatDuration(TimeSpan duration)
    {
        if (duration.TotalHours >= 1)
        {
            return $"{(int)duration.TotalHours}h {duration.Minutes}m";
        }
        else if (duration.TotalMinutes >= 1)
        {
            return $"{(int)duration.TotalMinutes}m {duration.Seconds}s";
        }
        else
        {
            return $"{(int)duration.TotalSeconds}s";
        }
    }

    /// <summary>
    /// Count lines in a string
    /// </summary>
    private static int CountLines(string text)
    {
        if (string.IsNullOrEmpty(text))
            return 0;

        int count = 1;
        foreach (char c in text)
        {
            if (c == '\n')
                count++;
        }
        return count;
    }
}
