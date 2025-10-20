using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Training Performance Profiler - Phase 14: Debugging & Diagnostics Tools
/// Provides detailed performance profiling and bottleneck identification
/// Enabled via LAB_DEBUG_MODE=1 or LAB_PROFILE=1 environment variable
/// </summary>
internal sealed class TrainingPerformanceProfiler
{
    private readonly ILogger<TrainingPerformanceProfiler> _logger;
    private readonly bool _enabled;
    private readonly Dictionary<string, ProfileSection> _sections = new();
    private readonly Dictionary<string, Stopwatch> _activeTimers = new();
    private readonly string _profileOutputPath;
    private readonly Stopwatch _sessionTimer = new();

    public TrainingPerformanceProfiler(ILogger<TrainingPerformanceProfiler> logger)
    {
        _logger = logger;
        
        // Check if profiling is enabled
        var debugMode = Environment.GetEnvironmentVariable("LAB_DEBUG_MODE");
        var profileMode = Environment.GetEnvironmentVariable("LAB_PROFILE");
        _enabled = debugMode == "1" || profileMode == "1";

        _profileOutputPath = Path.Combine(
            Directory.GetCurrentDirectory(),
            "artifacts",
            "debug",
            "performance-profile.txt");

        if (_enabled)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_profileOutputPath)!);
            _logger.LogInformation("[PROFILER] Performance profiling ENABLED");
            _sessionTimer.Start();
        }
    }

    /// <summary>
    /// Start profiling a section
    /// Phase 14.3: Performance Profiling System
    /// </summary>
    public void StartProfilingSection(string sectionName)
    {
        if (!_enabled) return;

        if (_activeTimers.ContainsKey(sectionName))
        {
            _logger.LogWarning("[PROFILER] Section {Section} already started", sectionName);
            return;
        }

        var stopwatch = Stopwatch.StartNew();
        _activeTimers[sectionName] = stopwatch;
        
        _logger.LogDebug("[PROFILER] Started: {Section}", sectionName);
    }

    /// <summary>
    /// End profiling a section
    /// Phase 14.3: Performance Profiling System
    /// </summary>
    public void EndProfilingSection(string sectionName)
    {
        if (!_enabled) return;

        if (!_activeTimers.TryGetValue(sectionName, out var stopwatch))
        {
            _logger.LogWarning("[PROFILER] Section {Section} not started", sectionName);
            return;
        }

        stopwatch.Stop();
        var elapsed = stopwatch.Elapsed;
        _activeTimers.Remove(sectionName);

        // Add or update section
        if (!_sections.ContainsKey(sectionName))
        {
            _sections[sectionName] = new ProfileSection
            {
                Name = sectionName,
                TotalTime = TimeSpan.Zero,
                CallCount = 0,
                MinTime = TimeSpan.MaxValue,
                MaxTime = TimeSpan.Zero
            };
        }

        var section = _sections[sectionName];
        section.TotalTime += elapsed;
        section.CallCount++;
        section.MinTime = elapsed < section.MinTime ? elapsed : section.MinTime;
        section.MaxTime = elapsed > section.MaxTime ? elapsed : section.MaxTime;
        section.AverageTime = TimeSpan.FromTicks(section.TotalTime.Ticks / section.CallCount);

        _logger.LogDebug("[PROFILER] Ended: {Section} ({Duration:F2}s)", sectionName, elapsed.TotalSeconds);
    }

    /// <summary>
    /// Generate performance profile report
    /// Phase 14.3: Performance Profiling System
    /// </summary>
    public async Task<string> GenerateProfileReportAsync(
        string sessionId,
        CancellationToken cancellationToken = default)
    {
        if (!_enabled)
        {
            return "Performance profiling not enabled";
        }

        _sessionTimer.Stop();
        var totalSessionTime = _sessionTimer.Elapsed;

        var report = new System.Text.StringBuilder();
        report.AppendLine($"PERFORMANCE PROFILE - Session {sessionId}");
        report.AppendLine("========================================");
        report.AppendLine($"Total Time: {FormatDuration(totalSessionTime)}");
        report.AppendLine();

        // Calculate time breakdown
        report.AppendLine("Time Breakdown:");
        var sortedSections = _sections.Values
            .OrderByDescending(s => s.TotalTime)
            .ToList();

        foreach (var section in sortedSections)
        {
            var percentage = (section.TotalTime.TotalSeconds / totalSessionTime.TotalSeconds) * 100;
            report.AppendLine($"- {section.Name,-30} {FormatDuration(section.TotalTime),15} ({percentage,5:F1}%)");
        }
        report.AppendLine();

        // Detailed statistics
        report.AppendLine("Detailed Statistics:");
        report.AppendLine($"{"Section",-30} {"Calls",8} {"Avg Time",12} {"Min Time",12} {"Max Time",12}");
        report.AppendLine(new string('-', 80));
        
        foreach (var section in sortedSections)
        {
            report.AppendLine($"{section.Name,-30} {section.CallCount,8} " +
                            $"{FormatDuration(section.AverageTime),12} " +
                            $"{FormatDuration(section.MinTime),12} " +
                            $"{FormatDuration(section.MaxTime),12}");
        }
        report.AppendLine();

        // Identify bottlenecks
        report.AppendLine("Bottlenecks Identified:");
        var bottleneckNum = 1;
        
        // Data loading bottleneck (>20% of time)
        var dataLoading = _sections.Values.FirstOrDefault(s => s.Name.Contains("DataLoading"));
        if (dataLoading != null)
        {
            var percentage = (dataLoading.TotalTime.TotalSeconds / totalSessionTime.TotalSeconds) * 100;
            if (percentage > 20)
            {
                report.AppendLine($"{bottleneckNum}. Data loading is slow ({percentage:F1}% of time) - consider caching");
                bottleneckNum++;
            }
        }

        // Model training bottleneck
        var modelTraining = _sections.Values.FirstOrDefault(s => s.Name.Contains("ModelTraining"));
        if (modelTraining != null && modelTraining.CallCount > 0)
        {
            var avgMinutes = modelTraining.AverageTime.TotalMinutes;
            if (avgMinutes > 6)
            {
                report.AppendLine($"{bottleneckNum}. Model training averages {avgMinutes:F1} minutes (expected <6 minutes)");
                bottleneckNum++;
            }
        }

        // Check for outliers
        foreach (var section in sortedSections.Where(s => s.CallCount > 1))
        {
            var avgSeconds = section.AverageTime.TotalSeconds;
            var maxSeconds = section.MaxTime.TotalSeconds;
            
            if (maxSeconds > avgSeconds * 2)
            {
                report.AppendLine($"{bottleneckNum}. {section.Name} has outlier: max {FormatDuration(section.MaxTime)} vs avg {FormatDuration(section.AverageTime)}");
                bottleneckNum++;
            }
        }

        if (bottleneckNum == 1)
        {
            report.AppendLine("No significant bottlenecks detected");
        }
        report.AppendLine();

        // Recommendations
        report.AppendLine("Recommendations:");
        
        if (dataLoading != null && (dataLoading.TotalTime.TotalSeconds / totalSessionTime.TotalSeconds) > 0.15)
        {
            report.AppendLine("- Pre-load historical data into memory before training starts");
            report.AppendLine("- Optimize data preprocessing pipeline");
        }

        if (modelTraining != null && modelTraining.AverageTime.TotalMinutes > 6)
        {
            report.AppendLine("- Consider GPU acceleration for model training");
            report.AppendLine("- Optimize model architecture for faster training");
        }

        var checkpointing = _sections.Values.FirstOrDefault(s => s.Name.Contains("Checkpointing"));
        if (checkpointing != null && (checkpointing.TotalTime.TotalSeconds / totalSessionTime.TotalSeconds) > 0.05)
        {
            report.AppendLine("- Reduce checkpoint frequency to improve performance");
        }

        // Save report to file
        var reportText = report.ToString();
        await File.WriteAllTextAsync(_profileOutputPath, reportText, cancellationToken).ConfigureAwait(false);
        
        _logger.LogInformation("[PROFILER] Performance report saved: {Path}", _profileOutputPath);
        
        return reportText;
    }

    /// <summary>
    /// Format duration for display
    /// </summary>
    private static string FormatDuration(TimeSpan duration)
    {
        if (duration.TotalHours >= 1)
            return $"{(int)duration.TotalHours}h {duration.Minutes}m {duration.Seconds}s";
        else if (duration.TotalMinutes >= 1)
            return $"{(int)duration.TotalMinutes}m {duration.Seconds}s";
        else
            return $"{duration.TotalSeconds:F2}s";
    }

    /// <summary>
    /// Get summary statistics
    /// </summary>
    public Dictionary<string, TimeSpan> GetSummaryStatistics()
    {
        return _sections.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value.TotalTime);
    }
}

/// <summary>
/// Profile section data
/// </summary>
internal class ProfileSection
{
    public string Name { get; set; } = string.Empty;
    public TimeSpan TotalTime { get; set; }
    public TimeSpan AverageTime { get; set; }
    public TimeSpan MinTime { get; set; }
    public TimeSpan MaxTime { get; set; }
    public int CallCount { get; set; }
}
