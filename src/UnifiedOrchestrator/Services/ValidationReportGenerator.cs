using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Phase 6.6: Validation Report Generator
/// Aggregates all validation results into comprehensive reports
/// Produces JSON (machine-readable) and Console (human-readable) formats
/// </summary>
internal sealed class ValidationReportGenerator
{
    private readonly ILogger<ValidationReportGenerator> _logger;
    private readonly string _reportsDirectory;
    
    public ValidationReportGenerator(ILogger<ValidationReportGenerator> logger)
    {
        _logger = logger;
        var baseDir = Directory.GetCurrentDirectory();
        _reportsDirectory = Path.Combine(baseDir, "reports", "validation");
        Directory.CreateDirectory(_reportsDirectory);
    }
    
    /// <summary>
    /// Generate comprehensive validation report from all validation results
    /// </summary>
    public async Task<Phase6ValidationReport> GenerateReportAsync(
        string sessionId,
        InferenceTestResults? canaryResults,
        ComparisonReport? comparisonResults,
        ForgettingDetectionResult? forgettingResults,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[VALIDATION-REPORT] Generating validation report for session {SessionId}", sessionId);
            
            var report = new Phase6ValidationReport
            {
                SessionId = sessionId,
                Timestamp = DateTime.UtcNow,
                CanaryTestResults = canaryResults,
                PerformanceComparisonResults = comparisonResults,
                ForgettingDetectionResults = forgettingResults
            };
            
            // Determine overall status
            report.OverallStatus = DetermineOverallValidationStatus(canaryResults, comparisonResults, forgettingResults);
            
            // Identify blockers
            report.Blockers = IdentifyBlockers(canaryResults, comparisonResults, forgettingResults);
            
            // Make promotion recommendation
            report.PromotionRecommendation = report.Blockers.Count == 0 ? "PROMOTE" : "REJECT";
            
            // Generate summary
            report.Summary = GenerateSummary(report);
            
            // Save reports
            await SaveJsonReportAsync(report, cancellationToken).ConfigureAwait(false);
            var consolePath = await SaveConsoleReportAsync(report, cancellationToken).ConfigureAwait(false);
            
            // Display console report
            DisplayConsoleReport(report);
            
            _logger.LogInformation("[VALIDATION-REPORT] Report generated: {Status}, Recommendation: {Rec}",
                report.OverallStatus, report.PromotionRecommendation);
            
            return report;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VALIDATION-REPORT] Failed to generate validation report");
            throw;
        }
    }
    
    /// <summary>
    /// Determine overall validation status from all checks
    /// </summary>
    private string DetermineOverallValidationStatus(
        InferenceTestResults? canaryResults,
        ComparisonReport? comparisonResults,
        ForgettingDetectionResult? forgettingResults)
    {
        var statuses = new List<string>();
        
        if (canaryResults != null)
            statuses.Add(canaryResults.Passed ? "PASS" : "FAIL");
        
        if (comparisonResults != null)
            statuses.Add(comparisonResults.Status);
        
        if (forgettingResults != null)
            statuses.Add(forgettingResults.Status);
        
        // If any FAIL, overall is FAIL
        if (statuses.Any(s => s == "FAIL" || s == "FAILED"))
            return "FAIL";
        
        // If any WARNING, overall is WARNING
        if (statuses.Any(s => s == "WARNING"))
            return "WARNING";
        
        // Otherwise PASS
        return "PASS";
    }
    
    /// <summary>
    /// Identify critical blockers preventing promotion
    /// </summary>
    private List<string> IdentifyBlockers(
        InferenceTestResults? canaryResults,
        ComparisonReport? comparisonResults,
        ForgettingDetectionResult? forgettingResults)
    {
        var blockers = new List<string>();
        
        // Canary test failures are blockers
        if (canaryResults != null && !canaryResults.Passed)
        {
            blockers.Add($"Canary tests failed: {canaryResults.ErrorCount} errors, " +
                        $"{canaryResults.ModelsLoaded}/{canaryResults.ModelsExpected} models loaded");
        }
        
        // Severe regressions are blockers
        if (comparisonResults != null && comparisonResults.Status == "FAILED")
        {
            blockers.Add($"Performance regressions detected: {comparisonResults.RegressionCount} models regressed");
        }
        
        // Severe catastrophic forgetting is blocker
        if (forgettingResults != null && forgettingResults.Status == "FAILED")
        {
            blockers.Add($"Catastrophic forgetting detected: {forgettingResults.SevereForgettingCount} models affected");
        }
        
        return blockers;
    }
    
    /// <summary>
    /// Generate human-readable summary
    /// </summary>
    private string GenerateSummary(Phase6ValidationReport report)
    {
        var sb = new StringBuilder();
        
        sb.AppendLine($"Validation session {report.SessionId} completed with status: {report.OverallStatus}");
        
        if (report.CanaryTestResults != null)
        {
            sb.AppendLine($"- Canary Tests: {report.CanaryTestResults.ModelsLoaded}/{report.CanaryTestResults.ModelsExpected} models loaded, " +
                         $"avg latency {report.CanaryTestResults.AverageLatencyMs:F1}ms");
        }
        
        if (report.PerformanceComparisonResults != null)
        {
            sb.AppendLine($"- Performance: {report.PerformanceComparisonResults.ImprovementCount} improved, " +
                         $"{report.PerformanceComparisonResults.RegressionCount} regressed, " +
                         $"avg {report.PerformanceComparisonResults.AverageImprovement:F1}% change");
        }
        
        if (report.ForgettingDetectionResults != null)
        {
            sb.AppendLine($"- Forgetting: {report.ForgettingDetectionResults.NoForgettingCount} OK, " +
                         $"{report.ForgettingDetectionResults.MildForgettingCount} mild, " +
                         $"{report.ForgettingDetectionResults.SevereForgettingCount} severe");
        }
        
        return sb.ToString();
    }
    
    /// <summary>
    /// Save JSON report to disk
    /// </summary>
    private async Task SaveJsonReportAsync(Phase6ValidationReport report, CancellationToken cancellationToken)
    {
        try
        {
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
            var filename = $"validation_{timestamp}.json";
            var filepath = Path.Combine(_reportsDirectory, filename);
            
            var options = new JsonSerializerOptions 
            { 
                WriteIndented = true,
                Converters = { new JsonStringEnumConverter() }
            };
            var json = JsonSerializer.Serialize(report, options);
            
            await File.WriteAllTextAsync(filepath, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("[VALIDATION-REPORT] JSON report saved: {Path}", filepath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[VALIDATION-REPORT] Failed to save JSON report");
        }
    }
    
    /// <summary>
    /// Save console report to disk
    /// </summary>
    private async Task<string> SaveConsoleReportAsync(Phase6ValidationReport report, CancellationToken cancellationToken)
    {
        try
        {
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
            var filename = $"validation_{timestamp}.txt";
            var filepath = Path.Combine(_reportsDirectory, filename);
            
            var consoleReport = FormatConsoleReport(report);
            await File.WriteAllTextAsync(filepath, consoleReport, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("[VALIDATION-REPORT] Console report saved: {Path}", filepath);
            return filepath;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[VALIDATION-REPORT] Failed to save console report");
            return string.Empty;
        }
    }
    
    /// <summary>
    /// Format human-readable console report
    /// </summary>
    public string FormatConsoleReport(Phase6ValidationReport report)
    {
        var sb = new StringBuilder();
        
        sb.AppendLine("═══════════════════════════════════════════════════════════════════");
        sb.AppendLine("                 POST-TRAINING VALIDATION REPORT");
        sb.AppendLine("═══════════════════════════════════════════════════════════════════");
        sb.AppendLine();
        sb.AppendLine($"Session ID:    {report.SessionId}");
        sb.AppendLine($"Timestamp:     {report.Timestamp:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine($"Overall Status: {GetStatusIcon(report.OverallStatus)} {report.OverallStatus}");
        sb.AppendLine($"Recommendation: {GetRecommendationIcon(report.PromotionRecommendation)} {report.PromotionRecommendation}");
        sb.AppendLine();
        
        // Canary Test Results
        if (report.CanaryTestResults != null)
        {
            sb.AppendLine("───────────────────────────────────────────────────────────────────");
            sb.AppendLine("CANARY TESTS");
            sb.AppendLine("───────────────────────────────────────────────────────────────────");
            var canary = report.CanaryTestResults;
            sb.AppendLine($"Status:           {GetStatusIcon(canary.Passed ? "PASS" : "FAIL")} {(canary.Passed ? "PASS" : "FAIL")}");
            sb.AppendLine($"Models Loaded:    {canary.ModelsLoaded}/{canary.ModelsExpected}");
            sb.AppendLine($"Avg Latency:      {canary.AverageLatencyMs:F1} ms");
            sb.AppendLine($"Max Latency:      {canary.MaxLatencyMs:F1} ms");
            sb.AppendLine($"Errors:           {canary.ErrorCount}");
            sb.AppendLine();
        }
        
        // Performance Comparison Results
        if (report.PerformanceComparisonResults != null)
        {
            sb.AppendLine("───────────────────────────────────────────────────────────────────");
            sb.AppendLine("PERFORMANCE COMPARISON");
            sb.AppendLine("───────────────────────────────────────────────────────────────────");
            var perf = report.PerformanceComparisonResults;
            sb.AppendLine($"Status:           {GetStatusIcon(perf.Status)} {perf.Status}");
            sb.AppendLine($"Improvements:     {perf.ImprovementCount}");
            sb.AppendLine($"Regressions:      {perf.RegressionCount}");
            sb.AppendLine($"Avg Improvement:  {perf.AverageImprovement:F1}%");
            
            if (perf.ModelComparisons.Any())
            {
                sb.AppendLine();
                sb.AppendLine("Top Improvements:");
                var topImprovements = perf.ModelComparisons
                    .Where(c => !c.IsNewModel)
                    .OrderByDescending(c => c.ImprovementPercent)
                    .Take(5);
                
                foreach (var model in topImprovements)
                {
                    sb.AppendLine($"  • {model.ModelName}: {model.PrimaryMetric} {model.ImprovementPercent:+0.0;-0.0}%");
                }
            }
            
            if (perf.Regressions.Any())
            {
                sb.AppendLine();
                sb.AppendLine("⚠️  Regressions:");
                foreach (var reg in perf.Regressions)
                {
                    sb.AppendLine($"  • {reg.ModelName}: {reg.Metric} {reg.RegressionPercent:F1}% ({reg.Severity})");
                }
            }
            sb.AppendLine();
        }
        
        // Forgetting Detection Results
        if (report.ForgettingDetectionResults != null)
        {
            sb.AppendLine("───────────────────────────────────────────────────────────────────");
            sb.AppendLine("CATASTROPHIC FORGETTING DETECTION");
            sb.AppendLine("───────────────────────────────────────────────────────────────────");
            var forget = report.ForgettingDetectionResults;
            sb.AppendLine($"Status:           {GetStatusIcon(forget.Status)} {forget.Status}");
            sb.AppendLine($"No Forgetting:    {forget.NoForgettingCount}");
            sb.AppendLine($"Mild Forgetting:  {forget.MildForgettingCount}");
            sb.AppendLine($"Severe Forgetting: {forget.SevereForgettingCount}");
            
            var severeModels = forget.ModelResults.Where(m => m.ForgettingSeverity == "SEVERE").ToList();
            if (severeModels.Any())
            {
                sb.AppendLine();
                sb.AppendLine("⚠️  Models with Severe Forgetting:");
                foreach (var model in severeModels)
                {
                    sb.AppendLine($"  • {model.ModelName}: {model.DegradationPercent:F1}% degradation on old data");
                }
            }
            sb.AppendLine();
        }
        
        // Blockers
        if (report.Blockers.Any())
        {
            sb.AppendLine("───────────────────────────────────────────────────────────────────");
            sb.AppendLine("❌ BLOCKERS");
            sb.AppendLine("───────────────────────────────────────────────────────────────────");
            foreach (var blocker in report.Blockers)
            {
                sb.AppendLine($"  • {blocker}");
            }
            sb.AppendLine();
        }
        
        // Summary
        sb.AppendLine("───────────────────────────────────────────────────────────────────");
        sb.AppendLine("SUMMARY");
        sb.AppendLine("───────────────────────────────────────────────────────────────────");
        sb.AppendLine(report.Summary);
        
        sb.AppendLine("═══════════════════════════════════════════════════════════════════");
        
        return sb.ToString();
    }
    
    /// <summary>
    /// Display console report to logger
    /// </summary>
    private void DisplayConsoleReport(Phase6ValidationReport report)
    {
        var consoleReport = FormatConsoleReport(report);
        
        // Log each line separately for better formatting
        foreach (var line in consoleReport.Split('\n'))
        {
            _logger.LogInformation(line.TrimEnd());
        }
    }
    
    /// <summary>
    /// Get status icon for display
    /// </summary>
    private string GetStatusIcon(string status)
    {
        return status.ToUpperInvariant() switch
        {
            "PASS" => "✅",
            "FAIL" => "❌",
            "FAILED" => "❌",
            "WARNING" => "⚠️",
            "NEUTRAL" => "ℹ️",
            _ => "❔"
        };
    }
    
    /// <summary>
    /// Get recommendation icon
    /// </summary>
    private string GetRecommendationIcon(string recommendation)
    {
        return recommendation.ToUpperInvariant() switch
        {
            "PROMOTE" => "✅",
            "REJECT" => "❌",
            _ => "❔"
        };
    }
}

/// <summary>
/// Phase 6 validation report (aggregates all Phase 6 validation results)
/// </summary>
public sealed class Phase6ValidationReport
{
    [JsonPropertyName("sessionId")]
    public string SessionId { get; set; } = string.Empty;
    
    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; }
    
    [JsonPropertyName("overallStatus")]
    public string OverallStatus { get; set; } = string.Empty;
    
    [JsonPropertyName("promotionRecommendation")]
    public string PromotionRecommendation { get; set; } = string.Empty;
    
    [JsonPropertyName("canaryTestResults")]
    public InferenceTestResults? CanaryTestResults { get; set; }
    
    [JsonPropertyName("performanceComparisonResults")]
    public ComparisonReport? PerformanceComparisonResults { get; set; }
    
    [JsonPropertyName("forgettingDetectionResults")]
    public ForgettingDetectionResult? ForgettingDetectionResults { get; set; }
    
    [JsonPropertyName("blockers")]
    public List<string> Blockers { get; set; } = new();
    
    [JsonPropertyName("summary")]
    public string Summary { get; set; } = string.Empty;
}
