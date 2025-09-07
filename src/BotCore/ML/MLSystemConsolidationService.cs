using Microsoft.Extensions.Logging;

namespace BotCore.ML;

/// <summary>
/// ML System Consolidation Service - Identifies and consolidates duplicate ML implementations
/// Removes duplicate code between BotCore/ML and Enhanced directories
/// </summary>
public sealed class MLSystemConsolidationService
{
    private readonly ILogger<MLSystemConsolidationService> _logger;
    private readonly List<ConsolidationAction> _consolidationActions = new();

    public class ConsolidationAction
    {
        public string Action { get; set; } = string.Empty;
        public string SourcePath { get; set; } = string.Empty;
        public string TargetPath { get; set; } = string.Empty;
        public string Reason { get; set; } = string.Empty;
        public ConsolidationStatus Status { get; set; } = ConsolidationStatus.Pending;
        public DateTime ActionTime { get; set; } = DateTime.UtcNow;
        public string? ErrorMessage { get; set; }
    }

    public enum ConsolidationStatus
    {
        Pending,
        Completed,
        Failed,
        Skipped
    }

    public MLSystemConsolidationService(ILogger<MLSystemConsolidationService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Analyze duplicate ML systems and plan consolidation
    /// </summary>
    public async Task<ConsolidationPlan> AnalyzeDuplicateSystemsAsync()
    {
        _logger.LogInformation("[ML-Consolidation] Starting duplicate ML system analysis");

        var plan = new ConsolidationPlan();

        // Check for Enhanced/MLRLSystem.cs duplicate MLMemoryManager
        var enhancedMLFile = Path.Combine("Enhanced", "MLRLSystem.cs");
        var botCoreMLMemoryManager = Path.Combine("src", "BotCore", "ML", "MLMemoryManager.cs");

        if (File.Exists(enhancedMLFile))
        {
            var content = await File.ReadAllTextAsync(enhancedMLFile);
            
            if (content.Contains("public class MLMemoryManager"))
            {
                _consolidationActions.Add(new ConsolidationAction
                {
                    Action = "Remove Duplicate MLMemoryManager",
                    SourcePath = enhancedMLFile,
                    TargetPath = botCoreMLMemoryManager,
                    Reason = "Duplicate MLMemoryManager class found in Enhanced directory. BotCore version has ONNX integration and is more complete."
                });

                plan.DuplicatesFound++;
            }

            // Check for other duplicate ML components
            if (content.Contains("class MLModelManager") || content.Contains("class ModelManager"))
            {
                _consolidationActions.Add(new ConsolidationAction
                {
                    Action = "Analyze ML Model Managers",
                    SourcePath = enhancedMLFile,
                    TargetPath = "Analysis needed",
                    Reason = "Multiple ML model manager implementations detected"
                });

                plan.DuplicatesFound++;
            }
        }

        // Check for other potential duplicates
        await AnalyzeMLDirectoriesAsync(plan);

        _logger.LogInformation("[ML-Consolidation] Analysis complete - {DuplicatesFound} duplicates found, {ActionsPlanned} actions planned",
            plan.DuplicatesFound, _consolidationActions.Count);

        return plan;
    }

    private async Task AnalyzeMLDirectoriesAsync(ConsolidationPlan plan)
    {
        var botCoreMLDir = Path.Combine("src", "BotCore", "ML");
        var enhancedDir = "Enhanced";

        if (!Directory.Exists(botCoreMLDir) || !Directory.Exists(enhancedDir))
        {
            _logger.LogWarning("[ML-Consolidation] ML directories not found for analysis");
            return;
        }

        // Get ML-related files in both directories
        var botCoreMLFiles = Directory.GetFiles(botCoreMLDir, "*.cs", SearchOption.AllDirectories)
            .Select(f => new { Path = f, Name = Path.GetFileName(f) });

        var enhancedFiles = Directory.GetFiles(enhancedDir, "*.cs", SearchOption.AllDirectories)
            .Select(f => new { Path = f, Name = Path.GetFileName(f) });

        // Look for potential name conflicts
        foreach (var enhancedFile in enhancedFiles)
        {
            var potentialDuplicate = botCoreMLFiles.FirstOrDefault(b => 
                b.Name.Equals(enhancedFile.Name, StringComparison.OrdinalIgnoreCase));

            if (potentialDuplicate != null)
            {
                _consolidationActions.Add(new ConsolidationAction
                {
                    Action = "Analyze File Conflict",
                    SourcePath = enhancedFile.Path,
                    TargetPath = potentialDuplicate.Path,
                    Reason = $"Potential file name conflict: {enhancedFile.Name}"
                });

                plan.ConflictsFound++;
            }
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Execute consolidation plan
    /// </summary>
    public async Task<ConsolidationResult> ExecuteConsolidationAsync(bool dryRun = true)
    {
        _logger.LogInformation("[ML-Consolidation] Starting consolidation execution (DryRun: {DryRun})", dryRun);

        var result = new ConsolidationResult
        {
            DryRun = dryRun,
            StartTime = DateTime.UtcNow
        };

        foreach (var action in _consolidationActions.Where(a => a.Status == ConsolidationStatus.Pending))
        {
            try
            {
                await ExecuteActionAsync(action, dryRun);
                result.ActionsCompleted++;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ML-Consolidation] Failed to execute action: {Action}", action.Action);
                action.Status = ConsolidationStatus.Failed;
                action.ErrorMessage = ex.Message;
                result.ActionsFailed++;
            }
        }

        result.EndTime = DateTime.UtcNow;
        result.Duration = result.EndTime - result.StartTime;

        _logger.LogInformation("[ML-Consolidation] Consolidation completed - {Completed} completed, {Failed} failed",
            result.ActionsCompleted, result.ActionsFailed);

        return result;
    }

    private async Task ExecuteActionAsync(ConsolidationAction action, bool dryRun)
    {
        _logger.LogInformation("[ML-Consolidation] Executing: {Action} - {SourcePath}", action.Action, action.SourcePath);

        if (dryRun)
        {
            _logger.LogInformation("[ML-Consolidation] DRY RUN: Would execute {Action}", action.Action);
            action.Status = ConsolidationStatus.Completed;
            return;
        }

        switch (action.Action)
        {
            case "Remove Duplicate MLMemoryManager":
                await RemoveDuplicateMLMemoryManagerAsync(action);
                break;

            case "Analyze ML Model Managers":
                await AnalyzeMLModelManagersAsync(action);
                break;

            case "Analyze File Conflict":
                await AnalyzeFileConflictAsync(action);
                break;

            default:
                _logger.LogWarning("[ML-Consolidation] Unknown action: {Action}", action.Action);
                action.Status = ConsolidationStatus.Skipped;
                break;
        }
    }

    private async Task RemoveDuplicateMLMemoryManagerAsync(ConsolidationAction action)
    {
        if (!File.Exists(action.SourcePath))
        {
            action.Status = ConsolidationStatus.Skipped;
            action.ErrorMessage = "Source file not found";
            return;
        }

        var content = await File.ReadAllTextAsync(action.SourcePath);
        
        // Create backup
        var backupPath = action.SourcePath + ".backup." + DateTime.Now.ToString("yyyyMMdd_HHmmss");
        await File.WriteAllTextAsync(backupPath, content);
        _logger.LogInformation("[ML-Consolidation] Created backup: {BackupPath}", backupPath);

        // Remove the MLMemoryManager class from Enhanced/MLRLSystem.cs
        var lines = content.Split('\n').ToList();
        var mlMemoryManagerStart = -1;
        var braceCount = 0;
        var inMLMemoryManager = false;

        for (int i = 0; i < lines.Count; i++)
        {
            var line = lines[i];
            
            if (line.Trim().Contains("public class MLMemoryManager") && !inMLMemoryManager)
            {
                mlMemoryManagerStart = i;
                inMLMemoryManager = true;
                braceCount = 0;
            }

            if (inMLMemoryManager)
            {
                braceCount += line.Count(c => c == '{');
                braceCount -= line.Count(c => c == '}');

                if (braceCount == 0 && mlMemoryManagerStart != -1)
                {
                    // Remove lines from start to current line
                    var linesToRemove = i - mlMemoryManagerStart + 1;
                    lines.RemoveRange(mlMemoryManagerStart, linesToRemove);
                    
                    _logger.LogInformation("[ML-Consolidation] Removed {LineCount} lines of duplicate MLMemoryManager", linesToRemove);
                    break;
                }
            }
        }

        // Write the cleaned content
        var cleanedContent = string.Join('\n', lines);
        await File.WriteAllTextAsync(action.SourcePath, cleanedContent);

        action.Status = ConsolidationStatus.Completed;
        _logger.LogInformation("[ML-Consolidation] Successfully removed duplicate MLMemoryManager from {SourcePath}", action.SourcePath);
    }

    private async Task AnalyzeMLModelManagersAsync(ConsolidationAction action)
    {
        // This would analyze different ML model manager implementations
        // For now, just log the analysis
        _logger.LogInformation("[ML-Consolidation] Analyzing ML model managers in {SourcePath}", action.SourcePath);
        action.Status = ConsolidationStatus.Completed;
        await Task.CompletedTask;
    }

    private async Task AnalyzeFileConflictAsync(ConsolidationAction action)
    {
        // This would analyze file conflicts and suggest resolution
        _logger.LogInformation("[ML-Consolidation] Analyzing file conflict: {SourcePath} vs {TargetPath}", 
            action.SourcePath, action.TargetPath);
        action.Status = ConsolidationStatus.Completed;
        await Task.CompletedTask;
    }

    /// <summary>
    /// Generate consolidation report
    /// </summary>
    public async Task<string> GenerateConsolidationReportAsync()
    {
        var plan = await AnalyzeDuplicateSystemsAsync();
        
        var report = $@"
=== ML System Consolidation Report ===
Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}

ANALYSIS SUMMARY:
- Duplicates Found: {plan.DuplicatesFound}
- Conflicts Found: {plan.ConflictsFound}
- Actions Planned: {_consolidationActions.Count}

RECOMMENDED ACTIONS:
";

        foreach (var action in _consolidationActions)
        {
            var statusIcon = action.Status switch
            {
                ConsolidationStatus.Completed => "✅",
                ConsolidationStatus.Failed => "❌",
                ConsolidationStatus.Skipped => "⏭️",
                _ => "⏳"
            };

            report += $"{statusIcon} {action.Action}\n";
            report += $"   Source: {action.SourcePath}\n";
            report += $"   Target: {action.TargetPath}\n";
            report += $"   Reason: {action.Reason}\n";
            
            if (!string.IsNullOrEmpty(action.ErrorMessage))
            {
                report += $"   Error: {action.ErrorMessage}\n";
            }
            
            report += "\n";
        }

        report += @"
CONSOLIDATION STRATEGY:
1. Keep BotCore/ML as the primary ML implementation directory
2. Remove duplicate MLMemoryManager from Enhanced/MLRLSystem.cs
3. Migrate any unique functionality from Enhanced to BotCore
4. Update references to use consolidated implementations
5. Remove or deprecate Enhanced ML duplicates

NEXT STEPS:
1. Review this report
2. Execute consolidation with dryRun=false
3. Update dependency injection registrations
4. Run tests to ensure functionality is preserved
5. Clean up unused Enhanced ML code
";

        _logger.LogInformation("[ML-Consolidation] Consolidation report generated");
        return report;
    }

    public class ConsolidationPlan
    {
        public int DuplicatesFound { get; set; }
        public int ConflictsFound { get; set; }
        public DateTime AnalysisTime { get; set; } = DateTime.UtcNow;
    }

    public class ConsolidationResult
    {
        public bool DryRun { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public TimeSpan Duration { get; set; }
        public int ActionsCompleted { get; set; }
        public int ActionsFailed { get; set; }
    }
}
