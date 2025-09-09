using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for workflow orchestration and conflict management
/// </summary>
public interface IWorkflowOrchestrationManager : IDisposable
{
    /// <summary>
    /// Initialize workflow orchestration services
    /// </summary>
    Task InitializeAsync();
    
    /// <summary>
    /// Request workflow execution with conflict detection
    /// </summary>
    Task<bool> RequestWorkflowExecutionAsync(string workflowName, Func<Task> action, List<string>? requiredResources = null);
    
    /// <summary>
    /// Resolve workflow conflicts
    /// </summary>
    Task<WorkflowOrchestrationManager.ConflictResolution> ResolveConflictsAsync();
    
    /// <summary>
    /// Get current orchestration status
    /// </summary>
    WorkflowOrchestrationManager.WorkflowOrchestrationStatus GetStatus();
}