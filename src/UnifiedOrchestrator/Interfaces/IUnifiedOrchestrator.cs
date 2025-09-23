using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.Abstractions;
using System.Collections.ObjectModel;
using static TradingBot.Abstractions.OrchestratorDefaults;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Core interface for the unified orchestrator system
/// </summary>
public interface IUnifiedOrchestrator
{
    /// <summary>
    /// Initialize the orchestrator with all required services
    /// </summary>
    Task InitializeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Start the orchestrator and begin workflow execution
    /// </summary>
    Task StartAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Stop the orchestrator gracefully
    /// </summary>
    Task StopAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get all registered workflows
    /// </summary>
    IReadOnlyList<UnifiedWorkflow> GetWorkflows();
    
    /// <summary>
    /// Get workflow by ID
    /// </summary>
    UnifiedWorkflow? GetWorkflow(string workflowId);
    
    /// <summary>
    /// Register a new workflow
    /// </summary>
    Task RegisterWorkflowAsync(UnifiedWorkflow workflow, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Execute a workflow manually
    /// </summary>
    Task<WorkflowExecutionResult> ExecuteWorkflowAsync(string workflowId, Dictionary<string, object>? parameters = null, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get execution history for a workflow
    /// </summary>
    IReadOnlyList<WorkflowExecutionContext> GetExecutionHistory(string workflowId, int limit);
    
    /// <summary>
    /// Get current system status
    /// </summary>
    Task<OrchestratorStatus> GetStatusAsync();
}

/// <summary>
/// Interface for workflow action execution
/// </summary>
public interface IWorkflowActionExecutor
{
    /// <summary>
    /// Execute a specific action within a workflow
    /// </summary>
    Task<WorkflowExecutionResult> ExecuteActionAsync(string action, WorkflowExecutionContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if this executor can handle the specified action
    /// </summary>
    bool CanExecute(string action);
    
    /// <summary>
    /// Get list of supported actions
    /// </summary>
    IReadOnlyList<string> SupportedActions { get; }
}

/// <summary>
/// Interface for trading-specific operations
/// </summary>
public interface ITradingOrchestrator : IWorkflowActionExecutor
{
    /// <summary>
    /// Connect to TopstepX API and hubs
    /// </summary>
    Task ConnectAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Disconnect from TopstepX
    /// </summary>
    Task DisconnectAsync();
    
    /// <summary>
    /// Execute ES/NQ trading signals
    /// </summary>
    Task ExecuteESNQTradingAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Manage portfolio heat and risk
    /// </summary>
    Task ManagePortfolioRiskAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Analyze market microstructure
    /// </summary>
    Task AnalyzeMicrostructureAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
}

/// <summary>
/// Interface for ML/RL intelligence operations
/// </summary>
public interface IIntelligenceOrchestrator : IWorkflowActionExecutor
{
    /// <summary>
    /// Run ML models for predictions
    /// </summary>
    Task RunMLModelsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Update RL training
    /// </summary>
    Task UpdateRLTrainingAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Generate predictions
    /// </summary>
    Task GeneratePredictionsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Analyze intermarket correlations
    /// </summary>
    Task AnalyzeCorrelationsAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
}

/// <summary>
/// Interface for data collection operations
/// </summary>
public interface IDataOrchestrator : IWorkflowActionExecutor
{
    /// <summary>
    /// Collect market data
    /// </summary>
    Task CollectMarketDataAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Store historical data
    /// </summary>
    Task StoreHistoricalDataAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Generate daily reports
    /// </summary>
    Task GenerateDailyReportAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default);
}

/// <summary>
/// Interface for workflow scheduling
/// </summary>
public interface IWorkflowScheduler
{
    /// <summary>
    /// Schedule a workflow for execution
    /// </summary>
    Task ScheduleWorkflowAsync(UnifiedWorkflow workflow, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Unschedule a workflow
    /// </summary>
    Task UnscheduleWorkflowAsync(string workflowId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get next scheduled execution time for a workflow
    /// </summary>
    DateTime? GetNextExecution(string workflowId);
    
    /// <summary>
    /// Start the scheduler
    /// </summary>
    Task StartAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Stop the scheduler
    /// </summary>
    Task StopAsync(CancellationToken cancellationToken = default);
}