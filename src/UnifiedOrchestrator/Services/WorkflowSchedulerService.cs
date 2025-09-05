using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using System.Collections.Concurrent;
using System.Text.RegularExpressions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Unified workflow scheduler that handles all workflow timing and execution
/// </summary>
public class WorkflowSchedulerService : IWorkflowScheduler, IDisposable
{
    private readonly ILogger<WorkflowSchedulerService> _logger;
    private readonly IServiceProvider _serviceProvider;
    
    private readonly ConcurrentDictionary<string, ScheduledWorkflow> _scheduledWorkflows = new();
    private readonly Timer _schedulerTimer;
    private readonly object _lockObject = new();
    private bool _isRunning = false;

    public WorkflowSchedulerService(
        ILogger<WorkflowSchedulerService> logger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        
        // Create timer that checks every minute for scheduled workflows
        _schedulerTimer = new Timer(ExecuteScheduledWorkflows, null, Timeout.Infinite, Timeout.Infinite);
    }

    #region IWorkflowScheduler Implementation

    public async Task ScheduleWorkflowAsync(UnifiedWorkflow workflow, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(workflow.Id))
            throw new ArgumentException("Workflow ID cannot be null or empty", nameof(workflow));

        var scheduledWorkflow = new ScheduledWorkflow
        {
            Workflow = workflow,
            NextExecution = CalculateNextExecution(workflow),
            IsScheduled = true
        };

        _scheduledWorkflows.AddOrUpdate(workflow.Id, scheduledWorkflow, (key, existing) => scheduledWorkflow);
        
        _logger.LogInformation("📅 Scheduled workflow: {WorkflowId} - Next execution: {NextExecution}", 
            workflow.Id, scheduledWorkflow.NextExecution);

        await Task.CompletedTask;
    }

    public async Task UnscheduleWorkflowAsync(string workflowId, CancellationToken cancellationToken = default)
    {
        if (_scheduledWorkflows.TryRemove(workflowId, out var scheduledWorkflow))
        {
            scheduledWorkflow.IsScheduled = false;
            _logger.LogInformation("📅 Unscheduled workflow: {WorkflowId}", workflowId);
        }

        await Task.CompletedTask;
    }

    public DateTime? GetNextExecution(string workflowId)
    {
        return _scheduledWorkflows.TryGetValue(workflowId, out var scheduledWorkflow) 
            ? scheduledWorkflow.NextExecution 
            : null;
    }

    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        if (_isRunning) return;

        _logger.LogInformation("⏰ Starting workflow scheduler...");
        
        _isRunning = true;
        
        // Start the scheduler timer to check every minute
        _schedulerTimer.Change(TimeSpan.Zero, TimeSpan.FromMinutes(1));
        
        _logger.LogInformation("✅ Workflow scheduler started");
        await Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        if (!_isRunning) return;

        _logger.LogInformation("⏰ Stopping workflow scheduler...");
        
        _isRunning = false;
        
        // Stop the scheduler timer
        _schedulerTimer.Change(Timeout.Infinite, Timeout.Infinite);
        
        _logger.LogInformation("✅ Workflow scheduler stopped");
        await Task.CompletedTask;
    }

    #endregion

    #region Private Methods

    private void ExecuteScheduledWorkflows(object? state)
    {
        if (!_isRunning) return;

        lock (_lockObject)
        {
            var now = DateTime.UtcNow;
            var workflowsToExecute = new List<ScheduledWorkflow>();

            // Find workflows that are due for execution
            foreach (var scheduledWorkflow in _scheduledWorkflows.Values)
            {
                if (scheduledWorkflow.IsScheduled && 
                    scheduledWorkflow.NextExecution.HasValue && 
                    scheduledWorkflow.NextExecution <= now)
                {
                    workflowsToExecute.Add(scheduledWorkflow);
                }
            }

            // Execute workflows asynchronously
            foreach (var scheduledWorkflow in workflowsToExecute)
            {
                _ = Task.Run(async () =>
                {
                    try
                    {
                        await ExecuteWorkflowAsync(scheduledWorkflow);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "❌ Error executing scheduled workflow: {WorkflowId}", 
                            scheduledWorkflow.Workflow.Id);
                    }
                });
            }
        }
    }

    private async Task ExecuteWorkflowAsync(ScheduledWorkflow scheduledWorkflow)
    {
        var workflow = scheduledWorkflow.Workflow;
        
        _logger.LogInformation("🔄 Executing scheduled workflow: {WorkflowId} - {WorkflowName}", 
            workflow.Id, workflow.Name);

        try
        {
            // Get the unified orchestrator to execute the workflow
            var orchestrator = _serviceProvider.GetService(typeof(IUnifiedOrchestrator)) as IUnifiedOrchestrator;
            if (orchestrator != null)
            {
                var result = await orchestrator.ExecuteWorkflowAsync(workflow.Id);
                
                if (result.Success)
                {
                    _logger.LogInformation("✅ Scheduled workflow completed: {WorkflowId} in {Duration}ms", 
                        workflow.Id, result.Duration.TotalMilliseconds);
                }
                else
                {
                    _logger.LogError("❌ Scheduled workflow failed: {WorkflowId} - {Error}", 
                        workflow.Id, result.ErrorMessage);
                }
            }

            // Calculate next execution time
            scheduledWorkflow.NextExecution = CalculateNextExecution(workflow);
            scheduledWorkflow.LastExecution = DateTime.UtcNow;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error in scheduled workflow execution: {WorkflowId}", workflow.Id);
            
            // Schedule retry in 5 minutes for failed workflows
            scheduledWorkflow.NextExecution = DateTime.UtcNow.AddMinutes(5);
        }
    }

    private DateTime? CalculateNextExecution(UnifiedWorkflow workflow)
    {
        var now = DateTime.UtcNow;
        var schedule = workflow.Schedule.GetActiveSchedule(now);
        
        if (string.IsNullOrEmpty(schedule))
            return null;

        try
        {
            return ParseCronExpression(schedule, now);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ Could not parse schedule for workflow {WorkflowId}: {Schedule}", 
                workflow.Id, schedule);
            return null;
        }
    }

    private DateTime ParseCronExpression(string cronExpression, DateTime fromTime)
    {
        // Simple cron parser for common patterns used in the orchestrators
        // Format: minute hour day month dayOfWeek
        var parts = cronExpression.Split(' ');
        if (parts.Length < 5) return fromTime.AddMinutes(5); // Default fallback

        var minute = parts[0];
        var hour = parts[1];
        
        var nextExecution = fromTime.AddMinutes(1); // Start from next minute
        
        // Handle */N patterns (every N minutes/hours)
        if (minute.StartsWith("*/"))
        {
            if (int.TryParse(minute.Substring(2), out var minuteInterval))
            {
                var minutesToAdd = minuteInterval - (nextExecution.Minute % minuteInterval);
                if (minutesToAdd == minuteInterval) minutesToAdd = 0;
                nextExecution = nextExecution.AddMinutes(minutesToAdd);
                return new DateTime(nextExecution.Year, nextExecution.Month, nextExecution.Day, 
                    nextExecution.Hour, nextExecution.Minute, 0, DateTimeKind.Utc);
            }
        }
        
        // Handle specific minute values
        if (int.TryParse(minute, out var specificMinute))
        {
            nextExecution = new DateTime(nextExecution.Year, nextExecution.Month, nextExecution.Day, 
                nextExecution.Hour, specificMinute, 0, DateTimeKind.Utc);
            
            if (nextExecution <= fromTime)
                nextExecution = nextExecution.AddHours(1);
        }
        
        // Handle hour constraints
        if (hour.Contains("-")) // Range like "9-16"
        {
            var hourParts = hour.Split('-');
            if (hourParts.Length == 2 && 
                int.TryParse(hourParts[0], out var startHour) && 
                int.TryParse(hourParts[1], out var endHour))
            {
                if (nextExecution.Hour < startHour)
                {
                    nextExecution = new DateTime(nextExecution.Year, nextExecution.Month, nextExecution.Day, 
                        startHour, nextExecution.Minute, 0, DateTimeKind.Utc);
                }
                else if (nextExecution.Hour > endHour)
                {
                    nextExecution = new DateTime(nextExecution.Year, nextExecution.Month, nextExecution.Day, 
                        startHour, nextExecution.Minute, 0, DateTimeKind.Utc).AddDays(1);
                }
            }
        }

        return nextExecution;
    }

    #endregion

    #region Supporting Classes

    private class ScheduledWorkflow
    {
        public UnifiedWorkflow Workflow { get; set; } = new();
        public DateTime? NextExecution { get; set; }
        public DateTime? LastExecution { get; set; }
        public bool IsScheduled { get; set; }
    }

    #endregion

    #region IDisposable

    public void Dispose()
    {
        _schedulerTimer?.Dispose();
    }

    #endregion
}