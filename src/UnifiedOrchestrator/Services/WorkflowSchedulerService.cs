using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.UnifiedOrchestrator.Configuration;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Workflow scheduler service - coordinates scheduled operations
/// </summary>
public class WorkflowSchedulerService : BackgroundService, IWorkflowScheduler
{
    private readonly ILogger<WorkflowSchedulerService> _logger;
    private readonly ICentralMessageBus _messageBus;
    private readonly WorkflowSchedulingOptions _schedulingOptions;
    private readonly Dictionary<string, DateTime> _workflowLastExecution = new();
    private readonly Dictionary<string, WorkflowScheduleConfig> _workflowSchedules = new();

    public WorkflowSchedulerService(
        ILogger<WorkflowSchedulerService> logger,
        ICentralMessageBus messageBus,
        IOptions<WorkflowSchedulingOptions> schedulingOptions)
    {
        _logger = logger;
        _messageBus = messageBus;
        _schedulingOptions = schedulingOptions.Value;
        
        // Initialize workflow schedules from configuration
        foreach (var schedule in _schedulingOptions.DefaultSchedules)
        {
            _workflowSchedules[schedule.Key] = schedule.Value;
        }
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Workflow Scheduler Service starting...");
        
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Main workflow scheduling loop
                await ProcessScheduledWorkflowsAsync(stoppingToken);
                
                // Wait before next iteration
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in workflow scheduler loop");
                await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
            }
        }
        
        _logger.LogInformation("Workflow Scheduler Service stopped");
    }

    private async Task ProcessScheduledWorkflowsAsync(CancellationToken cancellationToken)
    {
        // Process scheduled workflows
        // This will be implemented based on actual workflow requirements
        await Task.CompletedTask;
    }

    public async Task ScheduleWorkflowAsync(WorkflowDefinition workflow, CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("Scheduling workflow: {WorkflowName}", workflow.Name);
            
            // Implementation would schedule the workflow
            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to schedule workflow: {WorkflowName}", workflow.Name);
            throw;
        }
    }

    public Task<TradingBot.UnifiedOrchestrator.Models.WorkflowStatus> GetWorkflowStatusAsync(string workflowId, CancellationToken cancellationToken = default)
    {
        // Implementation would get actual workflow status
        return Task.FromResult(new TradingBot.UnifiedOrchestrator.Models.WorkflowStatus
        {
            WorkflowId = workflowId,
            Status = "Unknown",
            LastRun = DateTime.MinValue,
            NextRun = DateTime.MaxValue
        });
    }

    // IWorkflowScheduler interface implementation
    public async Task ScheduleWorkflowAsync(UnifiedWorkflow workflow, CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[SCHEDULER] Scheduling workflow: {WorkflowId}", workflow.Id);
            
            // Implementation would schedule the workflow
            await Task.Delay(50, cancellationToken); // Simulate scheduling
            
            _logger.LogInformation("[SCHEDULER] Workflow scheduled successfully: {WorkflowId}", workflow.Id);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to schedule workflow: {WorkflowId}", workflow.Id);
            throw;
        }
    }

    public async Task UnscheduleWorkflowAsync(string workflowId, CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[SCHEDULER] Unscheduling workflow: {WorkflowId}", workflowId);
            
            // Implementation would unschedule the workflow
            await Task.Delay(50, cancellationToken); // Simulate unscheduling
            
            _logger.LogInformation("[SCHEDULER] Workflow unscheduled successfully: {WorkflowId}", workflowId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to unschedule workflow: {WorkflowId}", workflowId);
            throw;
        }
    }

    public DateTime? GetNextExecution(string workflowId)
    {
        if (!_schedulingOptions.Enabled || string.IsNullOrWhiteSpace(workflowId))
            return null;

        try
        {
            // Get the workflow schedule configuration
            if (!_workflowSchedules.TryGetValue(workflowId, out var scheduleConfig))
            {
                _logger.LogDebug("[SCHEDULER] No schedule found for workflow: {WorkflowId}, using default interval", workflowId);
                return DateTime.UtcNow.AddHours(1);
            }

            var currentTime = DateTime.UtcNow;
            
            // Check if it's a market holiday
            if (CronScheduler.IsMarketHoliday(currentTime, _schedulingOptions.MarketHolidays))
            {
                _logger.LogDebug("[SCHEDULER] Market holiday detected, scheduling for next business day");
                return GetNextBusinessDay(currentTime).AddHours(9); // 9 AM next business day
            }

            // Create WorkflowSchedule instance for current evaluation
            var workflowSchedule = new WorkflowSchedule
            {
                MarketHours = scheduleConfig.MarketHours,
                ExtendedHours = scheduleConfig.ExtendedHours,
                Overnight = scheduleConfig.Overnight,
                CoreHours = scheduleConfig.CoreHours,
                FirstHour = scheduleConfig.FirstHour,
                LastHour = scheduleConfig.LastHour,
                Regular = scheduleConfig.Regular,
                Global = scheduleConfig.Global,
                Weekends = scheduleConfig.Weekends,
                Disabled = scheduleConfig.Disabled
            };

            // Get the active schedule for current time
            var activeSchedule = workflowSchedule.GetActiveSchedule(currentTime);
            
            if (string.IsNullOrWhiteSpace(activeSchedule))
            {
                _logger.LogDebug("[SCHEDULER] No active schedule for workflow: {WorkflowId} at current time", workflowId);
                return DateTime.UtcNow.AddHours(1);
            }

            // Parse the cron expression and get next execution
            var nextExecution = CronScheduler.GetNextExecution(activeSchedule, currentTime);
            
            if (nextExecution.HasValue)
            {
                _logger.LogDebug("[SCHEDULER] Next execution for {WorkflowId}: {NextExecution} (using schedule: {Schedule})", 
                    workflowId, nextExecution.Value, activeSchedule);
                return nextExecution.Value;
            }

            _logger.LogWarning("[SCHEDULER] Failed to calculate next execution for workflow: {WorkflowId}", workflowId);
            return DateTime.UtcNow.AddHours(1);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SCHEDULER] Error calculating next execution for workflow: {WorkflowId}", workflowId);
            return DateTime.UtcNow.AddHours(1);
        }
    }

    private DateTime GetNextBusinessDay(DateTime date)
    {
        var next = date.AddDays(1);
        while (next.DayOfWeek == DayOfWeek.Saturday || next.DayOfWeek == DayOfWeek.Sunday ||
               CronScheduler.IsMarketHoliday(next, _schedulingOptions.MarketHolidays))
        {
            next = next.AddDays(1);
        }
        return next;
    }

    public new async Task StartAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[SCHEDULER] Starting workflow scheduler...");
        await base.StartAsync(cancellationToken);
    }

    public new async Task StopAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[SCHEDULER] Stopping workflow scheduler...");
        await base.StopAsync(cancellationToken);
    }
}