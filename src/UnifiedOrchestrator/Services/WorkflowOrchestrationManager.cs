using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.UnifiedOrchestrator.Interfaces;
using System.Collections.Concurrent;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Workflow Orchestration Manager for preventing workflow collisions and managing execution priorities
/// Coordinates workflow execution across the unified orchestrator system
/// </summary>
public class WorkflowOrchestrationManager : IWorkflowOrchestrationManager
{
    private readonly ILogger<WorkflowOrchestrationManager> _logger;
    private readonly ConcurrentDictionary<string, WorkflowLock> _workflowLocks = new();
    private readonly PriorityQueue<WorkflowTask, int> _taskQueue = new();
    private readonly ConcurrentDictionary<string, WorkflowConflict> _conflicts = new();
    private readonly SemaphoreSlim _tradingMutex = new(1, 1);
    private readonly Timer _conflictResolver;
    private readonly object _lockObject = new();
    private bool _disposed = false;

    public class WorkflowLock
    {
        public string WorkflowId { get; set; } = string.Empty;
        public string ResourceType { get; set; } = string.Empty;
        public DateTime AcquiredAt { get; set; }
        public DateTime ExpiresAt { get; set; }
        public int Priority { get; set; }
        public bool IsExclusive { get; set; }
    }

    public class WorkflowTask
    {
        public string TaskId { get; set; } = string.Empty;
        public string WorkflowName { get; set; } = string.Empty;
        public int Priority { get; set; }
        public Func<Task>? Action { get; set; }
        public List<string> RequiredResources { get; set; } = new();
        public DateTime ScheduledTime { get; set; }
        public TimeSpan Timeout { get; set; }
        public int RetryCount { get; set; }
    }

    public class WorkflowConflict
    {
        public string ConflictId { get; set; } = string.Empty;
        public List<string> ConflictingWorkflows { get; set; } = new();
        public string ResourceInConflict { get; set; } = string.Empty;
        public DateTime DetectedAt { get; set; }
        public string Resolution { get; set; } = string.Empty;
    }

    public class ConflictResolution
    {
        public DateTime Timestamp { get; set; }
        public List<string> ConflictsResolved { get; set; } = new();
        public List<string> Actions { get; set; } = new();
    }

    // Workflow priorities (lower number = higher priority)
    private readonly Dictionary<string, int> _workflowPriorities = new()
    {
        ["es-nq-critical-trading"] = 1,
        ["risk-management"] = 2,
        ["position-reconciliation"] = 3,
        ["ultimate-ml-rl-intel"] = 4,
        ["microstructure-analysis"] = 5,
        ["options-flow-analysis"] = 6,
        ["portfolio-heat-management"] = 7,
        ["intermarket-correlations"] = 8,
        ["failed-patterns-detection"] = 9,
        ["volatility-surface-analysis"] = 10,
        ["supply-demand-zones"] = 11,
        ["market-maker-positioning"] = 12,
        ["daily-report"] = 15,
        ["ml-trainer"] = 20,
        ["data-collection"] = 25
    };

    public WorkflowOrchestrationManager(ILogger<WorkflowOrchestrationManager> logger)
    {
        _logger = logger;
        _conflictResolver = new Timer(ResolveConflictsTimer, null, Timeout.Infinite, Timeout.Infinite);
        
        _logger.LogInformation("[Workflow-Orchestration] WorkflowOrchestrationManager initialized");
    }

    /// <summary>
    /// Initialize conflict resolution and monitoring
    /// </summary>
    public async Task InitializeAsync()
    {
        _logger.LogInformation("[Workflow-Orchestration] Starting conflict resolution services");
        
        // Start conflict resolver (every 30 seconds)
        _conflictResolver.Change(TimeSpan.Zero, TimeSpan.FromSeconds(30));
        
        await Task.CompletedTask;
    }

    /// <summary>
    /// Request workflow execution with conflict detection and queuing
    /// </summary>
    public async Task<bool> RequestWorkflowExecutionAsync(string workflowName, Func<Task> action, List<string>? requiredResources = null)
    {
        if (string.IsNullOrEmpty(workflowName))
            throw new ArgumentException("Workflow name cannot be null or empty", nameof(workflowName));
            
        if (action == null)
            throw new ArgumentNullException(nameof(action));

        var task = new WorkflowTask
        {
            TaskId = Guid.NewGuid().ToString(),
            WorkflowName = workflowName,
            Priority = _workflowPriorities.GetValueOrDefault(workflowName, 50),
            Action = action,
            RequiredResources = requiredResources ?? new List<string>(),
            ScheduledTime = DateTime.UtcNow,
            Timeout = TimeSpan.FromMinutes(5),
            RetryCount = 0
        };

        _logger.LogDebug("[Workflow-Orchestration] Requesting execution for: {WorkflowName} (Priority: {Priority})", 
            workflowName, task.Priority);

        // Check for conflicts
        if (await HasConflictsAsync(task))
        {
            // Queue the task
            lock (_lockObject)
            {
                _taskQueue.Enqueue(task, task.Priority);
            }
            
            _logger.LogInformation("[Workflow-Orchestration] Queued workflow due to conflicts: {WorkflowName}", workflowName);
            return false;
        }

        // Execute immediately if no conflicts
        return await ExecuteWorkflowAsync(task);
    }

    private async Task<bool> HasConflictsAsync(WorkflowTask task)
    {
        // Critical trading workflows get exclusive access
        if (task.WorkflowName.Contains("critical") || task.Priority <= 2)
        {
            var hasOtherWorkflows = _workflowLocks.Any();
            if (hasOtherWorkflows)
            {
                _logger.LogDebug("[Workflow-Orchestration] Critical workflow {WorkflowName} blocked by existing workflows", 
                    task.WorkflowName);
            }
            return hasOtherWorkflows;
        }

        // Check resource conflicts
        foreach (var resource in task.RequiredResources)
        {
            if (_workflowLocks.Values.Any(l => l.ResourceType == resource && l.IsExclusive))
            {
                // Log conflict
                var conflict = new WorkflowConflict
                {
                    ConflictId = Guid.NewGuid().ToString(),
                    ConflictingWorkflows = new List<string> { task.WorkflowName },
                    ResourceInConflict = resource,
                    DetectedAt = DateTime.UtcNow
                };

                _conflicts[conflict.ConflictId] = conflict;
                
                _logger.LogWarning("[Workflow-Orchestration] Resource conflict detected: {WorkflowName} needs {Resource}", 
                    task.WorkflowName, resource);
                    
                return true;
            }
        }

        // Check for trading decision conflicts
        if (task.RequiredResources.Contains("trading_decision"))
        {
            var canAcquire = await _tradingMutex.WaitAsync(0);
            if (!canAcquire)
            {
                _logger.LogDebug("[Workflow-Orchestration] Trading decision conflict for: {WorkflowName}", task.WorkflowName);
                return true;
            }
            // Release immediately if we acquired it during check
            _tradingMutex.Release();
        }

        return false;
    }

    private async Task<bool> ExecuteWorkflowAsync(WorkflowTask task)
    {
        var lockId = Guid.NewGuid().ToString();

        try
        {
            // Acquire locks
            await AcquireLocksAsync(task, lockId);

            _logger.LogInformation("[Workflow-Orchestration] Executing workflow: {WorkflowName} (Task: {TaskId})", 
                task.WorkflowName, task.TaskId);

            // Execute with timeout
            using var cts = new CancellationTokenSource(task.Timeout);

            await Task.Run(async () =>
            {
                try
                {
                    if (task.Action != null)
                    {
                        await task.Action();
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[Workflow-Orchestration] Workflow execution failed: {WorkflowName}", task.WorkflowName);
                    throw;
                }
            }, cts.Token);

            _logger.LogInformation("[Workflow-Orchestration] Workflow completed successfully: {WorkflowName}", task.WorkflowName);
            return true;
        }
        catch (OperationCanceledException)
        {
            _logger.LogWarning("[Workflow-Orchestration] Workflow timeout: {WorkflowName} after {Timeout}", 
                task.WorkflowName, task.Timeout);

            // Retry if allowed
            if (task.RetryCount < 3)
            {
                task.RetryCount++;
                lock (_lockObject)
                {
                    _taskQueue.Enqueue(task, task.Priority);
                }
                _logger.LogInformation("[Workflow-Orchestration] Queued for retry: {WorkflowName} (Attempt {RetryCount})", 
                    task.WorkflowName, task.RetryCount + 1);
            }

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Workflow-Orchestration] Workflow execution error: {WorkflowName}", task.WorkflowName);
            return false;
        }
        finally
        {
            // Release locks
            ReleaseLocks(lockId);

            // Process queued tasks
            _ = Task.Run(ProcessQueuedTasksAsync);
        }
    }

    private async Task AcquireLocksAsync(WorkflowTask task, string lockId)
    {
        // Acquire trading mutex for critical workflows
        if (task.Priority <= 3)
        {
            await _tradingMutex.WaitAsync();
            _logger.LogDebug("[Workflow-Orchestration] Acquired trading mutex for: {WorkflowName}", task.WorkflowName);
        }

        // Create workflow lock
        var workflowLock = new WorkflowLock
        {
            WorkflowId = lockId,
            ResourceType = task.WorkflowName,
            AcquiredAt = DateTime.UtcNow,
            ExpiresAt = DateTime.UtcNow.Add(task.Timeout),
            Priority = task.Priority,
            IsExclusive = task.Priority <= 3
        };

        _workflowLocks[lockId] = workflowLock;

        // Lock required resources
        foreach (var resource in task.RequiredResources)
        {
            var resourceLock = new WorkflowLock
            {
                WorkflowId = lockId,
                ResourceType = resource,
                AcquiredAt = DateTime.UtcNow,
                ExpiresAt = DateTime.UtcNow.Add(task.Timeout),
                Priority = task.Priority,
                IsExclusive = resource == "trading_decision"
            };

            _workflowLocks[$"{lockId}_{resource}"] = resourceLock;
        }
    }

    private void ReleaseLocks(string lockId)
    {
        // Release all locks with this ID
        var locksToRemove = _workflowLocks.Keys.Where(k => k.Contains(lockId)).ToList();

        foreach (var key in locksToRemove)
        {
            _workflowLocks.TryRemove(key, out _);
        }

        // Release trading mutex if held
        try
        {
            if (_tradingMutex.CurrentCount == 0)
            {
                _tradingMutex.Release();
            }
        }
        catch (SemaphoreFullException)
        {
            // Mutex wasn't held
        }
    }

    private async Task ProcessQueuedTasksAsync()
    {
        try
        {
            WorkflowTask? task;
            int priority;
            
            lock (_lockObject)
            {
                if (!_taskQueue.TryDequeue(out task, out priority))
                {
                    return;
                }
            }

            if (task != null && !await HasConflictsAsync(task))
            {
                _logger.LogInformation("[Workflow-Orchestration] Processing queued task: {WorkflowName}", task.WorkflowName);
                _ = Task.Run(async () => await ExecuteWorkflowAsync(task));
            }
            else if (task != null)
            {
                // Re-queue if still conflicts
                lock (_lockObject)
                {
                    _taskQueue.Enqueue(task, priority);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Workflow-Orchestration] Error processing queued tasks");
        }
    }

    private void ResolveConflictsTimer(object? state)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                await ResolveConflictsAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[Workflow-Orchestration] Error in conflict resolution timer");
            }
        });
    }

    public async Task<ConflictResolution> ResolveConflictsAsync()
    {
        var resolution = new ConflictResolution
        {
            Timestamp = DateTime.UtcNow,
            ConflictsResolved = new List<string>(),
            Actions = new List<string>()
        };

        try
        {
            // Clear expired locks
            var expired = _workflowLocks.Where(kvp => kvp.Value.ExpiresAt < DateTime.UtcNow).ToList();
            foreach (var kvp in expired)
            {
                _workflowLocks.TryRemove(kvp.Key, out _);
                resolution.Actions.Add($"Cleared expired lock: {kvp.Value.ResourceType}");
                _logger.LogWarning("[Workflow-Orchestration] Cleared expired lock: {ResourceType}", kvp.Value.ResourceType);
            }

            // Resolve resource conflicts by priority
            var conflicts = _conflicts.Values.ToList();
            foreach (var conflict in conflicts)
            {
                var winner = conflict.ConflictingWorkflows
                    .OrderBy(w => _workflowPriorities.GetValueOrDefault(w, 100))
                    .First();

                conflict.Resolution = $"Priority given to {winner}";
                resolution.ConflictsResolved.Add(conflict.Resolution);
                
                _logger.LogInformation("[Workflow-Orchestration] Resolved conflict: {Resolution}", conflict.Resolution);
            }

            // Process queued tasks if locks were freed
            if (expired.Any())
            {
                _ = Task.Run(ProcessQueuedTasksAsync);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Workflow-Orchestration] Error during conflict resolution");
        }

        return resolution;
    }

    /// <summary>
    /// Get current workflow orchestration status
    /// </summary>
    public WorkflowOrchestrationStatus GetStatus()
    {
        return new WorkflowOrchestrationStatus
        {
            ActiveLocks = _workflowLocks.Count,
            QueuedTasks = _taskQueue.Count,
            ActiveConflicts = _conflicts.Count,
            TradingMutexAvailable = _tradingMutex.CurrentCount > 0
        };
    }

    public class WorkflowOrchestrationStatus
    {
        public int ActiveLocks { get; set; }
        public int QueuedTasks { get; set; }
        public int ActiveConflicts { get; set; }
        public bool TradingMutexAvailable { get; set; }
    }

    public void Dispose()
    {
        if (_disposed) return;

        _logger.LogInformation("[Workflow-Orchestration] Disposing WorkflowOrchestrationManager");

        _disposed = true;

        _conflictResolver?.Dispose();
        _tradingMutex?.Dispose();

        // Clear all locks and queues
        _workflowLocks.Clear();
        _conflicts.Clear();

        GC.SuppressFinalize(this);
    }
}