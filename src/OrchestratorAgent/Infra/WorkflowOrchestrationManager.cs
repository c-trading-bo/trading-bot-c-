using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Infra;

/// <summary>
/// COMPONENT 7: WORKFLOW COLLISION PREVENTION
/// Manages workflow execution to prevent conflicts and deadlocks through priority-based scheduling
/// </summary>
public class WorkflowOrchestrationManager : IDisposable
{
    private readonly ILogger<WorkflowOrchestrationManager> _logger;
    private readonly ConcurrentDictionary<string, WorkflowLock> _workflowLocks = new();
    private readonly PriorityQueue<WorkflowTask, int> _taskQueue = new();
    private readonly ConcurrentDictionary<string, WorkflowConflict> _conflicts = new();
    private readonly SemaphoreSlim _tradingMutex = new(1, 1);
    private readonly Timer? _conflictResolver;
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

    // Parameterless constructor for dependency injection
    public WorkflowOrchestrationManager() : this(null) { }

    public WorkflowOrchestrationManager(ILogger<WorkflowOrchestrationManager>? logger)
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<WorkflowOrchestrationManager>.Instance;
        
        // Start conflict resolution timer
        _conflictResolver = new Timer(async _ => await ResolveConflicts(), null, 
            TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
    }

    public async Task<bool> RequestWorkflowExecution(string workflowName, Func<Task> action, List<string>? requiredResources = null)
    {
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

        _logger.LogDebug("[Workflow] Request: {WorkflowName} priority {Priority}", workflowName, task.Priority);

        // Check for conflicts
        if (await HasConflicts(task))
        {
            // Queue the task
            lock (_taskQueue)
            {
                _taskQueue.Enqueue(task, task.Priority);
            }
            _logger.LogInformation("[Workflow] Queued {WorkflowName} due to conflicts", workflowName);
            return false;
        }

        // Execute immediately if no conflicts
        return await ExecuteWorkflow(task);
    }

    private async Task<bool> HasConflicts(WorkflowTask task)
    {
        // Critical trading workflows get exclusive access
        if (task.WorkflowName.Contains("critical") || task.Priority <= 2)
        {
            var hasActiveLocks = _workflowLocks.Any();
            if (hasActiveLocks)
            {
                _logger.LogDebug("[Workflow] Critical workflow {Name} blocked by active locks", task.WorkflowName);
            }
            return hasActiveLocks;
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
                _logger.LogWarning("[Workflow] Resource conflict detected: {Workflow} needs {Resource}", 
                    task.WorkflowName, resource);
                return true;
            }
        }

        // Check for trading decision conflicts
        if (task.RequiredResources.Contains("trading_decision"))
        {
            var canAcquire = await _tradingMutex.WaitAsync(0);
            if (canAcquire)
            {
                _tradingMutex.Release();
                return false;
            }
            return true;
        }

        return false;
    }

    private async Task<bool> ExecuteWorkflow(WorkflowTask task)
    {
        var lockId = Guid.NewGuid().ToString();

        try
        {
            // Acquire locks
            await AcquireLocks(task, lockId);

            _logger.LogInformation("[Workflow] Executing {WorkflowName} (ID: {TaskId})", 
                task.WorkflowName, task.TaskId);

            // Execute with timeout
            using var cts = new CancellationTokenSource(task.Timeout);

            await Task.Run(async () =>
            {
                try
                {
                    if (task.Action != null)
                        await task.Action();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[Workflow] Execution failed for {WorkflowName}", task.WorkflowName);
                    throw;
                }
            }, cts.Token);

            _logger.LogInformation("[Workflow] Completed {WorkflowName} successfully", task.WorkflowName);
            return true;
        }
        catch (OperationCanceledException)
        {
            _logger.LogWarning("[Workflow] Timeout executing {WorkflowName}", task.WorkflowName);

            // Retry if allowed
            if (task.RetryCount < 3)
            {
                task.RetryCount++;
                lock (_taskQueue)
                {
                    _taskQueue.Enqueue(task, task.Priority);
                }
                _logger.LogInformation("[Workflow] Retrying {WorkflowName} (attempt {Retry})", 
                    task.WorkflowName, task.RetryCount + 1);
            }

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Workflow] Failed to execute {WorkflowName}", task.WorkflowName);
            return false;
        }
        finally
        {
            // Release locks
            ReleaseLocks(lockId);

            // Process queued tasks
            await ProcessQueuedTasks();
        }
    }

    private async Task AcquireLocks(WorkflowTask task, string lockId)
    {
        // Acquire trading mutex for critical workflows
        if (task.Priority <= 3)
        {
            await _tradingMutex.WaitAsync();
            _logger.LogDebug("[Workflow] Acquired trading mutex for {WorkflowName}", task.WorkflowName);
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

        await Task.CompletedTask;
    }

    private void ReleaseLocks(string lockId)
    {
        // Release all locks with this ID
        var locksToRemove = _workflowLocks.Keys.Where(k => k.Contains(lockId)).ToList();

        foreach (var key in locksToRemove)
        {
            if (_workflowLocks.TryRemove(key, out var removedLock))
            {
                _logger.LogDebug("[Workflow] Released lock: {Resource}", removedLock.ResourceType);
            }
        }

        // Release trading mutex if held
        try
        {
            _tradingMutex.Release();
        }
        catch
        {
            // Mutex wasn't held by this thread
        }
    }

    private async Task ProcessQueuedTasks()
    {
        var processed = 0;
        const int maxBatch = 5; // Prevent infinite processing

        while (processed < maxBatch)
        {
            WorkflowTask? task = null;
            
            lock (_taskQueue)
            {
                if (_taskQueue.Count == 0)
                    break;
                    
                if (_taskQueue.TryDequeue(out task, out var priority))
                {
                    // Check if still has conflicts
                    if (HasConflicts(task).Result)
                    {
                        // Re-queue if still conflicts
                        _taskQueue.Enqueue(task, priority);
                        break; // Don't process more if conflicts persist
                    }
                }
            }

            if (task != null)
            {
                _logger.LogInformation("[Workflow] Processing queued task: {WorkflowName}", task.WorkflowName);
                _ = Task.Run(async () => await ExecuteWorkflow(task));
                processed++;
            }
            else
            {
                break;
            }
        }

        await Task.CompletedTask;
    }

    public async Task<ConflictResolution> ResolveConflicts()
    {
        var resolution = new ConflictResolution
        {
            Timestamp = DateTime.UtcNow,
            ConflictsResolved = new List<string>(),
            Actions = new List<string>()
        };

        try
        {
            // Detect deadlocks
            var deadlocks = DetectDeadlocks();
            foreach (var deadlock in deadlocks)
            {
                // Kill lower priority workflow
                var victim = deadlock.OrderByDescending(w => _workflowPriorities.GetValueOrDefault(w, 100)).First();
                await KillWorkflow(victim);

                resolution.ConflictsResolved.Add($"Resolved deadlock by killing {victim}");
                _logger.LogWarning("[Workflow] Resolved deadlock by terminating {Victim}", victim);
            }

            // Clear expired locks
            var expired = _workflowLocks.Where(kvp => kvp.Value.ExpiresAt < DateTime.UtcNow).ToList();
            foreach (var kvp in expired)
            {
                if (_workflowLocks.TryRemove(kvp.Key, out _))
                {
                    resolution.Actions.Add($"Cleared expired lock: {kvp.Value.ResourceType}");
                    _logger.LogInformation("[Workflow] Cleared expired lock: {Resource}", kvp.Value.ResourceType);
                }
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
                _logger.LogInformation("[Workflow] Conflict resolved: {Resolution}", conflict.Resolution);
            }

            // Clear resolved conflicts
            _conflicts.Clear();

            // Process any queued tasks now that conflicts are resolved
            await ProcessQueuedTasks();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Workflow] Error during conflict resolution");
        }

        return resolution;
    }

    private List<List<string>> DetectDeadlocks()
    {
        var deadlocks = new List<List<string>>();
        
        // Simple deadlock detection based on circular wait conditions
        var waitGraph = BuildWaitGraph();
        
        // Use basic cycle detection - in a production system would use more sophisticated algorithms
        var visited = new HashSet<string>();
        var recursionStack = new HashSet<string>();
        
        foreach (var node in waitGraph.Keys)
        {
            if (!visited.Contains(node))
            {
                var cycle = new List<string>();
                if (DetectCycle(node, waitGraph, visited, recursionStack, cycle))
                {
                    if (cycle.Count > 1)
                        deadlocks.Add(cycle);
                }
            }
        }
        
        return deadlocks;
    }

    private Dictionary<string, List<string>> BuildWaitGraph()
    {
        var graph = new Dictionary<string, List<string>>();
        
        // Build a wait-for graph based on current locks and conflicts
        foreach (var conflict in _conflicts.Values)
        {
            foreach (var workflow in conflict.ConflictingWorkflows)
            {
                if (!graph.ContainsKey(workflow))
                    graph[workflow] = new List<string>();
                    
                // Add dependencies based on resource conflicts
                var lockHolder = _workflowLocks.Values
                    .FirstOrDefault(l => l.ResourceType == conflict.ResourceInConflict)?
                    .WorkflowId;
                    
                if (!string.IsNullOrEmpty(lockHolder) && lockHolder != workflow)
                {
                    graph[workflow].Add(lockHolder);
                }
            }
        }
        
        return graph;
    }

    private bool DetectCycle(string node, Dictionary<string, List<string>> graph, 
        HashSet<string> visited, HashSet<string> recursionStack, List<string> cycle)
    {
        visited.Add(node);
        recursionStack.Add(node);
        cycle.Add(node);
        
        if (graph.ContainsKey(node))
        {
            foreach (var neighbor in graph[node])
            {
                if (!visited.Contains(neighbor))
                {
                    if (DetectCycle(neighbor, graph, visited, recursionStack, cycle))
                        return true;
                }
                else if (recursionStack.Contains(neighbor))
                {
                    // Cycle detected
                    return true;
                }
            }
        }
        
        recursionStack.Remove(node);
        cycle.Remove(node);
        return false;
    }

    private async Task KillWorkflow(string workflowName)
    {
        // Find and remove locks for this workflow
        var workflowLocks = _workflowLocks.Where(kvp => 
            kvp.Value.ResourceType == workflowName).ToList();
            
        foreach (var kvp in workflowLocks)
        {
            _workflowLocks.TryRemove(kvp.Key, out _);
        }
        
        _logger.LogWarning("[Workflow] Killed workflow: {WorkflowName}", workflowName);
        await Task.CompletedTask;
    }

    public WorkflowStatus GetWorkflowStatus()
    {
        return new WorkflowStatus
        {
            ActiveLocks = _workflowLocks.Count,
            QueuedTasks = _taskQueue.Count,
            ActiveConflicts = _conflicts.Count,
            LockedResources = _workflowLocks.Values.Select(l => l.ResourceType).Distinct().ToList()
        };
    }

    public class WorkflowStatus
    {
        public int ActiveLocks { get; set; }
        public int QueuedTasks { get; set; }
        public int ActiveConflicts { get; set; }
        public List<string> LockedResources { get; set; } = new();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            
            _conflictResolver?.Dispose();
            _tradingMutex?.Dispose();
            
            _logger.LogInformation("[Workflow] Workflow Orchestration Manager disposed");
        }
    }
}