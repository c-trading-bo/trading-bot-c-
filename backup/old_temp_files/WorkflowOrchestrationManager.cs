// ================================================================================
        public WorkflowOrchestrationManager(ILogger<WorkflowOrchestrationManager> logger)
        {
            _logger = logger;
            _conflictResolver = new Timer(ResolveConflicts, null, TimeSpan.Zero, TimeSpan.FromSeconds(15));
        }

        public Task InitializeAsync()
        {
            _logger.LogInformation("🛠️ Initializing Workflow Orchestration Manager...");
            
            // Start deadlock monitoring
            _deadlockMonitorTimer = new Timer(MonitorDeadlocks, null, TimeSpan.Zero, TimeSpan.FromSeconds(30));
            
            _logger.LogInformation("✅ Workflow Orchestration Manager initialized with collision prevention");
            return Task.CompletedTask;
        }

        public async Task<bool> IsWorkflowConflicted(string workflowId)
        {
            return _conflicts.ContainsKey(workflowId) || await HasResourceConflict(workflowId);
        }

        public async Task AcquireWorkflowLock(string workflowId)
        {
            await _tradingMutex.WaitAsync();
            try
            {
                var lockObj = new WorkflowLock
                {
                    WorkflowId = workflowId,
                    ResourceType = "trading",
                    AcquiredAt = DateTime.UtcNow,
                    ExpiresAt = DateTime.UtcNow.AddMinutes(30),
                    Priority = 1,
                    IsExclusive = true
                };
                
                _workflowLocks[workflowId] = lockObj;
                _logger.LogInformation("🔒 Acquired workflow lock for {WorkflowId}", workflowId);
            }
            finally
            {
                _tradingMutex.Release();
            }
        }

        public async Task ReleaseWorkflowLock(string workflowId)
        {
            await _tradingMutex.WaitAsync();
            try
            {
                if (_workflowLocks.TryRemove(workflowId, out var lockObj))
                {
                    _logger.LogInformation("🔓 Released workflow lock for {WorkflowId}", workflowId);
                }
            }
            finally
            {
                _tradingMutex.Release();
            }
        }

        private async Task<bool> HasResourceConflict(string workflowId)
        {
            // Check if workflow conflicts with any currently running workflows
            foreach (var lockObj in _workflowLocks.Values)
            {
                if (lockObj.WorkflowId != workflowId && lockObj.IsExclusive)
                {
                    return true;
                }
            }
            return false;
        }MPONENT 7: WORKFLOW COLLISION PREVENTION
// ================================================================================
// File: WorkflowOrchestrationManager.cs
// Purpose: Advanced workflow orchestration with collision prevention and priority queuing
// Integration: Enhances existing WorkflowSchedulerService.cs
// ================================================================================

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services
{
    public class WorkflowOrchestrationManager
    {
        private readonly ConcurrentDictionary<string, WorkflowLock> _workflowLocks = new();
        private readonly PriorityQueue<WorkflowTask, int> _taskQueue = new();
        private readonly ConcurrentDictionary<string, WorkflowConflict> _conflicts = new();
        private readonly SemaphoreSlim _tradingMutex = new(1, 1);
        private readonly Timer _conflictResolver;
        private readonly ILogger<WorkflowOrchestrationManager> _logger;
        
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
            public Func<Task> Action { get; set; } = null!;
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
            _conflictResolver = new Timer(async _ => await ResolveConflicts(), null, TimeSpan.Zero, TimeSpan.FromSeconds(30));
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
            
            _logger.LogInformation($"[WorkflowOrchestration] Requested execution: {workflowName} (Priority: {task.Priority})");
            
            // Check for conflicts
            if (await HasConflicts(task))
            {
                // Queue the task
                _taskQueue.Enqueue(task, task.Priority);
                _logger.LogWarning($"[WorkflowOrchestration] Workflow {workflowName} queued due to conflicts");
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
                return _workflowLocks.Any();
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
                    _logger.LogWarning($"[WorkflowOrchestration] Resource conflict detected: {resource}");
                    return true;
                }
            }
            
            // Check for trading decision conflicts
            if (task.RequiredResources.Contains("trading_decision"))
            {
                return !await _tradingMutex.WaitAsync(0);
            }
            
            return false;
        }
        
        private async Task<bool> ExecuteWorkflow(WorkflowTask task)
        {
            var lockId = Guid.NewGuid().ToString();
            
            try
            {
                _logger.LogInformation($"[WorkflowOrchestration] Starting execution: {task.WorkflowName}");
                
                // Acquire locks
                await AcquireLocks(task, lockId);
                
                // Execute with timeout
                using var cts = new CancellationTokenSource(task.Timeout);
                
                await Task.Run(async () =>
                {
                    try
                    {
                        await task.Action();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, $"[WorkflowOrchestration] Workflow {task.WorkflowName} failed");
                        throw;
                    }
                }, cts.Token);
                
                _logger.LogInformation($"[WorkflowOrchestration] Completed: {task.WorkflowName}");
                return true;
            }
            catch (OperationCanceledException)
            {
                _logger.LogWarning($"[WorkflowOrchestration] Workflow {task.WorkflowName} timed out");
                
                // Retry if allowed
                if (task.RetryCount < 3)
                {
                    task.RetryCount++;
                    _taskQueue.Enqueue(task, task.Priority);
                }
                
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
            
            _logger.LogDebug($"[WorkflowOrchestration] Acquired locks for {task.WorkflowName}");
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
                _tradingMutex.Release();
            }
            catch
            {
                // Mutex wasn't held
            }
            
            _logger.LogDebug($"[WorkflowOrchestration] Released locks for {lockId}");
        }
        
        private async Task ProcessQueuedTasks()
        {
            while (_taskQueue.Count > 0)
            {
                if (_taskQueue.TryDequeue(out var task, out var priority))
                {
                    if (!await HasConflicts(task))
                    {
                        _logger.LogInformation($"[WorkflowOrchestration] Processing queued task: {task.WorkflowName}");
                        _ = Task.Run(async () => await ExecuteWorkflow(task));
                    }
                    else
                    {
                        // Re-queue if still conflicts
                        _taskQueue.Enqueue(task, priority);
                        break;
                    }
                }
            }
        }
        
        public async Task<ConflictResolution> ResolveConflicts()
        {
            var resolution = new ConflictResolution
            {
                Timestamp = DateTime.UtcNow,
                ConflictsResolved = new List<string>(),
                Actions = new List<string>()
            };
            
            // Detect deadlocks
            var deadlocks = DetectDeadlocks();
            foreach (var deadlock in deadlocks)
            {
                // Kill lower priority workflow
                var victim = deadlock.OrderByDescending(w => _workflowPriorities.GetValueOrDefault(w, 100)).First();
                await KillWorkflow(victim);
                
                resolution.ConflictsResolved.Add($"Resolved deadlock by killing {victim}");
                _logger.LogWarning($"[WorkflowOrchestration] Resolved deadlock by terminating {victim}");
            }
            
            // Clear expired locks
            var expired = _workflowLocks.Where(kvp => kvp.Value.ExpiresAt < DateTime.UtcNow).ToList();
            foreach (var kvp in expired)
            {
                _workflowLocks.TryRemove(kvp.Key, out _);
                resolution.Actions.Add($"Cleared expired lock: {kvp.Value.ResourceType}");
                _logger.LogInformation($"[WorkflowOrchestration] Cleared expired lock: {kvp.Value.ResourceType}");
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
                _logger.LogInformation($"[WorkflowOrchestration] {conflict.Resolution}");
            }
            
            return resolution;
        }
        
        private List<List<string>> DetectDeadlocks()
        {
            var deadlocks = new List<List<string>>();
            var waitGraph = BuildWaitGraph();
            
            // Use cycle detection algorithm
            var visited = new HashSet<string>();
            var recursionStack = new HashSet<string>();
            
            foreach (var node in waitGraph.Keys)
            {
                if (DetectCycle(node, waitGraph, visited, recursionStack, new List<string>(), deadlocks))
                {
                    // Cycle detected
                }
            }
            
            return deadlocks;
        }

        private Dictionary<string, List<string>> BuildWaitGraph()
        {
            // Build wait-for graph for deadlock detection
            var graph = new Dictionary<string, List<string>>();
            
            foreach (var lockPair in _workflowLocks)
            {
                var lock1 = lockPair.Value;
                var dependencies = new List<string>();
                
                foreach (var otherLockPair in _workflowLocks)
                {
                    var lock2 = otherLockPair.Value;
                    if (lock1.WorkflowId != lock2.WorkflowId && 
                        lock1.ResourceType == lock2.ResourceType && 
                        lock1.AcquiredAt > lock2.AcquiredAt)
                    {
                        dependencies.Add(lock2.WorkflowId);
                    }
                }
                
                graph[lock1.WorkflowId] = dependencies;
            }
            
            return graph;
        }

        private bool DetectCycle(string node, Dictionary<string, List<string>> graph, HashSet<string> visited, 
            HashSet<string> recursionStack, List<string> path, List<List<string>> deadlocks)
        {
            visited.Add(node);
            recursionStack.Add(node);
            path.Add(node);
            
            if (graph.ContainsKey(node))
            {
                foreach (var neighbor in graph[node])
                {
                    if (!visited.Contains(neighbor))
                    {
                        if (DetectCycle(neighbor, graph, visited, recursionStack, path, deadlocks))
                        {
                            return true;
                        }
                    }
                    else if (recursionStack.Contains(neighbor))
                    {
                        // Cycle found
                        var cycleStart = path.IndexOf(neighbor);
                        deadlocks.Add(path.Skip(cycleStart).ToList());
                        return true;
                    }
                }
            }
            
            recursionStack.Remove(node);
            path.Remove(node);
            return false;
        }

        private async Task KillWorkflow(string workflowId)
        {
            // Implementation to terminate workflow
            _logger.LogWarning($"[WorkflowOrchestration] Terminating workflow: {workflowId}");
            
            // Remove all locks for this workflow
            var locksToRemove = _workflowLocks.Where(kvp => kvp.Value.WorkflowId == workflowId).ToList();
            foreach (var kvp in locksToRemove)
            {
                _workflowLocks.TryRemove(kvp.Key, out _);
            }
        }

        public WorkflowOrchestrationStatus GetStatus()
        {
            return new WorkflowOrchestrationStatus
            {
                ActiveLocks = _workflowLocks.Count,
                QueuedTasks = _taskQueue.Count,
                ActiveConflicts = _conflicts.Count,
                TradingMutexAvailable = _tradingMutex.CurrentCount > 0,
                Timestamp = DateTime.UtcNow
            };
        }
    }

    public class WorkflowOrchestrationStatus
    {
        public int ActiveLocks { get; set; }
        public int QueuedTasks { get; set; }
        public int ActiveConflicts { get; set; }
        public bool TradingMutexAvailable { get; set; }
        public DateTime Timestamp { get; set; }
    }
}
