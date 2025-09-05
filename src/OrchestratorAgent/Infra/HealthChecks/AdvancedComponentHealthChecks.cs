using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Infra.HealthChecks;

/// <summary>
/// Health checks for the ML Memory Manager component
/// </summary>
[HealthCheck(Category = "Memory Management", Priority = 1)]
public class MLMemoryManagerHealthCheck : IHealthCheck
{
    private readonly ILogger<MLMemoryManagerHealthCheck> _logger;
    private readonly MLMemoryManager? _memoryManager;

    // Parameterless constructor for auto-discovery
    public MLMemoryManagerHealthCheck() : this(null, null) { }

    public MLMemoryManagerHealthCheck(ILogger<MLMemoryManagerHealthCheck>? logger, MLMemoryManager? memoryManager)
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<MLMemoryManagerHealthCheck>.Instance;
        _memoryManager = memoryManager;
    }

    public string Name => "ML Memory Manager";
    public string Description => "Monitors ML memory usage, model versioning, and leak prevention";
    public string Category => "Memory Management";
    public int IntervalSeconds => 60; // Check every minute

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Test with a local instance if none provided
            using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            var memLogger = loggerFactory.CreateLogger<MLMemoryManager>();
            var manager = _memoryManager ?? new MLMemoryManager(memLogger);

            // Initialize if not already done
            await manager.InitializeMemoryManagement();

            // Test model loading
            var testModel = await manager.LoadModel<string>("test_model.onnx", "1.0");
            if (testModel == null)
            {
                return HealthCheckResult.Failed("Failed to load test model");
            }

            // Get memory snapshot
            var snapshot = manager.GetMemorySnapshot();
            
            // Check memory usage
            var memoryUsageMB = snapshot.TotalMemory / 1024 / 1024;
            var memoryPercentage = (double)snapshot.UsedMemory / (8L * 1024 * 1024 * 1024) * 100; // 8GB limit

            if (memoryPercentage > 90)
            {
                return HealthCheckResult.Failed($"Critical memory usage: {memoryPercentage:F1}% ({memoryUsageMB}MB)");
            }
            
            if (memoryPercentage > 75)
            {
                return HealthCheckResult.Warning($"High memory usage: {memoryPercentage:F1}% ({memoryUsageMB}MB)");
            }

            // Check for memory leaks
            if (snapshot.MemoryLeaks.Count > 0)
            {
                return HealthCheckResult.Warning($"Potential memory leaks detected: {snapshot.MemoryLeaks.Count}");
            }

            var message = $"Memory healthy: {memoryPercentage:F1}% used, {snapshot.LoadedModels} models loaded";
            return HealthCheckResult.Healthy(message, new
            {
                MemoryUsageMB = memoryUsageMB,
                MemoryPercentage = memoryPercentage,
                LoadedModels = snapshot.LoadedModels,
                MLMemoryMB = snapshot.MLMemory / 1024 / 1024
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH] ML Memory Manager health check failed");
            return HealthCheckResult.Failed($"ML Memory Manager check failed: {ex.Message}", ex);
        }
    }
}

/// <summary>
/// Health checks for the Workflow Orchestration Manager component
/// </summary>
[HealthCheck(Category = "Workflow Management", Priority = 2)]
public class WorkflowOrchestrationHealthCheck : IHealthCheck
{
    private readonly ILogger<WorkflowOrchestrationHealthCheck> _logger;
    private readonly WorkflowOrchestrationManager? _orchestrationManager;

    // Parameterless constructor for auto-discovery
    public WorkflowOrchestrationHealthCheck() : this(null, null) { }

    public WorkflowOrchestrationHealthCheck(ILogger<WorkflowOrchestrationHealthCheck>? logger, 
        WorkflowOrchestrationManager? orchestrationManager)
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<WorkflowOrchestrationHealthCheck>.Instance;
        _orchestrationManager = orchestrationManager;
    }

    public string Name => "Workflow Orchestration Manager";
    public string Description => "Monitors workflow execution, collision prevention, and deadlock resolution";
    public string Category => "Workflow Management";
    public int IntervalSeconds => 30; // Check every 30 seconds

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Test with a local instance if none provided
            using var loggerFactory2 = LoggerFactory.Create(builder => builder.AddConsole());
            var workflowLogger = loggerFactory2.CreateLogger<WorkflowOrchestrationManager>();
            var manager = _orchestrationManager ?? new WorkflowOrchestrationManager(workflowLogger);

            // Test workflow execution
            var testExecuted = false;
            await manager.RequestWorkflowExecution("test-workflow", async () =>
            {
                await Task.Delay(100, cancellationToken);
                testExecuted = true;
            });

            if (!testExecuted)
            {
                return HealthCheckResult.Warning("Test workflow was queued instead of executed immediately");
            }

            // Get workflow status
            var status = manager.GetWorkflowStatus();

            // Check for excessive conflicts
            if (status.ActiveConflicts > 10)
            {
                return HealthCheckResult.Failed($"Too many active conflicts: {status.ActiveConflicts}");
            }

            // Check queue length
            if (status.QueuedTasks > 50)
            {
                return HealthCheckResult.Warning($"Large task queue: {status.QueuedTasks} tasks waiting");
            }

            // Test conflict resolution
            var resolution = await manager.ResolveConflicts();

            var message = $"Workflow orchestration healthy: {status.QueuedTasks} queued, {status.ActiveLocks} locks, {status.ActiveConflicts} conflicts";
            return HealthCheckResult.Healthy(message, new
            {
                QueuedTasks = status.QueuedTasks,
                ActiveLocks = status.ActiveLocks,
                ActiveConflicts = status.ActiveConflicts,
                LockedResources = status.LockedResources.Count,
                ConflictsResolved = resolution.ConflictsResolved.Count
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH] Workflow Orchestration health check failed");
            return HealthCheckResult.Failed($"Workflow Orchestration check failed: {ex.Message}", ex);
        }
    }
}

/// <summary>
/// Health checks for the Redundant Data Feed Manager component
/// </summary>
[HealthCheck(Category = "Data Feeds", Priority = 1)]
public class RedundantDataFeedHealthCheck : IHealthCheck
{
    private readonly ILogger<RedundantDataFeedHealthCheck> _logger;
    private readonly RedundantDataFeedManager? _dataFeedManager;

    // Parameterless constructor for auto-discovery
    public RedundantDataFeedHealthCheck() : this(null, null) { }

    public RedundantDataFeedHealthCheck(ILogger<RedundantDataFeedHealthCheck>? logger, 
        RedundantDataFeedManager? dataFeedManager)
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<RedundantDataFeedHealthCheck>.Instance;
        _dataFeedManager = dataFeedManager;
    }

    public string Name => "Redundant Data Feed Manager";
    public string Description => "Monitors data feed health, failover capability, and data consistency";
    public string Category => "Data Feeds";
    public int IntervalSeconds => 15; // Check every 15 seconds

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Test with a local instance if none provided
            using var loggerFactory3 = LoggerFactory.Create(builder => builder.AddConsole());
            var dataFeedLogger = loggerFactory3.CreateLogger<RedundantDataFeedManager>();
            var manager = _dataFeedManager ?? new RedundantDataFeedManager(dataFeedLogger);

            // Initialize if not already done
            await manager.InitializeDataFeeds();

            // Test data retrieval
            var marketData = await manager.GetMarketData("ES");
            if (marketData == null)
            {
                return HealthCheckResult.Failed("Failed to retrieve market data for ES");
            }

            // Validate data quality
            if (marketData.Price <= 0)
            {
                return HealthCheckResult.Failed("Invalid market data: price is zero or negative");
            }

            if (DateTime.UtcNow - marketData.Timestamp > TimeSpan.FromMinutes(5))
            {
                return HealthCheckResult.Warning("Market data is stale (older than 5 minutes)");
            }

            // Get feed status
            var status = manager.GetFeedStatus();

            // Check feed availability
            if (status.HealthyFeeds == 0)
            {
                return HealthCheckResult.Failed("No healthy data feeds available");
            }

            if (status.HealthyFeeds == 1 && status.TotalFeeds > 1)
            {
                return HealthCheckResult.Warning($"Only 1 of {status.TotalFeeds} feeds healthy - no redundancy");
            }

            // Check primary feed
            if (string.IsNullOrEmpty(status.PrimaryFeed) || status.PrimaryFeed == "None")
            {
                return HealthCheckResult.Failed("No primary data feed selected");
            }

            // Check feed latency
            var highLatencyFeeds = status.FeedHealthDetails.Where(f => f.IsHealthy && f.Latency > 200).ToList();
            if (highLatencyFeeds.Any())
            {
                return HealthCheckResult.Warning($"High latency detected on {highLatencyFeeds.Count} feeds");
            }

            var message = $"Data feeds healthy: {status.HealthyFeeds}/{status.TotalFeeds} feeds, primary: {status.PrimaryFeed}";
            return HealthCheckResult.Healthy(message, new
            {
                HealthyFeeds = status.HealthyFeeds,
                TotalFeeds = status.TotalFeeds,
                PrimaryFeed = status.PrimaryFeed,
                DataLatency = DateTime.UtcNow - marketData.Timestamp,
                LastPrice = marketData.Price,
                DataSource = marketData.Source
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HEALTH] Redundant Data Feed health check failed");
            return HealthCheckResult.Failed($"Data Feed Manager check failed: {ex.Message}", ex);
        }
    }
}