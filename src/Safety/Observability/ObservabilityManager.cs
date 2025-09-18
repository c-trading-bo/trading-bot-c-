using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Hosting;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json;

namespace Trading.Safety.Observability;

/// <summary>
/// Production-grade observability and operations system
/// Provides unified correlation IDs, resource monitoring, latency tracking,
/// graceful shutdown, and dead letter queues for failed operations
/// </summary>
public interface IObservabilityManager
{
    string CreateCorrelationId();
    Task<ResourceMetrics> GetResourceMetricsAsync();
    Task TrackLatencyAsync(string operationName, TimeSpan latency, string? correlationId = null);
    Task<LatencyMetrics> GetLatencyMetricsAsync(string operationName);
    Task SendToDeadLetterQueueAsync<T>(T item, string reason, string? correlationId = null);
    Task<List<DeadLetterItem>> GetDeadLetterItemsAsync(string? queueName = null);
    Task ProcessGracefulShutdownAsync(CancellationToken cancellationToken);
    Task RegisterShutdownHookAsync(Func<CancellationToken, Task> shutdownHook);
    event Action<ResourceMetrics> OnResourceMetricsUpdated;
    event Action<LatencyAlert> OnLatencyBudgetExceeded;
    event Action<DeadLetterItem> OnDeadLetterItemAdded;
}

public class ObservabilityManager : IObservabilityManager, IHostedService
{
    private readonly ILogger<ObservabilityManager> _logger;
    private readonly ObservabilityConfig _config;
    private readonly Timer _metricsTimer;
    private readonly PerformanceCounter? _cpuCounter;
    private readonly PerformanceCounter? _memoryCounter;
    
    private readonly ConcurrentDictionary<string, LatencyTracker> _latencyTrackers = new();
    private readonly ConcurrentDictionary<string, ConcurrentQueue<DeadLetterItem>> _deadLetterQueues = new();
    private readonly List<Func<CancellationToken, Task>> _shutdownHooks = new();
    private readonly object _shutdownLock = new object();
    
    private ResourceMetrics _currentMetrics = new();
    private readonly string _instanceId = Environment.MachineName + "_" + Guid.NewGuid().ToString("N")[..8];

    public event Action<ResourceMetrics> OnResourceMetricsUpdated = delegate { };
    public event Action<LatencyAlert> OnLatencyBudgetExceeded = delegate { };
    public event Action<DeadLetterItem> OnDeadLetterItemAdded = delegate { };

    public ObservabilityManager(
        ILogger<ObservabilityManager> logger,
        IOptions<ObservabilityConfig> config)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _config = config?.Value ?? throw new ArgumentNullException(nameof(config));
        
        _metricsTimer = new Timer(CollectMetricsCallback, null, Timeout.Infinite, Timeout.Infinite);
        
        // Initialize performance counters (Windows-specific, adapt for Linux/Docker)
        try
        {
            if (OperatingSystem.IsWindows())
            {
                _cpuCounter = new PerformanceCounter("Processor", "% Processor Time", "_Total");
                _memoryCounter = new PerformanceCounter("Memory", "Available MBytes");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[OBSERVABILITY] Failed to initialize performance counters - metrics will be limited");
        }
    }

    public string CreateCorrelationId()
    {
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var random = Guid.NewGuid().ToString("N")[..6];
        var correlationId = $"{_instanceId}_{timestamp}_{random}";
        
        _logger.LogDebug("[OBSERVABILITY] Created correlation ID: {CorrelationId}", correlationId);
        return correlationId;
    }

    public async Task<ResourceMetrics> GetResourceMetricsAsync()
    {
        try
        {
            var process = Process.GetCurrentProcess();
            var metrics = new ResourceMetrics
            {
                Timestamp = DateTime.UtcNow,
                InstanceId = _instanceId,
                ProcessId = process.Id,
                ThreadCount = process.Threads.Count,
                WorkingSetMemoryMB = process.WorkingSet64 / (1024 * 1024),
                PrivateMemoryMB = process.PrivateMemorySize64 / (1024 * 1024),
                VirtualMemoryMB = process.VirtualMemorySize64 / (1024 * 1024),
                TotalProcessorTime = process.TotalProcessorTime,
                UserProcessorTime = process.UserProcessorTime,
                HandleCount = process.HandleCount
            };

            // Get CPU and memory usage if counters are available
            try
            {
                if (OperatingSystem.IsWindows() && _cpuCounter != null)
                {
                    metrics.CpuUsagePercent = _cpuCounter.NextValue();
                }
                
                if (OperatingSystem.IsWindows() && _memoryCounter != null)
                {
                    metrics.AvailableMemoryMB = _memoryCounter.NextValue();
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "[OBSERVABILITY] Error reading performance counters");
            }

            // Get GC statistics
            metrics.Gen0Collections = GC.CollectionCount(0);
            metrics.Gen1Collections = GC.CollectionCount(1);
            metrics.Gen2Collections = GC.CollectionCount(2);
            metrics.TotalMemoryMB = GC.GetTotalMemory(false) / (1024 * 1024);

            _currentMetrics = metrics;
            
            _logger.LogDebug("[OBSERVABILITY] Resource metrics collected: CPU={CpuUsage:F1}%, Memory={WorkingSet}MB, Threads={ThreadCount}",
                metrics.CpuUsagePercent, metrics.WorkingSetMemoryMB, metrics.ThreadCount);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[OBSERVABILITY] Error collecting resource metrics");
            return _currentMetrics;
        }
    }

    public async Task TrackLatencyAsync(string operationName, TimeSpan latency, string? correlationId = null)
    {
        try
        {
            var tracker = _latencyTrackers.GetOrAdd(operationName, _ => new LatencyTracker(operationName));
            tracker.RecordLatency(latency);

            var latencyMs = latency.TotalMilliseconds;
            _logger.LogDebug("[OBSERVABILITY] Latency tracked: {Operation} {Latency}ms [CorrelationId: {CorrelationId}]",
                operationName, latencyMs, correlationId ?? "none");

            // Check latency budget
            var budgetThreshold = _config.LatencyBudgets.GetValueOrDefault(operationName, _config.DefaultLatencyBudgetMs);
            if (latencyMs > budgetThreshold)
            {
                var alert = new LatencyAlert
                {
                    OperationName = operationName,
                    ActualLatencyMs = latencyMs,
                    BudgetThresholdMs = budgetThreshold,
                    Timestamp = DateTime.UtcNow,
                    CorrelationId = correlationId ?? "unknown",
                    Severity = latencyMs > budgetThreshold * 2 ? AlertSeverity.Critical : AlertSeverity.Warning
                };

                OnLatencyBudgetExceeded.Invoke(alert);
                
                _logger.LogWarning("[OBSERVABILITY] Latency budget exceeded: {Operation} {Actual}ms > {Budget}ms [CorrelationId: {CorrelationId}]",
                    operationName, latencyMs, budgetThreshold, correlationId);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[OBSERVABILITY] Error tracking latency for {Operation} [CorrelationId: {CorrelationId}]", 
                operationName, correlationId);
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task<LatencyMetrics> GetLatencyMetricsAsync(string operationName)
    {
        if (_latencyTrackers.TryGetValue(operationName, out var tracker))
        {
            return await Task.FromResult(tracker.GetMetrics()).ConfigureAwait(false);
        }

        return await Task.FromResult(new LatencyMetrics { OperationName = operationName }).ConfigureAwait(false);
    }

    public async Task SendToDeadLetterQueueAsync<T>(T item, string reason, string? correlationId = null)
    {
        try
        {
            var queueName = typeof(T).Name;
            var queue = _deadLetterQueues.GetOrAdd(queueName, _ => new ConcurrentQueue<DeadLetterItem>());

            var deadLetterItem = new DeadLetterItem
            {
                Id = Guid.NewGuid().ToString("N"),
                QueueName = queueName,
                ItemType = typeof(T).FullName ?? typeof(T).Name,
                SerializedItem = JsonSerializer.Serialize(item),
                Reason = reason,
                CorrelationId = correlationId ?? "unknown",
                Timestamp = DateTime.UtcNow,
                RetryCount = 0
            };

            queue.Enqueue(deadLetterItem);
            
            // Trim queue if it gets too large
            while (queue.Count > _config.MaxDeadLetterItemsPerQueue)
            {
                queue.TryDequeue(out _);
            }

            OnDeadLetterItemAdded.Invoke(deadLetterItem);
            
            _logger.LogWarning("[OBSERVABILITY] Item sent to dead letter queue: {QueueName} {ItemType} - {Reason} [CorrelationId: {CorrelationId}]",
                queueName, typeof(T).Name, reason, correlationId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[OBSERVABILITY] Error sending item to dead letter queue [CorrelationId: {CorrelationId}]", 
                correlationId);
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task<List<DeadLetterItem>> GetDeadLetterItemsAsync(string? queueName = null)
    {
        var items = new List<DeadLetterItem>();

        try
        {
            if (queueName != null)
            {
                if (_deadLetterQueues.TryGetValue(queueName, out var queue))
                {
                    items.AddRange(queue.ToArray());
                }
            }
            else
            {
                foreach (var queue in _deadLetterQueues.Values)
                {
                    items.AddRange(queue.ToArray());
                }
            }

            items = items.OrderByDescending(i => i.Timestamp).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[OBSERVABILITY] Error retrieving dead letter items for queue: {QueueName}", queueName);
        }

        return await Task.FromResult(items).ConfigureAwait(false);
    }

    public async Task ProcessGracefulShutdownAsync(CancellationToken cancellationToken)
    {
        var correlationId = CreateCorrelationId();
        _logger.LogInformation("[OBSERVABILITY] Starting graceful shutdown [CorrelationId: {CorrelationId}]", correlationId);

        try
        {
            var shutdownTasks = new List<Task>();

            lock (_shutdownLock)
            {
                foreach (var hook in _shutdownHooks)
                {
                    shutdownTasks.Add(ExecuteShutdownHookAsync(hook, cancellationToken, correlationId));
                }
            }

            // Execute all shutdown hooks with timeout
            var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(_config.ShutdownTimeout);

            try
            {
                await Task.WhenAll(shutdownTasks).WaitAsync(timeoutCts.Token).ConfigureAwait(false);
                _logger.LogInformation("[OBSERVABILITY] All shutdown hooks completed successfully [CorrelationId: {CorrelationId}]", correlationId);
            }
            catch (OperationCanceledException) when (timeoutCts.Token.IsCancellationRequested)
            {
                _logger.LogWarning("[OBSERVABILITY] Shutdown timeout reached - forcing shutdown [CorrelationId: {CorrelationId}]", correlationId);
            }

            // Final cleanup
            await FlushPendingOperationsAsync(correlationId).ConfigureAwait(false);
            
            _logger.LogInformation("[OBSERVABILITY] Graceful shutdown completed [CorrelationId: {CorrelationId}]", correlationId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[OBSERVABILITY] Error during graceful shutdown [CorrelationId: {CorrelationId}]", correlationId);
            throw;
        }
    }

    public async Task RegisterShutdownHookAsync(Func<CancellationToken, Task> shutdownHook)
    {
        lock (_shutdownLock)
        {
            _shutdownHooks.Add(shutdownHook);
        }
        
        _logger.LogDebug("[OBSERVABILITY] Shutdown hook registered: {HookCount} total hooks", _shutdownHooks.Count);
        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _metricsTimer.Change(TimeSpan.Zero, _config.MetricsCollectionInterval);
        _logger.LogInformation("[OBSERVABILITY] Started with instance ID: {InstanceId}", _instanceId);
        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _metricsTimer.Change(Timeout.Infinite, Timeout.Infinite);
        await ProcessGracefulShutdownAsync(cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("[OBSERVABILITY] Stopped");
    }

    private async Task ExecuteShutdownHookAsync(Func<CancellationToken, Task> hook, CancellationToken cancellationToken, string correlationId)
    {
        try
        {
            var stopwatch = Stopwatch.StartNew();
            await hook(cancellationToken).ConfigureAwait(false);
            stopwatch.Stop();
            
            _logger.LogDebug("[OBSERVABILITY] Shutdown hook completed in {ElapsedMs}ms [CorrelationId: {CorrelationId}]", 
                stopwatch.ElapsedMilliseconds, correlationId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[OBSERVABILITY] Shutdown hook failed [CorrelationId: {CorrelationId}]", correlationId);
        }
    }

    private async Task FlushPendingOperationsAsync(string correlationId)
    {
        try
        {
            // Flush any pending metrics, logs, or queues
            _logger.LogDebug("[OBSERVABILITY] Flushing pending operations [CorrelationId: {CorrelationId}]", correlationId);
            
            // Collect final metrics
            await GetResourceMetricsAsync().ConfigureAwait(false);
            
            // Log final dead letter queue status
            var totalDeadLetterItems = _deadLetterQueues.Values.Sum(q => q.Count);
            if (totalDeadLetterItems > 0)
            {
                _logger.LogWarning("[OBSERVABILITY] Shutdown with {DeadLetterCount} dead letter items remaining [CorrelationId: {CorrelationId}]", 
                    totalDeadLetterItems, correlationId);
            }
            
            await Task.Delay(100, CancellationToken.None).ConfigureAwait(false); // Brief pause for log flushing
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[OBSERVABILITY] Error flushing pending operations [CorrelationId: {CorrelationId}]", correlationId);
        }
    }

    private void CollectMetricsCallback(object? state)
    {
        try
        {
            _ = Task.Run(async () =>
            {
                var metrics = await GetResourceMetricsAsync().ConfigureAwait(false);
                OnResourceMetricsUpdated.Invoke(metrics);
                
                // Check for resource alerts
                CheckResourceAlerts(metrics);
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[OBSERVABILITY] Error in metrics collection callback");
        }
    }

    private void CheckResourceAlerts(ResourceMetrics metrics)
    {
        var correlationId = CreateCorrelationId();
        
        // CPU usage alert
        if (metrics.CpuUsagePercent > _config.CpuAlertThreshold)
        {
            _logger.LogWarning("[OBSERVABILITY] High CPU usage: {CpuUsage:F1}% > {Threshold}% [CorrelationId: {CorrelationId}]",
                metrics.CpuUsagePercent, _config.CpuAlertThreshold, correlationId);
        }

        // Memory usage alert
        if (metrics.WorkingSetMemoryMB > _config.MemoryAlertThresholdMB)
        {
            _logger.LogWarning("[OBSERVABILITY] High memory usage: {MemoryUsage}MB > {Threshold}MB [CorrelationId: {CorrelationId}]",
                metrics.WorkingSetMemoryMB, _config.MemoryAlertThresholdMB, correlationId);
        }

        // Thread count alert
        if (metrics.ThreadCount > _config.ThreadCountAlertThreshold)
        {
            _logger.LogWarning("[OBSERVABILITY] High thread count: {ThreadCount} > {Threshold} [CorrelationId: {CorrelationId}]",
                metrics.ThreadCount, _config.ThreadCountAlertThreshold, correlationId);
        }
    }

    public void Dispose()
    {
        _metricsTimer?.Dispose();
        _cpuCounter?.Dispose();
        _memoryCounter?.Dispose();
    }
}

// Supporting classes and data models
public class LatencyTracker
{
    private readonly Queue<double> _recentLatencies = new();
    private readonly object _lock = new object();
    private readonly int _maxSamples = 1000;
    
    public string OperationName { get; }
    public long TotalOperations { get; private set; }
    public double TotalLatencyMs { get; private set; }

    public LatencyTracker(string operationName)
    {
        OperationName = operationName;
    }

    public void RecordLatency(TimeSpan latency)
    {
        var latencyMs = latency.TotalMilliseconds;
        
        lock (_lock)
        {
            TotalOperations++;
            TotalLatencyMs += latencyMs;
            
            _recentLatencies.Enqueue(latencyMs);
            
            // Keep only recent samples for percentile calculations
            while (_recentLatencies.Count > _maxSamples)
            {
                _recentLatencies.Dequeue();
            }
        }
    }

    public LatencyMetrics GetMetrics()
    {
        lock (_lock)
        {
            var recentArray = _recentLatencies.ToArray();
            
            if (recentArray.Length == 0)
            {
                return new LatencyMetrics { OperationName = OperationName };
            }

            Array.Sort(recentArray);
            
            return new LatencyMetrics
            {
                OperationName = OperationName,
                TotalOperations = TotalOperations,
                AverageLatencyMs = TotalLatencyMs / TotalOperations,
                MinLatencyMs = recentArray[0],
                MaxLatencyMs = recentArray[^1],
                P50LatencyMs = GetPercentile(recentArray, 0.50),
                P95LatencyMs = GetPercentile(recentArray, 0.95),
                P99LatencyMs = GetPercentile(recentArray, 0.99),
                RecentSampleCount = recentArray.Length,
                LastUpdated = DateTime.UtcNow
            };
        }
    }

    private double GetPercentile(double[] sortedArray, double percentile)
    {
        if (sortedArray.Length == 0) return 0;
        
        var index = (int)Math.Ceiling(percentile * sortedArray.Length) - 1;
        index = Math.Max(0, Math.Min(index, sortedArray.Length - 1));
        
        return sortedArray[index];
    }
}

public class ResourceMetrics
{
    public DateTime Timestamp { get; set; }
    public string InstanceId { get; set; } = string.Empty;
    public int ProcessId { get; set; }
    public int ThreadCount { get; set; }
    public long WorkingSetMemoryMB { get; set; }
    public long PrivateMemoryMB { get; set; }
    public long VirtualMemoryMB { get; set; }
    public long TotalMemoryMB { get; set; }
    public double AvailableMemoryMB { get; set; }
    public double CpuUsagePercent { get; set; }
    public TimeSpan TotalProcessorTime { get; set; }
    public TimeSpan UserProcessorTime { get; set; }
    public int HandleCount { get; set; }
    public int Gen0Collections { get; set; }
    public int Gen1Collections { get; set; }
    public int Gen2Collections { get; set; }
}

public class LatencyMetrics
{
    public string OperationName { get; set; } = string.Empty;
    public long TotalOperations { get; set; }
    public double AverageLatencyMs { get; set; }
    public double MinLatencyMs { get; set; }
    public double MaxLatencyMs { get; set; }
    public double P50LatencyMs { get; set; }
    public double P95LatencyMs { get; set; }
    public double P99LatencyMs { get; set; }
    public int RecentSampleCount { get; set; }
    public DateTime LastUpdated { get; set; }
}

public class LatencyAlert
{
    public string OperationName { get; set; } = string.Empty;
    public double ActualLatencyMs { get; set; }
    public double BudgetThresholdMs { get; set; }
    public DateTime Timestamp { get; set; }
    public string CorrelationId { get; set; } = string.Empty;
    public AlertSeverity Severity { get; set; }
}

public class DeadLetterItem
{
    public string Id { get; set; } = string.Empty;
    public string QueueName { get; set; } = string.Empty;
    public string ItemType { get; set; } = string.Empty;
    public string SerializedItem { get; set; } = string.Empty;
    public string Reason { get; set; } = string.Empty;
    public string CorrelationId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public int RetryCount { get; set; }
}

public enum AlertSeverity
{
    Info,
    Warning,
    Critical
}

public class ObservabilityConfig
{
    public TimeSpan MetricsCollectionInterval { get; set; } = TimeSpan.FromMinutes(1);
    public TimeSpan ShutdownTimeout { get; set; } = TimeSpan.FromSeconds(30);
    public double CpuAlertThreshold { get; set; } = 80.0; // 80%
    public long MemoryAlertThresholdMB { get; set; } = 1024; // 1GB
    public int ThreadCountAlertThreshold { get; set; } = 100;
    public double DefaultLatencyBudgetMs { get; set; } = 1000; // 1 second
    public int MaxDeadLetterItemsPerQueue { get; set; } = 1000;
    
    public Dictionary<string, double> LatencyBudgets { get; set; } = new()
    {
        ["OrderPlacement"] = 500,    // 500ms
        ["MarketDataProcessing"] = 100, // 100ms
        ["RiskAssessment"] = 200,    // 200ms
        ["DecisionMaking"] = 1000,   // 1 second
        ["DataRetrieval"] = 2000     // 2 seconds
    };
}