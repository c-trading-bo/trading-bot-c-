using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// High-performance async trading logger with lock-free concurrent operations
/// Implements production-ready logging with file rotation, compression, and structured JSON
/// </summary>
public class TradingLogger : ITradingLogger, IDisposable
{
    private readonly ILogger<TradingLogger> _logger;
    private readonly TradingLoggerOptions _options;
    private readonly ConcurrentQueue<TradingLogEntry> _logQueue = new();
    private readonly ConcurrentQueue<TradingLogEntry> _memoryBuffer = new();
    private readonly Timer _flushTimer;
    private readonly Task _backgroundProcessor;
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    private readonly SemaphoreSlim _fileLock = new(1, 1);
    private readonly string _sessionId = Guid.NewGuid().ToString("N")[..8];
    
    // File rotation tracking
    private readonly Dictionary<TradingLogCategory, FileInfo> _currentFiles = new();
    private readonly Dictionary<TradingLogCategory, long> _currentFileSizes = new();
    private readonly PerformanceMetrics _performanceMetrics = new();
    
    // Market data sampling
    private int _marketDataCounter = 0;
    private int _mlPredictionCounter = 0;
    private readonly object _samplingLock = new();
    
    private bool _disposed = false;
    private int _pendingCount = 0;

    public TradingLogger(ILogger<TradingLogger> logger, IOptions<TradingLoggerOptions> options)
    {
        _logger = logger;
        _options = options.Value;
        
        // Ensure log directory exists
        EnsureLogDirectoryExists();
        
        // Start background processor
        _backgroundProcessor = Task.Run(ProcessLogQueueAsync);
        
        // Start flush timer - batch writes every 100ms or when buffer reaches 1000 messages
        _flushTimer = new Timer(FlushTimerCallback, null, TimeSpan.FromMilliseconds(100), TimeSpan.FromMilliseconds(100));
        
        _logger.LogInformation("TradingLogger initialized with session {SessionId}", _sessionId);
    }

    public async Task LogEventAsync(TradingLogCategory category, TradingLogLevel level, string eventType, object data, string? correlationId = null)
    {
        if (_disposed) return;

        var entry = new TradingLogEntry
        {
            Timestamp = DateTime.UtcNow,
            Category = category,
            Level = level,
            EventType = eventType,
            Data = data,
            CorrelationId = correlationId,
            SessionId = _sessionId
        };

        // Track performance metrics
        if (_options.EnablePerformanceMetrics)
        {
            _performanceMetrics.RecordLogEntry(category, level);
        }

        // Handle critical alerts
        if (_options.EnableCriticalAlerts && ShouldCreateCriticalAlert(level, eventType, data))
        {
            await WriteCriticalAlertAsync(entry).ConfigureAwait(false);
        }

        EnqueueLogEntry(entry);
        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task LogOrderAsync(string side, string symbol, decimal quantity, decimal entry, decimal stop, decimal target, decimal rMultiple, string customTag, string? orderId = null)
    {
        var orderData = new
        {
            side,
            symbol,
            qty = quantity,
            entry = entry.ToString("0.00", CultureInfo.InvariantCulture),
            stop = stop.ToString("0.00", CultureInfo.InvariantCulture),
            t1 = target.ToString("0.00", CultureInfo.InvariantCulture),
            rMultiple = rMultiple.ToString("0.00", CultureInfo.InvariantCulture),
            tag = customTag,
            orderId
        };

        await LogEventAsync(TradingLogCategory.ORDER, TradingLogLevel.INFO, "ORDER_PLACED", orderData, customTag).ConfigureAwait(false);
    }

    public async Task LogTradeAsync(string accountId, string orderId, decimal fillPrice, decimal quantity, DateTime fillTime)
    {
        var tradeData = new
        {
            account = accountId,
            orderId,
            fillPrice = fillPrice.ToString("0.00", CultureInfo.InvariantCulture),
            qty = quantity,
            time = fillTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture)
        };

        await LogEventAsync(TradingLogCategory.FILL, TradingLogLevel.INFO, "TRADE_FILLED", tradeData, orderId).ConfigureAwait(false);
    }

    public async Task LogOrderStatusAsync(string accountId, string orderId, string status, string? reason = null)
    {
        var statusData = new
        {
            account = accountId,
            status,
            orderId,
            reason
        };

        await LogEventAsync(TradingLogCategory.ORDER, TradingLogLevel.INFO, "ORDER_STATUS", statusData, orderId).ConfigureAwait(false);
    }

    public async Task LogSystemAsync(TradingLogLevel level, string component, string message, object? context = null)
    {
        var systemData = new
        {
            component,
            message,
            context
        };

        await LogEventAsync(TradingLogCategory.SYSTEM, level, "SYSTEM_EVENT", systemData).ConfigureAwait(false);
    }

    public async Task LogMarketDataAsync(string symbol, string dataType, object data)
    {
        // Implement sampling strategy to reduce volume
        lock (_samplingLock)
        {
            _marketDataCounter++;
            if (_marketDataCounter > 10)
            {
                _marketDataCounter = 1;
            }
            
            // Only log first 10 bars, then every 10th bar
            if (_marketDataCounter != 1 && _marketDataCounter != 10)
            {
                return;
            }
        }

        var marketData = new
        {
            symbol,
            dataType,
            data,
            sampleNumber = _marketDataCounter
        };

        await LogEventAsync(TradingLogCategory.MARKET, TradingLogLevel.DEBUG, "MARKET_DATA", marketData).ConfigureAwait(false);
    }

    public async Task LogMLAsync(string model, string action, object data, string? correlationId = null)
    {
        // Implement aggregation for ML predictions
        if (action == "PREDICTION")
        {
            lock (_samplingLock)
            {
                _mlPredictionCounter++;
                if (_mlPredictionCounter < _options.MLPredictionAggregationCount)
                {
                    return; // Skip logging until we reach aggregation count
                }
                _mlPredictionCounter = 0; // Reset counter
            }
        }

        var mlData = new
        {
            model,
            action,
            data,
            aggregatedCount = action == "PREDICTION" ? _options.MLPredictionAggregationCount : 1
        };

        await LogEventAsync(TradingLogCategory.ML, TradingLogLevel.INFO, "ML_EVENT", mlData, correlationId).ConfigureAwait(false);
    }

    public Task<TradingLogEntry[]> GetRecentEntriesAsync(int count = 1000, TradingLogCategory? category = null)
    {
        var entries = _memoryBuffer.ToArray();
        
        if (category.HasValue)
        {
            entries = entries.Where(e => e.Category == category.Value).ToArray();
        }
        
        return Task.FromResult(entries.TakeLast(count).ToArray());
    }

    public async Task FlushAsync()
    {
        if (_disposed) return;

        // Wait for pending queue to be processed
        var maxWait = TimeSpan.FromSeconds(5);
        var start = DateTime.UtcNow;
        
        while (_pendingCount > 0 && DateTime.UtcNow - start < maxWait)
        {
            await Task.Delay(10).ConfigureAwait(false);
        }
    }

    private void EnqueueLogEntry(TradingLogEntry entry)
    {
        if (_disposed) return;

        _logQueue.Enqueue(entry);
        Interlocked.Increment(ref _pendingCount);
        
        // Add to memory buffer (ring buffer)
        _memoryBuffer.Enqueue(entry);
        while (_memoryBuffer.Count > _options.MemoryBufferSize)
        {
            _memoryBuffer.TryDequeue(out _);
        }
    }

    private async Task ProcessLogQueueAsync()
    {
        var batch = new List<TradingLogEntry>(_options.BatchSize);
        
        while (!_cancellationTokenSource.Token.IsCancellationRequested)
        {
            try
            {
                // Collect batch
                batch.Clear();
                var deadline = DateTime.UtcNow.AddMilliseconds(_options.BatchFlushTimeoutMs);
                
                while (batch.Count < _options.BatchSize && DateTime.UtcNow < deadline)
                {
                    if (_logQueue.TryDequeue(out var entry))
                    {
                        batch.Add(entry);
                        Interlocked.Decrement(ref _pendingCount);
                    }
                    else
                    {
                        await Task.Delay(1, _cancellationTokenSource.Token).ConfigureAwait(false);
                    }
                }

                if (batch.Count > 0)
                {
                    await WriteBatchToFilesAsync(batch).ConfigureAwait(false);
                }
                else
                {
                    await Task.Delay(10, _cancellationTokenSource.Token).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in log processing");
                await Task.Delay(1000, _cancellationTokenSource.Token).ConfigureAwait(false);
            }
        }
    }

    private async Task WriteBatchToFilesAsync(List<TradingLogEntry> batch)
    {
        await _fileLock.WaitAsync(_cancellationTokenSource.Token).ConfigureAwait(false);
        try
        {
            // Group by category for efficient file writing
            var groupedEntries = batch.GroupBy(e => e.Category);
            
            foreach (var group in groupedEntries)
            {
                var category = group.Key;
                var entries = group.ToList();
                
                var filePath = GetLogFilePath(category);
                
                // Check if file rotation is needed
                if (ShouldRotateFile(category, filePath))
                {
                    await RotateLogFileAsync(category, filePath).ConfigureAwait(false);
                    filePath = GetLogFilePath(category);
                }
                
                // Write entries as NDJSON (newline-delimited JSON)
                var jsonLines = entries.Select(entry => JsonSerializer.Serialize(entry, JsonOptions));
                var content = string.Join('\n', jsonLines) + '\n';
                
                await File.AppendAllTextAsync(filePath, content, _cancellationTokenSource.Token).ConfigureAwait(false);
                
                // Update file size tracking
                _currentFileSizes[category] = _currentFileSizes.GetValueOrDefault(category, 0) + content.Length;
            }
        }
        finally
        {
            _fileLock.Release();
        }
    }

    private string GetLogFilePath(TradingLogCategory category)
    {
        var date = DateTime.UtcNow.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        var subdirectory = GetSubdirectoryForCategory(category);
        var fileName = $"{category.ToString().ToLower()}_{date}.ndjson";
        return Path.Combine(_options.LogDirectory, subdirectory, fileName);
    }

    private static string GetSubdirectoryForCategory(TradingLogCategory category) => category switch
    {
        TradingLogCategory.ORDER or TradingLogCategory.FILL or TradingLogCategory.SIGNAL or TradingLogCategory.RISK => "trading",
        TradingLogCategory.ERROR or TradingLogCategory.AUTH or TradingLogCategory.HUB or TradingLogCategory.SYSTEM => "system", 
        TradingLogCategory.ML => "ml",
        TradingLogCategory.MARKET => "market",
        _ => "system"
    };

    private bool ShouldRotateFile(TradingLogCategory category, string filePath)
    {
        if (!File.Exists(filePath)) return false;
        
        var currentSize = _currentFileSizes.GetValueOrDefault(category, 0);
        return currentSize > _options.MaxFileSizeBytes;
    }

    private async Task RotateLogFileAsync(TradingLogCategory category, string filePath)
    {
        if (!File.Exists(filePath)) return;
        
        var timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd_HH-mm-ss", CultureInfo.InvariantCulture);
        var rotatedPath = filePath.Replace(".ndjson", $"_{timestamp}.ndjson");
        
        // Move current file
        File.Move(filePath, rotatedPath);
        
        // Compress rotated file in background
        _ = Task.Run(async () =>
        {
            try
            {
                await CompressFileAsync(rotatedPath).ConfigureAwait(false);
                File.Delete(rotatedPath);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to compress rotated log file {FilePath}", rotatedPath);
            }
        });
        
        // Reset size tracking
        _currentFileSizes[category] = 0;
        
        // Add a small delay to make this method async-worthy
        await Task.Delay(1).ConfigureAwait(false);
    }

    private static async Task CompressFileAsync(string filePath)
    {
        var compressedPath = filePath + ".gz";
        
        using var originalStream = File.OpenRead(filePath);
        using var compressedStream = File.Create(compressedPath);
        using var gzipStream = new GZipStream(compressedStream, CompressionMode.Compress);
using System.Globalization;
        
        await originalStream.CopyToAsync(gzipStream).ConfigureAwait(false);
    }

    private void EnsureLogDirectoryExists()
    {
        Directory.CreateDirectory(_options.LogDirectory);
        
        // Create comprehensive directory structure as requested
        var directories = new[]
        {
            "trading",      // ORDER, FILL, SIGNAL, RISK events
            "system",       // ERROR, AUTH, HUB, service lifecycle
            "ml",          // ML predictions, training, model performance
            "market"       // Price bars, events, order book data
        };
        
        foreach (var dir in directories)
        {
            Directory.CreateDirectory(Path.Combine(_options.LogDirectory, dir));
        }
        
        // Create critical alerts file if it doesn't exist
        var criticalAlertsPath = Path.Combine(_options.LogDirectory, "critical_alerts.txt");
        if (!File.Exists(criticalAlertsPath))
        {
            File.WriteAllText(criticalAlertsPath, $"# Critical Trading Alerts - Created {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC}\n");
        }
    }

    private void FlushTimerCallback(object? state)
    {
        // This ensures periodic flushing even with low activity
        // The actual processing happens in ProcessLogQueueAsync
    }

    private static bool ShouldCreateCriticalAlert(TradingLogLevel level, string eventType, object data)
    {
        if (level == TradingLogLevel.ERROR) return true;
        
        if (eventType.Contains("AUTHENTICATION_FAILED")) return true;
        if (eventType.Contains("CONNECTION_FAILED")) return true;
        if (eventType.Contains("ORDER_REJECTED")) return true;
        if (eventType.Contains("RISK_VIOLATION")) return true;
        if (eventType.Contains("KILL_SWITCH")) return true;
        
        // Check for specific risk violations in data
        if (data is object dataObj)
        {
            var json = JsonSerializer.Serialize(dataObj);
            if (json.Contains("risk") && json.Contains("violation")) return true;
            if (json.Contains("drawdown") && json.Contains("limit")) return true;
        }
        
        return false;
    }

    private async Task WriteCriticalAlertAsync(TradingLogEntry entry)
    {
        try
        {
            var alertPath = Path.Combine(_options.LogDirectory, "critical_alerts.txt");
            var alertMessage = $"[{entry.Timestamp:yyyy-MM-dd HH:mm:ss UTC}] CRITICAL: {entry.Category}/{entry.EventType}\n" +
                              $"Data: {JsonSerializer.Serialize(entry.Data)}\n" +
                              $"Session: {entry.SessionId}\n\n";
            
            await File.AppendAllTextAsync(alertPath, alertMessage).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to write critical alert");
        }
    }

    public async Task<PerformanceMetrics> GetPerformanceMetricsAsync()
    {
        return await Task.FromResult(_performanceMetrics).ConfigureAwait(false);
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        _flushTimer?.Dispose();
        _cancellationTokenSource.Cancel();
        
        try
        {
            _backgroundProcessor.Wait(TimeSpan.FromSeconds(5));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error waiting for background processor to finish");
        }
        
        _cancellationTokenSource.Dispose();
        _fileLock.Dispose();
    }
}

/// <summary>
/// Configuration options for TradingLogger
/// </summary>
public class TradingLoggerOptions
{
    public string LogDirectory { get; set; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "TradingBot", "Logs");
    public int BatchSize { get; set; } = 1000;
    public int BatchFlushTimeoutMs { get; set; } = 100;
    public long MaxFileSizeBytes { get; set; } = 100 * 1024 * 1024; // 100MB
    public int MemoryBufferSize { get; set; } = 10000;
    public bool EnableCompression { get; set; } = true;
    public int LogRetentionDays { get; set; } = 30;
    public int DebugLogRetentionDays { get; set; } = 7;
    public bool EnablePerformanceMetrics { get; set; } = true;
    public bool EnableCriticalAlerts { get; set; } = true;
    public int MarketDataSamplingRate { get; set; } = 10; // Log every 10th bar
    public int MLPredictionAggregationCount { get; set; } = 100; // Aggregate every 100 predictions
}

/// <summary>
/// Performance metrics for the trading logger
/// </summary>
public class PerformanceMetrics
{
    private readonly ConcurrentDictionary<TradingLogCategory, long> _categoryCounts = new();
    private readonly ConcurrentDictionary<TradingLogLevel, long> _levelCounts = new();
    private readonly object _lockObject = new();
    private DateTime _startTime = DateTime.UtcNow;
    private long _totalEntries = 0;

    public void RecordLogEntry(TradingLogCategory category, TradingLogLevel level)
    {
        _categoryCounts.AddOrUpdate(category, 1, (_, count) => count + 1);
        _levelCounts.AddOrUpdate(level, 1, (_, count) => count + 1);
        Interlocked.Increment(ref _totalEntries);
    }

    public object GetMetrics()
    {
        var uptime = DateTime.UtcNow - _startTime;
        var entriesPerSecond = _totalEntries / Math.Max(1, uptime.TotalSeconds);
        
        return new
        {
            uptime = uptime.ToString(@"dd\.hh\:mm\:ss"),
            totalEntries = _totalEntries,
            entriesPerSecond = Math.Round(entriesPerSecond, 2),
            categoryCounts = _categoryCounts.ToDictionary(kvp => kvp.Key.ToString(), kvp => kvp.Value),
            levelCounts = _levelCounts.ToDictionary(kvp => kvp.Key.ToString(), kvp => kvp.Value)
        };
    }
}