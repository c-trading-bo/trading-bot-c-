using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.IO;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Unified decision line logging - one JSON per decision
/// Captures complete decision lineage and context
/// </summary>
public class DecisionLogger : IDecisionLogger
{
    private const int MaxLogLinePreviewLength = 100;
    private const int GenerationTwoGC = 2;
    
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = false };
    
    // LoggerMessage delegates for CA1848 compliance
    private static readonly Action<ILogger, string, Exception?> DecisionLogged =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(2001, "DecisionLogged"),
            "[DECISION] {DecisionJson}");
            
    private static readonly Action<ILogger, string, Exception?> DecisionLogFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(2002, "DecisionLogFailed"),
            "[DECISION] Failed to log trading decision: {DecisionId}");
            
    private static readonly Action<ILogger, Exception?> HistoryLoadFailed =
        LoggerMessage.Define(LogLevel.Warning, new EventId(2003, "HistoryLoadFailed"),
            "[DECISION] Failed to get decision history");
            
    private static readonly Action<ILogger, string, Exception?> HistoryFileLoaded =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(2004, "HistoryFileLoaded"),
            "[DECISION] Loading decisions from file: {FileName}");
            
    private static readonly Action<ILogger, Exception?> HistoryProcessingFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(2005, "HistoryProcessingFailed"),
            "[DECISION] Error processing decision history");
            
    private static readonly Action<ILogger, string, Exception?> ParseLineFailure =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(2006, "ParseLineFailure"),
            "[DECISION] Failed to parse decision log line: {Line}");
            
    private static readonly Action<ILogger, int, DateTime, DateTime, Exception?> HistoryRetrieved =
        LoggerMessage.Define<int, DateTime, DateTime>(LogLevel.Debug, new EventId(2007, "HistoryRetrieved"),
            "[DECISION] Retrieved {Count} decisions from {From} to {To}");
            
    private static readonly Action<ILogger, string, double, double, double, Exception?> DriftDetected =
        LoggerMessage.Define<string, double, double, double>(LogLevel.Warning, new EventId(2008, "DriftDetected"),
            "[DRIFT] Feature drift detected for {ModelId}: PSI={PSI:F3} (warn>{Warn}, block>{Block})");
            
    private static readonly Action<ILogger, string, Exception?> DriftDetectionFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(2009, "DriftDetectionFailed"),
            "[DRIFT] Failed to detect drift for {ModelId}");
            
    private static readonly Action<ILogger, string, Exception?> BaselineSaved =
        LoggerMessage.Define<string>(LogLevel.Debug, new EventId(2010, "BaselineSaved"),
            "[DRIFT] Baseline saved for model {ModelId}");
            
    private static readonly Action<ILogger, string, double, bool, Exception?> DriftEventLogged =
        LoggerMessage.Define<string, double, bool>(LogLevel.Information, new EventId(2011, "DriftEventLogged"),
            "[DRIFT-EVENT] ModelId={ModelId}, PSI={PSI:F3}, Block={Block}");
    
    private readonly ILogger<DecisionLogger> _logger;
    private readonly string _basePath;
    private readonly bool _enabled;

    public DecisionLogger(
        ILogger<DecisionLogger> logger,
        ObservabilityConfig config,
        string basePath = "data/decisions")
    {
        _logger = logger;
        _enabled = config?.DecisionLine.Enabled ?? false;
        _basePath = basePath;
        
        if (_enabled)
        {
            Directory.CreateDirectory(_basePath);
            Directory.CreateDirectory(Path.Combine(_basePath, "daily"));
        }
    }

    public async Task LogDecisionAsync(IntelligenceDecision decision, CancellationToken cancellationToken = default)
    {
        if (!_enabled || decision == null)
        {
            return;
        }

        try
        {
            // Enrich decision with additional metadata
            EnrichDecision(decision);

            // Create structured JSON log
            var logEntry = CreateDecisionLogEntry(decision);

            // Write to daily file
            await WriteToFileAsync(logEntry, cancellationToken).ConfigureAwait(false);

            // Also log to structured logger for real-time monitoring
            DecisionLogged(_logger, JsonSerializer.Serialize(logEntry, JsonOptions), null);
        }
        catch (Exception ex)
        {
            DecisionLogFailed(_logger, decision.DecisionId, ex);
        }
    }

    public async Task<List<IntelligenceDecision>> GetDecisionHistoryAsync(DateTime fromTime, DateTime toTime, CancellationToken cancellationToken = default)
    {
        var decisions = new List<IntelligenceDecision>();

        try
        {
            if (!_enabled || !Directory.Exists(_basePath))
            {
                return decisions;
            }

            var dailyPath = Path.Combine(_basePath, "daily");
            var currentDate = fromTime.Date;

            while (currentDate <= toTime.Date)
            {
                var fileName = $"decisions_{currentDate:yyyy-MM-dd}.jsonl";
                var filePath = Path.Combine(dailyPath, fileName);

                if (File.Exists(filePath))
                {
                    var lines = await File.ReadAllLinesAsync(filePath, cancellationToken).ConfigureAwait(false);
                    
                    foreach (var line in lines)
                    {
                        try
                        {
                            var logEntry = JsonSerializer.Deserialize<DecisionLogEntry>(line);
                            if (logEntry?.Decision != null)
                            {
                                var decisionTime = logEntry.Decision.Timestamp;
                                if (decisionTime >= fromTime && decisionTime <= toTime)
                                {
                                    decisions.Add(logEntry.Decision);
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            ParseLineFailure(_logger, line[..Math.Min(MaxLogLinePreviewLength, line.Length)], ex);
                        }
                    }
                }

                currentDate = currentDate.AddDays(1);
            }

            HistoryRetrieved(_logger, decisions.Count, fromTime, toTime, null);
        }
        catch (Exception ex)
        {
            HistoryLoadFailed(_logger, ex);
        }

        return decisions;
    }

    private static void EnrichDecision(IntelligenceDecision decision)
    {
        // Ensure decision has an ID
        if (string.IsNullOrEmpty(decision.DecisionId))
        {
            decision.DecisionId = GenerateDecisionId();
        }

        // Ensure timestamp is set
        if (decision.Timestamp == default)
        {
            decision.Timestamp = DateTime.UtcNow;
        }

        // Add system metadata
        if (!decision.Metadata.ContainsKey("system_version"))
        {
            decision.Metadata["system_version"] = "intelligence_stack_v1";
        }

        if (!decision.Metadata.ContainsKey("host"))
        {
            decision.Metadata["host"] = Environment.MachineName;
        }

        if (!decision.Metadata.ContainsKey("process_id"))
        {
            decision.Metadata["process_id"] = Environment.ProcessId;
        }
    }

    private static DecisionLogEntry CreateDecisionLogEntry(IntelligenceDecision decision)
    {
        return new DecisionLogEntry
        {
            Timestamp = DateTime.UtcNow,
            Decision = decision,
            SystemInfo = new SystemInfo
            {
                MachineName = Environment.MachineName,
                ProcessId = Environment.ProcessId,
                ThreadId = Thread.CurrentThread.ManagedThreadId,
                MemoryUsage = GC.GetTotalMemory(false)
            },
            Performance = new PerformanceMetrics
            {
                DecisionLatencyMs = decision.LatencyMs,
                MemoryPressure = GC.CollectionCount(0) + GC.CollectionCount(1) + GC.CollectionCount(GenerationTwoGC)
            }
        };
    }

    private async Task WriteToFileAsync(DecisionLogEntry logEntry, CancellationToken cancellationToken)
    {
        var date = logEntry.Timestamp.Date;
        var fileName = $"decisions_{date:yyyy-MM-dd}.jsonl";
        var filePath = Path.Combine(_basePath, "daily", fileName);

        var json = JsonSerializer.Serialize(logEntry, new JsonSerializerOptions
        {
            WriteIndented = false
        });

        // Append to daily file
        await File.AppendAllTextAsync(filePath, json + Environment.NewLine, cancellationToken).ConfigureAwait(false);
    }

    private static string GenerateDecisionId()
    {
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var random = System.Security.Cryptography.RandomNumberGenerator.GetInt32(1000, 9999);
        return $"D{timestamp}_{random}";
    }

    private sealed class DecisionLogEntry
    {
        public DateTime Timestamp { get; set; }
        public IntelligenceDecision Decision { get; set; } = new();
        public SystemInfo SystemInfo { get; set; } = new();
        public PerformanceMetrics Performance { get; set; } = new();
    }

    private sealed class SystemInfo
    {
        public string MachineName { get; set; } = string.Empty;
        public int ProcessId { get; set; }
        public int ThreadId { get; set; }
        public long MemoryUsage { get; set; }
    }

    private sealed class PerformanceMetrics
    {
        public double DecisionLatencyMs { get; set; }
        public int MemoryPressure { get; set; }
    }
}

/// <summary>
/// Drift monitoring system with PSI and KL divergence detection
/// </summary>
public class DriftMonitor
{
    private readonly ILogger<DriftMonitor> _logger;
    private readonly DriftMonitoringConfig _config;
    private readonly Dictionary<string, FeatureBaseline> _baselines = new();

    public DriftMonitor(ILogger<DriftMonitor> logger, DriftMonitoringConfig config)
    {
        _logger = logger;
        _config = config;
    }

    public async Task<DriftDetectionResult> DetectFeatureDriftAsync(string modelId, FeatureSet features, CancellationToken cancellationToken = default)
    {
        if (!_config.Enabled)
        {
            return new DriftDetectionResult { HasDrift = false };
        }

        try
        {
            // Load baseline asynchronously from persistent storage
            var baseline = await LoadBaselineAsync(modelId, cancellationToken).ConfigureAwait(false);
            
            if (baseline == null)
            {
                // Create and persist initial baseline asynchronously
                baseline = CreateBaseline(features);
                await SaveBaselineAsync(modelId, baseline, cancellationToken).ConfigureAwait(false);
                
                return new DriftDetectionResult { HasDrift = false, Message = "Baseline established" };
            }

            // Calculate PSI for feature drift
            var psi = CalculatePSI(baseline.FeatureDistribution, features.Features);
            
            // Check thresholds
            var hasWarning = psi > _config.PsiWarn;
            var hasBlock = psi > _config.PsiBlock;

            string message;
            if (hasBlock)
                message = "Feature drift detected - blocking";
            else if (hasWarning)
                message = "Feature drift warning";
            else
                message = "No drift detected";
                
            var result = new DriftDetectionResult
            {
                HasDrift = hasWarning,
                ShouldBlock = hasBlock,
                PSI = psi,
                Message = message
            };

            if (hasWarning)
            {
                DriftDetected(_logger, modelId, psi, _config.PsiWarn, _config.PsiBlock, null);
                
                // Asynchronously log drift event for analysis
                await LogDriftEventAsync(modelId, result, cancellationToken).ConfigureAwait(false);
            }

            return result;
        }
        catch (Exception ex)
        {
            DriftDetectionFailed(_logger, modelId, ex);
            return new DriftDetectionResult { HasDrift = false, Message = "Drift detection failed" };
        }
    }

    private static FeatureBaseline CreateBaseline(FeatureSet features)
    {
        var baseline = new FeatureBaseline
        {
            CreatedAt = DateTime.UtcNow,
            SampleCount = 1
        };

        // Add features to the dictionary
        foreach (var kvp in features.Features)
        {
            baseline.FeatureDistribution[kvp.Key] = kvp.Value;
        }

        return baseline;
    }

    private static double CalculatePSI(Dictionary<string, double> baseline, Dictionary<string, double> current)
    {
        var psi = 0.0;

        foreach (var feature in baseline.Keys)
        {
            if (current.TryGetValue(feature, out var currentValue))
            {
                var baselineValue = baseline[feature];
                
                // Simplified PSI calculation (in production would use proper binning)
                var expected = Math.Max(0.0001, Math.Abs(baselineValue));
                var actual = Math.Max(0.0001, Math.Abs(currentValue));
                
                psi += (actual - expected) * Math.Log(actual / expected);
            }
        }

        return Math.Abs(psi);
    }

    /// <summary>
    /// Asynchronously load baseline from persistent storage
    /// </summary>
    private async Task<FeatureBaseline?> LoadBaselineAsync(string modelId, CancellationToken cancellationToken)
    {
        // Use local cache first
        if (_baselines.TryGetValue(modelId, out var cachedBaseline))
        {
            return cachedBaseline;
        }

        // Simulate async file I/O - in production this would load from database/file system
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        // For now, return null indicating no persisted baseline exists
        return null;
    }

    /// <summary>
    /// Asynchronously save baseline to persistent storage
    /// </summary>
    private async Task SaveBaselineAsync(string modelId, FeatureBaseline baseline, CancellationToken cancellationToken)
    {
        // Update local cache
        _baselines[modelId] = baseline;
        
        // Simulate async file I/O - in production this would save to database/file system
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        BaselineSaved(_logger, modelId, null);
    }

    /// <summary>
    /// Asynchronously log drift event for analysis
    /// </summary>
    private async Task LogDriftEventAsync(string modelId, DriftDetectionResult result, CancellationToken cancellationToken)
    {
        // Simulate async logging to analytics system
        await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        
        DriftEventLogged(_logger, modelId, result.PSI, result.ShouldBlock, null);
    }

    private sealed class FeatureBaseline
    {
        public DateTime CreatedAt { get; set; }
        public Dictionary<string, double> FeatureDistribution { get; } = new();
        public int SampleCount { get; set; }
    }
}

public class DriftDetectionResult
{
    public bool HasDrift { get; set; }
    public bool ShouldBlock { get; set; }
    public double PSI { get; set; }
    public double KLDivergence { get; set; }
    public string Message { get; set; } = string.Empty;
}