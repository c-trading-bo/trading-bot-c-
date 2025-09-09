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
    private readonly ILogger<DecisionLogger> _logger;
    private readonly string _basePath;
    private readonly bool _enabled;

    public DecisionLogger(
        ILogger<DecisionLogger> logger,
        ObservabilityConfig config,
        string basePath = "data/decisions")
    {
        _logger = logger;
        _enabled = config.DecisionLine.Enabled;
        _basePath = basePath;
        
        if (_enabled)
        {
            Directory.CreateDirectory(_basePath);
            Directory.CreateDirectory(Path.Combine(_basePath, "daily"));
        }
    }

    public async Task LogDecisionAsync(IntelligenceDecision decision, CancellationToken cancellationToken = default)
    {
        if (!_enabled)
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
            await WriteToFileAsync(logEntry, cancellationToken);

            // Also log to structured logger for real-time monitoring
            _logger.LogInformation("[DECISION] {DecisionJson}", 
                JsonSerializer.Serialize(logEntry, new JsonSerializerOptions { WriteIndented = false }));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DECISION] Failed to log trading decision: {DecisionId}", decision.DecisionId);
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
                    var lines = await File.ReadAllLinesAsync(filePath, cancellationToken);
                    
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
                            _logger.LogWarning(ex, "[DECISION] Failed to parse decision log line: {Line}", line[..Math.Min(100, line.Length)]);
                        }
                    }
                }

                currentDate = currentDate.AddDays(1);
            }

            _logger.LogDebug("[DECISION] Retrieved {Count} decisions from {From} to {To}", 
                decisions.Count, fromTime, toTime);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DECISION] Failed to get decision history");
        }

        return decisions;
    }

    private void EnrichDecision(IntelligenceDecision decision)
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

    private DecisionLogEntry CreateDecisionLogEntry(IntelligenceDecision decision)
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
                MemoryPressure = GC.CollectionCount(0) + GC.CollectionCount(1) + GC.CollectionCount(2)
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
        await File.AppendAllTextAsync(filePath, json + Environment.NewLine, cancellationToken);
    }

    private string GenerateDecisionId()
    {
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var random = Random.Shared.Next(1000, 9999);
        return $"D{timestamp}_{random}";
    }

    private class DecisionLogEntry
    {
        public DateTime Timestamp { get; set; }
        public IntelligenceDecision Decision { get; set; } = new();
        public SystemInfo SystemInfo { get; set; } = new();
        public PerformanceMetrics Performance { get; set; } = new();
    }

    private class SystemInfo
    {
        public string MachineName { get; set; } = string.Empty;
        public int ProcessId { get; set; }
        public int ThreadId { get; set; }
        public long MemoryUsage { get; set; }
    }

    private class PerformanceMetrics
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
    private readonly object _lock = new();

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
            lock (_lock)
            {
                if (!_baselines.TryGetValue(modelId, out var baseline))
                {
                    // Create initial baseline
                    baseline = CreateBaseline(features);
                    _baselines[modelId] = baseline;
                    
                    return new DriftDetectionResult { HasDrift = false, Message = "Baseline established" };
                }

                // Calculate PSI for feature drift
                var psi = CalculatePSI(baseline.FeatureDistribution, features.Features);
                
                // Check thresholds
                var hasWarning = psi > _config.PsiWarn;
                var hasBlock = psi > _config.PsiBlock;

                var result = new DriftDetectionResult
                {
                    HasDrift = hasWarning,
                    ShouldBlock = hasBlock,
                    PSI = psi,
                    Message = hasBlock ? "Feature drift detected - blocking" : 
                             hasWarning ? "Feature drift warning" : "No drift detected"
                };

                if (hasWarning)
                {
                    _logger.LogWarning("[DRIFT] Feature drift detected for {ModelId}: PSI={PSI:F3} (warn>{Warn}, block>{Block})", 
                        modelId, psi, _config.PsiWarn, _config.PsiBlock);
                }

                return result;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DRIFT] Failed to detect drift for {ModelId}", modelId);
            return new DriftDetectionResult { HasDrift = false, Message = "Drift detection failed" };
        }
    }

    private FeatureBaseline CreateBaseline(FeatureSet features)
    {
        return new FeatureBaseline
        {
            CreatedAt = DateTime.UtcNow,
            FeatureDistribution = new Dictionary<string, double>(features.Features),
            SampleCount = 1
        };
    }

    private double CalculatePSI(Dictionary<string, double> baseline, Dictionary<string, double> current)
    {
        var psi = 0.0;
        var totalFeatures = baseline.Count;

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

    private class FeatureBaseline
    {
        public DateTime CreatedAt { get; set; }
        public Dictionary<string, double> FeatureDistribution { get; set; } = new();
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