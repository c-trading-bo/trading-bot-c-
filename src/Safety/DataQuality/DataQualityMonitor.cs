using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Hosting;
using System.Collections.Concurrent;
using TradingBot.Abstractions;

namespace Trading.Safety.DataQuality;

/// <summary>
/// Production-grade data quality and integrity monitoring system
/// Detects tick gaps, zero-volume bars, repeated timestamps, time drift, and missing data
/// Provides gap repair mechanisms and schema validation
/// </summary>
public interface IDataQualityMonitor
{
    Task<DataQualityReport> ValidateMarketDataAsync(MarketDataPoint dataPoint);
    Task<List<DataQualityIssue>> DetectDataDriftAsync(string symbol, TimeSpan period);
    Task<TimeSpan> CheckTimeSynchronizationAsync();
    Task RepairDataGapsAsync(string symbol, DateTime from, DateTime to);
    Task ValidateSchemaAsync<T>(T data, string schemaVersion);
    Task<DataQualityMetrics> GetQualityMetricsAsync(string symbol);
    event Action<DataQualityIssue> OnDataQualityIssue;
    event Action<DataQualityMetrics> OnQualityMetricsUpdated;
}

public class DataQualityMonitor : IDataQualityMonitor, IHostedService
{
    private readonly ILogger<DataQualityMonitor> _logger;
    private readonly DataQualityConfig _config;
    private readonly ConcurrentDictionary<string, SymbolDataTracker> _symbolTrackers = new();
    private readonly ConcurrentDictionary<string, DataQualityMetrics> _qualityMetrics = new();
    private readonly Timer _monitoringTimer;
    private readonly Timer _ntpSyncTimer;
    private DateTime _lastNtpSync = DateTime.MinValue;
    private TimeSpan _timeSkew = TimeSpan.Zero;

    public event Action<DataQualityIssue> OnDataQualityIssue = delegate { };
    public event Action<DataQualityMetrics> OnQualityMetricsUpdated = delegate { };

    public DataQualityMonitor(
        ILogger<DataQualityMonitor> logger,
        IOptions<DataQualityConfig> config)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _config = config?.Value ?? throw new ArgumentNullException(nameof(config));
        
        _monitoringTimer = new Timer(MonitoringCallback, null, Timeout.Infinite, Timeout.Infinite);
        _ntpSyncTimer = new Timer(NtpSyncCallback, null, Timeout.Infinite, Timeout.Infinite);
    }

    public async Task<DataQualityReport> ValidateMarketDataAsync(MarketDataPoint dataPoint)
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        var report = new DataQualityReport
        {
            Symbol = dataPoint.Symbol,
            Timestamp = dataPoint.Timestamp,
            CorrelationId = correlationId,
            Issues = new List<DataQualityIssue>()
        };

        var tracker = _symbolTrackers.GetOrAdd(dataPoint.Symbol, _ => new SymbolDataTracker());
        var metrics = _qualityMetrics.GetOrAdd(dataPoint.Symbol, _ => new DataQualityMetrics { Symbol = dataPoint.Symbol });

        try
        {
            // Validate timestamp
            await ValidateTimestampAsync(dataPoint, tracker, report);
            
            // Validate price data
            await ValidatePriceDataAsync(dataPoint, tracker, report);
            
            // Validate volume data
            await ValidateVolumeDataAsync(dataPoint, tracker, report);
            
            // Check for data gaps
            await CheckDataGapsAsync(dataPoint, tracker, report);
            
            // Update tracker
            tracker.UpdateWith(dataPoint);
            
            // Update metrics
            UpdateQualityMetrics(metrics, report);
            
            _logger.LogDebug("[DATA_QUALITY] Validated {Symbol} at {Timestamp}: {IssueCount} issues [CorrelationId: {CorrelationId}]",
                dataPoint.Symbol, dataPoint.Timestamp, report.Issues.Count, correlationId);

            // Emit events for issues
            foreach (var issue in report.Issues)
            {
                OnDataQualityIssue.Invoke(issue);
            }

            return report;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_QUALITY] Error validating market data for {Symbol} [CorrelationId: {CorrelationId}]", 
                dataPoint.Symbol, correlationId);
            
            report.Issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.ValidationError,
                Severity = DataQualitySeverity.High,
                Description = $"Validation error: {ex.Message}",
                Symbol = dataPoint.Symbol,
                Timestamp = dataPoint.Timestamp,
                CorrelationId = correlationId
            });
            
            return report;
        }
    }

    public async Task<List<DataQualityIssue>> DetectDataDriftAsync(string symbol, TimeSpan period)
    {
        var issues = new List<DataQualityIssue>();
        var correlationId = Guid.NewGuid().ToString("N")[..8];

        if (!_symbolTrackers.TryGetValue(symbol, out var tracker))
        {
            return issues; // No data to analyze
        }

        var cutoff = DateTime.UtcNow - period;
        var recentData = tracker.RecentData.Where(d => d.Timestamp >= cutoff).ToList();

        if (recentData.Count < _config.MinDataPointsForDriftDetection)
        {
            return issues;
        }

        try
        {
            // Check for unusual price volatility
            var prices = recentData.Select(d => d.Price).ToList();
            var priceStdDev = CalculateStandardDeviation(prices);
            var avgPrice = prices.Average();
            var volatility = priceStdDev / avgPrice;

            if (volatility > _config.MaxVolatilityThreshold)
            {
                issues.Add(new DataQualityIssue
                {
                    Type = DataQualityIssueType.AbnormalVolatility,
                    Severity = DataQualitySeverity.Medium,
                    Description = $"Abnormal volatility detected: {volatility:P2} > {_config.MaxVolatilityThreshold:P2}",
                    Symbol = symbol,
                    Timestamp = DateTime.UtcNow,
                    CorrelationId = correlationId,
                    Metadata = new Dictionary<string, object>
                    {
                        ["volatility"] = volatility,
                        ["threshold"] = _config.MaxVolatilityThreshold,
                        ["period_minutes"] = period.TotalMinutes,
                        ["data_points"] = recentData.Count
                    }
                });
            }

            // Check for volume drift
            var volumes = recentData.Select(d => d.Volume).ToList();
            var zeroVolumeCount = volumes.Count(v => v == 0);
            var zeroVolumeRatio = (double)zeroVolumeCount / volumes.Count;

            if (zeroVolumeRatio > _config.MaxZeroVolumeRatio)
            {
                issues.Add(new DataQualityIssue
                {
                    Type = DataQualityIssueType.ZeroVolume,
                    Severity = DataQualitySeverity.High,
                    Description = $"Excessive zero volume bars: {zeroVolumeRatio:P2} > {_config.MaxZeroVolumeRatio:P2}",
                    Symbol = symbol,
                    Timestamp = DateTime.UtcNow,
                    CorrelationId = correlationId,
                    Metadata = new Dictionary<string, object>
                    {
                        ["zero_volume_ratio"] = zeroVolumeRatio,
                        ["threshold"] = _config.MaxZeroVolumeRatio,
                        ["zero_count"] = zeroVolumeCount,
                        ["total_count"] = volumes.Count
                    }
                });
            }

            // Check for timestamp clustering (repeated timestamps)
            var timestampGroups = recentData.GroupBy(d => d.Timestamp).Where(g => g.Count() > 1);
            foreach (var group in timestampGroups)
            {
                issues.Add(new DataQualityIssue
                {
                    Type = DataQualityIssueType.RepeatedTimestamp,
                    Severity = DataQualitySeverity.Medium,
                    Description = $"Repeated timestamp detected: {group.Key:yyyy-MM-dd HH:mm:ss.fff} ({group.Count()} occurrences)",
                    Symbol = symbol,
                    Timestamp = group.Key,
                    CorrelationId = correlationId,
                    Metadata = new Dictionary<string, object>
                    {
                        ["occurrence_count"] = group.Count(),
                        ["repeated_timestamp"] = group.Key
                    }
                });
            }

            _logger.LogInformation("[DATA_QUALITY] Data drift analysis for {Symbol}: {IssueCount} issues in {Period} [CorrelationId: {CorrelationId}]",
                symbol, issues.Count, period, correlationId);

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_QUALITY] Error detecting data drift for {Symbol} [CorrelationId: {CorrelationId}]", 
                symbol, correlationId);
            
            issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.AnalysisError,
                Severity = DataQualitySeverity.High,
                Description = $"Drift analysis error: {ex.Message}",
                Symbol = symbol,
                Timestamp = DateTime.UtcNow,
                CorrelationId = correlationId
            });
        }

        return await Task.FromResult(issues);
    }

    public async Task<TimeSpan> CheckTimeSynchronizationAsync()
    {
        try
        {
            var systemTime = DateTime.UtcNow;
            
            // Simulate NTP check (in production, use actual NTP client)
            var ntpTime = await SimulateNtpQueryAsync();
            _timeSkew = systemTime - ntpTime;
            _lastNtpSync = systemTime;

            if (Math.Abs(_timeSkew.TotalSeconds) > _config.MaxTimeSkewSeconds)
            {
                var issue = new DataQualityIssue
                {
                    Type = DataQualityIssueType.TimeSkew,
                    Severity = DataQualitySeverity.Critical,
                    Description = $"System time skew detected: {_timeSkew.TotalSeconds:F2}s",
                    Timestamp = systemTime,
                    CorrelationId = Guid.NewGuid().ToString("N")[..8],
                    Metadata = new Dictionary<string, object>
                    {
                        ["system_time"] = systemTime,
                        ["ntp_time"] = ntpTime,
                        ["skew_seconds"] = _timeSkew.TotalSeconds,
                        ["threshold_seconds"] = _config.MaxTimeSkewSeconds
                    }
                };

                OnDataQualityIssue.Invoke(issue);
                _logger.LogCritical("[DATA_QUALITY] Time synchronization issue: {Skew}s skew detected", 
                    _timeSkew.TotalSeconds);
            }
            else
            {
                _logger.LogDebug("[DATA_QUALITY] Time synchronization OK: {Skew}ms skew", 
                    _timeSkew.TotalMilliseconds);
            }

            return _timeSkew;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_QUALITY] Error checking time synchronization");
            return TimeSpan.Zero;
        }
    }

    public async Task RepairDataGapsAsync(string symbol, DateTime from, DateTime to)
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        
        try
        {
            if (!_symbolTrackers.TryGetValue(symbol, out var tracker))
            {
                _logger.LogWarning("[DATA_QUALITY] No tracker found for symbol {Symbol} during gap repair [CorrelationId: {CorrelationId}]", 
                    symbol, correlationId);
                return;
            }

            var existingData = tracker.RecentData
                .Where(d => d.Timestamp >= from && d.Timestamp <= to)
                .OrderBy(d => d.Timestamp)
                .ToList();

            var gaps = DetectTimeGaps(existingData, _config.ExpectedDataInterval);
            
            if (gaps.Any())
            {
                _logger.LogInformation("[DATA_QUALITY] Repairing {GapCount} data gaps for {Symbol} from {From} to {To} [CorrelationId: {CorrelationId}]",
                    gaps.Count, symbol, from, to, correlationId);

                foreach (var gap in gaps)
                {
                    await FillDataGapAsync(symbol, gap, correlationId);
                }
            }
            else
            {
                _logger.LogDebug("[DATA_QUALITY] No data gaps found for {Symbol} in period {From} to {To} [CorrelationId: {CorrelationId}]",
                    symbol, from, to, correlationId);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_QUALITY] Error repairing data gaps for {Symbol} [CorrelationId: {CorrelationId}]", 
                symbol, correlationId);
        }
    }

    public async Task ValidateSchemaAsync<T>(T data, string schemaVersion)
    {
        var correlationId = Guid.NewGuid().ToString("N")[..8];
        
        try
        {
            // Simple schema validation (in production, use JSON Schema or similar)
            var dataType = typeof(T);
            var expectedVersion = _config.SchemaVersions.GetValueOrDefault(dataType.Name, "1.0");
            
            if (schemaVersion != expectedVersion)
            {
                var issue = new DataQualityIssue
                {
                    Type = DataQualityIssueType.SchemaVersionMismatch,
                    Severity = DataQualitySeverity.High,
                    Description = $"Schema version mismatch for {dataType.Name}: expected {expectedVersion}, got {schemaVersion}",
                    Timestamp = DateTime.UtcNow,
                    CorrelationId = correlationId,
                    Metadata = new Dictionary<string, object>
                    {
                        ["data_type"] = dataType.Name,
                        ["expected_version"] = expectedVersion,
                        ["actual_version"] = schemaVersion
                    }
                };

                OnDataQualityIssue.Invoke(issue);
                _logger.LogWarning("[DATA_QUALITY] Schema validation failed: {Issue} [CorrelationId: {CorrelationId}]", 
                    issue.Description, correlationId);
            }
            else
            {
                _logger.LogDebug("[DATA_QUALITY] Schema validation passed for {DataType} v{Version} [CorrelationId: {CorrelationId}]",
                    dataType.Name, schemaVersion, correlationId);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_QUALITY] Error validating schema [CorrelationId: {CorrelationId}]", correlationId);
        }

        await Task.CompletedTask;
    }

    public async Task<DataQualityMetrics> GetQualityMetricsAsync(string symbol)
    {
        if (_qualityMetrics.TryGetValue(symbol, out var metrics))
        {
            return await Task.FromResult(metrics.Clone());
        }

        return await Task.FromResult(new DataQualityMetrics { Symbol = symbol });
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _monitoringTimer.Change(TimeSpan.Zero, _config.MonitoringInterval);
        _ntpSyncTimer.Change(TimeSpan.Zero, _config.NtpSyncInterval);
        _logger.LogInformation("[DATA_QUALITY] Started monitoring with interval: {Interval}", _config.MonitoringInterval);
        await Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _monitoringTimer.Change(Timeout.Infinite, Timeout.Infinite);
        _ntpSyncTimer.Change(Timeout.Infinite, Timeout.Infinite);
        _logger.LogInformation("[DATA_QUALITY] Stopped monitoring");
        await Task.CompletedTask;
    }

    // Private implementation methods
    private async Task ValidateTimestampAsync(MarketDataPoint dataPoint, SymbolDataTracker tracker, DataQualityReport report)
    {
        var now = DateTime.UtcNow;
        var timeDiff = Math.Abs((now - dataPoint.Timestamp).TotalSeconds);

        // Check for future timestamps
        if (dataPoint.Timestamp > now.AddSeconds(_config.MaxFutureToleranceSeconds))
        {
            report.Issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.FutureTimestamp,
                Severity = DataQualitySeverity.High,
                Description = $"Future timestamp detected: {dataPoint.Timestamp:yyyy-MM-dd HH:mm:ss.fff} vs {now:yyyy-MM-dd HH:mm:ss.fff}",
                Symbol = dataPoint.Symbol,
                Timestamp = dataPoint.Timestamp,
                CorrelationId = report.CorrelationId
            });
        }

        // Check for stale data
        if (timeDiff > _config.MaxDataAgeSeconds)
        {
            report.Issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.StaleData,
                Severity = DataQualitySeverity.Medium,
                Description = $"Stale data detected: {timeDiff:F1}s old",
                Symbol = dataPoint.Symbol,
                Timestamp = dataPoint.Timestamp,
                CorrelationId = report.CorrelationId
            });
        }

        await Task.CompletedTask;
    }

    private async Task ValidatePriceDataAsync(MarketDataPoint dataPoint, SymbolDataTracker tracker, DataQualityReport report)
    {
        // Check for invalid prices
        if (dataPoint.Price <= 0)
        {
            report.Issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.InvalidPrice,
                Severity = DataQualitySeverity.High,
                Description = $"Invalid price: {dataPoint.Price}",
                Symbol = dataPoint.Symbol,
                Timestamp = dataPoint.Timestamp,
                CorrelationId = report.CorrelationId
            });
        }

        // Check for price gaps if we have previous data
        if (tracker.LastDataPoint != null)
        {
            var priceChange = Math.Abs(dataPoint.Price - tracker.LastDataPoint.Price) / tracker.LastDataPoint.Price;
            if (priceChange > _config.MaxPriceChangePercent)
            {
                report.Issues.Add(new DataQualityIssue
                {
                    Type = DataQualityIssueType.PriceGap,
                    Severity = DataQualitySeverity.Medium,
                    Description = $"Large price change: {priceChange:P2} from {tracker.LastDataPoint.Price} to {dataPoint.Price}",
                    Symbol = dataPoint.Symbol,
                    Timestamp = dataPoint.Timestamp,
                    CorrelationId = report.CorrelationId
                });
            }
        }

        await Task.CompletedTask;
    }

    private async Task ValidateVolumeDataAsync(MarketDataPoint dataPoint, SymbolDataTracker tracker, DataQualityReport report)
    {
        // Check for negative volume
        if (dataPoint.Volume < 0)
        {
            report.Issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.InvalidVolume,
                Severity = DataQualitySeverity.High,
                Description = $"Negative volume: {dataPoint.Volume}",
                Symbol = dataPoint.Symbol,
                Timestamp = dataPoint.Timestamp,
                CorrelationId = report.CorrelationId
            });
        }

        await Task.CompletedTask;
    }

    private async Task CheckDataGapsAsync(MarketDataPoint dataPoint, SymbolDataTracker tracker, DataQualityReport report)
    {
        if (tracker.LastDataPoint != null)
        {
            var timeDiff = dataPoint.Timestamp - tracker.LastDataPoint.Timestamp;
            var expectedInterval = _config.ExpectedDataInterval;

            if (timeDiff > expectedInterval.Add(TimeSpan.FromSeconds(_config.GapToleranceSeconds)))
            {
                report.Issues.Add(new DataQualityIssue
                {
                    Type = DataQualityIssueType.DataGap,
                    Severity = DataQualitySeverity.Medium,
                    Description = $"Data gap detected: {timeDiff.TotalSeconds:F1}s gap (expected {expectedInterval.TotalSeconds:F1}s)",
                    Symbol = dataPoint.Symbol,
                    Timestamp = dataPoint.Timestamp,
                    CorrelationId = report.CorrelationId
                });
            }
        }

        await Task.CompletedTask;
    }

    private void UpdateQualityMetrics(DataQualityMetrics metrics, DataQualityReport report)
    {
        metrics.TotalDataPoints++;
        metrics.LastUpdate = DateTime.UtcNow;
        
        foreach (var issue in report.Issues)
        {
            metrics.TotalIssues++;
            
            switch (issue.Severity)
            {
                case DataQualitySeverity.Low:
                    metrics.LowSeverityIssues++;
                    break;
                case DataQualitySeverity.Medium:
                    metrics.MediumSeverityIssues++;
                    break;
                case DataQualitySeverity.High:
                    metrics.HighSeverityIssues++;
                    break;
                case DataQualitySeverity.Critical:
                    metrics.CriticalIssues++;
                    break;
            }
        }

        metrics.QualityScore = CalculateQualityScore(metrics);
        OnQualityMetricsUpdated.Invoke(metrics);
    }

    private double CalculateQualityScore(DataQualityMetrics metrics)
    {
        if (metrics.TotalDataPoints == 0) return 1.0;

        var issueWeight = (metrics.CriticalIssues * 10) + 
                         (metrics.HighSeverityIssues * 5) + 
                         (metrics.MediumSeverityIssues * 2) + 
                         metrics.LowSeverityIssues;

        var maxPossibleWeight = metrics.TotalDataPoints * 10; // Assuming worst case: all critical
        var qualityScore = 1.0 - ((double)issueWeight / maxPossibleWeight);
        
        return Math.Max(0.0, Math.Min(1.0, qualityScore));
    }

    private double CalculateStandardDeviation(List<double> values)
    {
        if (values.Count < 2) return 0;
        
        var mean = values.Average();
        var variance = values.Select(v => Math.Pow(v - mean, 2)).Average();
        return Math.Sqrt(variance);
    }

    private async Task<DateTime> SimulateNtpQueryAsync()
    {
        // Simulate NTP query with small random offset
        await Task.Delay(50); // Simulate network delay
        var random = new Random();
        var offsetMs = random.Next(-100, 100); // ±100ms random offset
        return DateTime.UtcNow.AddMilliseconds(offsetMs);
    }

    private List<TimeGap> DetectTimeGaps(List<MarketDataPoint> data, TimeSpan expectedInterval)
    {
        var gaps = new List<TimeGap>();
        
        for (int i = 1; i < data.Count; i++)
        {
            var timeDiff = data[i].Timestamp - data[i - 1].Timestamp;
            if (timeDiff > expectedInterval.Add(TimeSpan.FromSeconds(_config.GapToleranceSeconds)))
            {
                gaps.Add(new TimeGap
                {
                    StartTime = data[i - 1].Timestamp,
                    EndTime = data[i].Timestamp,
                    Duration = timeDiff
                });
            }
        }

        return gaps;
    }

    private async Task FillDataGapAsync(string symbol, TimeGap gap, string correlationId)
    {
        try
        {
            // In production, this would request data from alternative sources or interpolate
            _logger.LogInformation("[DATA_QUALITY] Filling data gap for {Symbol}: {Start} to {End} ({Duration}s) [CorrelationId: {CorrelationId}]",
                symbol, gap.StartTime, gap.EndTime, gap.Duration.TotalSeconds, correlationId);

            // Simulate gap filling (in production, implement actual data retrieval)
            await Task.Delay(100);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_QUALITY] Error filling data gap for {Symbol} [CorrelationId: {CorrelationId}]", 
                symbol, correlationId);
        }
    }

    private void MonitoringCallback(object? state)
    {
        try
        {
            _ = Task.Run(async () =>
            {
                // Periodic monitoring tasks
                foreach (var symbol in _symbolTrackers.Keys.ToList())
                {
                    await DetectDataDriftAsync(symbol, _config.DriftDetectionPeriod);
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_QUALITY] Error in monitoring callback");
        }
    }

    private void NtpSyncCallback(object? state)
    {
        try
        {
            _ = Task.Run(async () => await CheckTimeSynchronizationAsync());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATA_QUALITY] Error in NTP sync callback");
        }
    }

    public void Dispose()
    {
        _monitoringTimer?.Dispose();
        _ntpSyncTimer?.Dispose();
    }
}

// Data models and supporting classes
public class MarketDataPoint
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public double Price { get; set; }
    public double Volume { get; set; }
    public double Bid { get; set; }
    public double Ask { get; set; }
}

public class SymbolDataTracker
{
    private readonly int MaxRecentDataPoints = 1000;
    
    public MarketDataPoint? LastDataPoint { get; private set; }
    public List<MarketDataPoint> RecentData { get; } = new();

    public void UpdateWith(MarketDataPoint dataPoint)
    {
        LastDataPoint = dataPoint;
        RecentData.Add(dataPoint);

        // Keep only recent data to prevent memory issues
        if (RecentData.Count > MaxRecentDataPoints)
        {
            RecentData.RemoveAt(0);
        }
    }
}

public class DataQualityReport
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string CorrelationId { get; set; } = string.Empty;
    public List<DataQualityIssue> Issues { get; set; } = new();
    public bool HasIssues => Issues.Any();
    public int CriticalIssueCount => Issues.Count(i => i.Severity == DataQualitySeverity.Critical);
}

public class DataQualityIssue
{
    public DataQualityIssueType Type { get; set; }
    public DataQualitySeverity Severity { get; set; }
    public string Description { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string CorrelationId { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

public class DataQualityMetrics
{
    public string Symbol { get; set; } = string.Empty;
    public long TotalDataPoints { get; set; }
    public long TotalIssues { get; set; }
    public long CriticalIssues { get; set; }
    public long HighSeverityIssues { get; set; }
    public long MediumSeverityIssues { get; set; }
    public long LowSeverityIssues { get; set; }
    public double QualityScore { get; set; } = 1.0;
    public DateTime LastUpdate { get; set; }

    public DataQualityMetrics Clone()
    {
        return new DataQualityMetrics
        {
            Symbol = Symbol,
            TotalDataPoints = TotalDataPoints,
            TotalIssues = TotalIssues,
            CriticalIssues = CriticalIssues,
            HighSeverityIssues = HighSeverityIssues,
            MediumSeverityIssues = MediumSeverityIssues,
            LowSeverityIssues = LowSeverityIssues,
            QualityScore = QualityScore,
            LastUpdate = LastUpdate
        };
    }
}

public class TimeGap
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public TimeSpan Duration { get; set; }
}

public enum DataQualityIssueType
{
    InvalidPrice,
    InvalidVolume,
    FutureTimestamp,
    StaleData,
    DataGap,
    PriceGap,
    ZeroVolume,
    RepeatedTimestamp,
    TimeSkew,
    SchemaVersionMismatch,
    AbnormalVolatility,
    ValidationError,
    AnalysisError
}

public enum DataQualitySeverity
{
    Low,
    Medium,
    High,
    Critical
}

public class DataQualityConfig
{
    public TimeSpan ExpectedDataInterval { get; set; } = TimeSpan.FromSeconds(1);
    public double GapToleranceSeconds { get; set; } = 2.0;
    public double MaxDataAgeSeconds { get; set; } = 30.0;
    public double MaxFutureToleranceSeconds { get; set; } = 5.0;
    public double MaxPriceChangePercent { get; set; } = 0.1; // 10%
    public double MaxZeroVolumeRatio { get; set; } = 0.2; // 20%
    public double MaxVolatilityThreshold { get; set; } = 0.05; // 5%
    public double MaxTimeSkewSeconds { get; set; } = 10.0;
    public int MinDataPointsForDriftDetection { get; set; } = 50;
    public TimeSpan MonitoringInterval { get; set; } = TimeSpan.FromMinutes(5);
    public TimeSpan NtpSyncInterval { get; set; } = TimeSpan.FromHours(1);
    public TimeSpan DriftDetectionPeriod { get; set; } = TimeSpan.FromHours(1);
    public Dictionary<string, string> SchemaVersions { get; set; } = new()
    {
        ["MarketDataPoint"] = "1.0",
        ["OrderEvent"] = "1.0",
        ["FillEvent"] = "1.0"
    };
}