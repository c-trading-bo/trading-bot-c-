using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Globalization;

namespace TopstepX.Bot.Core.Services
{
    /// <summary>
    /// Comprehensive Error Handling and Monitoring System
    /// Tracks system health, errors, and performance metrics
    /// </summary>
    public class ErrorHandlingMonitoringSystem
    {
        private readonly ILogger<ErrorHandlingMonitoringSystem> _logger;
        private readonly ConcurrentDictionary<string, ErrorRecord> _recentErrors = new();
        private readonly ConcurrentDictionary<string, ComponentHealth> _componentHealth = new();
        private readonly Timer _healthCheckTimer;
        private readonly Timer _reportingTimer;
        private readonly string _errorLogPath;
        private readonly object _lockObject = new();
        private volatile bool _isSystemHealthy = true;
        
        public event EventHandler<CriticalErrorEventArgs>? CriticalErrorDetected;
        public event EventHandler<HealthStatusEventArgs>? HealthStatusChanged;
        
        public class ErrorRecord
        {
            public string ErrorId { get; set; } = string.Empty;
            public DateTime Timestamp { get; set; }
            public string Component { get; set; } = string.Empty;
            public string ErrorType { get; set; } = string.Empty;
            public string Message { get; set; } = string.Empty;
            public string? StackTrace { get; set; }
            public ErrorSeverity Severity { get; set; }
            public int OccurrenceCount { get; set; } = 1;
            public bool IsResolved { get; set; }
            public string? Resolution { get; set; }
        }
        
        public class ComponentHealth
        {
            public string ComponentName { get; set; } = string.Empty;
            public HealthStatus Status { get; set; } = HealthStatus.Unknown;
            public DateTime LastCheck { get; set; }
            public string? LastError { get; set; }
            public int ErrorCount { get; set; }
            public double SuccessRate { get; set; } = 100.0;
            public long ResponseTimeMs { get; set; }
            public Dictionary<string, object> Metrics { get; } = new();
        }
        
        public enum ErrorSeverity
        {
            Low = 1,
            Medium = 2,
            High = 3,
            Critical = 4
        }
        
        public enum HealthStatus
        {
            Healthy = 1,
            Warning = 2,
            Critical = 3,
            Unknown = 4
        }
        
        public ErrorHandlingMonitoringSystem(ILogger<ErrorHandlingMonitoringSystem> logger)
        {
            _logger = logger;
            _errorLogPath = Path.Combine(Directory.GetCurrentDirectory(), "logs", "errors");
            
            // Ensure log directory exists
            Directory.CreateDirectory(_errorLogPath);
            
            // Setup timers
            _healthCheckTimer = new Timer(PerformHealthChecks, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
            _reportingTimer = new Timer(GenerateHealthReport, null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));
            
            _logger.LogInformation("🏥 Error Handling & Monitoring System initialized");
        }
        
        /// <summary>
        /// Log an error with automatic categorization and alerting
        /// </summary>
        public async Task LogErrorAsync(string component, Exception exception, ErrorSeverity severity = ErrorSeverity.Medium, Dictionary<string, object>? additionalData = null)
        {
            if (component is null) throw new ArgumentNullException(nameof(component));
            if (exception is null) throw new ArgumentNullException(nameof(exception));

            try
            {
                var errorId = GenerateErrorId(component, exception);
                var timestamp = DateTime.UtcNow;
                
                // Check if this is a recurring error
                if (_recentErrors.TryGetValue(errorId, out var existingError))
                {
                    existingError.OccurrenceCount++;
                    existingError.Timestamp = timestamp;
                    
                    // Escalate severity if error is recurring frequently
                    if (existingError.OccurrenceCount > 5 && severity < ErrorSeverity.Critical)
                    {
                        severity = ErrorSeverity.High;
                    }
                }
                else
                {
                    var errorRecord = new ErrorRecord
                    {
                        ErrorId = errorId,
                        Timestamp = timestamp,
                        Component = component,
                        ErrorType = exception.GetType().Name,
                        Message = exception.Message,
                        StackTrace = exception.StackTrace,
                        Severity = severity
                    };
                    
                    _recentErrors[errorId] = errorRecord;
                }
                
                // Log to structured logging
                _logger.LogError(exception, "❌ [{Component}] {ErrorType}: {Message} | Severity: {Severity} | ErrorId: {ErrorId}", 
                    component, exception.GetType().Name, exception.Message, severity, errorId);
                
                // Write to error log file
                await WriteErrorToFileAsync(component, exception, severity, errorId, additionalData).ConfigureAwait(false);
                
                // Update component health
                UpdateComponentHealth(component, HealthStatus.Warning, exception.Message);
                
                // Check if this is a critical error that requires immediate attention
                if (severity >= ErrorSeverity.Critical)
                {
                    await HandleCriticalErrorAsync(component, exception, errorId).ConfigureAwait(false);
                }
                
                // Clean up old error records (keep last 24 hours)
                CleanupOldErrors();
            }
            catch (Exception ex)
            {
                // Failsafe logging - don't let error logging itself crash the system
                _logger.LogCritical(ex, "🚨 CRITICAL: Error in error logging system");
            }
        }
        
        private string GenerateErrorId(string component, Exception exception)
        {
            var baseString = $"{component}_{exception.GetType().Name}_{exception.Message?.GetHashCode()}";
            return Convert.ToHexString(System.Text.Encoding.UTF8.GetBytes(baseString))[..12];
        }
        
        private async Task WriteErrorToFileAsync(string component, Exception exception, ErrorSeverity severity, string errorId, Dictionary<string, object>? additionalData)
        {
            try
            {
                var errorLogEntry = new
                {
                    Timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture),
                    ErrorId = errorId,
                    Component = component,
                    Severity = severity.ToString(),
                    ErrorType = exception.GetType().Name,
                    Message = exception.Message,
                    StackTrace = exception.StackTrace,
                    AdditionalData = additionalData,
                    ProcessId = Environment.ProcessId,
                    MachineName = Environment.MachineName
                };
                
                var json = JsonSerializer.Serialize(errorLogEntry, new JsonSerializerOptions { WriteIndented = true });
                var fileName = $"error_{DateTime.UtcNow:yyyyMMdd}_{errorId}.json";
                var filePath = Path.Combine(_errorLogPath, fileName);
                
                await File.WriteAllTextAsync(filePath, json).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ Failed to write error to file");
            }
        }
        
        private async Task HandleCriticalErrorAsync(string component, Exception exception, string errorId)
        {
            try
            {
                _isSystemHealthy = false;
                
                var eventArgs = new CriticalErrorEventArgs
                {
                    Component = component,
                    Exception = exception,
                    ErrorId = errorId,
                    Timestamp = DateTime.UtcNow,
                    RequiresImmediateAction = true
                };
                
                _logger.LogCritical("🚨 CRITICAL ERROR DETECTED in {Component}: {Message}", component, exception.Message);
                
                // Fire critical error event
                CriticalErrorDetected?.Invoke(this, eventArgs);
                
                // Create immediate alert file
                var alertPath = Path.Combine(Directory.GetCurrentDirectory(), $"CRITICAL_ALERT_{DateTime.UtcNow:yyyyMMdd_HHmmss}.txt");
                var alertContent = $"""
                    CRITICAL SYSTEM ERROR
                    =====================
                    Time: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC
                    Component: {component}
                    Error: {exception.Message}
                    Error ID: {errorId}
                    
                    IMMEDIATE ACTION REQUIRED:
                    - Check system logs for details
                    - Verify trading positions are safe
                    - Consider emergency stop if trading related
                    
                    Stack Trace:
                    {exception.StackTrace}
                    """;
                    
                await File.WriteAllTextAsync(alertPath, alertContent).ConfigureAwait(false);
                
                _logger.LogInformation("📋 Critical alert file created: {AlertPath}", alertPath);
            }
            catch (Exception ex)
            {
                _logger.LogCritical(ex, "🚨 Failed to handle critical error");
            }
        }
        
        /// <summary>
        /// Update component health status
        /// </summary>
        public void UpdateComponentHealth(string componentName, HealthStatus status, string? errorMessage = null, Dictionary<string, object>? metrics = null)
        {
            lock (_lockObject)
            {
                if (!_componentHealth.TryGetValue(componentName, out var health))
                {
                    health = new ComponentHealth { ComponentName = componentName };
                    _componentHealth[componentName] = health;
                }
                
                health.Status = status;
                health.LastCheck = DateTime.UtcNow;
                
                if (!string.IsNullOrEmpty(errorMessage))
                {
                    health.LastError = errorMessage;
                    health.ErrorCount++;
                }
                
                if (metrics != null)
                {
                    foreach (var metric in metrics)
                    {
                        health.Metrics[metric.Key] = metric.Value;
                    }
                }
                
                // Calculate success rate
                var totalChecks = health.ErrorCount + (health.Metrics.ContainsKey("SuccessCount") ? Convert.ToInt32(health.Metrics["SuccessCount"]) : 1);
                health.SuccessRate = Math.Max(0, 100.0 - (health.ErrorCount * 100.0 / totalChecks));
                
                _logger.LogDebug("📊 Component health updated: {Component} - {Status} (Success Rate: {SuccessRate:F1}%)", 
                    componentName, status, health.SuccessRate);
            }
        }
        
        /// <summary>
        /// Record successful operation for component health
        /// </summary>
        public void RecordSuccess(string componentName, long responseTimeMs = 0)
        {
            var metrics = new Dictionary<string, object>
            {
                ["SuccessCount"] = (_componentHealth.TryGetValue(componentName, out var health) && 
                                  health.Metrics.ContainsKey("SuccessCount")) ? 
                                  Convert.ToInt32(health.Metrics["SuccessCount"]) + 1 : 1
            };
            
            if (responseTimeMs > 0)
            {
                metrics["ResponseTimeMs"] = responseTimeMs;
            }
            
            UpdateComponentHealth(componentName, HealthStatus.Healthy, null, metrics);
        }
        
        private void PerformHealthChecks(object? state)
        {
            try
            {
                var overallHealth = CalculateOverallHealth();
                var previousHealth = _isSystemHealthy;
                _isSystemHealthy = overallHealth >= 80.0; // 80% threshold for healthy system
                
                if (previousHealth != _isSystemHealthy)
                {
                    var eventArgs = new HealthStatusEventArgs
                    {
                        IsHealthy = _isSystemHealthy,
                        OverallHealthScore = overallHealth,
                        Timestamp = DateTime.UtcNow
                    };
                    
                    // Add component health summary to the dictionary
                    var componentHealthSummary = GetComponentHealthSummary();
                    foreach (var item in componentHealthSummary)
                    {
                        eventArgs.ComponentHealthSummary[item.Key] = item.Value;
                    }
                    
                    _logger.LogWarning("🏥 System health status changed: {IsHealthy} (Score: {HealthScore:F1}%)", 
                        _isSystemHealthy ? "HEALTHY" : "UNHEALTHY", overallHealth);
                    
                    HealthStatusChanged?.Invoke(this, eventArgs);
                }
                
                // Check for stale components (no updates in 5 minutes)
                var staleComponents = _componentHealth.Values
                    .Where(c => DateTime.UtcNow - c.LastCheck > TimeSpan.FromMinutes(5))
                    .ToList();
                
                foreach (var stale in staleComponents)
                {
                    stale.Status = HealthStatus.Unknown;
                    _logger.LogWarning("⚠️ Component {Component} is stale (last update: {LastCheck})", 
                        stale.ComponentName, stale.LastCheck);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error during health check");
            }
        }
        
        private double CalculateOverallHealth()
        {
            if (!_componentHealth.Any()) return 100.0;
            
            var totalScore = 0.0;
            var componentCount = _componentHealth.Count;
            
            foreach (var health in _componentHealth.Values)
            {
                var componentScore = health.Status switch
                {
                    HealthStatus.Healthy => 100.0,
                    HealthStatus.Warning => 70.0,
                    HealthStatus.Critical => 20.0,
                    HealthStatus.Unknown => 50.0,
                    _ => 0.0
                };
                
                // Factor in success rate
                componentScore = componentScore * (health.SuccessRate / 100.0);
                totalScore += componentScore;
            }
            
            return totalScore / componentCount;
        }
        
        private Dictionary<string, object> GetComponentHealthSummary()
        {
            return _componentHealth.Values.ToDictionary(
                h => h.ComponentName,
                h => new
                {
                    Status = h.Status.ToString(),
                    SuccessRate = h.SuccessRate,
                    ErrorCount = h.ErrorCount,
                    LastCheck = h.LastCheck,
                    LastError = h.LastError
                } as object
            );
        }
        
        private void GenerateHealthReport(object? state)
        {
            try
            {
                var report = new
                {
                    GeneratedAt = DateTime.UtcNow,
                    OverallHealth = CalculateOverallHealth(),
                    IsSystemHealthy = _isSystemHealthy,
                    ComponentHealth = _componentHealth.Values.ToList(),
                    RecentErrors = _recentErrors.Values
                        .Where(e => DateTime.UtcNow - e.Timestamp < TimeSpan.FromHours(1))
                        .OrderByDescending(e => e.Timestamp)
                        .Take(20)
                        .ToList(),
                    Summary = new
                    {
                        TotalComponents = _componentHealth.Count,
                        HealthyComponents = _componentHealth.Values.Count(c => c.Status == HealthStatus.Healthy),
                        CriticalComponents = _componentHealth.Values.Count(c => c.Status == HealthStatus.Critical),
                        RecentErrorCount = _recentErrors.Values.Count(e => DateTime.UtcNow - e.Timestamp < TimeSpan.FromHours(1))
                    }
                };
                
                var reportPath = Path.Combine(Directory.GetCurrentDirectory(), "logs", $"health_report_{DateTime.UtcNow:yyyyMMdd_HHmm}.json");
                Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
                
                var json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(reportPath, json);
                
                _logger.LogInformation("📋 Health report generated: {ReportPath} | Health Score: {HealthScore:F1}%", 
                    reportPath, report.OverallHealth);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error generating health report");
            }
        }
        
        private void CleanupOldErrors()
        {
            try
            {
                var cutoffTime = DateTime.UtcNow.AddHours(-24);
                var staleErrors = _recentErrors.Values
                    .Where(e => e.Timestamp < cutoffTime)
                    .Select(e => e.ErrorId)
                    .ToList();
                
                foreach (var errorId in staleErrors)
                {
                    _recentErrors.TryRemove(errorId, out _);
                }
                
                if (staleErrors.Any())
                {
                    _logger.LogDebug("🧹 Cleaned up {Count} old error records", staleErrors.Count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ Error during error cleanup");
            }
        }
        
        /// <summary>
        /// Get current system health status
        /// </summary>
        public SystemHealthStatus GetSystemHealth()
        {
            return new SystemHealthStatus
            {
                IsHealthy = _isSystemHealthy,
                OverallHealthScore = CalculateOverallHealth(),
                ComponentCount = _componentHealth.Count,
                HealthyComponents = _componentHealth.Values.Count(c => c.Status == HealthStatus.Healthy),
                CriticalComponents = _componentHealth.Values.Count(c => c.Status == HealthStatus.Critical),
                RecentErrorCount = _recentErrors.Values.Count(e => DateTime.UtcNow - e.Timestamp < TimeSpan.FromHours(1)),
                LastHealthCheck = DateTime.UtcNow
            };
        }
        
        /// <summary>
        /// Get recent errors
        /// </summary>
        public List<ErrorRecord> GetRecentErrors(int maxCount = 50)
        {
            return _recentErrors.Values
                .OrderByDescending(e => e.Timestamp)
                .Take(maxCount)
                .ToList();
        }
        
        /// <summary>
        /// Get component health details
        /// </summary>
        public List<ComponentHealth> GetComponentHealth()
        {
            return _componentHealth.Values
                .OrderBy(c => c.ComponentName)
                .ToList();
        }
        
        /// <summary>
        /// Mark error as resolved
        /// </summary>
        public void MarkErrorResolved(string errorId, string resolution)
        {
            if (_recentErrors.TryGetValue(errorId, out var error))
            {
                error.IsResolved = true;
                error.Resolution = resolution;
                _logger.LogInformation("✅ Error resolved: {ErrorId} - {Resolution}", errorId, resolution);
            }
        }
        
        public void Dispose()
        {
            _healthCheckTimer?.Dispose();
            _reportingTimer?.Dispose();
        }
    }
    
    public class CriticalErrorEventArgs : EventArgs
    {
        public string Component { get; set; } = string.Empty;
        public Exception Exception { get; set; } = new InvalidOperationException();
        public string ErrorId { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public bool RequiresImmediateAction { get; set; }
    }
    
    public class HealthStatusEventArgs : EventArgs
    {
        public bool IsHealthy { get; set; }
        public double OverallHealthScore { get; set; }
        public DateTime Timestamp { get; set; }
        public Dictionary<string, object> ComponentHealthSummary { get; } = new();
    }
    
    public class SystemHealthStatus
    {
        public bool IsHealthy { get; set; }
        public double OverallHealthScore { get; set; }
        public int ComponentCount { get; set; }
        public int HealthyComponents { get; set; }
        public int CriticalComponents { get; set; }
        public int RecentErrorCount { get; set; }
        public DateTime LastHealthCheck { get; set; }
    }
}