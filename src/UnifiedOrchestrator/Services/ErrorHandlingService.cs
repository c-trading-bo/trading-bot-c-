using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Error handling service with graceful degradation and fallback logging
/// Provides circuit breaker pattern and Windows Event Log fallback for critical errors
/// </summary>
public class ErrorHandlingService : IHostedService
{
    private readonly ILogger<ErrorHandlingService> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly TradingLoggerOptions _options;
    private readonly EventLog? _eventLog;
    private readonly Timer _circuitBreakerTimer;
    private volatile bool _fileSystemAvailable = true;
    private volatile bool _eventLogAvailable = false;
    private int _consecutiveFileErrors = 0;
    private readonly object _lockObject = new();

    public ErrorHandlingService(
        ILogger<ErrorHandlingService> logger,
        ITradingLogger tradingLogger,
        Microsoft.Extensions.Options.IOptions<TradingLoggerOptions> options)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _options = options.Value;
        
        // Initialize Windows Event Log on Windows platforms
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            try
            {
                if (!EventLog.SourceExists("TradingBot"))
                {
                    EventLog.CreateEventSource("TradingBot", "Application");
                }
                _eventLog = new EventLog("Application") { Source = "TradingBot" };
                _eventLogAvailable = true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to initialize Windows Event Log - fallback logging unavailable");
            }
        }

        // Circuit breaker timer - check every 60 seconds
        _circuitBreakerTimer = new Timer(CheckCircuitBreaker, null, 
            TimeSpan.FromSeconds(60), TimeSpan.FromSeconds(60));
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "ErrorHandler", 
            "Error handling service started", new
            {
                fileSystemAvailable = _fileSystemAvailable,
                eventLogAvailable = _eventLogAvailable,
                platform = RuntimeInformation.OSDescription
            });
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "ErrorHandler", 
            "Error handling service stopped");
            
        _circuitBreakerTimer?.Dispose();
        _eventLog?.Dispose();
    }

    /// <summary>
    /// Log critical error with fallback mechanisms
    /// </summary>
    public async Task LogCriticalErrorAsync(string component, string message, Exception? exception = null)
    {
        var errorData = new
        {
            component,
            message,
            exception = exception?.Message,
            exceptionType = exception?.GetType().Name,
            timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC")
        };

        try
        {
            // Try primary logging first
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, component, message, errorData);
            
            // Reset error counter on success
            lock (_lockObject)
            {
                _consecutiveFileErrors = 0;
                _fileSystemAvailable = true;
            }
        }
        catch (Exception ex)
        {
            // Primary logging failed - use fallback mechanisms
            HandleLoggingFailure(component, message, exception, ex);
        }

        // Also use Windows Event Log for critical errors
        if (_eventLogAvailable && _eventLog != null && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            try
            {
                var eventMessage = $"TradingBot Critical Error - {component}: {message}";
                if (exception != null)
                {
                    eventMessage += $"\nException: {exception.Message}\nStackTrace: {exception.StackTrace}";
                }
                
                _eventLog.WriteEntry(eventMessage, EventLogEntryType.Error);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to write to Windows Event Log");
            }
        }
    }

    private void HandleLoggingFailure(string component, string message, Exception? originalException, Exception loggingException)
    {
        lock (_lockObject)
        {
            _consecutiveFileErrors++;
            
            if (_consecutiveFileErrors >= 5)
            {
                _fileSystemAvailable = false;
                _logger.LogError("File system logging circuit breaker activated after {Errors} consecutive failures", 
                    _consecutiveFileErrors);
            }
        }

        // Fallback to console logging
        var fallbackMessage = $"CRITICAL ERROR [{component}]: {message}";
        if (originalException != null)
        {
            fallbackMessage += $" | Exception: {originalException.Message}";
        }
        fallbackMessage += $" | Logging Error: {loggingException.Message}";
        
        Console.WriteLine($"[{DateTime.UtcNow:HH:mm:ss.fff}] 🔴 ERROR FALLBACK: {fallbackMessage}");

        // Try to create emergency log file
        TryCreateEmergencyLog(component, message, originalException, loggingException);
    }

    private void TryCreateEmergencyLog(string component, string message, Exception? originalException, Exception loggingException)
    {
        try
        {
            var emergencyPath = Path.Combine(Path.GetTempPath(), "TradingBot_Emergency.log");
            var emergencyEntry = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC}] CRITICAL: {component} - {message}\n";
            
            if (originalException != null)
            {
                emergencyEntry += $"Original Exception: {originalException}\n";
            }
            
            emergencyEntry += $"Logging Exception: {loggingException}\n\n";
            
            File.AppendAllText(emergencyPath, emergencyEntry);
        }
        catch
        {
            // If even emergency logging fails, there's nothing more we can do
        }
    }

    private async void CheckCircuitBreaker(object? state)
    {
        try
        {
            // Test file system availability
            var testPath = Path.Combine(_options.LogDirectory, "health_check.tmp");
            
            try
            {
                await File.WriteAllTextAsync(testPath, DateTime.UtcNow.ToString());
                File.Delete(testPath);
                
                // File system is working - reset circuit breaker
                lock (_lockObject)
                {
                    if (!_fileSystemAvailable)
                    {
                        _fileSystemAvailable = true;
                        _consecutiveFileErrors = 0;
                        _logger.LogInformation("File system logging circuit breaker reset - logging restored");
                    }
                }
            }
            catch (Exception)
            {
                // File system still not available
                lock (_lockObject)
                {
                    _fileSystemAvailable = false;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error during circuit breaker check");
        }
    }

    /// <summary>
    /// Get current error handling status for monitoring
    /// </summary>
    public object GetStatus()
    {
        return new
        {
            fileSystemAvailable = _fileSystemAvailable,
            eventLogAvailable = _eventLogAvailable,
            consecutiveFileErrors = _consecutiveFileErrors,
            circuitBreakerActive = !_fileSystemAvailable
        };
    }
}