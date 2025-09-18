using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Security;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System.Globalization;

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
    private volatile bool _fileLoggingFallbackAvailable = false;
    private string? _fallbackLogPath;
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
            catch (SecurityException ex)
            {
                _logger.LogWarning(ex, "Failed to initialize Windows Event Log due to permissions - setting up file logging fallback");
                SetupFileLoggingFallback();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to initialize Windows Event Log - setting up file logging fallback");
                SetupFileLoggingFallback();
            }
        }
        else
        {
            // On non-Windows platforms, always use file logging
            SetupFileLoggingFallback();
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
            }).ConfigureAwait(false);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "ErrorHandler", 
            "Error handling service stopped").ConfigureAwait(false);
            
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
            timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC", CultureInfo.InvariantCulture)
        };

        try
        {
            // Try primary logging first
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, component, message, errorData).ConfigureAwait(false);
            
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

        // Try file logging fallback first
        if (_fileLoggingFallbackAvailable)
        {
            var fallbackMessage = $"CRITICAL ERROR [{component}]: {message} | Logging Error: {loggingException.Message}";
            WriteToFileLoggingFallback("CRITICAL", fallbackMessage, originalException);
        }
        
        // Fallback to console logging
        var consoleMessage = $"CRITICAL ERROR [{component}]: {message}";
        if (originalException != null)
        {
            consoleMessage += $" | Exception: {originalException.Message}";
        }
        consoleMessage += $" | Logging Error: {loggingException.Message}";
        
        Console.WriteLine($"[{DateTime.UtcNow:HH:mm:ss.fff}] 🔴 ERROR FALLBACK: {consoleMessage}");

        // Try to create emergency log file as last resort
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
                await File.WriteAllTextAsync(testPath, DateTime.UtcNow.ToString()).ConfigureAwait(false);
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
    /// Setup file logging fallback when Event Log is not available
    /// </summary>
    private void SetupFileLoggingFallback()
    {
        try
        {
            var logDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TradingBot", "Logs");
            Directory.CreateDirectory(logDirectory);
            
            _fallbackLogPath = Path.Combine(logDirectory, $"trading-errors-{DateTime.UtcNow:yyyyMMdd}.log");
            
            // Test write to ensure the path is accessible
            var testMessage = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}] File logging fallback initialized{Environment.NewLine}";
            File.AppendAllText(_fallbackLogPath, testMessage);
            
            _fileLoggingFallbackAvailable = true;
            _logger.LogInformation("File logging fallback initialized at: {FallbackLogPath}", _fallbackLogPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to setup file logging fallback - error handling will be limited");
            _fileLoggingFallbackAvailable = false;
        }
    }

    /// <summary>
    /// Write critical error to file fallback
    /// </summary>
    private void WriteToFileLoggingFallback(string level, string message, Exception? exception = null)
    {
        if (!_fileLoggingFallbackAvailable || string.IsNullOrEmpty(_fallbackLogPath))
            return;

        try
        {
            var logEntry = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}] [{level}] {message}";
            if (exception != null)
            {
                logEntry += $" Exception: {exception.GetType().Name}: {exception.Message}";
                if (exception.StackTrace != null)
                {
                    logEntry += Environment.NewLine + exception.StackTrace;
                }
            }
            logEntry += Environment.NewLine;

            File.AppendAllText(_fallbackLogPath, logEntry);
        }
        catch (Exception ex)
        {
            // If file logging also fails, we can only log to the regular logger
            _logger.LogError(ex, "File logging fallback failed for message: {Message}", message);
            _fileLoggingFallbackAvailable = false;
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
            fileLoggingFallbackAvailable = _fileLoggingFallbackAvailable,
            fallbackLogPath = _fallbackLogPath,
            consecutiveFileErrors = _consecutiveFileErrors,
            circuitBreakerActive = !_fileSystemAvailable
        };
    }
}