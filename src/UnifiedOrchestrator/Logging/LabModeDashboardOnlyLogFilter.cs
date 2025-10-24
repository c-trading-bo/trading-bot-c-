using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Logging;

/// <summary>
/// Custom log filter that suppresses ALL console logging when Lab Mode dashboard is active
/// This ensures only the dashboard is visible in the terminal (no training logs)
/// </summary>
public sealed class LabModeDashboardOnlyLogFilter : ILoggerProvider
{
    private readonly ILoggerProvider _innerProvider;
    private readonly bool _labModeEnabled;

    public LabModeDashboardOnlyLogFilter(ILoggerProvider innerProvider)
    {
        _innerProvider = innerProvider;
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        _labModeEnabled = labMode == "1";
    }

    public ILogger CreateLogger(string categoryName)
    {
        var innerLogger = _innerProvider.CreateLogger(categoryName);
        
        // If Lab Mode is enabled, wrap logger to suppress console output
        if (_labModeEnabled)
        {
            return new LabModeSuppressingLogger(innerLogger, categoryName);
        }
        
        return innerLogger;
    }

    public void Dispose()
    {
        _innerProvider.Dispose();
    }

    /// <summary>
    /// Logger wrapper that suppresses all log output to console when Lab Mode dashboard is active
    /// File logging still works, only console output is suppressed
    /// </summary>
    private sealed class LabModeSuppressingLogger : ILogger
    {
        private readonly ILogger _innerLogger;
        private readonly string _categoryName;

        public LabModeSuppressingLogger(ILogger innerLogger, string categoryName)
        {
            _innerLogger = innerLogger;
            _categoryName = categoryName;
        }

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull
        {
            return _innerLogger.BeginScope(state);
        }

        public bool IsEnabled(LogLevel logLevel)
        {
            return _innerLogger.IsEnabled(logLevel);
        }

        public void Log<TState>(
            LogLevel logLevel,
            EventId eventId,
            TState state,
            Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            // In Lab Mode, suppress all console logging
            // Logs still go to files if file logging is configured
            // This prevents training logs from interfering with the dashboard display
            
            // Don't call innerLogger.Log() - this suppresses console output
            // File logging is handled separately by other configured providers
        }
    }
}
