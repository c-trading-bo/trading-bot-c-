using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Training;

namespace TradingBot.UnifiedOrchestrator.Logging;

/// <summary>
/// Custom log filter that suppresses console logging but routes warnings/errors to Lab Mode dashboard
/// This ensures only the dashboard is visible in the terminal (no training logs)
/// </summary>
public sealed class LabModeDashboardOnlyLogFilter : ILoggerProvider
{
    private readonly ILoggerProvider _innerProvider;
    private readonly bool _labModeEnabled;
    private LabModeDashboardStateManager? _dashboardStateManager;

    public LabModeDashboardOnlyLogFilter(ILoggerProvider innerProvider)
    {
        _innerProvider = innerProvider;
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        _labModeEnabled = labMode == "1";
    }

    /// <summary>
    /// Set the dashboard state manager to receive alerts
    /// </summary>
    public void SetDashboardStateManager(LabModeDashboardStateManager dashboardStateManager)
    {
        _dashboardStateManager = dashboardStateManager;
    }

    public ILogger CreateLogger(string categoryName)
    {
        var innerLogger = _innerProvider.CreateLogger(categoryName);
        
        // If Lab Mode is enabled, wrap logger to suppress console output but route warnings/errors to dashboard
        if (_labModeEnabled)
        {
            return new LabModeDashboardLogger(innerLogger, categoryName, _dashboardStateManager);
        }
        
        return innerLogger;
    }

    public void Dispose()
    {
        _innerProvider.Dispose();
    }

    /// <summary>
    /// Logger wrapper that suppresses console output but routes warnings/errors to dashboard
    /// File logging still works, only console output is suppressed
    /// </summary>
    private sealed class LabModeDashboardLogger : ILogger
    {
        private readonly ILogger _innerLogger;
        private readonly string _categoryName;
        private readonly LabModeDashboardStateManager? _dashboardStateManager;

        public LabModeDashboardLogger(ILogger innerLogger, string categoryName, LabModeDashboardStateManager? dashboardStateManager)
        {
            _innerLogger = innerLogger;
            _categoryName = categoryName;
            _dashboardStateManager = dashboardStateManager;
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
            // Route warnings and errors to dashboard as alerts
            if (_dashboardStateManager != null && 
                (logLevel >= LogLevel.Warning))
            {
                var message = formatter(state, exception);
                var levelStr = logLevel.ToString().ToLowerInvariant();
                
                // Extract source from category name (last part)
                var sourceParts = _categoryName.Split('.');
                var source = sourceParts.Length > 0 ? sourceParts[^1] : _categoryName;
                
                _dashboardStateManager.AddAlert(levelStr, source, message);
            }
            
            // In Lab Mode, suppress all console logging
            // Logs still go to files if file logging is configured
            // This prevents training logs from interfering with the dashboard display
            
            // Don't call innerLogger.Log() - this suppresses console output
            // File logging is handled separately by other configured providers
        }
    }
}
