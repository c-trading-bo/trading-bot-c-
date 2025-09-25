using Microsoft.Extensions.Logging;
using System;

namespace BotCore.Utilities
{
    /// <summary>
    /// Utility class to reduce code duplication in logging patterns
    /// Addresses SonarCloud duplication concerns by providing standardized logging methods
    /// Uses LoggerMessage delegates for improved performance (CA1848 compliance)
    /// </summary>
    public static class LoggingHelper
    {
        // LoggerMessage delegates for improved performance (CA1848 compliance)
        private static readonly Action<ILogger, string, string, Exception?> _logInitializationSimple = 
            LoggerMessage.Define<string, string>(LogLevel.Information, new EventId(1001, "ComponentInit"), 
                "[{Component}] {ComponentName} initialized");
                
        private static readonly Action<ILogger, string, string, string, Exception?> _logInitializationWithInfo = 
            LoggerMessage.Define<string, string, string>(LogLevel.Information, new EventId(1002, "ComponentInitInfo"), 
                "[{Component}] {ComponentName} initialized - {Info}");
                
        private static readonly Action<ILogger, string, string, TimeSpan, Exception?> _logServiceStarted = 
            LoggerMessage.Define<string, string, TimeSpan>(LogLevel.Information, new EventId(1003, "ServiceStarted"), 
                "[{Service}] Started - {Description} every {Interval}");
                
        private static readonly Action<ILogger, string, string, object?, Exception?> _logErrorWithContext = 
            LoggerMessage.Define<string, string, object?>(LogLevel.Error, new EventId(1004, "ErrorWithContext"), 
                "[{Component}] Failed to {Operation}: {Context}");
                
        private static readonly Action<ILogger, string, string, Exception?> _logErrorSimple = 
            LoggerMessage.Define<string, string>(LogLevel.Error, new EventId(1005, "ErrorSimple"), 
                "[{Component}] Failed to {Operation}");
                
        private static readonly Action<ILogger, string, string, string, Exception?> _logDeprecation = 
            LoggerMessage.Define<string, string, string>(LogLevel.Warning, new EventId(1006, "Deprecation"), 
                "[{Component}] DEPRECATED: {Reason}. Use {Alternative} instead.");

        /// <summary>
        /// Logs component initialization with standardized format
        /// Eliminates duplication of initialization logging patterns
        /// </summary>
        /// <param name="logger">The logger instance</param>
        /// <param name="componentName">Name of the component being initialized</param>
        /// <param name="additionalInfo">Optional additional information</param>
        public static void LogInitialization(ILogger logger, string componentName, string? additionalInfo = null)
        {
            if (string.IsNullOrEmpty(additionalInfo))
            {
                _logInitializationSimple(logger, componentName, componentName, null);
            }
            else
            {
                _logInitializationWithInfo(logger, componentName, componentName, additionalInfo, null);
            }
        }

        /// <summary>
        /// Logs service started with interval information (common pattern)
        /// </summary>
        /// <param name="logger">The logger instance</param>
        /// <param name="serviceName">Name of the service</param>
        /// <param name="interval">The interval for the service</param>
        /// <param name="description">Description of what the service does</param>
        public static void LogServiceStarted(ILogger logger, string serviceName, TimeSpan interval, string description)
        {
            _logServiceStarted(logger, serviceName, description, interval, null);
        }

        /// <summary>
        /// Logs standard error with component context
        /// Eliminates duplication of error logging patterns
        /// </summary>
        /// <param name="logger">The logger instance</param>
        /// <param name="exception">The exception to log</param>
        /// <param name="componentName">Name of the component where error occurred</param>
        /// <param name="operation">The operation that failed</param>
        /// <param name="additionalContext">Optional additional context</param>
        public static void LogError(ILogger logger, Exception exception, string componentName, string operation, object? additionalContext = null)
        {
            if (additionalContext != null)
            {
                _logErrorWithContext(logger, componentName, operation, additionalContext, exception);
            }
            else
            {
                _logErrorSimple(logger, componentName, operation, exception);
            }
        }

        /// <summary>
        /// Logs deprecation warning with standard format
        /// </summary>
        /// <param name="logger">The logger instance</param>
        /// <param name="componentName">Name of the deprecated component</param>
        /// <param name="reason">Reason for deprecation</param>
        /// <param name="alternative">Recommended alternative</param>
        public static void LogDeprecation(ILogger logger, string componentName, string reason, string alternative)
        {
            _logDeprecation(logger, componentName, reason, alternative, null);
        }
    }
}