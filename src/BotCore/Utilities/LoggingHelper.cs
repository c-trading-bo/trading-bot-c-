using Microsoft.Extensions.Logging;
using System;

namespace BotCore.Utilities
{
    /// <summary>
    /// Utility class to reduce code duplication in logging patterns
    /// Addresses SonarCloud duplication concerns by providing standardized logging methods
    /// </summary>
    public static class LoggingHelper
    {
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
                logger.LogInformation("[{Component}] {Component} initialized", componentName, componentName);
            }
            else
            {
                logger.LogInformation("[{Component}] {Component} initialized - {Info}", componentName, componentName, additionalInfo);
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
            logger.LogInformation("[{Service}] Started - {Description} every {Interval}", serviceName, description, interval);
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
                logger.LogError(exception, "[{Component}] Failed to {Operation}: {Context}", componentName, operation, additionalContext);
            }
            else
            {
                logger.LogError(exception, "[{Component}] Failed to {Operation}", componentName, operation);
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
            logger.LogWarning("[{Component}] DEPRECATED: {Reason}. Use {Alternative} instead.", componentName, reason, alternative);
        }
    }
}