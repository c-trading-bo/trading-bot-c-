using Microsoft.Extensions.Logging;
using System;
using System.Threading.Tasks;

namespace BotCore.Utilities
{
    /// <summary>
    /// Utility class to reduce code duplication in exception handling patterns
    /// Addresses SonarCloud duplication concerns by providing standardized exception handling
    /// </summary>
    public static class ExceptionHelper
    {
        /// <summary>
        /// Handles exceptions in async operations with standardized logging
        /// Eliminates duplication of try-catch-log patterns
        /// </summary>
        /// <param name="operation">The async operation to execute</param>
        /// <param name="logger">Logger instance for error reporting</param>
        /// <param name="componentName">Name of the component performing the operation</param>
        /// <param name="operationDescription">Description of what operation failed</param>
        /// <param name="suppressExceptions">Whether to suppress exceptions (default: true)</param>
        /// <returns>True if operation succeeded, false if failed</returns>
        public static async Task<bool> ExecuteWithLogging(
            Func<Task> operation, 
            ILogger logger, 
            string componentName, 
            string operationDescription, 
            bool suppressExceptions = true)
        {
            if (operation is null) throw new ArgumentNullException(nameof(operation));
            if (logger is null) throw new ArgumentNullException(nameof(logger));
            if (componentName is null) throw new ArgumentNullException(nameof(componentName));
            if (operationDescription is null) throw new ArgumentNullException(nameof(operationDescription));

            try
            {
                await operation().ConfigureAwait(false);
                return true;
            }
            catch (Exception ex)
            {
                LoggingHelper.LogError(logger, ex, componentName, operationDescription);
                if (!suppressExceptions)
                    throw;
                return false;
            }
        }

        /// <summary>
        /// Handles exceptions in sync operations with standardized logging
        /// </summary>
        public static bool ExecuteWithLogging(
            Action operation, 
            ILogger logger, 
            string componentName, 
            string operationDescription, 
            bool suppressExceptions = true)
        {
            if (operation is null) throw new ArgumentNullException(nameof(operation));
            if (logger is null) throw new ArgumentNullException(nameof(logger));
            if (componentName is null) throw new ArgumentNullException(nameof(componentName));
            if (operationDescription is null) throw new ArgumentNullException(nameof(operationDescription));

            try
            {
                operation();
                return true;
            }
            catch (Exception ex)
            {
                LoggingHelper.LogError(logger, ex, componentName, operationDescription);
                if (!suppressExceptions)
                    throw;
                return false;
            }
        }

        /// <summary>
        /// Handles exceptions with return values and standardized logging
        /// </summary>
        public static async Task<T?> ExecuteWithLogging<T>(
            Func<Task<T>> operation, 
            ILogger logger, 
            string componentName, 
            string operationDescription, 
            T? defaultValue = default,
            bool suppressExceptions = true)
        {
            if (operation is null) throw new ArgumentNullException(nameof(operation));
            if (logger is null) throw new ArgumentNullException(nameof(logger));
            if (componentName is null) throw new ArgumentNullException(nameof(componentName));
            if (operationDescription is null) throw new ArgumentNullException(nameof(operationDescription));

            try
            {
                return await operation().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                LoggingHelper.LogError(logger, ex, componentName, operationDescription);
                if (!suppressExceptions)
                    throw;
                return defaultValue;
            }
        }
    }
}