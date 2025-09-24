using System;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Utilities
{
    /// <summary>
    /// Utility class to reduce code duplication in Timer initialization patterns
    /// Addresses SonarCloud duplication concerns by providing reusable Timer creation methods
    /// </summary>
    public static class TimerHelper
    {
        /// <summary>
        /// Creates a Timer that executes an async method with fire-and-forget pattern
        /// Eliminates duplication of Timer(_ => _ = AsyncMethod()) pattern
        /// </summary>
        /// <param name="asyncCallback">The async method to execute</param>
        /// <param name="dueTime">The time interval to wait before first execution</param>
        /// <param name="period">The time interval between subsequent executions</param>
        /// <returns>Configured Timer instance</returns>
        public static Timer CreateAsyncTimer(Func<Task> asyncCallback, TimeSpan dueTime, TimeSpan period)
        {
            return new Timer(_ => _ = asyncCallback(), null, 
                (int)dueTime.TotalMilliseconds, 
                (int)period.TotalMilliseconds);
        }

        /// <summary>
        /// Creates a Timer that executes a synchronous method with immediate start
        /// Eliminates duplication of Timer(_ => SyncMethod()) pattern
        /// </summary>
        /// <param name="syncCallback">The synchronous method to execute</param>
        /// <param name="period">The time interval between executions</param>
        /// <returns>Configured Timer instance</returns>
        public static Timer CreateSyncTimer(Action syncCallback, TimeSpan period)
        {
            return new Timer(_ => syncCallback(), null, 0, (int)period.TotalMilliseconds);
        }

        /// <summary>
        /// Creates a Timer that executes an async method with immediate start
        /// Most common pattern used across the codebase
        /// </summary>
        /// <param name="asyncCallback">The async method to execute</param>
        /// <param name="period">The time interval between executions</param>
        /// <returns>Configured Timer instance</returns>
        public static Timer CreateAsyncTimerWithImmediateStart(Func<Task> asyncCallback, TimeSpan period)
        {
            return CreateAsyncTimer(asyncCallback, TimeSpan.Zero, period);
        }

        /// <summary>
        /// Creates a Timer for periodic health checks (common pattern)
        /// </summary>
        /// <param name="healthCheckCallback">The health check method to execute</param>
        /// <param name="interval">The check interval</param>
        /// <returns>Configured Timer instance</returns>
        public static Timer CreateHealthCheckTimer(Func<Task> healthCheckCallback, TimeSpan interval)
        {
            return CreateAsyncTimerWithImmediateStart(healthCheckCallback, interval);
        }
    }
}