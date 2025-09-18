using Microsoft.Extensions.Logging;
using System;
using System.Security;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Centralized order management with unified cancellation API
    /// Routes cancel requests to the appropriate broker adapter
    /// Provides consistent logging and error handling across all brokers
    /// </summary>
    public sealed class OrderManager
    {
        private readonly ILogger<OrderManager> _logger;
        private readonly Dictionary<string, IBrokerAdapter> _brokerAdapters;

        // LoggerMessage delegates for performance (CA1848)
        private static readonly Action<ILogger, int, string, Exception?> LogOrderManagerInitialized =
            LoggerMessage.Define<int, string>(
                LogLevel.Information,
                new EventId(1, nameof(LogOrderManagerInitialized)),
                "[OrderManager] Initialized with {AdapterCount} broker adapters: {Brokers}");

        private static readonly Action<ILogger, Exception?> LogCannotCancelOrderNullId =
            LoggerMessage.Define(
                LogLevel.Warning,
                new EventId(2, nameof(LogCannotCancelOrderNullId)),
                "[OrderManager] Cannot cancel order - orderId is null or empty");

        private static readonly Action<ILogger, string, string, string, Exception?> LogCancellingOrder =
            LoggerMessage.Define<string, string, string>(
                LogLevel.Information,
                new EventId(3, nameof(LogCancellingOrder)),
                "[OrderManager] Cancelling order {OrderId} via {Broker}: {Reason}");

        private static readonly Action<ILogger, string, string, Exception?> LogAttemptingCancel =
            LoggerMessage.Define<string, string>(
                LogLevel.Debug,
                new EventId(4, nameof(LogAttemptingCancel)),
                "[OrderManager] Attempting cancel via {Broker} for order {OrderId}");

        private static readonly Action<ILogger, string, string, string, Exception?> LogOrderCancelledSuccessfully =
            LoggerMessage.Define<string, string, string>(
                LogLevel.Information,
                new EventId(5, nameof(LogOrderCancelledSuccessfully)),
                "[OrderManager] ✅ Order {OrderId} cancelled successfully via {Broker}: {Reason}");

        private static readonly Action<ILogger, string, string, Exception?> LogInvalidOperationDuringCancel =
            LoggerMessage.Define<string, string>(
                LogLevel.Debug,
                new EventId(6, nameof(LogInvalidOperationDuringCancel)),
                "[OrderManager] Invalid operation during cancel via {Broker} for order {OrderId}");

        private static readonly Action<ILogger, string, string, Exception?> LogTimeoutDuringCancel =
            LoggerMessage.Define<string, string>(
                LogLevel.Debug,
                new EventId(7, nameof(LogTimeoutDuringCancel)),
                "[OrderManager] Timeout during cancel via {Broker} for order {OrderId}");

        private static readonly Action<ILogger, string, string, Exception?> LogCancelAttemptFailed =
            LoggerMessage.Define<string, string>(
                LogLevel.Debug,
                new EventId(8, nameof(LogCancelAttemptFailed)),
                "[OrderManager] Cancel attempt failed via {Broker} for order {OrderId}");

        private static readonly Action<ILogger, string, string, Exception?> LogFailedCancelViaAnyAdapter =
            LoggerMessage.Define<string, string>(
                LogLevel.Warning,
                new EventId(9, nameof(LogFailedCancelViaAnyAdapter)),
                "[OrderManager] ❌ Failed to cancel order {OrderId} via any broker adapter: {Reason}");

        private static readonly Action<ILogger, string, string, Exception?> LogUnknownBrokerAdapter =
            LoggerMessage.Define<string, string>(
                LogLevel.Error,
                new EventId(10, nameof(LogUnknownBrokerAdapter)),
                "[OrderManager] ❌ Unknown broker adapter: {Broker} for order {OrderId}");

        private static readonly Action<ILogger, string, string, string, Exception?> LogOrderCancelFailed =
            LoggerMessage.Define<string, string, string>(
                LogLevel.Warning,
                new EventId(11, nameof(LogOrderCancelFailed)),
                "[OrderManager] ❌ Failed to cancel order {OrderId} via {Broker}: {Reason}");

        private static readonly Action<ILogger, string, string, string, Exception?> LogInvalidArgumentCancelling =
            LoggerMessage.Define<string, string, string>(
                LogLevel.Error,
                new EventId(12, nameof(LogInvalidArgumentCancelling)),
                "[OrderManager] ❌ Invalid argument cancelling order {OrderId} via {Broker}: {Reason}");

        private static readonly Action<ILogger, string, string, string, Exception?> LogInvalidOperationCancelling =
            LoggerMessage.Define<string, string, string>(
                LogLevel.Error,
                new EventId(13, nameof(LogInvalidOperationCancelling)),
                "[OrderManager] ❌ Invalid operation cancelling order {OrderId} via {Broker}: {Reason}");

        private static readonly Action<ILogger, string, string, string, Exception?> LogExceptionCancelling =
            LoggerMessage.Define<string, string, string>(
                LogLevel.Error,
                new EventId(14, nameof(LogExceptionCancelling)),
                "[OrderManager] ❌ Exception cancelling order {OrderId} via {Broker}: {Reason}");

        private static readonly Action<ILogger, int, int, Exception?> LogBatchCancelCompleted =
            LoggerMessage.Define<int, int>(
                LogLevel.Information,
                new EventId(15, nameof(LogBatchCancelCompleted)),
                "[OrderManager] Batch cancel completed: {Success}/{Total} orders cancelled");

        public OrderManager(ILogger<OrderManager> logger, IEnumerable<IBrokerAdapter> brokerAdapters)
        {
            ArgumentNullException.ThrowIfNull(logger);
            ArgumentNullException.ThrowIfNull(brokerAdapters);
            
            _logger = logger;
            _brokerAdapters = new Dictionary<string, IBrokerAdapter>();
            
            foreach (var adapter in brokerAdapters)
            {
                _brokerAdapters[adapter.BrokerName] = adapter;
            }
            
            LogOrderManagerInitialized(_logger, _brokerAdapters.Count, string.Join(", ", _brokerAdapters.Keys), null);
        }

        /// <summary>
        /// Centralized cancel order API that routes to the correct broker adapter
        /// All cancel requests go through this single method for consistency
        /// </summary>
        public async Task<bool> CancelOrderAsync(string orderId, string? brokerName = null, string? reason = null, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(orderId))
            {
                LogCannotCancelOrderNullId(_logger, null);
                return false;
            }

            reason ??= "Manual cancellation";
            
            LogCancellingOrder(_logger, orderId, brokerName ?? "auto-detect", reason, null);

            // If no broker specified, try all adapters until one succeeds
            if (string.IsNullOrEmpty(brokerName))
            {
                foreach (var adapter in _brokerAdapters.Values)
                {
                    try
                    {
                        LogAttemptingCancel(_logger, adapter.BrokerName, orderId, null);
                        
                        var result = await adapter.CancelOrderAsync(orderId, cancellationToken).ConfigureAwait(false);
                        if (result)
                        {
                            LogOrderCancelledSuccessfully(_logger, orderId, adapter.BrokerName, reason, null);
                            return true;
                        }
                    }
                    catch (InvalidOperationException ex)
                    {
                        LogInvalidOperationDuringCancel(_logger, adapter.BrokerName, orderId, ex);
                    }
                    catch (TimeoutException ex)
                    {
                        LogTimeoutDuringCancel(_logger, adapter.BrokerName, orderId, ex);
                    }
                    catch (Exception ex) when (!ex.IsFatal())
                    {
                        LogCancelAttemptFailed(_logger, adapter.BrokerName, orderId, ex);
                    }
                }
                
                LogFailedCancelViaAnyAdapter(_logger, orderId, reason, null);
                return false;
            }

            // Use specific broker adapter
            if (!_brokerAdapters.TryGetValue(brokerName, out var brokerAdapter))
            {
                LogUnknownBrokerAdapter(_logger, brokerName, orderId, null);
                return false;
            }

            try
            {
                var result = await brokerAdapter.CancelOrderAsync(orderId, cancellationToken).ConfigureAwait(false);
                
                if (result)
                {
                    LogOrderCancelledSuccessfully(_logger, orderId, brokerName, reason, null);
                }
                else
                {
                    LogOrderCancelFailed(_logger, orderId, brokerName, reason, null);
                }
                
                return result;
            }
            catch (ArgumentException ex)
            {
                LogInvalidArgumentCancelling(_logger, orderId, brokerName, reason, ex);
                return false;
            }
            catch (InvalidOperationException ex)
            {
                LogInvalidOperationCancelling(_logger, orderId, brokerName, reason, ex);
                return false;
            }
            catch (Exception ex) when (!ex.IsFatal())
            {
                LogExceptionCancelling(_logger, orderId, brokerName, reason, ex);
                return false;
            }
        }

        /// <summary>
        /// Cancel multiple orders in batch
        /// </summary>
        public async Task<Dictionary<string, bool>> CancelOrdersAsync(IEnumerable<string> orderIds, string? brokerName = null, string? reason = null, CancellationToken cancellationToken = default)
        {
            var results = new Dictionary<string, bool>();
            
            foreach (var orderId in orderIds)
            {
                results[orderId] = await CancelOrderAsync(orderId, brokerName, reason, cancellationToken).ConfigureAwait(false);
            }
            
            var successCount = results.Values.Count(r => r);
            LogBatchCancelCompleted(_logger, successCount, results.Count, null);
            
            return results;
        }

        /// <summary>
        /// Get available broker adapters
        /// </summary>
        public IEnumerable<string> GetAvailableBrokers() => _brokerAdapters.Keys;

        /// <summary>
        /// Check if a broker adapter is available
        /// </summary>
        public bool IsBrokerAvailable(string brokerName) => _brokerAdapters.ContainsKey(brokerName);
    }

    /// <summary>
    /// Interface for broker adapters to implement consistent cancellation
    /// </summary>
    public interface IBrokerAdapter
    {
        string BrokerName { get; }
        Task<bool> CancelOrderAsync(string orderId, CancellationToken cancellationToken = default);
    }
}

internal static class ExceptionExtensions
{
    /// <summary>
    /// Determines if an exception is fatal and should be rethrown
    /// </summary>
    public static bool IsFatal(this Exception ex)
    {
        return ex is OutOfMemoryException ||
               ex is StackOverflowException ||
               ex is AccessViolationException ||
               ex is AppDomainUnloadedException ||
               ex is ThreadAbortException ||
               ex is System.Security.SecurityException;
    }
}