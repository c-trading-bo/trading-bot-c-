using Microsoft.Extensions.Logging;
using System;
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

        public OrderManager(ILogger<OrderManager> logger, IEnumerable<IBrokerAdapter> brokerAdapters)
        {
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
                        _logger.LogDebug("[OrderManager] Attempting cancel via {Broker} for order {OrderId}", 
                            adapter.BrokerName, orderId);
                        
                        var result = await adapter.CancelOrderAsync(orderId, cancellationToken).ConfigureAwait(false);
                        if (result)
                        {
                            _logger.LogInformation("[OrderManager] ✅ Order {OrderId} cancelled successfully via {Broker}: {Reason}", 
                                orderId, adapter.BrokerName, reason);
                            return true;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, "[OrderManager] Cancel attempt failed via {Broker} for order {OrderId}", 
                            adapter.BrokerName, orderId);
                    }
                }
                
                _logger.LogWarning("[OrderManager] ❌ Failed to cancel order {OrderId} via any broker adapter: {Reason}", 
                    orderId, reason);
                return false;
            }

            // Use specific broker adapter
            if (!_brokerAdapters.TryGetValue(brokerName, out var brokerAdapter))
            {
                _logger.LogError("[OrderManager] ❌ Unknown broker adapter: {Broker} for order {OrderId}", 
                    brokerName, orderId);
                return false;
            }

            try
            {
                var result = await brokerAdapter.CancelOrderAsync(orderId, cancellationToken).ConfigureAwait(false);
                
                if (result)
                {
                    _logger.LogInformation("[OrderManager] ✅ Order {OrderId} cancelled successfully via {Broker}: {Reason}", 
                        orderId, brokerName, reason);
                }
                else
                {
                    _logger.LogWarning("[OrderManager] ❌ Failed to cancel order {OrderId} via {Broker}: {Reason}", 
                        orderId, brokerName, reason);
                }
                
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[OrderManager] ❌ Exception cancelling order {OrderId} via {Broker}: {Reason}", 
                    orderId, brokerName, reason);
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
            _logger.LogInformation("[OrderManager] Batch cancel completed: {Success}/{Total} orders cancelled", 
                successCount, results.Count);
            
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