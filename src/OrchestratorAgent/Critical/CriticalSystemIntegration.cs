// ================================================================================
// CRITICAL SYSTEM INTEGRATION
// ================================================================================
// File: CriticalSystemIntegration.cs  
// Purpose: Integration layer for critical trading system components
// Author: kevinsuero072897-collab
// Date: 2025-01-09
// ================================================================================

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Critical;

namespace OrchestratorAgent.Critical
{
    /// <summary>
    /// Manages integration and lifecycle of critical trading system components
    /// </summary>
    public class CriticalSystemManager : IDisposable
    {
        private readonly ILogger<CriticalSystemManager> _logger;
        private ExecutionVerificationSystem? _executionVerificationSystem;
        private DisasterRecoverySystem? _disasterRecoverySystem;
        private CorrelationProtectionSystem? _correlationProtectionSystem;
        private HubConnection? _userHubConnection;
        private bool _isInitialized;

        public CriticalSystemManager(ILogger<CriticalSystemManager> logger)
        {
            _logger = logger;
        }

        public async Task InitializeAsync(HubConnection userHubConnection, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("[CriticalSystem] Initializing critical trading system components...");

                _userHubConnection = userHubConnection;

                // Validate required credentials first
                EnhancedCredentialManager.ValidateRequiredCredentials();
                _logger.LogInformation("[CriticalSystem] Credential validation completed");

                // Initialize Execution Verification System
                _executionVerificationSystem = new ExecutionVerificationSystem(
                    _userHubConnection, 
                    Microsoft.Extensions.Logging.LoggerFactory.Create(builder => builder.AddConsole())
                        .CreateLogger<ExecutionVerificationSystem>()
                );
                await _executionVerificationSystem.InitializeVerificationSystem().ConfigureAwait(false);
                _logger.LogInformation("[CriticalSystem] ExecutionVerificationSystem initialized");

                // Initialize Disaster Recovery System
                _disasterRecoverySystem = new DisasterRecoverySystem(
                    Microsoft.Extensions.Logging.LoggerFactory.Create(builder => builder.AddConsole())
                        .CreateLogger<DisasterRecoverySystem>()
                );
                await _disasterRecoverySystem.InitializeRecoverySystem().ConfigureAwait(false);
                _logger.LogInformation("[CriticalSystem] DisasterRecoverySystem initialized");

                // Initialize Correlation Protection System
                _correlationProtectionSystem = new CorrelationProtectionSystem(
                    Microsoft.Extensions.Logging.LoggerFactory.Create(builder => builder.AddConsole())
                        .CreateLogger<CorrelationProtectionSystem>()
                );
                await _correlationProtectionSystem.InitializeCorrelationMonitor().ConfigureAwait(false);
                _logger.LogInformation("[CriticalSystem] CorrelationProtectionSystem initialized");

                _isInitialized = true;
                _logger.LogInformation("[CriticalSystem] ✅ All critical trading system components initialized successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CriticalSystem] ❌ Failed to initialize critical trading system components");
                throw;
            }
        }

        /// <summary>
        /// Pre-validate an order before execution using all critical systems
        /// </summary>
        public async Task<bool> ValidateOrderAsync(string symbol, int quantity, string direction, decimal price)
        {
            if (!_isInitialized)
            {
                _logger.LogWarning("[CriticalSystem] Systems not initialized - rejecting order");
                return false;
            }

            try
            {
                // 1. Check correlation protection
                if (_correlationProtectionSystem != null)
                {
                    var correlationOk = await _correlationProtectionSystem.ValidateNewPosition(symbol, quantity, direction).ConfigureAwait(false);
                    if (!correlationOk)
                    {
                        _logger.LogWarning("[CriticalSystem] Order rejected by correlation protection: {Symbol} {Qty} {Direction}", 
                            symbol, quantity, direction);
                        return false;
                    }
                }

                // 2. Add other validation checks here as needed

                _logger.LogInformation("[CriticalSystem] Order validation passed: {Symbol} {Qty} {Direction} @ {Price}", 
                    symbol, quantity, direction, price);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CriticalSystem] Order validation failed for {Symbol}", symbol);
                return false;
            }
        }

        /// <summary>
        /// Register a pending order for execution verification
        /// </summary>
        public void RegisterPendingOrder(string orderId, string clientOrderId, string symbol, int quantity, decimal price, string side)
        {
            if (_executionVerificationSystem != null)
            {
                var orderRecord = new ExecutionVerificationSystem.OrderRecord
                {
                    OrderId = orderId,
                    ClientOrderId = clientOrderId,
                    SubmittedTime = DateTime.UtcNow,
                    Symbol = symbol,
                    Quantity = quantity,
                    Price = price,
                    Side = side,
                    Status = "PENDING"
                };

                _executionVerificationSystem.AddPendingOrder(orderRecord);
                _logger.LogInformation("[CriticalSystem] Registered pending order: {OrderId} {Symbol} {Side} {Qty}@{Price}", 
                    orderId, symbol, side, quantity, price);
            }
        }

        /// <summary>
        /// Verify order execution with optional slippage tolerance
        /// </summary>
        public async Task<bool> VerifyExecutionAsync(string orderId, int expectedQuantity, decimal maxSlippage = 1.0m)
        {
            if (_executionVerificationSystem != null)
            {
                return await _executionVerificationSystem.VerifyExecution(orderId, expectedQuantity, maxSlippage).ConfigureAwait(false);
            }
            return false;
        }

        /// <summary>
        /// Update position exposure for correlation tracking
        /// </summary>
        public void UpdateExposure(string symbol, decimal exposure)
        {
            _correlationProtectionSystem?.UpdateExposure(symbol, exposure);
        }

        /// <summary>
        /// Add position to disaster recovery tracking
        /// </summary>
        public void AddPosition(string symbol, int quantity, decimal entryPrice, string strategyId)
        {
            if (_disasterRecoverySystem != null)
            {
                var position = new DisasterRecoverySystem.Position
                {
                    Symbol = symbol,
                    Quantity = quantity,
                    EntryPrice = entryPrice,
                    CurrentPrice = entryPrice,
                    EntryTime = DateTime.UtcNow,
                    StrategyId = strategyId
                };

                _disasterRecoverySystem.AddPosition(position);
                _logger.LogInformation("[CriticalSystem] Added position to recovery tracking: {Symbol} {Qty}@{Price}", 
                    symbol, quantity, entryPrice);
            }
        }

        /// <summary>
        /// Get credential using enhanced credential manager
        /// </summary>
        public static string GetCredential(string key, string? defaultValue = null)
        {
            return EnhancedCredentialManager.GetCredential(key, defaultValue);
        }

        /// <summary>
        /// Try to get credential safely
        /// </summary>
        public static bool TryGetCredential(string key, out string value)
        {
            return EnhancedCredentialManager.TryGetCredential(key, out value);
        }

        public void Dispose()
        {
            try
            {
                _executionVerificationSystem?.Dispose();
                _disasterRecoverySystem?.Dispose();
                _correlationProtectionSystem?.Dispose();
                _logger.LogInformation("[CriticalSystem] Disposed all critical system components");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CriticalSystem] Error during disposal");
            }
        }
    }

    /// <summary>
    /// Enhanced application environment helper that uses critical system credential management
    /// </summary>
    public static class CriticalAppEnv
    {
        public static string Get(string key, string? def = null)
        {
            try
            {
                return CriticalSystemManager.GetCredential(key, def);
            }
            catch when (def != null)
            {
                return def;
            }
        }

        public static int Int(string key, int def)
        {
            if (CriticalSystemManager.TryGetCredential(key, out var value) && int.TryParse(value, out var result))
            {
                return result;
            }
            return def;
        }

        public static bool Flag(string key, bool defaultTrue)
        {
            if (!CriticalSystemManager.TryGetCredential(key, out var raw))
            {
                return defaultTrue;
            }

            raw = raw.Trim();
            return raw.Equals("1", StringComparison.OrdinalIgnoreCase)
                || raw.Equals("true", StringComparison.OrdinalIgnoreCase)
                || raw.Equals("yes", StringComparison.OrdinalIgnoreCase);
        }

        public static void Set(string key, string value)
        {
            Environment.SetEnvironmentVariable(key, value);
        }
    }
}