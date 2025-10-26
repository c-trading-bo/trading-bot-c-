using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using BotCore.Configuration;
using BotCore.Models;
using TopstepX.Bot.Core.Services;

namespace BotCore.Services
{
    /// <summary>
    /// Layer 2: Stuck Position Monitor
    /// Runs every 30 seconds identifying positions that need emergency intervention
    /// Detects stuck exits, aged positions, and runaway losses
    /// </summary>
    public sealed class StuckPositionMonitor : BackgroundService
    {
        private readonly ILogger<StuckPositionMonitor> _logger;
        private readonly StuckPositionRecoveryConfiguration _config;
        private readonly PositionTrackingSystem _positionTracking;
        private readonly EmergencyExitExecutor _emergencyExecutor;

        // Track positions currently under recovery
        private readonly ConcurrentDictionary<string, RecoveryTrackingInfo> _positionsUnderRecovery = new();

        public class RecoveryTrackingInfo
        {
            public DateTime RecoveryStartTime { get; set; }
            public RecoveryLevel CurrentLevel { get; set; }
            public int AttemptsMade { get; set; }
        }

        public StuckPositionMonitor(
            ILogger<StuckPositionMonitor> logger,
            IOptions<StuckPositionRecoveryConfiguration> config,
            PositionTrackingSystem positionTracking,
            EmergencyExitExecutor emergencyExecutor)
        {
            _logger = logger;
            _config = config?.Value ?? throw new ArgumentNullException(nameof(config));
            _positionTracking = positionTracking ?? throw new ArgumentNullException(nameof(positionTracking));
            _emergencyExecutor = emergencyExecutor ?? throw new ArgumentNullException(nameof(emergencyExecutor));

            _config.Validate();
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            if (!_config.Enabled)
            {
                _logger.LogInformation("👁️ [MONITOR] Stuck Position Monitor disabled via configuration");
                // FIXED: Wait indefinitely instead of returning to prevent shutdown signal
                await Task.Delay(Timeout.Infinite, stoppingToken).ConfigureAwait(false);
                return;
            }

            _logger.LogInformation("👁️ [MONITOR] Stuck Position Monitor started - checking every {Interval}s",
                _config.MonitorCheckIntervalSeconds);

            // Wait 30 seconds for bot initialization before first check
            await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken).ConfigureAwait(false);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await CheckForStuckPositionsAsync(stoppingToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // Expected during shutdown
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [MONITOR] Error during stuck position check");
                }

                // Wait for next check interval
                await Task.Delay(
                    TimeSpan.FromSeconds(_config.MonitorCheckIntervalSeconds),
                    stoppingToken
                ).ConfigureAwait(false);
            }

            _logger.LogInformation("👁️ [MONITOR] Stuck Position Monitor stopped");
        }

        /// <summary>
        /// Check all open positions for stuck conditions
        /// </summary>
        private async Task CheckForStuckPositionsAsync(CancellationToken cancellationToken)
        {
            _logger.LogDebug("👁️ [MONITOR] Starting stuck position check...");

            try
            {
                // Get all open positions from position tracking system
                var openPositions = _positionTracking.GetAllPositions()
                    .Where(p => p.NetQuantity != 0)
                    .ToList();

                if (openPositions.Count == 0)
                {
                    _logger.LogDebug("👁️ [MONITOR] No open positions to check");
                    return;
                }

                _logger.LogDebug("👁️ [MONITOR] Checking {Count} open positions", openPositions.Count);

                var healthyCount = 0;
                var stuckCount = 0;

                foreach (var position in openPositions)
                {
                    // Skip if already under recovery
                    if (_positionsUnderRecovery.ContainsKey(position.Symbol))
                    {
                        _logger.LogDebug("👁️ [MONITOR] {Symbol} already under recovery - skipping", position.Symbol);
                        continue;
                    }

                    // Classify the position
                    var classification = ClassifyPosition(position);

                    if (classification == StuckPositionClassification.Healthy)
                    {
                        healthyCount++;
                        continue;
                    }

                    // Position needs recovery
                    stuckCount++;

                    _logger.LogWarning(
                        "⚠️ [MONITOR] Stuck position detected: {Symbol} - Classification: {Classification}",
                        position.Symbol, classification);

                    // Create alert
                    var alert = CreateStuckPositionAlert(position, classification);

                    // Track that we've initiated recovery
                    _positionsUnderRecovery[position.Symbol] = new RecoveryTrackingInfo
                    {
                        RecoveryStartTime = DateTime.UtcNow,
                        CurrentLevel = RecoveryLevel.None,
                        AttemptsMade = 0
                    };

                    // Hand off to emergency executor
                    await _emergencyExecutor.HandleStuckPositionAsync(alert, cancellationToken).ConfigureAwait(false);
                }

                if (stuckCount > 0)
                {
                    _logger.LogWarning(
                        "⚠️ [MONITOR] Check complete: {StuckCount} stuck positions detected, {HealthyCount} healthy",
                        stuckCount, healthyCount);
                }
                else
                {
                    _logger.LogInformation(
                        "✅ [MONITOR] Check complete: All {HealthyCount} positions healthy",
                        healthyCount);
                }

                // Clean up resolved positions from tracking
                CleanupResolvedPositions();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [MONITOR] Exception during stuck position check");
                throw;
            }
        }

        /// <summary>
        /// Classify a position as healthy or stuck
        /// </summary>
        private StuckPositionClassification ClassifyPosition(BotCore.Models.Position position)
        {
            var now = DateTime.UtcNow;

            // Check for runaway loss (highest priority)
            if (position.UnrealizedPnL < _config.RunawayLossThresholdUsd)
            {
                _logger.LogWarning(
                    "🚨 [MONITOR] {Symbol} - Runaway loss detected: P&L ${PnL:F2} below threshold ${Threshold:F2}",
                    position.Symbol, position.UnrealizedPnL, _config.RunawayLossThresholdUsd);
                return StuckPositionClassification.RunawayLoss;
            }

            // Check for aged position
            var positionAge = now - position.LastUpdate;
            var maxAge = TimeSpan.FromHours(_config.MaxPositionAgeHours);

            if (positionAge > maxAge)
            {
                _logger.LogWarning(
                    "⏰ [MONITOR] {Symbol} - Position aged out: Age {Age:F1} hours exceeds maximum {Max} hours",
                    position.Symbol, positionAge.TotalHours, _config.MaxPositionAgeHours);
                return StuckPositionClassification.AgedOut;
            }

            // Check for stuck exit
            // This requires checking if there are failed exit orders
            // For now, we'll use a simplified check based on position tracking data
            var hasFailedExits = CheckForFailedExitOrders(position);
            if (hasFailedExits)
            {
                _logger.LogWarning(
                    "🔒 [MONITOR] {Symbol} - Stuck exit detected: Exit order failed more than {Minutes} minutes ago",
                    position.Symbol, _config.StuckExitMinutesThreshold);
                return StuckPositionClassification.StuckExit;
            }

            // Position is healthy
            return StuckPositionClassification.Healthy;
        }

        /// <summary>
        /// Check if position has failed exit orders
        /// </summary>
        private bool CheckForFailedExitOrders(BotCore.Models.Position position)
        {
            // This would check the order history for the position
            // Looking for exit orders that are:
            // 1. Submitted more than StuckExitMinutesThreshold minutes ago
            // 2. Failed or rejected
            // 3. No new exit attempts in last MinutesSinceLastExitAttempt minutes

            // For now, return false as we need order history access
            // This will be implemented when integrating with OrderExecutionService
            return false;
        }

        /// <summary>
        /// Create stuck position alert from position data
        /// </summary>
        private StuckPositionAlert CreateStuckPositionAlert(
            BotCore.Models.Position position,
            StuckPositionClassification classification)
        {
            var alert = new StuckPositionAlert
            {
                PositionId = $"{position.Symbol}-{DateTime.UtcNow:yyyyMMddHHmmss}",
                Symbol = position.Symbol,
                Quantity = Math.Abs(position.NetQuantity),
                EntryPrice = position.AveragePrice,
                IsLong = position.NetQuantity > 0,
                EntryTimestamp = position.LastUpdate,
                CurrentPrice = position.AveragePrice, // Would be updated with current market price
                UnrealizedPnL = position.UnrealizedPnL,
                Classification = classification,
                DetectionTimestamp = DateTime.UtcNow,
                Reason = GetClassificationReason(classification, position)
            };

            // Add any exit attempts from position history
            // This would be populated from order history
            alert.ExitAttempts = new List<ExitAttempt>();

            return alert;
        }

        /// <summary>
        /// Get human-readable reason for classification
        /// </summary>
        private string GetClassificationReason(
            StuckPositionClassification classification,
            BotCore.Models.Position position)
        {
            return classification switch
            {
                StuckPositionClassification.StuckExit =>
                    $"Exit order submitted more than {_config.StuckExitMinutesThreshold} minutes ago but position still open",

                StuckPositionClassification.AgedOut =>
                    $"Position held for {(DateTime.UtcNow - position.LastUpdate).TotalHours:F1} hours, exceeds maximum {_config.MaxPositionAgeHours} hours",

                StuckPositionClassification.RunawayLoss =>
                    $"Unrealized P&L ${position.UnrealizedPnL:F2} below emergency threshold ${_config.RunawayLossThresholdUsd:F2}",

                _ => "Unknown reason"
            };
        }

        /// <summary>
        /// Clean up resolved positions from tracking
        /// </summary>
        private void CleanupResolvedPositions()
        {
            var resolvedPositions = new List<string>();

            foreach (var kvp in _positionsUnderRecovery)
            {
                var symbol = kvp.Key;

                // Check if position is still open
                var position = _positionTracking.GetAllPositions()
                    .FirstOrDefault(p => p.Symbol.Equals(symbol, StringComparison.OrdinalIgnoreCase));

                if (position == null || position.NetQuantity == 0)
                {
                    // Position is closed, remove from tracking
                    resolvedPositions.Add(symbol);
                }
            }

            foreach (var symbol in resolvedPositions)
            {
                if (_positionsUnderRecovery.TryRemove(symbol, out var info))
                {
                    var recoveryTime = DateTime.UtcNow - info.RecoveryStartTime;
                    _logger.LogInformation(
                        "✅ [MONITOR] {Symbol} recovery completed - Total time: {Time:F1} seconds",
                        symbol, recoveryTime.TotalSeconds);
                }
            }
        }

        /// <summary>
        /// Get current recovery status for monitoring
        /// </summary>
        public Dictionary<string, RecoveryTrackingInfo> GetRecoveryStatus()
        {
            return new Dictionary<string, RecoveryTrackingInfo>(_positionsUnderRecovery);
        }
    }
}
