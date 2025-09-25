using System;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Configuration snapshot service for audit logging and position epoch freezing
    /// Ensures configuration values don't change mid-trade
    /// </summary>
    public class ConfigurationSnapshotService
    {
        private readonly ILogger<ConfigurationSnapshotService> _logger;
        private readonly Dictionary<string, ConfigSnapshot> _activeSnapshots = new();
        private readonly object _snapshotLock = new();

        public ConfigurationSnapshotService(ILogger<ConfigurationSnapshotService> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Create a configuration snapshot for a position epoch
        /// </summary>
        public ConfigSnapshot CreateSnapshot(string positionId, IServiceProvider serviceProvider)
        {
            if (positionId is null) throw new ArgumentNullException(nameof(positionId));
            if (serviceProvider is null) throw new ArgumentNullException(nameof(serviceProvider));
            
            lock (_snapshotLock)
            {
                var snapshot = new ConfigSnapshot
                {
                    Id = Guid.NewGuid().ToString("N")[..12], // 12-character snapshot ID
                    PositionId = positionId,
                    CreatedAt = DateTime.UtcNow
                };
                
                snapshot.ReplaceValues(CaptureConfigurationValues(serviceProvider));

                _activeSnapshots[positionId] = snapshot;

                _logger.LogInformation("📸 [CONFIG-SNAPSHOT] Created snapshot {SnapshotId} for position {PositionId}", 
                    snapshot.Id, positionId);

                LogSnapshotValues(snapshot);

                return snapshot;
            }
        }

        /// <summary>
        /// Get frozen configuration values for a position
        /// </summary>
        public ConfigSnapshot? GetSnapshot(string positionId)
        {
            lock (_snapshotLock)
            {
                return _activeSnapshots.TryGetValue(positionId, out var snapshot) ? snapshot : null;
            }
        }

        /// <summary>
        /// Release snapshot when position closes
        /// </summary>
        public void ReleaseSnapshot(string positionId)
        {
            lock (_snapshotLock)
            {
                if (_activeSnapshots.Remove(positionId, out var snapshot))
                {
                    _logger.LogInformation("🗑️ [CONFIG-SNAPSHOT] Released snapshot {SnapshotId} for position {PositionId}", 
                        snapshot.Id, positionId);
                }
            }
        }

        private Dictionary<string, object> CaptureConfigurationValues(IServiceProvider serviceProvider)
        {
            var values = new Dictionary<string, object>();

            try
            {
                // Capture ML Configuration values
                var mlConfig = serviceProvider.GetService(typeof(IMLConfigurationService)) as IMLConfigurationService;
                if (mlConfig != null)
                {
                    values["ml.ai_confidence_threshold"] = mlConfig.GetAIConfidenceThreshold();
                    values["ml.minimum_confidence"] = mlConfig.GetMinimumConfidence();
                    values["ml.position_size_multiplier"] = mlConfig.GetPositionSizeMultiplier();
                    values["ml.regime_detection_threshold"] = mlConfig.GetRegimeDetectionThreshold();
                    values["ml.stop_loss_buffer_percentage"] = mlConfig.GetStopLossBufferPercentage();
                    values["ml.reward_risk_ratio_threshold"] = mlConfig.GetRewardRiskRatioThreshold();
                }

                // Capture Risk Configuration values
                var riskConfig = serviceProvider.GetService(typeof(IRiskConfig)) as IRiskConfig;
                if (riskConfig != null)
                {
                    values["risk.max_position_size"] = riskConfig.GetMaxPositionSize();
                    values["risk.daily_loss_limit"] = riskConfig.GetDailyLossLimit();
                    values["risk.per_trade_risk"] = riskConfig.GetPerTradeRisk();
                    values["risk.max_drawdown_pct"] = riskConfig.GetMaxDrawdownPercentage();
                    values["risk.cvar_confidence"] = riskConfig.GetCvarConfidenceLevel();
                }

                // Capture Execution Configuration values
                var execConfig = serviceProvider.GetService(typeof(IExecutionGuardsConfig)) as IExecutionGuardsConfig;
                if (execConfig != null)
                {
                    values["execution.max_spread_ticks"] = execConfig.GetMaxSpreadTicks();
                    values["execution.max_latency_ms"] = execConfig.GetMaxLatencyMs();
                    values["execution.min_volume"] = execConfig.GetMinVolumeThreshold();
                    values["execution.max_imbalance"] = execConfig.GetMaxImbalanceRatio();
                }

                // Capture Bracket Configuration values
                var bracketConfig = serviceProvider.GetService(typeof(IBracketConfig)) as IBracketConfig;
                if (bracketConfig != null)
                {
                    values["bracket.default_stop_atr_multiple"] = bracketConfig.GetDefaultStopAtrMultiple();
                    values["bracket.default_target_atr_multiple"] = bracketConfig.GetDefaultTargetAtrMultiple();
                    values["bracket.trailing_stop_enabled"] = bracketConfig.EnableTrailingStop;
                    values["bracket.reduce_only_mode"] = bracketConfig.ReduceOnlyMode;
                }

            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error capturing configuration snapshot values");
                values["capture_error"] = ex.Message;
            }

            return values;
        }

        private void LogSnapshotValues(ConfigSnapshot snapshot)
        {
            _logger.LogInformation("📊 [CONFIG-SNAPSHOT] Values for {SnapshotId}:", snapshot.Id);
            
            foreach (var kvp in snapshot.Values)
            {
                _logger.LogInformation("   {Key} = {Value}", kvp.Key, kvp.Value);
            }

            // Log as structured JSON for analysis
            var json = JsonSerializer.Serialize(snapshot.Values, new JsonSerializerOptions { WriteIndented = true });
            _logger.LogInformation("📋 [CONFIG-SNAPSHOT] JSON: {Json}", json);
        }
    }

    /// <summary>
    /// Frozen configuration snapshot for a trading position
    /// </summary>
    public class ConfigSnapshot
    {
        public string Id { get; set; } = string.Empty;
        public string PositionId { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        private readonly Dictionary<string, object> _values = new();
        public IReadOnlyDictionary<string, object> Values => _values;
        
        public void ReplaceValues(IEnumerable<KeyValuePair<string, object>> values)
        {
            _values.Clear();
            if (values != null)
            {
                foreach (var kvp in values)
                    _values[kvp.Key] = kvp.Value;
            }
        }
        
        public void SetValue(string key, object value)
        {
            if (key != null) _values[key] = value;
        }

        /// <summary>
        /// Get a typed configuration value from the snapshot
        /// </summary>
        public T GetValue<T>(string key, T defaultValue = default!)
        {
            if (Values.TryGetValue(key, out var value))
            {
                try
                {
                    return (T)Convert.ChangeType(value, typeof(T));
                }
                catch
                {
                    return defaultValue;
                }
            }
            return defaultValue;
        }
    }
}