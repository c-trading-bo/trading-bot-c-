using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Models;
using TopstepX.Bot.Core.Services;
using Zones;

namespace BotCore.Services
{
    /// <summary>
    /// Zone Break Monitoring Service - Event-driven implementation
    /// 
    /// Monitors real-time price updates to detect when supply/demand zones are broken.
    /// Publishes zone break events that trigger position management actions:
    /// - Long position + demand zone break → Early exit warning
    /// - Short position + supply zone break → Early exit warning
    /// - Strong zone break → Aggressive entry signal boost
    /// 
    /// Converted to event-driven IHostedService with Timer (Phase 5 refactoring)
    /// </summary>
    public sealed class ZoneBreakMonitoringService : IHostedService, IDisposable
    {
        private readonly ILogger<ZoneBreakMonitoringService> _logger;
        private readonly IServiceProvider _serviceProvider;
        private readonly ConcurrentDictionary<string, ZoneBreakState> _zoneStates = new();
        private readonly ConcurrentQueue<ZoneBreakEvent> _breakEvents = new();
        private Timer? _monitoringTimer;
        private bool _disposed;
        
        // Monitoring interval - check for zone breaks every 2 seconds
        private const int MonitoringIntervalSeconds = 2;
        
        // Zone break thresholds
        private const decimal StrongBreakStrengthThreshold = 0.75m; // Strong zone must have strength > 0.75
        private const decimal WeakBreakStrengthThreshold = 0.30m;   // Weak zone has strength < 0.30
        private const decimal BreakConfirmationTicks = 2; // Price must be X ticks beyond zone
        private const decimal EsTickSize = 0.25m; // ES/MES tick size for price precision
        
        // Break severity thresholds
        private const decimal MediumStrengthThreshold = 0.5m; // Medium severity threshold
        private const int MinTouchesForCritical = 3; // Minimum touches for critical severity
        private const int MinTouchesForHigh = 2; // Minimum touches for high severity
        
        public ZoneBreakMonitoringService(
            ILogger<ZoneBreakMonitoringService> logger,
            IServiceProvider serviceProvider)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
        }
        
        public Task StartAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🔍 [ZONE-BREAK] Zone Break Monitoring Service starting (event-driven mode)...");
            
            // Start timer for zone break monitoring
            var interval = TimeSpan.FromSeconds(MonitoringIntervalSeconds);
            _monitoringTimer = new Timer(MonitorZoneBreaksCallback, null, TimeSpan.Zero, interval);
            
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🛑 [ZONE-BREAK] Zone Break Monitoring Service stopping");
            _monitoringTimer?.Change(Timeout.Infinite, 0);
            return Task.CompletedTask;
        }

        /// <summary>
        /// Timer callback for zone break monitoring - uses fire-and-forget pattern
        /// </summary>
        private void MonitorZoneBreaksCallback(object? state)
        {
            if (_disposed) return;

            // Fire-and-forget pattern for async operation in timer callback
            _ = Task.Run(async () =>
            {
                try
                {
                    await MonitorZoneBreaksAsync(CancellationToken.None).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // Expected during shutdown
                }
                catch (InvalidOperationException ex)
                {
                    _logger.LogError(ex, "❌ [ZONE-BREAK] Invalid operation in zone break monitoring");
                }
                catch (ArgumentException ex)
                {
                    _logger.LogError(ex, "❌ [ZONE-BREAK] Invalid argument in zone break monitoring");
                }
            });
        }
        
        private async Task MonitorZoneBreaksAsync(CancellationToken cancellationToken)
        {
            var zoneService = _serviceProvider.GetService<IZoneService>();
            var positionTracker = _serviceProvider.GetService<PositionTrackingSystem>();
            
            if (zoneService == null || positionTracker == null)
            {
                // Services not available - skip this cycle
                return;
            }
            
            // Get all open positions to monitor
            var positions = positionTracker.AllPositions;
            if (positions == null || positions.Count == 0)
            {
                // No positions to monitor
                return;
            }
            
            foreach (var position in positions.Values)
            {
                try
                {
                    await CheckPositionForZoneBreaksAsync(position, zoneService, cancellationToken).ConfigureAwait(false);
                }
                catch (InvalidOperationException ex)
                {
                    _logger.LogWarning(ex, "⚠️ [ZONE-BREAK] Invalid operation checking zone breaks for position {Symbol}", position.Symbol);
                }
                catch (ArgumentException ex)
                {
                    _logger.LogWarning(ex, "⚠️ [ZONE-BREAK] Invalid argument checking zone breaks for position {Symbol}", position.Symbol);
                }
            }
            
            // Process any queued zone break events
            await ProcessZoneBreakEventsAsync(cancellationToken).ConfigureAwait(false);
        }
        
        private Task CheckPositionForZoneBreaksAsync(
            BotCore.Models.Position position,
            IZoneService zoneService,
            CancellationToken cancellationToken)
        {
            var snapshot = zoneService.GetSnapshot(position.Symbol);
            if (snapshot == null)
            {
                return Task.CompletedTask;
            }
            
            var currentPrice = CalculateCurrentPrice(position);
            if (currentPrice <= 0)
            {
                return Task.CompletedTask;
            }
            
            var stateKey = $"{position.Symbol}_{(position.NetQuantity > 0 ? "LONG" : "SHORT")}";
            var state = _zoneStates.GetOrAdd(stateKey, _ => new ZoneBreakState
            {
                Symbol = position.Symbol,
                IsLong = position.NetQuantity > 0,
                LastCheckedPrice = currentPrice,
                LastCheckedTime = DateTime.UtcNow
            });
            
            // Check nearest demand and supply zones for breaks
            if (snapshot.NearestDemand != null)
            {
                CheckZoneForBreak(snapshot.NearestDemand, currentPrice, position, state);
            }
            
            if (snapshot.NearestSupply != null)
            {
                CheckZoneForBreak(snapshot.NearestSupply, currentPrice, position, state);
            }
            
            // Update state
            state.LastCheckedPrice = currentPrice;
            state.LastCheckedTime = DateTime.UtcNow;
            
            return Task.CompletedTask;
        }
        
        private void CheckZoneForBreak(
            Zone zone,
            decimal currentPrice,
            BotCore.Models.Position position,
            ZoneBreakState state)
        {
            var isLong = position.NetQuantity > 0;
            var zoneKey = $"{position.Symbol}_{zone.PriceLow}_{zone.PriceHigh}";
            
            // Check if this zone was already marked as broken
            if (state.BrokenZones.Contains(zoneKey))
            {
                return;
            }
            
            // Determine if zone is supply or demand based on position
            bool isSupplyZone = zone.PriceHigh < position.AveragePrice; // Zone below entry = support/demand
            bool isDemandZone = zone.PriceLow > position.AveragePrice; // Zone above entry = resistance/supply
            
            // For long positions, monitor demand zones breaking (support breaking = bad)
            if (isLong && isSupplyZone)
            {
                // Check if price broke below this demand/support zone
                if (currentPrice < zone.PriceLow - (BreakConfirmationTicks * EsTickSize))
                {
                    // Demand zone broken!
                    var breakEvent = new ZoneBreakEvent
                    {
                        Symbol = position.Symbol,
                        ZoneKey = zoneKey,
                        ZoneLow = zone.PriceLow,
                        ZoneHigh = zone.PriceHigh,
                        ZoneStrength = zone.Pressure,
                        BreakPrice = currentPrice,
                        BreakTime = DateTime.UtcNow,
                        BreakType = zone.Pressure > StrongBreakStrengthThreshold ? ZoneBreakType.StrongDemandBreak : ZoneBreakType.WeakDemandBreak,
                        PositionType = "LONG",
                        Severity = CalculateBreakSeverity(zone.Pressure, zone.TouchCount)
                    };
                    
                    _breakEvents.Enqueue(breakEvent);
                    state.BrokenZones.Add(zoneKey);
                    
                    _logger.LogWarning("⚠️ [ZONE-BREAK] {Type} for LONG {Symbol} - Zone [{Lo}-{Hi}] Strength={Strength:F2} - Price broke to {Price}",
                        breakEvent.BreakType, position.Symbol, zone.PriceLow, zone.PriceHigh, zone.Pressure, currentPrice);
                }
            }
            
            // For short positions, monitor supply zones breaking (resistance breaking = bad)
            else if (!isLong && isDemandZone && currentPrice > zone.PriceHigh + (BreakConfirmationTicks * EsTickSize))
            {
                // Supply zone broken!
                var breakEvent = new ZoneBreakEvent
                {
                    Symbol = position.Symbol,
                    ZoneKey = zoneKey,
                    ZoneLow = zone.PriceLow,
                    ZoneHigh = zone.PriceHigh,
                    ZoneStrength = zone.Pressure,
                    BreakPrice = currentPrice,
                    BreakTime = DateTime.UtcNow,
                    BreakType = zone.Pressure > StrongBreakStrengthThreshold ? ZoneBreakType.StrongSupplyBreak : ZoneBreakType.WeakSupplyBreak,
                    PositionType = "SHORT",
                    Severity = CalculateBreakSeverity(zone.Pressure, zone.TouchCount)
                };
                
                _breakEvents.Enqueue(breakEvent);
                state.BrokenZones.Add(zoneKey);
                
                _logger.LogWarning("⚠️ [ZONE-BREAK] {Type} for SHORT {Symbol} - Zone [{Lo}-{Hi}] Strength={Strength:F2} - Price broke to {Price}",
                    breakEvent.BreakType, position.Symbol, zone.PriceLow, zone.PriceHigh, zone.Pressure, currentPrice);
            }
            
            // Also check for strong zone breaks in the favorable direction (entry signals)
            if (isLong && isDemandZone && zone.Pressure > StrongBreakStrengthThreshold)
            {
                // Strong supply zone breaking upward = bullish signal
                if (currentPrice > zone.PriceHigh + (BreakConfirmationTicks * EsTickSize))
                {
                    var breakEvent = new ZoneBreakEvent
                    {
                        Symbol = position.Symbol,
                        ZoneKey = zoneKey,
                        ZoneLow = zone.PriceLow,
                        ZoneHigh = zone.PriceHigh,
                        ZoneStrength = zone.Pressure,
                        BreakPrice = currentPrice,
                        BreakTime = DateTime.UtcNow,
                        BreakType = ZoneBreakType.StrongSupplyBreakBullish,
                        PositionType = "LONG",
                        Severity = "POSITIVE"
                    };
                    
                    _breakEvents.Enqueue(breakEvent);
                    state.BrokenZones.Add(zoneKey);
                    
                    _logger.LogInformation("✅ [ZONE-BREAK] Bullish breakout for LONG {Symbol} - Strong supply broken at {Price}", 
                        position.Symbol, currentPrice);
                }
            }
            else if (!isLong && isSupplyZone && zone.Pressure > StrongBreakStrengthThreshold && currentPrice < zone.PriceLow - (BreakConfirmationTicks * EsTickSize))
            {
                // Strong demand zone breaking downward = bearish signal
                var breakEvent = new ZoneBreakEvent
                {
                    Symbol = position.Symbol,
                    ZoneKey = zoneKey,
                    ZoneLow = zone.PriceLow,
                    ZoneHigh = zone.PriceHigh,
                    ZoneStrength = zone.Pressure,
                    BreakPrice = currentPrice,
                    BreakTime = DateTime.UtcNow,
                    BreakType = ZoneBreakType.StrongDemandBreakBearish,
                    PositionType = "SHORT",
                    Severity = "POSITIVE"
                };
                
                _breakEvents.Enqueue(breakEvent);
                state.BrokenZones.Add(zoneKey);
                
                _logger.LogInformation("✅ [ZONE-BREAK] Bearish breakout for SHORT {Symbol} - Strong demand broken at {Price}",
                    position.Symbol, currentPrice);
            }
        }
        
        private static string CalculateBreakSeverity(decimal pressure, int touchCount)
        {
            // Higher pressure + more touches = more severe break
            if (pressure > StrongBreakStrengthThreshold && touchCount >= MinTouchesForCritical)
            {
                return "CRITICAL";
            }
            else if (pressure > MediumStrengthThreshold && touchCount >= MinTouchesForHigh)
            {
                return "HIGH";
            }
            else if (pressure > WeakBreakStrengthThreshold)
            {
                return "MEDIUM";
            }
            else
            {
                return "LOW";
            }
        }
        
        private async Task ProcessZoneBreakEventsAsync(CancellationToken cancellationToken)
        {
            var positionMgmt = _serviceProvider.GetService<UnifiedPositionManagementService>();
            if (positionMgmt == null)
            {
                return;
            }
            
            while (_breakEvents.TryDequeue(out var breakEvent))
            {
                try
                {
                    // Notify position management service of zone break
                    await positionMgmt.OnZoneBreak(breakEvent).ConfigureAwait(false);
                    
                    _logger.LogInformation("📢 [ZONE-BREAK] Notified position management of {Type} event for {Symbol}",
                        breakEvent.BreakType, breakEvent.Symbol);
                }
                catch (InvalidOperationException ex)
                {
                    _logger.LogError(ex, "❌ [ZONE-BREAK] Invalid operation processing zone break event for {Symbol}", breakEvent.Symbol);
                }
                catch (ArgumentException ex)
                {
                    _logger.LogError(ex, "❌ [ZONE-BREAK] Invalid argument processing zone break event for {Symbol}", breakEvent.Symbol);
                }
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
        }
        
        private static decimal CalculateCurrentPrice(BotCore.Models.Position position)
        {
            // Calculate current market price from unrealized P&L
            // Formula: currentPrice = (unrealizedPnL / netQuantity) + avgPrice
            if (position.NetQuantity == 0)
            {
                return 0;
            }
            
            var estimatedPrice = (position.UnrealizedPnL / position.NetQuantity) + position.AveragePrice;
            return estimatedPrice;
        }
        
        /// <summary>
        /// Get all zone break events for a symbol (for testing/monitoring)
        /// </summary>
        public IReadOnlyList<ZoneBreakEvent> GetRecentBreaks(string symbol, TimeSpan lookback)
        {
            var cutoff = DateTime.UtcNow - lookback;
            var events = new List<ZoneBreakEvent>();
            
            foreach (var breakEvent in _breakEvents)
            {
                if (breakEvent.Symbol == symbol && breakEvent.BreakTime >= cutoff)
                {
                    events.Add(breakEvent);
                }
            }
            
            return events;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _monitoringTimer?.Dispose();
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }
    
    /// <summary>
    /// Zone break state for tracking which zones have been broken
    /// </summary>
    internal sealed class ZoneBreakState
    {
        public string Symbol { get; set; } = string.Empty;
        public bool IsLong { get; set; }
        public decimal LastCheckedPrice { get; set; }
        public DateTime LastCheckedTime { get; set; }
        public HashSet<string> BrokenZones { get; set; } = new();
    }
    
    /// <summary>
    /// Zone break event - published when a zone is broken
    /// </summary>
    public sealed class ZoneBreakEvent
    {
        public string Symbol { get; set; } = string.Empty;
        public string ZoneKey { get; set; } = string.Empty;
        public decimal ZoneLow { get; set; }
        public decimal ZoneHigh { get; set; }
        public decimal ZoneStrength { get; set; }
        public decimal BreakPrice { get; set; }
        public DateTime BreakTime { get; set; }
        public ZoneBreakType BreakType { get; set; }
        public string PositionType { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// Types of zone breaks
    /// </summary>
    public enum ZoneBreakType
    {
        StrongDemandBreak,         // Strong support broken (bearish for longs)
        WeakDemandBreak,           // Weak support broken (mildly bearish for longs)
        StrongSupplyBreak,         // Strong resistance broken (bearish for shorts)
        WeakSupplyBreak,           // Weak resistance broken (mildly bearish for shorts)
        StrongSupplyBreakBullish,  // Strong resistance broken upward (bullish)
        StrongDemandBreakBearish   // Strong support broken downward (bearish)
    }
}
