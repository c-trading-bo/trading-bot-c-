using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BotCore.Services
{
    /// <summary>
    /// Production-ready trading readiness tracker
    /// Centralized tracking of BarsSeen and other readiness metrics
    /// Fixes the historical bar seeding pipeline issue
    /// </summary>
    public class TradingReadinessTracker : ITradingReadinessTracker
    {
        private readonly ILogger<TradingReadinessTracker> _logger;
        private readonly TradingReadinessConfiguration _config;
        private readonly object _lockObject = new();
        private volatile TradingReadinessContext _context;

        public TradingReadinessTracker(
            ILogger<TradingReadinessTracker> logger,
            IOptions<TradingReadinessConfiguration> config)
        {
            _logger = logger;
            _config = config.Value;
            _context = new TradingReadinessContext
            {
                State = TradingReadinessState.Initializing,
                LastMarketDataUpdate = DateTime.UtcNow
            };
        }

        public void IncrementBarsSeen(int count = 1)
        {
            lock (_lockObject)
            {
                _context.TotalBarsSeen += count;
                UpdateState();
                
                _logger.LogDebug("[READINESS-TRACKER] BarsSeen incremented by {Count} -> {Total}/{Min}", 
                    count, _context.TotalBarsSeen, GetMinBarsSeen());
            }
        }

        public void IncrementSeededBars(int count = 1)
        {
            lock (_lockObject)
            {
                _context.SeededBars += count;
                UpdateState();
                
                _logger.LogDebug("[READINESS-TRACKER] SeededBars incremented by {Count} -> {Total}/{Min}", 
                    count, _context.SeededBars, GetMinSeededBars());
            }
        }

        public void IncrementLiveTicks(int count = 1)
        {
            lock (_lockObject)
            {
                _context.LiveTicks += count;
                _context.LastMarketDataUpdate = DateTime.UtcNow;
                UpdateState();
                
                _logger.LogDebug("[READINESS-TRACKER] LiveTicks incremented by {Count} -> {Total}/{Min}", 
                    count, _context.LiveTicks, GetMinLiveTicks());
            }
        }

        public void UpdateLastMarketDataTimestamp()
        {
            lock (_lockObject)
            {
                _context.LastMarketDataUpdate = DateTime.UtcNow;
                UpdateState();
            }
        }

        public TradingReadinessContext GetReadinessContext()
        {
            lock (_lockObject)
            {
                // Return a copy to prevent external modifications
                return new TradingReadinessContext
                {
                    TotalBarsSeen = _context.TotalBarsSeen,
                    SeededBars = _context.SeededBars,
                    LiveTicks = _context.LiveTicks,
                    LastMarketDataUpdate = _context.LastMarketDataUpdate,
                    HubsConnected = _context.HubsConnected,
                    CanTrade = _context.CanTrade,
                    ContractId = _context.ContractId,
                    State = _context.State
                };
            }
        }

        public async Task<ReadinessValidationResult> ValidateReadinessAsync()
        {
            await Task.CompletedTask.ConfigureAwait(false); // For async interface compatibility

            lock (_lockObject)
            {
                var minBarsSeen = GetMinBarsSeen();
                var minSeededBars = GetMinSeededBars();
                var minLiveTicks = GetMinLiveTicks();

                var isReady = _context.TotalBarsSeen >= minBarsSeen &&
                              _context.SeededBars >= minSeededBars &&
                              _context.LiveTicks >= minLiveTicks &&
                              _context.TimeSinceLastData.TotalSeconds <= _config.MarketDataTimeoutSeconds;

                var readinessScore = CalculateReadinessScore();

                var result = new ReadinessValidationResult
                {
                    IsReady = isReady,
                    State = _context.State,
                    ReadinessScore = readinessScore,
                    Reason = GenerateReadinessReason(minBarsSeen, minSeededBars, minLiveTicks),
                    Recommendations = GenerateRecommendations()
                };

                return result;
            }
        }

        public void Reset()
        {
            lock (_lockObject)
            {
                _context = new TradingReadinessContext
                {
                    State = TradingReadinessState.Initializing,
                    LastMarketDataUpdate = DateTime.UtcNow
                };
                
                _logger.LogInformation("[READINESS-TRACKER] Readiness state reset");
            }
        }

        public void SetSystemReady()
        {
            lock (_lockObject)
            {
                _context.State = TradingReadinessState.FullyReady;
                _context.CanTrade = true;
                
                _logger.LogInformation("[READINESS] System marked ready");
                
                var readyData = new
                {
                    timestamp = DateTime.UtcNow,
                    component = "readiness_tracker",
                    operation = "set_system_ready",
                    bars_seen = _context.TotalBarsSeen,
                    seeded_bars = _context.SeededBars,
                    live_ticks = _context.LiveTicks,
                    can_trade = _context.CanTrade,
                    state = _context.State.ToString()
                };

                _logger.LogInformation("SYSTEM_READY: {ReadyData}", 
                    System.Text.Json.JsonSerializer.Serialize(readyData));
            }
        }

        private void UpdateState()
        {
            var minBarsSeen = GetMinBarsSeen();
            var minSeededBars = GetMinSeededBars();
            var minLiveTicks = GetMinLiveTicks();

            var oldState = _context.State;

            if (_context.TotalBarsSeen >= minBarsSeen && 
                _context.SeededBars >= minSeededBars && 
                _context.LiveTicks >= minLiveTicks)
            {
                _context.State = TradingReadinessState.FullyReady;
                _context.CanTrade = true;
            }
            else if (_context.LiveTicks > 0)
            {
                _context.State = TradingReadinessState.LiveTickReceived;
                _context.CanTrade;
            }
            else if (_context.SeededBars > 0)
            {
                _context.State = TradingReadinessState.Seeded;
                _context.CanTrade;
            }
            else
            {
                _context.State = TradingReadinessState.Initializing;
                _context.CanTrade;
            }

            // Check for data timeout
            if (_context.TimeSinceLastData.TotalSeconds > _config.MarketDataTimeoutSeconds)
            {
                _context.State = TradingReadinessState.Degraded;
                _context.CanTrade;
            }

            if (oldState != _context.State)
            {
                _logger.LogInformation("[READINESS-TRACKER] State changed: {OldState} -> {NewState} | CanTrade: {CanTrade}", 
                    oldState, _context.State, _context.CanTrade);
            }
        }

        private double CalculateReadinessScore()
        {
            var minBarsSeen = GetMinBarsSeen();
            var minSeededBars = GetMinSeededBars();
            var minLiveTicks = GetMinLiveTicks();

            var barsScore = Math.Min(1.0, (double)_context.TotalBarsSeen / minBarsSeen);
            var seededScore = Math.Min(1.0, (double)_context.SeededBars / minSeededBars);
            var ticksScore = Math.Min(1.0, (double)_context.LiveTicks / minLiveTicks);
            var timeScore = _context.TimeSinceLastData.TotalSeconds <= _config.MarketDataTimeoutSeconds ? 1.0 : 0.0;

            return (barsScore + seededScore + ticksScore + timeScore) / 4.0;
        }

        private string GenerateReadinessReason(int minBarsSeen, int minSeededBars, int minLiveTicks)
        {
            if (_context.State == TradingReadinessState.FullyReady)
                return "System fully ready for trading";

            var reasons = new List<string>();

            if (_context.TotalBarsSeen < minBarsSeen)
                reasons.Add($"BarsSeen {_context.TotalBarsSeen}/{minBarsSeen}");

            if (_context.SeededBars < minSeededBars)
                reasons.Add($"SeededBars {_context.SeededBars}/{minSeededBars}");

            if (_context.LiveTicks < minLiveTicks)
                reasons.Add($"LiveTicks {_context.LiveTicks}/{minLiveTicks}");

            if (_context.TimeSinceLastData.TotalSeconds > _config.MarketDataTimeoutSeconds)
                reasons.Add($"DataTimeout {_context.TimeSinceLastData.TotalSeconds:F0}s/{_config.MarketDataTimeoutSeconds}s");

            return string.Join(", ", reasons);
        }

        private string[] GenerateRecommendations()
        {
            var recommendations = new List<string>();

            if (_context.State == TradingReadinessState.Initializing)
                recommendations.Add("Wait for historical data seeding to complete");

            if (_context.SeededBars < GetMinSeededBars())
                recommendations.Add("Enable historical data seeding");

            if (_context.LiveTicks == 0)
                recommendations.Add("Verify market data connections");

            if (_context.TotalBarsSeen < GetMinBarsSeen())
                recommendations.Add("Wait for more bar data to accumulate");

            return recommendations.ToArray();
        }

        private int GetMinBarsSeen()
        {
            var env = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT");
            return env == "Development" ? _config.Environment.Dev.MinBarsSeen : _config.Environment.Production.MinBarsSeen;
        }

        private int GetMinSeededBars()
        {
            var env = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT");
            return env == "Development" ? _config.Environment.Dev.MinSeededBars : _config.Environment.Production.MinSeededBars;
        }

        private int GetMinLiveTicks()
        {
            var env = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT");
            return env == "Development" ? _config.Environment.Dev.MinLiveTicks : _config.Environment.Production.MinLiveTicks;
        }
    }
}