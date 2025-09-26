using System;
using System.ComponentModel.DataAnnotations;

namespace BotCore.Services
{
    /// <summary>
    /// Production-ready trading readiness configuration system
    /// Implements configurable warm-up thresholds and multi-stage validation
    /// </summary>
    public class TradingReadinessConfiguration
    {
        // Configuration constants for validation ranges
        private const int MinBarsSeenMinimum = 1;
        private const int MinBarsSeenMaximum = 100;
        private const int MinSeededBarsMinimum = 0;
        private const int MinSeededBarsMaximum = 50;
        private const int MinLiveTicksMinimum = 1;
        private const int MinLiveTicksMaximum = 10;
        private const int HistoricalDataAgeMinHours = 1;
        private const int HistoricalDataAgeMaxHours = 168; // 1 week
        private const int MarketDataTimeoutMinSeconds = 30;
        private const int MarketDataTimeoutMaxSeconds = 600;
        
        // Default values for configuration properties
        private const int DefaultMinBarsSeen = 10;
        private const int DefaultMinSeededBars = 12;
        private const int DefaultMinLiveTicks = 2;
        private const int DefaultMaxHistoricalDataAgeHours = 24;
        private const int DefaultMarketDataTimeoutSeconds = 300;

        /// <summary>
        /// Minimum bars required before trading is allowed
        /// Configurable per environment (dev vs production)
        /// </summary>
        [Range(MinBarsSeenMinimum, MinBarsSeenMaximum)]
        public int MinBarsSeen { get; set; } = DefaultMinBarsSeen;

        /// <summary>
        /// Minimum seeded bars from historical data
        /// </summary>
        [Range(MinSeededBarsMinimum, MinSeededBarsMaximum)]
        public int MinSeededBars { get; set; } = DefaultMinSeededBars;

        /// <summary>
        /// Minimum live ticks required after seeding
        /// Ensures at least some live data before trading
        /// </summary>
        [Range(MinLiveTicksMinimum, MinLiveTicksMaximum)]
        public int MinLiveTicks { get; set; } = DefaultMinLiveTicks;

        /// <summary>
        /// Maximum age for historical seeding data in hours
        /// </summary>
        [Range(HistoricalDataAgeMinHours, HistoricalDataAgeMaxHours)]
        public int MaxHistoricalDataAgeHours { get; set; } = DefaultMaxHistoricalDataAgeHours;

        /// <summary>
        /// Market data health timeout in seconds
        /// If no data received within this time, system is considered unhealthy
        /// </summary>
        [Range(MarketDataTimeoutMinSeconds, MarketDataTimeoutMaxSeconds)]
        public int MarketDataTimeoutSeconds { get; set; } = DefaultMarketDataTimeoutSeconds;

        /// <summary>
        /// Enable historical data seeding at startup
        /// </summary>
        public bool EnableHistoricalSeeding { get; set; } = true;

        /// <summary>
        /// Enable progressive readiness states
        /// </summary>
        public bool EnableProgressiveReadiness { get; set; } = true;

        /// <summary>
        /// Contracts to seed with historical data
        /// </summary>
        public string[] SeedingContracts { get; set; } = { "CON.F.US.EP.Z25", "CON.F.US.ENQ.Z25" };

        /// <summary>
        /// Environment-specific settings
        /// </summary>
        public EnvironmentSettings Environment { get; set; } = new();
    }

    /// <summary>
    /// Environment-specific trading readiness settings
    /// </summary>
    public class EnvironmentSettings
    {
        /// <summary>
        /// Environment name (dev, staging, production)
        /// </summary>
        public string Name { get; set; } = "production";

        /// <summary>
        /// Development environment settings (more relaxed)
        /// </summary>
        public DevEnvironmentSettings Dev { get; set; } = new();

        /// <summary>
        /// Production environment settings (strict)
        /// </summary>
        public ProductionEnvironmentSettings Production { get; set; } = new();
    }

    public class DevEnvironmentSettings
    {
        // Development environment constants (more relaxed settings)
        private const int DevMinBarsSeen = 5;
        private const int DevMinSeededBars = 3;
        private const int DevMinLiveTicks = 1;
        
        public int MinBarsSeen { get; set; } = DevMinBarsSeen;
        public int MinSeededBars { get; set; } = DevMinSeededBars;
        public int MinLiveTicks { get; set; } = DevMinLiveTicks;
        public bool AllowMockData { get; set; } = true;
    }

    public class ProductionEnvironmentSettings
    {
        // Production environment constants (strict settings)
        private const int ProductionMinBarsSeen = 10;
        private const int ProductionMinSeededBars = 12;
        private const int ProductionMinLiveTicks = 2;
        
        public int MinBarsSeen { get; set; } = ProductionMinBarsSeen;
        public int MinSeededBars { get; set; } = ProductionMinSeededBars;
        public int MinLiveTicks { get; set; } = ProductionMinLiveTicks;
        public bool AllowMockData { get; set; }
    }

    /// <summary>
    /// Progressive trading readiness states
    /// </summary>
    public enum TradingReadinessState
    {
        /// <summary>
        /// System starting up, no data yet
        /// </summary>
        Initializing = 0,

        /// <summary>
        /// Historical data seeded, waiting for live data
        /// </summary>
        Seeded = 1,

        /// <summary>
        /// First live tick received, system warming up
        /// </summary>
        LiveTickReceived = 2,

        /// <summary>
        /// Minimum bars satisfied, system ready for trading
        /// </summary>
        FullyReady = 3,

        /// <summary>
        /// Data flow interrupted, system degraded
        /// </summary>
        Degraded = 4,

        /// <summary>
        /// Emergency stop or critical failure
        /// </summary>
        Emergency = 5
    }

    /// <summary>
    /// Trading readiness context for validation
    /// </summary>
    public class TradingReadinessContext
    {
        public int TotalBarsSeen { get; set; }
        public int SeededBars { get; set; }
        public int LiveTicks { get; set; }
        public DateTime LastMarketDataUpdate { get; set; }
        public bool HubsConnected { get; set; }
        public bool CanTrade { get; set; }
        public string? ContractId { get; set; }
        public TradingReadinessState State { get; set; }
        public TimeSpan TimeSinceLastData => DateTime.UtcNow - LastMarketDataUpdate;
    }

    /// <summary>
    /// Readiness validation result
    /// </summary>
    public class ReadinessValidationResult
    {
        public bool IsReady { get; set; }
        public TradingReadinessState State { get; set; }
        public string Reason { get; set; } = string.Empty;
        public double ReadinessScore { get; set; } // 0.0 to 1.0
        public string[] Recommendations { get; set; } = Array.Empty<string>();
    }
}