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
        /// <summary>
        /// Minimum bars required before trading is allowed
        /// Configurable per environment (dev vs production)
        /// </summary>
        [Range(1, 100)]
        public int MinBarsSeen { get; set; } = 10;

        /// <summary>
        /// Minimum seeded bars from historical data
        /// </summary>
        [Range(0, 50)]
        public int MinSeededBars { get; set; } = 12;

        /// <summary>
        /// Minimum live ticks required after seeding
        /// Ensures at least some live data before trading
        /// </summary>
        [Range(1, 10)]
        public int MinLiveTicks { get; set; } = 2;

        /// <summary>
        /// Maximum age for historical seeding data in hours
        /// </summary>
        [Range(1, 168)] // 1 hour to 1 week
        public int MaxHistoricalDataAgeHours { get; set; } = 24;

        /// <summary>
        /// Market data health timeout in seconds
        /// If no data received within this time, system is considered unhealthy
        /// </summary>
        [Range(30, 600)]
        public int MarketDataTimeoutSeconds { get; set; } = 300;

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
        public int MinBarsSeen { get; set; } = 5;
        public int MinSeededBars { get; set; } = 3;
        public int MinLiveTicks { get; set; } = 1;
        public bool AllowMockData { get; set; } = true;
    }

    public class ProductionEnvironmentSettings
    {
        public int MinBarsSeen { get; set; } = 10;
        public int MinSeededBars { get; set; } = 12;
        public int MinLiveTicks { get; set; } = 2;
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