// ES & NQ 24/7 Trading Schedule Configuration
// Defines comprehensive trading sessions with instrument-specific strategies and position sizing
using System;
using System.Collections.Generic;

namespace BotCore.Config
{
    /// <summary>
    /// 24/7 ES & NQ futures trading schedule with time-optimized strategy allocation
    /// Covers all major trading sessions from Asian overnight to US close
    /// </summary>
    public static class ES_NQ_TradingSchedule
    {
        // ES & NQ FUTURES TRADING HOURS (23/5 - Sunday 5PM to Friday 4PM CT)
        public static readonly Dictionary<string, TradingSession> Sessions = new()
        {
            // ============================================
            // OVERNIGHT SESSION (BEST FOR NQ)
            // ============================================
            ["AsianSession"] = new TradingSession
            {
                Start = new TimeSpan(18, 0, 0), // 6:00 PM CT (7PM ET)
                End = new TimeSpan(23, 59, 0),  // 11:59 PM CT
                Instruments = new[] { "NQ", "ES" },
                PrimaryInstrument = "NQ", // Tech-heavy Asian trading
                Strategies = new Dictionary<string, string[]>
                {
                    ["NQ"] = new[] { "S2", "S11", "S3" }, // Mean reversion works well
                    ["ES"] = new[] { "S2", "S11" } // Lower volume, fade extremes
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["NQ"] = 0.8,
                    ["ES"] = 0.6
                },
                Description = "Asian Session - NQ more active",
                IsActive = true
            },

            ["EuropeanPreOpen"] = new TradingSession
            {
                Start = new TimeSpan(0, 0, 0),   // 12:00 AM CT (1AM ET)
                End = new TimeSpan(2, 0, 0),     // 2:00 AM CT (3AM ET)
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "ES",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S2", "S3" }, // European prep
                    ["NQ"] = new[] { "S2" } // Follow ES
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 0.7,
                    ["NQ"] = 0.5
                },
                Description = "European Pre-Open positioning",
                IsActive = true
            },

            ["EuropeanOpen"] = new TradingSession
            {
                Start = new TimeSpan(2, 0, 0),   // 2:00 AM CT (3AM ET, 8AM London)
                End = new TimeSpan(5, 0, 0),     // 5:00 AM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "ES",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S3", "S6", "S2" }, // Breakouts common
                    ["NQ"] = new[] { "S3", "S2" }
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 0.9,
                    ["NQ"] = 0.8
                },
                Description = "European Open - Increased volatility",
                IsActive = true
            },

            ["LondonMorning"] = new TradingSession
            {
                Start = new TimeSpan(5, 0, 0),   // 5:00 AM CT
                End = new TimeSpan(8, 30, 0),    // 8:30 AM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "ES",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S2", "S3", "S11" },
                    ["NQ"] = new[] { "S2", "S3" }
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 1.0,
                    ["NQ"] = 0.9
                },
                Description = "London Morning - Good liquidity",
                IsActive = true
            },

            // ============================================
            // US PRE-MARKET (CRITICAL SETUP PERIOD)
            // ============================================
            ["USPreMarket"] = new TradingSession
            {
                Start = new TimeSpan(8, 30, 0),  // 8:30 AM CT (9:30 AM ET)
                End = new TimeSpan(9, 28, 0),    // 9:28 AM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "BOTH",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S3", "S2" }, // Compression setups
                    ["NQ"] = new[] { "S3", "S2" }  // Compression setups
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 0.8,
                    ["NQ"] = 0.8
                },
                Description = "Pre-Market - Position for open",
                IsActive = true
            },

            // ============================================
            // OPENING DRIVE (HIGHEST OPPORTUNITY)
            // ============================================
            ["OpeningDrive"] = new TradingSession
            {
                Start = new TimeSpan(9, 28, 0),  // 9:28 AM CT
                End = new TimeSpan(10, 0, 0),    // 10:00 AM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "BOTH",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S6" }, // Opening Drive ONLY
                    ["NQ"] = new[] { "S6" }  // Opening Drive ONLY
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 1.2, // Increase size for high probability
                    ["NQ"] = 1.2
                },
                Description = "CRITICAL: Opening Drive Window",
                IsActive = true
            },

            // ============================================
            // MORNING TREND (BEST TRENDING PERIOD)
            // ============================================
            ["MorningTrend"] = new TradingSession
            {
                Start = new TimeSpan(10, 0, 0),  // 10:00 AM CT
                End = new TimeSpan(11, 30, 0),   // 11:30 AM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "NQ", // Tech often leads
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S3", "S2", "S11" },
                    ["NQ"] = new[] { "S3", "S2", "S11" }
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 1.0,
                    ["NQ"] = 1.1 // NQ trends stronger
                },
                Description = "Morning Trend - Best trends",
                IsActive = true
            },

            // ============================================
            // LUNCH CHOP (REDUCE ACTIVITY)
            // ============================================
            ["LunchChop"] = new TradingSession
            {
                Start = new TimeSpan(11, 30, 0), // 11:30 AM CT
                End = new TimeSpan(13, 30, 0),   // 1:30 PM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "ES",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S2" }, // Mean reversion only
                    ["NQ"] = new[] { "S2" }  // Mean reversion only
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 0.5, // Reduce size in chop
                    ["NQ"] = 0.5
                },
                Description = "Lunch - Low volume chop",
                IsActive = true
            },

            // ============================================
            // AFTERNOON TREND (ADR EXHAUSTION TIME)
            // ============================================
            ["AfternoonTrend"] = new TradingSession
            {
                Start = new TimeSpan(13, 30, 0), // 1:30 PM CT
                End = new TimeSpan(14, 30, 0),   // 2:30 PM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "BOTH",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S11", "S3", "S2" }, // ADR exhaustion prime time
                    ["NQ"] = new[] { "S11", "S3", "S2" }
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 1.0,
                    ["NQ"] = 1.0
                },
                Description = "Afternoon - ADR exhaustion setups",
                IsActive = true
            },

            // ============================================
            // CLOSING HOUR (HIGH VOLATILITY)
            // ============================================
            ["PowerHour"] = new TradingSession
            {
                Start = new TimeSpan(14, 30, 0), // 2:30 PM CT
                End = new TimeSpan(15, 0, 0),    // 3:00 PM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "BOTH",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S3", "S11", "S2" },
                    ["NQ"] = new[] { "S3", "S11", "S2" }
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 0.9,
                    ["NQ"] = 0.9
                },
                Description = "Power Hour - Increased volatility",
                IsActive = true
            },

            ["MarketClose"] = new TradingSession
            {
                Start = new TimeSpan(15, 0, 0),  // 3:00 PM CT
                End = new TimeSpan(16, 0, 0),    // 4:00 PM CT (Market close)
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "ES",
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S2", "S11" }, // Fade extremes
                    ["NQ"] = new[] { "S2", "S11" }
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 0.7,
                    ["NQ"] = 0.7
                },
                Description = "Close - Position squaring",
                IsActive = true
            },

            // ============================================
            // AFTER HOURS (REDUCED BUT ACTIVE)
            // ============================================
            ["AfterHours"] = new TradingSession
            {
                Start = new TimeSpan(16, 0, 0),  // 4:00 PM CT
                End = new TimeSpan(18, 0, 0),    // 6:00 PM CT
                Instruments = new[] { "ES", "NQ" },
                PrimaryInstrument = "NQ", // Tech earnings often AH
                Strategies = new Dictionary<string, string[]>
                {
                    ["ES"] = new[] { "S2" },
                    ["NQ"] = new[] { "S2", "S3" } // Earnings reactions
                },
                PositionSizeMultiplier = new Dictionary<string, double>
                {
                    ["ES"] = 0.5,
                    ["NQ"] = 0.6
                },
                Description = "After Hours - Earnings/News",
                IsActive = true
            }
        };

        /// <summary>
        /// Get the currently active trading session based on Central Time
        /// </summary>
        /// <param name="currentTime">Current time in Central Time</param>
        /// <returns>Active trading session or null if market closed</returns>
        public static TradingSession? GetCurrentSession(TimeSpan currentTime)
        {
            foreach (var kvp in Sessions)
            {
                var session = kvp.Value;
                if (!session.IsActive) continue;

                // Handle sessions that cross midnight
                if (session.Start > session.End)
                {
                    if (currentTime >= session.Start || currentTime <= session.End)
                        return session;
                }
                else
                {
                    if (currentTime >= session.Start && currentTime <= session.End)
                        return session;
                }
            }

            return null; // Market closed
        }

        /// <summary>
        /// Get the current session by name
        /// </summary>
        public static TradingSession? GetSessionByName(string sessionName)
        {
            return Sessions.TryGetValue(sessionName, out var session) ? session : null;
        }

        /// <summary>
        /// Check if an instrument should be trading in the current session
        /// </summary>
        public static bool ShouldTradeInstrument(string instrument, TimeSpan currentTime)
        {
            var session = GetCurrentSession(currentTime);
            return session?.Instruments.Contains(instrument) == true;
        }

        /// <summary>
        /// Get strategies for an instrument in the current session
        /// </summary>
        public static string[] GetActiveStrategies(string instrument, TimeSpan currentTime)
        {
            var session = GetCurrentSession(currentTime);
            if (session?.Strategies.TryGetValue(instrument, out var strategies) == true)
                return strategies;

            return Array.Empty<string>();
        }

        /// <summary>
        /// Get position size multiplier for an instrument in current session
        /// </summary>
        public static double GetPositionSizeMultiplier(string instrument, TimeSpan currentTime)
        {
            var session = GetCurrentSession(currentTime);
            if (session?.PositionSizeMultiplier.TryGetValue(instrument, out var multiplier) == true)
                return multiplier;

            return 1.0; // Default multiplier
        }
    }

    /// <summary>
    /// Represents a trading session with instrument-specific configuration
    /// </summary>
    public class TradingSession
    {
        public TimeSpan Start { get; set; }
        public TimeSpan End { get; set; }
        public string[] Instruments { get; set; } = Array.Empty<string>();
        public string PrimaryInstrument { get; set; } = "";
        public Dictionary<string, string[]> Strategies { get; set; } = new();
        public Dictionary<string, double> PositionSizeMultiplier { get; set; } = new();
        public string Description { get; set; } = "";
        public bool IsActive { get; set; } = true;
    }
}