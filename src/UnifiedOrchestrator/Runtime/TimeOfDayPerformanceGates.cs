using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Collections.Concurrent;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Time-of-Day Performance Gates with EST-based futures trading hours
/// Production-grade temporal filtering for autonomous institutional trading
/// </summary>
public class TimeOfDayPerformanceGates
{
    private readonly ILogger<TimeOfDayPerformanceGates> _logger;
    private readonly ConcurrentDictionary<string, SymbolPerformanceProfile> _symbolProfiles;
    private readonly string _performanceFile;
    private readonly object _profileLock = new();
    
    // EST timezone for futures trading
    private static readonly TimeZoneInfo EstTimeZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");

    public TimeOfDayPerformanceGates(ILogger<TimeOfDayPerformanceGates> logger)
    {
        _logger = logger;
        _symbolProfiles = new ConcurrentDictionary<string, SymbolPerformanceProfile>();
        
        var stateDir = Path.Combine(Directory.GetCurrentDirectory(), "state");
        Directory.CreateDirectory(stateDir);
        _performanceFile = Path.Combine(stateDir, "time_of_day_performance.json");
        
        LoadPerformanceProfiles();
        _logger.LogInformation("⏰ [TIME-OF-DAY-GATES] Initialized with {ProfileCount} symbol performance profiles", _symbolProfiles.Count);
    }

    /// <summary>
    /// Check if trading is allowed based on historical time-of-day performance (EST-based)
    /// </summary>
    public bool IsTradeAllowedAtCurrentTime(string symbol, string strategy = "ALL")
    {
        var estTime = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, EstTimeZone);
        return IsTradeAllowedAtTime(symbol, estTime, strategy);
    }

    /// <summary>
    /// Check if trading is allowed at specific EST time
    /// </summary>
    public bool IsTradeAllowedAtTime(string symbol, DateTime estTime, string strategy = "ALL")
    {
        // Check futures trading session first
        if (!IsFuturesSessionOpen(estTime))
        {
            _logger.LogDebug("🕒 [TIME-OF-DAY-GATES] Futures session closed at {EstTime} for {Symbol}", 
                estTime.ToString("HH:mm:ss"), symbol);
            return false;
        }

        var profile = GetSymbolProfile(symbol);
        var hour = estTime.Hour;
        var minute = estTime.Minute;
        var timeSlot = GetTimeSlot(hour, minute);

        var performance = profile.HourlyPerformance.TryGetValue(timeSlot, out var perf) ? perf : new HourlyPerformance();
        
        // Check if this time period is profitable enough
        var isAllowed = IsTimeSlotProfitable(performance, symbol, strategy);
        
        if (!isAllowed)
        {
            _logger.LogDebug("📊 [TIME-OF-DAY-GATES] Trading BLOCKED for {Symbol} at {EstTime} (time slot: {TimeSlot}, win rate: {WinRate:F2}, avg PnL: {AvgPnL:F2})",
                symbol, estTime.ToString("HH:mm:ss"), timeSlot, performance.WinRate, performance.AveragePnL);
        }
        
        return isAllowed;
    }

    /// <summary>
    /// Record trading outcome for time-of-day learning
    /// </summary>
    public void RecordTradingOutcome(string symbol, string strategy, decimal pnl, bool wasWinner, DateTime estTime)
    {
        var profile = GetSymbolProfile(symbol);
        var timeSlot = GetTimeSlot(estTime.Hour, estTime.Minute);

        lock (_profileLock)
        {
            if (!profile.HourlyPerformance.TryGetValue(timeSlot, out var performance))
            {
                performance = new HourlyPerformance();
                profile.HourlyPerformance[timeSlot] = performance;
            }

            // Update performance metrics
            performance.TotalTrades++;
            performance.TotalPnL += pnl;
            performance.AveragePnL = performance.TotalPnL / performance.TotalTrades;
            
            if (wasWinner)
            {
                performance.WinningTrades++;
            }
            
            performance.WinRate = (decimal)performance.WinningTrades / performance.TotalTrades;
            performance.LastTradeTime = DateTime.UtcNow;

            // Update strategy-specific performance if provided
            if (!string.IsNullOrEmpty(strategy) && strategy != "ALL")
            {
                if (!performance.StrategyPerformance.TryGetValue(strategy, out var stratPerf))
                {
                    stratPerf = new StrategyPerformance();
                    performance.StrategyPerformance[strategy] = stratPerf;
                }

                stratPerf.TotalTrades++;
                stratPerf.TotalPnL += pnl;
                stratPerf.AveragePnL = stratPerf.TotalPnL / stratPerf.TotalTrades;
                
                if (wasWinner)
                {
                    stratPerf.WinningTrades++;
                }
                
                stratPerf.WinRate = (decimal)stratPerf.WinningTrades / stratPerf.TotalTrades;
            }
        }

        _logger.LogDebug("📈 [TIME-OF-DAY-GATES] Recorded outcome for {Symbol} at {TimeSlot}: PnL={PnL:F2}, Winner={Winner}, Strategy={Strategy}",
            symbol, timeSlot, pnl, wasWinner, strategy);

        // Trigger periodic save
        if (performance.TotalTrades % 10 == 0) // Save every 10 trades
        {
            Task.Run(SavePerformanceProfiles);
        }
    }

    /// <summary>
    /// Get best trading hours for a symbol
    /// </summary>
    public IReadOnlyList<string> GetBestTradingHours(string symbol, int topCount = 5)
    {
        var profile = GetSymbolProfile(symbol);
        
        return profile.HourlyPerformance
            .Where(kvp => kvp.Value.TotalTrades >= 10) // Minimum sample size
            .OrderByDescending(kvp => kvp.Value.AveragePnL)
            .ThenByDescending(kvp => kvp.Value.WinRate)
            .Take(topCount)
            .Select(kvp => kvp.Key)
            .ToList()
            .AsReadOnly();
    }

    /// <summary>
    /// Get worst trading hours for a symbol
    /// </summary>
    public IReadOnlyList<string> GetWorstTradingHours(string symbol, int bottomCount = 5)
    {
        var profile = GetSymbolProfile(symbol);
        
        return profile.HourlyPerformance
            .Where(kvp => kvp.Value.TotalTrades >= 10) // Minimum sample size
            .OrderBy(kvp => kvp.Value.AveragePnL)
            .ThenBy(kvp => kvp.Value.WinRate)
            .Take(bottomCount)
            .Select(kvp => kvp.Key)
            .ToList()
            .AsReadOnly();
    }

    /// <summary>
    /// Check if futures session is open (EST-based)
    /// </summary>
    private static bool IsFuturesSessionOpen(DateTime estTime)
    {
        var dayOfWeek = estTime.DayOfWeek;
        var timeOfDay = estTime.TimeOfDay;

        return dayOfWeek switch
        {
            DayOfWeek.Sunday => timeOfDay >= TimeSpan.FromHours(18), // 6:00 PM EST Sunday
            DayOfWeek.Monday or DayOfWeek.Tuesday or DayOfWeek.Wednesday or DayOfWeek.Thursday => true, // All day
            DayOfWeek.Friday => timeOfDay < TimeSpan.FromHours(17), // Until 5:00 PM EST Friday
            DayOfWeek.Saturday => false, // Closed Saturday
            _ => false
        };
    }

    /// <summary>
    /// Get symbol-specific performance profile
    /// </summary>
    private SymbolPerformanceProfile GetSymbolProfile(string symbol)
    {
        return _symbolProfiles.GetOrAdd(symbol, _ => CreateDefaultProfile(symbol));
    }

    /// <summary>
    /// Create default performance profile for a symbol
    /// </summary>
    private static SymbolPerformanceProfile CreateDefaultProfile(string symbol)
    {
        var profile = new SymbolPerformanceProfile
        {
            Symbol = symbol,
            CreatedTime = DateTime.UtcNow,
            HourlyPerformance = new Dictionary<string, HourlyPerformance>()
        };

        // Initialize all time slots with default performance
        for (int hour = 0; hour < 24; hour++)
        {
            for (int minute = 0; minute < 60; minute += 30) // 30-minute slots
            {
                var timeSlot = GetTimeSlot(hour, minute);
                profile.HourlyPerformance[timeSlot] = new HourlyPerformance
                {
                    WinRate = 0.5m, // Start neutral
                    AveragePnL = 0.0m,
                    TotalTrades = 0
                };
            }
        }

        return profile;
    }

    /// <summary>
    /// Get time slot identifier (30-minute intervals)
    /// </summary>
    private static string GetTimeSlot(int hour, int minute)
    {
        var slotMinute = minute < 30 ? 0 : 30;
        return $"{hour:D2}:{slotMinute:D2}";
    }

    /// <summary>
    /// Determine if a time slot is profitable enough for trading
    /// </summary>
    private bool IsTimeSlotProfitable(HourlyPerformance performance, string symbol, string strategy)
    {
        // If no historical data, default to allowing trades
        if (performance.TotalTrades < 5)
        {
            return true;
        }

        // Define profitability thresholds based on symbol
        var (minWinRate, minAvgPnL) = GetProfitabilityThresholds(symbol);

        // Check strategy-specific performance if available
        if (strategy != "ALL" && performance.StrategyPerformance.TryGetValue(strategy, out var stratPerf))
        {
            if (stratPerf.TotalTrades >= 3) // Minimum sample for strategy-specific decision
            {
                return stratPerf.WinRate >= minWinRate && stratPerf.AveragePnL >= minAvgPnL;
            }
        }

        // Fall back to overall performance
        return performance.WinRate >= minWinRate && performance.AveragePnL >= minAvgPnL;
    }

    /// <summary>
    /// Get profitability thresholds based on symbol characteristics
    /// </summary>
    private static (decimal minWinRate, decimal minAvgPnL) GetProfitabilityThresholds(string symbol)
    {
        return symbol.ToUpperInvariant() switch
        {
            // ES - Most liquid, higher standards
            var s when s.StartsWith("ES") => (0.52m, 5.0m),
            
            // MES - Micro contracts, lower absolute PnL threshold
            var s when s.StartsWith("MES") => (0.50m, 1.0m),
            
            // NQ - Higher volatility, adjust thresholds
            var s when s.StartsWith("NQ") => (0.51m, 10.0m),
            
            // MNQ - Micro NASDAQ, lower threshold
            var s when s.StartsWith("MNQ") => (0.50m, 2.0m),
            
            // Default thresholds
            _ => (0.50m, 2.0m)
        };
    }

    /// <summary>
    /// Load performance profiles from persistence
    /// </summary>
    private void LoadPerformanceProfiles()
    {
        try
        {
            if (!File.Exists(_performanceFile))
            {
                _logger.LogInformation("📂 [TIME-OF-DAY-GATES] No performance file found, starting fresh");
                return;
            }

            var json = File.ReadAllText(_performanceFile);
            var persistedData = JsonSerializer.Deserialize<PerformanceData>(json);
            
            if (persistedData?.Profiles != null)
            {
                foreach (var profile in persistedData.Profiles)
                {
                    _symbolProfiles.TryAdd(profile.Symbol, profile);
                }

                _logger.LogInformation("📂 [TIME-OF-DAY-GATES] Loaded {Count} performance profiles (last save: {LastSave})", 
                    persistedData.Profiles.Count, persistedData.LastSavedUtc);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [TIME-OF-DAY-GATES] Failed to load performance profiles");
        }
    }

    /// <summary>
    /// Save performance profiles to persistence
    /// </summary>
    private void SavePerformanceProfiles()
    {
        try
        {
            var persistedData = new PerformanceData
            {
                Profiles = _symbolProfiles.Values.ToList(),
                LastSavedUtc = DateTime.UtcNow,
                Version = "2.0"
            };

            var json = JsonSerializer.Serialize(persistedData, new JsonSerializerOptions 
            { 
                WriteIndented = false,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            });
            
            // Atomic write
            var tempFile = _performanceFile + ".tmp";
            File.WriteAllText(tempFile, json);
            File.Move(tempFile, _performanceFile, overwrite: true);

            _logger.LogDebug("💾 [TIME-OF-DAY-GATES] Saved {Count} performance profiles", _symbolProfiles.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [TIME-OF-DAY-GATES] Failed to save performance profiles");
        }
    }

    /// <summary>
    /// Get performance statistics for monitoring
    /// </summary>
    public TimeOfDayStats GetStats()
    {
        var allProfiles = _symbolProfiles.Values.ToList();
        
        return new TimeOfDayStats
        {
            TotalSymbols = allProfiles.Count,
            TotalTimeSlots = allProfiles.Sum(p => p.HourlyPerformance.Count),
            TotalTrades = allProfiles.Sum(p => p.HourlyPerformance.Values.Sum(h => h.TotalTrades)),
            OverallWinRate = CalculateOverallWinRate(allProfiles),
            BestPerformingHour = GetBestPerformingHour(allProfiles),
            WorstPerformingHour = GetWorstPerformingHour(allProfiles)
        };
    }

    private static decimal CalculateOverallWinRate(List<SymbolPerformanceProfile> profiles)
    {
        var totalTrades = profiles.Sum(p => p.HourlyPerformance.Values.Sum(h => h.TotalTrades));
        var totalWins = profiles.Sum(p => p.HourlyPerformance.Values.Sum(h => h.WinningTrades));
        
        return totalTrades > 0 ? (decimal)totalWins / totalTrades : 0.5m;
    }

    private static string GetBestPerformingHour(List<SymbolPerformanceProfile> profiles)
    {
        return profiles
            .SelectMany(p => p.HourlyPerformance)
            .Where(kvp => kvp.Value.TotalTrades >= 10)
            .OrderByDescending(kvp => kvp.Value.AveragePnL)
            .ThenByDescending(kvp => kvp.Value.WinRate)
            .FirstOrDefault().Key ?? "Unknown";
    }

    private static string GetWorstPerformingHour(List<SymbolPerformanceProfile> profiles)
    {
        return profiles
            .SelectMany(p => p.HourlyPerformance)
            .Where(kvp => kvp.Value.TotalTrades >= 10)
            .OrderBy(kvp => kvp.Value.AveragePnL)
            .ThenBy(kvp => kvp.Value.WinRate)
            .FirstOrDefault().Key ?? "Unknown";
    }
}

/// <summary>
/// Symbol-specific performance profile
/// </summary>
public class SymbolPerformanceProfile
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime CreatedTime { get; set; }
    public Dictionary<string, HourlyPerformance> HourlyPerformance { get; set; } = new();
}

/// <summary>
/// Performance data for a specific hour/time slot
/// </summary>
public class HourlyPerformance
{
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public decimal WinRate { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal AveragePnL { get; set; }
    public DateTime LastTradeTime { get; set; }
    public Dictionary<string, StrategyPerformance> StrategyPerformance { get; set; } = new();
}

/// <summary>
/// Strategy-specific performance within a time slot
/// </summary>
public class StrategyPerformance
{
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public decimal WinRate { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal AveragePnL { get; set; }
}

/// <summary>
/// Persistence data structure
/// </summary>
public class PerformanceData
{
    public List<SymbolPerformanceProfile> Profiles { get; set; } = new();
    public DateTime LastSavedUtc { get; set; }
    public string Version { get; set; } = "2.0";
}

/// <summary>
/// Statistics for monitoring
/// </summary>
public class TimeOfDayStats
{
    public int TotalSymbols { get; set; }
    public int TotalTimeSlots { get; set; }
    public int TotalTrades { get; set; }
    public decimal OverallWinRate { get; set; }
    public string BestPerformingHour { get; set; } = string.Empty;
    public string WorstPerformingHour { get; set; } = string.Empty;
}