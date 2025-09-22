using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace BotCore.Services;

/// <summary>
/// Session-aware runtime gates for ES/NQ futures (24×5) trading
/// Handles RTH/ETH sessions, daily maintenance halt, and Sunday reopen stabilization
/// Implements the specific session logic requested for complete futures hours coverage
/// </summary>
public class SessionAwareRuntimeGates
{
    private readonly ILogger<SessionAwareRuntimeGates> _logger;
    private readonly IConfiguration _configuration;
    private readonly SessionConfiguration _sessionConfig;

    public SessionAwareRuntimeGates(ILogger<SessionAwareRuntimeGates> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        _sessionConfig = LoadSessionConfiguration();
    }

    /// <summary>
    /// Check if trading is currently allowed based on session rules
    /// Covers: RTH/ETH permissions, maintenance breaks, Sunday reopen curbs
    /// </summary>
    public async Task<bool> IsTradingAllowedAsync(string symbol = "ES", CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var etNow = NowEt();
        
        // Check maintenance break (daily halt 5:00-6:00 PM ET)
        if (IsInMaintenanceBreak(etNow))
        {
            _logger.LogDebug("Trading blocked: In maintenance break (17:00-18:00 ET)");
            return false;
        }
        
        // Check weekend closure
        if (IsWeekendClosed(etNow))
        {
            _logger.LogDebug("Trading blocked: Weekend closure");
            return false;
        }
        
        // Check Sunday reopen stabilization curb
        if (IsSundayReopenCurb(etNow))
        {
            _logger.LogDebug("Trading blocked: Sunday reopen stabilization (first {CurbMins} minutes)", 
                _sessionConfig.SundayReopen.CurbMins);
            return false;
        }
        
        // Check ETH first minutes curb after daily reopen
        if (IsEthFirstMinsCurb(etNow))
        {
            _logger.LogDebug("Trading blocked: ETH first minutes curb (first {CurbMins} minutes after reopen)", 
                _sessionConfig.ETH.CurbFirstMins);
            return false;
        }
        
        // Check if ETH trading is allowed
        if (IsEthSession(etNow) && !_sessionConfig.ETH.Allow)
        {
            _logger.LogDebug("Trading blocked: ETH trading disabled");
            return false;
        }
        
        _logger.LogDebug("Trading allowed: {Session} session active", GetCurrentSession(etNow));
        return true;
    }

    /// <summary>
    /// Get current trading session (RTH, ETH, MAINTENANCE, CLOSED)
    /// </summary>
    public string GetCurrentSession(DateTime? etTime = null)
    {
        var et = etTime ?? NowEt();
        
        if (IsInMaintenanceBreak(et))
            return "MAINTENANCE";
            
        if (IsWeekendClosed(et))
            return "CLOSED";
            
        if (IsRthSession(et))
            return "RTH";
            
        if (_sessionConfig.ETH.Allow)
            return "ETH";
            
        return "CLOSED";
    }

    /// <summary>
    /// Check if currently in Regular Trading Hours (9:30 AM - 4:00 PM ET)
    /// </summary>
    public bool IsRthSession(DateTime? etTime = null)
    {
        var et = etTime ?? NowEt();
        var timeOfDay = et.TimeOfDay;
        
        return InRange(_sessionConfig.RTH.Start, _sessionConfig.RTH.End, et);
    }

    /// <summary>
    /// Check if currently in Extended Trading Hours (outside RTH but market open)
    /// </summary>
    public bool IsEthSession(DateTime? etTime = null)
    {
        var et = etTime ?? NowEt();
        
        // Not weekend, not maintenance, not RTH = ETH
        return !IsWeekendClosed(et) && 
               !IsInMaintenanceBreak(et) && 
               !IsRthSession(et);
    }

    /// <summary>
    /// Get detailed session status with timing information
    /// </summary>
    public SessionStatus GetSessionStatus(DateTime? etTime = null)
    {
        var et = etTime ?? NowEt();
        var session = GetCurrentSession(et);
        var tradingAllowed = IsTradingAllowedAsync().Result; // Safe for synchronous context
        
        return new SessionStatus
        {
            CurrentSession = session,
            TradingAllowed = tradingAllowed,
            IsRth = IsRthSession(et),
            IsEth = IsEthSession(et),
            IsMaintenanceBreak = IsInMaintenanceBreak(et),
            IsWeekendClosed = IsWeekendClosed(et),
            IsSundayReopenCurb = IsSundayReopenCurb(et),
            IsEthFirstMinsCurb = IsEthFirstMinsCurb(et),
            EasternTime = et,
            NextSessionChange = GetNextSessionChange(et),
            IsWithinReopenCurbWindow = IsWithinReopenCurbWindow(et),
            ReopenCurbTimeRemaining = GetReopenCurbTimeRemaining(et)
        };
    }

    #region Private Helper Methods

    /// <summary>
    /// Get current Eastern Time (handles DST automatically)
    /// </summary>
    private static DateTime NowEt()
    {
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, easternZone);
        }
        catch
        {
            // Fallback if timezone not found
            return DateTime.UtcNow.AddHours(-5);
        }
    }

    /// <summary>
    /// Check if time is within a specific range (handles cross-day ranges)
    /// </summary>
    private static bool InRange(string hhmm, string hhmm2, DateTime et)
    {
        var (h1, m1) = (int.Parse(hhmm[..2]), int.Parse(hhmm[3..]));
        var (h2, m2) = (int.Parse(hhmm2[..2]), int.Parse(hhmm2[3..]));
        var a = new TimeSpan(h1, m1, 0);
        var b = new TimeSpan(h2, m2, 0);
        var t = et.TimeOfDay;
        
        return a <= b ? (t >= a && t < b) : (t >= a || t < b);
    }

    /// <summary>
    /// Check if in daily maintenance break (17:00-18:00 ET)
    /// </summary>
    private bool IsInMaintenanceBreak(DateTime et)
    {
        return InRange(_sessionConfig.MaintenanceBreak.Start, _sessionConfig.MaintenanceBreak.End, et);
    }

    /// <summary>
    /// Check if weekend closure (Friday 17:00 ET to Sunday 18:00 ET)
    /// </summary>
    private bool IsWeekendClosed(DateTime et)
    {
        var dayOfWeek = et.DayOfWeek;
        var timeOfDay = et.TimeOfDay;
        
        // Saturday is always closed
        if (dayOfWeek == DayOfWeek.Saturday)
            return true;
            
        // Friday after 17:00 ET
        if (dayOfWeek == DayOfWeek.Friday && timeOfDay >= TimeSpan.Parse(_sessionConfig.MaintenanceBreak.Start))
            return true;
            
        // Sunday before 18:00 ET
        if (dayOfWeek == DayOfWeek.Sunday && timeOfDay < TimeSpan.Parse(_sessionConfig.MaintenanceBreak.End))
            return true;
            
        return false;
    }

    /// <summary>
    /// Check if in Sunday reopen stabilization period
    /// </summary>
    private bool IsSundayReopenCurb(DateTime et)
    {
        if (!_sessionConfig.SundayReopen.Enable)
            return false;
            
        if (et.DayOfWeek != DayOfWeek.Sunday)
            return false;
            
        var reopenTime = TimeSpan.Parse(_sessionConfig.MaintenanceBreak.End); // 18:00
        var curbEndTime = reopenTime.Add(TimeSpan.FromMinutes(_sessionConfig.SundayReopen.CurbMins));
        
        return et.TimeOfDay >= reopenTime && et.TimeOfDay < curbEndTime;
    }

    /// <summary>
    /// Check if in ETH first minutes curb after daily reopen
    /// Enhanced logic tracks maintenance break end times and curbs trading for first few minutes
    /// </summary>
    private bool IsEthFirstMinsCurb(DateTime et)
    {
        if (_sessionConfig.ETH.CurbFirstMins <= 0)
            return false;
            
        // Check if just after daily reopen (18:00 ET Monday-Thursday)
        var dayOfWeek = et.DayOfWeek;
        if (dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Thursday)
        {
            var reopenTime = TimeSpan.Parse(_sessionConfig.MaintenanceBreak.End); // 18:00
            var curbEndTime = reopenTime.Add(TimeSpan.FromMinutes(_sessionConfig.ETH.CurbFirstMins));
            
            return et.TimeOfDay >= reopenTime && et.TimeOfDay < curbEndTime;
        }
        
        return false;
    }

    /// <summary>
    /// Enhanced reopen curbing logic - tracks session restart times and applies curbs
    /// Stores maintenance break end times and checks if current time is within curbing window
    /// </summary>
    public bool IsWithinReopenCurbWindow(DateTime? etTime = null)
    {
        var et = etTime ?? NowEt();
        
        // Check ETH first minutes curb
        if (IsEthFirstMinsCurb(et))
        {
            _logger.LogDebug("[REOPEN_CURB] ETH first minutes curb active: {CurbMins} minutes after {ReopenTime}", 
                _sessionConfig.ETH.CurbFirstMins, _sessionConfig.MaintenanceBreak.End);
            return true;
        }
        
        // Check Sunday reopen curb
        if (IsSundayReopenCurb(et))
        {
            _logger.LogDebug("[REOPEN_CURB] Sunday reopen curb active: {CurbMins} minutes after Sunday {ReopenTime}", 
                _sessionConfig.SundayReopen.CurbMins, _sessionConfig.MaintenanceBreak.End);
            return true;
        }
        
        return false;
    }

    /// <summary>
    /// Get time remaining in current reopen curb window (if any)
    /// </summary>
    public TimeSpan? GetReopenCurbTimeRemaining(DateTime? etTime = null)
    {
        var et = etTime ?? NowEt();
        
        if (!IsWithinReopenCurbWindow(et))
            return null;
        
        var reopenTime = TimeSpan.Parse(_sessionConfig.MaintenanceBreak.End); // 18:00
        
        // Check ETH curb
        if (IsEthFirstMinsCurb(et))
        {
            var curbEndTime = reopenTime.Add(TimeSpan.FromMinutes(_sessionConfig.ETH.CurbFirstMins));
            return curbEndTime - et.TimeOfDay;
        }
        
        // Check Sunday curb
        if (IsSundayReopenCurb(et))
        {
            var curbEndTime = reopenTime.Add(TimeSpan.FromMinutes(_sessionConfig.SundayReopen.CurbMins));
            return curbEndTime - et.TimeOfDay;
        }
        
        return null;
    }

    /// <summary>
    /// Get next session change time
    /// </summary>
    private DateTime? GetNextSessionChange(DateTime et)
    {
        var timeOfDay = et.TimeOfDay;
        var today = et.Date;
        
        // Check upcoming session changes today
        var rthStart = TimeSpan.Parse(_sessionConfig.RTH.Start);
        var rthEnd = TimeSpan.Parse(_sessionConfig.RTH.End);
        var maintenanceStart = TimeSpan.Parse(_sessionConfig.MaintenanceBreak.Start);
        var maintenanceEnd = TimeSpan.Parse(_sessionConfig.MaintenanceBreak.End);
        
        if (timeOfDay < rthStart)
            return today.Add(rthStart);
        if (timeOfDay < rthEnd)
            return today.Add(rthEnd);
        if (timeOfDay < maintenanceStart)
            return today.Add(maintenanceStart);
        if (timeOfDay < maintenanceEnd)
            return today.Add(maintenanceEnd);
            
        // Next change is tomorrow's RTH start
        return today.AddDays(1).Add(rthStart);
    }

    /// <summary>
    /// Load session configuration from appsettings
    /// </summary>
    private SessionConfiguration LoadSessionConfiguration()
    {
        var section = _configuration.GetSection("Sessions");
        
        return new SessionConfiguration
        {
            TimeZone = section["TimeZone"] ?? "America/New_York",
            MaintenanceBreak = new MaintenanceBreakConfig
            {
                Start = section["MaintenanceBreak:Start"] ?? "17:00",
                End = section["MaintenanceBreak:End"] ?? "18:00"
            },
            RTH = new RthConfig
            {
                Start = section["RTH:Start"] ?? "09:30",
                End = section["RTH:End"] ?? "16:00"
            },
            ETH = new EthConfig
            {
                Allow = section.GetValue<bool>("ETH:Allow", true),
                CurbFirstMins = section.GetValue<int>("ETH:CurbFirstMins", 3)
            },
            SundayReopen = new SundayReopenConfig
            {
                Enable = section.GetValue<bool>("SundayReopen:Enable", true),
                CurbMins = section.GetValue<int>("SundayReopen:CurbMins", 5)
            }
        };
    }

    #endregion
}

/// <summary>
/// Session configuration model
/// </summary>
public class SessionConfiguration
{
    public string TimeZone { get; set; } = "America/New_York";
    public MaintenanceBreakConfig MaintenanceBreak { get; set; } = new();
    public RthConfig RTH { get; set; } = new();
    public EthConfig ETH { get; set; } = new();
    public SundayReopenConfig SundayReopen { get; set; } = new();
}

public class MaintenanceBreakConfig
{
    public string Start { get; set; } = "17:00";
    public string End { get; set; } = "18:00";
}

public class RthConfig
{
    public string Start { get; set; } = "09:30";
    public string End { get; set; } = "16:00";
}

public class EthConfig
{
    public bool Allow { get; set; } = true;
    public int CurbFirstMins { get; set; } = 3;
}

public class SundayReopenConfig
{
    public bool Enable { get; set; } = true;
    public int CurbMins { get; set; } = 5;
}

/// <summary>
/// Detailed session status
/// </summary>
public class SessionStatus
{
    public string CurrentSession { get; set; } = string.Empty;
    public bool TradingAllowed { get; set; }
    public bool IsRth { get; set; }
    public bool IsEth { get; set; }
    public bool IsMaintenanceBreak { get; set; }
    public bool IsWeekendClosed { get; set; }
    public bool IsSundayReopenCurb { get; set; }
    public bool IsEthFirstMinsCurb { get; set; }
    public DateTime EasternTime { get; set; }
    public DateTime? NextSessionChange { get; set; }
    public bool IsWithinReopenCurbWindow { get; set; }
    public TimeSpan? ReopenCurbTimeRemaining { get; set; }
}