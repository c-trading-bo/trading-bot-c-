using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading.Tasks;

namespace BotCore.Market;

/// <summary>
/// Interface for economic event management and filtering
/// </summary>
public interface IEconomicEventManager
{
    /// <summary>
    /// Gets upcoming economic events within the specified time window
    /// </summary>
    Task<IEnumerable<EconomicEvent>> GetUpcomingEventsAsync(TimeSpan timeWindow);

    /// <summary>
    /// Gets events filtered by impact level
    /// </summary>
    Task<IEnumerable<EconomicEvent>> GetEventsByImpactAsync(EventImpact minImpact);

    /// <summary>
    /// Checks if trading should be restricted due to upcoming events
    /// </summary>
    Task<bool> ShouldRestrictTradingAsync(string symbol, TimeSpan lookAhead);

    /// <summary>
    /// Gets the current trading restriction status
    /// </summary>
    Task<TradingRestriction> GetTradingRestrictionAsync(string symbol);

    /// <summary>
    /// Event fired when a high-impact event is approaching
    /// </summary>
    event EventHandler<EconomicEventAlert>? OnHighImpactEventApproaching;

    /// <summary>
    /// Event fired when trading restrictions change
    /// </summary>
    event EventHandler<TradingRestriction>? OnTradingRestrictionChanged;

    /// <summary>
    /// Initializes the event manager and starts monitoring
    /// </summary>
    Task InitializeAsync();

    /// <summary>
    /// Shuts down the event manager
    /// </summary>
    Task ShutdownAsync();
}

/// <summary>
/// Economic event impact levels
/// </summary>
public enum EventImpact
{
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4
}

/// <summary>
/// Economic event data structure
/// </summary>
public class EconomicEvent
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Country { get; set; } = string.Empty;
    public string Currency { get; set; } = string.Empty;
    public DateTime ScheduledTime { get; set; }
    public EventImpact Impact { get; set; }
    public string Category { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    private readonly List<string> _affectedSymbols = new();
    public IReadOnlyList<string> AffectedSymbols => _affectedSymbols;
    public bool IsActual { get; set; }
    public string? ActualValue { get; set; }
    public string? ForecastValue { get; set; }
    public string? PreviousValue { get; set; }
    
    internal void AddAffectedSymbol(string symbol) => _affectedSymbols.Add(symbol);
}

/// <summary>
/// Trading restriction information
/// </summary>
public class TradingRestriction
{
    public string Symbol { get; set; } = string.Empty;
    public bool IsRestricted { get; set; }
    public string Reason { get; set; } = string.Empty;
    public DateTime RestrictedUntil { get; set; }
    private readonly List<EconomicEvent> _causingEvents = new();
    public IReadOnlyList<EconomicEvent> CausingEvents => _causingEvents;
    public EventImpact MaxImpact { get; set; }
    
    internal void AddCausingEvent(EconomicEvent eventItem) => _causingEvents.Add(eventItem);
}

/// <summary>
/// Economic event alert
/// </summary>
public class EconomicEventAlert
{
    public EconomicEvent Event { get; set; } = new();
    public TimeSpan TimeUntilEvent { get; set; }
    private readonly List<string> _affectedSymbols = new();
    public IReadOnlyList<string> AffectedSymbols => _affectedSymbols;
    public string RecommendedAction { get; set; } = string.Empty;
    
    internal void AddAffectedSymbol(string symbol) => _affectedSymbols.Add(symbol);
}