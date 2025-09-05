using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Text.Json;

namespace BotCore.Market;

// ================================================================================
// COMPONENT 10: ECONOMIC EVENT FILTER
// ================================================================================

/// <summary>
/// Economic Event Management System for preventing trading during high-impact events
/// Manages economic event monitoring, impact assessment, and trading restrictions
/// </summary>
public class EconomicEventManager : IEconomicEventManager, IDisposable
{
    private readonly ILogger<EconomicEventManager> _logger;
    private readonly List<EconomicEvent> _events = new();
    private readonly ConcurrentDictionary<string, TradingRestriction> _tradingRestrictions = new();
    private readonly Timer _eventMonitor;
    private readonly Timer _restrictionUpdater;
    private readonly object _lockObject = new();
    private bool _disposed = false;
    private bool _initialized = false;

    // Configuration
    private static readonly TimeSpan DEFAULT_LOOKHEAD = TimeSpan.FromHours(2);
    private static readonly TimeSpan CRITICAL_EVENT_BUFFER = TimeSpan.FromMinutes(30);
    private static readonly TimeSpan HIGH_EVENT_BUFFER = TimeSpan.FromMinutes(15);
    private static readonly TimeSpan MEDIUM_EVENT_BUFFER = TimeSpan.FromMinutes(5);

    public event EventHandler<EconomicEventAlert>? OnHighImpactEventApproaching;
    public event EventHandler<TradingRestriction>? OnTradingRestrictionChanged;

    public EconomicEventManager(ILogger<EconomicEventManager> logger)
    {
        _logger = logger;
        
        // Initialize monitoring timers
        _eventMonitor = new Timer(CheckUpcomingEvents, null, Timeout.Infinite, Timeout.Infinite);
        _restrictionUpdater = new Timer(UpdateTradingRestrictions, null, Timeout.Infinite, Timeout.Infinite);
        
        _logger.LogInformation("[EconomicEventManager] Economic Event Filter initialized");
    }

    public async Task InitializeAsync()
    {
        if (_initialized)
        {
            _logger.LogWarning("[EconomicEventManager] Already initialized");
            return;
        }

        try
        {
            _logger.LogInformation("[EconomicEventManager] Initializing economic event monitoring...");

            // Load initial economic events
            await LoadEconomicEventsAsync();

            // Start monitoring timers
            _eventMonitor.Change(TimeSpan.Zero, TimeSpan.FromMinutes(5));
            _restrictionUpdater.Change(TimeSpan.Zero, TimeSpan.FromMinutes(1));

            _initialized = true;
            _logger.LogInformation("[EconomicEventManager] Economic event monitoring started successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[EconomicEventManager] Failed to initialize economic event manager");
            throw;
        }
    }

    public async Task<IEnumerable<EconomicEvent>> GetUpcomingEventsAsync(TimeSpan timeWindow)
    {
        var cutoffTime = DateTime.UtcNow.Add(timeWindow);
        
        lock (_lockObject)
        {
            var upcomingEvents = _events
                .Where(e => e.ScheduledTime >= DateTime.UtcNow && e.ScheduledTime <= cutoffTime)
                .OrderBy(e => e.ScheduledTime)
                .ToList();

            _logger.LogDebug("[EconomicEventManager] Found {Count} upcoming events in next {TimeWindow}", 
                upcomingEvents.Count, timeWindow);

            return upcomingEvents;
        }
    }

    public async Task<IEnumerable<EconomicEvent>> GetEventsByImpactAsync(EventImpact minImpact)
    {
        lock (_lockObject)
        {
            var filteredEvents = _events
                .Where(e => e.Impact >= minImpact && e.ScheduledTime >= DateTime.UtcNow)
                .OrderBy(e => e.ScheduledTime)
                .ToList();

            _logger.LogDebug("[EconomicEventManager] Found {Count} events with impact >= {Impact}", 
                filteredEvents.Count, minImpact);

            return filteredEvents;
        }
    }

    public async Task<bool> ShouldRestrictTradingAsync(string symbol, TimeSpan lookAhead)
    {
        var restriction = await GetTradingRestrictionAsync(symbol);
        
        if (restriction.IsRestricted)
        {
            _logger.LogInformation("[EconomicEventManager] Trading restricted for {Symbol}: {Reason}", 
                symbol, restriction.Reason);
            return true;
        }

        // Check for upcoming high-impact events
        var upcomingEvents = await GetUpcomingEventsAsync(lookAhead);
        var relevantEvents = upcomingEvents.Where(e => 
            e.AffectedSymbols.Contains(symbol) || 
            e.Impact >= EventImpact.Critical ||
            IsSymbolAffectedByEvent(symbol, e)).ToList();

        if (relevantEvents.Any())
        {
            _logger.LogWarning("[EconomicEventManager] {Count} relevant events found for {Symbol} in next {TimeSpan}", 
                relevantEvents.Count, symbol, lookAhead);
            return true;
        }

        return false;
    }

    public async Task<TradingRestriction> GetTradingRestrictionAsync(string symbol)
    {
        if (_tradingRestrictions.TryGetValue(symbol, out var restriction))
        {
            // Check if restriction has expired
            if (restriction.RestrictedUntil <= DateTime.UtcNow)
            {
                _tradingRestrictions.TryRemove(symbol, out _);
                _logger.LogInformation("[EconomicEventManager] Trading restriction expired for {Symbol}", symbol);
                
                var expiredRestriction = new TradingRestriction 
                { 
                    Symbol = symbol, 
                    IsRestricted = false, 
                    Reason = "Restriction expired" 
                };
                
                OnTradingRestrictionChanged?.Invoke(this, expiredRestriction);
                return expiredRestriction;
            }

            return restriction;
        }

        return new TradingRestriction { Symbol = symbol, IsRestricted = false };
    }

    public async Task ShutdownAsync()
    {
        try
        {
            _logger.LogInformation("[EconomicEventManager] Shutting down economic event monitoring...");

            await _eventMonitor.DisposeAsync();
            await _restrictionUpdater.DisposeAsync();

            _initialized = false;
            _logger.LogInformation("[EconomicEventManager] Economic event monitoring stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[EconomicEventManager] Error during shutdown");
        }
    }

    private async Task LoadEconomicEventsAsync()
    {
        try
        {
            // In a real implementation, this would load from external data sources
            // For now, we'll simulate with some common high-impact events
            var mockEvents = GenerateMockEconomicEvents();

            lock (_lockObject)
            {
                _events.Clear();
                _events.AddRange(mockEvents);
            }

            _logger.LogInformation("[EconomicEventManager] Loaded {Count} economic events", mockEvents.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[EconomicEventManager] Failed to load economic events");
        }
    }

    private List<EconomicEvent> GenerateMockEconomicEvents()
    {
        var now = DateTime.UtcNow;
        var events = new List<EconomicEvent>();

        // Generate some upcoming events for testing
        var eventTemplates = new[]
        {
            new { Name = "FOMC Interest Rate Decision", Impact = EventImpact.Critical, Currency = "USD", Category = "Monetary Policy" },
            new { Name = "Non-Farm Payrolls", Impact = EventImpact.High, Currency = "USD", Category = "Employment" },
            new { Name = "CPI Inflation Data", Impact = EventImpact.High, Currency = "USD", Category = "Inflation" },
            new { Name = "GDP Growth Rate", Impact = EventImpact.Medium, Currency = "USD", Category = "Economic Growth" },
            new { Name = "Unemployment Rate", Impact = EventImpact.Medium, Currency = "USD", Category = "Employment" },
            new { Name = "Retail Sales", Impact = EventImpact.Medium, Currency = "USD", Category = "Consumer Spending" }
        };

        for (int i = 0; i < eventTemplates.Length; i++)
        {
            var template = eventTemplates[i];
            var eventTime = now.AddHours(2 + i * 6); // Space events 6 hours apart

            events.Add(new EconomicEvent
            {
                Id = Guid.NewGuid().ToString(),
                Name = template.Name,
                Country = "United States",
                Currency = template.Currency,
                ScheduledTime = eventTime,
                Impact = template.Impact,
                Category = template.Category,
                Description = $"Scheduled {template.Name} release",
                AffectedSymbols = GetAffectedSymbols(template.Currency, template.Impact),
                IsActual = false
            });
        }

        return events;
    }

    private List<string> GetAffectedSymbols(string currency, EventImpact impact)
    {
        var symbols = new List<string>();

        // Major USD-based instruments
        if (currency == "USD")
        {
            symbols.AddRange(new[] { "ES", "NQ", "YM", "RTY", "EURUSD", "GBPUSD", "USDJPY" });

            if (impact >= EventImpact.High)
            {
                symbols.AddRange(new[] { "DXY", "GC", "SI", "CL" }); // Dollar index, gold, silver, oil
            }
        }

        return symbols;
    }

    private bool IsSymbolAffectedByEvent(string symbol, EconomicEvent economicEvent)
    {
        // Basic logic to determine if a symbol is affected by an event
        if (economicEvent.AffectedSymbols.Contains(symbol))
            return true;

        // ES and NQ are affected by major USD events
        if ((symbol == "ES" || symbol == "NQ") && economicEvent.Currency == "USD" && economicEvent.Impact >= EventImpact.High)
            return true;

        // Forex pairs affected by their base/quote currencies
        if (symbol.Contains(economicEvent.Currency))
            return true;

        return false;
    }

    private void CheckUpcomingEvents(object? state)
    {
        try
        {
            var now = DateTime.UtcNow;
            var upcomingEvents = _events
                .Where(e => e.ScheduledTime > now && e.ScheduledTime <= now.Add(DEFAULT_LOOKHEAD))
                .Where(e => e.Impact >= EventImpact.High)
                .ToList();

            foreach (var economicEvent in upcomingEvents)
            {
                var timeUntilEvent = economicEvent.ScheduledTime - now;
                var shouldAlert = ShouldAlertForEvent(economicEvent, timeUntilEvent);

                if (shouldAlert)
                {
                    var alert = new EconomicEventAlert
                    {
                        Event = economicEvent,
                        TimeUntilEvent = timeUntilEvent,
                        AffectedSymbols = economicEvent.AffectedSymbols,
                        RecommendedAction = GetRecommendedAction(economicEvent)
                    };

                    _logger.LogWarning("[EconomicEventManager] High-impact event approaching: {Name} in {TimeUntil}", 
                        economicEvent.Name, timeUntilEvent);

                    OnHighImpactEventApproaching?.Invoke(this, alert);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[EconomicEventManager] Error checking upcoming events");
        }
    }

    private bool ShouldAlertForEvent(EconomicEvent economicEvent, TimeSpan timeUntilEvent)
    {
        return economicEvent.Impact switch
        {
            EventImpact.Critical => timeUntilEvent <= CRITICAL_EVENT_BUFFER,
            EventImpact.High => timeUntilEvent <= HIGH_EVENT_BUFFER,
            EventImpact.Medium => timeUntilEvent <= MEDIUM_EVENT_BUFFER,
            _ => false
        };
    }

    private string GetRecommendedAction(EconomicEvent economicEvent)
    {
        return economicEvent.Impact switch
        {
            EventImpact.Critical => "HALT_TRADING",
            EventImpact.High => "REDUCE_POSITION_SIZE",
            EventImpact.Medium => "INCREASE_MONITORING",
            _ => "CONTINUE_NORMAL"
        };
    }

    private void UpdateTradingRestrictions(object? state)
    {
        try
        {
            var now = DateTime.UtcNow;
            var affectedSymbols = new HashSet<string>();

            // Collect all symbols that should be restricted
            foreach (var economicEvent in _events.Where(e => e.ScheduledTime >= now))
            {
                var timeUntilEvent = economicEvent.ScheduledTime - now;
                var shouldRestrict = ShouldRestrictForEvent(economicEvent, timeUntilEvent);

                if (shouldRestrict)
                {
                    foreach (var symbol in economicEvent.AffectedSymbols)
                    {
                        affectedSymbols.Add(symbol);
                        UpdateRestrictionForSymbol(symbol, economicEvent, timeUntilEvent);
                    }
                }
            }

            // Remove expired restrictions
            var expiredSymbols = _tradingRestrictions.Keys
                .Where(symbol => !affectedSymbols.Contains(symbol))
                .ToList();

            foreach (var symbol in expiredSymbols)
            {
                if (_tradingRestrictions.TryRemove(symbol, out var expiredRestriction))
                {
                    _logger.LogInformation("[EconomicEventManager] Removed trading restriction for {Symbol}", symbol);
                    
                    var newRestriction = new TradingRestriction 
                    { 
                        Symbol = symbol, 
                        IsRestricted = false, 
                        Reason = "No upcoming high-impact events" 
                    };
                    
                    OnTradingRestrictionChanged?.Invoke(this, newRestriction);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[EconomicEventManager] Error updating trading restrictions");
        }
    }

    private bool ShouldRestrictForEvent(EconomicEvent economicEvent, TimeSpan timeUntilEvent)
    {
        return economicEvent.Impact switch
        {
            EventImpact.Critical => timeUntilEvent <= CRITICAL_EVENT_BUFFER,
            EventImpact.High => timeUntilEvent <= HIGH_EVENT_BUFFER,
            _ => false
        };
    }

    private void UpdateRestrictionForSymbol(string symbol, EconomicEvent economicEvent, TimeSpan timeUntilEvent)
    {
        var restrictedUntil = economicEvent.ScheduledTime.Add(GetPostEventBuffer(economicEvent.Impact));
        
        var restriction = new TradingRestriction
        {
            Symbol = symbol,
            IsRestricted = true,
            Reason = $"High-impact event: {economicEvent.Name}",
            RestrictedUntil = restrictedUntil,
            CausingEvents = new List<EconomicEvent> { economicEvent },
            MaxImpact = economicEvent.Impact
        };

        var isNewRestriction = !_tradingRestrictions.ContainsKey(symbol);
        _tradingRestrictions.AddOrUpdate(symbol, restriction, (key, existing) => restriction);

        if (isNewRestriction)
        {
            _logger.LogWarning("[EconomicEventManager] Trading restricted for {Symbol} until {Until} due to {Event}", 
                symbol, restrictedUntil, economicEvent.Name);
            
            OnTradingRestrictionChanged?.Invoke(this, restriction);
        }
    }

    private TimeSpan GetPostEventBuffer(EventImpact impact)
    {
        return impact switch
        {
            EventImpact.Critical => TimeSpan.FromMinutes(60),
            EventImpact.High => TimeSpan.FromMinutes(30),
            EventImpact.Medium => TimeSpan.FromMinutes(10),
            _ => TimeSpan.FromMinutes(5)
        };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            try
            {
                _eventMonitor?.Dispose();
                _restrictionUpdater?.Dispose();
                _disposed = true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "[EconomicEventManager] Error during disposal");
            }
        }
    }
}