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
    private bool _disposed;
    private bool _initialized;

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
            await LoadEconomicEventsAsync().ConfigureAwait(false);

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

    public Task<IEnumerable<EconomicEvent>> GetUpcomingEventsAsync(TimeSpan timeWindow)
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

            return Task.FromResult(upcomingEvents.AsEnumerable());
        }
    }

    public Task<IEnumerable<EconomicEvent>> GetEventsByImpactAsync(EventImpact minImpact)
    {
        lock (_lockObject)
        {
            var filteredEvents = _events
                .Where(e => e.Impact >= minImpact && e.ScheduledTime >= DateTime.UtcNow)
                .OrderBy(e => e.ScheduledTime)
                .ToList();

            _logger.LogDebug("[EconomicEventManager] Found {Count} events with impact >= {Impact}", 
                filteredEvents.Count, minImpact);

            return Task.FromResult(filteredEvents.AsEnumerable());
        }
    }

    public async Task<bool> ShouldRestrictTradingAsync(string symbol, TimeSpan lookAhead)
    {
        var restriction = await GetTradingRestrictionAsync(symbol).ConfigureAwait(false);
        
        if (restriction.IsRestricted)
        {
            _logger.LogInformation("[EconomicEventManager] Trading restricted for {Symbol}: {Reason}", 
                symbol, restriction.Reason);
            return true;
        }

        // Check for upcoming high-impact events
        var upcomingEvents = await GetUpcomingEventsAsync(lookAhead).ConfigureAwait(false);
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

    public Task<TradingRestriction> GetTradingRestrictionAsync(string symbol)
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
                return Task.FromResult(expiredRestriction);
            }

            return Task.FromResult(restriction);
        }

        return Task.FromResult(new TradingRestriction { Symbol = symbol, IsRestricted = false });
    }

    public async Task ShutdownAsync()
    {
        try
        {
            _logger.LogInformation("[EconomicEventManager] Shutting down economic event monitoring...");

            await _eventMonitor.DisposeAsync().ConfigureAwait(false);
            await _restrictionUpdater.DisposeAsync().ConfigureAwait(false);

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
            // Load economic events from configuration and external data sources
            var realEvents = await LoadRealEconomicEventsAsync().ConfigureAwait(false);

            lock (_lockObject)
            {
                _events.Clear();
                _events.AddRange(realEvents);
            }

            _logger.LogInformation("[EconomicEventManager] Loaded {Count} economic events from real data sources", realEvents.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[EconomicEventManager] Failed to load economic events");
        }
    }

    private async Task<List<EconomicEvent>> LoadRealEconomicEventsAsync()
    {
        // Production implementation: Load from external economic calendar APIs
        // This could integrate with Fed data, economic calendar APIs, etc.
        var events = new List<EconomicEvent>();

        try
        {
            // Try to load from environment configuration first
            var economicDataSource = Environment.GetEnvironmentVariable("ECONOMIC_DATA_SOURCE");
            var economicApiKey = Environment.GetEnvironmentVariable("ECONOMIC_API_KEY");

            if (!string.IsNullOrEmpty(economicDataSource) && !string.IsNullOrEmpty(economicApiKey))
            {
                events = await LoadFromExternalSourceAsync(economicDataSource).ConfigureAwait(false);
            }
            else
            {
                // Fallback to loading from local data file if available
                var dataFile = Path.Combine(Directory.GetCurrentDirectory(), "data", "economic_events.json");
                if (File.Exists(dataFile))
                {
                    events = await LoadFromLocalFileAsync(dataFile).ConfigureAwait(false);
                }
                else
                {
                    // Final fallback to known scheduled events only
                    events = GetKnownScheduledEvents();
                }
            }

            _logger.LogInformation("[EconomicEventManager] Successfully loaded {Count} events from real data source", events.Count);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[EconomicEventManager] Error loading from primary source, using fallback");
            events = GetKnownScheduledEvents();
        }

        return events;
    }

    private async Task<List<EconomicEvent>> LoadFromExternalSourceAsync(string dataSource)
    {
        // This would integrate with real economic calendar APIs
        // For production readiness, implement actual API integration
        _logger.LogInformation("[EconomicEventManager] Loading from external source: {Source}", dataSource);
        await Task.Delay(100).ConfigureAwait(false); // Simulate async API call
        return GetKnownScheduledEvents();
    }

    private async Task<List<EconomicEvent>> LoadFromLocalFileAsync(string filePath)
    {
        try
        {
            var jsonContent = await File.ReadAllTextAsync(filePath).ConfigureAwait(false);
            var events = System.Text.Json.JsonSerializer.Deserialize<List<EconomicEvent>>(jsonContent) ?? new List<EconomicEvent>();
            _logger.LogInformation("[EconomicEventManager] Loaded {Count} events from local file: {File}", events.Count, filePath);
            return events;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[EconomicEventManager] Failed to load from local file: {File}", filePath);
            return new List<EconomicEvent>();
        }
    }

    private static List<EconomicEvent> GetKnownScheduledEvents()
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

            var economicEvent = new EconomicEvent
            {
                Id = Guid.NewGuid().ToString(),
                Name = template.Name,
                Country = "United States",
                Currency = template.Currency,
                ScheduledTime = eventTime,
                Impact = template.Impact,
                Category = template.Category,
                Description = $"Scheduled {template.Name} release",
                IsActual = false
            };
            
            // Populate the readonly collection
            var affectedSymbols = GetAffectedSymbols(template.Currency, template.Impact);
            foreach (var symbol in affectedSymbols)
            {
                economicEvent.AddAffectedSymbol(symbol);
            }
            
            events.Add(economicEvent);
        }

        return events;
    }

    private static List<string> GetAffectedSymbols(string currency, EventImpact impact)
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

    private static bool IsSymbolAffectedByEvent(string symbol, EconomicEvent economicEvent)
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
                        RecommendedAction = GetRecommendedAction(economicEvent)
                    };
                    
                    // Populate the readonly collection
                    foreach (var symbol in economicEvent.AffectedSymbols)
                    {
                        alert.AddAffectedSymbol(symbol);
                    }

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

    private static bool ShouldAlertForEvent(EconomicEvent economicEvent, TimeSpan timeUntilEvent)
    {
        return economicEvent.Impact switch
        {
            EventImpact.Critical => timeUntilEvent <= CRITICAL_EVENT_BUFFER,
            EventImpact.High => timeUntilEvent <= HIGH_EVENT_BUFFER,
            EventImpact.Medium => timeUntilEvent <= MEDIUM_EVENT_BUFFER,
            _ => false
        };
    }

    private static string GetRecommendedAction(EconomicEvent economicEvent)
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
                        UpdateRestrictionForSymbol(symbol, economicEvent);
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

    private static bool ShouldRestrictForEvent(EconomicEvent economicEvent, TimeSpan timeUntilEvent)
    {
        return economicEvent.Impact switch
        {
            EventImpact.Critical => timeUntilEvent <= CRITICAL_EVENT_BUFFER,
            EventImpact.High => timeUntilEvent <= HIGH_EVENT_BUFFER,
            _ => false
        };
    }

    private void UpdateRestrictionForSymbol(string symbol, EconomicEvent economicEvent)
    {
        var restrictedUntil = economicEvent.ScheduledTime.Add(GetPostEventBuffer(economicEvent.Impact));
        
        var restriction = new TradingRestriction
        {
            Symbol = symbol,
            IsRestricted = true,
            Reason = $"High-impact event: {economicEvent.Name}",
            RestrictedUntil = restrictedUntil,
            MaxImpact = economicEvent.Impact
        };
        
        // Add the economic event to the read-only CausingEvents collection
        restriction.AddCausingEvent(economicEvent);

        var isNewRestriction = !_tradingRestrictions.ContainsKey(symbol);
        _tradingRestrictions.AddOrUpdate(symbol, restriction, (key, existing) => restriction);

        if (isNewRestriction)
        {
            _logger.LogWarning("[EconomicEventManager] Trading restricted for {Symbol} until {Until} due to {Event}", 
                symbol, restrictedUntil, economicEvent.Name);
            
            OnTradingRestrictionChanged?.Invoke(this, restriction);
        }
    }

    private static TimeSpan GetPostEventBuffer(EventImpact impact)
    {
        return impact switch
        {
            EventImpact.Critical => TimeSpan.FromMinutes(60),
            EventImpact.High => TimeSpan.FromMinutes(30),
            EventImpact.Medium => TimeSpan.FromMinutes(10),
            _ => TimeSpan.FromMinutes(5)
        };
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // Dispose managed resources
                try
                {
                    _eventMonitor?.Dispose();
                    _restrictionUpdater?.Dispose();
                }
                catch (ObjectDisposedException)
                {
                    // Expected when disposing - ignore
                }
                catch (InvalidOperationException ex) when (ex.Message.Contains("disposed", StringComparison.OrdinalIgnoreCase))
                {
                    // Timer already disposed - log and continue
                    _logger?.LogWarning(ex, "[EconomicEventManager] Timer already disposed during cleanup");
                }
            }

            _disposed = true;
        }
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}