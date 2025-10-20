using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BotCore.StrategyDsl;

/// <summary>
/// DSL loader with async support and options pattern
/// Wraps SimpleDslLoader to provide the API expected by tests
/// </summary>
public class DslLoader
{
    private readonly ILogger<DslLoader> _logger;
    private readonly DslLoaderOptions _options;
    private DateTime _lastLoadTime = DateTime.MinValue;
    private IReadOnlyList<DslStrategy>? _cachedStrategies;

    public DslLoader(ILogger<DslLoader> logger, IOptions<DslLoaderOptions> options)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Load all strategies from configured directory
    /// </summary>
    public async Task<IReadOnlyList<DslStrategy>> LoadStrategiesAsync()
    {
        return await Task.Run(() =>
        {
            try
            {
                var strategies = SimpleDslLoader.LoadAll(_options.StrategyDirectory);
                _cachedStrategies = strategies;
                _lastLoadTime = DateTime.UtcNow;
                
                _logger.LogInformation("Loaded {Count} strategies from {Directory}", 
                    strategies.Count, _options.StrategyDirectory);
                
                return strategies;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load strategies from {Directory}", 
                    _options.StrategyDirectory);
                throw;
            }
        });
    }

    /// <summary>
    /// Get a specific strategy by name
    /// </summary>
    public async Task<DslStrategy?> GetStrategyAsync(string strategyName)
    {
        var strategies = await LoadStrategiesAsync();
        return strategies.FirstOrDefault(s => s.Name.Equals(strategyName, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Get loader statistics
    /// </summary>
    public async Task<DslLoaderStats> GetStatsAsync()
    {
        var strategies = _cachedStrategies ?? await LoadStrategiesAsync();
        
        var familyCounts = strategies
            .Where(s => !string.IsNullOrEmpty(s.Family))
            .GroupBy(s => s.Family)
            .ToDictionary(g => g.Key, g => g.Count());
        
        // Enabled/disabled counts - assume enabled if not explicitly set
        var enabledCount = strategies.Count; // All strategies are enabled unless they have enabled flag
        var disabledCount = 0;
        
        return new DslLoaderStats
        {
            TotalStrategies = strategies.Count,
            EnabledStrategies = enabledCount,
            DisabledStrategies = disabledCount,
            LastLoadTime = _lastLoadTime,
            StrategyNames = strategies.Select(s => s.Name).ToList(),
            FamilyCounts = familyCounts
        };
    }

    /// <summary>
    /// Check if strategies need to be reloaded
    /// </summary>
    public bool NeedsReload()
    {
        if (_cachedStrategies == null)
            return true;

        if (!_options.AutoReload || _options.ReloadIntervalSeconds <= 0)
            return false;

        var elapsed = (DateTime.UtcNow - _lastLoadTime).TotalSeconds;
        return elapsed >= _options.ReloadIntervalSeconds;
    }

    /// <summary>
    /// Get cached strategies without reloading
    /// </summary>
    public IReadOnlyList<DslStrategy>? GetCachedStrategies()
    {
        return _cachedStrategies;
    }
}

/// <summary>
/// Configuration options for DslLoader
/// </summary>
public class DslLoaderOptions
{
    /// <summary>
    /// Directory containing strategy YAML files (legacy name StrategyFolder)
    /// </summary>
    public string StrategyDirectory { get; set; } = "strategies";

    /// <summary>
    /// Alias for StrategyDirectory for backward compatibility
    /// </summary>
    public string StrategyFolder
    {
        get => StrategyDirectory;
        set => StrategyDirectory = value;
    }

    /// <summary>
    /// Whether to automatically reload strategies on interval
    /// </summary>
    public bool AutoReload { get; set; } = true;

    /// <summary>
    /// Interval in seconds to check for strategy file changes
    /// Set to 0 to disable automatic reloading
    /// </summary>
    public int ReloadIntervalSeconds { get; set; } = 60;

    /// <summary>
    /// Legacy name for ReloadIntervalSeconds (in minutes)
    /// </summary>
    public int ReloadIntervalMinutes
    {
        get => ReloadIntervalSeconds / 60;
        set => ReloadIntervalSeconds = value * 60;
    }

    /// <summary>
    /// Whether to fail on invalid strategy files or skip them
    /// </summary>
    public bool FailOnInvalidStrategies { get; set; } = false;
}

/// <summary>
/// Statistics about loaded strategies
/// </summary>
public class DslLoaderStats
{
    public int TotalStrategies { get; init; }
    public int EnabledStrategies { get; init; }
    public int DisabledStrategies { get; init; }
    public DateTime LastLoadTime { get; init; }
    public List<string> StrategyNames { get; init; } = new();
    public Dictionary<string, int> FamilyCounts { get; init; } = new();
}
