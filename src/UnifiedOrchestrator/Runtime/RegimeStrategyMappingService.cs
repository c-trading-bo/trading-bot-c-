using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Collections.Concurrent;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Regime-Strategy Mapping Service for autonomous strategy selection
/// Connects sophisticated regime detection to strategy filtering for institutional trading
/// </summary>
public class RegimeStrategyMappingService
{
    private readonly ILogger<RegimeStrategyMappingService> _logger;
    private readonly ConcurrentDictionary<string, RegimeDefinition> _regimes;
    private readonly ConcurrentDictionary<string, StrategyDefinition> _strategies;
    private readonly Dictionary<string, HashSet<string>> _regimeStrategyMatrix;
    private readonly object _matrixLock = new();
    private readonly string _configFile;

    public RegimeStrategyMappingService(ILogger<RegimeStrategyMappingService> logger)
    {
        _logger = logger;
        _regimes = new ConcurrentDictionary<string, RegimeDefinition>();
        _strategies = new ConcurrentDictionary<string, StrategyDefinition>();
        _regimeStrategyMatrix = new Dictionary<string, HashSet<string>>();
        
        var configDir = Path.Combine(Directory.GetCurrentDirectory(), "config");
        Directory.CreateDirectory(configDir);
        _configFile = Path.Combine(configDir, "regime-strategy-matrix.json");
        
        LoadRegimeStrategyMatrix();
        _logger.LogInformation("🎯 [REGIME-STRATEGY-MAPPING] Initialized with {RegimeCount} regimes and {StrategyCount} strategies", 
            _regimes.Count, _strategies.Count);
    }

    /// <summary>
    /// Check if a strategy is allowed in the current market regime
    /// </summary>
    public bool IsStrategyAllowedInRegime(string strategyName, string currentRegime, decimal confidence = 0.5m)
    {
        if (string.IsNullOrEmpty(currentRegime))
        {
            _logger.LogDebug("🤔 [REGIME-STRATEGY-MAPPING] No regime detected, allowing strategy {Strategy}", strategyName);
            return true; // Allow all strategies if regime is unknown
        }

        lock (_matrixLock)
        {
            if (_regimeStrategyMatrix.TryGetValue(currentRegime, out var allowedStrategies))
            {
                var isAllowed = allowedStrategies.Contains(strategyName);
                
                if (!isAllowed)
                {
                    _logger.LogWarning("🚫 [REGIME-STRATEGY-MAPPING] Strategy {Strategy} BLOCKED in regime {Regime} (confidence: {Confidence:F3})", 
                        strategyName, currentRegime, confidence);
                }
                else
                {
                    _logger.LogDebug("✅ [REGIME-STRATEGY-MAPPING] Strategy {Strategy} ALLOWED in regime {Regime} (confidence: {Confidence:F3})", 
                        strategyName, currentRegime, confidence);
                }
                
                return isAllowed;
            }
        }

        _logger.LogWarning("⚠️ [REGIME-STRATEGY-MAPPING] Unknown regime {Regime}, defaulting to ALLOW for strategy {Strategy}", 
            currentRegime, strategyName);
        return true; // Default to allowing unknown regimes
    }

    /// <summary>
    /// Get allowed strategies for a specific regime
    /// </summary>
    public IReadOnlyList<string> GetAllowedStrategiesForRegime(string regimeName)
    {
        lock (_matrixLock)
        {
            if (_regimeStrategyMatrix.TryGetValue(regimeName, out var strategies))
            {
                return strategies.ToList().AsReadOnly();
            }
        }
        
        return new List<string>().AsReadOnly();
    }

    /// <summary>
    /// Get all strategies that should be blocked in current regime
    /// </summary>
    public IReadOnlyList<string> GetBlockedStrategiesForRegime(string regimeName)
    {
        var allStrategies = _strategies.Keys.ToHashSet();
        var allowedStrategies = GetAllowedStrategiesForRegime(regimeName).ToHashSet();
        
        return allStrategies.Except(allowedStrategies).ToList().AsReadOnly();
    }

    /// <summary>
    /// Update regime-strategy compatibility in real-time
    /// </summary>
    public void UpdateRegimeStrategyMapping(string regimeName, string strategyName, bool isAllowed)
    {
        lock (_matrixLock)
        {
            if (!_regimeStrategyMatrix.TryGetValue(regimeName, out var strategies))
            {
                strategies = new HashSet<string>();
                _regimeStrategyMatrix[regimeName] = strategies;
            }

            if (isAllowed)
            {
                strategies.Add(strategyName);
                _logger.LogInformation("➕ [REGIME-STRATEGY-MAPPING] Added {Strategy} to allowed list for regime {Regime}", 
                    strategyName, regimeName);
            }
            else
            {
                strategies.Remove(strategyName);
                _logger.LogInformation("➖ [REGIME-STRATEGY-MAPPING] Removed {Strategy} from allowed list for regime {Regime}", 
                    strategyName, regimeName);
            }
        }

        // Save updated matrix
        SaveRegimeStrategyMatrix();
    }

    /// <summary>
    /// Get regime compatibility statistics
    /// </summary>
    public RegimeStrategyStats GetStats()
    {
        lock (_matrixLock)
        {
            var stats = new RegimeStrategyStats
            {
                TotalRegimes = _regimes.Count,
                TotalStrategies = _strategies.Count,
                TotalMappings = _regimeStrategyMatrix.Sum(kvp => kvp.Value.Count),
                RegimeCompatibility = new Dictionary<string, int>()
            };

            foreach (var (regime, strategies) in _regimeStrategyMatrix)
            {
                stats.RegimeCompatibility[regime] = strategies.Count;
            }

            return stats;
        }
    }

    /// <summary>
    /// Load regime-strategy matrix from configuration
    /// </summary>
    private void LoadRegimeStrategyMatrix()
    {
        try
        {
            if (!File.Exists(_configFile))
            {
                CreateDefaultRegimeStrategyMatrix();
                return;
            }

            var json = File.ReadAllText(_configFile);
            var config = JsonSerializer.Deserialize<RegimeStrategyConfig>(json);
            
            if (config != null)
            {
                // Load regimes
                foreach (var regime in config.Regimes)
                {
                    _regimes.TryAdd(regime.Key, regime.Value);
                }

                // Load strategies
                foreach (var strategy in config.Strategies)
                {
                    _strategies.TryAdd(strategy.Key, strategy.Value);
                }

                // Load matrix
                lock (_matrixLock)
                {
                    foreach (var mapping in config.RegimeStrategyMatrix)
                    {
                        _regimeStrategyMatrix[mapping.Key] = new HashSet<string>(mapping.Value);
                    }
                }

                _logger.LogInformation("📂 [REGIME-STRATEGY-MAPPING] Loaded configuration from {ConfigFile}", _configFile);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [REGIME-STRATEGY-MAPPING] Failed to load configuration, creating defaults");
            CreateDefaultRegimeStrategyMatrix();
        }
    }

    /// <summary>
    /// Create default regime-strategy matrix for production use
    /// </summary>
    private void CreateDefaultRegimeStrategyMatrix()
    {
        // Define market regimes
        var regimes = new Dictionary<string, RegimeDefinition>
        {
            ["trending_bull"] = new RegimeDefinition
            {
                Name = "Trending Bull",
                Description = "Strong upward momentum with increasing volume",
                Indicators = new[] { "momentum > 0.7", "volatility < 0.3", "volume_trend > 1.2" },
                TypicalDuration = "2-6 hours",
                Confidence = 0.8m
            },
            ["trending_bear"] = new RegimeDefinition
            {
                Name = "Trending Bear", 
                Description = "Strong downward momentum with fear indicators",
                Indicators = new[] { "momentum < -0.7", "volatility < 0.4", "vix > 20" },
                TypicalDuration = "1-4 hours",
                Confidence = 0.8m
            },
            ["range_bound"] = new RegimeDefinition
            {
                Name = "Range Bound",
                Description = "Sideways consolidation with mean reversion behavior", 
                Indicators = new[] { "momentum < 0.3", "volatility < 0.2", "atr_ratio < 1.1" },
                TypicalDuration = "4-8 hours",
                Confidence = 0.7m
            },
            ["high_volatility"] = new RegimeDefinition
            {
                Name = "High Volatility",
                Description = "Elevated volatility with unpredictable movements",
                Indicators = new[] { "volatility > 0.5", "atr_ratio > 1.8", "gap_size > 0.5%" },
                TypicalDuration = "30 minutes - 2 hours", 
                Confidence = 0.9m
            },
            ["breakout"] = new RegimeDefinition
            {
                Name = "Breakout",
                Description = "Breaking through key resistance/support levels",
                Indicators = new[] { "volume > 1.5x avg", "momentum > 0.6", "level_break = true" },
                TypicalDuration = "15 minutes - 1 hour",
                Confidence = 0.85m
            },
            ["exhaustion"] = new RegimeDefinition
            {
                Name = "Exhaustion",
                Description = "End of trend with declining momentum and volume",
                Indicators = new[] { "momentum declining", "volume declining", "rsi > 70 or < 30" },
                TypicalDuration = "30 minutes - 2 hours",
                Confidence = 0.75m
            }
        };

        // Define trading strategies
        var strategies = new Dictionary<string, StrategyDefinition>
        {
            ["S2"] = new StrategyDefinition
            {
                Name = "VWAP Mean Reversion",
                Description = "Mean reversion strategy using VWAP as anchor",
                OptimalRegimes = new[] { "range_bound", "exhaustion" },
                AvoidRegimes = new[] { "trending_bull", "trending_bear", "breakout" },
                RiskMultiplier = 1.0m,
                MaxPositionSize = 3
            },
            ["S3"] = new StrategyDefinition
            {
                Name = "Bollinger Compression",
                Description = "Volatility compression breakout strategy",
                OptimalRegimes = new[] { "breakout", "high_volatility" },
                AvoidRegimes = new[] { "range_bound" },
                RiskMultiplier = 1.2m,
                MaxPositionSize = 2
            },
            ["S6"] = new StrategyDefinition
            {
                Name = "Opening Drive",
                Description = "Momentum strategy for market open periods",
                OptimalRegimes = new[] { "trending_bull", "trending_bear", "breakout" },
                AvoidRegimes = new[] { "range_bound", "exhaustion" },
                RiskMultiplier = 0.8m,
                MaxPositionSize = 4
            },
            ["S11"] = new StrategyDefinition
            {
                Name = "Afternoon Fade",
                Description = "Counter-trend strategy for late-day reversals",
                OptimalRegimes = new[] { "exhaustion", "high_volatility" },
                AvoidRegimes = new[] { "trending_bull", "trending_bear" },
                RiskMultiplier = 1.1m,
                MaxPositionSize = 2
            }
        };

        // Define regime-strategy matrix (which strategies are allowed in which regimes)
        var matrix = new Dictionary<string, string[]>
        {
            ["trending_bull"] = new[] { "S6" }, // Only momentum strategies in strong trends
            ["trending_bear"] = new[] { "S6" }, // Only momentum strategies in strong trends  
            ["range_bound"] = new[] { "S2", "S11" }, // Mean reversion strategies in sideways markets
            ["high_volatility"] = new[] { "S3", "S11" }, // Volatility and counter-trend strategies
            ["breakout"] = new[] { "S3", "S6" }, // Breakout and momentum strategies
            ["exhaustion"] = new[] { "S2", "S11" } // Mean reversion and fade strategies
        };

        // Store in class properties
        foreach (var (key, regime) in regimes)
        {
            _regimes.TryAdd(key, regime);
        }

        foreach (var (key, strategy) in strategies)
        {
            _strategies.TryAdd(key, strategy);
        }

        lock (_matrixLock)
        {
            foreach (var (regime, strategyList) in matrix)
            {
                _regimeStrategyMatrix[regime] = new HashSet<string>(strategyList);
            }
        }

        // Save to file
        SaveRegimeStrategyMatrix();
        
        _logger.LogInformation("✨ [REGIME-STRATEGY-MAPPING] Created default regime-strategy matrix with {RegimeCount} regimes and {StrategyCount} strategies",
            regimes.Count, strategies.Count);
    }

    /// <summary>
    /// Save regime-strategy matrix to configuration file
    /// </summary>
    private void SaveRegimeStrategyMatrix()
    {
        try
        {
            var config = new RegimeStrategyConfig
            {
                Regimes = _regimes.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                Strategies = _strategies.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                RegimeStrategyMatrix = new Dictionary<string, string[]>(),
                LastUpdated = DateTime.UtcNow,
                Version = "2.0"
            };

            lock (_matrixLock)
            {
                foreach (var (regime, strategies) in _regimeStrategyMatrix)
                {
                    config.RegimeStrategyMatrix[regime] = strategies.ToArray();
                }
            }

            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(_configFile, json);

            _logger.LogDebug("💾 [REGIME-STRATEGY-MAPPING] Saved configuration to {ConfigFile}", _configFile);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [REGIME-STRATEGY-MAPPING] Failed to save configuration");
        }
    }
}

/// <summary>
/// Market regime definition
/// </summary>
public class RegimeDefinition
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string[] Indicators { get; set; } = Array.Empty<string>();
    public string TypicalDuration { get; set; } = string.Empty;
    public decimal Confidence { get; set; }
}

/// <summary>
/// Trading strategy definition
/// </summary>
public class StrategyDefinition
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string[] OptimalRegimes { get; set; } = Array.Empty<string>();
    public string[] AvoidRegimes { get; set; } = Array.Empty<string>();
    public decimal RiskMultiplier { get; set; } = 1.0m;
    public int MaxPositionSize { get; set; } = 1;
}

/// <summary>
/// Configuration structure for persistence
/// </summary>
public class RegimeStrategyConfig
{
    public Dictionary<string, RegimeDefinition> Regimes { get; set; } = new();
    public Dictionary<string, StrategyDefinition> Strategies { get; set; } = new();
    public Dictionary<string, string[]> RegimeStrategyMatrix { get; set; } = new();
    public DateTime LastUpdated { get; set; }
    public string Version { get; set; } = "2.0";
}

/// <summary>
/// Statistics for monitoring
/// </summary>
public class RegimeStrategyStats
{
    public int TotalRegimes { get; set; }
    public int TotalStrategies { get; set; }
    public int TotalMappings { get; set; }
    public Dictionary<string, int> RegimeCompatibility { get; set; } = new();
}