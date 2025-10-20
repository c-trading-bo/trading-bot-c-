using BotCore.StrategyDsl;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace Tests.Unit.StrategyDsl;

public class DslLoaderTests
{
    private readonly string _tempDirectory;

    public DslLoaderTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"dsl_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDirectory);
    }

    [Fact]
    public async Task LoadStrategiesAsync_WithValidYamlFiles_LoadsStrategiesSuccessfully()
    {
        // Arrange
        var yamlContent = @"
name: TestStrategy
family: test
priority: high
enabled: true
description: ""Test strategy for unit testing""

telemetry_tags:
  - ""Test""
  - ""UnitTest""

regime_filters:
  required:
    - ""market_regime == 'Range'""
  blocked:
    - ""volatility_z_score > 2.0""

zone_conditions:
  required:
    - ""zone.distance_atr <= 0.8""
  preferred:
    - ""zone.strength >= 0.6""

pattern_conditions:
  confluence:
    - ""pattern.kind::Doji""
    - ""pattern.kind::Hammer""

micro_conditions:
  entry_triggers:
    - ""momentum.z_score >= -0.5""
    - ""vwap.distance <= 0.002""

risk_management:
  position_sizing:
    base_size: 1
    confidence_multiplier: true
    max_size: 3
  stop_loss:
    type: ""atr_based""
    distance_atr: 0.8
  take_profit:
    type: ""dynamic""
    initial_target_atr: 1.5

confidence_calculation:
  base_confidence: 0.65
  boosters:
    zone.breakout_score >= 0.8: 0.15
    pattern.bull_score >= 0.7: 0.10
  penalties:
    zone.test_count >= 4: -0.10

direction_logic:
  long_conditions:
    - ""zone.type == 'Support'""
    - ""pattern.bull_score > pattern.bear_score""
  short_conditions:
    - ""zone.type == 'Resistance'""
    - ""pattern.bear_score > pattern.bull_score""

metadata:
  version: ""1.0""
  author: ""TestAuthor""
";

        await File.WriteAllTextAsync(Path.Combine(_tempDirectory, "test_strategy.yml"), yamlContent);

        var options = Options.Create(new DslLoaderOptions
        {
            StrategyFolder = _tempDirectory,
            AutoReload = false
        });

        var loader = new DslLoader(NullLogger<DslLoader>.Instance, options);

        // Act
        var strategies = await loader.LoadStrategiesAsync();

        // Assert
        Assert.Single(strategies);
        var strategy = strategies.First();
        
        Assert.Equal("TestStrategy", strategy.Name);
        Assert.Equal("test", strategy.Family);
        Assert.Equal("high", strategy.Priority);
        Assert.True(strategy.Enabled);
        Assert.Contains("Test", strategy.TelemetryTags);
        Assert.Contains("UnitTest", strategy.TelemetryTags);
        
        Assert.Contains("market_regime == 'Range'", strategy.RegimeFilters.Required);
        Assert.Contains("volatility_z_score > 2.0", strategy.RegimeFilters.Blocked);
        
        Assert.Equal(0.65, strategy.ConfidenceCalculation.BaseConfidence);
        Assert.Equal(1, strategy.RiskManagement.PositionSizing.BaseSize);
        Assert.Equal(3, strategy.RiskManagement.PositionSizing.MaxSize);
    }

    [Fact]
    public async Task LoadStrategiesAsync_WithNonexistentFolder_ReturnsEmptyList()
    {
        // Arrange
        var options = Options.Create(new DslLoaderOptions
        {
            StrategyFolder = "/nonexistent/folder",
            AutoReload = false
        });

        var loader = new DslLoader(NullLogger<DslLoader>.Instance, options);

        // Act
        var strategies = await loader.LoadStrategiesAsync();

        // Assert
        Assert.Empty(strategies);
    }

    [Fact]
    public async Task LoadStrategiesAsync_WithInvalidYaml_LogsErrorAndContinues()
    {
        // Arrange
        var invalidYaml = "invalid: yaml: content: [unclosed";
        await File.WriteAllTextAsync(Path.Combine(_tempDirectory, "invalid.yml"), invalidYaml);

        var validYaml = @"
name: ValidStrategy  
family: test
enabled: true
description: ""Valid strategy""
";
        await File.WriteAllTextAsync(Path.Combine(_tempDirectory, "valid.yml"), validYaml);

        var options = Options.Create(new DslLoaderOptions
        {
            StrategyFolder = _tempDirectory,
            AutoReload = false
        });

        var loader = new DslLoader(NullLogger<DslLoader>.Instance, options);

        // Act
        var strategies = await loader.LoadStrategiesAsync();

        // Assert
        Assert.Single(strategies); // Only the valid strategy should be loaded
        Assert.Equal("ValidStrategy", strategies.First().Name);
    }

    [Fact]
    public async Task GetStrategyAsync_WithExistingStrategy_ReturnsStrategy()
    {
        // Arrange
        var yamlContent = @"
name: SpecificStrategy
family: test
enabled: true
description: ""Specific test strategy""
";
        await File.WriteAllTextAsync(Path.Combine(_tempDirectory, "specific.yml"), yamlContent);

        var options = Options.Create(new DslLoaderOptions
        {
            StrategyFolder = _tempDirectory,
            AutoReload = false
        });

        var loader = new DslLoader(NullLogger<DslLoader>.Instance, options);

        // Act
        var strategy = await loader.GetStrategyAsync("SpecificStrategy");

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("SpecificStrategy", strategy.Name);
    }

    [Fact]
    public async Task GetStrategyAsync_WithNonexistentStrategy_ReturnsNull()
    {
        // Arrange
        var options = Options.Create(new DslLoaderOptions
        {
            StrategyFolder = _tempDirectory,
            AutoReload = false
        });

        var loader = new DslLoader(NullLogger<DslLoader>.Instance, options);

        // Act
        var strategy = await loader.GetStrategyAsync("NonexistentStrategy");

        // Assert
        Assert.Null(strategy);
    }

    [Fact]
    public async Task GetStatsAsync_WithMultipleStrategies_ReturnsCorrectStats()
    {
        // Arrange
        var strategy1 = @"
name: Strategy1
family: family1
enabled: true
description: ""First strategy""
";
        var strategy2 = @"
name: Strategy2  
family: family1
enabled: false
description: ""Second strategy""
";
        var strategy3 = @"
name: Strategy3
family: family2
enabled: true  
description: ""Third strategy""
";

        await File.WriteAllTextAsync(Path.Combine(_tempDirectory, "strategy1.yml"), strategy1);
        await File.WriteAllTextAsync(Path.Combine(_tempDirectory, "strategy2.yml"), strategy2);
        await File.WriteAllTextAsync(Path.Combine(_tempDirectory, "strategy3.yml"), strategy3);

        var options = Options.Create(new DslLoaderOptions
        {
            StrategyFolder = _tempDirectory,
            AutoReload = false
        });

        var loader = new DslLoader(NullLogger<DslLoader>.Instance, options);

        // Act
        var stats = await loader.GetStatsAsync();

        // Assert
        Assert.Equal(3, stats.TotalStrategies);
        Assert.Equal(2, stats.EnabledStrategies);
        Assert.Equal(1, stats.DisabledStrategies);
        Assert.Equal(2, stats.FamilyCounts["family1"]);
        Assert.Equal(1, stats.FamilyCounts["family2"]);
    }

    [Fact]
    public void NeedsReload_WithAutoReloadAndExpiredTime_ReturnsTrue()
    {
        // Arrange
        var options = Options.Create(new DslLoaderOptions
        {
            StrategyFolder = _tempDirectory,
            AutoReload = true,
            ReloadIntervalMinutes = 1
        });

        var loader = new DslLoader(NullLogger<DslLoader>.Instance, options);

        // Act & Assert
        Assert.True(loader.NeedsReload()); // Should need reload initially
    }

    [Fact]
    public void NeedsReload_WithoutAutoReload_ReturnsFalse()
    {
        // Arrange
        var options = Options.Create(new DslLoaderOptions
        {
            StrategyFolder = _tempDirectory,
            AutoReload = false
        });

        var loader = new DslLoader(NullLogger<DslLoader>.Instance, options);

        // Act & Assert
        Assert.False(loader.NeedsReload());
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }
}