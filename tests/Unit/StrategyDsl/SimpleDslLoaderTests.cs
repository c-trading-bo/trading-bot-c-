using Xunit;
using BotCore.StrategyDsl;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System;

namespace BotCore.Tests.StrategyDsl;

/// <summary>
/// Unit tests for SimpleDslLoader - validates YAML strategy file loading and parsing
/// </summary>
public sealed class SimpleDslLoaderTests
{
    [Fact]
    public void LoadAll_ValidYamlFiles_ReturnsStrategies()
    {
        // Arrange
        var tempDir = CreateTempDirectoryWithValidStrategies();

        try
        {
            // Act
            var strategies = SimpleDslLoader.LoadAll(tempDir);

            // Assert
            Assert.NotNull(strategies);
            Assert.Equal(2, strategies.Count);
            
            var s2Strategy = strategies.First(s => s.Name == "S2");
            Assert.Equal("MeanReversion", s2Strategy.Label);
            Assert.Equal("RangeFade", s2Strategy.Family);
            Assert.Equal("both", s2Strategy.Bias);
            Assert.NotNull(s2Strategy.When);
            Assert.Contains("Range", s2Strategy.When.Regime!);
            Assert.Contains("LowVol", s2Strategy.When.Regime!);
            Assert.NotEmpty(s2Strategy.When.Micro);
            Assert.NotEmpty(s2Strategy.Confluence);
            Assert.Contains("S2", s2Strategy.TelemetryTags);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void LoadAll_NonExistentDirectory_ThrowsDirectoryNotFoundException()
    {
        // Arrange
        var nonExistentDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());

        // Act & Assert
        Assert.Throws<DirectoryNotFoundException>(() => SimpleDslLoader.LoadAll(nonExistentDir));
    }

    [Fact]
    public void LoadAll_InvalidYamlSyntax_ThrowsInvalidOperationException()
    {
        // Arrange
        var tempDir = CreateTempDirectoryWithInvalidYaml();

        try
        {
            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => SimpleDslLoader.LoadAll(tempDir));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void LoadAll_EmptyDirectory_ReturnsEmptyList()
    {
        // Arrange
        var tempDir = Directory.CreateTempSubdirectory().FullName;

        try
        {
            // Act
            var strategies = SimpleDslLoader.LoadAll(tempDir);

            // Assert
            Assert.NotNull(strategies);
            Assert.Empty(strategies);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void LoadAll_MixedFileTypes_LoadsOnlyYamlFiles()
    {
        // Arrange
        var tempDir = CreateTempDirectoryWithMixedFiles();

        try
        {
            // Act
            var strategies = SimpleDslLoader.LoadAll(tempDir);

            // Assert
            Assert.NotNull(strategies);
            Assert.Single(strategies); // Only one YAML file should be loaded
            Assert.Equal("S2", strategies.First().Name);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    private static string CreateTempDirectoryWithValidStrategies()
    {
        var tempDir = Directory.CreateTempSubdirectory().FullName;
        
        var s2Yaml = @"
name: S2
label: MeanReversion
family: RangeFade
bias: both
when:
  regime:
    - Range
    - LowVol
  micro:
    - ""zone.dist_to_supply_atr >= 0.5""
    - ""zone.dist_to_demand_atr >= 0.5""
contra:
  - ""pattern.bear_score > 0.7 and intent == Long""
  - ""pattern.bull_score > 0.7 and intent == Short""
confluence:
  - ""vwap.distance_atr <= 0.6""
playbook:
  entry: limit_reversion
  bracket: ""tp_at_zone_edge; sl_outside_zone buffer_ticks: 6""
telemetryTags:
  - S2
  - RangeFade
";

        var s3Yaml = @"
name: S3
label: Compression
family: Breakout
bias: both
when:
  regime:
    - Range
    - LowVol
  micro:
    - ""vdc <= 0.6""
    - ""inside_bars_lookback >= 2""
confluence:
  - ""zone.breakout_score >= 0.65""
playbook:
  entry: ""stop_breakout confirm_atr: 0.25""
  bracket: ""measured_move_or_next_zone buffer_ticks: 6""
telemetryTags:
  - S3
  - Compression
";

        File.WriteAllText(Path.Combine(tempDir, "S2.yaml"), s2Yaml);
        File.WriteAllText(Path.Combine(tempDir, "S3.yaml"), s3Yaml);

        return tempDir;
    }

    private static string CreateTempDirectoryWithInvalidYaml()
    {
        var tempDir = Directory.CreateTempSubdirectory().FullName;
        
        var invalidYaml = @"
name: S2
label: MeanReversion
invalid_yaml_syntax: [unclosed bracket
";

        File.WriteAllText(Path.Combine(tempDir, "invalid.yaml"), invalidYaml);

        return tempDir;
    }

    private static string CreateTempDirectoryWithMixedFiles()
    {
        var tempDir = Directory.CreateTempSubdirectory().FullName;
        
        var validYaml = @"
name: S2
label: MeanReversion
family: RangeFade
bias: both
when:
  regime:
    - Range
  micro:
    - ""zone.dist_to_supply_atr >= 0.5""
confluence:
  - ""vwap.distance_atr <= 0.6""
playbook:
  entry: limit_reversion
  bracket: ""tp_at_zone_edge""
telemetryTags:
  - S2
";

        File.WriteAllText(Path.Combine(tempDir, "S2.yaml"), validYaml);
        File.WriteAllText(Path.Combine(tempDir, "README.txt"), "This is not a YAML file");
        File.WriteAllText(Path.Combine(tempDir, "config.json"), "{ \"key\": \"value\" }");

        return tempDir;
    }
}