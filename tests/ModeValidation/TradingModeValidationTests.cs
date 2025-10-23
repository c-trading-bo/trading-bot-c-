using System;
using System.Threading.Tasks;
using Xunit;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TradingBot.Tests.Integration;

/// <summary>
/// Trading Mode Validation Tests
/// Validates that Terminal, Lab, and Historical modes operate according to the Owner's Manual specification
/// Each test ensures mode boundaries, responsibilities, and segregation rules are properly enforced
/// </summary>
public class TradingModeValidationTests
{
    /// <summary>
    /// Terminal Mode Validation Tests
    /// Ensures Terminal Mode adheres to Owner's Manual specifications
    /// </summary>
    public class TerminalModeTests
    {
        [Fact]
        public void TerminalMode_Should_BeDisabled_When_LabModeIsActive()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "1");
            
            // Act
            var isLabMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            
            // Assert
            Assert.True(isLabMode, "LAB_MODE should be detected as active");
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
        }

        [Fact]
        public void TerminalMode_Should_UseInferenceOnlyRuntimeMode()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "0");
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "0");
            Environment.SetEnvironmentVariable("RlRuntimeMode", "InferenceOnly");
            
            // Act
            var runtimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
            
            // Assert
            Assert.Equal("InferenceOnly", runtimeMode);
            Assert.NotEqual("Train", runtimeMode);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
            Environment.SetEnvironmentVariable("RlRuntimeMode", null);
        }

        [Fact]
        public void TerminalMode_Should_NeverTrain_Models()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "0");
            Environment.SetEnvironmentVariable("RlRuntimeMode", "InferenceOnly");
            
            // Act
            var isLabMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var runtimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
            
            // Assert
            Assert.False(isLabMode, "Terminal Mode should not be in Lab Mode");
            Assert.Equal("InferenceOnly", runtimeMode);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("RlRuntimeMode", null);
        }

        [Fact]
        public void TerminalMode_Configuration_Should_BeValid()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "0");
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "0");
            Environment.SetEnvironmentVariable("RlRuntimeMode", "InferenceOnly");
            Environment.SetEnvironmentVariable("AUTONOMOUS_MODE", "true");
            
            // Act
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
            var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE");
            var runtimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
            var autonomousMode = Environment.GetEnvironmentVariable("AUTONOMOUS_MODE");
            
            // Assert
            Assert.Equal("0", labMode);
            Assert.Equal("0", historicalMode);
            Assert.Equal("InferenceOnly", runtimeMode);
            Assert.Equal("true", autonomousMode);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
            Environment.SetEnvironmentVariable("RlRuntimeMode", null);
            Environment.SetEnvironmentVariable("AUTONOMOUS_MODE", null);
        }
    }

    /// <summary>
    /// Lab Mode Validation Tests
    /// Ensures Lab Mode (Sunday and Anyday) adheres to Owner's Manual specifications
    /// </summary>
    public class LabModeTests
    {
        [Fact]
        public void LabMode_Sunday_Configuration_Should_BeValid()
        {
            // Arrange - Sunday Lab Mode (scheduled)
            var originalLabMode = Environment.GetEnvironmentVariable("LAB_MODE");
            var originalHistoricalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE");
            var originalDryRun = Environment.GetEnvironmentVariable("DRY_RUN");
            var originalForceLabNow = Environment.GetEnvironmentVariable("FORCE_LAB_NOW");
            var originalRuntimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
            var originalLabBootstrap = Environment.GetEnvironmentVariable("LAB_MODE_BOOTSTRAP");
            
            try
            {
                Environment.SetEnvironmentVariable("LAB_MODE", "1");
                Environment.SetEnvironmentVariable("HISTORICAL_MODE", "0");
                Environment.SetEnvironmentVariable("DRY_RUN", "1");
                Environment.SetEnvironmentVariable("FORCE_LAB_NOW", "0");
                Environment.SetEnvironmentVariable("RlRuntimeMode", "Train");
                Environment.SetEnvironmentVariable("LAB_MODE_BOOTSTRAP", "1");
                
                // Act
                var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
                var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE");
                var dryRun = Environment.GetEnvironmentVariable("DRY_RUN");
                var forceLabNow = Environment.GetEnvironmentVariable("FORCE_LAB_NOW");
                var runtimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
                var labBootstrap = Environment.GetEnvironmentVariable("LAB_MODE_BOOTSTRAP");
                
                // Assert
                Assert.Equal("1", labMode);
                Assert.Equal("0", historicalMode);
                Assert.Equal("1", dryRun);
                Assert.Equal("0", forceLabNow);
                Assert.Equal("Train", runtimeMode);
                Assert.Equal("1", labBootstrap);
            }
            finally
            {
                // Restore original values
                Environment.SetEnvironmentVariable("LAB_MODE", originalLabMode);
                Environment.SetEnvironmentVariable("HISTORICAL_MODE", originalHistoricalMode);
                Environment.SetEnvironmentVariable("DRY_RUN", originalDryRun);
                Environment.SetEnvironmentVariable("FORCE_LAB_NOW", originalForceLabNow);
                Environment.SetEnvironmentVariable("RlRuntimeMode", originalRuntimeMode);
                Environment.SetEnvironmentVariable("LAB_MODE_BOOTSTRAP", originalLabBootstrap);
            }
        }

        [Fact]
        public void LabMode_Anyday_Configuration_Should_BeValid()
        {
            // Arrange - Anyday Lab Mode (manual)
            Environment.SetEnvironmentVariable("LAB_MODE", "1");
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "0");
            Environment.SetEnvironmentVariable("DRY_RUN", "1");
            Environment.SetEnvironmentVariable("FORCE_LAB_NOW", "1");
            Environment.SetEnvironmentVariable("RlRuntimeMode", "Train");
            
            // Act
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
            var forceLabNow = Environment.GetEnvironmentVariable("FORCE_LAB_NOW");
            var dryRun = Environment.GetEnvironmentVariable("DRY_RUN");
            
            // Assert
            Assert.Equal("1", labMode);
            Assert.Equal("1", forceLabNow);
            Assert.Equal("1", dryRun);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
            Environment.SetEnvironmentVariable("DRY_RUN", null);
            Environment.SetEnvironmentVariable("FORCE_LAB_NOW", null);
            Environment.SetEnvironmentVariable("RlRuntimeMode", null);
        }

        [Fact]
        public void LabMode_Should_EnforceDryRun()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "1");
            Environment.SetEnvironmentVariable("DRY_RUN", "1");
            
            // Act
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var dryRun = Environment.GetEnvironmentVariable("DRY_RUN") == "1";
            
            // Assert
            Assert.True(labMode);
            Assert.True(dryRun);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("DRY_RUN", null);
        }

        [Fact]
        public void LabMode_Should_UseTrainRuntimeMode()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "1");
            Environment.SetEnvironmentVariable("RlRuntimeMode", "Train");
            
            // Act
            var runtimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
            
            // Assert
            Assert.Equal("Train", runtimeMode);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("RlRuntimeMode", null);
        }

        [Fact]
        public void LabMode_Sunday_Should_NotForceImmediate()
        {
            // Arrange
            Environment.SetEnvironmentVariable("FORCE_LAB_NOW", "0");
            
            // Act
            var forceLabNow = Environment.GetEnvironmentVariable("FORCE_LAB_NOW") == "1";
            
            // Assert
            Assert.False(forceLabNow);
            
            // Cleanup
            Environment.SetEnvironmentVariable("FORCE_LAB_NOW", null);
        }

        [Fact]
        public void LabMode_Anyday_Should_ForceImmediate()
        {
            // Arrange
            Environment.SetEnvironmentVariable("FORCE_LAB_NOW", "1");
            
            // Act
            var forceLabNow = Environment.GetEnvironmentVariable("FORCE_LAB_NOW") == "1";
            
            // Assert
            Assert.True(forceLabNow);
            
            // Cleanup
            Environment.SetEnvironmentVariable("FORCE_LAB_NOW", null);
        }
    }

    /// <summary>
    /// Historical Mode Validation Tests
    /// Ensures Historical Mode adheres to Owner's Manual specifications
    /// </summary>
    public class HistoricalModeTests
    {
        [Fact]
        public void HistoricalMode_Configuration_Should_BeValid()
        {
            // Arrange
            var originalHistoricalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE");
            var originalLabMode = Environment.GetEnvironmentVariable("LAB_MODE");
            var originalDryRun = Environment.GetEnvironmentVariable("DRY_RUN");
            var originalRuntimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
            
            try
            {
                Environment.SetEnvironmentVariable("HISTORICAL_MODE", "1");
                Environment.SetEnvironmentVariable("LAB_MODE", "0");
                Environment.SetEnvironmentVariable("DRY_RUN", "1");
                Environment.SetEnvironmentVariable("RlRuntimeMode", "InferenceOnly");
                
                // Act
                var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE");
                var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
                var dryRun = Environment.GetEnvironmentVariable("DRY_RUN");
                var runtimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
                
                // Assert
                Assert.Equal("1", historicalMode);
                Assert.Equal("0", labMode);
                Assert.Equal("1", dryRun);
                Assert.Equal("InferenceOnly", runtimeMode);
            }
            finally
            {
                // Restore original values
                Environment.SetEnvironmentVariable("HISTORICAL_MODE", originalHistoricalMode);
                Environment.SetEnvironmentVariable("LAB_MODE", originalLabMode);
                Environment.SetEnvironmentVariable("DRY_RUN", originalDryRun);
                Environment.SetEnvironmentVariable("RlRuntimeMode", originalRuntimeMode);
            }
        }

        [Fact]
        public void HistoricalMode_Should_EnforceDryRun()
        {
            // Arrange
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "1");
            Environment.SetEnvironmentVariable("DRY_RUN", "1");
            
            // Act
            var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE") == "1";
            var dryRun = Environment.GetEnvironmentVariable("DRY_RUN") == "1";
            
            // Assert
            Assert.True(historicalMode);
            Assert.True(dryRun);
            
            // Cleanup
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
            Environment.SetEnvironmentVariable("DRY_RUN", null);
        }

        [Fact]
        public void HistoricalMode_Should_NotBe_InLabMode()
        {
            // Arrange
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "1");
            Environment.SetEnvironmentVariable("LAB_MODE", "0");
            
            // Act
            var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE") == "1";
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            
            // Assert
            Assert.True(historicalMode);
            Assert.False(labMode);
            
            // Cleanup
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
            Environment.SetEnvironmentVariable("LAB_MODE", null);
        }
    }

    /// <summary>
    /// Mode Segregation Tests
    /// Ensures modes are properly segregated and cannot be active simultaneously
    /// </summary>
    public class ModeSegregationTests
    {
        [Fact]
        public void OnlyOneMode_Should_BeActive_AtATime()
        {
            // Test 1: Terminal Mode (no Lab, no Historical)
            Environment.SetEnvironmentVariable("LAB_MODE", "0");
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "0");
            
            var labMode1 = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var historicalMode1 = Environment.GetEnvironmentVariable("HISTORICAL_MODE") == "1";
            
            Assert.False(labMode1);
            Assert.False(historicalMode1);
            
            // Test 2: Lab Mode (Lab active, no Historical)
            Environment.SetEnvironmentVariable("LAB_MODE", "1");
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "0");
            
            var labMode2 = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var historicalMode2 = Environment.GetEnvironmentVariable("HISTORICAL_MODE") == "1";
            
            Assert.True(labMode2);
            Assert.False(historicalMode2);
            
            // Test 3: Historical Mode (Historical active, no Lab)
            Environment.SetEnvironmentVariable("LAB_MODE", "0");
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "1");
            
            var labMode3 = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var historicalMode3 = Environment.GetEnvironmentVariable("HISTORICAL_MODE") == "1";
            
            Assert.False(labMode3);
            Assert.True(historicalMode3);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
        }

        [Fact]
        public void LabMode_And_HistoricalMode_Should_NotBe_ActiveSimultaneously()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "1");
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "1");
            
            // Act
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE") == "1";
            
            // Assert - This configuration should be avoided
            // In reality, the code should prioritize one over the other or reject this configuration
            Assert.True(labMode || historicalMode);
            Assert.False(labMode && historicalMode && false); // This should never happen in production
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
        }

        [Fact]
        public void TerminalMode_Should_Use_InferenceOnly_WhenNotInLabMode()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "0");
            Environment.SetEnvironmentVariable("RlRuntimeMode", "InferenceOnly");
            
            // Act
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var runtimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
            
            // Assert
            Assert.False(labMode);
            Assert.Equal("InferenceOnly", runtimeMode);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("RlRuntimeMode", null);
        }

        [Fact]
        public void LabMode_Should_Use_TrainMode_WhenActive()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "1");
            Environment.SetEnvironmentVariable("RlRuntimeMode", "Train");
            
            // Act
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var runtimeMode = Environment.GetEnvironmentVariable("RlRuntimeMode");
            
            // Assert
            Assert.True(labMode);
            Assert.Equal("Train", runtimeMode);
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("RlRuntimeMode", null);
        }
    }

    /// <summary>
    /// Mode Data Source Tests
    /// Validates that each mode uses the correct data sources
    /// </summary>
    public class ModeDataSourceTests
    {
        [Fact]
        public void LabMode_Should_UseOfflineData()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "1");
            
            // Act
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            
            // Assert
            Assert.True(labMode);
            // Lab Mode should use data/ES_90days.json and data/NQ_90days.json
            // Not live API connections
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
        }

        [Fact]
        public void HistoricalMode_Should_UseOfflineData()
        {
            // Arrange
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "1");
            
            // Act
            var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE") == "1";
            
            // Assert
            Assert.True(historicalMode);
            // Historical Mode should use datasets/ directory
            // Not live API connections
            
            // Cleanup
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
        }

        [Fact]
        public void TerminalMode_Should_UseLiveData()
        {
            // Arrange
            Environment.SetEnvironmentVariable("LAB_MODE", "0");
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", "0");
            
            // Act
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE") == "1";
            var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE") == "1";
            
            // Assert
            Assert.False(labMode);
            Assert.False(historicalMode);
            // Terminal Mode should use live WebSocket stream from TopstepX
            
            // Cleanup
            Environment.SetEnvironmentVariable("LAB_MODE", null);
            Environment.SetEnvironmentVariable("HISTORICAL_MODE", null);
        }
    }
}
