using System;
using System.IO;
using Xunit;
using BotCore.Config;

namespace TradingBot.Tests.Unit;

public class EnvConfigTests : IDisposable
{
    private readonly string _testEnvFile = ".env.test";
    private readonly string _originalEnvValue;

    public EnvConfigTests()
    {
        // Save original TEST_VAR if it exists
        _originalEnvValue = Environment.GetEnvironmentVariable("TEST_VAR") ?? "";
        
        // Clean up any existing test environment variable
        Environment.SetEnvironmentVariable("TEST_VAR", null);
    }

    public void Dispose()
    {
        // Restore original environment variable
        Environment.SetEnvironmentVariable("TEST_VAR", _originalEnvValue);
        
        // Clean up test files
        if (File.Exists(_testEnvFile))
        {
            File.Delete(_testEnvFile);
        }
    }

    [Fact]
    public void GetBool_TrueValues_ShouldReturnTrue()
    {
        // Arrange & Act & Assert
        Environment.SetEnvironmentVariable("TEST_BOOL", "1");
        Assert.True(EnvConfig.GetBool("TEST_BOOL"));

        Environment.SetEnvironmentVariable("TEST_BOOL", "true");
        Assert.True(EnvConfig.GetBool("TEST_BOOL"));

        Environment.SetEnvironmentVariable("TEST_BOOL", "TRUE");
        Assert.True(EnvConfig.GetBool("TEST_BOOL"));

        Environment.SetEnvironmentVariable("TEST_BOOL", "yes");
        Assert.True(EnvConfig.GetBool("TEST_BOOL"));

        // Clean up
        Environment.SetEnvironmentVariable("TEST_BOOL", null);
    }

    [Fact]
    public void GetBool_FalseValues_ShouldReturnFalse()
    {
        // Arrange & Act & Assert
        Environment.SetEnvironmentVariable("TEST_BOOL", "0");
        Assert.False(EnvConfig.GetBool("TEST_BOOL"));

        Environment.SetEnvironmentVariable("TEST_BOOL", "false");
        Assert.False(EnvConfig.GetBool("TEST_BOOL"));

        Environment.SetEnvironmentVariable("TEST_BOOL", "no");
        Assert.False(EnvConfig.GetBool("TEST_BOOL"));

        Environment.SetEnvironmentVariable("TEST_BOOL", "");
        Assert.False(EnvConfig.GetBool("TEST_BOOL"));

        // Clean up
        Environment.SetEnvironmentVariable("TEST_BOOL", null);
    }

    [Fact]
    public void GetBool_DefaultValue_ShouldReturnDefault()
    {
        // Act & Assert
        Assert.True(EnvConfig.GetBool("NON_EXISTENT_VAR", true));
        Assert.False(EnvConfig.GetBool("NON_EXISTENT_VAR", false));
    }

    [Fact]
    public void GetInt_ValidValues_ShouldReturnParsedInt()
    {
        // Arrange
        Environment.SetEnvironmentVariable("TEST_INT", "42");

        // Act
        var result = EnvConfig.GetInt("TEST_INT");

        // Assert
        Assert.Equal(42, result);

        // Clean up
        Environment.SetEnvironmentVariable("TEST_INT", null);
    }

    [Fact]
    public void GetInt_InvalidValue_ShouldReturnDefault()
    {
        // Arrange
        Environment.SetEnvironmentVariable("TEST_INT", "not_a_number");

        // Act
        var result = EnvConfig.GetInt("TEST_INT", 99);

        // Assert
        Assert.Equal(99, result);

        // Clean up
        Environment.SetEnvironmentVariable("TEST_INT", null);
    }

    [Fact]
    public void GetDecimal_ValidValues_ShouldReturnParsedDecimal()
    {
        // Arrange
        Environment.SetEnvironmentVariable("TEST_DECIMAL", "123.45");

        // Act
        var result = EnvConfig.GetDecimal("TEST_DECIMAL");

        // Assert
        Assert.Equal(123.45m, result);

        // Clean up
        Environment.SetEnvironmentVariable("TEST_DECIMAL", null);
    }

    [Fact]
    public void IsKillFilePresent_WhenFileExists_ShouldReturnTrue()
    {
        // Arrange
        File.WriteAllText("kill.txt", "test");

        try
        {
            // Act
            var result = EnvConfig.IsKillFilePresent();

            // Assert
            Assert.True(result);
        }
        finally
        {
            // Clean up
            if (File.Exists("kill.txt"))
            {
                File.Delete("kill.txt");
            }
        }
    }

    [Fact]
    public void IsKillFilePresent_WhenFileNotExists_ShouldReturnFalse()
    {
        // Ensure file doesn't exist
        if (File.Exists("kill.txt"))
        {
            File.Delete("kill.txt");
        }

        // Act
        var result = EnvConfig.IsKillFilePresent();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GetExecutionMode_KillFilePresent_ShouldReturnDryRun()
    {
        // Arrange
        File.WriteAllText("kill.txt", "test");
        var context = new ExecutionContext
        {
            BarsSeen = 15,
            HubsConnected = true,
            CanTrade = true,
            ContractId = "TEST_CONTRACT"
        };

        try
        {
            // Act
            var result = EnvConfig.GetExecutionMode(context);

            // Assert
            Assert.Equal(ExecutionMode.DryRun, result);
        }
        finally
        {
            // Clean up
            if (File.Exists("kill.txt"))
            {
                File.Delete("kill.txt");
            }
        }
    }

    [Fact]
    public void GetExecutionMode_ExecuteFalse_ShouldReturnDryRun()
    {
        // Arrange
        Environment.SetEnvironmentVariable("EXECUTE", "false");
        var context = new ExecutionContext
        {
            BarsSeen = 15,
            HubsConnected = true,
            CanTrade = true,
            ContractId = "TEST_CONTRACT"
        };

        try
        {
            // Act
            var result = EnvConfig.GetExecutionMode(context);

            // Assert
            Assert.Equal(ExecutionMode.DryRun, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable("EXECUTE", null);
        }
    }

    [Fact]
    public void GetExecutionMode_InsufficientBars_ShouldReturnDryRun()
    {
        // Arrange
        Environment.SetEnvironmentVariable("EXECUTE", "true");
        var context = new ExecutionContext
        {
            BarsSeen = 5, // Less than 10
            HubsConnected = true,
            CanTrade = true,
            ContractId = "TEST_CONTRACT"
        };

        try
        {
            // Act
            var result = EnvConfig.GetExecutionMode(context);

            // Assert
            Assert.Equal(ExecutionMode.DryRun, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable("EXECUTE", null);
        }
    }

    [Fact]
    public void GetExecutionMode_AllConditionsMet_ShouldReturnAutoExecute()
    {
        // Arrange
        Environment.SetEnvironmentVariable("EXECUTE", "true");
        var context = new ExecutionContext
        {
            BarsSeen = 15,
            HubsConnected = true,
            CanTrade = true,
            ContractId = "TEST_CONTRACT"
        };

        try
        {
            // Act
            var result = EnvConfig.GetExecutionMode(context);

            // Assert
            Assert.Equal(ExecutionMode.AutoExecute, result);
        }
        finally
        {
            Environment.SetEnvironmentVariable("EXECUTE", null);
        }
    }

    [Fact]
    public void ShouldQuickExit_True_ShouldReturnTrue()
    {
        // Arrange
        Environment.SetEnvironmentVariable("BOT_QUICK_EXIT", "1");

        try
        {
            // Act
            var result = EnvConfig.ShouldQuickExit();

            // Assert
            Assert.True(result);
        }
        finally
        {
            Environment.SetEnvironmentVariable("BOT_QUICK_EXIT", null);
        }
    }

    [Fact]
    public void ShouldQuickExit_False_ShouldReturnFalse()
    {
        // Arrange
        Environment.SetEnvironmentVariable("BOT_QUICK_EXIT", "0");

        try
        {
            // Act
            var result = EnvConfig.ShouldQuickExit();

            // Assert
            Assert.False(result);
        }
        finally
        {
            Environment.SetEnvironmentVariable("BOT_QUICK_EXIT", null);
        }
    }

    [Fact]
    public void GetAllRedacted_ShouldRedactSecrets()
    {
        // Arrange
        Environment.SetEnvironmentVariable("TEST_TOKEN", "secret123");
        Environment.SetEnvironmentVariable("TEST_NORMAL", "normal_value");

        try
        {
            // Act
            var result = EnvConfig.GetAllRedacted();

            // Assert
            Assert.Contains("TEST_TOKEN", result.Keys);
            Assert.Contains("TEST_NORMAL", result.Keys);
            Assert.Equal("[REDACTED]", result["TEST_TOKEN"]);
            Assert.Equal("normal_value", result["TEST_NORMAL"]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TEST_TOKEN", null);
            Environment.SetEnvironmentVariable("TEST_NORMAL", null);
        }
    }
}