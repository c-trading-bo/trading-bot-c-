using Xunit;
using TradingBot.Infrastructure.TopstepX;

namespace TradingBot.Tests.Unit;

public class PxHelpersTests
{
    [Theory]
    [InlineData(5500.12m, 5500.00m)]
    [InlineData(5500.37m, 5500.25m)]
    [InlineData(5500.63m, 5500.75m)]
    [InlineData(5500.88m, 5501.00m)]
    [InlineData(5499.99m, 5500.00m)]
    public void RoundToTick_ES_ShouldRoundToQuarterTick(decimal input, decimal expected)
    {
        // Act
        var result = Px.RoundToTick(input);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(0.05m, 0.00m)]
    [InlineData(0.10m, 0.00m)]
    [InlineData(0.15m, 0.25m)]
    [InlineData(0.30m, 0.25m)]
    [InlineData(0.40m, 0.50m)]
    public void RoundToTick_CustomTick_ShouldRoundCorrectly(decimal input, decimal expected)
    {
        // Act
        var result = Px.RoundToTick(input, 0.25m);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(5500.123456m, "5500.12")]
    [InlineData(5500.999999m, "5501.00")]
    [InlineData(0.001m, "0.00")]
    [InlineData(99999.99m, "99999.99")]
    public void F2_ShouldFormatToTwoDecimalPlaces(decimal input, string expected)
    {
        // Act
        var result = Px.F2(input);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(5500m, 5495m, 5510m, true, 2.0)] // Long: Risk=5, Reward=10, R=2.0
    [InlineData(5500m, 5505m, 5490m, false, 2.0)] // Short: Risk=5, Reward=10, R=2.0
    [InlineData(5500m, 5495m, 5497.5m, true, 0.5)] // Long: Risk=5, Reward=2.5, R=0.5
    [InlineData(5500m, 5500m, 5510m, true, 0)] // Long: Risk=0, Reward=10, R=0 (invalid risk)
    public void RMultiple_ShouldCalculateCorrectly(decimal entry, decimal stop, decimal target, bool isLong, decimal expected)
    {
        // Act
        var result = Px.RMultiple(entry, stop, target, isLong);

        // Assert
        Assert.Equal(expected, result);
    }

    [Fact]
    public void RMultiple_ZeroRisk_ShouldReturnZero()
    {
        // Arrange
        decimal entry = 5500m;
        decimal stop = 5500m; // Same as entry = zero risk
        decimal target = 5510m;

        // Act
        var result = Px.RMultiple(entry, stop, target, true);

        // Assert
        Assert.Equal(0m, result);
    }

    [Fact]
    public void RMultiple_NegativeRisk_ShouldReturnZero()
    {
        // Arrange - stop is in wrong direction for long position
        decimal entry = 5500m;
        decimal stop = 5505m; // Stop above entry for long = negative risk
        decimal target = 5510m;

        // Act
        var result = Px.RMultiple(entry, stop, target, true);

        // Assert
        Assert.Equal(0m, result);
    }
}