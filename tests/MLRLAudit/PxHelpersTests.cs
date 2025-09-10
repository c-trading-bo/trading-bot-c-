using Xunit;
using TradingBot.Infrastructure.TopstepX;

namespace TradingBot.Tests.Unit;

public class PxHelpersTests
{
    [Theory]
    [InlineData(5500.12, 5500.00)]
    [InlineData(5500.37, 5500.25)]
    [InlineData(5500.63, 5500.75)]
    [InlineData(5500.88, 5501.00)]
    [InlineData(5499.99, 5500.00)]
    public void RoundToTick_ES_ShouldRoundToQuarterTick(double input, double expected)
    {
        // Act
        var result = Px.RoundToTick((decimal)input);

        // Assert
        Assert.Equal((decimal)expected, result);
    }

    [Theory]
    [InlineData(0.05, 0.00)]
    [InlineData(0.10, 0.00)]
    [InlineData(0.15, 0.25)]
    [InlineData(0.30, 0.25)]
    [InlineData(0.40, 0.50)]
    public void RoundToTick_CustomTick_ShouldRoundCorrectly(double input, double expected)
    {
        // Act
        var result = Px.RoundToTick((decimal)input, 0.25m);

        // Assert
        Assert.Equal((decimal)expected, result);
    }

    [Theory]
    [InlineData(5500.123456, "5500.12")]
    [InlineData(5500.999999, "5501.00")]
    [InlineData(0.001, "0.00")]
    [InlineData(99999.99, "99999.99")]
    public void F2_ShouldFormatToTwoDecimalPlaces(double input, string expected)
    {
        // Act
        var result = Px.F2((decimal)input);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(5500, 5495, 5510, true, 2.0)] // Long: Risk=5, Reward=10, R=2.0
    [InlineData(5500, 5505, 5490, false, 2.0)] // Short: Risk=5, Reward=10, R=2.0
    [InlineData(5500, 5495, 5497.5, true, 0.5)] // Long: Risk=5, Reward=2.5, R=0.5
    [InlineData(5500, 5500, 5510, true, 0)] // Long: Risk=0, Reward=10, R=0 (invalid risk)
    public void RMultiple_ShouldCalculateCorrectly(double entry, double stop, double target, bool isLong, double expected)
    {
        // Act
        var result = Px.RMultiple((decimal)entry, (decimal)stop, (decimal)target, isLong);

        // Assert
        Assert.Equal((decimal)expected, result);
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