using System;
using System.Linq;
using Xunit;
using Trading.Strategies;

namespace Trading.Tests.Unit;

public class AnalyticsTests
{
    [Fact]
    public void CalculatePearsonCorrelation_PerfectPositiveCorrelation_ReturnsOne()
    {
        // Arrange
        var returnsX = new[] { 0.01, 0.02, 0.03, 0.04, 0.05 };
        var returnsY = new[] { 0.02, 0.04, 0.06, 0.08, 0.10 }; // Perfect positive correlation (2x)
        
        // Act
        var correlation = Analytics.CalculatePearsonCorrelation(returnsX, returnsY);
        
        // Assert
        Assert.Equal(1.0, correlation, 6); // 6 decimal places precision
    }
    
    [Fact]
    public void CalculatePearsonCorrelation_PerfectNegativeCorrelation_ReturnsMinusOne()
    {
        // Arrange
        var returnsX = new[] { 0.01, 0.02, 0.03, 0.04, 0.05 };
        var returnsY = new[] { -0.01, -0.02, -0.03, -0.04, -0.05 }; // Perfect negative correlation
        
        // Act
        var correlation = Analytics.CalculatePearsonCorrelation(returnsX, returnsY);
        
        // Assert
        Assert.Equal(-1.0, correlation, 6);
    }
    
    [Fact]
    public void CalculatePearsonCorrelation_NoCorrelation_ReturnsNearZero()
    {
        // Arrange
        var returnsX = new[] { 0.01, 0.02, 0.01, 0.02, 0.01 };
        var returnsY = new[] { 0.02, 0.01, 0.02, 0.01, 0.02 }; // Alternating pattern, no correlation
        
        // Act
        var correlation = Analytics.CalculatePearsonCorrelation(returnsX, returnsY);
        
        // Assert
        Assert.True(Math.Abs(correlation) < 0.1); // Should be close to zero
    }
    
    [Fact]
    public void CalculatePearsonCorrelation_KnownData_ReturnsExpectedValue()
    {
        // Arrange - Using known data with expected correlation
        var returnsX = new[] { 0.01, 0.02, -0.01, 0.03, -0.02 };
        var returnsY = new[] { 0.015, 0.025, -0.005, 0.035, -0.015 };
        
        // Act
        var correlation = Analytics.CalculatePearsonCorrelation(returnsX, returnsY);
        
        // Assert
        Assert.True(correlation > 0.9); // Should be highly correlated
        Assert.True(correlation <= 1.0);
    }
    
    [Fact]
    public void CalculatePearsonCorrelation_EmptyArrays_ThrowsArgumentException()
    {
        // Arrange
        var emptyArray = Array.Empty<double>();
        var validArray = new[] { 0.01, 0.02 };
        
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Analytics.CalculatePearsonCorrelation(null, validArray));
        Assert.Throws<ArgumentNullException>(() => Analytics.CalculatePearsonCorrelation(validArray, null));
    }
    
    [Fact]
    public void CalculatePearsonCorrelation_DifferentLengths_ThrowsArgumentException()
    {
        // Arrange
        var arrayX = new[] { 0.01, 0.02, 0.03 };
        var arrayY = new[] { 0.01, 0.02 };
        
        // Act & Assert
        Assert.Throws<ArgumentException>(() => Analytics.CalculatePearsonCorrelation(arrayX, arrayY));
    }
    
    [Fact]
    public void CalculatePearsonCorrelation_SingleValue_ReturnsNaN()
    {
        // Arrange
        var singleValue = new[] { 0.01 };
        
        // Act
        var correlation = Analytics.CalculatePearsonCorrelation(singleValue, singleValue);
        
        // Assert
        Assert.True(double.IsNaN(correlation));
    }
    
    [Fact]
    public void CalculateRollingReturns_ValidPrices_ReturnsCorrectReturns()
    {
        // Arrange
        var prices = new[] { 100.0, 102.0, 104.0, 103.0, 105.0 };
        
        // Act
        var returns = Analytics.CalculateRollingReturns(prices).ToArray();
        
        // Assert
        Assert.Equal(4, returns.Length);
        Assert.Equal(0.02, returns[0], 6); // (102 - 100) / 100
        Assert.Equal(0.0196, returns[1], 4); // (104 - 102) / 102
        Assert.Equal(-0.0096, returns[2], 4); // (103 - 104) / 104
        Assert.Equal(0.0194, returns[3], 4); // (105 - 103) / 103
    }
    
    [Fact]
    public void CalculateRollingReturns_MultiPeriod_ReturnsCorrectReturns()
    {
        // Arrange
        var prices = new[] { 100.0, 102.0, 104.0, 103.0, 105.0 };
        
        // Act
        var returns = Analytics.CalculateRollingReturns(prices, 2).ToArray();
        
        // Assert
        Assert.Equal(3, returns.Length);
        Assert.Equal(0.04, returns[0], 6); // (104 - 100) / 100
        Assert.Equal(0.0098, returns[1], 4); // (103 - 102) / 102
        Assert.Equal(0.0096, returns[2], 4); // (105 - 104) / 104
    }
    
    [Fact]
    public void CalculateSharpeRatio_ValidReturns_ReturnsCorrectRatio()
    {
        // Arrange
        var returns = new[] { 0.01, 0.02, -0.01, 0.03, -0.005, 0.015 };
        
        // Act
        var sharpeRatio = Analytics.CalculateSharpeRatio(returns, 0.02, 252.0);
        
        // Assert
        Assert.False(double.IsNaN(sharpeRatio));
        Assert.True(sharpeRatio != 0.0);
    }
    
    [Fact]
    public void CalculateRollingCorrelation_ValidData_ReturnsCorrelationSeries()
    {
        // Arrange
        var pricesX = new[] { 100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0 };
        var pricesY = new[] { 200.0, 204.0, 208.0, 206.0, 210.0, 214.0, 212.0 }; // 2x pricesX
        var windowSize = 3;
        
        // Act
        var rollingCorrelations = Analytics.CalculateRollingCorrelation(pricesX, pricesY, windowSize).ToArray();
        
        // Assert
        Assert.True(rollingCorrelations.Length > 0);
        Assert.All(rollingCorrelations, corr => Assert.True(corr >= -1.0 && corr <= 1.0));
    }
}