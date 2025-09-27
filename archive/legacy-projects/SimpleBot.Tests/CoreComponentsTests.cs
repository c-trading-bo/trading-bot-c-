using Xunit;
using Trading.Strategies;

namespace SimpleBot.Tests;

public class CoreComponentsTests
{
    [Fact]
    public void StrategyIds_GenerateStrategyId_ShouldReturnValidId()
    {
        // Arrange
        var strategyName = "TestStrategy";
        
        // Act
        var result = StrategyIds.GenerateStrategyId(strategyName);
        
        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result);
        Assert.Contains("TestStrategy", result);
        Assert.Contains("_", result);
    }
    
    [Fact]
    public void StrategyIds_GenerateStrategyId_WithDate_ShouldReturnValidId()
    {
        // Arrange
        var strategyName = "TestStrategy";
        var testDate = new DateTime(2025, 9, 8);
        
        // Act
        var result = StrategyIds.GenerateStrategyId(strategyName, testDate);
        
        // Assert
        Assert.NotNull(result);
        Assert.Equal("TestStrategy_20250908", result);
    }
    
    [Fact]
    public void Analytics_CalculatePearsonCorrelation_PerfectCorrelation_ShouldReturnOne()
    {
        // Arrange
        var data1 = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var data2 = new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 };
        
        // Act
        var result = Analytics.CalculatePearsonCorrelation(data1, data2);
        
        // Assert
        Assert.Equal(1.0, result, 5); // 5 decimal places precision
    }
    
    [Fact]
    public void Analytics_CalculatePearsonCorrelation_NoCorrelation_ShouldReturnZero()
    {
        // Arrange
        var data1 = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var data2 = new double[] { 5.0, 4.0, 3.0, 2.0, 1.0 };
        
        // Act
        var result = Analytics.CalculatePearsonCorrelation(data1, data2);
        
        // Assert
        Assert.Equal(-1.0, result, 5); // Perfect negative correlation
    }
    
    [Fact]
    public void Analytics_CalculatePearsonCorrelation_EmptyData_ShouldReturnNaN()
    {
        // Arrange
        var data1 = new double[] { };
        var data2 = new double[] { };
        
        // Act
        var result = Analytics.CalculatePearsonCorrelation(data1, data2);
        
        // Assert
        Assert.True(double.IsNaN(result));
    }
    
    [Fact]
    public void Analytics_CalculatePearsonCorrelation_NullData_ShouldThrowException()
    {
        // Act & Assert - Should throw ArgumentNullException for null inputs
        Assert.Throws<ArgumentNullException>(() => 
            Analytics.CalculatePearsonCorrelation(null!, new double[] { 1, 2, 3 }));
        
        Assert.Throws<ArgumentNullException>(() => 
            Analytics.CalculatePearsonCorrelation(new double[] { 1, 2, 3 }, null!));
    }
}