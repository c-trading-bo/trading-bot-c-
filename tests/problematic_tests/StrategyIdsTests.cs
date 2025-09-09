using System;
using Xunit;
using Trading.Strategies;

namespace Trading.Tests.Unit;

public class StrategyIdsTests
{
    [Fact]
    public void GenerateStrategyId_ValidName_ReturnsCorrectFormat()
    {
        // Arrange
        var strategyName = "TestStrategy";
        var date = new DateTime(2025, 1, 15);
        
        // Act
        var strategyId = StrategyIds.GenerateStrategyId(strategyName, date);
        
        // Assert
        Assert.Equal("TestStrategy_20250115", strategyId);
    }
    
    [Fact]
    public void GenerateStrategyId_NameWithSpecialCharacters_SanitizesCorrectly()
    {
        // Arrange
        var strategyName = "Test-Strategy!@#$%";
        var date = new DateTime(2025, 1, 15);
        
        // Act
        var strategyId = StrategyIds.GenerateStrategyId(strategyName, date);
        
        // Assert
        Assert.Equal("Test-Strategy_____20250115", strategyId);
    }
    
    [Fact]
    public void GenerateStrategyId_EmptyName_ThrowsArgumentException()
    {
        // Arrange
        var emptyName = "";
        
        // Act & Assert
        Assert.Throws<ArgumentException>(() => StrategyIds.GenerateStrategyId(emptyName));
    }
    
    [Fact]
    public void GenerateStrategyId_NullName_ThrowsArgumentException()
    {
        // Arrange
        string nullName = null;
        
        // Act & Assert
        Assert.Throws<ArgumentException>(() => StrategyIds.GenerateStrategyId(nullName));
    }
    
    [Fact]
    public void GenerateStrategyId_SameInputs_ReturnsSameId()
    {
        // Arrange
        var strategyName = "ConsistentStrategy";
        var date = new DateTime(2025, 1, 15);
        
        // Act
        var id1 = StrategyIds.GenerateStrategyId(strategyName, date);
        var id2 = StrategyIds.GenerateStrategyId(strategyName, date);
        
        // Assert
        Assert.Equal(id1, id2);
    }
    
    [Fact]
    public void GenerateStrategyIdWithConfig_SameConfig_ReturnsSameId()
    {
        // Arrange
        var strategyName = "ConfigStrategy";
        var config = new { RiskLevel = 0.5, MaxPositions = 3, Symbol = "ES" };
        var date = new DateTime(2025, 1, 15);
        
        // Act
        var id1 = StrategyIds.GenerateStrategyIdWithConfig(strategyName, config, date);
        var id2 = StrategyIds.GenerateStrategyIdWithConfig(strategyName, config, date);
        
        // Assert
        Assert.Equal(id1, id2);
        Assert.Contains("ConfigStrategy_20250115_", id1);
    }
    
    [Fact]
    public void GenerateStrategyIdWithConfig_DifferentConfig_ReturnsDifferentIds()
    {
        // Arrange
        var strategyName = "ConfigStrategy";
        var config1 = new { RiskLevel = 0.5, MaxPositions = 3 };
        var config2 = new { RiskLevel = 0.7, MaxPositions = 3 };
        var date = new DateTime(2025, 1, 15);
        
        // Act
        var id1 = StrategyIds.GenerateStrategyIdWithConfig(strategyName, config1, date);
        var id2 = StrategyIds.GenerateStrategyIdWithConfig(strategyName, config2, date);
        
        // Assert
        Assert.NotEqual(id1, id2);
        Assert.Contains("ConfigStrategy_20250115_", id1);
        Assert.Contains("ConfigStrategy_20250115_", id2);
    }
    
    [Fact]
    public void GenerateConfigHash_SameObject_ReturnsSameHash()
    {
        // Arrange
        var config = new { Value1 = 42, Value2 = "test", Value3 = 3.14 };
        
        // Act
        var hash1 = StrategyIds.GenerateConfigHash(config);
        var hash2 = StrategyIds.GenerateConfigHash(config);
        
        // Assert
        Assert.Equal(hash1, hash2);
        Assert.Equal(8, hash1.Length);
        Assert.Matches("^[0-9a-f]{8}$", hash1); // Should be 8 hex characters
    }
    
    [Fact]
    public void GenerateConfigHash_DifferentObjects_ReturnsDifferentHashes()
    {
        // Arrange
        var config1 = new { Value = 42 };
        var config2 = new { Value = 43 };
        
        // Act
        var hash1 = StrategyIds.GenerateConfigHash(config1);
        var hash2 = StrategyIds.GenerateConfigHash(config2);
        
        // Assert
        Assert.NotEqual(hash1, hash2);
    }
    
    [Fact]
    public void GenerateConfigHash_NullConfig_ThrowsArgumentNullException()
    {
        // Arrange
        object nullConfig = null;
        
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => StrategyIds.GenerateConfigHash(nullConfig));
    }
    
    [Fact]
    public void ParseStrategyId_ValidId_ReturnsCorrectParts()
    {
        // Arrange
        var strategyId = "TestStrategy_20250115";
        
        // Act
        var result = StrategyIds.ParseStrategyId(strategyId);
        
        // Assert
        Assert.NotNull(result);
        Assert.Equal("TestStrategy", result.Value.StrategyName);
        Assert.Equal(new DateTime(2025, 1, 15), result.Value.Date);
    }
    
    [Fact]
    public void ParseStrategyId_ComplexName_ReturnsCorrectParts()
    {
        // Arrange
        var strategyId = "Multi_Word_Strategy_Name_20250115";
        
        // Act
        var result = StrategyIds.ParseStrategyId(strategyId);
        
        // Assert
        Assert.NotNull(result);
        Assert.Equal("Multi_Word_Strategy_Name", result.Value.StrategyName);
        Assert.Equal(new DateTime(2025, 1, 15), result.Value.Date);
    }
    
    [Fact]
    public void ParseStrategyId_InvalidFormat_ReturnsNull()
    {
        // Arrange
        var invalidIds = new[] { "NoDate", "Invalid_Date_Format", "", null, "Strategy_202501" };
        
        // Act & Assert
        foreach (var invalidId in invalidIds)
        {
            var result = StrategyIds.ParseStrategyId(invalidId);
            Assert.Null(result);
        }
    }
    
    [Fact]
    public void StrategyId_RoundTrip_PreservesData()
    {
        // Arrange
        var originalName = "RoundTrip_Test_Strategy";
        var originalDate = new DateTime(2025, 1, 15);
        
        // Act
        var strategyId = StrategyIds.GenerateStrategyId(originalName, originalDate);
        var parsed = StrategyIds.ParseStrategyId(strategyId);
        
        // Assert
        Assert.NotNull(parsed);
        Assert.Equal(originalName, parsed.Value.StrategyName);
        Assert.Equal(originalDate, parsed.Value.Date);
    }
}