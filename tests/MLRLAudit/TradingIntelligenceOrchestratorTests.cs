using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using Microsoft.Extensions.Logging;
using Moq;
using Trading.Strategies;

namespace TradingBot.Tests.Unit
{
    /// <summary>
    /// Tests for OnnxModelWrapper to validate production functionality
    /// </summary>
    public class OnnxModelWrapperTests
    {
        private readonly Mock<ILogger<OnnxModelWrapper>> _mockLogger;
        private readonly OnnxModelWrapper _wrapper;

        public OnnxModelWrapperTests()
        {
            _mockLogger = new Mock<ILogger<OnnxModelWrapper>>();
            _wrapper = new OnnxModelWrapper(_mockLogger.Object);
        }

        [Fact]
        public async Task PredictConfidenceAsync_Should_Return_Valid_Confidence()
        {
            // Arrange
            var features = new Dictionary<string, double>
            {
                ["vix_level"] = 20.0,
                ["volume_ratio"] = 1.5,
                ["momentum"] = 0.02,
                ["rsi"] = 65.0,
                ["volatility"] = 0.8
            };

            // Act
            var confidence = await _wrapper.PredictConfidenceAsync(features);

            // Assert
            Assert.InRange(confidence, 0.0, 1.0);
        }

        [Fact]
        public async Task PredictConfidenceAsync_With_Named_Parameters_Should_Work()
        {
            // Act
            var confidence = await _wrapper.PredictConfidenceAsync(
                ("vix_level", 15.0),
                ("volume_ratio", 2.0),
                ("momentum", 0.03)
            );

            // Assert
            Assert.InRange(confidence, 0.0, 1.0);
        }

        [Fact]
        public async Task PredictConfidenceAsync_With_Empty_Features_Should_Return_Default()
        {
            // Arrange
            var features = new Dictionary<string, double>();

            // Act
            var confidence = await _wrapper.PredictConfidenceAsync(features);

            // Assert
            Assert.InRange(confidence, 0.0, 1.0);
            Assert.True(confidence > 0); // Should not be zero
        }

        [Fact]
        public async Task PredictConfidenceAsync_With_Extreme_Values_Should_Be_Normalized()
        {
            // Arrange
            var features = new Dictionary<string, double>
            {
                ["vix_level"] = 500.0, // Extreme value
                ["volume_ratio"] = -5.0, // Negative value
                ["rsi"] = 150.0 // Out of range
            };

            // Act
            var confidence = await _wrapper.PredictConfidenceAsync(features);

            // Assert
            Assert.InRange(confidence, 0.0, 1.0);
        }

        [Fact]
        public void IsModelAvailable_Should_Return_Boolean()
        {
            // Act
            var isAvailable = _wrapper.IsModelAvailable;

            // Assert
            Assert.IsType<bool>(isAvailable);
        }

        [Theory]
        [InlineData(10.0, 1.0, 0.01, 50.0, 0.3)]
        [InlineData(30.0, 0.5, -0.02, 30.0, 1.5)]
        [InlineData(25.0, 2.0, 0.05, 80.0, 0.1)]
        public async Task PredictConfidenceAsync_Should_Handle_Various_Market_Conditions(
            double vix, double volume, double momentum, double rsi, double volatility)
        {
            // Arrange
            var features = new Dictionary<string, double>
            {
                ["vix_level"] = vix,
                ["volume_ratio"] = volume,
                ["momentum"] = momentum,
                ["rsi"] = rsi,
                ["volatility"] = volatility
            };

            // Act
            var confidence = await _wrapper.PredictConfidenceAsync(features);

            // Assert
            Assert.InRange(confidence, 0.0, 1.0);
        }

        [Fact]
        public async Task ConfidenceHelper_PredictAsync_Should_Work()
        {
            // Act
            var confidence = await ConfidenceHelper.PredictAsync(
                ("vix_level", 20.0),
                ("volume_ratio", 1.5)
            );

            // Assert
            Assert.InRange(confidence, 0.0, 1.0);
        }

        [Fact]
        public async Task Multiple_Calls_Should_Be_Consistent()
        {
            // Arrange
            var features = new Dictionary<string, double>
            {
                ["vix_level"] = 20.0,
                ["volume_ratio"] = 1.0,
                ["momentum"] = 0.0,
                ["rsi"] = 50.0,
                ["volatility"] = 0.5
            };

            // Act
            var confidence1 = await _wrapper.PredictConfidenceAsync(features);
            var confidence2 = await _wrapper.PredictConfidenceAsync(features);

            // Assert
            // Should be similar but may have small variations due to noise
            Assert.InRange(Math.Abs(confidence1 - confidence2), 0.0, 0.2);
        }
    }

    /// <summary>
    /// Integration tests for production functionality
    /// </summary>
    public class ProductionIntegrationTests
    {
        [Fact]
        public void All_Production_Classes_Should_Be_Instantiable()
        {
            // Test that our production classes can be instantiated without errors
            var logger = Mock.Of<ILogger<OnnxModelWrapper>>();
            var wrapper = new OnnxModelWrapper(logger);
            
            Assert.NotNull(wrapper);
        }

        [Fact]
        public async Task Production_Workflow_Should_Execute_Without_Errors()
        {
            // Simulate a production workflow
            var logger = Mock.Of<ILogger<OnnxModelWrapper>>();
            var wrapper = new OnnxModelWrapper(logger);

            var marketConditions = new Dictionary<string, double>
            {
                ["vix_level"] = 18.5,
                ["volume_ratio"] = 1.2,
                ["momentum"] = 0.015,
                ["rsi"] = 62.0,
                ["volatility"] = 0.4,
                ["trend_strength"] = 0.8,
                ["macd_signal"] = 0.5
            };

            // This should complete without exceptions
            var confidence = await wrapper.PredictConfidenceAsync(marketConditions);
            
            Assert.InRange(confidence, 0.0, 1.0);
        }
    }
}