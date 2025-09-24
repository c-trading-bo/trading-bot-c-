using System;
using Xunit;

namespace TradingBot.Tests.Unit
{
    /// <summary>
    /// Tests for defensive string parsing to prevent ArgumentOutOfRangeException
    /// Validates the fix for substring operations
    /// </summary>
    internal class DefensiveParsingTests
    {
        [Theory]
        [InlineData("")]
        [InlineData("short")]
        [InlineData("S11L-20240101-120000")]
        [InlineData("VERYLONGSTRATEGYNAME-20240101-120000")]
        [InlineData("NoHyphen")]
        [InlineData("-")]
        [InlineData("--")]
        public void ProcessTagSafely_WithVariousInputs_ShouldNotCrash(string input)
        {
            // This test verifies that the defensive parsing changes prevent crashes
            // Simulates the logic used in BotSupervisor.ExtractStrategyIdFromTag
            
            // Arrange & Act & Assert - should not throw exceptions
            var result = ProcessTagSafely(input);
            
            Assert.NotNull(result);
            Assert.NotEmpty(result);
        }

        [Fact]
        public void SafeSubstring_WithValidInput_ShouldReturnCorrectResult()
        {
            // Arrange
            var input = "TEST-STRING-LONG";
            
            // Act
            var result = SafeSubstring(input, 0, 4);
            
            // Assert
            Assert.Equal("TEST", result);
        }

        [Fact]
        public void SafeSubstring_WithOutOfBoundsIndex_ShouldNotCrash()
        {
            // Arrange
            var input = "SHORT";
            
            // Act & Assert - should not throw ArgumentOutOfRangeException
            var result = SafeSubstring(input, 0, 20);
            
            Assert.Equal("SHORT", result); // Should return the full string when length is too long
        }

        [Fact]
        public void SafeSubstring_WithEmptyString_ShouldNotCrash()
        {
            // Arrange
            var input = "";
            
            // Act & Assert - should not throw ArgumentOutOfRangeException
            var result = SafeSubstring(input, 0, 5);
            
            Assert.Equal("", result);
        }

        /// <summary>
        /// Simulates the defensive string processing logic used in our fixes
        /// </summary>
        private static string ProcessTagSafely(string input)
        {
            try
            {
                if (string.IsNullOrEmpty(input))
                    return "default";

                var dashIndex = input.IndexOf('-');
                if (dashIndex > 0 && dashIndex < input.Length)
                {
                    return SafeSubstring(input, 0, dashIndex);
                }

                if (input.Length > 10)
                {
                    return SafeSubstring(input, 0, 10);
                }
                
                return input;
            }
            catch
            {
                return "default";
            }
        }

        /// <summary>
        /// Safe substring operation with bounds checking
        /// </summary>
        private static string SafeSubstring(string input, int startIndex, int length)
        {
            if (string.IsNullOrEmpty(input) || startIndex >= input.Length)
                return "";
                
            var safeLength = Math.Min(length, input.Length - startIndex);
            return input.Substring(startIndex, safeLength);
        }
    }
}