using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Xunit;
using BotCore.Services;

namespace TradingBot.Tests.Unit
{
    /// <summary>
    /// Tests for the TradingReadinessTracker to ensure BarsSeen counter works correctly
    /// Validates the fix for historical bar seeding pipeline issue
    /// </summary>
    internal class TradingReadinessTrackerTests
    {
        private readonly ILogger<TradingReadinessTracker> _logger;
        private readonly TradingReadinessConfiguration _config;

        public TradingReadinessTrackerTests()
        {
            var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            _logger = loggerFactory.CreateLogger<TradingReadinessTracker>();
            
            _config = new TradingReadinessConfiguration
            {
                MinBarsSeen = 10,
                MinSeededBars = 5,
                MinLiveTicks = 2,
                Environment = new EnvironmentSettings
                {
                    Dev = new DevEnvironmentSettings { MinBarsSeen = 5, MinSeededBars = 3, MinLiveTicks = 1 },
                    Production = new ProductionEnvironmentSettings { MinBarsSeen = 10, MinSeededBars = 5, MinLiveTicks = 2 }
                }
            };
        }

        [Fact]
        public void IncrementBarsSeen_ShouldUpdateCounter()
        {
            // Arrange
            var tracker = new TradingReadinessTracker(_logger, Options.Create(_config));
            
            // Act
            tracker.IncrementBarsSeen(5);
            
            // Assert
            var context = tracker.GetReadinessContext();
            Assert.Equal(5, context.TotalBarsSeen);
            Assert.Equal(TradingReadinessState.Initializing, context.State);
        }

        [Fact]
        public void IncrementSeededBars_ShouldUpdateCounterAndState()
        {
            // Arrange
            var tracker = new TradingReadinessTracker(_logger, Options.Create(_config));
            
            // Act
            tracker.IncrementSeededBars(8);
            
            // Assert
            var context = tracker.GetReadinessContext();
            Assert.Equal(8, context.SeededBars);
            Assert.Equal(TradingReadinessState.Seeded, context.State);
        }

        [Fact]
        public async Task ValidateReadinessAsync_WithEnoughBars_ShouldBeReady()
        {
            // Arrange
            var tracker = new TradingReadinessTracker(_logger, Options.Create(_config));
            
            // Act
            tracker.IncrementBarsSeen(10);
            tracker.IncrementSeededBars(5);
            tracker.IncrementLiveTicks(2);
            
            var result = await tracker.ValidateReadinessAsync().ConfigureAwait(false);
            
            // Assert
            Assert.True(result.IsReady);
            Assert.Equal(TradingReadinessState.FullyReady, result.State);
            Assert.Equal(1.0, result.ReadinessScore);
        }

        [Fact]
        public async Task ValidateReadinessAsync_WithInsufficientBars_ShouldNotBeReady()
        {
            // Arrange
            var tracker = new TradingReadinessTracker(_logger, Options.Create(_config));
            
            // Act
            tracker.IncrementBarsSeen(3);
            tracker.IncrementSeededBars(2);
            
            var result = await tracker.ValidateReadinessAsync().ConfigureAwait(false);
            
            // Assert
            Assert.False(result.IsReady);
            Assert.Contains("BarsSeen", result.Reason);
            Assert.Contains("SeededBars", result.Reason);
            Assert.Contains("LiveTicks", result.Reason);
        }

        [Fact]
        public void Reset_ShouldClearAllCounters()
        {
            // Arrange
            var tracker = new TradingReadinessTracker(_logger, Options.Create(_config));
            tracker.IncrementBarsSeen(10);
            tracker.IncrementSeededBars(5);
            tracker.IncrementLiveTicks(3);
            
            // Act
            tracker.Reset();
            
            // Assert
            var context = tracker.GetReadinessContext();
            Assert.Equal(0, context.TotalBarsSeen);
            Assert.Equal(0, context.SeededBars);
            Assert.Equal(0, context.LiveTicks);
            Assert.Equal(TradingReadinessState.Initializing, context.State);
        }
    }
}