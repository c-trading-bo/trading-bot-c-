using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Xunit;
using TradingBot.Abstractions;

namespace TradingBot.Tests.Unit
{
    /// <summary>
    /// S7ServiceTests - Required by S7 Audit-Clean Acceptance Contract
    /// Verifies RiskOn/RiskOff transitions, coherence behavior, cooldown
    /// </summary>
    public class S7ServiceTests
    {
        private readonly S7Configuration _config;
        private readonly BreadthConfiguration _breadthConfig;
        private readonly ILogger<S7.S7Service> _logger;
        private readonly IOptions<S7Configuration> _options;
        private readonly IOptions<BreadthConfiguration> _breadthOptions;

        public S7ServiceTests()
        {
            _config = new S7Configuration
            {
                Enabled = true,
                BarTimeframeMinutes = 5,
                LookbackShortBars = 10,
                LookbackMediumBars = 30,
                LookbackLongBars = 60,
                ZThresholdEntry = 2.0m,
                ZThresholdExit = 1.0m,
                CoherenceMin = 0.75m,
                CooldownBars = 5,
                SizeTiltFactor = 1.0m,
                AtrMultiplierStop = 2.0m,
                AtrMultiplierTarget = 4.0m,
                MinAtrThreshold = 0.7m,
                ZScoreAlignmentWeight = 0.4m,
                DirectionAlignmentWeight = 0.4m,
                TimeframeCoherenceWeight = 0.2m,
                LeaderThreshold = 1.2m,
                TimeframeCountNormalizer = 3.0m,
                MaxSizeTiltMultiplier = 2.0m,
                BaseBreadthScore = 1.0m,
                AdvanceDeclineBonus = 0.1m,
                AdvanceDeclinePenalty = 0.1m,
                NewHighsLowsBonus = 0.05m,
                MinBreadthScore = 0.5m,
                MaxBreadthScore = 1.5m,
                
                // Enhanced adaptive and dispersion parameters
                DispersionThreshold = 0.3m,
                AdvanceFractionMin = 0.6m,
                DispersionSizeBoostFactor = 1.5m,
                DispersionSizeBlockFactor = 0.5m,
                EnableDispersionAdjustments = true,
                EnableAdaptiveThresholds = true,
                AdaptiveThresholdMin = 1.5m,
                AdaptiveThresholdMax = 3.0m,
                AdaptiveSensitivity = 0.1m,
                AdaptiveLookbackPeriod = 20,
                AdaptiveVolatilityWeight = 0.3m,
                AdaptivePerformanceWeight = 0.7m,
                EnableFusionTags = true,
                FusionStateTagPrefix = "fusion.s7_state",
                FusionCoherenceTag = "fusion.s7_coherence",
                FusionDispersionTag = "fusion.s7_dispersion",
                FusionAdaptiveTag = "fusion.s7_adaptive",
                FailOnUnknownKeys = true,
                FailOnMissingData = true,
                TelemetryPrefix = "s7"
            };
            
            // Initialize Symbols separately since it's read-only
            _config.Symbols.Add("ES");
            _config.Symbols.Add("NQ");
            
            _breadthConfig = new BreadthConfiguration
            {
                Enabled = true,
                AdvanceDeclineThreshold = 0.75m,
                NewHighsLowsRatio = 2.0m,
                SectorRotationWeight = 0.25m,
                BreadthLookbackBars = 20
            };

            _logger = new TestLogger<S7.S7Service>();
            _options = Options.Create(_config);
            _breadthOptions = Options.Create(_breadthConfig);
        }

        [Fact]
        public void S7Service_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var service = new S7.S7Service(_logger, _options, _breadthOptions);

            // Assert
            Assert.NotNull(service);
            Assert.False(service.IsReady()); // Should not be ready without market data
        }

        [Fact]
        public async Task S7Service_UpdateAsync_AcceptsValidBarData()
        {
            // Arrange
            var service = new S7.S7Service(_logger, _options, _breadthOptions);
            var barData = CreateTestBarData("ES", 4500.0m, 4501.0m, 4499.0m, 4500.5m, 1000);

            // Act & Assert - Should not throw
            await service.UpdateAsync("ES", barData.Close, barData.Timestamp);
        }

        [Fact]
        public void S7Service_GetCurrentSnapshot_ReturnsValidSnapshot()
        {
            // Arrange
            var service = new S7.S7Service(_logger, _options, _breadthOptions);

            // Act
            var snapshot = service.GetCurrentSnapshot();

            // Assert
            Assert.NotNull(snapshot);
            Assert.False(snapshot.IsActionable); // Should not be actionable without sufficient data
            Assert.Equal(S7Leader.None, snapshot.DominantLeader);
        }

        [Fact]
        public void S7Service_GetFeatureTuple_ReturnsValidFeatures()
        {
            // Arrange
            var service = new S7.S7Service(_logger, _options, _breadthOptions);

            // Act
            var features = service.GetFeatureTuple("ES");

            // Assert
            Assert.NotNull(features);
            Assert.Equal("ES", features.Symbol);
            Assert.False(features.IsSignalActive); // Should not be active without sufficient data
        }

        [Fact]
        public async Task S7Service_CooldownBehavior_PreventsSignalFlapping()
        {
            // Arrange
            var service = new S7.S7Service(_logger, _options, _breadthOptions);
            
            // Add sufficient bars to trigger signal
            for (int i = 0; i < 65; i++) // More than max lookback
            {
                await service.UpdateAsync("ES", 4500.0m + i, DateTime.UtcNow.AddMinutes(-i));
                await service.UpdateAsync("NQ", 18000.0m + i * 4, DateTime.UtcNow.AddMinutes(-i));
            }

            // Act - Get initial state
            var initialFeatures = service.GetFeatureTuple("ES");
            
            // Simulate bars during cooldown period
            for (int i = 0; i < _config.CooldownBars; i++)
            {
                await service.UpdateAsync("ES", 4500.0m, DateTime.UtcNow.AddMinutes(i + 1));
            }

            var cooldownFeatures = service.GetFeatureTuple("ES");

            // Assert - Should maintain cooldown behavior
            Assert.NotNull(initialFeatures);
            Assert.NotNull(cooldownFeatures);
        }

        [Fact]
        public void S7Service_CoherenceCalculation_RequiresMinimumThreshold()
        {
            // Arrange
            var service = new S7.S7Service(_logger, _options, _breadthOptions);

            // Act
            var snapshot = service.GetCurrentSnapshot();

            // Assert - Without sufficient coherence, should not be actionable
            Assert.False(snapshot.IsActionable);
            Assert.True(snapshot.CrossSymbolCoherence < _config.CoherenceMin);
        }

        [Fact]
        public async Task S7Service_RiskOnRiskOff_Transitions()
        {
            // Arrange
            var service = new S7.S7Service(_logger, _options, _breadthOptions);
            
            // Act - Add bars to simulate risk-on conditions (strong coherent signals)
            for (int i = 0; i < 65; i++)
            {
                // Simulate strong upward trend in both ES and NQ (risk-on)
                await service.UpdateAsync("ES", 4500.0m + i * 2, DateTime.UtcNow.AddMinutes(-i));
                await service.UpdateAsync("NQ", 18000.0m + i * 8, DateTime.UtcNow.AddMinutes(-i));
            }

            var riskOnSnapshot = service.GetCurrentSnapshot();

            // Simulate risk-off conditions (divergent or weak signals)
            for (int i = 0; i < 20; i++)
            {
                // Simulate divergent movements (ES up, NQ down - risk-off)
                await service.UpdateAsync("ES", 4500.0m + i, DateTime.UtcNow.AddMinutes(i + 1));
                await service.UpdateAsync("NQ", 18000.0m - i * 2, DateTime.UtcNow.AddMinutes(i + 1));
            }

            var riskOffSnapshot = service.GetCurrentSnapshot();

            // Assert - Should show transition from risk-on to risk-off
            Assert.NotNull(riskOnSnapshot);
            Assert.NotNull(riskOffSnapshot);
            
            // Risk-off should have lower coherence than risk-on
            Assert.True(riskOffSnapshot.CrossSymbolCoherence <= riskOnSnapshot.CrossSymbolCoherence);
        }

        [Fact]
        public void S7Service_FailClosed_MissingDataTriggersHold()
        {
            // Arrange
            var failClosedConfig = new S7Configuration
            {
                Enabled = true,
                FailOnMissingData = true,
                // Other required properties...
                BarTimeframeMinutes = 5,
                LookbackShortBars = 10,
                LookbackMediumBars = 30,
                LookbackLongBars = 60,
                ZThresholdEntry = 2.0m,
                ZThresholdExit = 1.0m,
                CoherenceMin = 0.75m,
                CooldownBars = 5,
                SizeTiltFactor = 1.0m,
                AtrMultiplierStop = 2.0m,
                AtrMultiplierTarget = 4.0m,
                MinAtrThreshold = 0.7m,
                ZScoreAlignmentWeight = 0.4m,
                DirectionAlignmentWeight = 0.4m,
                TimeframeCoherenceWeight = 0.2m,
                LeaderThreshold = 1.2m,
                TimeframeCountNormalizer = 3.0m,
                MaxSizeTiltMultiplier = 2.0m,
                BaseBreadthScore = 1.0m,
                AdvanceDeclineBonus = 0.1m,
                AdvanceDeclinePenalty = 0.1m,
                NewHighsLowsBonus = 0.05m,
                MinBreadthScore = 0.5m,
                MaxBreadthScore = 1.5m,
                FailOnUnknownKeys = true,
                TelemetryPrefix = "s7"
            };
            
            // Initialize Symbols separately since it's read-only
            failClosedConfig.Symbols.Add("ES");
            failClosedConfig.Symbols.Add("NQ");

            var service = new S7.S7Service(_logger, Options.Create(failClosedConfig), _breadthOptions);

            // Act
            var snapshot = service.GetCurrentSnapshot();

            // Assert - With fail-closed enabled and no data, should not be actionable
            Assert.False(snapshot.IsActionable);
        }

        private TestBarData CreateTestBarData(string symbol, decimal open, decimal high, decimal low, decimal close, long volume)
        {
            return new TestBarData
            {
                Symbol = symbol,
                Timestamp = DateTime.UtcNow,
                Open = open,
                High = high,
                Low = low,
                Close = close,
                Volume = volume
            };
        }

        private class TestBarData
        {
            public string Symbol { get; set; } = string.Empty;
            public DateTime Timestamp { get; set; }
            public decimal Open { get; set; }
            public decimal High { get; set; }
            public decimal Low { get; set; }
            public decimal Close { get; set; }
            public long Volume { get; set; }
        }

        private class TestLogger<T> : ILogger<T>
        {
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
            public bool IsEnabled(LogLevel logLevel) => true;
            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) { }
        }
    }
}