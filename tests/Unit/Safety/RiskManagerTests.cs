using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Xunit;
using Moq;
using TradingBot.Abstractions;
using Trading.Safety;
using BotCore.Models;
using SafetyRiskBreach = Trading.Safety.RiskBreach;

namespace TradingBot.Tests.Unit.Safety
{
    /// <summary>
    /// Unit tests for RiskManager - Critical for production safety
    /// Addresses audit finding: Insufficient test coverage for risk management
    /// </summary>
    public class RiskManagerTests
    {
        private readonly Mock<ILogger<RiskManager>> _mockLogger;
        private readonly AppOptions _appOptions;
        private readonly RiskManager _riskManager;

        public RiskManagerTests()
        {
            _mockLogger = new Mock<ILogger<RiskManager>>();
            _appOptions = new AppOptions
            {
                MaxDailyLoss = -1000m,
                MaxPositionSize = 5,
                DrawdownLimit = -2000m,
                EnableDryRunMode = true
            };
            var options = Options.Create(_appOptions);
            _riskManager = new RiskManager(_mockLogger.Object, options);
        }

        [Fact]
        public void Constructor_InitializesCorrectly()
        {
            // Assert
            Assert.False(_riskManager.IsRiskBreached);
            var metrics = _riskManager.GetCurrentMetrics();
            Assert.Equal(0m, metrics.DailyPnL);
            Assert.Equal(0m, metrics.MaxDrawdown);
            Assert.Equal(0m, metrics.LargestPosition);
            Assert.False(metrics.IsBreached);
        }

        [Fact]
        public async Task ValidateOrderAsync_WithinLimits_ReturnsTrue()
        {
            // Arrange
            var order = new PlaceOrderRequest
            {
                Symbol = "ES",
                Quantity = 3,
                Price = 4500m,
                Side = "BUY"
            };

            // Act
            var result = await _riskManager.ValidateOrderAsync(order);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public async Task ValidateOrderAsync_ExceedsMaxPositionSize_ReturnsFalse()
        {
            // Arrange
            var order = new PlaceOrderRequest
            {
                Symbol = "ES",
                Quantity = 10, // Exceeds max position size of 5
                Price = 4500m,
                Side = "BUY"
            };

            // Act
            var result = await _riskManager.ValidateOrderAsync(order);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public async Task UpdateDailyPnLAsync_ExceedsMaxDailyLoss_TriggersRiskBreach()
        {
            // Arrange
            var breachTriggered = false;
            TradingBot.Abstractions.RiskBreach? capturedBreach = null;
            
            _riskManager.OnRiskBreach += (sender, args) =>
            {
                breachTriggered = true;
                capturedBreach = args.RiskBreach;
            };

            // Act
            await _riskManager.UpdateDailyPnLAsync(-1500m); // Exceeds -1000 limit

            // Assert
            Assert.True(breachTriggered);
            Assert.NotNull(capturedBreach);
            Assert.Equal("MaxDailyLoss", capturedBreach.Type);
            Assert.True(_riskManager.IsRiskBreached);
        }

        [Fact]
        public async Task UpdatePositionAsync_ValidPosition_UpdatesCorrectly()
        {
            // Arrange
            var symbol = "ES";
            var currentPrice = 4500m;
            var quantity = 2;

            // Act
            await _riskManager.UpdatePositionAsync(symbol, currentPrice, quantity);

            // Assert
            var metrics = _riskManager.GetCurrentMetrics();
            Assert.Equal(2m, metrics.LargestPosition);
        }

        [Fact]
        public async Task UpdateDailyPnLAsync_PositivePnL_UpdatesMetricsCorrectly()
        {
            // Arrange
            var profitAmount = 500m;

            // Act
            await _riskManager.UpdateDailyPnLAsync(profitAmount);

            // Assert
            var metrics = _riskManager.GetCurrentMetrics();
            Assert.Equal(profitAmount, metrics.DailyPnL);
            Assert.False(_riskManager.IsRiskBreached);
        }

        [Theory]
        [InlineData(-500, false)]
        [InlineData(-1000, false)] // At limit
        [InlineData(-1001, true)]  // Over limit
        [InlineData(-2000, true)]  // Way over limit
        public async Task UpdateDailyPnLAsync_VariousLossAmounts_HandlesCorrectly(decimal pnl, bool shouldBreach)
        {
            // Act
            await _riskManager.UpdateDailyPnLAsync(pnl);

            // Assert
            Assert.Equal(shouldBreach, _riskManager.IsRiskBreached);
        }

        [Fact]
        public async Task DrawdownCalculation_WorksCorrectly()
        {
            // Arrange - simulate a trading session with profit then loss
            
            // Start with profit to establish peak
            await _riskManager.UpdateDailyPnLAsync(1000m);
            var metricsAfterProfit = _riskManager.GetCurrentMetrics();
            Assert.Equal(0m, metricsAfterProfit.MaxDrawdown); // No drawdown yet

            // Then simulate loss
            await _riskManager.UpdateDailyPnLAsync(-500m);
            var metricsAfterLoss = _riskManager.GetCurrentMetrics();
            
            // Should show drawdown from peak
            Assert.True(metricsAfterLoss.MaxDrawdown < 0);
        }

        [Fact]
        public async Task RiskBreach_DrawdownLimit_TriggersCorrectly()
        {
            // Arrange
            var breachTriggered = false;
            TradingBot.Abstractions.RiskBreach? capturedBreach = null;
            
            _riskManager.OnRiskBreach += (sender, args) =>
            {
                breachTriggered = true;
                capturedBreach = args.RiskBreach;
            };

            // Act - simulate large drawdown
            await _riskManager.UpdateDailyPnLAsync(1000m); // Peak
            await _riskManager.UpdateDailyPnLAsync(-1500m); // Large loss causing drawdown

            // Assert
            Assert.True(breachTriggered);
            Assert.NotNull(capturedBreach);
            Assert.Equal("DrawdownLimit", capturedBreach.Type);
        }
    }
}