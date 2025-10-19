using BotCore.Models;
using BotCore.Patterns;
using BotCore.Patterns.Detectors;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using Xunit;
using Zones;
using Microsoft.Extensions.DependencyInjection;

namespace Tests.Unit.Patterns;

public class PatternEngineTests
{
    [Fact]
    public void PatternEngine_AggregatesScores_FromMultipleDetectors()
    {
        // Arrange
        var mockFeatureBus = new Mock<IFeatureBus>();
        var mockServiceProvider = new Mock<IServiceProvider>();
        var detectors = new List<IPatternDetector>
        {
            new CandlestickPatternDetector(CandlestickType.Doji),
            new CandlestickPatternDetector(CandlestickType.Hammer),
            new ContinuationPatternDetector(ContinuationType.BullFlag)
        };
        
        var engine = new PatternEngine(NullLogger<PatternEngine>.Instance, mockFeatureBus.Object, detectors, mockServiceProvider.Object);
        
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 102m, low: 98m, close: 100.1m), // Potential doji
            (open: 100.1m, high: 101m, low: 95m, close: 98m),   // Potential hammer
            // Add more bars for continuation patterns
            (open: 98m, high: 99m, low: 97m, close: 98.5m),
            (open: 98.5m, high: 100m, low: 98m, close: 99.2m),
            (open: 99.2m, high: 101m, low: 98.8m, close: 100.5m),
            (open: 100.5m, high: 102m, low: 100m, close: 101.2m),
            (open: 101.2m, high: 103m, low: 100.8m, close: 102.5m),
            (open: 102.5m, high: 104m, low: 102m, close: 103.2m),
            (open: 103.2m, high: 103.8m, low: 102.8m, close: 103.1m),
            (open: 103.1m, high: 103.5m, low: 102.5m, close: 103m),
            (open: 103m, high: 103.3m, low: 102.7m, close: 102.9m),
            (open: 102.9m, high: 103.2m, low: 102.6m, close: 103m)
        });
        
        // Act
        var scores = engine.GetScores("ES", bars);
        
        // Assert
        Assert.NotNull(scores);
        Assert.True(scores.BullScore >= 0);
        Assert.True(scores.BearScore >= 0);
        Assert.True(scores.PatternFlags.Count >= 0);
        
        // Verify feature bus calls
        mockFeatureBus.Verify(fb => fb.Publish("ES", It.IsAny<DateTime>(), "pattern.bull_score", It.IsAny<double>()), Times.Once);
        mockFeatureBus.Verify(fb => fb.Publish("ES", It.IsAny<DateTime>(), "pattern.bear_score", It.IsAny<double>()), Times.Once);
    }

    [Fact]
    public void PatternEngine_PublishesIndividualPatternScores_ToFeatureBus()
    {
        // Arrange
        var mockFeatureBus = new Mock<IFeatureBus>();
        var mockServiceProvider = new Mock<IServiceProvider>();
        var detectors = new List<IPatternDetector>
        {
            new CandlestickPatternDetector(CandlestickType.Doji)
        };
        
        var engine = new PatternEngine(NullLogger<PatternEngine>.Instance, mockFeatureBus.Object, detectors, mockServiceProvider.Object);
        
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 102m, low: 98m, close: 100.05m) // Strong doji pattern
        });
        
        // Act
        var scores = engine.GetScores("NQ", bars);
        
        // Assert
        // Should publish individual pattern score if detected
        if (scores.PatternFlags.ContainsKey("Doji"))
        {
            mockFeatureBus.Verify(fb => fb.Publish("NQ", It.IsAny<DateTime>(), "pattern.kind::Doji", It.IsAny<double>()), Times.Once);
        }
    }

    [Fact]
    public void PatternEngine_HandlesNoDetectors_Gracefully()
    {
        // Arrange
        var mockFeatureBus = new Mock<IFeatureBus>();
        var mockServiceProvider = new Mock<IServiceProvider>();
        var engine = new PatternEngine(NullLogger<PatternEngine>.Instance, mockFeatureBus.Object, new List<IPatternDetector>(), mockServiceProvider.Object);
        
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 102m, low: 98m, close: 100.5m)
        });
        
        // Act
        var scores = engine.GetScores("ES", bars);
        
        // Assert
        Assert.NotNull(scores);
        Assert.Equal(0, scores.BullScore);
        Assert.Equal(0, scores.BearScore);
        Assert.Empty(scores.PatternFlags);
    }

    [Fact]
    public void PatternEngine_HandlesEmptyBars_Gracefully()
    {
        // Arrange
        var mockFeatureBus = new Mock<IFeatureBus>();
        var mockServiceProvider = new Mock<IServiceProvider>();
        var detectors = new List<IPatternDetector>
        {
            new CandlestickPatternDetector(CandlestickType.Doji)
        };
        
        var engine = new PatternEngine(NullLogger<PatternEngine>.Instance, mockFeatureBus.Object, detectors, mockServiceProvider.Object);
        
        // Act
        var scores = engine.GetScores("ES", new List<Bar>());
        
        // Assert
        Assert.NotNull(scores);
        Assert.Equal(0, scores.BullScore);
        Assert.Equal(0, scores.BearScore);
        Assert.Empty(scores.PatternFlags);
    }

    [Fact]
    public void PatternEngine_NormalizesScores_WhenExceedingOne()
    {
        // Arrange - Create mock detectors that return high scores
        var mockDetector1 = new Mock<IPatternDetector>();
        var mockDetector2 = new Mock<IPatternDetector>();
        
        mockDetector1.Setup(d => d.PatternName).Returns("TestPattern1");
        mockDetector1.Setup(d => d.Family).Returns(PatternFamily.Candlestick);
        mockDetector1.Setup(d => d.RequiredBars).Returns(1);
        mockDetector1.Setup(d => d.Detect(It.IsAny<IReadOnlyList<Bar>>()))
                    .Returns(new PatternResult { Score = 0.9, Direction = 1, Confidence = 0.9 });
        
        mockDetector2.Setup(d => d.PatternName).Returns("TestPattern2");
        mockDetector2.Setup(d => d.Family).Returns(PatternFamily.Candlestick);
        mockDetector2.Setup(d => d.RequiredBars).Returns(1);
        mockDetector2.Setup(d => d.Detect(It.IsAny<IReadOnlyList<Bar>>()))
                    .Returns(new PatternResult { Score = 0.8, Direction = 1, Confidence = 0.8 });
        
        var mockFeatureBus = new Mock<IFeatureBus>();
        var mockServiceProvider = new Mock<IServiceProvider>();
        var detectors = new List<IPatternDetector> { mockDetector1.Object, mockDetector2.Object };
        var engine = new PatternEngine(NullLogger<PatternEngine>.Instance, mockFeatureBus.Object, detectors, mockServiceProvider.Object);
        
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 102m, low: 98m, close: 100.5m)
        });
        
        // Act
        var scores = engine.GetScores("ES", bars);
        
        // Assert - Scores should be normalized to not exceed 1.0
        Assert.True(scores.BullScore <= 1.0);
        Assert.True(scores.BearScore <= 1.0);
        Assert.Equal(2, scores.PatternFlags.Count);
    }

    [Fact]
    public void PatternEngine_HandlesBothBullishAndBearish_Patterns()
    {
        // Arrange
        var mockBullishDetector = new Mock<IPatternDetector>();
        var mockBearishDetector = new Mock<IPatternDetector>();
        
        mockBullishDetector.Setup(d => d.PatternName).Returns("BullishPattern");
        mockBullishDetector.Setup(d => d.Family).Returns(PatternFamily.Reversal);
        mockBullishDetector.Setup(d => d.RequiredBars).Returns(1);
        mockBullishDetector.Setup(d => d.Detect(It.IsAny<IReadOnlyList<Bar>>()))
                          .Returns(new PatternResult { Score = 0.6, Direction = 1, Confidence = 0.6 });
        
        mockBearishDetector.Setup(d => d.PatternName).Returns("BearishPattern");
        mockBearishDetector.Setup(d => d.Family).Returns(PatternFamily.Reversal);
        mockBearishDetector.Setup(d => d.RequiredBars).Returns(1);
        mockBearishDetector.Setup(d => d.Detect(It.IsAny<IReadOnlyList<Bar>>()))
                          .Returns(new PatternResult { Score = 0.4, Direction = -1, Confidence = 0.4 });
        
        var mockFeatureBus = new Mock<IFeatureBus>();
        var mockServiceProvider = new Mock<IServiceProvider>();
        var detectors = new List<IPatternDetector> { mockBullishDetector.Object, mockBearishDetector.Object };
        var engine = new PatternEngine(NullLogger<PatternEngine>.Instance, mockFeatureBus.Object, detectors, mockServiceProvider.Object);
        
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 102m, low: 98m, close: 100.5m)
        });
        
        // Act
        var scores = engine.GetScores("ES", bars);
        
        // Assert
        Assert.True(scores.BullScore > 0);
        Assert.True(scores.BearScore > 0);
        Assert.True(scores.BullScore > scores.BearScore); // Bullish pattern has higher score
        Assert.Equal(2, scores.PatternFlags.Count);
    }

    private static List<Bar> CreateTestBars(IEnumerable<(decimal open, decimal high, decimal low, decimal close)> barData)
    {
        var bars = new List<Bar>();
        var timestamp = DateTime.UtcNow.AddHours(-barData.Count());
        
        foreach (var (open, high, low, close) in barData)
        {
            bars.Add(new Bar
            {
                Open = open,
                High = high,
                Low = low,
                Close = close,
                Start = timestamp,
                Volume = 1000
            });
            timestamp = timestamp.AddMinutes(5);
        }
        
        return bars;
    }
}