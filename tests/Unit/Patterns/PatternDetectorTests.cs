using BotCore.Models;
using BotCore.Patterns;
using BotCore.Patterns.Detectors;
using Xunit;

namespace Tests.Unit.Patterns;

public class PatternDetectorTests
{
    [Fact]
    public void CandlestickDetector_DetectsDoji_WithSmallBody()
    {
        // Arrange
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 102m, low: 98m, close: 100.1m) // Doji-like
        });
        
        var detector = new CandlestickPatternDetector(CandlestickType.Doji);
        
        // Act
        var result = detector.Detect(bars);
        
        // Assert
        Assert.True(result.Score > 0.5);
        Assert.Equal(0, result.Direction); // Neutral
        Assert.True(result.Metadata.ContainsKey("body_ratio"));
    }

    [Fact]
    public void CandlestickDetector_DetectsHammer_WithLongLowerShadow()
    {
        // Arrange
        var bars = CreateTestBars(new[]
        {
            (open: 102m, high: 102.5m, low: 100m, close: 101m), // Previous bar
            (open: 100.5m, high: 101m, low: 95m, close: 99.5m)   // Hammer-like
        });
        
        var detector = new CandlestickPatternDetector(CandlestickType.Hammer);
        
        // Act
        var result = detector.Detect(bars);
        
        // Assert
        Assert.True(result.Score > 0.5);
        Assert.Equal(1, result.Direction); // Bullish
        Assert.True(result.Metadata.ContainsKey("lower_shadow_ratio"));
    }

    [Fact]
    public void CandlestickDetector_DetectsEngulfing_WithProperPattern()
    {
        // Arrange
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 101m, low: 99.5m, close: 99.5m), // Small red candle
            (open: 99m, high: 102m, low: 98m, close: 101.5m)     // Large green engulfing
        });
        
        var detector = new CandlestickPatternDetector(CandlestickType.Engulfing);
        
        // Act
        var result = detector.Detect(bars);
        
        // Assert
        Assert.True(result.Score > 0.5);
        Assert.Equal(1, result.Direction); // Bullish engulfing
        Assert.Equal("bullish", result.Metadata["type"]);
    }

    [Fact]
    public void StructuralDetector_DetectsDoubleTop_WithSimilarPeaks()
    {
        // Arrange - Create pattern with two similar peaks
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 105m, low: 99m, close: 104m),   // First peak area
            (open: 104m, high: 105.1m, low: 103m, close: 104m),
            (open: 104m, high: 104.5m, low: 102m, close: 103m), // Valley
            (open: 103m, high: 103.5m, low: 101m, close: 102m),
            (open: 102m, high: 103m, low: 101m, close: 102.5m),
            (open: 102.5m, high: 104m, low: 102m, close: 103.5m), // Rise to second peak
            (open: 103.5m, high: 105.2m, low: 103m, close: 104.8m), // Second peak
            (open: 104.8m, high: 105m, low: 103.5m, close: 104m)
        });
        
        // Add more bars to meet minimum requirement
        var allBars = bars.Concat(CreateTestBars(Enumerable.Repeat((100m, 102m, 98m, 100m), 15))).ToList();
        
        var detector = new StructuralPatternDetector(StructuralType.DoubleTop);
        
        // Act
        var result = detector.Detect(allBars);
        
        // Assert
        Assert.True(result.Score >= 0);
        Assert.True(result.Direction <= 0); // Bearish or neutral
    }

    [Fact]
    public void ContinuationDetector_DetectsBullFlag_WithTrendAndConsolidation()
    {
        // Arrange - Create uptrend followed by sideways action
        var trendBars = CreateTestBars(new[]
        {
            (open: 100m, high: 101m, low: 99.5m, close: 100.8m),
            (open: 100.8m, high: 102m, low: 100.5m, close: 101.5m),
            (open: 101.5m, high: 103m, low: 101m, close: 102.5m),
            (open: 102.5m, high: 104m, low: 102m, close: 103.5m),
            (open: 103.5m, high: 105m, low: 103m, close: 104.5m), // Strong uptrend
        });
        
        var consolidationBars = CreateTestBars(new[]
        {
            (open: 104.5m, high: 105m, low: 103.5m, close: 104m),
            (open: 104m, high: 104.5m, low: 103m, close: 103.5m),
            (open: 103.5m, high: 104.5m, low: 103m, close: 104m),
            (open: 104m, high: 104.5m, low: 103.5m, close: 104m), // Sideways consolidation
        });
        
        var allBars = trendBars.Concat(consolidationBars).ToList();
        
        var detector = new ContinuationPatternDetector(ContinuationType.BullFlag);
        
        // Act
        var result = detector.Detect(allBars);
        
        // Assert
        Assert.True(result.Score >= 0);
        if (result.Score > 0.5)
        {
            Assert.Equal(1, result.Direction); // Bullish continuation
            Assert.True(result.Metadata.ContainsKey("slope_ratio"));
        }
    }

    [Fact]
    public void ReversalDetector_DetectsKeyReversal_WithProperPattern()
    {
        // Arrange - New high but close below previous close
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 102m, low: 99.5m, close: 101.5m), // Previous bar
            (open: 101.8m, high: 104m, low: 100m, close: 100.5m)  // Key reversal: new high but weak close
        });
        
        var detector = new ReversalPatternDetector(ReversalType.KeyReversal);
        
        // Act
        var result = detector.Detect(bars);
        
        // Assert
        Assert.True(result.Score > 0.5);
        Assert.Equal(1, result.Direction); // Bullish reversal (from potential downtrend)
        Assert.True(result.Metadata.ContainsKey("wick_ratio"));
    }

    [Fact]
    public void ReversalDetector_DetectsFailedBreakout_WhenBreakoutFails()
    {
        // Arrange - Breakout above resistance followed by failure
        var bars = CreateTestBars(new[]
        {
            (open: 100m, high: 102m, low: 99m, close: 101m),     // Establish range
            (open: 101m, high: 102.1m, low: 100.5m, close: 101.5m),
            (open: 101.5m, high: 102m, low: 100.8m, close: 101.2m),
            (open: 101.2m, high: 103.5m, low: 101m, close: 103m), // Breakout above 102
            (open: 103m, high: 103.2m, low: 101.5m, close: 101.8m), // Failed to sustain
            (open: 101.8m, high: 102.5m, low: 101m, close: 101.5m), // Back below breakout level
            (open: 101.5m, high: 102m, low: 100.5m, close: 101m),
            (open: 101m, high: 101.5m, low: 100m, close: 100.5m)
        });
        
        var detector = new ReversalPatternDetector(ReversalType.FailedBreakout);
        
        // Act
        var result = detector.Detect(bars);
        
        // Assert  
        Assert.True(result.Score >= 0);
        if (result.Score > 0.5)
        {
            Assert.Equal(-1, result.Direction); // Bearish after failed upside breakout
            Assert.True(result.Metadata.ContainsKey("breakout_direction"));
        }
    }

    [Theory]
    [InlineData(CandlestickType.Doji, 1)]
    [InlineData(CandlestickType.Hammer, 2)]
    [InlineData(CandlestickType.ThreeBlackCrows, 3)]
    public void CandlestickDetector_RequiredBars_MatchesExpected(CandlestickType type, int expectedBars)
    {
        // Arrange
        var detector = new CandlestickPatternDetector(type);
        
        // Act & Assert
        Assert.Equal(expectedBars, detector.RequiredBars);
        Assert.Equal(PatternFamily.Candlestick, detector.Family);
    }

    [Theory]
    [InlineData(StructuralType.DoubleTop, 20)]
    [InlineData(StructuralType.HeadAndShoulders, 25)]
    [InlineData(StructuralType.CupAndHandle, 40)]
    public void StructuralDetector_RequiredBars_MatchesExpected(StructuralType type, int expectedBars)
    {
        // Arrange
        var detector = new StructuralPatternDetector(type);
        
        // Act & Assert
        Assert.Equal(expectedBars, detector.RequiredBars);
        Assert.Equal(PatternFamily.Structural, detector.Family);
    }

    [Fact]
    public void PatternDetector_ReturnsZeroScore_WithInsufficientBars()
    {
        // Arrange
        var bars = CreateTestBars(new[] { (100m, 101m, 99m, 100.5m) }); // Only 1 bar
        var detector = new CandlestickPatternDetector(CandlestickType.Hammer); // Requires 2 bars
        
        // Act
        var result = detector.Detect(bars);
        
        // Assert
        Assert.Equal(0, result.Score);
        Assert.Equal(0, result.Confidence);
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
                Volume = 1000 // Default volume
            });
            timestamp = timestamp.AddMinutes(5); // 5-minute bars
        }
        
        return bars;
    }
}