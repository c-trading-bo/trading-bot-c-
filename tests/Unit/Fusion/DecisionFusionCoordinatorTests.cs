using Xunit;
using Microsoft.Extensions.Logging;
using Moq;
using BotCore.Fusion;
using BotCore.Strategy;
using BotCore.StrategyDsl;
using System.Collections.Generic;
using System;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Tests.Fusion;

/// <summary>
/// Unit tests for DecisionFusionCoordinator - validates ML decision fusion logic
/// </summary>
public sealed class DecisionFusionCoordinatorTests
{
    private readonly Mock<IStrategyKnowledgeGraph> _knowledgeGraphMock;
    private readonly Mock<IUcbStrategyChooser> _ucbChooserMock;
    private readonly Mock<IPpoSizer> _ppoSizerMock;
    private readonly Mock<IMLConfigurationService> _mlConfigMock;
    private readonly Mock<IMetrics> _metricsMock;
    private readonly Mock<ILogger<DecisionFusionCoordinator>> _loggerMock;
    private readonly Mock<IServiceProvider> _serviceProviderMock;
    private readonly DecisionFusionCoordinator _fusionCoordinator;

    public DecisionFusionCoordinatorTests()
    {
        _knowledgeGraphMock = new Mock<IStrategyKnowledgeGraph>();
        _ucbChooserMock = new Mock<IUcbStrategyChooser>();
        _ppoSizerMock = new Mock<IPpoSizer>();
        _mlConfigMock = new Mock<IMLConfigurationService>();
        _metricsMock = new Mock<IMetrics>();
        _loggerMock = new Mock<ILogger<DecisionFusionCoordinator>>();
        _serviceProviderMock = new Mock<IServiceProvider>();

        _fusionCoordinator = new DecisionFusionCoordinator(
            _knowledgeGraphMock.Object,
            _ucbChooserMock.Object,
            _ppoSizerMock.Object,
            _mlConfigMock.Object,
            _metricsMock.Object,
            _loggerMock.Object,
            _serviceProviderMock.Object);
    }

    [Fact]
    public async Task DecideAsync_BothSystemsAgreeAboveThreshold_ReturnsKnowledgeGraphRecommendation()
    {
        // Arrange
        const string symbol = "ES";
        var knowledgeRec = CreateStrategyRecommendation("S2", StrategyIntent.Buy, 0.8);
        var ucbPrediction = ("S2", StrategyIntent.Buy, 0.7);
        var config = CreateConfigDictionary(minConfidence: 0.65, holdOnDisagree: 1);

        SetupMocks(symbol, new[] { knowledgeRec }, ucbPrediction, config);

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("S2", result.StrategyName);
        Assert.Equal(StrategyIntent.Buy, result.Intent);
    }

    [Fact]
    public async Task DecideAsync_SystemsDisagreeWithHoldOnDisagree_ReturnsNull()
    {
        // Arrange
        const string symbol = "ES";
        var knowledgeRec = CreateStrategyRecommendation("S2", StrategyIntent.Buy, 0.8);
        var ucbPrediction = ("S6", StrategyIntent.Sell, 0.7); // Different strategy and direction
        var config = CreateConfigDictionary(minConfidence: 0.65, holdOnDisagree: 1);

        SetupMocks(symbol, new[] { knowledgeRec }, ucbPrediction, config);

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public async Task DecideAsync_SystemsDisagreeWithoutHoldOnDisagree_ReturnsKnowledgeGraphRecommendation()
    {
        // Arrange
        const string symbol = "ES";
        var knowledgeRec = CreateStrategyRecommendation("S2", StrategyIntent.Buy, 0.8);
        var ucbPrediction = ("S6", StrategyIntent.Sell, 0.7);
        var config = CreateConfigDictionary(minConfidence: 0.65, holdOnDisagree: 0); // Allow disagreement

        SetupMocks(symbol, new[] { knowledgeRec }, ucbPrediction, config);

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("S2", result.StrategyName); // Should prefer knowledge graph
        Assert.Equal(StrategyIntent.Buy, result.Intent);
    }

    [Fact]
    public async Task DecideAsync_BlendedConfidenceBelowThreshold_ReturnsNull()
    {
        // Arrange
        const string symbol = "ES";
        var knowledgeRec = CreateStrategyRecommendation("S2", StrategyIntent.Buy, 0.5); // Low confidence
        var ucbPrediction = ("S2", StrategyIntent.Buy, 0.4); // Low confidence
        var config = CreateConfigDictionary(knowledgeWeight: 0.6, ucbWeight: 0.4, minConfidence: 0.65);

        SetupMocks(symbol, new[] { knowledgeRec }, ucbPrediction, config);

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.Null(result);
        
        // Blended score should be: 0.6 * 0.5 + 0.4 * 0.4 = 0.46, which is below 0.65 threshold
    }

    [Fact]
    public async Task DecideAsync_OnlyKnowledgeGraphRecommendation_ReturnsRecommendation()
    {
        // Arrange
        const string symbol = "ES";
        var knowledgeRec = CreateStrategyRecommendation("S2", StrategyIntent.Buy, 0.8);
        var ucbPrediction = ("", StrategyIntent.Buy, 0.0); // No UCB recommendation
        var config = CreateConfigDictionary(minConfidence: 0.65);

        SetupMocks(symbol, new[] { knowledgeRec }, ucbPrediction, config);

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("S2", result.StrategyName);
        Assert.Equal(StrategyIntent.Buy, result.Intent);
    }

    [Fact]
    public async Task DecideAsync_OnlyUcbRecommendation_ReturnsUcbRecommendation()
    {
        // Arrange
        const string symbol = "ES";
        var ucbPrediction = ("S6", StrategyIntent.Sell, 0.8);
        var config = CreateConfigDictionary(minConfidence: 0.65);

        SetupMocks(symbol, new StrategyRecommendation[0], ucbPrediction, config); // No knowledge graph

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("S6", result.StrategyName);
        Assert.Equal(StrategyIntent.Sell, result.Intent);
    }

    [Fact]
    public async Task DecideAsync_NoRecommendationsFromEitherSystem_ReturnsNull()
    {
        // Arrange
        const string symbol = "ES";
        var ucbPrediction = ("", StrategyIntent.Buy, 0.0);
        var config = CreateConfigDictionary(minConfidence: 0.65);

        SetupMocks(symbol, new StrategyRecommendation[0], ucbPrediction, config);

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public async Task DecideAsync_InvalidSymbol_ThrowsArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => _fusionCoordinator.DecideAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => _fusionCoordinator.DecideAsync(null!));
        await Assert.ThrowsAsync<ArgumentException>(() => _fusionCoordinator.DecideAsync("   "));
    }

    [Fact]
    public async Task DecideAsync_ExceptionInKnowledgeGraph_HandlesGracefullyAndUsesUcb()
    {
        // Arrange
        const string symbol = "ES";
        var ucbPrediction = ("S6", StrategyIntent.Buy, 0.8);
        var config = CreateConfigDictionary(minConfidence: 0.65);

        _knowledgeGraphMock
            .Setup(kg => kg.EvaluateAsync(symbol, It.IsAny<DateTime>(), It.IsAny<CancellationToken>()))
            .ThrowsAsync(new InvalidOperationException("Knowledge graph failed"));

        _ucbChooserMock
            .Setup(ucb => ucb.PredictAsync(symbol, It.IsAny<CancellationToken>()))
            .ReturnsAsync(ucbPrediction);

        _mlConfigMock
            .Setup(cfg => cfg.GetConfigurationAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(config);

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.Null(result); // Should return null on exception for safety
    }

    [Theory]
    [InlineData(0.7, 0.6, 0.8, 0.2, 0.68)] // 0.8 * 0.7 + 0.2 * 0.6 = 0.68
    [InlineData(0.5, 0.8, 0.6, 0.4, 0.62)] // 0.6 * 0.5 + 0.4 * 0.8 = 0.62
    [InlineData(0.9, 0.1, 1.0, 0.0, 0.9)]  // 1.0 * 0.9 + 0.0 * 0.1 = 0.9
    public async Task DecideAsync_BlendingWeights_CalculatesCorrectly(double knowledgeScore, double ucbScore, 
        double knowledgeWeight, double ucbWeight, double expectedBlended)
    {
        // Arrange
        const string symbol = "ES";
        var knowledgeRec = CreateStrategyRecommendation("S2", StrategyIntent.Buy, knowledgeScore);
        var ucbPrediction = ("S2", StrategyIntent.Buy, ucbScore);
        var config = CreateConfigDictionary(
            knowledgeWeight: knowledgeWeight, 
            ucbWeight: ucbWeight, 
            minConfidence: 0.5); // Low threshold to allow testing

        SetupMocks(symbol, new[] { knowledgeRec }, ucbPrediction, config);

        // Act
        var result = await _fusionCoordinator.DecideAsync(symbol);

        // Assert
        Assert.NotNull(result);
    }

    private static StrategyRecommendation CreateStrategyRecommendation(string strategyName, StrategyIntent intent, double confidence)
    {
        return new StrategyRecommendation(
            strategyName,
            intent,
            confidence,
            new List<StrategyEvidence> { new("test_evidence", 1.0, "Test evidence") },
            new[] { "TestTag" });
    }

    private static Dictionary<string, object> CreateConfigDictionary(double knowledgeWeight = 0.6, double ucbWeight = 0.4, 
        double minConfidence = 0.65, int holdOnDisagree = 1)
    {
        return new Dictionary<string, object>
        {
            ["fusion_knowledge_weight"] = knowledgeWeight,
            ["fusion_ucb_weight"] = ucbWeight,
            ["fusion_min_confidence"] = minConfidence,
            ["fusion_hold_on_disagree"] = (double)holdOnDisagree
        };
    }

    private void SetupMocks(string symbol, IReadOnlyList<StrategyRecommendation> knowledgeRecs, 
        (string Strategy, StrategyIntent Intent, double Score) ucbPrediction, Dictionary<string, object> config)
    {
        _knowledgeGraphMock
            .Setup(kg => kg.EvaluateAsync(symbol, It.IsAny<DateTime>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(knowledgeRecs);

        _ucbChooserMock
            .Setup(ucb => ucb.PredictAsync(symbol, It.IsAny<CancellationToken>()))
            .ReturnsAsync(ucbPrediction);

        _mlConfigMock
            .Setup(cfg => cfg.GetConfigurationAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(config);
        
        _ppoSizerMock
            .Setup(ppo => ppo.PredictSizeAsync(It.IsAny<string>(), It.IsAny<StrategyIntent>(), It.IsAny<double>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(1.0);
    }
}