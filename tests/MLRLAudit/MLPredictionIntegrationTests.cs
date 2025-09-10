using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Moq;
using Xunit;
using TradingBot.IntelligenceStack;
using TradingBot.RLAgent;

// Use alias to resolve ambiguous references
using IntelStackEnsemblePrediction = TradingBot.IntelligenceStack.EnsemblePrediction;
using IntelStackModelPrediction = TradingBot.IntelligenceStack.ModelPrediction;
using RLAgentEnsemblePrediction = TradingBot.RLAgent.EnsemblePrediction;
using RLAgentModelPrediction = TradingBot.RLAgent.ModelPrediction;

namespace TradingBot.Tests.Unit
{
    /// <summary>
    /// Unit tests for ML prediction integration
    /// Tests the GetOnlinePredictionAsync implementation and ML gating behavior
    /// </summary>
    public class MLPredictionIntegrationTests
    {
        private readonly Mock<ILogger<IntelligenceOrchestrator>> _mockLogger;
        private readonly Mock<IServiceProvider> _mockServiceProvider;
        private readonly Mock<OnnxEnsembleWrapper> _mockOnnxEnsemble;
        private readonly IntelligenceStackConfig _config;

        public MLPredictionIntegrationTests()
        {
            _mockLogger = new Mock<ILogger<IntelligenceOrchestrator>>();
            _mockServiceProvider = new Mock<IServiceProvider>();
            _mockOnnxEnsemble = new Mock<OnnxEnsembleWrapper>(
                Mock.Of<ILogger<OnnxEnsembleWrapper>>(),
                Options.Create(new OnnxEnsembleOptions()));
            
            _config = new IntelligenceStackConfig
            {
                ML = new MLConfig
                {
                    Confidence = new ConfidenceConfig
                    {
                        EdgeConversionOffset = 0.5,
                        EdgeConversionMultiplier = 2.0,
                        KellyClip = 0.25,
                        ConfidenceMultiplierOffset = 0.3,
                        ConfidenceMultiplierScale = 2.0
                    }
                }
            };
        }

        [Fact]
        public async Task GetOnlinePredictionAsync_WithOnnxEnsemble_ReturnsValidPrediction()
        {
            // Arrange
            var mockServiceProvider = new Mock<IServiceProvider>();
            var mockEnsemble = new Mock<OnnxEnsembleWrapper>(
                Mock.Of<ILogger<OnnxEnsembleWrapper>>(),
                Options.Create(new OnnxEnsembleOptions()));
            
            // Setup ONNX ensemble to return valid prediction
            var expectedPrediction = new IntelStackEnsemblePrediction
            {
                Confidence = 0.85,
                EnsembleResult = 0.75f,
                IsAnomaly = false,
                LatencyMs = 15.0,
                Predictions = new System.Collections.Generic.Dictionary<string, IntelStackModelPrediction>
                {
                    ["test_model"] = new IntelStackModelPrediction { Value = 0.75f, Confidence = 0.85, ModelName = "test_model" }
                }
            };

            mockEnsemble
                .Setup(x => x.PredictAsync(It.IsAny<float[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(expectedPrediction);

            mockServiceProvider
                .Setup(x => x.GetService(typeof(TradingBot.RLAgent.OnnxEnsembleWrapper)))
                .Returns(mockEnsemble.Object);

            var orchestrator = CreateIntelligenceOrchestrator(mockServiceProvider);

            // Act
            var result = await orchestrator.GetOnlinePredictionAsync("ES", "TestStrategy", CancellationToken.None);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("ES", result.Symbol);
            Assert.Equal(0.85, result.Confidence);
            Assert.Equal("BUY", result.Direction); // 0.75 > 0.55 threshold
            Assert.Equal("ensemble_TestStrategy", result.ModelId);
            Assert.True(result.IsValid);
            Assert.Contains("ensemble_result", result.Metadata.Keys);
            Assert.Contains("latency_ms", result.Metadata.Keys);
            Assert.Contains("is_anomaly", result.Metadata.Keys);
        }

        [Fact]
        public async Task GetOnlinePredictionAsync_WithoutOnnxEnsemble_FallsBackToLatestPrediction()
        {
            // Arrange
            var mockServiceProvider = new Mock<IServiceProvider>();
            mockServiceProvider
                .Setup(x => x.GetService(typeof(TradingBot.RLAgent.OnnxEnsembleWrapper)))
                .Returns((object?)null);

            var orchestrator = CreateIntelligenceOrchestrator(mockServiceProvider);

            // Act
            var result = await orchestrator.GetOnlinePredictionAsync("ES", "TestStrategy", CancellationToken.None);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("ES", result.Symbol);
            // Should get fallback prediction with reasonable confidence
            Assert.True(result.Confidence >= 0.05 && result.Confidence <= 0.95);
        }

        [Fact]
        public async Task GetOnlinePredictionAsync_WithAnomalousInput_ReturnsInvalidPrediction()
        {
            // Arrange
            var mockServiceProvider = new Mock<IServiceProvider>();
            var mockEnsemble = new Mock<OnnxEnsembleWrapper>(
                Mock.Of<ILogger<OnnxEnsembleWrapper>>(),
                Options.Create(new OnnxEnsembleOptions()));
            
            // Setup ONNX ensemble to return anomalous prediction
            var anomalousPrediction = new IntelStackEnsemblePrediction
            {
                Confidence = 0.3,
                EnsembleResult = 0.1f,
                IsAnomaly = true,
                LatencyMs = 25.0,
                Predictions = new System.Collections.Generic.Dictionary<string, IntelStackModelPrediction>()
            };

            mockEnsemble
                .Setup(x => x.PredictAsync(It.IsAny<float[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(anomalousPrediction);

            mockServiceProvider
                .Setup(x => x.GetService(typeof(TradingBot.RLAgent.OnnxEnsembleWrapper)))
                .Returns(mockEnsemble.Object);

            var orchestrator = CreateIntelligenceOrchestrator(mockServiceProvider);

            // Act
            var result = await orchestrator.GetOnlinePredictionAsync("ES", "TestStrategy", CancellationToken.None);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("ES", result.Symbol);
            Assert.Equal(0.3, result.Confidence);
            Assert.False(result.IsValid); // Should be invalid due to anomaly
            Assert.True((bool)result.Metadata["is_anomaly"]);
        }

        [Theory]
        [InlineData(0.8, "BUY")]
        [InlineData(0.4, "SELL")]
        [InlineData(0.5, "HOLD")]
        public async Task GetOnlinePredictionAsync_DirectionMapping_CorrectlyMapsEnsembleResult(float ensembleResult, string expectedDirection)
        {
            // Arrange
            var mockServiceProvider = new Mock<IServiceProvider>();
            var mockEnsemble = new Mock<OnnxEnsembleWrapper>(
                Mock.Of<ILogger<OnnxEnsembleWrapper>>(),
                Options.Create(new OnnxEnsembleOptions()));
            
            var prediction = new IntelStackEnsemblePrediction
            {
                Confidence = 0.75,
                EnsembleResult = ensembleResult,
                IsAnomaly = false,
                LatencyMs = 10.0,
                Predictions = new System.Collections.Generic.Dictionary<string, IntelStackModelPrediction>()
            };

            mockEnsemble
                .Setup(x => x.PredictAsync(It.IsAny<float[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(prediction);

            mockServiceProvider
                .Setup(x => x.GetService(typeof(TradingBot.RLAgent.OnnxEnsembleWrapper)))
                .Returns(mockEnsemble.Object);

            var orchestrator = CreateIntelligenceOrchestrator(mockServiceProvider);

            // Act
            var result = await orchestrator.GetOnlinePredictionAsync("ES", "TestStrategy", CancellationToken.None);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(expectedDirection, result.Direction);
        }

        private IntelligenceOrchestrator CreateIntelligenceOrchestrator(Mock<IServiceProvider> mockServiceProvider)
        {
            // Create minimal mocks for required dependencies
            var mockRegimeDetector = new Mock<IRegimeDetector>();
            var mockFeatureStore = new Mock<IFeatureStore>();
            var mockModelRegistry = new Mock<IModelRegistry>();
            var mockCalibrationManager = new Mock<ICalibrationManager>();
            var mockDecisionLogger = new Mock<IDecisionLogger>();
            var mockStartupValidator = new Mock<TradingBot.Abstractions.IStartupValidator>();
            var mockIdempotentOrderService = new Mock<IIdempotentOrderService>();
            var mockHttpClient = new Mock<System.Net.Http.HttpClient>();
            var mockCloudFlowOptions = Options.Create(new CloudFlowOptions());

            var orchestrator = new IntelligenceOrchestrator(
                _mockLogger.Object,
                mockServiceProvider.Object,
                _config,
                mockRegimeDetector.Object,
                mockFeatureStore.Object,
                mockModelRegistry.Object,
                mockCalibrationManager.Object,
                mockDecisionLogger.Object,
                mockStartupValidator.Object,
                mockIdempotentOrderService.Object,
                new System.Net.Http.HttpClient(),
                mockCloudFlowOptions);

            return orchestrator;
        }
    }

    /// <summary>
    /// Minimal test configuration classes
    /// </summary>
    public class IntelligenceStackConfig
    {
        public MLConfig ML { get; set; } = new();
    }

    public class MLConfig
    {
        public ConfidenceConfig Confidence { get; set; } = new();
    }

    public class ConfidenceConfig
    {
        public double EdgeConversionOffset { get; set; }
        public double EdgeConversionMultiplier { get; set; }
        public double KellyClip { get; set; }
        public double ConfidenceMultiplierOffset { get; set; }
        public double ConfidenceMultiplierScale { get; set; }
    }

    public class CloudFlowOptions
    {
        public string CloudEndpoint { get; set; } = "https://test.endpoint";
        public int TimeoutSeconds { get; set; } = 30;
    }

    public class OnnxEnsembleOptions
    {
        public int MaxQueueSize { get; set; } = 100;
        public int MaxConcurrentBatches { get; set; } = 4;
        public int MaxBatchSize { get; set; } = 8;
        public int BatchTimeoutMs { get; set; } = 100;
        public double AnomalyThreshold { get; set; } = 2.0;
        public bool BlockAnomalousInputs { get; set; } = false;
    }
}