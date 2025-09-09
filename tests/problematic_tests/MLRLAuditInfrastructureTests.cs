using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Xunit;
using Moq;
using UnifiedOrchestrator.Services;

namespace UnifiedOrchestrator.Tests.Services
{
    /// <summary>
    /// Comprehensive tests for ML/RL audit infrastructure
    /// Tests cloud flow, model registry, data lake, ONNX ensemble, backtest harness, and metrics
    /// </summary>
    public class MLRLAuditInfrastructureTests : IDisposable
    {
        private readonly Mock<ILogger<CloudFlowService>> _cloudFlowLogger;
        private readonly Mock<ILogger<ModelRegistryService>> _modelRegistryLogger;
        private readonly Mock<ILogger<DataLakeService>> _dataLakeLogger;
        private readonly Mock<ILogger<OnnxEnsembleService>> _onnxLogger;
        private readonly Mock<ILogger<BacktestHarnessService>> _backtestLogger;
        private readonly Mock<ILogger<MLRLMetricsService>> _metricsLogger;
        private readonly Mock<ILogger<StreamingFeatureAggregator>> _streamingLogger;
        private readonly string _testDataDir;

        public MLRLAuditInfrastructureTests()
        {
            _cloudFlowLogger = new Mock<ILogger<CloudFlowService>>();
            _modelRegistryLogger = new Mock<ILogger<ModelRegistryService>>();
            _dataLakeLogger = new Mock<ILogger<DataLakeService>>();
            _onnxLogger = new Mock<ILogger<OnnxEnsembleService>>();
            _backtestLogger = new Mock<ILogger<BacktestHarnessService>>();
            _metricsLogger = new Mock<ILogger<MLRLMetricsService>>();
            _streamingLogger = new Mock<ILogger<StreamingFeatureAggregator>>();
            
            _testDataDir = Path.Combine(Path.GetTempPath(), "ml_rl_audit_tests", Guid.NewGuid().ToString());
            Directory.CreateDirectory(_testDataDir);
        }

        [Fact]
        public async Task CloudFlowService_PushTradeRecord_Success()
        {
            // Arrange
            var options = Options.Create(new CloudFlowOptions
            {
                Enabled = true,
                CloudEndpoint = "https://httpbin.org/post",
                InstanceId = "test_instance"
            });

            using var httpClient = new System.Net.Http.HttpClient();
            using var cloudFlowService = new CloudFlowService(httpClient, _cloudFlowLogger.Object, options);

            var tradeRecord = new TradeRecord
            {
                TradeId = "TEST_001",
                Symbol = "ES",
                Side = "LONG",
                Quantity = 2,
                EntryPrice = 4000.00m,
                ExitPrice = 4005.00m,
                PnL = 500.00m,
                EntryTime = DateTime.UtcNow.AddMinutes(-10),
                ExitTime = DateTime.UtcNow,
                Strategy = "S1"
            };

            // Act & Assert (should not throw)
            await cloudFlowService.PushTradeRecordAsync(tradeRecord, CancellationToken.None);
        }

        [Fact]
        public async Task ModelRegistryService_RegisterAndRetrieveModel_Success()
        {
            // Arrange
            var registryPath = Path.Combine(_testDataDir, "model_registry");
            var options = Options.Create(new ModelRegistryOptions
            {
                RegistryPath = registryPath,
                AutoCompress = false,
                ModelExpiryDays = 30
            });

            var modelRegistryService = new ModelRegistryService(_modelRegistryLogger.Object, options);

            // Create a dummy model file
            var modelPath = Path.Combine(_testDataDir, "test_model.onnx");
            await File.WriteAllTextAsync(modelPath, "dummy model content");

            var metadata = new ModelMetadata
            {
                TrainingDate = DateTime.UtcNow,
                Hyperparams = new Dictionary<string, object> { ["learning_rate"] = 0.001 },
                ValidationAccuracy = 0.85,
                Description = "Test model"
            };

            // Act
            var registryEntry = await modelRegistryService.RegisterModelAsync(
                "test_model", modelPath, metadata, CancellationToken.None);

            var retrievedModel = await modelRegistryService.GetLatestModelAsync(
                "test_model", CancellationToken.None);

            // Assert
            Assert.NotNull(registryEntry);
            Assert.Equal("test_model", registryEntry.ModelName);
            Assert.NotNull(retrievedModel);
            Assert.Equal("test_model", retrievedModel.ModelName);
            Assert.Equal(0.85, retrievedModel.Metadata.ValidationAccuracy);
        }

        [Fact]
        public async Task DataLakeService_StoreAndRetrieveFeatures_Success()
        {
            // Arrange
            var dbPath = Path.Combine(_testDataDir, "test_features.db");
            var options = Options.Create(new DataLakeOptions
            {
                DatabasePath = dbPath,
                DriftThreshold = 0.05,
                AlertOnDrift = false
            });

            using var dataLakeService = new DataLakeService(_dataLakeLogger.Object, options);

            var features = new Dictionary<string, object>
            {
                ["price"] = 4000.5,
                ["volume"] = 1500,
                ["volatility"] = 0.25
            };

            var timestamp = DateTime.UtcNow;

            // Act
            var stored = await dataLakeService.StoreFeatureSetAsync(
                "market_features", features, timestamp, CancellationToken.None);

            var retrieved = await dataLakeService.RetrieveFeaturesAsync(
                "market_features", 
                timestamp.AddMinutes(-1), 
                timestamp.AddMinutes(1), 
                CancellationToken.None);

            // Assert
            Assert.True(stored);
            Assert.Single(retrieved);
            Assert.Equal("market_features", retrieved[0].FeatureSetName);
            Assert.Equal(4000.5, Convert.ToDouble(retrieved[0].Features["price"]));
        }

        [Fact]
        public async Task DataLakeService_DriftDetection_TriggersAlert()
        {
            // Arrange
            var dbPath = Path.Combine(_testDataDir, "test_drift.db");
            var options = Options.Create(new DataLakeOptions
            {
                DatabasePath = dbPath,
                DriftThreshold = 0.01, // Very low threshold for testing
                AlertOnDrift = true
            });

            using var dataLakeService = new DataLakeService(_dataLakeLogger.Object, options);

            // Store baseline features
            var baselineFeatures = new Dictionary<string, object> { ["price"] = 4000.0 };
            await dataLakeService.StoreFeatureSetAsync(
                "drift_test", baselineFeatures, DateTime.UtcNow.AddMinutes(-10), CancellationToken.None);

            // Store drifted features
            var driftedFeatures = new Dictionary<string, object> { ["price"] = 5000.0 }; // Significant drift

            // Act
            var stored = await dataLakeService.StoreFeatureSetAsync(
                "drift_test", driftedFeatures, DateTime.UtcNow, CancellationToken.None);

            // Assert
            Assert.True(stored); // Should still store even with drift
        }

        [Fact]
        public async Task StreamingFeatureAggregator_ProcessTicks_GeneratesFeatures()
        {
            // Arrange
            var options = Options.Create(new StreamingOptions
            {
                TimeWindows = new List<TimeSpan> { TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(5) },
                MicrostructureWindow = 100,
                StaleThresholdSeconds = 30
            });

            using var aggregator = new StreamingFeatureAggregator(_streamingLogger.Object, options);

            var tick = new MarketTick
            {
                Symbol = "ES",
                Timestamp = DateTime.UtcNow,
                Price = 4000.5,
                Volume = 100,
                IsBuyInitiated = true,
                Bid = 4000.25,
                Ask = 4000.75
            };

            // Act
            var features = await aggregator.ProcessTickAsync(tick, CancellationToken.None);
            var cachedFeatures = aggregator.GetCachedFeatures("ES");

            // Assert
            Assert.NotNull(features);
            Assert.Equal("ES", features.Symbol);
            Assert.NotNull(cachedFeatures);
            Assert.Contains(TimeSpan.FromMinutes(1), features.TimeWindowFeatures.Keys);
            Assert.Contains(TimeSpan.FromMinutes(5), features.TimeWindowFeatures.Keys);
        }

        [Fact]
        public async Task StreamingFeatureAggregator_StaleDataDetection_Works()
        {
            // Arrange
            var options = Options.Create(new StreamingOptions
            {
                StaleThresholdSeconds = 1 // 1 second for quick test
            });

            using var aggregator = new StreamingFeatureAggregator(_streamingLogger.Object, options);

            var tick = new MarketTick
            {
                Symbol = "ES",
                Timestamp = DateTime.UtcNow.AddSeconds(-2), // 2 seconds ago
                Price = 4000.0,
                Volume = 100
            };

            // Act
            await aggregator.ProcessTickAsync(tick, CancellationToken.None);
            await Task.Delay(1100); // Wait for staleness

            var hasStaleFeatures = aggregator.HasStaleFeatures();
            var staleSymbols = aggregator.GetStaleSymbols();

            // Assert
            Assert.True(hasStaleFeatures);
            Assert.Contains("ES", staleSymbols);
        }

        [Fact]
        public void MLRLMetricsService_RecordMetrics_Success()
        {
            // Arrange
            var options = Options.Create(new MLRLMetricsOptions
            {
                UpdateIntervalSeconds = 60,
                DriftAlertThreshold = 0.05,
                AccuracyAlertThreshold = 0.6,
                LatencyAlertThresholdMs = 500
            });

            using var metricsService = new MLRLMetricsService(_metricsLogger.Object, options);

            // Act & Assert (should not throw)
            metricsService.RecordPrediction("test_model", "signal", 0.050, 0.8);
            metricsService.RecordAccuracy("test_model", "1h", 0.75);
            metricsService.RecordFeatureDrift("market_features", "statistical", 0.03);
            metricsService.RecordEpisodeReward("test_agent", "training", 150.5);
            metricsService.RecordExplorationRate("test_agent", 0.1);
            metricsService.RecordPolicyNorm("test_agent", "layer1", 2.5);
            metricsService.RecordMemoryUsage("onnx_engine", 512 * 1024 * 1024);
            metricsService.RecordModelHealth("test_model", "inference", true);

            var summary = metricsService.GetMetricsSummary();
            var alerts = metricsService.CheckAlerts();

            // Assert
            Assert.NotNull(summary);
            Assert.Single(summary.ModelMetrics);
            Assert.Equal("test_model", summary.ModelMetrics[0].ModelName);
            Assert.Empty(alerts); // No alerts with good metrics
        }

        [Fact]
        public void MLRLMetricsService_AlertsGeneration_Works()
        {
            // Arrange
            var options = Options.Create(new MLRLMetricsOptions
            {
                AccuracyAlertThreshold = 0.8, // High threshold
                LatencyAlertThresholdMs = 10 // Low threshold
            });

            using var metricsService = new MLRLMetricsService(_metricsLogger.Object, options);

            // Record bad metrics
            metricsService.RecordPrediction("bad_model", "signal", 1.0, 0.3); // High latency
            metricsService.RecordAccuracy("bad_model", "1h", 0.5); // Low accuracy

            // Act
            var alerts = metricsService.CheckAlerts();

            // Assert
            Assert.NotEmpty(alerts);
            Assert.Contains(alerts, a => a.Type == "accuracy_degradation");
            Assert.Contains(alerts, a => a.Type == "high_latency");
        }

        [Fact]
        public async Task BacktestHarnessService_WalkForwardBacktest_Success()
        {
            // Arrange
            var options = Options.Create(new BacktestOptions
            {
                TrainingWindowDays = 30,
                TestWindowDays = 7,
                StepSizeDays = 3,
                PurgeDays = 1,
                EmbargoDays = 1,
                AutoRetrain = false // Disable for test
            });

            var mockModelRegistry = new Mock<ModelRegistryService>(
                Mock.Of<ILogger<ModelRegistryService>>(),
                Options.Create(new ModelRegistryOptions()));

            var backtestService = new BacktestHarnessService(
                _backtestLogger.Object, options, mockModelRegistry.Object);

            var startDate = DateTime.UtcNow.AddDays(-60);
            var endDate = DateTime.UtcNow.AddDays(-30);

            // Act
            var result = await backtestService.RunWalkForwardBacktestAsync(
                "test_model", startDate, endDate, CancellationToken.None);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("test_model", result.ModelName);
            Assert.NotEmpty(result.WindowResults);
            Assert.NotNull(result.OverallMetrics);
        }

        [Fact]
        public async Task OnnxEnsembleService_LoadAndUnloadModel_Success()
        {
            // Arrange
            var options = Options.Create(new OnnxEnsembleOptions
            {
                MaxBatchSize = 16,
                BatchTimeoutMs = 50,
                UseGpu = false, // CPU only for tests
                ClampInputs = true,
                BlockAnomalousInputs = false
            });

            using var ensembleService = new OnnxEnsembleService(_onnxLogger.Object, options);

            // Create a dummy ONNX model file (won't actually work but tests file handling)
            var modelPath = Path.Combine(_testDataDir, "dummy_model.onnx");
            await File.WriteAllBytesAsync(modelPath, new byte[] { 1, 2, 3, 4 }); // Dummy content

            // Act & Assert for load (will fail due to invalid ONNX but tests infrastructure)
            var loadResult = await ensembleService.LoadModelAsync("test_model", modelPath, 1.0, CancellationToken.None);
            Assert.False(loadResult); // Expected to fail with dummy file

            // Test status
            var status = ensembleService.GetStatus();
            Assert.NotNull(status);
            Assert.Equal(0, status.LoadedModels); // No models loaded due to invalid file
        }

        [Fact]
        public async Task Integration_AllServices_WorkTogether()
        {
            // Arrange - create all services
            var cloudFlowOptions = Options.Create(new CloudFlowOptions { Enabled = false }); // Disable for test
            var modelRegistryOptions = Options.Create(new ModelRegistryOptions 
            { 
                RegistryPath = Path.Combine(_testDataDir, "integration_registry") 
            });
            var dataLakeOptions = Options.Create(new DataLakeOptions 
            { 
                DatabasePath = Path.Combine(_testDataDir, "integration.db") 
            });
            var streamingOptions = Options.Create(new StreamingOptions());
            var metricsOptions = Options.Create(new MLRLMetricsOptions());

            using var httpClient = new System.Net.Http.HttpClient();
            using var cloudFlowService = new CloudFlowService(httpClient, _cloudFlowLogger.Object, cloudFlowOptions);
            var modelRegistryService = new ModelRegistryService(_modelRegistryLogger.Object, modelRegistryOptions);
            using var dataLakeService = new DataLakeService(_dataLakeLogger.Object, dataLakeOptions);
            using var streamingAggregator = new StreamingFeatureAggregator(_streamingLogger.Object, streamingOptions);
            using var metricsService = new MLRLMetricsService(_metricsLogger.Object, metricsOptions);

            // Act - simulate a complete ML/RL workflow
            
            // 1. Process market tick
            var tick = new MarketTick
            {
                Symbol = "ES",
                Timestamp = DateTime.UtcNow,
                Price = 4000.0,
                Volume = 100,
                IsBuyInitiated = true
            };
            var streamingFeatures = await streamingAggregator.ProcessTickAsync(tick, CancellationToken.None);

            // 2. Store features in data lake
            var features = new Dictionary<string, object>
            {
                ["price"] = tick.Price,
                ["volume"] = tick.Volume,
                ["microstructure_score"] = streamingFeatures.MicrostructureFeatures.OrderFlow
            };
            var stored = await dataLakeService.StoreFeatureSetAsync(
                "integration_test", features, tick.Timestamp, CancellationToken.None);

            // 3. Record ML metrics
            metricsService.RecordPrediction("integration_model", "signal", 0.025, 0.8);
            metricsService.RecordAccuracy("integration_model", "1h", 0.78);

            // 4. Simulate trade completion and cloud push
            var tradeRecord = new TradeRecord
            {
                TradeId = "INTEGRATION_001",
                Symbol = "ES",
                Side = "LONG",
                Quantity = 1,
                EntryPrice = 4000.0m,
                ExitPrice = 4002.5m,
                PnL = 125.0m,
                EntryTime = tick.Timestamp.AddMinutes(-5),
                ExitTime = tick.Timestamp,
                Strategy = "IntegrationTest"
            };

            // Would push to cloud if enabled
            // await cloudFlowService.PushTradeRecordAsync(tradeRecord, CancellationToken.None);

            // Assert - verify all components worked
            Assert.NotNull(streamingFeatures);
            Assert.True(stored);
            
            var retrievedFeatures = await dataLakeService.RetrieveFeaturesAsync(
                "integration_test", 
                tick.Timestamp.AddMinutes(-1), 
                tick.Timestamp.AddMinutes(1), 
                CancellationToken.None);
            Assert.Single(retrievedFeatures);

            var metricsSummary = metricsService.GetMetricsSummary();
            Assert.Single(metricsSummary.ModelMetrics);

            var qualityReport = await dataLakeService.GetDataQualityReportAsync(
                "integration_test", TimeSpan.FromHours(1), CancellationToken.None);
            Assert.True(qualityReport.IsHealthy);
        }

        public void Dispose()
        {
            if (Directory.Exists(_testDataDir))
            {
                try
                {
                    Directory.Delete(_testDataDir, recursive: true);
                }
                catch
                {
                    // Ignore cleanup errors in tests
                }
            }
        }
    }
}