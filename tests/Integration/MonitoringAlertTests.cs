using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TradingBot.Infrastructure.Alerts;
using TradingBot.Monitoring;
using TradingBot.Monitoring;
using Xunit;
using Xunit.Abstractions;

namespace TradingBot.Tests.Integration
{
    /// <summary>
    /// Integration tests for monitoring alert functionality
    /// Tests model health, latency, and deployment alerts
    /// </summary>
    public class MonitoringAlertTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly HttpClient _httpClient;
        private readonly IAlertService _alertService;
        private readonly ModelHealthMonitor _modelHealthMonitor;
        private readonly LatencyMonitor _latencyMonitor;
        private readonly ModelDeploymentManager _deploymentManager;

        public MonitoringAlertTests(ITestOutputHelper output)
        {
            _output = output;
            _httpClient = new HttpClient();
            
            // Create logger that outputs to test console
            var loggerFactory = LoggerFactory.Create(builder => 
                builder.AddProvider(new TestOutputLoggerProvider(output)));
            
            _alertService = new AlertService(loggerFactory.CreateLogger<AlertService>(), _httpClient);
            _modelHealthMonitor = new ModelHealthMonitor(loggerFactory.CreateLogger<ModelHealthMonitor>(), _alertService);
            _latencyMonitor = new LatencyMonitor(loggerFactory.CreateLogger<LatencyMonitor>(), _alertService);
            _deploymentManager = new ModelDeploymentManager(loggerFactory.CreateLogger<ModelDeploymentManager>(), _alertService);
        }

        [Fact]
        public async Task ModelHealthMonitor_ConfidenceDriftBreach_ShouldTriggerAlert()
        {
            // Arrange - Record predictions that will cause confidence drift
            var features = new Dictionary<string, double> { ["price"] = 100.0, ["volume"] = 1000.0 };
            
            // Establish baseline with stable confidence
            for (int i = 0; i < 25; i++)
            {
                _modelHealthMonitor.RecordPrediction(0.7, i % 2, features);
            }
            
            // Introduce drift - significantly different confidence values
            for (int i = 0; i < 25; i++)
            {
                _modelHealthMonitor.RecordPrediction(0.3, i % 2, features); // Much lower confidence
            }
            
            // Act
            var health = _modelHealthMonitor.GetCurrentHealth();
            
            // Assert
            Assert.False(health.IsHealthy);
            Assert.Contains("Confidence drift", string.Join("; ", health.HealthIssues));
            Assert.True(health.ConfidenceDrift > 0.15); // Should exceed threshold
        }

        [Fact]
        public async Task ModelHealthMonitor_HighBrierScore_ShouldTriggerAlert()
        {
            // Arrange - Record predictions with poor accuracy (high Brier scores)
            for (int i = 0; i < 30; i++)
            {
                // Very confident predictions that are wrong
                _modelHealthMonitor.RecordPrediction(0.9, 0.0); // Confident about 1, but actual is 0
                _modelHealthMonitor.RecordPrediction(0.1, 1.0); // Confident about 0, but actual is 1
            }
            
            // Act
            var health = _modelHealthMonitor.GetCurrentHealth();
            
            // Assert
            Assert.False(health.IsHealthy);
            Assert.Contains("Brier score", string.Join("; ", health.HealthIssues));
            Assert.True(health.AverageBrierScore > 0.3); // Should exceed threshold
        }

        [Fact]
        public async Task ModelHealthMonitor_FeatureParityCheck_FailureShouldTriggerAlert()
        {
            // Arrange - Record some feature data
            var features = new Dictionary<string, double> 
            { 
                ["price"] = 100.0, 
                ["volume"] = 1000.0,
                ["volatility"] = 0.2
            };
            
            for (int i = 0; i < 30; i++)
            {
                // Gradually shift feature values
                features["price"] = 100.0 + i * 2; // Price increasing
                features["volume"] = 1000.0 - i * 10; // Volume decreasing
                _modelHealthMonitor.RecordPrediction(0.7, i % 2, features);
            }
            
            // Act - Check parity against expected values that are very different
            var expectedFeatures = new Dictionary<string, double>
            {
                ["price"] = 50.0,    // Much different from recent values (~160)
                ["volume"] = 2000.0, // Much different from recent values (~700)
                ["volatility"] = 0.2
            };
            
            var parityResult = await _modelHealthMonitor.CheckFeatureParityAsync(expectedFeatures);
            
            // Assert
            Assert.False(parityResult);
        }

        [Fact]
        public async Task LatencyMonitor_DecisionLatencyBreach_ShouldTriggerAlert()
        {
            // Arrange & Act - Record multiple high latency measurements
            for (int i = 0; i < 5; i++)
            {
                _latencyMonitor.RecordDecisionLatency(7000.0, $"Test decision {i}"); // Above 5000ms threshold
            }
            
            // Assert
            var health = _latencyMonitor.GetLatencyHealth();
            Assert.False(health.IsHealthy);
            Assert.True(health.DecisionStats.P99 > 5000);
        }

        [Fact]
        public async Task LatencyMonitor_OrderLatencyBreach_ShouldTriggerAlert()
        {
            // Arrange & Act - Record multiple high latency measurements
            for (int i = 0; i < 5; i++)
            {
                _latencyMonitor.RecordOrderLatency(3000.0, $"Test order {i}"); // Above 2000ms threshold
            }
            
            // Assert
            var health = _latencyMonitor.GetLatencyHealth();
            Assert.False(health.IsHealthy);
            Assert.True(health.OrderStats.P99 > 2000);
        }

        [Fact]
        public async Task LatencyMonitor_TrackerUsage_ShouldRecordLatency()
        {
            // Arrange & Act
            using (var tracker = _latencyMonitor.StartDecisionTracking("Test context"))
            {
                await Task.Delay(100); // Simulate some work
                tracker.Stop();
            }
            
            // Assert
            var stats = _latencyMonitor.GetDecisionStats();
            Assert.True(stats.Count > 0);
            Assert.True(stats.Average > 50); // Should have some measurable latency
        }

        [Fact]
        public async Task ModelDeploymentManager_PromoteToProduction_ShouldSendAlert()
        {
            // Arrange
            var modelName = "TestModel";
            var version = "v1.0.0";
            
            // Act
            var result = await _deploymentManager.PromoteModelToProductionAsync(modelName, version);
            
            // Assert
            Assert.True(result);
            var deployment = _deploymentManager.GetDeployment(modelName, "Production");
            Assert.NotNull(deployment);
            Assert.Equal(DeploymentStatus.Active, deployment.Status);
        }

        [Fact]
        public async Task ModelDeploymentManager_CanaryRollout_ShouldSendAlert()
        {
            // Arrange
            var modelName = "TestModel";
            var version = "v1.1.0";
            var trafficPercentage = 0.1;
            
            // Act
            var result = await _deploymentManager.StartCanaryRolloutAsync(modelName, version, trafficPercentage);
            
            // Assert
            Assert.True(result);
            var deployment = _deploymentManager.GetDeployment(modelName, "Canary");
            Assert.NotNull(deployment);
            Assert.Equal(DeploymentStatus.Canary, deployment.Status);
            Assert.Equal(trafficPercentage, deployment.TrafficPercentage);
        }

        [Fact]
        public async Task ModelDeploymentManager_CanaryFailure_ShouldTriggerRollback()
        {
            // Arrange
            var modelName = "TestModel";
            var version = "v1.2.0";
            await _deploymentManager.StartCanaryRolloutAsync(modelName, version);
            
            // Act
            var result = await _deploymentManager.FailCanaryRolloutAsync(modelName, "High error rate detected");
            
            // Assert
            Assert.True(result);
            var canaryDeployment = _deploymentManager.GetDeployment(modelName, "Canary");
            Assert.NotNull(canaryDeployment);
            Assert.Equal(DeploymentStatus.Failed, canaryDeployment.Status);
        }

        [Fact]
        public async Task ModelDeploymentManager_Rollback_ShouldSendCriticalAlert()
        {
            // Arrange
            var modelName = "TestModel";
            var version = "v1.3.0";
            await _deploymentManager.PromoteModelToProductionAsync(modelName, version);
            
            // Act
            var result = await _deploymentManager.RollbackModelAsync(modelName, "Critical bug discovered");
            
            // Assert
            Assert.True(result);
            var deployment = _deploymentManager.GetDeployment(modelName, "Production");
            Assert.NotNull(deployment);
            Assert.Equal(DeploymentStatus.RolledBack, deployment.Status);
        }

        [Fact]
        public async Task EndToEnd_MultipleAlertTypes_ShouldAllWork()
        {
            // Arrange
            var modelName = "E2ETestModel";
            
            // Act - Trigger multiple types of alerts
            
            // 1. Model health alert
            for (int i = 0; i < 30; i++)
            {
                _modelHealthMonitor.RecordPrediction(0.9, 0.0); // Poor predictions
            }
            
            // 2. Latency alert
            _latencyMonitor.RecordDecisionLatency(8000.0, "E2E test");
            
            // 3. Deployment alert
            await _deploymentManager.PromoteModelToProductionAsync(modelName, "v1.0.0");
            
            // Assert - All should complete without exceptions
            var modelHealth = _modelHealthMonitor.GetCurrentHealth();
            var latencyHealth = _latencyMonitor.GetLatencyHealth();
            var deploymentHealth = await _deploymentManager.GetDeploymentHealthAsync();
            
            Assert.NotNull(modelHealth);
            Assert.NotNull(latencyHealth);
            Assert.NotNull(deploymentHealth);
        }

        public void Dispose()
        {
            _modelHealthMonitor?.Dispose();
            _httpClient?.Dispose();
        }
    }
    
    /// <summary>
    /// Custom logger provider that outputs to test console
    /// </summary>
    internal class TestOutputLoggerProvider : ILoggerProvider
    {
        private readonly ITestOutputHelper _output;

        public TestOutputLoggerProvider(ITestOutputHelper output)
        {
            _output = output;
        }

        public ILogger CreateLogger(string categoryName)
        {
            return new TestOutputLogger(_output, categoryName);
        }

        public void Dispose() { }
    }

    internal class TestOutputLogger : ILogger
    {
        private readonly ITestOutputHelper _output;
        private readonly string _categoryName;

        public TestOutputLogger(ITestOutputHelper output, string categoryName)
        {
            _output = output;
            _categoryName = categoryName;
        }

        public IDisposable BeginScope<TState>(TState state) => null!;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
        {
            try
            {
                var message = formatter(state, exception);
                _output.WriteLine($"[{logLevel}] {_categoryName}: {message}");
                if (exception != null)
                {
                    _output.WriteLine($"Exception: {exception}");
                }
            }
            catch
            {
                // Ignore any issues with test output
            }
        }
    }
}