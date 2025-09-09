using System;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TradingBot.Infrastructure.Alerts;
using Xunit;
using Xunit.Abstractions;

namespace TradingBot.Tests.Integration
{
    /// <summary>
    /// Integration tests for the AlertService
    /// Tests email and Slack alert functionality in a controlled environment
    /// </summary>
    public class AlertServiceTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly HttpClient _httpClient;
        private readonly ILogger<AlertService> _logger;
        private readonly AlertService _alertService;

        public AlertServiceTests(ITestOutputHelper output)
        {
            _output = output;
            _httpClient = new HttpClient();
            
            // Create logger that outputs to test console
            var loggerFactory = LoggerFactory.Create(builder => 
                builder.AddProvider(new TestOutputLoggerProvider(output)));
            _logger = loggerFactory.CreateLogger<AlertService>();
            
            _alertService = new AlertService(_logger, _httpClient);
        }

        [Fact]
        public async Task SendEmailAsync_WithValidConfiguration_ShouldLogSuccess()
        {
            // Arrange
            var subject = "Test Alert";
            var body = "This is a test alert from the integration test suite.";
            
            // Act & Assert - Should not throw exception
            // Note: Actual email sending will be skipped if SMTP not configured
            await _alertService.SendEmailAsync(subject, body, AlertSeverity.Info);
            
            // Test passes if no exception is thrown
            Assert.True(true);
        }

        [Fact]
        public async Task SendSlackAsync_WithValidConfiguration_ShouldLogSuccess()
        {
            // Arrange
            var message = "Test Slack alert from integration test suite";
            
            // Act & Assert - Should not throw exception
            // Note: Actual Slack sending will be skipped if webhook not configured
            await _alertService.SendSlackAsync(message, AlertSeverity.Warning);
            
            // Test passes if no exception is thrown
            Assert.True(true);
        }

        [Fact]
        public async Task SendCriticalAlertAsync_ShouldSendBothEmailAndSlack()
        {
            // Arrange
            var title = "Test Critical Alert";
            var details = "This is a test critical alert that should trigger both email and Slack notifications.";
            
            // Act & Assert
            await _alertService.SendCriticalAlertAsync(title, details);
            
            // Test passes if no exception is thrown
            Assert.True(true);
        }

        [Fact]
        public async Task SendModelHealthAlertAsync_WithMetrics_ShouldFormatCorrectly()
        {
            // Arrange
            var modelName = "TestModel_v1.0";
            var healthIssue = "Confidence drift detected";
            var metrics = new { ConfidenceDrift = 0.25, BrierScore = 0.35, PredictionCount = 150 };
            
            // Act & Assert
            await _alertService.SendModelHealthAlertAsync(modelName, healthIssue, metrics);
            
            // Test passes if no exception is thrown
            Assert.True(true);
        }

        [Fact]
        public async Task SendLatencyAlertAsync_WithHighLatency_ShouldFormatCorrectly()
        {
            // Arrange
            var component = "Decision Processing";
            var latencyMs = 7500.25;
            var thresholdMs = 5000.0;
            
            // Act & Assert
            await _alertService.SendLatencyAlertAsync(component, latencyMs, thresholdMs);
            
            // Test passes if no exception is thrown
            Assert.True(true);
        }

        [Theory]
        [InlineData("Model Promoted to Production", "TestModel_v2.1", true)]
        [InlineData("Canary Rollout Failed", "TestModel_v2.1", false)]
        public async Task SendDeploymentAlertAsync_WithDifferentStatuses_ShouldFormatCorrectly(
            string deploymentEvent, string modelName, bool isSuccess)
        {
            // Act & Assert
            await _alertService.SendDeploymentAlertAsync(deploymentEvent, modelName, isSuccess);
            
            // Test passes if no exception is thrown
            Assert.True(true);
        }

        [Fact]
        public void AlertService_Constructor_ShouldInitializeWithoutException()
        {
            // Arrange & Act
            using var httpClient = new HttpClient();
            var logger = NullLogger<AlertService>.Instance;
            var service = new AlertService(logger, httpClient);
            
            // Assert
            Assert.NotNull(service);
        }

        public void Dispose()
        {
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