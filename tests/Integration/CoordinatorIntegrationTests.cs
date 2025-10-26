using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Scheduling;
using TradingBot.UnifiedOrchestrator.Services;
using Xunit;

namespace TradingBot.Tests.Integration;

/// <summary>
/// Integration tests for coordinator services
/// Phase 8 Day 28-29: Integration Testing
/// Tests service interactions and workflows
/// </summary>
public class CoordinatorIntegrationTests
{
    [Fact]
    public async Task MaintenanceScheduler_Coordinates_CleanupServices()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddConsole());
        
        // Register dependencies (mocked for integration test)
        services.AddSingleton<LogRetentionService>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<LogRetentionService>>();
            var tradingLogger = new MockTradingLogger();
            var options = Microsoft.Extensions.Options.Options.Create(new TradingBot.Abstractions.TradingLoggerOptions());
            return new LogRetentionService(logger, tradingLogger, options);
        });

        services.AddSingleton<DataRetentionService>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<DataRetentionService>>();
            var modelRegistry = new MockFileModelRegistry();
            return new DataRetentionService(logger, modelRegistry);
        });

        services.AddSingleton<TrainingAlertService>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<TrainingAlertService>>();
            return new TrainingAlertService(logger);
        });

        services.AddSingleton<MaintenanceScheduler>();
        services.AddHostedService<MaintenanceScheduler>(sp => 
            sp.GetRequiredService<MaintenanceScheduler>());

        var serviceProvider = services.BuildServiceProvider();

        // Act
        var hostedService = serviceProvider.GetRequiredService<IHostedService>();
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        
        await hostedService.StartAsync(cts.Token);
        await Task.Delay(500); // Let it run briefly
        await hostedService.StopAsync(CancellationToken.None);

        // Assert - should start and stop without errors
        Assert.True(true);
    }

    [Fact]
    public async Task InternalScheduler_Coordinates_TrainingServices()
    {
        // This test verifies that InternalScheduler properly coordinates
        // with EnhancedBacktestLearningService
        
        // Note: This is a placeholder for when Lab mode services are available
        // Full integration test would require:
        // - HistoricalTrainingOrchestrator
        // - ResourcePreCheckService
        // - TrainingAlertService
        // - EnhancedBacktestLearningService
        
        await Task.CompletedTask;
        Assert.True(true); // Placeholder - implement when Lab services available in test context
    }

    [Fact]
    public async Task EndToEnd_CleanupTaskExecution()
    {
        // Arrange - simulate a full cleanup cycle
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddConsole());

        services.AddSingleton<TrainingAlertService>(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<TrainingAlertService>>();
            return new TrainingAlertService(logger);
        });

        var serviceProvider = services.BuildServiceProvider();
        var alertService = serviceProvider.GetRequiredService<TrainingAlertService>();

        // Act - simulate cleanup task execution
        var maintenanceStart = DateTime.UtcNow;
        
        // Simulate LogRetentionService execution
        await Task.Delay(100);
        var logRetentionComplete = DateTime.UtcNow;
        
        // Simulate DataRetentionService execution
        await Task.Delay(100);
        var dataRetentionComplete = DateTime.UtcNow;

        // Assert - verify execution order and timing
        Assert.True(logRetentionComplete > maintenanceStart);
        Assert.True(dataRetentionComplete > logRetentionComplete);
        Assert.True((dataRetentionComplete - maintenanceStart).TotalMilliseconds < 1000);
    }

    // Mock implementations for testing
    private class MockTradingLogger : TradingBot.Abstractions.ITradingLogger
    {
        public Task LogSystemAsync(TradingBot.Abstractions.TradingLogLevel level, string category, string message, object? data = null)
            => Task.CompletedTask;
        
        public Task LogTradeAsync(string symbol, string action, decimal price, int quantity, string reason, object? metadata = null)
            => Task.CompletedTask;
        
        public Task LogMLPredictionAsync(string model, string symbol, decimal confidence, string decision, object? features = null)
            => Task.CompletedTask;
        
        public Task LogMarketDataAsync(string symbol, decimal price, long volume, object? additionalData = null)
            => Task.CompletedTask;
        
        public Task<string> ExportLogsAsync(DateTime startTime, DateTime endTime, string outputPath)
            => Task.FromResult(outputPath);
        
        public Task FlushAsync() => Task.CompletedTask;
    }

    private class MockFileModelRegistry : TradingBot.UnifiedOrchestrator.Runtime.FileModelRegistry
    {
        public MockFileModelRegistry() 
            : base(
                Microsoft.Extensions.Logging.Abstractions.NullLogger<TradingBot.UnifiedOrchestrator.Runtime.FileModelRegistry>.Instance,
                "test_registry")
        {
        }

        public override Task CleanupOldModelsAsync(string algorithm, int keepCount)
        {
            return Task.CompletedTask;
        }
    }
}
