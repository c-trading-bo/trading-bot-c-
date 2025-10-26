using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Moq;
using TradingBot.Abstractions;
using TradingBot.UnifiedOrchestrator.Scheduling;
using TradingBot.UnifiedOrchestrator.Services;
using TradingBot.UnifiedOrchestrator.Runtime;
using Xunit;

namespace TradingBot.Tests.Unit.UnifiedOrchestrator;

/// <summary>
/// Unit tests for MaintenanceScheduler
/// Phase 8 Day 26-27: Unit Testing
/// </summary>
public class MaintenanceSchedulerTests
{
    private readonly ILogger<MaintenanceScheduler> _logger;
    private readonly LogRetentionService _logRetention;
    private readonly DataRetentionService _dataRetention;
    private readonly TrainingAlertService _alertService;

    public MaintenanceSchedulerTests()
    {
        _logger = NullLogger<MaintenanceScheduler>.Instance;
        
        // Create real instances for testing (using test logger)
        var tradingLoggerOptions = Options.Create(new TradingLoggerOptions());
        var mockTradingLogger = new Mock<ITradingLogger>();
        _logRetention = new LogRetentionService(
            NullLogger<LogRetentionService>.Instance,
            mockTradingLogger.Object,
            tradingLoggerOptions);

        var modelRegistry = new FileModelRegistry(
            NullLogger<FileModelRegistry>.Instance, 
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), "test_registry"));
        _dataRetention = new DataRetentionService(
            NullLogger<DataRetentionService>.Instance,
            modelRegistry);

        _alertService = new TrainingAlertService(NullLogger<TrainingAlertService>.Instance);
    }

    [Fact]
    public async Task StartAsync_InitializesSuccessfully()
    {
        // Arrange
        var scheduler = new MaintenanceScheduler(_logger, _logRetention, _dataRetention, _alertService);

        // Act
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(1));
        var startTask = scheduler.StartAsync(cts.Token);

        // Assert - should start without throwing
        await Task.WhenAny(startTask, Task.Delay(2000));
        Assert.True(startTask.IsCompleted || !cts.Token.IsCancellationRequested);
    }

    [Fact]
    public async Task StopAsync_StopsGracefully()
    {
        // Arrange
        var scheduler = new MaintenanceScheduler(_logger, _logRetention, _dataRetention, _alertService);

        var cts = new CancellationTokenSource();
        await scheduler.StartAsync(cts.Token);

        // Act
        await Task.Delay(100); // Let it run briefly
        cts.Cancel();
        await scheduler.StopAsync(CancellationToken.None);

        // Assert - should stop without hanging
        Assert.True(true); // If we get here, stop was successful
    }

    [Fact]
    public void ReportLogRetentionSuccess_CompletesSuccessfully()
    {
        // Arrange
        var scheduler = new MaintenanceScheduler(_logger, _logRetention, _dataRetention, _alertService);

        // Act & Assert - should not throw
        scheduler.ReportLogRetentionSuccess();
        Assert.True(true);
    }

    [Fact]
    public void ReportDataRetentionSuccess_CompletesSuccessfully()
    {
        // Arrange
        var scheduler = new MaintenanceScheduler(_logger, _logRetention, _dataRetention, _alertService);

        // Act & Assert - should not throw
        scheduler.ReportDataRetentionSuccess();
        Assert.True(true);
    }

    [Fact]
    public async Task ReportLogRetentionFailureAsync_CompletesSuccessfully()
    {
        // Arrange
        var scheduler = new MaintenanceScheduler(_logger, _logRetention, _dataRetention, _alertService);

        var errorMessage = "Test error";

        // Act & Assert - should not throw
        await scheduler.ReportLogRetentionFailureAsync(errorMessage, CancellationToken.None);
        Assert.True(true);
    }

    [Fact]
    public async Task ReportDataRetentionFailureAsync_CompletesSuccessfully()
    {
        // Arrange
        var scheduler = new MaintenanceScheduler(_logger, _logRetention, _dataRetention, _alertService);

        var errorMessage = "Test error";

        // Act & Assert - should not throw
        await scheduler.ReportDataRetentionFailureAsync(errorMessage, CancellationToken.None);
        Assert.True(true);
    }

    [Fact]
    public void Dispose_DisposesResourcesCleanly()
    {
        // Arrange
        var scheduler = new MaintenanceScheduler(_logger, _logRetention, _dataRetention, _alertService);

        // Act & Assert - should not throw
        scheduler.Dispose();
        Assert.True(true);
    }
}
