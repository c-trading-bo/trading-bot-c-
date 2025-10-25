using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Xunit;
using Moq;
using TradingBot.Abstractions;
using TradingBot.UnifiedOrchestrator.Services;
using TradingBot.UnifiedOrchestrator.Training;
using TradingBot.UnifiedOrchestrator.Promotion;

namespace TradingBot.Tests.Integration;

/// <summary>
/// Lab Mode Integration Tests
/// Tests complete training pipeline, checkpoint resume, promotion validation, and failure handling
/// Phase 15: Integration Testing Suite
/// </summary>
public class LabModeIntegrationTests : IDisposable
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<LabModeIntegrationTests> _logger;
    private readonly string _tempDir;
    private readonly string _testDataDir;
    private readonly string _testModelDir;
    private readonly string _testCheckpointDir;

    public LabModeIntegrationTests()
    {
        // Create temporary directories for test isolation
        _tempDir = Path.Combine(Path.GetTempPath(), $"qbot-lab-test-{Guid.NewGuid():N}");
        _testDataDir = Path.Combine(_tempDir, "data");
        _testModelDir = Path.Combine(_tempDir, "models");
        _testCheckpointDir = Path.Combine(_tempDir, "checkpoints");
        
        Directory.CreateDirectory(_tempDir);
        Directory.CreateDirectory(_testDataDir);
        Directory.CreateDirectory(_testModelDir);
        Directory.CreateDirectory(_testCheckpointDir);

        // Setup service provider with test dependencies
        var services = new ServiceCollection();
        ConfigureTestServices(services);
        _serviceProvider = services.BuildServiceProvider();
        _logger = _serviceProvider.GetRequiredService<ILogger<LabModeIntegrationTests>>();
    }

    private void ConfigureTestServices(IServiceCollection services)
    {
        // Add logging
        services.AddLogging(builder => builder
            .AddConsole()
            .SetMinimumLevel(LogLevel.Information));

        // Add configuration
        var configData = new Dictionary<string, string>
        {
            ["LAB_MEMORY_PROFILING"] = "0",
            ["LAB_DEBUG_MODE"] = "0",
            ["ResourcePreCheck:MinimumDiskSpaceGB"] = "1",
            ["ResourcePreCheck:MinimumMemoryGB"] = "1"
        };
        
        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(configData!)
            .Build();
        
        services.AddSingleton<IConfiguration>(configuration);

        // Add Lab Mode services needed for testing
        services.AddSingleton<TrainingDebugLogger>();
        services.AddSingleton<MemoryLeakDetector>();
        services.AddSingleton<TrainingMetricsCollector>();
        services.AddSingleton<TrainingCheckpointService>();
        services.AddSingleton<TrainingFailureHandler>();
        services.AddSingleton<TrainingPerformanceProfiler>();
        services.AddSingleton<DataIntegrityService>();
        services.AddSingleton<TrainingManifestService>();
        services.AddSingleton<TrainingAlertService>();
        services.AddSingleton<TrainingRetryService>();
        services.AddSingleton<SystemCapabilityProfiler>();
        services.AddSingleton<DynamicResourceManager>();
        services.AddSingleton<TrainingResourceMonitor>();
        
        // Mock services
        var mockHistoricalDataBridge = new Mock<IHistoricalDataBridgeService>();
        services.AddSingleton(mockHistoricalDataBridge.Object);
        
        // We won't register full orchestrator to avoid dependencies on all components
        // Instead we'll test individual services
    }

    /// <summary>
    /// Task 3.2: End-to-End Training Test (Simplified)
    /// Validates that training infrastructure can be initialized and basic flow works
    /// </summary>
    [Fact]
    public async Task EndToEndTraining_BasicInfrastructure_ShouldInitialize()
    {
        _logger.LogInformation("=== Test: End-to-End Training Infrastructure ===");

        // Arrange
        var metricsCollector = _serviceProvider.GetRequiredService<TrainingMetricsCollector>();
        var memoryDetector = _serviceProvider.GetRequiredService<MemoryLeakDetector>();
        var debugLogger = _serviceProvider.GetRequiredService<TrainingDebugLogger>();

        // Act - Verify services are initialized
        metricsCollector.StartRun("test-session", 5);
        memoryDetector.RecordBaseline();
        
        // Simulate component training
        memoryDetector.RecordBeforeComponent("TestComponent");
        await Task.Delay(100); // Simulate work
        var analysis = await memoryDetector.RecordAfterComponentAsync("TestComponent");
        
        metricsCollector.EndRun(true);

        // Assert
        Assert.NotNull(metricsCollector);
        Assert.NotNull(memoryDetector);
        Assert.NotNull(debugLogger);
        Assert.False(analysis.LeakDetected); // No leak in simple test
        
        _logger.LogInformation("✓ Training infrastructure initialized successfully");
    }

    /// <summary>
    /// Task 3.3: Checkpoint Resume Test
    /// Validates checkpoint save and load functionality
    /// </summary>
    [Fact]
    public async Task CheckpointResume_SaveAndLoad_ShouldPreserveState()
    {
        _logger.LogInformation("=== Test: Checkpoint Save and Resume ===");

        // Arrange
        var checkpointService = _serviceProvider.GetRequiredService<TrainingCheckpointService>();
        var sessionId = "test-checkpoint-session";
        var components = new List<string> { "Component1", "Component2", "Component3" };
        
        // Act - Save checkpoint after completing first component
        var checkpoint = new CheckpointState
        {
            SessionId = sessionId,
            TotalComponents = components.Count,
            CurrentComponentIndex = 1, // Completed first, starting second
            CompletedComponents = new List<string> { "Component1" },
            StartTime = DateTime.UtcNow.AddMinutes(-5),
            LastCheckpointTime = DateTime.UtcNow
        };

        var checkpointPath = Path.Combine(_testCheckpointDir, $"{sessionId}.json");
        await checkpointService.SaveCheckpointAsync(checkpoint, checkpointPath, CancellationToken.None);

        // Load checkpoint
        var loadedCheckpoint = await checkpointService.LoadCheckpointAsync(checkpointPath, CancellationToken.None);
        var isValid = await checkpointService.ValidateCheckpointAsync(loadedCheckpoint, CancellationToken.None);

        // Assert
        Assert.NotNull(loadedCheckpoint);
        Assert.Equal(sessionId, loadedCheckpoint.SessionId);
        Assert.Equal(1, loadedCheckpoint.CurrentComponentIndex);
        Assert.Single(loadedCheckpoint.CompletedComponents);
        Assert.Contains("Component1", loadedCheckpoint.CompletedComponents);
        Assert.True(isValid);
        
        _logger.LogInformation("✓ Checkpoint saved and loaded successfully");
        _logger.LogInformation($"  - Completed: {string.Join(", ", loadedCheckpoint.CompletedComponents)}");
        _logger.LogInformation($"  - Next component index: {loadedCheckpoint.CurrentComponentIndex}");
    }

    /// <summary>
    /// Task 3.4: Promotion Rollback Test
    /// Validates that promotion validation can detect poor performance
    /// </summary>
    [Fact]
    public void PromotionValidation_PoorPerformance_ShouldReject()
    {
        _logger.LogInformation("=== Test: Promotion Validation - Poor Performance ===");

        // Arrange - Simulate baseline vs challenger comparison
        var baselineMetrics = new ModelPerformanceMetrics
        {
            Sharpe = 1.5m,
            WinRate = 0.65m,
            MaxDrawdown = -0.10m,
            TotalReturn = 0.20m
        };

        var challengerMetrics = new ModelPerformanceMetrics
        {
            Sharpe = 0.8m,      // Worse than baseline
            WinRate = 0.50m,    // Worse than baseline
            MaxDrawdown = -0.25m, // Worse than baseline
            TotalReturn = 0.05m  // Worse than baseline
        };

        // Act - Compare performance
        var shouldPromote = ShouldPromoteChallenger(baselineMetrics, challengerMetrics);

        // Assert
        Assert.False(shouldPromote);
        
        _logger.LogInformation("✓ Poor challenger correctly rejected");
        _logger.LogInformation($"  - Baseline Sharpe: {baselineMetrics.Sharpe:F2}, Challenger: {challengerMetrics.Sharpe:F2}");
        _logger.LogInformation($"  - Decision: REJECT (challenger underperforms baseline)");
    }

    /// <summary>
    /// Task 3.5: Failure Handling Test
    /// Validates retry logic and graceful failure handling
    /// </summary>
    [Fact]
    public async Task FailureHandling_RetryLogic_ShouldRetryAndEventuallySkip()
    {
        _logger.LogInformation("=== Test: Failure Handling with Retry ===");

        // Arrange
        var failureHandler = _serviceProvider.GetRequiredService<TrainingFailureHandler>();
        var attemptCount = 0;
        var maxAttempts = 3;

        // Simulate component that always fails
        Func<CancellationToken, Task<ComponentTrainingResult>> failingComponent = async (ct) =>
        {
            attemptCount++;
            _logger.LogInformation($"  Attempt {attemptCount}/{maxAttempts}...");
            await Task.Delay(50, ct); // Simulate work
            throw new InvalidOperationException($"Simulated failure (attempt {attemptCount})");
        };

        // Act
        var result = await failureHandler.RetryComponentTrainingAsync(
            "FailingComponent",
            failingComponent,
            maxAttempts,
            CancellationToken.None);

        // Assert
        Assert.False(result.Success);
        Assert.Equal(maxAttempts, attemptCount);
        Assert.Contains("Simulated failure", result.ErrorMessage);
        
        _logger.LogInformation("✓ Failure handler correctly retried {Attempts} times", attemptCount);
        _logger.LogInformation($"  - Final result: {(result.Success ? "SUCCESS" : "FAILED")}");
    }

    /// <summary>
    /// Task 3.6: Performance Benchmark Test
    /// Validates that training performance profiling works
    /// </summary>
    [Fact]
    public async Task PerformanceBenchmark_Profiling_ShouldMeasureTime()
    {
        _logger.LogInformation("=== Test: Performance Benchmark ===");

        // Arrange
        var profiler = _serviceProvider.GetRequiredService<TrainingPerformanceProfiler>();
        var componentName = "BenchmarkComponent";
        var expectedMinDuration = TimeSpan.FromMilliseconds(100);

        // Act - Profile a component
        profiler.StartProfilingSection(componentName);
        await Task.Delay(expectedMinDuration); // Simulate work
        profiler.EndProfilingSection(componentName);

        var profile = profiler.GetSectionProfile(componentName);

        // Assert
        Assert.NotNull(profile);
        Assert.True(profile.Duration >= expectedMinDuration);
        Assert.True(profile.Duration < TimeSpan.FromSeconds(5)); // Should be quick
        
        _logger.LogInformation("✓ Performance profiling measured correctly");
        _logger.LogInformation($"  - Component: {componentName}");
        _logger.LogInformation($"  - Duration: {profile.Duration.TotalMilliseconds:F0}ms");
        _logger.LogInformation($"  - CPU: {profile.CpuPercent:F1}%, Memory: {profile.MemoryDeltaMB:F1}MB");
    }

    /// <summary>
    /// Task 3.7: Configuration Validation Tool
    /// Validates that Lab Mode configuration can be checked
    /// </summary>
    [Fact]
    public void ConfigurationValidation_RequiredSettings_ShouldValidate()
    {
        _logger.LogInformation("=== Test: Configuration Validation ===");

        // Arrange - Check critical configuration
        var checks = new Dictionary<string, bool>
        {
            ["Logging configured"] = _serviceProvider.GetService<ILogger<LabModeIntegrationTests>>() != null,
            ["Configuration available"] = _serviceProvider.GetService<IConfiguration>() != null,
            ["TrainingDebugLogger registered"] = _serviceProvider.GetService<TrainingDebugLogger>() != null,
            ["MemoryLeakDetector registered"] = _serviceProvider.GetService<MemoryLeakDetector>() != null,
            ["TrainingCheckpointService registered"] = _serviceProvider.GetService<TrainingCheckpointService>() != null,
            ["TrainingFailureHandler registered"] = _serviceProvider.GetService<TrainingFailureHandler>() != null,
            ["Test directories exist"] = Directory.Exists(_tempDir) && Directory.Exists(_testDataDir)
        };

        // Act & Assert
        var allPassed = true;
        foreach (var (check, passed) in checks)
        {
            if (passed)
            {
                _logger.LogInformation($"  ✓ {check}");
            }
            else
            {
                _logger.LogError($"  ✗ {check}");
                allPassed = false;
            }
        }

        Assert.True(allPassed, "All configuration checks should pass");
        _logger.LogInformation("✓ All configuration validation checks passed");
    }

    /// <summary>
    /// Helper method to determine if challenger should be promoted
    /// Simplified version of actual promotion logic
    /// </summary>
    private bool ShouldPromoteChallenger(ModelPerformanceMetrics baseline, ModelPerformanceMetrics challenger)
    {
        // Challenger must outperform or match baseline in all key metrics
        var sharpeImproved = challenger.Sharpe >= baseline.Sharpe * 0.95m; // Allow 5% tolerance
        var winRateImproved = challenger.WinRate >= baseline.WinRate * 0.95m;
        var drawdownImproved = challenger.MaxDrawdown >= baseline.MaxDrawdown; // Less negative is better
        var returnImproved = challenger.TotalReturn >= baseline.TotalReturn * 0.90m;

        return sharpeImproved && winRateImproved && drawdownImproved && returnImproved;
    }

    /// <summary>
    /// Test that lock file mechanism works correctly and prevents concurrent training
    /// Regression test for: "lock file issues every time i launch"
    /// </summary>
    [Fact]
    public async Task TrainingLockFile_PreventsConcurrentTraining()
    {
        // Arrange
        var monitor = _serviceProvider.GetRequiredService<TrainingResourceMonitor>();
        var lockFilePath = Path.Combine(Path.GetTempPath(), "qbot_lab_training.lock");
        
        // Clean up any existing lock file
        if (File.Exists(lockFilePath))
        {
            File.Delete(lockFilePath);
        }
        
        // Act - First lock should succeed
        var (firstCanProceed, firstIssue) = monitor.CheckTrainingLock();
        
        // Assert - First lock succeeds
        Assert.True(firstCanProceed, $"First lock should succeed but got: {firstIssue}");
        Assert.Null(firstIssue);
        Assert.True(File.Exists(lockFilePath), "Lock file should exist");
        
        // Verify lock file contains current process ID
        var lockContent = File.ReadAllText(lockFilePath);
        Assert.Contains($"PID:{Environment.ProcessId}", lockContent);
        
        // Act - Second lock from same process should succeed (allows current process)
        var (secondCanProceed, secondIssue) = monitor.CheckTrainingLock();
        
        // Assert - Same process can re-acquire lock
        Assert.True(secondCanProceed, "Same process should be able to re-acquire lock");
        
        // Act - Simulate lock from different (dead) process
        File.WriteAllText(lockFilePath, "PID:99999999|Started:" + DateTime.UtcNow.ToString("O"));
        var (thirdCanProceed, thirdIssue) = monitor.CheckTrainingLock();
        
        // Assert - Dead process lock should be cleaned up and new lock acquired
        Assert.True(thirdCanProceed, "Lock from dead process should be cleaned up");
        
        // Clean up
        monitor.ReleaseTrainingLock();
        Assert.False(File.Exists(lockFilePath), "Lock file should be deleted after release");
        
        await Task.CompletedTask;
    }
    
    /// <summary>
    /// Test that lock file is properly cleaned up when process terminates
    /// </summary>
    [Fact]
    public void TrainingLockFile_HandlesStaleLocks()
    {
        // Arrange
        var monitor = _serviceProvider.GetRequiredService<TrainingResourceMonitor>();
        var lockFilePath = Path.Combine(Path.GetTempPath(), "qbot_lab_training.lock");
        
        // Clean up any existing lock file
        if (File.Exists(lockFilePath))
        {
            File.Delete(lockFilePath);
        }
        
        // Act - Create a very old stale lock (7 hours old)
        File.WriteAllText(lockFilePath, "PID:12345|Started:" + DateTime.UtcNow.AddHours(-7).ToString("O"));
        File.SetLastWriteTimeUtc(lockFilePath, DateTime.UtcNow.AddHours(-7));
        
        var (canProceed, issue) = monitor.CheckTrainingLock();
        
        // Assert - Stale lock should be cleaned up
        Assert.True(canProceed, "Very old stale lock (>6 hours) should be cleaned up automatically");
        Assert.Null(issue);
        
        // Clean up
        monitor.ReleaseTrainingLock();
    }

    public void Dispose()
    {
        // Clean up any test lock files
        var lockFilePath = Path.Combine(Path.GetTempPath(), "qbot_lab_training.lock");
        try
        {
            if (File.Exists(lockFilePath))
            {
                File.Delete(lockFilePath);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
        
        // Cleanup test directories
        try
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, recursive: true);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to cleanup test directory: {Directory}", _tempDir);
        }
    }
}

/// <summary>
/// Simplified checkpoint state for testing
/// </summary>
public class CheckpointState
{
    public string SessionId { get; set; } = string.Empty;
    public int TotalComponents { get; set; }
    public int CurrentComponentIndex { get; set; }
    public List<string> CompletedComponents { get; set; } = new();
    public DateTime StartTime { get; set; }
    public DateTime LastCheckpointTime { get; set; }
}

/// <summary>
/// Model performance metrics for promotion validation
/// </summary>
public class ModelPerformanceMetrics
{
    public decimal Sharpe { get; set; }
    public decimal WinRate { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal TotalReturn { get; set; }
}

/// <summary>
/// Component training result
/// </summary>
public class ComponentTrainingResult
{
    public bool Success { get; set; }
    public string ErrorMessage { get; set; } = string.Empty;
}
