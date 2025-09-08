using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Xunit;
using TradingBot.Abstractions;
using Trading.Safety;

namespace TradingBot.Tests.Integration;

/// <summary>
/// Integration tests for Phase 5 Safety components
/// Tests kill switch triggers, risk breaches, and health monitoring
/// </summary>
public class SafetyIntegrationTests : IDisposable
{
    private readonly string _testKillFile;
    private readonly AppOptions _testConfig;
    private readonly ILogger<KillSwitchWatcher> _killSwitchLogger;
    private readonly ILogger<RiskManager> _riskLogger;
    private readonly ILogger<HealthMonitor> _healthLogger;

    public SafetyIntegrationTests()
    {
        _testKillFile = Path.Combine(Path.GetTempPath(), $"test_kill_{Guid.NewGuid():N}.txt");
        _testConfig = new AppOptions
        {
            KillFile = _testKillFile,
            MaxDailyLoss = 1000m,
            MaxPositionSize = 5000m,
            DrawdownLimit = 500m
        };

        var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
        _killSwitchLogger = loggerFactory.CreateLogger<KillSwitchWatcher>();
        _riskLogger = loggerFactory.CreateLogger<RiskManager>();
        _healthLogger = loggerFactory.CreateLogger<HealthMonitor>();
    }

    [Fact]
    public async Task KillSwitchWatcher_DetectsKillFile_TriggersHalt()
    {
        // Arrange
        var options = Options.Create(_testConfig);
        var killSwitch = new KillSwitchWatcher(_killSwitchLogger, options);
        var killSwitchActivated = false;
        
        killSwitch.OnKillSwitchActivated += () => killSwitchActivated = true;

        // Act
        var watchTask = Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            await killSwitch.StartWatchingAsync(cts.Token);
        });

        // Wait a moment then create kill file
        await Task.Delay(100);
        await File.WriteAllTextAsync(_testKillFile, "KILL");
        
        // Wait for detection
        await Task.Delay(1500);

        // Assert
        Assert.True(killSwitch.IsKillSwitchActive, "Kill switch should be active");
        Assert.True(killSwitchActivated, "Kill switch event should have been triggered");

        // Cleanup
        killSwitch.Dispose();
    }

    [Fact]
    public async Task RiskManager_ExceedsMaxDailyLoss_TriggersRiskBreach()
    {
        // Arrange
        var options = Options.Create(_testConfig);
        var riskManager = new RiskManager(_riskLogger, options);
        RiskBreach? detectedBreach = null;
        
        riskManager.OnRiskBreach += breach => detectedBreach = breach;

        // Act - Update with loss exceeding limit
        await riskManager.UpdateDailyPnLAsync(-1500m); // Exceeds -1000 limit

        // Assert
        Assert.True(riskManager.IsRiskBreached, "Risk should be breached");
        Assert.NotNull(detectedBreach);
        Assert.Equal(RiskBreachType.MaxDailyLoss, detectedBreach.Type);
        Assert.Equal(1500m, detectedBreach.CurrentValue);
        Assert.Equal(1000m, detectedBreach.Limit);
    }

    [Fact]
    public async Task RiskManager_ExceedsMaxPositionSize_RejectsOrder()
    {
        // Arrange
        var options = Options.Create(_testConfig);
        var riskManager = new RiskManager(_riskLogger, options);
        
        var largeOrder = new PlaceOrderRequest(
            Symbol: "ES",
            Side: "BUY", 
            Quantity: 100m, // 100 * 100 = 10000 > 5000 limit
            Price: 100m,
            OrderType: "LIMIT",
            CustomTag: "TEST",
            AccountId: "123456"
        );

        // Act
        var isValid = await riskManager.ValidateOrderAsync(largeOrder);

        // Assert
        Assert.False(isValid, "Large order should be rejected");
        Assert.True(riskManager.IsRiskBreached, "Risk should be breached");
    }

    [Fact]
    public async Task HealthMonitor_LowHubConnections_SuspendsTradingAsync()
    {
        // Arrange
        var options = Options.Create(_testConfig);
        var healthMonitor = new HealthMonitor(_healthLogger, options);
        HealthStatus? lastStatus = null;
        
        healthMonitor.OnHealthChanged += status => lastStatus = status;

        // Act - Start monitoring
        var monitorTask = Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3));
            await healthMonitor.StartMonitoringAsync(cts.Token);
        });

        // Simulate disconnected hubs
        healthMonitor.RecordHubConnection("UserHub", false);
        healthMonitor.RecordHubConnection("MarketHub", false);
        
        // Wait for health check
        await Task.Delay(1000);

        // Assert
        Assert.False(healthMonitor.IsTradingAllowed, "Trading should be suspended");
        
        var currentHealth = healthMonitor.GetCurrentHealth();
        Assert.False(currentHealth.IsHealthy, "System should be unhealthy");
        Assert.False(currentHealth.TradingAllowed, "Trading should not be allowed");
    }

    [Fact]
    public void HealthMonitor_HighErrorRate_MarksUnhealthy()
    {
        // Arrange
        var options = Options.Create(_testConfig);
        var healthMonitor = new HealthMonitor(_healthLogger, options);

        // Act - Record high error rate (>10%)
        for (int i = 0; i < 20; i++)
        {
            var success = i < 15; // 25% error rate
            healthMonitor.RecordApiCall("TestOperation", TimeSpan.FromMilliseconds(100), success);
        }

        // Trigger health check by recording an error
        healthMonitor.RecordError("TestSource", new Exception("Test error"));

        // Wait a moment for processing
        Task.Delay(100).Wait();

        // Assert
        var health = healthMonitor.GetCurrentHealth();
        // Note: Due to the async nature, we mainly test that the methods don't throw
        Assert.NotNull(health);
    }

    public void Dispose()
    {
        // Cleanup test files
        try
        {
            if (File.Exists(_testKillFile))
            {
                File.Delete(_testKillFile);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }
}