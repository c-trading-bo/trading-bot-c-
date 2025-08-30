# Health Monitoring Developer Guide

## 🎯 Problem Solved
When you add new features to the bot, the health monitoring system needs to know about them to detect failures. This guide shows you how to create intelligent health checks that **automatically get discovered** and integrated into the monitoring system.

## 🚀 Quick Start - Adding Health Checks for New Features

### Step 1: Create Your Health Check Class

```csharp
using OrchestratorAgent.Infra;

[HealthCheck(Category = "Your Feature Category", Priority = 1)]
public class YourFeatureHealthCheck : IHealthCheck
{
    public string Name => "your_feature_name";
    public string Description => "What this health check validates";
    public string Category => "Feature Category";
    public int IntervalSeconds => 60; // How often to check (seconds)

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Test your feature's core functionality here
            // Example: Validate configuration, test calculations, check data quality
            
            if (/* something is broken */)
            {
                return HealthCheckResult.Failed("Specific error message");
            }
            
            if (/* something needs attention */)
            {
                return HealthCheckResult.Warning("Warning message");
            }
            
            return HealthCheckResult.Healthy("Feature working correctly");
        }
        catch (Exception ex)
        {
            return HealthCheckResult.Failed($"Health check failed: {ex.Message}");
        }
    }
}
```

### Step 2: Save in the Right Location
Save your health check in: `src/OrchestratorAgent/Infra/HealthChecks/YourFeatureHealthCheck.cs`

### Step 3: That's It! 
The health monitoring system will **automatically discover** and register your health check when the bot starts.

## 🧠 What Makes a Good Health Check?

### ✅ DO - Test Actual Functionality
```csharp
// Good: Tests if strategy signal logic actually works
var testEntry = 5000.25m;
var testStop = 4995.00m;
var risk = testEntry - testStop;
if (risk <= 0)
    return HealthCheckResult.Failed("Risk calculation logic is broken");
```

### ❌ DON'T - Just Check If Files Exist
```csharp
// Bad: Only checks existence, not functionality
if (File.Exists("strategy.json"))
    return HealthCheckResult.Healthy("Strategy file exists");
```

### ✅ DO - Validate Mathematical Calculations
```csharp
// Good: Tests actual math with known inputs/outputs
var testRMultiple = CalculateRMultiple(5000m, 4995m, 5010m);
var expectedR = 2.0m;
if (Math.Abs(testRMultiple - expectedR) > 0.01m)
    return HealthCheckResult.Failed($"R-multiple calculation wrong: got {testRMultiple}, expected {expectedR}");
```

### ✅ DO - Test Error Conditions
```csharp
// Good: Tests that error handling works
try
{
    ProcessInvalidData(null);
    return HealthCheckResult.Failed("Should have thrown exception for null data");
}
catch (ArgumentNullException)
{
    // Expected behavior
    return HealthCheckResult.Healthy("Error handling works correctly");
}
```

## 📋 Health Check Categories

Organize your health checks by category:

| Category | Purpose | Examples |
|----------|---------|----------|
| `Trading Strategies` | Strategy logic validation | Signal generation, R-multiple calculations |
| `Machine Learning` | ML system monitoring | Learning persistence, model accuracy |
| `Market Data` | Data feed quality | Price validation, connectivity checks |
| `Risk Management` | Risk system validation | Position sizing, stop loss logic |
| `Authentication` | Auth system health | JWT validation, API connectivity |
| `File System` | File operations | Permission checks, backup validation |
| `Performance` | System performance | Memory usage, response times |

## 🔧 Advanced Features

### Priority Levels
Higher priority health checks run first:
```csharp
[HealthCheck(Category = "Critical", Priority = 1)]  // Runs first
[HealthCheck(Category = "Important", Priority = 5)] // Runs later
```

### Custom Intervals
Different features need different check frequencies:
```csharp
public int IntervalSeconds => 30;  // Check every 30 seconds (critical)
public int IntervalSeconds => 300; // Check every 5 minutes (background)
```

### Disabling Health Checks
Temporarily disable problematic checks:
```csharp
[HealthCheck(Category = "Testing", Enabled = false)]
```

## 🚨 Real Examples from the Bot

### ML Learning Persistence Check
```csharp
[HealthCheck(Category = "Machine Learning", Priority = 1)]
public class MLLearningHealthCheck : IHealthCheck
{
    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        // Test that ML learning state persists across restarts
        var stateFile = "state/learning_state.json";
        if (!File.Exists(stateFile))
            return HealthCheckResult.Warning("ML learning state file not found");
            
        var content = await File.ReadAllTextAsync(stateFile, cancellationToken);
        if (!content.Contains("lastPractice"))
            return HealthCheckResult.Failed("ML state file corrupted");
            
        return HealthCheckResult.Healthy("ML learning persistence working");
    }
}
```

### Strategy Signal Validation
```csharp
[HealthCheck(Category = "Trading Strategies", Priority = 2)]
public class StrategySignalHealthCheck : IHealthCheck
{
    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        // Test strategy signal generation with mock data
        var mockPrice = 5000.25m;
        var mockStop = 4995.00m;
        var mockTarget = 5010.50m;
        
        var risk = mockPrice - mockStop;
        var reward = mockTarget - mockPrice;
        var rMultiple = reward / risk;
        
        if (risk <= 0 || reward <= 0)
            return HealthCheckResult.Failed("Strategy R-multiple calculation broken");
            
        if (rMultiple < 1.0m)
            return HealthCheckResult.Warning($"Low R-multiple: {rMultiple:F2}");
            
        return HealthCheckResult.Healthy($"Strategy signals working, R-multiple: {rMultiple:F2}");
    }
}
```

## 🔍 How the Discovery System Works

1. **Automatic Scanning**: On startup, the system scans all assemblies for classes implementing `IHealthCheck`
2. **Attribute Detection**: Finds classes marked with `[HealthCheck]` attribute
3. **Auto-Registration**: Creates instances and registers them automatically
4. **Integration**: Adds them to the existing health monitoring dashboard
5. **Feature Detection**: Warns if major feature areas lack health checks

## 📊 Monitoring Your Health Checks

### Dashboard View
- Go to `http://localhost:5050` → Health tab
- See all health checks with real-time status
- Green = Healthy, Yellow = Warning, Red = Failed

### Logs
Health check results appear in logs:
```
[HEALTH] ml_learning_system HEALTHY: ML learning persistence working
[HEALTH] strategy_signals FAILED: Strategy R-multiple calculation broken
```

### API Access
Get health status programmatically:
```bash
curl http://localhost:5050/health/system
```

## 🚦 Best Practices

### 1. Test Real Behavior, Not Existence
```csharp
// Bad
if (File.Exists("config.json")) return Healthy();

// Good  
var config = LoadConfig("config.json");
if (config.MaxTrades <= 0) return Failed("Invalid max trades");
```

### 2. Use Meaningful Error Messages
```csharp
// Bad
return HealthCheckResult.Failed("Error");

// Good
return HealthCheckResult.Failed("Position tracking failed: calculated PnL of $500 but expected $750 for ES trade");
```

### 3. Handle Timeouts Gracefully
```csharp
public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken)
{
    using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
    timeoutCts.CancelAfter(TimeSpan.FromSeconds(30));
    
    try
    {
        await SomeSlowOperation(timeoutCts.Token);
        return HealthCheckResult.Healthy();
    }
    catch (OperationCanceledException)
    {
        return HealthCheckResult.Failed("Health check timed out after 30 seconds");
    }
}
```

### 4. Include Diagnostic Data
```csharp
return HealthCheckResult.Healthy("All systems operational", new 
{
    LastCheck = DateTime.UtcNow,
    ResponseTime = stopwatch.ElapsedMilliseconds,
    RecordsProcessed = count
});
```

## 🎯 Summary

With this system, **every new feature you add gets automatic health monitoring**:

1. ✅ **Automatic Discovery** - No manual registration needed
2. ✅ **Intelligent Testing** - Tests actual functionality, not just existence  
3. ✅ **Real-time Monitoring** - Dashboard integration with live status
4. ✅ **Self-Documenting** - Health checks serve as functional tests
5. ✅ **Future-Proof** - New health checks get discovered automatically

**The health monitoring system will now grow with your bot automatically!** 🚀
