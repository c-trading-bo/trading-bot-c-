# Advanced System Integration - Usage Guide

## Overview

This guide demonstrates how to integrate the new `MLMemoryManager` and `WorkflowOrchestrationManager` components into your existing trading bot infrastructure.

## Component Summary

### 1. MLMemoryManager (`BotCore/ML/MLMemoryManager.cs`)
- **Purpose**: Prevents memory leaks in ML pipeline operations
- **Features**: 
  - Automatic model lifecycle management
  - Memory monitoring and cleanup
  - Integration with existing `StrategyMlModelManager`
  - Configurable memory limits and GC policies

### 2. WorkflowOrchestrationManager (`UnifiedOrchestrator/Services/WorkflowOrchestrationManager.cs`)
- **Purpose**: Prevents workflow collisions and manages execution priorities
- **Features**:
  - Priority-based workflow execution
  - Resource conflict detection and resolution
  - Automatic queuing and retry mechanisms
  - Trading decision coordination

### 3. RedundantDataFeedManager (`BotCore/Market/RedundantDataFeedManager.cs`)
- **Purpose**: High-availability market data with automatic failover
- **Features**:
  - Multiple data source support
  - Automatic failover on feed failures
  - Data quality monitoring
  - Latency and health tracking

## Quick Start Integration

### Step 1: Add Services to Dependency Injection

```csharp
// In your Program.cs or Startup.cs
using BotCore.Infra;
using TradingBot.UnifiedOrchestrator.Infrastructure;

var services = new ServiceCollection();

// Add logging
services.AddLogging(builder => {
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Information);
});

// Add BotCore advanced services
services.AddMLMemoryManagement();
services.AddEnhancedMLModelManager();

// Add UnifiedOrchestrator services
services.AddWorkflowOrchestration();
services.AddSingleton<AdvancedSystemIntegrationService>();

var serviceProvider = services.BuildServiceProvider();
```

### Step 2: Initialize Advanced Systems

```csharp
// Initialize advanced systems
await BotCore.Infra.AdvancedSystemConfiguration.InitializeAdvancedSystemAsync(serviceProvider);
await TradingBot.UnifiedOrchestrator.Infrastructure.WorkflowOrchestrationConfiguration.InitializeWorkflowOrchestrationAsync(serviceProvider);

// Wire systems together
TradingBot.UnifiedOrchestrator.Infrastructure.WorkflowOrchestrationConfiguration.WireWorkflowOrchestration(serviceProvider);
```

### Step 3: Use Integrated Services

```csharp
// Get the integration service
var integrationService = serviceProvider.GetRequiredService<AdvancedSystemIntegrationService>();
await integrationService.InitializeAsync();

// Execute workflows with advanced coordination
await integrationService.ExecuteWorkflowWithAdvancedCoordinationAsync(
    "es-nq-critical-trading",
    async () => {
        // Your trading logic here
        Console.WriteLine("Executing critical trading workflow");
        await YourTradingLogic();
    },
    new List<string> { "trading_decision", "market_data" }
);

// Get optimized position sizes using ML with memory management
var positionMultiplier = await integrationService.GetOptimizedPositionSizeAsync(
    "your-strategy", "ES", 4500.00m, 25.50m, 1.8m, 0.75m, marketBars
);
```

## Integration with Existing Services

### With StrategyMlModelManager

The enhanced `StrategyMlModelManager` now optionally integrates with `MLMemoryManager`:

```csharp
// The existing StrategyMlModelManager automatically uses MLMemoryManager if available
var strategyMlManager = serviceProvider.GetRequiredService<StrategyMlModelManager>();

// Position sizing now includes memory management
var multiplier = strategyMlManager.GetPositionSizeMultiplier(
    strategyId, symbol, price, atr, score, qScore, bars);

// Check memory usage
var memorySnapshot = strategyMlManager.GetMemorySnapshot();
if (memorySnapshot != null) {
    Console.WriteLine($"ML Memory Usage: {memorySnapshot.MLMemory / 1024 / 1024:F1}MB");
}
```

### With UnifiedOrchestrator

Workflow orchestration integrates seamlessly with existing orchestrator services:

```csharp
var orchestrationManager = serviceProvider.GetRequiredService<IWorkflowOrchestrationManager>();

// Execute workflows with conflict prevention
var success = await orchestrationManager.RequestWorkflowExecutionAsync(
    "portfolio-heat-management",
    async () => {
        // Your portfolio management logic
        await AnalyzePortfolioHeat();
        await AdjustPositions();
    },
    new List<string> { "portfolio_data", "risk_calculation" }
);

// Check orchestration status
var status = orchestrationManager.GetStatus();
Console.WriteLine($"Active workflows: {status.ActiveLocks}, Queued: {status.QueuedTasks}");
```

### With Market Data

Use redundant data feeds for high availability:

```csharp
var dataFeedManager = serviceProvider.GetRequiredService<RedundantDataFeedManager>();

// Get market data with automatic failover
try {
    var marketData = await dataFeedManager.GetMarketDataAsync("ES");
    Console.WriteLine($"ES Price: {marketData.Price} from {marketData.Source}");
} catch (InvalidOperationException ex) {
    Console.WriteLine($"All data feeds unavailable: {ex.Message}");
}

// Monitor feed failovers
dataFeedManager.OnFeedFailover += (sender, feedName) => {
    Console.WriteLine($"Failed over to data feed: {feedName}");
};
```

## System Health Monitoring

```csharp
// Get comprehensive system status
var status = integrationService.GetSystemStatus();

Console.WriteLine($"System Health: {(status.IsHealthy ? "Healthy" : "Issues Detected")}");

foreach (var component in status.Components) {
    var icon = component.Value ? "✅" : "❌";
    Console.WriteLine($"{icon} {component.Key}");
}

if (status.MemorySnapshot != null) {
    var memoryUsagePercent = (double)status.MemorySnapshot.UsedMemory / (8L * 1024 * 1024 * 1024) * 100;
    Console.WriteLine($"Memory Usage: {memoryUsagePercent:F1}%");
}

if (status.Issues.Any()) {
    Console.WriteLine("Issues:");
    foreach (var issue in status.Issues) {
        Console.WriteLine($"  • {issue}");
    }
}
```

## Configuration Options

### MLMemoryManager Configuration

```csharp
// Memory limits can be configured via constants in MLMemoryManager
// Default: 8GB max memory, 3 max model versions

// Environment variable to enable/disable RL
Environment.SetEnvironmentVariable("RL_ENABLED", "1");
```

### Workflow Priorities

The workflow orchestration uses priority-based execution:

```csharp
// Built-in priorities (lower number = higher priority):
// es-nq-critical-trading: 1
// risk-management: 2  
// position-reconciliation: 3
// ultimate-ml-rl-intel: 4
// daily-report: 15
// ml-trainer: 20
// data-collection: 25
```

## Test Results

✅ **Integration Test Successful**: All components initialized and integrated correctly
✅ **Memory Management**: ML memory tracking and cleanup working
✅ **Workflow Orchestration**: Conflict detection and priority execution working  
✅ **Data Feed Redundancy**: Failover and health monitoring working
✅ **System Integration**: All components work together seamlessly

## Key Benefits

1. **Memory Leak Prevention**: Automatic ML model cleanup prevents memory issues
2. **Workflow Coordination**: Prevents conflicts between trading operations
3. **High Availability**: Redundant data feeds ensure continuous operation
4. **Minimal Integration**: Works with existing code with minimal changes
5. **Comprehensive Monitoring**: Real-time system health and status tracking

## Next Steps

1. **Performance Testing**: Validate under production load
2. **Custom Configuration**: Adjust memory limits and priorities for your needs
3. **Monitoring Integration**: Connect to your existing monitoring systems
4. **Documentation**: Update your internal documentation with new capabilities

This integration provides enterprise-grade reliability and performance optimizations while maintaining full compatibility with your existing trading bot infrastructure.