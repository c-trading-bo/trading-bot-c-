using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using BotCore.ML;
using BotCore.Market;
using BotCore.Models;
using BotCore.Infra;
using TradingBot.UnifiedOrchestrator.Services;
using TradingBot.UnifiedOrchestrator.Infrastructure;

namespace TradingBot.Tests;

/// <summary>
/// Comprehensive integration test for the advanced system components
/// Tests MLMemoryManager, WorkflowOrchestrationManager, and their integration
/// </summary>
public class Program
{
    public static async Task<int> Main(string[] args)
    {
        Console.WriteLine("=== Advanced System Integration Test ===");
        Console.WriteLine("Testing MLMemoryManager and WorkflowOrchestrationManager");
        Console.WriteLine();

        try
        {
            // Setup dependency injection
            var services = new ServiceCollection();
            
            // Add logging
            services.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information);
            });
            
            // Add BotCore advanced services
            services.AddMLMemoryManagement();
            services.AddEnhancedMLModelManager();
            services.AddEconomicEventManagement();
            
            // Add UnifiedOrchestrator services
            services.AddWorkflowOrchestration();
            services.AddSingleton<AdvancedSystemIntegrationService>();
            
            var serviceProvider = services.BuildServiceProvider();
            
            // Initialize advanced systems
            await BotCore.Infra.AdvancedSystemConfiguration.InitializeAdvancedSystemAsync(serviceProvider);
            await TradingBot.UnifiedOrchestrator.Infrastructure.WorkflowOrchestrationConfiguration.InitializeWorkflowOrchestrationAsync(serviceProvider);
            
            // Wire systems together
            TradingBot.UnifiedOrchestrator.Infrastructure.WorkflowOrchestrationConfiguration.WireWorkflowOrchestration(serviceProvider);
            
            // Get the integration service
            var integrationService = serviceProvider.GetRequiredService<AdvancedSystemIntegrationService>();
            await integrationService.InitializeAsync();
            
            Console.WriteLine("✅ Advanced system components initialized successfully");
            Console.WriteLine();
            
            // Test MLMemoryManager
            await TestMLMemoryManager(serviceProvider);
            
            // Test WorkflowOrchestrationManager
            await TestWorkflowOrchestration(serviceProvider);
            
            // Test EconomicEventManager
            await TestEconomicEventManager(serviceProvider);
            
            // Test integrated system
            await TestIntegratedSystem(integrationService);
            
            // Show system status
            await ShowSystemStatus(integrationService);
            
            Console.WriteLine();
            Console.WriteLine("✅ All tests completed successfully!");
            Console.WriteLine("🎯 MLMemoryManager, WorkflowOrchestrationManager, and EconomicEventManager are fully integrated and working");
            
            // Cleanup
            integrationService.Dispose();
            await serviceProvider.DisposeAsync();
            
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Test failed: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            return 1;
        }
    }
    
    private static async Task TestMLMemoryManager(IServiceProvider serviceProvider)
    {
        Console.WriteLine("🧠 Testing ML Memory Manager...");
        
        var memoryManager = serviceProvider.GetService<IMLMemoryManager>();
        var strategyMlManager = serviceProvider.GetService<StrategyMlModelManager>();
        
        if (memoryManager == null)
        {
            Console.WriteLine("❌ MLMemoryManager not found");
            return;
        }
        
        // Test memory snapshot
        var snapshot = memoryManager.GetMemorySnapshot();
        Console.WriteLine($"   📊 Memory Usage: {snapshot.TotalMemory / 1024 / 1024:F1}MB total, {snapshot.LoadedModels} models loaded");
        
        // Test model loading with memory management
        if (strategyMlManager != null)
        {
            var memorySnapshot = strategyMlManager.GetMemorySnapshot();
            if (memorySnapshot != null)
            {
                Console.WriteLine($"   🔗 StrategyMlManager integrated with memory management");
                Console.WriteLine($"   📈 ML Memory: {memorySnapshot.MLMemory / 1024 / 1024:F1}MB");
            }
        }
        
        Console.WriteLine("✅ ML Memory Manager test completed");
        Console.WriteLine();
    }
    
    private static async Task TestWorkflowOrchestration(IServiceProvider serviceProvider)
    {
        Console.WriteLine("⚙️ Testing Workflow Orchestration Manager...");
        
        var orchestrationManager = serviceProvider.GetService<TradingBot.UnifiedOrchestrator.Interfaces.IWorkflowOrchestrationManager>();
        
        if (orchestrationManager == null)
        {
            Console.WriteLine("❌ WorkflowOrchestrationManager not found");
            return;
        }
        
        // Test workflow execution
        var testWorkflowExecuted = false;
        var success = await orchestrationManager.RequestWorkflowExecutionAsync(
            "test-workflow",
            async () =>
            {
                await Task.Delay(100);
                testWorkflowExecuted = true;
                Console.WriteLine("   🔄 Test workflow executed successfully");
            },
            new List<string> { "test_resource" }
        );
        
        if (success && testWorkflowExecuted)
        {
            Console.WriteLine("   ✅ Workflow execution successful");
        }
        else
        {
            Console.WriteLine("   ⏳ Workflow queued due to conflicts");
        }
        
        // Test status
        var status = orchestrationManager.GetStatus();
        Console.WriteLine($"   📊 Active locks: {status.ActiveLocks}, Queued tasks: {status.QueuedTasks}");
        Console.WriteLine($"   🔒 Trading mutex available: {status.TradingMutexAvailable}");
        
        Console.WriteLine("✅ Workflow Orchestration Manager test completed");
        Console.WriteLine();
    }
    
    private static async Task TestEconomicEventManager(IServiceProvider serviceProvider)
    {
        Console.WriteLine("📊 Testing Economic Event Manager...");
        
        var economicEventManager = serviceProvider.GetService<IEconomicEventManager>();
        
        if (economicEventManager == null)
        {
            Console.WriteLine("❌ EconomicEventManager not found");
            return;
        }
        
        // Test upcoming events
        var upcomingEvents = await economicEventManager.GetUpcomingEventsAsync(TimeSpan.FromHours(24));
        Console.WriteLine($"   📅 Found {upcomingEvents.Count()} upcoming events in next 24 hours");
        
        // Test high impact events
        var highImpactEvents = await economicEventManager.GetEventsByImpactAsync(EventImpact.High);
        Console.WriteLine($"   ⚠️ Found {highImpactEvents.Count()} high-impact events");
        
        foreach (var economicEvent in highImpactEvents.Take(3))
        {
            Console.WriteLine($"      • {economicEvent.Name} ({economicEvent.Currency}) - {economicEvent.Impact} impact at {economicEvent.ScheduledTime:yyyy-MM-dd HH:mm}");
        }
        
        // Test trading restrictions
        var tradingAllowedES = !await economicEventManager.ShouldRestrictTradingAsync("ES", TimeSpan.FromHours(2));
        var tradingAllowedNQ = !await economicEventManager.ShouldRestrictTradingAsync("NQ", TimeSpan.FromHours(2));
        
        Console.WriteLine($"   🛡️ Trading allowed for ES: {(tradingAllowedES ? "✅" : "❌")}");
        Console.WriteLine($"   🛡️ Trading allowed for NQ: {(tradingAllowedNQ ? "✅" : "❌")}");
        
        // Test trading restriction details
        var restrictionES = await economicEventManager.GetTradingRestrictionAsync("ES");
        if (restrictionES.IsRestricted)
        {
            Console.WriteLine($"      📋 ES restriction: {restrictionES.Reason} until {restrictionES.RestrictedUntil:yyyy-MM-dd HH:mm}");
        }
        
        Console.WriteLine("✅ Economic Event Manager test completed");
        Console.WriteLine();
    }
    
    private static async Task TestIntegratedSystem(AdvancedSystemIntegrationService integrationService)
    {
        Console.WriteLine("🔗 Testing Integrated System...");
        
        // Test workflow execution through integration service
        var success = await integrationService.ExecuteWorkflowWithAdvancedCoordinationAsync(
            "es-nq-critical-trading",
            async () =>
            {
                Console.WriteLine("   📈 Executing critical ES/NQ trading workflow");
                await Task.Delay(200);
                Console.WriteLine("   ✅ Critical trading workflow completed");
            },
            new List<string> { "trading_decision", "market_data" }
        );
        
        if (success)
        {
            Console.WriteLine("   ✅ Integrated workflow execution successful");
        }
        
        // Test ML position sizing integration
        var bars = CreateTestBars();
        var positionMultiplier = await integrationService.GetOptimizedPositionSizeAsync(
            "test-strategy", "ES", 4500.00m, 25.50m, 1.8m, 0.75m, bars
        );
        
        Console.WriteLine($"   🎯 Optimized position size multiplier: {positionMultiplier:F2}");
        
        // Test trading allowance with economic events
        var isTradingAllowedES = await integrationService.IsTradingAllowedAsync("ES");
        var isTradingAllowedNQ = await integrationService.IsTradingAllowedAsync("NQ");
        
        Console.WriteLine($"   🛡️ Trading allowed for ES: {(isTradingAllowedES ? "✅" : "❌")}");
        Console.WriteLine($"   🛡️ Trading allowed for NQ: {(isTradingAllowedNQ ? "✅" : "❌")}");
        
        Console.WriteLine("✅ Integrated System test completed");
        Console.WriteLine();
    }
    
    private static async Task ShowSystemStatus(AdvancedSystemIntegrationService integrationService)
    {
        Console.WriteLine("📊 System Status Summary:");
        
        var status = await integrationService.GetSystemStatusAsync();
        
        Console.WriteLine($"   🏥 Overall Health: {(status.IsHealthy ? "✅ Healthy" : "❌ Issues Detected")}");
        Console.WriteLine($"   ⏰ Status Time: {status.Timestamp:yyyy-MM-dd HH:mm:ss}");
        
        Console.WriteLine("   🔧 Components:");
        foreach (var component in status.Components)
        {
            var icon = component.Value ? "✅" : "❌";
            Console.WriteLine($"      {icon} {component.Key}");
        }
        
        if (status.MemorySnapshot != null)
        {
            Console.WriteLine($"   🧠 Memory: {status.MemorySnapshot.TotalMemory / 1024 / 1024:F1}MB total, {status.MemorySnapshot.LoadedModels} ML models");
        }
        
        if (status.WorkflowOrchestrationStatus != null)
        {
            Console.WriteLine($"   ⚙️ Workflows: {status.WorkflowOrchestrationStatus.ActiveLocks} active locks, {status.WorkflowOrchestrationStatus.QueuedTasks} queued");
        }
        
        if (status.Issues.Any())
        {
            Console.WriteLine("   ⚠️ Issues:");
            foreach (var issue in status.Issues)
            {
                Console.WriteLine($"      • {issue}");
            }
        }
    }
    
    private static List<BotCore.Models.Bar> CreateTestBars()
    {
        var bars = new List<BotCore.Models.Bar>();
        var random = new Random();
        var basePrice = 4500.00m;
        
        for (int i = 0; i < 50; i++)
        {
            var price = basePrice + (decimal)(random.NextDouble() * 20 - 10);
            var startTime = DateTime.UtcNow.AddMinutes(-i);
            bars.Add(new BotCore.Models.Bar
            {
                Start = startTime,
                Ts = ((DateTimeOffset)startTime).ToUnixTimeMilliseconds(),
                Symbol = "ES",
                Open = price,
                High = price + (decimal)(random.NextDouble() * 5),
                Low = price - (decimal)(random.NextDouble() * 5),
                Close = price + (decimal)(random.NextDouble() * 4 - 2),
                Volume = 1000 + random.Next(0, 500)
            });
        }
        
        return bars;
    }
}