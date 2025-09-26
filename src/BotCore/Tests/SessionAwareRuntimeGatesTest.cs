using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Services;

namespace BotCore.Tests;

/// <summary>
/// Simple test to validate SessionAwareRuntimeGates functionality
/// </summary>
public static class SessionAwareRuntimeGatesTest
{
    public static async Task RunBasicTestAsync()
    {
        Console.WriteLine("🧪 Testing SessionAwareRuntimeGates functionality...");
        
        // Setup basic configuration
        var config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                {"Sessions:TimeZone", "America/New_York"},
                {"Sessions:MaintenanceBreak:Start", "17:00"},
                {"Sessions:MaintenanceBreak:End", "18:00"},
                {"Sessions:RTH:Start", "09:30"},
                {"Sessions:RTH:End", "16:00"},
                {"Sessions:ETH:Allow", "true"},
                {"Sessions:ETH:CurbFirstMins", "3"},
                {"Sessions:SundayReopen:Enable", "true"},
                {"Sessions:SundayReopen:CurbMins", "5"}
            })
            .Build();
        
        // Setup logging
        var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
        var logger = loggerFactory.CreateLogger<SessionAwareRuntimeGates>();
        
        // Create service
        var sessionGates = new SessionAwareRuntimeGates(logger, config);
        
        // Test basic functionality
        Console.WriteLine("📊 Current Session Status:");
        var status = sessionGates.GetSessionStatus();
        
        Console.WriteLine($"  Current Session: {status.CurrentSession}");
        Console.WriteLine($"  Trading Allowed: {status.TradingAllowed}");
        Console.WriteLine($"  Is RTH: {status.IsRth}");
        Console.WriteLine($"  Is ETH: {status.IsEth}");
        Console.WriteLine($"  Eastern Time: {status.EasternTime:yyyy-MM-dd HH:mm:ss} ET");
        Console.WriteLine($"  Next Change: {status.NextSessionChange?.ToString("yyyy-MM-dd HH:mm:ss") ?? "N/A"}");
        
        // Test trading permission
        var tradingAllowed = await sessionGates.IsTradingAllowedAsync("ES").ConfigureAwait(false);
        Console.WriteLine($"  ES Trading Allowed: {tradingAllowed}");
        
        // Test session detection
        Console.WriteLine($"  Current Session Name: {sessionGates.GetCurrentSession()}");
        
        Console.WriteLine("✅ SessionAwareRuntimeGates test completed successfully!");
    }
}

/// <summary>
/// Demo program to show SessionAwareRuntimeGates in action
/// </summary>
public static class SessionAwareDemoProgram
{
    public static async Task Main(string[] args)
    {
        try
        {
            await SessionAwareRuntimeGatesTest.RunBasicTestAsync().ConfigureAwait(false);
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"❌ Test failed with invalid operation: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
        catch (TimeoutException ex)
        {
            Console.WriteLine($"❌ Test failed with timeout: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"❌ Test failed with argument error: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
        catch (NotSupportedException ex)
        {
            Console.WriteLine($"❌ Test failed with unsupported operation: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }
}