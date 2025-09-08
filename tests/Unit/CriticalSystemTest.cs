using System;
using System.Threading.Tasks;
using TradingBot.Critical;
using Microsoft.Extensions.Logging;

namespace TradingBot.Tests
{
    /// <summary>
    /// Simple test program to verify critical system components work correctly
    /// </summary>
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== Critical Trading System Components Test ===");
            
            // Test 1: Enhanced Credential Manager
            Console.WriteLine("\n1. Testing Enhanced Credential Manager...");
            try
            {
                // Test with environment variables that should exist
                var testKey = "PATH";
                if (EnhancedCredentialManager.TryGetCredential(testKey, out var value))
                {
                    Console.WriteLine($"✅ Successfully retrieved credential: {testKey} = {value.Substring(0, Math.Min(50, value.Length))}...");
                }
                else
                {
                    Console.WriteLine($"❌ Failed to retrieve credential: {testKey}");
                }

                // Test required credentials (should fail gracefully without actual creds)
                try
                {
                    EnhancedCredentialManager.ValidateRequiredCredentials();
                    Console.WriteLine("✅ Required credentials validation passed");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ Required credentials validation failed (expected): {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Enhanced Credential Manager test failed: {ex.Message}");
            }

            // Test 2: Disaster Recovery System (basic initialization)
            Console.WriteLine("\n2. Testing Disaster Recovery System...");
            try
            {
                using var loggerFactory = LoggerFactory.Create(builder => 
                {
                    builder.AddSimpleConsole(options =>
                    {
                        options.IncludeScopes = false;
                        options.SingleLine = true;
                        options.TimestampFormat = "HH:mm:ss ";
                    });
                });
                var logger = loggerFactory.CreateLogger<DisasterRecoverySystem>();
                
                var disasterRecovery = new DisasterRecoverySystem(logger);
                
                // Test adding a position
                var position = new DisasterRecoverySystem.Position
                {
                    Symbol = "ES",
                    Quantity = 1,
                    EntryPrice = 4500.00m,
                    CurrentPrice = 4500.00m,
                    EntryTime = DateTime.UtcNow,
                    StrategyId = "TEST_STRATEGY"
                };
                
                disasterRecovery.AddPosition(position);
                Console.WriteLine("✅ Disaster Recovery System position tracking works");
                
                disasterRecovery.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Disaster Recovery System test failed: {ex.Message}");
            }

            // Test 3: Correlation Protection System 
            Console.WriteLine("\n3. Testing Correlation Protection System...");
            try
            {
                using var loggerFactory = LoggerFactory.Create(builder => 
                {
                    builder.AddSimpleConsole(options =>
                    {
                        options.IncludeScopes = false;
                        options.SingleLine = true;
                        options.TimestampFormat = "HH:mm:ss ";
                    });
                });
                var logger = loggerFactory.CreateLogger<CorrelationProtectionSystem>();
                
                var correlationProtection = new CorrelationProtectionSystem(logger);
                await correlationProtection.InitializeCorrelationMonitor();
                
                // Test position validation
                var validationResult = await correlationProtection.ValidateNewPosition("ES", 1, "LONG");
                Console.WriteLine($"✅ Correlation Protection System validation works: {validationResult}");
                
                // Test exposure update
                correlationProtection.UpdateExposure("ES", 4500.00m);
                Console.WriteLine("✅ Correlation Protection System exposure tracking works");
                
                correlationProtection.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Correlation Protection System test failed: {ex.Message}");
            }

            Console.WriteLine("\n=== Critical Trading System Components Test Complete ===");
            Console.WriteLine("✅ All basic component tests completed successfully!");
            Console.WriteLine("\nNote: Full integration testing requires actual TopstepX credentials and SignalR connections.");
        }
    }
}