using System;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Testing;

/// <summary>
/// Simple test to verify contract entitlement probe functionality
/// </summary>
internal static class EntitlementTest
{
    public static async Task TestContractEntitlements()
    {
        try
        {
            Console.WriteLine("🔧 [TEST] Starting contract entitlement probe test");

            // 1. Resolve standardized contracts
            var (es, nq) = ContractEntitlementProbe.ResolveContracts();
            Console.WriteLine($"[TEST] Contracts resolved - ES: {es}, NQ: {nq}");

            // 2. Get fresh JWT
            var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            if (string.IsNullOrEmpty(jwt))
            {
                Console.WriteLine("[TEST] ❌ No JWT token available");
                return;
            }

            Console.WriteLine($"[TEST] JWT found: {jwt.Length} characters");

            // 3. Create a simple logger (console-based)
            var logger = new ConsoleLogger();

            // 4. Probe entitlements
            Console.WriteLine("[TEST] Probing contract entitlements...");
            
            var esProbeOk = await ContractEntitlementProbe.ProbeSnapshotAsync(es, jwt, logger).ConfigureAwait(false);
            var nqProbeOk = await ContractEntitlementProbe.ProbeSnapshotAsync(nq, jwt, logger).ConfigureAwait(false);

            Console.WriteLine($"[TEST] Results:");
            Console.WriteLine($"  ES ({es}): {(esProbeOk ? "✅ OK" : "❌ FAILED")}");
            Console.WriteLine($"  NQ ({nq}): {(nqProbeOk ? "✅ OK" : "❌ FAILED")}");

            if (!esProbeOk && !nqProbeOk)
            {
                Console.WriteLine("[TEST] ❌ Both contracts failed - entitlement/ID issue confirmed");
            }
            else if (esProbeOk && nqProbeOk)
            {
                Console.WriteLine("[TEST] ✅ Both contracts accessible - live data issue is SignalR-side");
            }
            else
            {
                Console.WriteLine("[TEST] ⚠️ Mixed results - partial entitlements or contract ID mismatch");
            }

            Console.WriteLine("🎉 [TEST] Entitlement probe test complete!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[TEST] ❌ Exception: {ex.Message}");
        }
    }
}

/// <summary>
/// Simple console logger for testing
/// </summary>
internal class ConsoleLogger : Microsoft.Extensions.Logging.ILogger
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;
    
    public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, Microsoft.Extensions.Logging.EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        var message = formatter(state, exception);
        Console.WriteLine($"[{logLevel}] {message}");
        if (exception != null)
        {
            Console.WriteLine($"Exception: {exception}");
        }
    }
}