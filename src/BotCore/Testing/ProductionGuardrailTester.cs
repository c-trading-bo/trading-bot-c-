using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using BotCore.Services;
using BotCore.Extensions;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Testing;

/// <summary>
/// Simple test program to verify production guardrails are working
/// </summary>
public class ProductionGuardrailTester
{
    private readonly ILogger<ProductionGuardrailTester> _logger;
    private readonly ProductionGuardrailOrchestrator _orchestrator;
    private readonly ProductionOrderEvidenceService _evidenceService;

    public ProductionGuardrailTester(
        ILogger<ProductionGuardrailTester> logger,
        ProductionGuardrailOrchestrator orchestrator,
        ProductionOrderEvidenceService evidenceService)
    {
        _logger = logger;
        _orchestrator = orchestrator;
        _evidenceService = evidenceService;
    }

    /// <summary>
    /// Run comprehensive production guardrail tests
    /// </summary>
    public async Task<bool> RunAllTestsAsync()
    {
        _logger.LogInformation("🧪 [TEST] Starting production guardrail tests...");

        var allPassed = true;

        // Test 1: DRY_RUN precedence
        allPassed &= await TestDryRunPrecedence().ConfigureAwait(false);

        // Test 2: Kill switch functionality
        allPassed &= await TestKillSwitchFunctionality().ConfigureAwait(false);

        // Test 3: Price validation (ES/MES tick rounding)
        allPassed &= TestPriceValidation();

        // Test 4: Risk validation (reject ≤ 0)
        allPassed &= TestRiskValidation();

        // Test 5: Order evidence requirements
        allPassed &= await TestOrderEvidenceRequirements().ConfigureAwait(false);

        if (allPassed)
        {
            _logger.LogInformation("✅ [TEST] All production guardrail tests PASSED");
        }
        else
        {
            _logger.LogCritical("🔴 [TEST] Some production guardrail tests FAILED");
        }

        return allPassed;
    }

    private async Task<bool> TestDryRunPrecedence()
    {
        _logger.LogInformation("🧪 [TEST] Testing DRY_RUN precedence...");

        try
        {
            // Set DRY_RUN=true and verify it takes precedence
            Environment.SetEnvironmentVariable("DRY_RUN", "true");
            Environment.SetEnvironmentVariable("EXECUTE", "true");
            Environment.SetEnvironmentVariable("AUTO_EXECUTE", "true");

            var isDryRun = ProductionKillSwitchService.IsDryRunMode();
            
            if (isDryRun)
            {
                _logger.LogInformation("✅ [TEST] DRY_RUN precedence test PASSED");
                return true;
            }
            else
            {
                _logger.LogError("❌ [TEST] DRY_RUN precedence test FAILED - DRY_RUN should override execution flags");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [TEST] DRY_RUN precedence test FAILED with exception");
            return false;
        }
        finally
        {
            // Clean up
            Environment.SetEnvironmentVariable("DRY_RUN", null);
            Environment.SetEnvironmentVariable("EXECUTE", null);
            Environment.SetEnvironmentVariable("AUTO_EXECUTE", null);
            await Task.CompletedTask.ConfigureAwait(false);
        }
    }

    private async Task<bool> TestKillSwitchFunctionality()
    {
        _logger.LogInformation("🧪 [TEST] Testing kill switch functionality...");

        try
        {
            // Create kill.txt file
            await File.WriteAllTextAsync("kill.txt", "Test kill switch").ConfigureAwait(false);

            // Give file watcher time to detect
            await Task.Delay(100).ConfigureAwait(false);

            var killSwitchActive = ProductionKillSwitchService.IsKillSwitchActive();
            var isDryRun = ProductionKillSwitchService.IsDryRunMode();

            if (killSwitchActive && isDryRun)
            {
                _logger.LogInformation("✅ [TEST] Kill switch test PASSED");
                return true;
            }
            else
            {
                _logger.LogError("❌ [TEST] Kill switch test FAILED - kill.txt should force DRY_RUN");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [TEST] Kill switch test FAILED with exception");
            return false;
        }
        finally
        {
            // Clean up
            try
            {
                if (File.Exists("kill.txt"))
                {
                    File.Delete("kill.txt");
                }
            }
            catch { /* Ignore cleanup errors */ }
        }
    }

    private bool TestPriceValidation()
    {
        _logger.LogInformation("🧪 [TEST] Testing ES/MES price validation...");

        try
        {
            // Test ES tick rounding (0.25)
            var price1 = 4125.13m;
            var price2 = 4125.38m;

            var rounded1 = ProductionPriceService.RoundToTick(price1);
            var rounded2 = ProductionPriceService.RoundToTick(price2);

            if (rounded1 == 4125.00m && rounded2 == 4125.50m)
            {
                _logger.LogInformation("✅ [TEST] ES/MES tick rounding test PASSED ({Price1} -> {Rounded1}, {Price2} -> {Rounded2})",
                    price1, rounded1, price2, rounded2);
                return true;
            }
            else
            {
                _logger.LogError("❌ [TEST] ES/MES tick rounding test FAILED ({Price1} -> {Rounded1}, {Price2} -> {Rounded2})",
                    price1, rounded1, price2, rounded2);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [TEST] Price validation test FAILED with exception");
            return false;
        }
    }

    private bool TestRiskValidation()
    {
        _logger.LogInformation("🧪 [TEST] Testing risk validation (reject ≤ 0)...");

        try
        {
            // Test case 1: Valid risk (> 0)
            var validR = ProductionPriceService.RMultiple(4125.00m, 4124.00m, 4127.00m, true, _logger);
            
            // Test case 2: Invalid risk (= 0)
            var invalidR = ProductionPriceService.RMultiple(4125.00m, 4125.00m, 4127.00m, true, _logger);
            
            // Test case 3: Invalid risk (< 0)
            var negativeR = ProductionPriceService.RMultiple(4125.00m, 4126.00m, 4127.00m, true, _logger);

            if (validR.HasValue && !invalidR.HasValue && !negativeR.HasValue)
            {
                _logger.LogInformation("✅ [TEST] Risk validation test PASSED (valid: {Valid}, zero: {Zero}, negative: {Negative})",
                    validR?.ToString("0.00") ?? "null", invalidR?.ToString("0.00") ?? "null", negativeR?.ToString("0.00") ?? "null");
                return true;
            }
            else
            {
                _logger.LogError("❌ [TEST] Risk validation test FAILED (valid: {Valid}, zero: {Zero}, negative: {Negative})",
                    validR?.ToString("0.00") ?? "null", invalidR?.ToString("0.00") ?? "null", negativeR?.ToString("0.00") ?? "null");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [TEST] Risk validation test FAILED with exception");
            return false;
        }
    }

    private async Task<bool> TestOrderEvidenceRequirements()
    {
        _logger.LogInformation("🧪 [TEST] Testing order evidence requirements...");

        try
        {
            // Test with both orderId and fill event (should pass)
            var goodEvidence = await _evidenceService.VerifyOrderFillEvidenceAsync(
                "test-order-123", 
                new TradingBot.Abstractions.GatewayUserTrade 
                { 
                    OrderId = "test-order-123", 
                    FillPrice = 4125.25m, 
                    Quantity = 1 
                },
                "TEST-TAG-GOOD").ConfigureAwait(false);

            // Test with missing evidence (should fail)  
            var badEvidence = await _evidenceService.VerifyOrderFillEvidenceAsync(
                null, null, "TEST-TAG-BAD").ConfigureAwait(false);

            if (goodEvidence.HasSufficientEvidence && !badEvidence.HasSufficientEvidence)
            {
                _logger.LogInformation("✅ [TEST] Order evidence test PASSED (good: {Good}, bad: {Bad})",
                    goodEvidence.HasSufficientEvidence, badEvidence.HasSufficientEvidence);
                return true;
            }
            else
            {
                _logger.LogError("❌ [TEST] Order evidence test FAILED (good: {Good}, bad: {Bad})",
                    goodEvidence.HasSufficientEvidence, badEvidence.HasSufficientEvidence);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [TEST] Order evidence test FAILED with exception");
            return false;
        }
    }

    /// <summary>
    /// Quick test method for console apps
    /// </summary>
    public static async Task<bool> QuickTestAsync()
    {
        var services = new ServiceCollection()
            .AddProductionTradingServices()
            .BuildServiceProvider();

        try
        {
            var tester = ActivatorUtilities.CreateInstance<ProductionGuardrailTester>(services);
            return await tester.RunAllTestsAsync().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Quick test failed: {ex.Message}");
            return false;
        }
    }
}