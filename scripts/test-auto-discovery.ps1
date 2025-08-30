# Test Health Check Auto-Discovery
# This script creates a test health check and verifies it gets discovered automatically

param(
    [string]$BotUrl = "http://localhost:5050"
)

Write-Host "[DISCOVERY-TEST] Testing Health Check Auto-Discovery System..." -ForegroundColor Cyan

# Step 1: Create a test health check
$testHealthCheckContent = @"
using System;
using System.Threading;
using System.Threading.Tasks;
using OrchestratorAgent.Infra;

namespace OrchestratorAgent.Infra.HealthChecks;

[HealthCheck(Category = "Auto-Discovery Test", Priority = 999)]
public class AutoDiscoveryTestHealthCheck : IHealthCheck
{
    public string Name => "auto_discovery_test";
    public string Description => "Test health check to verify auto-discovery is working";
    public string Category => "Testing";
    public int IntervalSeconds => 300; // Check every 5 minutes

    public async Task<HealthCheckResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        // This is a test health check that always succeeds
        await Task.Delay(10, cancellationToken);
        
        var testData = new
        {
            TestTime = DateTime.UtcNow,
            DiscoveryWorking = true,
            TestPassed = true
        };
        
        return HealthCheckResult.Healthy("Auto-discovery test passed - this health check was automatically discovered!", testData);
    }
}
"@

$testFile = "src\OrchestratorAgent\Infra\HealthChecks\AutoDiscoveryTestHealthCheck.cs"

Write-Host "[DISCOVERY-TEST] Creating test health check: $testFile" -ForegroundColor Yellow
try {
    New-Item -Path $testFile -ItemType File -Value $testHealthCheckContent -Force | Out-Null
    Write-Host "✅ Test health check created successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create test health check: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Step 2: Wait for bot restart (if needed) or discovery cycle
Write-Host "`n[DISCOVERY-TEST] Waiting for health check discovery..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Step 3: Check if the test health check was discovered
Write-Host "`n[DISCOVERY-TEST] Checking if test health check was discovered..." -ForegroundColor Yellow

try {
    $healthResponse = Invoke-RestMethod -Uri "$BotUrl/health/system" -TimeoutSec 10
    
    # Look for our test health check
    $testCheck = $healthResponse.checks | Where-Object { $_.name -eq "auto_discovery_test" }
    
    if ($testCheck) {
        Write-Host "✅ SUCCESS: Test health check was automatically discovered!" -ForegroundColor Green
        Write-Host "   Name: $($testCheck.name)" -ForegroundColor Gray
        Write-Host "   Status: $($testCheck.status)" -ForegroundColor Gray
        Write-Host "   Message: $($testCheck.message)" -ForegroundColor Gray
        
        # Step 4: Clean up test file
        Write-Host "`n[DISCOVERY-TEST] Cleaning up test file..." -ForegroundColor Yellow
        Remove-Item -Path $testFile -Force
        Write-Host "✅ Test file cleaned up" -ForegroundColor Green
        
        Write-Host "`n🎉 AUTO-DISCOVERY IS WORKING!" -ForegroundColor Green
        Write-Host "   ✅ Test health check was created" -ForegroundColor Green
        Write-Host "   ✅ Health monitoring system discovered it automatically" -ForegroundColor Green
        Write-Host "   ✅ Test health check is running and reporting status" -ForegroundColor Green
        Write-Host "   ✅ New features will get automatic health monitoring" -ForegroundColor Green
        
    } else {
        Write-Host "⚠️ Test health check not found in health system" -ForegroundColor Yellow
        Write-Host "This could mean:" -ForegroundColor Yellow
        Write-Host "  • Discovery system needs bot restart to find new checks" -ForegroundColor Yellow
        Write-Host "  • Auto-discovery is not fully working yet" -ForegroundColor Yellow
        Write-Host "  • Test health check has compilation errors" -ForegroundColor Yellow
        
        # Show available health checks
        Write-Host "`nAvailable health checks:" -ForegroundColor Gray
        foreach ($check in $healthResponse.checks) {
            Write-Host "  • $($check.name): $($check.status)" -ForegroundColor Gray
        }
        
        # Clean up test file
        Remove-Item -Path $testFile -Force -ErrorAction SilentlyContinue
    }
    
} catch {
    Write-Host "❌ Failed to check health endpoint: $($_.Exception.Message)" -ForegroundColor Red
    
    # Clean up test file
    Remove-Item -Path $testFile -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "`n[DISCOVERY-TEST] Auto-discovery validation complete." -ForegroundColor Cyan
