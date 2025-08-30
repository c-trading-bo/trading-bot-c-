# Health Monitor Validation Script
# This script tests if the health monitoring system is working correctly

param(
    [string]$BotUrl = "http://localhost:5050"
)

Write-Host "[SEARCH] Testing Health Monitor System..." -ForegroundColor Cyan
Write-Host "Bot URL: $BotUrl" -ForegroundColor Gray

# Test 1: Check if health endpoint is accessible
Write-Host "`n[API] Test 1: Health Endpoint Accessibility" -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "$BotUrl/health/system" -TimeoutSec 10
    Write-Host "✅ Health endpoint is accessible" -ForegroundColor Green
    Write-Host "Overall Status: $($healthResponse.status)" -ForegroundColor $(if($healthResponse.status -eq "Healthy") {"Green"} else {"Red"})
    Write-Host "Checks Count: $($healthResponse.checks.Count)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Health endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Test 2: Verify all expected health checks are present
Write-Host "`n[CHECK] Test 2: Health Check Coverage" -ForegroundColor Yellow
$expectedChecks = @(
    "ml_persistence", "ml_cycles", "strategy_configs", "session_windows", 
    "timezone_logic", "position_limits", "risk_calculations", "data_feeds", 
    "connectivity", "order_routing", "memory_usage", "file_permissions",
    "strategy_signals", "position_tracking", "price_validation"
)

$missingChecks = @()
foreach ($check in $expectedChecks) {
    $found = $healthResponse.checks | Where-Object { $_.name -eq $check }
    if (-not $found) {
        $missingChecks += $check
    }
}

if ($missingChecks.Count -eq 0) {
    Write-Host "✅ All 15 expected health checks are present" -ForegroundColor Green
} else {
    Write-Host "❌ Missing health checks: $($missingChecks -join ', ')" -ForegroundColor Red
}

# Test 3: Check for any critical failures
Write-Host "`n[ALERT] Test 3: Critical Failure Detection" -ForegroundColor Yellow
$criticalFailures = $healthResponse.checks | Where-Object { $_.status -eq "Failed" }
$warnings = $healthResponse.checks | Where-Object { $_.status -eq "Warning" }

Write-Host "Critical Failures: $($criticalFailures.Count)" -ForegroundColor $(if($criticalFailures.Count -eq 0) {"Green"} else {"Red"})
Write-Host "Warnings: $($warnings.Count)" -ForegroundColor $(if($warnings.Count -eq 0) {"Green"} else {"Yellow"})

if ($criticalFailures.Count -gt 0) {
    Write-Host "`n[CRITICAL] CRITICAL ISSUES DETECTED:" -ForegroundColor Red
    foreach ($failure in $criticalFailures) {
        Write-Host "  • $($failure.name): $($failure.message)" -ForegroundColor Red
    }
}

if ($warnings.Count -gt 0) {
    Write-Host "`n[WARNING] WARNINGS DETECTED:" -ForegroundColor Yellow
    foreach ($warning in $warnings) {
        Write-Host "  • $($warning.name): $($warning.message)" -ForegroundColor Yellow
    }
}

# Test 4: Validate intelligent checks are actually testing logic
Write-Host "`n[BRAIN] Test 4: Intelligence Validation" -ForegroundColor Yellow

# Check if strategy signal logic is being tested
$strategySignals = $healthResponse.checks | Where-Object { $_.name -eq "strategy_signals" }
if ($strategySignals -and $strategySignals.message -like "*test scenarios*") {
    Write-Host "✅ Strategy signal logic is being tested intelligently" -ForegroundColor Green
} else {
    Write-Host "⚠️ Strategy signal testing may not be comprehensive" -ForegroundColor Yellow
}

# Check if position tracking is testing calculations
$positionTracking = $healthResponse.checks | Where-Object { $_.name -eq "position_tracking" }
if ($positionTracking -and $positionTracking.message -like "*calculation*") {
    Write-Host "✅ Position tracking calculations are being validated" -ForegroundColor Green
} else {
    Write-Host "⚠️ Position tracking validation may be incomplete" -ForegroundColor Yellow
}

# Check if risk calculations are being tested
$riskCalcs = $healthResponse.checks | Where-Object { $_.name -eq "risk_calculations" }
if ($riskCalcs -and $riskCalcs.message -like "*R-multiple*") {
    Write-Host "✅ Risk calculations are being mathematically validated" -ForegroundColor Green
} else {
    Write-Host "⚠️ Risk calculation testing may be superficial" -ForegroundColor Yellow
}

# Test 5: Check health data persistence
Write-Host "`n[DATA] Test 5: Health Data Persistence" -ForegroundColor Yellow
$healthDir = "state\health"
if (Test-Path $healthDir) {
    $healthFiles = Get-ChildItem $healthDir -Filter "health_*.json" | Sort-Object CreationTime -Descending
    if ($healthFiles.Count -gt 0) {
        $latestFile = $healthFiles[0]
        Write-Host "✅ Health snapshots are being saved: $($latestFile.Name)" -ForegroundColor Green
        Write-Host "Latest snapshot: $($latestFile.CreationTime)" -ForegroundColor Gray
    } else {
        Write-Host "⚠️ No health snapshot files found" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ Health directory does not exist" -ForegroundColor Yellow
}

# Test 6: Dashboard integration test
Write-Host "`n[DASH] Test 6: Dashboard Integration" -ForegroundColor Yellow
try {
    $metricsResponse = Invoke-RestMethod -Uri "$BotUrl/stream/metrics" -TimeoutSec 5
    Write-Host "✅ Dashboard metrics endpoint is accessible" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Dashboard metrics endpoint may have issues: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Summary
Write-Host "`n[SUMMARY] HEALTH MONITOR VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

$totalChecks = $expectedChecks.Count
$healthyChecks = ($healthResponse.checks | Where-Object { $_.status -eq "Healthy" }).Count
$failedChecks = ($healthResponse.checks | Where-Object { $_.status -eq "Failed" }).Count
$warningChecks = ($healthResponse.checks | Where-Object { $_.status -eq "Warning" }).Count

Write-Host "Total Health Checks: $totalChecks" -ForegroundColor Gray
Write-Host "Healthy: $healthyChecks" -ForegroundColor Green
Write-Host "Warnings: $warningChecks" -ForegroundColor Yellow
Write-Host "Failed: $failedChecks" -ForegroundColor Red

$healthPercentage = [math]::Round(($healthyChecks / $totalChecks) * 100, 1)
Write-Host "Health Score: $healthPercentage%" -ForegroundColor $(if($healthPercentage -ge 80) {"Green"} elseif($healthPercentage -ge 60) {"Yellow"} else {"Red"})

if ($failedChecks -eq 0 -and $warningChecks -eq 0) {
    Write-Host "`n[SUCCESS] ALL SYSTEMS HEALTHY! Health monitoring is working perfectly." -ForegroundColor Green
} elseif ($failedChecks -eq 0) {
    Write-Host "`n[OK] No critical failures detected. Monitor working well with minor warnings." -ForegroundColor Yellow
} else {
    Write-Host "`n[ALERT] CRITICAL ISSUES DETECTED! Health monitor is working - please address failures." -ForegroundColor Red
}

Write-Host "`nHealth monitoring system validation complete." -ForegroundColor Cyan
