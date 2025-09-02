#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Test the new GitHub Actions dashboard integration

.DESCRIPTION
    Tests the new dashboard features including bot control API endpoints
    and GitHub Actions integration
#>

param(
    [int]$Port = 5050
)

function Write-TestResult {
    param([string]$Test, [bool]$Passed, [string]$Message = "")
    
    $status = if ($Passed) { "✅ PASS" } else { "❌ FAIL" }
    $color = if ($Passed) { "Green" } else { "Red" }
    
    Write-Host "$status - $Test" -ForegroundColor $color
    if ($Message) {
        Write-Host "    $Message" -ForegroundColor Gray
    }
}

function Test-FileExists {
    param([string]$Path, [string]$Description)
    
    $exists = Test-Path $Path
    Write-TestResult $Description $exists $(if (-not $exists) { "File not found: $Path" })
    return $exists
}

function Test-BuildSuccess {
    Write-Host "`n🔧 Testing Solution Build..." -ForegroundColor Yellow
    
    try {
        $output = dotnet build --verbosity quiet 2>&1
        $success = $LASTEXITCODE -eq 0
        Write-TestResult "Solution builds without errors" $success $(if (-not $success) { $output })
        return $success
    } catch {
        Write-TestResult "Solution builds without errors" $false $_.Exception.Message
        return $false
    }
}

function Test-GitHubWorkflow {
    Write-Host "`n📋 Testing GitHub Actions Workflow..." -ForegroundColor Yellow
    
    $workflowPath = ".github/workflows/live-dashboard.yml"
    $exists = Test-FileExists $workflowPath "GitHub Actions workflow exists"
    
    if ($exists) {
        try {
            $content = Get-Content $workflowPath -Raw
            
            # Check for key components
            $hasPages = $content -match "github-pages"
            $hasSchedule = $content -match "cron:"
            $hasPython = $content -match "python"
            $hasArtifact = $content -match "upload-pages-artifact"
            
            Write-TestResult "Workflow has GitHub Pages deployment" $hasPages
            Write-TestResult "Workflow has scheduled execution" $hasSchedule
            Write-TestResult "Workflow has Python setup" $hasPython
            Write-TestResult "Workflow uploads dashboard artifacts" $hasArtifact
            
            return $hasPages -and $hasSchedule -and $hasPython -and $hasArtifact
        } catch {
            Write-TestResult "Workflow file is valid YAML" $false $_.Exception.Message
            return $false
        }
    }
    
    return $false
}

function Test-DashboardModule {
    Write-Host "`n🎮 Testing Dashboard Module..." -ForegroundColor Yellow
    
    $modulePath = "src/Dashboard/DashboardModule.cs"
    $exists = Test-FileExists $modulePath "Dashboard module exists"
    
    if ($exists) {
        try {
            $content = Get-Content $modulePath -Raw
            
            # Check for new API endpoints
            $hasStartEndpoint = $content -match "/api/bot/start"
            $hasStopEndpoint = $content -match "/api/bot/stop"
            $hasModeEndpoint = $content -match "/api/bot/mode"
            $hasStatusEndpoint = $content -match "/api/status"
            $hasGitHubEndpoint = $content -match "/api/github/status"
            
            Write-TestResult "Has bot start API endpoint" $hasStartEndpoint
            Write-TestResult "Has bot stop API endpoint" $hasStopEndpoint
            Write-TestResult "Has bot mode API endpoint" $hasModeEndpoint
            Write-TestResult "Has status API endpoint" $hasStatusEndpoint
            Write-TestResult "Has GitHub integration endpoint" $hasGitHubEndpoint
            
            return $hasStartEndpoint -and $hasStopEndpoint -and $hasModeEndpoint -and $hasStatusEndpoint -and $hasGitHubEndpoint
        } catch {
            Write-TestResult "Dashboard module is readable" $false $_.Exception.Message
            return $false
        }
    }
    
    return $false
}

function Test-EnhancedDashboard {
    Write-Host "`n🎨 Testing Enhanced Dashboard..." -ForegroundColor Yellow
    
    $dashboardPath = "wwwroot/dashboard.html"
    $exists = Test-FileExists $dashboardPath "Dashboard HTML exists"
    
    if ($exists) {
        try {
            $content = Get-Content $dashboardPath -Raw
            
            # Check for new features
            $hasGitHubIntegration = $content -match "GitHub.*dashboard"
            $hasBotControls = $content -match "startBotWithMode|stopBot|changeBotMode"
            $hasModeSelector = $content -match "paper.*shadow.*live"
            $hasAutoConnect = $content -match "checkGitHubDashboardConnection"
            
            Write-TestResult "Has GitHub Actions integration" $hasGitHubIntegration
            Write-TestResult "Has bot control functions" $hasBotControls
            Write-TestResult "Has mode selector" $hasModeSelector
            Write-TestResult "Has auto-connect functionality" $hasAutoConnect
            
            return $hasGitHubIntegration -and $hasBotControls -and $hasModeSelector -and $hasAutoConnect
        } catch {
            Write-TestResult "Dashboard HTML is readable" $false $_.Exception.Message
            return $false
        }
    }
    
    return $false
}

function Test-LauncherScript {
    Write-Host "`n🚀 Testing Enhanced Launcher..." -ForegroundColor Yellow
    
    $launcherPath = "launch-bot-enhanced.ps1"
    $exists = Test-FileExists $launcherPath "Enhanced launcher exists"
    
    if ($exists) {
        try {
            $content = Get-Content $launcherPath -Raw
            
            # Check for key features
            $hasModeParam = $content -match "ValidateSet.*paper.*shadow.*live"
            $hasGitHubIntegration = $content -match "GitHub.*dashboard"
            $hasWarnings = $content -match "WARNING.*LIVE.*TRADING"
            $hasRegistration = $content -match "registerWithGitHubDashboard|RegisterWithGitHubDashboard"
            
            Write-TestResult "Has trading mode validation" $hasModeParam
            Write-TestResult "Has GitHub dashboard integration" $hasGitHubIntegration
            Write-TestResult "Has live trading warnings" $hasWarnings
            Write-TestResult "Has dashboard registration" $hasRegistration
            
            return $hasModeParam -and $hasGitHubIntegration -and $hasWarnings
        } catch {
            Write-TestResult "Launcher script is readable" $false $_.Exception.Message
            return $false
        }
    }
    
    return $false
}

# Main test execution
Write-Host "🧪 Testing GitHub Actions Dashboard Integration" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

$allTests = @()

# Test file existence and basic functionality
$allTests += Test-BuildSuccess
$allTests += Test-GitHubWorkflow
$allTests += Test-DashboardModule
$allTests += Test-EnhancedDashboard
$allTests += Test-LauncherScript

# Summary
Write-Host "`n📊 Test Summary" -ForegroundColor Cyan
Write-Host "=" * 30 -ForegroundColor Cyan

$passedTests = ($allTests | Where-Object { $_ -eq $true }).Count
$totalTests = $allTests.Count
$passRate = [math]::Round(($passedTests / $totalTests) * 100, 1)

Write-Host "Passed: $passedTests / $totalTests ($passRate%)" -ForegroundColor $(if ($passRate -eq 100) { "Green" } else { "Yellow" })

if ($passRate -eq 100) {
    Write-Host "`n✅ All tests passed! Dashboard implementation is ready." -ForegroundColor Green
    Write-Host "🚀 You can now:" -ForegroundColor Cyan
    Write-Host "  • Run the enhanced launcher: .\launch-bot-enhanced.ps1" -ForegroundColor White
    Write-Host "  • Access the local dashboard: http://localhost:$Port/dashboard" -ForegroundColor White
    Write-Host "  • View the GitHub Actions dashboard: https://kevinsuero072897-collab.github.io/trading-bot-c-/" -ForegroundColor White
} else {
    Write-Host "`n⚠️  Some tests failed. Please review the implementation." -ForegroundColor Yellow
}

Write-Host "`n🔗 Dashboard URLs:" -ForegroundColor Cyan
Write-Host "  Local:  http://localhost:$Port/dashboard" -ForegroundColor White
Write-Host "  GitHub: https://kevinsuero072897-collab.github.io/trading-bot-c-/" -ForegroundColor White
Write-Host "  Actions: https://github.com/kevinsuero072897-collab/trading-bot-c-/actions" -ForegroundColor White