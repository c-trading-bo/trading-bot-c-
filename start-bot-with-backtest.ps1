#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start trading bot with historical data adapter for backtesting
    
.DESCRIPTION
    This script starts the TopstepX adapter first (to load 90 days of historical data),
    then starts the bot so it can connect and run backtests via EnhancedBacktestLearningService.
    
.NOTES
    - Adapter must start FIRST and be ready before bot starts
    - Bot's EnhancedBacktestLearningService needs adapter on localhost:8765
    - Adapter loads 90 days of historical data for comprehensive backtesting
#>

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🤖 Trading Bot Startup with Backtesting Infrastructure      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Load environment variables
Write-Host "📦 Loading environment variables from .env..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Host "❌ ERROR: .env file not found!" -ForegroundColor Red
    exit 1
}

Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        
        # Skip comments and empty lines
        if ($name -notmatch '^#' -and $name -ne '') {
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
}

Write-Host "✅ Environment variables loaded" -ForegroundColor Green
Write-Host ""

# Step 2: Verify critical configuration
Write-Host "🔍 Verifying configuration..." -ForegroundColor Yellow

$criticalVars = @{
    'ENABLE_HISTORICAL_LEARNING' = '1'
    'DRY_RUN' = '1'
    'LIVE_ORDERS' = '0'
    'ENABLED_STRATEGIES' = 'S2,S3,S6,S11'
    'CONCURRENT_LEARNING_INTERVAL_MINUTES' = '5'
}

$configOk = $true
foreach ($var in $criticalVars.GetEnumerator()) {
    $value = [Environment]::GetEnvironmentVariable($var.Key)
    if ($value -eq $var.Value) {
        Write-Host "  ✅ $($var.Key) = $value" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ $($var.Key) = $value (expected: $($var.Value))" -ForegroundColor Yellow
        $configOk = $false
    }
}

Write-Host ""

# Step 3: Clean up any existing processes
Write-Host "🧹 Cleaning up existing processes..." -ForegroundColor Yellow

$existingBot = Get-Process -Name "UnifiedOrchestrator" -ErrorAction SilentlyContinue
if ($existingBot) {
    Write-Host "  ⚠️ Stopping existing bot (PID: $($existingBot.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $existingBot.Id -Force
    Start-Sleep -Seconds 2
}

$existingAdapter = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    $cmdLine = $_.CommandLine
    $cmdLine -like "*topstep_x_adapter*"
}

if ($existingAdapter) {
    Write-Host "  ⚠️ Stopping existing adapter processes..." -ForegroundColor Yellow
    $existingAdapter | Stop-Process -Force
    Start-Sleep -Seconds 2
}

Write-Host "✅ Cleanup complete" -ForegroundColor Green
Write-Host ""

# Step 4: Start TopstepX Adapter FIRST
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  STEP 1: Starting TopstepX Historical Data Adapter           ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 The adapter will load 90 days of historical data for ES and NQ" -ForegroundColor Cyan
Write-Host "⏳ This may take 30-60 seconds..." -ForegroundColor Cyan
Write-Host ""

# Start adapter in background
$adapterProcess = Start-Process -FilePath "pwsh" -ArgumentList "-File", "start-adapter.ps1" -PassThru -WindowStyle Minimized

if (-not $adapterProcess) {
    Write-Host "❌ ERROR: Failed to start adapter!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Adapter process started (PID: $($adapterProcess.Id))" -ForegroundColor Green
Write-Host ""

# Step 5: Wait for adapter to initialize and load data
Write-Host "⏳ Waiting for adapter to load 90 days of historical data..." -ForegroundColor Yellow
Write-Host "   This is critical for EnhancedBacktestLearningService to work!" -ForegroundColor Cyan
Write-Host ""

$maxWaitSeconds = 90
$waitInterval = 5
$elapsedSeconds = 0
$adapterReady = $false

while ($elapsedSeconds -lt $maxWaitSeconds) {
    Start-Sleep -Seconds $waitInterval
    $elapsedSeconds += $waitInterval
    
    Write-Host "   ⏱️ Elapsed: $elapsedSeconds seconds..." -ForegroundColor Gray
    
    # Check if adapter is listening on port 8765
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port 8765 -InformationLevel Quiet -WarningAction SilentlyContinue
        if ($connection) {
            Write-Host ""
            Write-Host "✅ Adapter is ready and serving on localhost:8765!" -ForegroundColor Green
            $adapterReady = $true
            break
        }
    } catch {
        # Continue waiting
    }
    
    # Check if adapter process is still running
    if (-not (Get-Process -Id $adapterProcess.Id -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "❌ ERROR: Adapter process crashed!" -ForegroundColor Red
        Write-Host "   Check adapter logs for details" -ForegroundColor Red
        exit 1
    }
}

if (-not $adapterReady) {
    Write-Host ""
    Write-Host "⚠️ WARNING: Adapter did not respond on port 8765 after $maxWaitSeconds seconds" -ForegroundColor Yellow
    Write-Host "   Continuing anyway, but backtest may not work..." -ForegroundColor Yellow
    Write-Host ""
}

# Step 6: Start the Trading Bot
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  STEP 2: Starting Trading Bot                                 ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Find the bot executable
$botExe = "src/UnifiedOrchestrator/bin/Debug/net8.0/UnifiedOrchestrator.exe"
if (-not (Test-Path $botExe)) {
    $botExe = "src/UnifiedOrchestrator/bin/Release/net8.0/UnifiedOrchestrator.exe"
}

if (-not (Test-Path $botExe)) {
    Write-Host "❌ ERROR: Bot executable not found!" -ForegroundColor Red
    Write-Host "   Run: dotnet build" -ForegroundColor Yellow
    exit 1
}

Write-Host "🚀 Starting bot: $botExe" -ForegroundColor Cyan
Write-Host ""

# Start bot
& $botExe

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅ Bot Startup Complete                                      ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "📊 To view backtest results, run:" -ForegroundColor Cyan
Write-Host "   ./run-strategy-backtest.ps1" -ForegroundColor White
Write-Host ""
Write-Host "📋 To monitor logs:" -ForegroundColor Cyan
Write-Host "   Get-Content logs/*.log -Tail 50 -Wait" -ForegroundColor White
Write-Host ""
