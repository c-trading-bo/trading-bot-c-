# Concurrent Learning Startup Script for Windows
# This script ensures your bot runs with concurrent historical learning enabled

Write-Host "🤖 Starting Trading Bot with Concurrent Historical Learning..." -ForegroundColor Green

# Load concurrent learning configuration
$envFile = ".env.concurrent-learning"
if (Test-Path $envFile) {
    Write-Host "📊 Loading concurrent learning configuration..." -ForegroundColor Cyan
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
} else {
    Write-Host "⚠️  No .env.concurrent-learning file found, using defaults..." -ForegroundColor Yellow
    # Set default values for concurrent learning
    $env:CONCURRENT_LEARNING = "1"
    $env:RUN_LEARNING = "1"
    $env:BACKTEST_MODE = "1"
    $env:MAX_CONCURRENT_OPERATIONS = "2"
    $env:LEARNING_PRIORITY = "LOW"
    $env:LIVE_TRADING_PRIORITY = "HIGH"
    $env:CONCURRENT_LEARNING_INTERVAL_MINUTES = "60"
    $env:OFFLINE_LEARNING_INTERVAL_MINUTES = "15"
    $env:CONCURRENT_LEARNING_DAYS = "7"
    $env:OFFLINE_LEARNING_DAYS = "30"
}

# Load main environment configuration
$mainEnvFile = ".env"
if (Test-Path $mainEnvFile) {
    Write-Host "🔧 Loading main environment configuration..." -ForegroundColor Cyan
    Get-Content $mainEnvFile | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

# Display learning configuration
Write-Host ""
Write-Host "📈 CONCURRENT LEARNING CONFIGURATION:" -ForegroundColor Magenta
Write-Host "   CONCURRENT_LEARNING: $env:CONCURRENT_LEARNING" -ForegroundColor White
Write-Host "   RUN_LEARNING: $env:RUN_LEARNING" -ForegroundColor White
Write-Host "   BACKTEST_MODE: $env:BACKTEST_MODE" -ForegroundColor White
Write-Host "   Learning during market hours: Every $env:CONCURRENT_LEARNING_INTERVAL_MINUTES minutes" -ForegroundColor White
Write-Host "   Learning during market closed: Every $env:OFFLINE_LEARNING_INTERVAL_MINUTES minutes" -ForegroundColor White
Write-Host ""

# Start the bot
Write-Host "🚀 Starting Unified Trading Orchestrator with concurrent learning..." -ForegroundColor Green
Set-Location "src\UnifiedOrchestrator"
dotnet run

Write-Host "🛑 Trading Bot with concurrent learning stopped" -ForegroundColor Red