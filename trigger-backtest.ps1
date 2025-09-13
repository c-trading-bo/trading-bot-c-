# 🎯 Manual Backtesting Trigger Script
# This script will trigger your backtesting/learning system with historical data

Write-Host "🤖 Starting Backtesting & Learning System..." -ForegroundColor Green

# Set environment variables for backtesting mode
$env:RUN_LEARNING = "1"
$env:PROMOTE_TUNER = "1"
$env:INSTANT_ALLOW_LIVE = "1"
$env:LEARN_STRICT = "true"
$env:BACKTEST_MODE = "1"

# Use demo contract IDs for backtesting (these will work for historical data fetching)
$env:TOPSTEPX_EVAL_ES_ID = "11111111-2222-3333-4444-555555555555"
$env:TOPSTEPX_EVAL_NQ_ID = "22222222-3333-4444-5555-666666666666"

Write-Host "✅ Environment configured for backtesting" -ForegroundColor Green
Write-Host "📊 RUN_LEARNING: $env:RUN_LEARNING" -ForegroundColor Yellow
Write-Host "🔧 PROMOTE_TUNER: $env:PROMOTE_TUNER" -ForegroundColor Yellow
Write-Host "⚡ INSTANT_ALLOW_LIVE: $env:INSTANT_ALLOW_LIVE" -ForegroundColor Yellow

# Build the project first
Write-Host "🔨 Building UnifiedOrchestrator..." -ForegroundColor Blue
dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful!" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "🚀 Launching UnifiedOrchestrator with backtesting enabled..." -ForegroundColor Green
    Write-Host "📈 Your ML/RL system will train on historical market data" -ForegroundColor Cyan
    Write-Host "🧠 AdaptiveLearner, NightlyParameterTuner, and TuningRunner are active" -ForegroundColor Cyan
    Write-Host ""
    
    # Launch the orchestrator with backtesting mode
    dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj
} else {
    Write-Host "❌ Build failed. Please check the errors above." -ForegroundColor Red
}