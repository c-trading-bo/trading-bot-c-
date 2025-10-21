# Force Training NOW - Bypass Sunday Schedule
# This script triggers immediate training session in Lab Mode

Write-Host "🚀 FORCE TRAINING NOW" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Set environment variables to force immediate training
$env:LAB_MODE = "1"
$env:FORCE_LAB_NOW = "1"
$env:HISTORICAL_MODE = "0"
$env:DRY_RUN = "1"
$env:SKIP_MODE_PROMPT = "1"  # Skip interactive mode selection

Write-Host "`n✅ Environment configured:" -ForegroundColor Green
Write-Host "   LAB_MODE = 1" -ForegroundColor Gray
Write-Host "   FORCE_LAB_NOW = 1 (bypasses Sunday schedule)" -ForegroundColor Yellow
Write-Host "   SKIP_MODE_PROMPT = 1 (no menu - goes straight to training)" -ForegroundColor Yellow
Write-Host "   HISTORICAL_MODE = 0" -ForegroundColor Gray
Write-Host "   DRY_RUN = 1" -ForegroundColor Gray

Write-Host "`n📊 Starting bot in Lab Mode with immediate training..." -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Build and run
dotnet build TopstepX.Bot.sln --configuration Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Build successful - starting training session..." -ForegroundColor Green
Write-Host "`n⏱️  Expected duration: 2-6 hours for full training" -ForegroundColor Yellow
Write-Host "📈 Progress will be logged to console and log file" -ForegroundColor Yellow
Write-Host "`n🔄 Training will:" -ForegroundColor Cyan
Write-Host "   1. Load 90 days of historical ES/NQ data" -ForegroundColor Gray
Write-Host "   2. Train CVaR-PPO, Neural-UCB, LSTM models" -ForegroundColor Gray
Write-Host "   3. Export ONNX model files" -ForegroundColor Gray
Write-Host "   4. Run promotion evaluation" -ForegroundColor Gray
Write-Host "   5. Promote champions if criteria met" -ForegroundColor Gray
Write-Host "`n💡 TIP: The bot will start in LAB MODE and trigger training immediately" -ForegroundColor Yellow
Write-Host "   Training runs automatically - no menu selection needed!" -ForegroundColor Yellow
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

# Run the bot (LAB_MODE=1 and FORCE_LAB_NOW=1 will auto-select Lab Mode)
dotnet run --project src\UnifiedOrchestrator\UnifiedOrchestrator.csproj --configuration Release -- --select-mode 2
