#!/usr/bin/env pwsh
# Simple bot launcher - Just run this and it trains!

Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🤖 TRADING BOT - AUTOMATIC TRAINING MODE              ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Step 1: Kill any old bot processes
Write-Host "🛑 Stopping old bot instances..." -ForegroundColor Yellow
Get-Process -Name "dotnet","UnifiedOrchestrator" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Write-Host "✅ Old processes stopped`n" -ForegroundColor Green

# Step 2: Clean up ALL lock files
Write-Host "🧹 Cleaning ALL lock files..." -ForegroundColor Yellow
Remove-Item -Path "state/*.lock" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$env:TEMP/qbot*.lock" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$env:TEMP/training.lock" -Force -ErrorAction SilentlyContinue
$labLock = "$env:LOCALAPPDATA\Temp\qbot_lab_training.lock"
if (Test-Path $labLock) {
    Remove-Item -Path $labLock -Force
    Write-Host "  ✓ Removed lab training lock" -ForegroundColor Green
}
Write-Host "✅ All lock files cleaned`n" -ForegroundColor Green

# Step 3: Set training environment
Write-Host "⚙️  Setting training mode..." -ForegroundColor Yellow
$env:LAB_MODE = "1"                     # Training mode (offline)
$env:FORCE_LAB_NOW = "1"                # Start training immediately
$env:FORCE_STRATEGY_EVAL_LAB = "1"      # S6/S11 can learn from all data
$env:DRY_RUN = "1"                      # Safe mode
$env:SKIP_MODE_PROMPT = "1"             # No prompts
Write-Host "✅ Training environment ready`n" -ForegroundColor Green

# Step 4: Launch bot
Write-Host "🚀 Launching bot - Training will start automatically..." -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 What the bot will do:" -ForegroundColor Yellow
Write-Host "  • Load 1,520 historical trading experiences"
Write-Host "  • Train 25 ML/RL components (Heavy/Medium/Light phases)"
Write-Host "  • Update models every 5 minutes automatically"
Write-Host "  • Learn from S6/S11 strategies across all time periods"
Write-Host ""
Write-Host "Press Ctrl+C to stop training" -ForegroundColor Gray
Write-Host "═══════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

dotnet run --project src\UnifiedOrchestrator\UnifiedOrchestrator.csproj --configuration Release
