#!/usr/bin/env pwsh
# Clean lock files and launch Lab Mode training

Write-Host "`n🧹 Cleaning stale lock files..." -ForegroundColor Yellow
Remove-Item -Path "state/*.lock" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$env:TEMP/qbot*.lock" -Force -ErrorAction SilentlyContinue
Write-Host "✅ Lock files cleaned`n" -ForegroundColor Green

Write-Host "🚀 Launching bot in Lab Mode with forced training..." -ForegroundColor Cyan
Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "  LAB_MODE=1                    # Training mode (offline)"
Write-Host "  FORCE_LAB_NOW=1               # Skip Sunday schedule"
Write-Host "  FORCE_STRATEGY_EVAL_LAB=1     # S6/S11 anytime evaluation"
Write-Host "  DRY_RUN=1                     # Safe simulation"
Write-Host "  SKIP_MODE_PROMPT=1            # No interactive menu"
Write-Host ""

$env:LAB_MODE = "1"
$env:FORCE_LAB_NOW = "1"
$env:FORCE_STRATEGY_EVAL_LAB = "1"
$env:DRY_RUN = "1"
$env:SKIP_MODE_PROMPT = "1"

dotnet run --project src\UnifiedOrchestrator\UnifiedOrchestrator.csproj --configuration Release
