#!/usr/bin/env pwsh
# Launch bot with interactive mode selection menu
# Clears any existing mode environment variables first

Write-Host "🧹 Clearing mode environment variables..." -ForegroundColor Cyan

# Clear all mode-related environment variables
Remove-Item Env:\LAB_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\HISTORICAL_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\BOT_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\OFFLINE_TRAINING -ErrorAction SilentlyContinue
Remove-Item Env:\RL_RUNTIME_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\SKIP_MODE_PROMPT -ErrorAction SilentlyContinue
Remove-Item Env:\DRY_RUN -ErrorAction SilentlyContinue

Write-Host "✅ Environment cleared" -ForegroundColor Green
Write-Host "📊 Current environment:"
Write-Host "   BOT_MODE: $env:BOT_MODE" -ForegroundColor Yellow
Write-Host "   LAB_MODE: $env:LAB_MODE" -ForegroundColor Yellow
Write-Host "   HISTORICAL_MODE: $env:HISTORICAL_MODE" -ForegroundColor Yellow
Write-Host ""

# Launch bot - will show interactive menu
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj
