# Auto-launch Lab Mode training with proper menu selection
# This script automatically selects:
#   Option 2: Lab Mode (Historical Training)
#   Option 2: Manual Training (Run Now)

Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║            QBot Lab Mode - Automatic Launch                    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will start Lab Mode training immediately (bypassing Sunday schedule)" -ForegroundColor Yellow
Write-Host ""

# Set environment variables for Lab Mode
$env:ASPNETCORE_ENVIRONMENT = "Lab"
$env:LAB_MODE = "1"
$env:FORCE_LAB_NOW = "1"

Write-Host "[*] Building project..." -ForegroundColor Cyan
dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -c Release --verbosity quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "[✗] Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[✓] Build successful" -ForegroundColor Green
Write-Host ""
Write-Host "[*] Launching Lab Mode..." -ForegroundColor Cyan
Write-Host "    Auto-selecting: [2] Lab Mode → [2] Manual Training" -ForegroundColor Gray
Write-Host ""

# Create input file with proper line endings (CRLF for Windows)
$menuSelections = "2`r`n2`r`n"
$menuSelections | Out-File -FilePath "$env:TEMP\lab_input.txt" -Encoding ASCII -NoNewline

# Launch with input redirection
Get-Content "$env:TEMP\lab_input.txt" | dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -c Release

# Cleanup
Remove-Item "$env:TEMP\lab_input.txt" -ErrorAction SilentlyContinue
