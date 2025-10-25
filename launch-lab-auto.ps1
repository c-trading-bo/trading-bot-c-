# Auto-launch Lab Mode training
$env:ASPNETCORE_ENVIRONMENT="Lab"
$env:FORCE_LAB_NOW="1"

Write-Host "Launching Lab Mode..."
Write-Host "Selecting option 2 (Lab Mode)..."
Write-Host "Selecting option 2 (Manual Training - Run Now)..."

# Create input file with menu selections
"2`n2" | Out-File -FilePath "$env:TEMP\lab_input.txt" -Encoding ASCII -NoNewline

# Launch with input redirection
Get-Content "$env:TEMP\lab_input.txt" | dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build
