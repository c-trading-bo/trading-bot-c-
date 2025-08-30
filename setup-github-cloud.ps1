#!/usr/bin/env pwsh
# GitHub Cloud Learning Setup (No AWS Required!)

Write-Host "Setting up FREE GitHub Cloud Learning..." -ForegroundColor Cyan

# Create directories
Write-Host "Creating training directories..." -ForegroundColor Blue
New-Item -ItemType Directory -Path "data/rl_training" -Force | Out-Null
New-Item -ItemType Directory -Path "models/rl" -Force | Out-Null
New-Item -ItemType Directory -Path "cloud_sync" -Force | Out-Null

# Create .env with GitHub cloud settings
$envFile = ".env"
if (Test-Path $envFile) {
    $envContent = Get-Content $envFile -Raw
} else {
    $envContent = ""
}

$cloudSettings = @"

# GitHub Cloud Learning (FREE!)
CLOUD_PROVIDER=github
RL_ENABLED=1
GITHUB_CLOUD_LEARNING=1

"@

if ($envContent -notmatch "GITHUB_CLOUD_LEARNING") {
    Add-Content -Path $envFile -Value $cloudSettings
    Write-Host "Environment configured for GitHub cloud learning" -ForegroundColor Green
} else {
    Write-Host "GitHub cloud settings already exist" -ForegroundColor Yellow
}

# Create a sample training data file to test
$sampleData = @'
{"timestamp":"2025-08-30T12:00:00Z","price":5000.0,"volume":100,"features":{"rsi":45.2,"atr":12.5}}
'@

$sampleData | Out-File -FilePath "data/rl_training/sample_training_data.jsonl" -Encoding UTF8

Write-Host ""
Write-Host "FREE GitHub Cloud Learning Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "What you get:" -ForegroundColor Yellow
Write-Host "  ✅ FREE cloud training every 6 hours" -ForegroundColor White
Write-Host "  ✅ Models stored in GitHub releases" -ForegroundColor White  
Write-Host "  ✅ No AWS costs or setup required" -ForegroundColor White
Write-Host "  ✅ Learning continues 24/7 automatically" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Push your code: git add . && git commit -m 'Add cloud learning' && git push" -ForegroundColor Cyan
Write-Host "  2. Go to GitHub Actions tab and run 'Cloud ML Training Pipeline'" -ForegroundColor Cyan
Write-Host "  3. Start your bot: .\launch-bot.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your bot will learn continuously even when your computer is OFF!" -ForegroundColor Green
