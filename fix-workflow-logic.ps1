#!/usr/bin/env powershell
# Fix Workflow Logic Issues
# Updates GitHub Actions versions and adds permissions

$workflowDir = ".github\workflows"
$workflows = @(
    "congress_trades.yml",
    "cot_report.yml", 
    "failed_patterns.yml",
    "fed_liquidity.yml",
    "intermarket.yml",
    "mm_positioning.yml",
    "opex_calendar.yml", 
    "options_flow.yml",
    "overnight.yml",
    "rebalancing.yml",
    "regime_detector.yml",
    "seasonality.yml",
    "sector_rotation.yml",
    "volatility_surface.yml",
    "zones_identifier.yml"
)

Write-Host "Fixing workflow logic issues..." -ForegroundColor Green

foreach ($workflow in $workflows) {
    $filePath = Join-Path $workflowDir $workflow
    if (Test-Path $filePath) {
        Write-Host "Fixing $workflow..." -ForegroundColor Yellow
        
        # Read content
        $content = Get-Content $filePath -Raw
        
        # Fix GitHub Actions versions
        $content = $content -replace "actions/checkout@v2", "actions/checkout@v4"
        $content = $content -replace "actions/setup-python@v2", "actions/setup-python@v4"
        
        # Add permissions block if missing and has git push
        if ($content -match "git push" -and $content -notmatch "permissions:") {
            # Insert permissions after the 'on:' block
            $content = $content -replace "(on:\s*\r?\n(?:.*\r?\n)*?(?=jobs:))", "`$1`npermissions:`n  contents: write`n`n"
        }
        
        # Write back
        Set-Content $filePath $content -NoNewline
        Write-Host "Fixed $workflow" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "All workflow logic fixes completed!" -ForegroundColor Green
