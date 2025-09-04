# PowerShell script to restore all workflows from working commit
$workingCommit = "f9aa43a"
$workflowDir = ".github/workflows"

# List of workflow files that need to be restored
$workflows = @(
    "auto-label-automerge.yml",
    "ci.yml",
    "cloud-ml-training.yml", 
    "congress_trades.yml",
    "cot_report.yml",
    "daily_report.yml",
    "dashboard-enhanced.yml",
    "dotnet-desktop.yml",
    "earnings_whisper.yml",
    "failed_patterns.yml",
    "fed_liquidity.yml",
    "intermarket.yml",
    "live-dashboard.yml",
    "market_data.yml",
    "microstructure.yml",
    "ml_trainer.yml",
    "mm_positioning.yml",
    "news_pulse.yml",
    "opex_calendar.yml",
    "options_flow.yml",
    "overnight.yml",
    "quality-assurance.yml",
    "rebalancing.yml",
    "regime_detector.yml",
    "seasonality.yml",
    "sector_rotation.yml",
    "social_momentum.yml",
    "status-badges.yml",
    "train-continuous-clean.yml",
    "train-continuous-final.yml",
    "train-continuous-fixed.yml",
    "train-continuous.yml",
    "train-github-only.yml",
    "volatility_surface.yml",
    "zones_identifier.yml"
)

Write-Host "🔄 RESTORING WORKFLOWS FROM COMMIT $workingCommit" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

$restored = 0
$errors = 0

foreach ($workflow in $workflows) {
    try {
        Write-Host "Restoring $workflow..." -ForegroundColor Yellow
        
        # Restore workflow from git
        git show "${workingCommit}:${workflowDir}/${workflow}" | Out-File -FilePath "${workflowDir}/${workflow}" -Encoding UTF8
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Restored $workflow" -ForegroundColor Green
            $restored++
        } else {
            Write-Host "❌ Failed to restore $workflow" -ForegroundColor Red
            $errors++
        }
    } catch {
        Write-Host "❌ Error restoring $workflow: $_" -ForegroundColor Red
        $errors++
    }
}

Write-Host "" 
Write-Host "📊 RESTORATION SUMMARY:" -ForegroundColor Cyan
Write-Host "✅ Restored: $restored workflows" -ForegroundColor Green
Write-Host "❌ Errors: $errors workflows" -ForegroundColor Red
Write-Host ""
Write-Host "🔍 Checking restored workflows..." -ForegroundColor Yellow

# Verify restored workflows
$fixedWorkflows = Get-ChildItem "$workflowDir/*.yml" | Where-Object { 
    (Get-Content $_.FullName -Raw) -match "name: Fixed Workflow" 
}

if ($fixedWorkflows.Count -gt 0) {
    Write-Host "⚠️ WARNING: $($fixedWorkflows.Count) workflows still have 'Fixed Workflow' template:" -ForegroundColor Yellow
    $fixedWorkflows | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Red }
} else {
    Write-Host "✅ No 'Fixed Workflow' templates found - restoration successful!" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Review restored workflows: git diff" -ForegroundColor White  
Write-Host "2. Test a workflow: git commit and push" -ForegroundColor White
Write-Host "3. Monitor workflow runs: gh run list" -ForegroundColor White
