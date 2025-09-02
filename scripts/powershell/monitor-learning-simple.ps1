#!/usr/bin/env pwsh
# Bot Learning Monitor

Write-Host "Bot Learning Status Monitor" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan

# 1. Check if bot is running
$bot = Get-Process -Name "OrchestratorAgent" -ErrorAction SilentlyContinue
if ($bot) {
    Write-Host "Bot Status: RUNNING (PID: $($bot.Id))" -ForegroundColor Green
} else {
    Write-Host "Bot Status: NOT RUNNING" -ForegroundColor Red
    Write-Host "   Start with: .\launch-bot.ps1" -ForegroundColor Yellow
}

# 2. Check training data collection
Write-Host "`nTraining Data Collection:" -ForegroundColor Blue
$dataFiles = Get-ChildItem "data/rl_training/*.jsonl" -ErrorAction SilentlyContinue
if ($dataFiles) {
    Write-Host "Training Files: $($dataFiles.Count)" -ForegroundColor Green
    foreach ($file in $dataFiles) {
        $lines = (Get-Content $file.FullName | Measure-Object -Line).Lines
        $size = [math]::Round($file.Length / 1KB, 2)
        Write-Host "   $($file.Name): $lines records, ${size}KB" -ForegroundColor White
    }
} else {
    Write-Host "No training data yet - bot will generate as it trades" -ForegroundColor Yellow
}

# 3. Check RL models
Write-Host "`nRL Models:" -ForegroundColor Blue
$modelFiles = Get-ChildItem "models/rl/*.onnx" -ErrorAction SilentlyContinue
if ($modelFiles) {
    Write-Host "AI Models: $($modelFiles.Count)" -ForegroundColor Green
    foreach ($model in $modelFiles) {
        $age = [math]::Round(((Get-Date) - $model.LastWriteTime).TotalHours, 1)
        Write-Host "   $($model.Name): Updated ${age}h ago" -ForegroundColor White
    }
} else {
    Write-Host "No models yet - will be created after collecting training data" -ForegroundColor Yellow
}

# 4. Check cloud learning status
Write-Host "`nCloud Learning:" -ForegroundColor Blue
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "GITHUB_CLOUD_LEARNING=1") {
        Write-Host "GitHub Actions: Enabled (trains every 6 hours)" -ForegroundColor Green
        Write-Host "   Check: https://github.com/kevinsuero072897-collab/trading-bot-c-/actions" -ForegroundColor Cyan
    } else {
        Write-Host "Cloud learning not configured" -ForegroundColor Yellow
    }
} else {
    Write-Host ".env file not found" -ForegroundColor Yellow
}

# 5. Check bot dashboard
Write-Host "`nBot Dashboard:" -ForegroundColor Blue
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5050/healthz" -UseBasicParsing -TimeoutSec 3
    if ($response.StatusCode -eq 200) {
        Write-Host "Dashboard: http://localhost:5050/dashboard" -ForegroundColor Green
        Write-Host "Health API: http://localhost:5050/healthz" -ForegroundColor Green
    }
} catch {
    Write-Host "Dashboard: Not accessible (bot may be starting)" -ForegroundColor Red
}

# 6. Real-time learning indicators to watch for
Write-Host "`nWhat to Watch for in Bot Logs:" -ForegroundColor Yellow
Write-Host "   Trade signals: [Strategy] entries/exits" -ForegroundColor White
Write-Host "   Position sizing: [RlSizer] using X% allocation" -ForegroundColor White
Write-Host "   Data collection: [RlTrainingDataCollector] writing features" -ForegroundColor White
Write-Host "   Model training: [AutoRlTrainer] background training" -ForegroundColor White
Write-Host "   Cloud sync: [CloudRL] upload/download activities" -ForegroundColor White

# 7. Learning timeline
Write-Host "`nLearning Timeline:" -ForegroundColor Magenta
Write-Host "   Immediate: Data collection (17 features per trade)" -ForegroundColor White
Write-Host "   Every 6 hours: Local model training (AutoRlTrainer)" -ForegroundColor White
Write-Host "   Every 6 hours: Cloud model training (GitHub Actions)" -ForegroundColor White
Write-Host "   Continuous: Model updates and performance improvement" -ForegroundColor White

Write-Host "`nYour bot is learning 24/7! Check back periodically to see progress." -ForegroundColor Green
