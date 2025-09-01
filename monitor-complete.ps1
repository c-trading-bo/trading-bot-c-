#!/usr/bin/env pwsh
# 🎯 Complete Trading Bot Monitor
# Enhanced monitoring script with cloud learning status

param(
    [switch]$CloudOnly,
    [switch]$LocalOnly,
    [switch]$Quick,
    [switch]$Detailed
)

Write-Host "🤖 Complete Trading Bot Monitoring Dashboard" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Function to check cloud learning status
function Test-CloudLearning {
    Write-Host "`n☁️ Cloud Learning Pipeline:" -ForegroundColor Blue
    
    try {
        # Check GitHub Actions API (requires token)
        $repoUrl = "https://api.github.com/repos/kevinsuero072897-collab/trading-bot-c-/actions/workflows/train-continuous-clean.yml/runs"
        
        # Try to get workflow status (will fail without token, but shows URL)
        Write-Host "📋 Training Workflow: https://github.com/kevinsuero072897-collab/trading-bot-c-/actions/workflows/train-continuous-clean.yml" -ForegroundColor Cyan
        Write-Host "📊 All Workflows: https://github.com/kevinsuero072897-collab/trading-bot-c-/actions" -ForegroundColor Cyan
        
        # Check for GitHub Actions indicators
        if (Test-Path ".github/workflows/train-continuous-clean.yml") {
            Write-Host "✅ Training Workflow: Configured (runs every 30 minutes)" -ForegroundColor Green
        } else {
            Write-Host "❌ Training Workflow: Not found" -ForegroundColor Red
        }
        
        if (Test-Path ".github/workflows/dashboard-enhanced.yml") {
            Write-Host "✅ Dashboard Monitor: Configured (updates every 5 minutes)" -ForegroundColor Green
        } else {
            Write-Host "❌ Dashboard Monitor: Not found" -ForegroundColor Red
        }
        
        if (Test-Path ".github/workflows/status-badges.yml") {
            Write-Host "✅ Status Badges: Configured (updates every 2 minutes)" -ForegroundColor Green
        } else {
            Write-Host "❌ Status Badges: Not found" -ForegroundColor Red
        }
        
        if (Test-Path ".github/workflows/quality-assurance.yml") {
            Write-Host "✅ Quality Monitor: Configured (checks every 2 hours)" -ForegroundColor Green
        } else {
            Write-Host "❌ Quality Monitor: Not found" -ForegroundColor Red
        }
        
    } catch {
        Write-Host "⚠️ Cloud status check failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Function to check local bot status
function Test-LocalBot {
    Write-Host "`n💻 Local Bot Status:" -ForegroundColor Blue
    
    # Check if bot process is running
    $botProcess = Get-Process -Name "OrchestratorAgent" -ErrorAction SilentlyContinue
    if ($botProcess) {
        Write-Host "✅ Bot Process: RUNNING (PID: $($botProcess.Id))" -ForegroundColor Green
        
        # Check memory usage
        $memoryMB = [math]::Round($botProcess.WorkingSet / 1MB, 2)
        Write-Host "📊 Memory Usage: ${memoryMB} MB" -ForegroundColor White
        
        # Check CPU usage (approximate)
        $cpuPercent = [math]::Round($botProcess.CPU / [Environment]::ProcessorCount, 2)
        Write-Host "⚡ CPU Usage: ${cpuPercent}%" -ForegroundColor White
    } else {
        Write-Host "❌ Bot Process: NOT RUNNING" -ForegroundColor Red
        Write-Host "   Start with: .\launch-bot.ps1" -ForegroundColor Yellow
    }
    
    # Check dashboard accessibility
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5050/healthz" -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Dashboard: http://localhost:5050/dashboard" -ForegroundColor Green
            Write-Host "✅ Health API: Responding" -ForegroundColor Green
            
            # Try to get health data
            try {
                $healthData = Invoke-RestMethod -Uri "http://localhost:5050/health/system" -TimeoutSec 3
                if ($healthData.status) {
                    Write-Host "📊 System Health: $($healthData.status)" -ForegroundColor Green
                }
            } catch {
                Write-Host "⚠️ Health data unavailable" -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "❌ Dashboard: Not accessible" -ForegroundColor Red
        Write-Host "   Check if bot is running and binding to port 5050" -ForegroundColor Yellow
    }
}

# Function to check data and models
function Test-DataAndModels {
    Write-Host "`n📊 Data & Models:" -ForegroundColor Blue
    
    # Check training data
    $dataFiles = Get-ChildItem "data/rl_training/*.jsonl" -ErrorAction SilentlyContinue
    if ($dataFiles) {
        Write-Host "✅ Training Data: $($dataFiles.Count) files" -ForegroundColor Green
        $totalSize = ($dataFiles | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($totalSize / 1MB, 2)
        Write-Host "📁 Total Size: ${sizeMB} MB" -ForegroundColor White
        
        foreach ($file in $dataFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 3) {
            $age = [math]::Round(((Get-Date) - $file.LastWriteTime).TotalHours, 1)
            $lines = (Get-Content $file.FullName | Measure-Object -Line).Lines
            Write-Host "   📄 $($file.Name): $lines records, ${age}h old" -ForegroundColor White
        }
    } else {
        Write-Host "⏳ Training Data: No files yet (bot generates during trading)" -ForegroundColor Yellow
    }
    
    # Check RL models
    $modelFiles = Get-ChildItem "models/rl/*.onnx" -ErrorAction SilentlyContinue
    if ($modelFiles) {
        Write-Host "✅ AI Models: $($modelFiles.Count) models" -ForegroundColor Green
        foreach ($model in $modelFiles | Sort-Object LastWriteTime -Descending) {
            $age = [math]::Round(((Get-Date) - $model.LastWriteTime).TotalHours, 1)
            $sizeMB = [math]::Round($model.Length / 1MB, 2)
            Write-Host "   🤖 $($model.Name): ${sizeMB}MB, ${age}h old" -ForegroundColor White
        }
    } else {
        Write-Host "⏳ AI Models: Not found (created after training data collection)" -ForegroundColor Yellow
    }
    
    # Check for model manifest
    if (Test-Path "models/current.json") {
        Write-Host "✅ Model Manifest: Found" -ForegroundColor Green
        try {
            $manifest = Get-Content "models/current.json" | ConvertFrom-Json
            Write-Host "   📦 Version: $($manifest.version)" -ForegroundColor White
            if ($manifest.signature) {
                Write-Host "   🔐 Security: HMAC signed" -ForegroundColor Green
            }
        } catch {
            Write-Host "   ⚠️ Manifest parsing failed" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⏳ Model Manifest: Not found" -ForegroundColor Yellow
    }
}

# Function to show monitoring URLs
function Show-MonitoringUrls {
    Write-Host "`n🔗 Monitoring URLs:" -ForegroundColor Magenta
    Write-Host "📊 Local Dashboard: http://localhost:5050/dashboard" -ForegroundColor Cyan
    Write-Host "☁️ Enhanced Dashboard: https://kevinsuero072897-collab.github.io/trading-bot-c-/monitoring.html" -ForegroundColor Cyan
    Write-Host "📋 GitHub Actions: https://github.com/kevinsuero072897-collab/trading-bot-c-/actions" -ForegroundColor Cyan
    Write-Host "🚀 Training Workflow: https://github.com/kevinsuero072897-collab/trading-bot-c-/actions/workflows/train-continuous-clean.yml" -ForegroundColor Cyan
    Write-Host "📦 Model Releases: https://github.com/kevinsuero072897-collab/trading-bot-c-/releases" -ForegroundColor Cyan
}

# Function to show quick actions
function Show-QuickActions {
    Write-Host "`n⚡ Quick Actions:" -ForegroundColor Yellow
    Write-Host "🚀 Start Bot: .\launch-bot.ps1" -ForegroundColor White
    Write-Host "📊 Monitor Learning: .\monitor-learning.ps1" -ForegroundColor White
    Write-Host "🔄 Refresh Status: .\monitor-complete.ps1" -ForegroundColor White
    Write-Host "🧪 Test Cloud ML: .\test-cloud-ml.ps1" -ForegroundColor White
    Write-Host "📈 View Dashboard: Start-Process 'http://localhost:5050/dashboard'" -ForegroundColor White
}

# Function to check environment configuration
function Test-Environment {
    Write-Host "`n🔧 Environment Configuration:" -ForegroundColor Blue
    
    if (Test-Path ".env") {
        Write-Host "✅ Environment File: Found" -ForegroundColor Green
        
        $envContent = Get-Content ".env" -Raw
        
        if ($envContent -match "GITHUB_CLOUD_LEARNING=1") {
            Write-Host "✅ Cloud Learning: Enabled" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Cloud Learning: Not configured" -ForegroundColor Yellow
        }
        
        if ($envContent -match "MODEL_MANIFEST_URL") {
            Write-Host "✅ Model Updates: Configured" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Model Updates: Not configured" -ForegroundColor Yellow
        }
        
        if ($envContent -match "MANIFEST_HMAC_KEY") {
            Write-Host "✅ Security: HMAC key configured" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Security: HMAC key missing" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Environment File: Not found" -ForegroundColor Red
        Write-Host "   Copy from .env.sample.local and configure" -ForegroundColor Yellow
    }
    
    # Check for GitHub secrets documentation
    if (Test-Path "GITHUB_SECRETS_SETUP.md") {
        Write-Host "✅ GitHub Secrets Guide: Available" -ForegroundColor Green
    }
    
    if (Test-Path "COMPLETE_MONITORING_GUIDE.md") {
        Write-Host "✅ Monitoring Guide: Available" -ForegroundColor Green
    }
}

# Function to show system status summary
function Show-StatusSummary {
    Write-Host "`n📊 System Status Summary:" -ForegroundColor Green
    Write-Host "=========================" -ForegroundColor Green
    
    $warnings = 0
    $errors = 0
    $checks = 0
    
    # Bot running check
    $checks++
    if (-not (Get-Process -Name "OrchestratorAgent" -ErrorAction SilentlyContinue)) {
        $errors++
        Write-Host "❌ Bot not running" -ForegroundColor Red
    } else {
        Write-Host "✅ Bot running" -ForegroundColor Green
    }
    
    # Dashboard check
    $checks++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5050/healthz" -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Dashboard accessible" -ForegroundColor Green
        } else {
            $warnings++
            Write-Host "⚠️ Dashboard responding with errors" -ForegroundColor Yellow
        }
    } catch {
        $errors++
        Write-Host "❌ Dashboard not accessible" -ForegroundColor Red
    }
    
    # Training data check
    $checks++
    if (Get-ChildItem "data/rl_training/*.jsonl" -ErrorAction SilentlyContinue) {
        Write-Host "✅ Training data present" -ForegroundColor Green
    } else {
        $warnings++
        Write-Host "⚠️ No training data yet" -ForegroundColor Yellow
    }
    
    # Environment check
    $checks++
    if (Test-Path ".env") {
        Write-Host "✅ Environment configured" -ForegroundColor Green
    } else {
        $warnings++
        Write-Host "⚠️ Environment not configured" -ForegroundColor Yellow
    }
    
    # Workflow files check
    $checks++
    if (Test-Path ".github/workflows/train-continuous-clean.yml") {
        Write-Host "✅ Cloud learning configured" -ForegroundColor Green
    } else {
        $warnings++
        Write-Host "⚠️ Cloud learning not configured" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "📊 Health Score: $($checks - $errors - $warnings)/$checks" -ForegroundColor $(if ($errors -eq 0 -and $warnings -eq 0) {"Green"} elseif ($errors -eq 0) {"Yellow"} else {"Red"})
    if ($errors -gt 0) {
        Write-Host "🔴 Errors: $errors" -ForegroundColor Red
    }
    if ($warnings -gt 0) {
        Write-Host "🟡 Warnings: $warnings" -ForegroundColor Yellow
    }
}

# Main execution
try {
    if ($Quick) {
        Show-StatusSummary
    } elseif ($CloudOnly) {
        Test-CloudLearning
        Show-MonitoringUrls
    } elseif ($LocalOnly) {
        Test-LocalBot
        Test-DataAndModels
    } else {
        # Full monitoring
        Test-Environment
        Test-LocalBot
        Test-DataAndModels
        Test-CloudLearning
        
        if ($Detailed) {
            Show-MonitoringUrls
            Show-QuickActions
        }
        
        Show-StatusSummary
    }
    
    Write-Host "`n🎯 Complete monitoring dashboard is active!" -ForegroundColor Green
    Write-Host "For detailed guide, see: COMPLETE_MONITORING_GUIDE.md" -ForegroundColor Cyan
    
} catch {
    Write-Host "`n❌ Monitoring check failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}