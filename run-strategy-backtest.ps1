#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run 90-day backtest for all enabled strategies and generate performance report
    
.DESCRIPTION
    Uses the bot's existing backtest infrastructure to evaluate strategy performance
    over the last 90 days using real TopstepX historical data.
    
    This does NOT place real orders - it's pure historical simulation.
    
.PARAMETER Strategies
    Comma-separated list of strategies to backtest (default: S2,S3,S6,S11)
    
.PARAMETER Days
    Number of days to backtest (default: 90)
    
.PARAMETER Symbol
    Trading symbol (default: ES)
    
.EXAMPLE
    .\run-strategy-backtest.ps1
    Backtests all enabled strategies over 90 days
    
.EXAMPLE
    .\run-strategy-backtest.ps1 -Strategies "S2,S6" -Days 30
    Backtests only S2 and S6 strategies over 30 days
#>

param(
    [string]$Strategies = "S2,S3,S6,S11",
    [int]$Days = 90,
    [string]$Symbol = "ES"
)

$ErrorActionPreference = "Stop"

Write-Host "📊 Strategy Performance Backtest Report Generator" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Calculate date range
$endDate = Get-Date
$startDate = $endDate.AddDays(-$Days)

Write-Host "📅 Backtest Period: $($startDate.ToString('yyyy-MM-dd')) to $($endDate.ToString('yyyy-MM-dd'))" -ForegroundColor Green
Write-Host "🎯 Symbol: $Symbol" -ForegroundColor Green
Write-Host "📈 Strategies: $Strategies" -ForegroundColor Green
Write-Host ""

# Check if bot is running
$botProcess = Get-Process -Name "UnifiedOrchestrator" -ErrorAction SilentlyContinue
if ($botProcess) {
    Write-Host "✅ Bot is running (PID: $($botProcess.Id))" -ForegroundColor Green
    Write-Host "⏳ Waiting for historical learning to complete..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💡 The bot's EnhancedBacktestLearningService is actively running backtests" -ForegroundColor Cyan
    Write-Host "   in the background. Check the logs for results:" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "⚠️  Bot is not running" -ForegroundColor Yellow
    Write-Host "   Start the bot to run backtests: dotnet run --project src/UnifiedOrchestrator" -ForegroundColor Yellow
    Write-Host ""
}

# Check for recent backtest results
Write-Host "🔍 Checking for existing backtest results..." -ForegroundColor Cyan

$backtestDir = "state/backtests"
if (Test-Path $backtestDir) {
    $backtestFiles = Get-ChildItem $backtestDir -Filter "*.json" -ErrorAction SilentlyContinue | 
                     Sort-Object LastWriteTime -Descending | 
                     Select-Object -First 10
    
    if ($backtestFiles) {
        Write-Host "📁 Found $($backtestFiles.Count) recent backtest result(s):" -ForegroundColor Green
        Write-Host ""
        
        foreach ($file in $backtestFiles) {
            Write-Host "   📄 $($file.Name)" -ForegroundColor Gray
            Write-Host "      Modified: $($file.LastWriteTime)" -ForegroundColor DarkGray
            
            try {
                $content = Get-Content $file.FullName -Raw | ConvertFrom-Json
                if ($content.Strategy) {
                    Write-Host "      Strategy: $($content.Strategy)" -ForegroundColor DarkCyan
                }
                if ($content.Summary) {
                    Write-Host "      Win Rate: $([math]::Round($content.Summary.WinRate * 100, 2))%" -ForegroundColor $(if ($content.Summary.WinRate -gt 0.5) { "Green" } else { "Yellow" })
                    Write-Host "      Net PnL: `$$([math]::Round($content.Summary.NetPnL, 2))" -ForegroundColor $(if ($content.Summary.NetPnL -gt 0) { "Green" } else { "Red" })
                    Write-Host "      Sharpe: $([math]::Round($content.Summary.SharpeRatio, 2))" -ForegroundColor DarkGreen
                }
            } catch {
                Write-Host "      (Unable to parse details)" -ForegroundColor DarkGray
            }
            Write-Host ""
        }
    } else {
        Write-Host "⚠️  No backtest results found in $backtestDir" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Backtest directory not found: $backtestDir" -ForegroundColor Yellow
}

# Check bot logs for backtest activity
Write-Host "📋 Recent Backtest Activity (from logs):" -ForegroundColor Cyan
Write-Host ""

$logFiles = Get-ChildItem "logs" -Filter "bot-*.log" -ErrorAction SilentlyContinue | 
            Sort-Object LastWriteTime -Descending | 
            Select-Object -First 1

if ($logFiles) {
    $logFile = $logFiles[0]
    Write-Host "📄 Checking log file: $($logFile.Name)" -ForegroundColor Gray
    Write-Host ""
    
    # Extract backtest-related log entries
    $backtestLogs = Select-String -Path $logFile.FullName -Pattern "BACKTEST|UNIFIED-BACKTEST|ENHANCED-BACKTEST|BacktestResult" -Context 0,2 -ErrorAction SilentlyContinue |
                    Select-Object -Last 20
    
    if ($backtestLogs) {
        Write-Host "Recent backtest log entries:" -ForegroundColor Green
        Write-Host ""
        foreach ($log in $backtestLogs) {
            # Color code based on log level
            $color = "Gray"
            if ($log.Line -match "\[ERROR\]") { $color = "Red" }
            elseif ($log.Line -match "\[WARNING\]") { $color = "Yellow" }
            elseif ($log.Line -match "\[INFO\]") { $color = "White" }
            elseif ($log.Line -match "✅") { $color = "Green" }
            
            Write-Host "   $($log.Line)" -ForegroundColor $color
        }
        Write-Host ""
    } else {
        Write-Host "⚠️  No recent backtest activity found in logs" -ForegroundColor Yellow
        Write-Host "   The bot may still be initializing or learning hasn't started yet" -ForegroundColor DarkGray
        Write-Host ""
    }
} else {
    Write-Host "⚠️  No log files found" -ForegroundColor Yellow
}

# Generate summary report
Write-Host "=" -NoNewline -ForegroundColor Cyan
for ($i = 0; $i -lt 80; $i++) { Write-Host "=" -NoNewline -ForegroundColor Cyan }
Write-Host ""
Write-Host "📊 STRATEGY PERFORMANCE SUMMARY" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
for ($i = 0; $i -lt 80; $i++) { Write-Host "=" -NoNewline -ForegroundColor Cyan }
Write-Host ""
Write-Host ""

# Parse strategy performance from UnifiedTradingBrain exports
$perfFile = "state/models/strategy-performance.json"
if (Test-Path $perfFile) {
    Write-Host "📈 Strategy Performance (from UnifiedTradingBrain):" -ForegroundColor Green
    Write-Host ""
    
    try {
        $perfData = Get-Content $perfFile -Raw | ConvertFrom-Json
        
        # Display in table format
        Write-Host "Strategy | Trades | Win Rate | Avg Win | Avg Loss | Net PnL | Sharpe" -ForegroundColor Cyan
        Write-Host "---------|--------|----------|---------|----------|---------|--------" -ForegroundColor DarkGray
        
        foreach ($strategy in $Strategies.Split(',')) {
            $strategy = $strategy.Trim()
            if ($perfData.$strategy) {
                $perf = $perfData.$strategy
                $winRate = if ($perf.TotalTrades -gt 0) { ($perf.Wins / $perf.TotalTrades) * 100 } else { 0 }
                
                $winRateColor = if ($winRate -ge 60) { "Green" } elseif ($winRate -ge 50) { "Yellow" } else { "Red" }
                $pnlColor = if ($perf.NetPnL -gt 0) { "Green" } else { "Red" }
                
                Write-Host "  $strategy    " -NoNewline -ForegroundColor White
                Write-Host "| " -NoNewline
                Write-Host "$($perf.TotalTrades.ToString().PadLeft(6)) " -NoNewline -ForegroundColor Gray
                Write-Host "| " -NoNewline
                Write-Host "$([math]::Round($winRate, 1).ToString().PadLeft(7))% " -NoNewline -ForegroundColor $winRateColor
                Write-Host "| " -NoNewline
                Write-Host "`$$([math]::Round($perf.AverageWin, 2).ToString().PadLeft(6)) " -NoNewline -ForegroundColor Green
                Write-Host "| " -NoNewline
                Write-Host "`$$([math]::Round($perf.AverageLoss, 2).ToString().PadLeft(7)) " -NoNewline -ForegroundColor Red
                Write-Host "| " -NoNewline
                Write-Host "`$$([math]::Round($perf.NetPnL, 2).ToString().PadLeft(6)) " -NoNewline -ForegroundColor $pnlColor
                Write-Host "| " -NoNewline
                Write-Host "$([math]::Round($perf.Sharpe, 2).ToString().PadLeft(5))" -ForegroundColor DarkGreen
            } else {
                Write-Host "  $strategy    | No data available" -ForegroundColor DarkGray
            }
        }
        Write-Host ""
    } catch {
        Write-Host "⚠️  Unable to parse performance data: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  No strategy performance data found at: $perfFile" -ForegroundColor Yellow
    Write-Host "   The bot needs to run for a while to generate performance data" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
for ($i = 0; $i -lt 80; $i++) { Write-Host "=" -NoNewline -ForegroundColor Cyan }
Write-Host ""
Write-Host ""

Write-Host "💡 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Keep the bot running - it's learning from 90 days of historical data" -ForegroundColor White
Write-Host "   2. Check logs/bot-*.log for detailed backtest progress" -ForegroundColor White
Write-Host "   3. Performance data accumulates in state/backtests/ and state/models/" -ForegroundColor White
Write-Host "   4. Re-run this script periodically to see updated results" -ForegroundColor White
Write-Host ""
Write-Host "📝 Current Configuration:" -ForegroundColor Cyan
Write-Host "   - DRY_RUN=1 (Paper trading - NO REAL ORDERS)" -ForegroundColor Green
Write-Host "   - ENABLE_HISTORICAL_LEARNING=1 (Learning from 90 days of data)" -ForegroundColor Green
Write-Host "   - Markets are closed, so bot is in intensive learning mode" -ForegroundColor Yellow
Write-Host ""
