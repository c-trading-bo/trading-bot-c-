#!/usr/bin/env pwsh
# ========================================
# LOCAL BOT MECHANIC - INTEGRATED LAUNCHER
# STRICTLY LOCAL ONLY - NO CLOUD FEATURES
# Auto-starts with dashboard integration
# ========================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "🧠 LOCAL BOT MECHANIC - INTEGRATED DASHBOARD LAUNCHER" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "STRICTLY LOCAL - runs on YOUR computer only" -ForegroundColor Green
Write-Host "Auto-integrates with main bot dashboard" -ForegroundColor Green
Write-Host "No cloud features, no GitHub minutes used" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and ensure it's in your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create directories if they don't exist
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "Intelligence\mechanic\local" -Force | Out-Null
New-Item -ItemType Directory -Path "Intelligence\mechanic\database" -Force | Out-Null
Write-Host "✅ Directories created" -ForegroundColor Green

Write-Host "🔧 Installing/checking dependencies..." -ForegroundColor Yellow

# Install required Python packages
$packages = @("pandas", "numpy", "scikit-learn", "requests", "yfinance", "flask")
foreach ($package in $packages) {
    Write-Host "  📦 Installing $package..." -NoNewline
    try {
        & python -m pip install $package --quiet 2>&1 | Out-Null
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " ⚠️" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🚀 Starting Local Bot Mechanic with Dashboard Integration..." -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "🌐 Dashboard will be available at: http://localhost:5051" -ForegroundColor Cyan
Write-Host "🔗 Will auto-integrate with main dashboard at: http://localhost:5050" -ForegroundColor Cyan
Write-Host "🔍 Auto-monitoring and repair enabled" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Start the integrated auto-launcher
try {
    & python "Intelligence\mechanic\local\start_local_mechanic.py"
} catch {
    Write-Host "❌ Failed to start Local Bot Mechanic: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
