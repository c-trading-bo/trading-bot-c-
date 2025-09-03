# Auto-Background Bot Mechanic Launcher
# Runs fully automatically in background with bot

Write-Host "🤖 Starting Bot with Auto-Background Mechanic..." -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Create necessary directories silently
$directories = @(
    "Intelligence",
    "Intelligence\mechanic",
    "Intelligence\mechanic\local", 
    "Intelligence\mechanic\database",
    "Intelligence\mechanic\reports",
    "Intelligence\scripts",
    "Intelligence\scripts\strategies",
    "Intelligence\scripts\ml",
    "Intelligence\scripts\data",
    "Intelligence\data",
    "Intelligence\models"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Auto-install required packages silently
Write-Host "📦 Auto-checking dependencies..." -ForegroundColor Yellow
$packages = @("pandas", "numpy", "scikit-learn", "requests", "yfinance", "flask")

foreach ($package in $packages) {
    python -c "import $($package.Replace('-', '_').Split('.')[0])" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        pip install $package --quiet | Out-Null
    }
}

Write-Host "🚀 Starting Auto-Background Mechanic (silent mode)..." -ForegroundColor Cyan

# Start the auto-background mechanic in background
Start-Process -FilePath "python" -ArgumentList "auto_background_mechanic.py" -WindowStyle Hidden -PassThru

Write-Host "✅ Auto-Background Mechanic started successfully" -ForegroundColor Green
Write-Host "🔄 Running silently in background with all features" -ForegroundColor Green
Write-Host "🧠 Auto-fixing issues, monitoring health, generating reports" -ForegroundColor Green
Write-Host "📊 Dashboard integration active" -ForegroundColor Green
Write-Host ""
Write-Host "Your bot mechanic is now running fully automatically!" -ForegroundColor Cyan
