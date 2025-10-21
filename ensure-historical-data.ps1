# Automated Historical Data Maintenance Script
# Ensures bot always has fresh 90-day historical data before startup

param(
    [switch]$Force  # Force refresh even if data exists
)

Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Historical Data Maintenance - Auto Refresh System     ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ ERROR: .env file not found" -ForegroundColor Red
    Write-Host "   Please copy .env.example to .env and configure your credentials" -ForegroundColor Yellow
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python not found in PATH" -ForegroundColor Red
    exit 1
}

# Check if python-dotenv is installed
Write-Host "🔍 Checking Python dependencies..." -ForegroundColor Cyan
$dotenvCheck = pip show python-dotenv 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "📦 Installing python-dotenv..." -ForegroundColor Yellow
    pip install python-dotenv | Out-Null
    Write-Host "✅ python-dotenv installed" -ForegroundColor Green
}

# Check if historical data exists and is recent
$needsRefresh = $false
$esFile = "data\historical\ES_90days.json"
$nqFile = "data\historical\NQ_90days.json"

if ($Force) {
    Write-Host "🔄 Force refresh requested" -ForegroundColor Yellow
    $needsRefresh = $true
} elseif (-not (Test-Path $esFile) -or -not (Test-Path $nqFile)) {
    Write-Host "⚠️  Historical data files missing" -ForegroundColor Yellow
    $needsRefresh = $true
} else {
    # Check if data is older than 24 hours
    $esAge = (Get-Date) - (Get-Item $esFile).LastWriteTime
    $nqAge = (Get-Date) - (Get-Item $nqFile).LastWriteTime
    
    if ($esAge.TotalHours -gt 24 -or $nqAge.TotalHours -gt 24) {
        Write-Host "⚠️  Historical data is older than 24 hours" -ForegroundColor Yellow
        $needsRefresh = $true
    } else {
        Write-Host "✅ Historical data is recent (< 24 hours old)" -ForegroundColor Green
        
        # Show current data stats
        try {
            $esData = Get-Content $esFile | ConvertFrom-Json
            $nqData = Get-Content $nqFile | ConvertFrom-Json
            Write-Host "   ES: $($esData.bars.Count) bars" -ForegroundColor Gray
            Write-Host "   NQ: $($nqData.bars.Count) bars" -ForegroundColor Gray
        } catch {
            Write-Host "⚠️  Could not read data files, will refresh" -ForegroundColor Yellow
            $needsRefresh = $true
        }
    }
}

if ($needsRefresh) {
    Write-Host ""
    Write-Host "📥 Fetching 90 days of historical data..." -ForegroundColor Cyan
    Write-Host "   This may take 2-5 minutes..." -ForegroundColor Gray
    Write-Host ""
    
    # Run Python script with dotenv loaded and FULL refresh mode
    $pythonScript = @"
from dotenv import load_dotenv
import os
import subprocess

# Load .env file
load_dotenv()

# Verify credentials are loaded
api_key = os.getenv('TOPSTEPX_API_KEY') or os.getenv('PROJECT_X_API_KEY')
username = os.getenv('TOPSTEPX_USERNAME') or os.getenv('PROJECT_X_USERNAME')

if not api_key or not username:
    print('❌ ERROR: TopstepX credentials not found in .env')
    print('   Required: TOPSTEPX_API_KEY and TOPSTEPX_USERNAME')
    exit(1)

print(f'✅ Credentials loaded: {username}')

# Set refresh mode to FULL for complete 90-day fetch
os.environ['REFRESH_MODE'] = 'full'

# Run the historical data fetch script
result = subprocess.run(['python', 'fetch-and-save-historical-data.py'])
exit(result.returncode)
"@
    
    # Execute the Python script
    $pythonScript | python
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Historical data refresh completed successfully!" -ForegroundColor Green
        
        # Show updated stats
        try {
            $esData = Get-Content $esFile | ConvertFrom-Json
            $nqData = Get-Content $nqFile | ConvertFrom-Json
            Write-Host "   ES: $($esData.bars.Count) bars" -ForegroundColor Gray
            Write-Host "   NQ: $($nqData.bars.Count) bars" -ForegroundColor Gray
            
            # Calculate date range
            $firstDate = [DateTime]::Parse($esData.bars[0].timestamp)
            $lastDate = [DateTime]::Parse($esData.bars[-1].timestamp)
            $daysCovered = ($lastDate - $firstDate).Days
            Write-Host "   Coverage: $daysCovered days ($($firstDate.ToString('MMM dd')) - $($lastDate.ToString('MMM dd yyyy')))" -ForegroundColor Gray
        } catch {
            Write-Host "   Data files created successfully" -ForegroundColor Gray
        }
    } else {
        Write-Host ""
        Write-Host "❌ Historical data refresh FAILED" -ForegroundColor Red
        Write-Host "   Check the error messages above" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✅ No refresh needed - data is current" -ForegroundColor Green
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ Historical data maintenance complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
