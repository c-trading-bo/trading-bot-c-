#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start TopstepX Python adapter in stream mode with environment variables from .env
.DESCRIPTION
    Loads .env file and starts the Python adapter to provide historical data on localhost:8765
#>

# Load .env file
Write-Host "📦 Loading environment variables from .env..." -ForegroundColor Cyan
if (Test-Path ".env") {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)\s*=\s*(.+)\s*$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            if ($name -match "API_KEY") {
                Write-Host "  ✅ $name = $($value.Substring(0, [Math]::Min(10, $value.Length)))..." -ForegroundColor Green
            } else {
                Write-Host "  ✅ $name = $value" -ForegroundColor Green
            }
        }
    }
} else {
    Write-Host "❌ .env file not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🚀 Starting TopstepX adapter in stream mode..." -ForegroundColor Cyan
Write-Host "   📊 Historical data will be available on localhost:8765" -ForegroundColor Yellow
Write-Host "   ⚡ Bot will be able to learn from historical data" -ForegroundColor Yellow
Write-Host ""

# Get Python executable from environment variable
$pythonExe = [Environment]::GetEnvironmentVariable("PYTHON_EXECUTABLE", "Process")
if (-not $pythonExe -or -not (Test-Path $pythonExe)) {
    # Fallback to 'python' command
    $pythonExe = "python"
}

Write-Host "   🐍 Using Python: $pythonExe" -ForegroundColor Gray
Write-Host ""

# Start the adapter
& $pythonExe src/adapters/topstep_x_adapter.py stream
