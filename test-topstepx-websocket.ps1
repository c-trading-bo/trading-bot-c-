# TopstepX WebSocket Connection Diagnostic Wrapper
# Loads .env file and runs Python diagnostic script

Write-Output "🔧 Loading .env file..."

# Load .env file
if (Test-Path ".env") {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            # Remove quotes if present
            $value = $value -replace '^["'']|["'']$', ''
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            
            # Show masked value for sensitive vars
            if ($name -match 'KEY|JWT|PASSWORD') {
                $masked = if ($value.Length -gt 12) { 
                    $value.Substring(0, 8) + "..." + $value.Substring($value.Length - 4) 
                } else { 
                    "***" 
                }
                Write-Output "  ✅ $name = $masked"
            } else {
                Write-Output "  ✅ $name = $value"
            }
        }
    }
    Write-Output "✅ Environment loaded from .env"
} else {
    Write-Output "❌ .env file not found"
    exit 1
}

Write-Output ""
Write-Output "🚀 Running TopstepX diagnostic tests..."
Write-Output ""

# Get Python path from environment or use default
$pythonPath = $env:PYTHON_EXECUTABLE
if (-not $pythonPath) {
    $pythonPath = "C:\Users\kevin\AppData\Local\Programs\Python\Python312\python.exe"
}

Write-Output "Using Python: $pythonPath"
Write-Output ""

# Run Python diagnostic script
& $pythonPath test-topstepx-websocket.py

$exitCode = $LASTEXITCODE
Write-Output ""
Write-Output "=================================================="
if ($exitCode -eq 0) {
    Write-Output "✅ All diagnostics passed"
} else {
    Write-Output "❌ Some diagnostics failed (exit code: $exitCode)"
}
Write-Output "=================================================="

exit $exitCode
