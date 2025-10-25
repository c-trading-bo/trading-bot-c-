# Lab Mode Pre-Flight Check Script (PowerShell)
# Validates that all required historical data files exist and are valid

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lab Mode Pre-Flight Validation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$pass = 0
$warn = 0
$fail = 0

# Check 1: Data directory exists
Write-Host -NoNewline "Checking data directory... "
if (Test-Path "data\historical" -PathType Container) {
    Write-Host "✓ PASS" -ForegroundColor Green
    $pass++
} else {
    Write-Host "✗ FAIL" -ForegroundColor Red
    Write-Host "  Directory 'data\historical' does not exist" -ForegroundColor Red
    Write-Host "  Run: mkdir data\historical" -ForegroundColor Yellow
    $fail++
    exit 1
}

# Check 2: Required 5m data files exist
Write-Host ""
Write-Host "Checking required 5-minute data files..."
foreach ($symbol in @("ES", "NQ")) {
    $file = "data\historical\${symbol}_90days.json"
    Write-Host -NoNewline "  ${symbol}_90days.json... "
    
    if (Test-Path $file) {
        $size = (Get-Item $file).Length
        if ($size -gt 102400) { # > 100 KB
            # Check if file contains expected JSON structure
            $content = Get-Content $file -Raw
            if ($content -match '"bars"' -and $content -match '"timestamp"') {
                # Consistent size formatting
                if ($size -ge 1MB) {
                    $sizeFormatted = "{0:N1} MB" -f ($size / 1MB)
                } else {
                    $sizeFormatted = "{0:N0} KB" -f ($size / 1KB)
                }
                Write-Host "✓ PASS ($sizeFormatted)" -ForegroundColor Green
                $pass++
            } else {
                Write-Host "✗ FAIL (Invalid JSON structure)" -ForegroundColor Red
                $fail++
            }
        } else {
            $sizeFormatted = "{0:N0} bytes" -f $size
            Write-Host "✗ FAIL (File too small: $sizeFormatted)" -ForegroundColor Red
            Write-Host "     Expected > 100 KB" -ForegroundColor Yellow
            $fail++
        }
    } else {
        Write-Host "✗ FAIL (File not found)" -ForegroundColor Red
        $fail++
    }
}

# Check 3: Optional 1m data files
Write-Host ""
Write-Host "Checking optional 1-minute data files..."
foreach ($symbol in @("ES", "NQ")) {
    $file = "data\historical\${symbol}_1m_90days.json"
    Write-Host -NoNewline "  ${symbol}_1m_90days.json... "
    
    if (Test-Path $file) {
        $size = (Get-Item $file).Length
        if ($size -gt 102400) {
            # Consistent size formatting
            if ($size -ge 1MB) {
                $sizeFormatted = "{0:N1} MB" -f ($size / 1MB)
            } else {
                $sizeFormatted = "{0:N0} KB" -f ($size / 1KB)
            }
            Write-Host "✓ PRESENT ($sizeFormatted)" -ForegroundColor Green
            $pass++
        } else {
            Write-Host "⚠ WARNING (File too small)" -ForegroundColor Yellow
            $warn++
        }
    } else {
        Write-Host "⚠ NOT FOUND (Training will use 5m data only)" -ForegroundColor Yellow
        $warn++
    }
}

# Check 4: Python executable
Write-Host ""
Write-Host -NoNewline "Checking Python availability... "
try {
    $pythonVersion = (python --version 2>&1) -join ""
    if ($pythonVersion -match "Python") {
        Write-Host "✓ PASS ($pythonVersion)" -ForegroundColor Green
        $pass++
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "⚠ WARNING (Python not found in PATH)" -ForegroundColor Yellow
    Write-Host "  Python is needed to fetch historical data" -ForegroundColor Yellow
    Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    $warn++
}

# Check 5: fetch-and-save-historical-data.py script
Write-Host ""
Write-Host -NoNewline "Checking data fetch script... "
if (Test-Path "fetch-and-save-historical-data.py") {
    Write-Host "✓ PASS" -ForegroundColor Green
    $pass++
} else {
    Write-Host "⚠ WARNING" -ForegroundColor Yellow
    Write-Host "  Script 'fetch-and-save-historical-data.py' not found" -ForegroundColor Yellow
    Write-Host "  Data refresh will not be available" -ForegroundColor Yellow
    $warn++
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Passed:   $pass" -ForegroundColor Green
Write-Host "Warnings: $warn" -ForegroundColor Yellow
Write-Host "Failed:   $fail" -ForegroundColor Red
Write-Host ""

if ($fail -eq 0) {
    Write-Host "✅ All critical checks passed!" -ForegroundColor Green
    Write-Host "Lab Mode should be able to start training." -ForegroundColor Green
    Write-Host ""
    Write-Host "To launch Lab Mode:" -ForegroundColor Cyan
    Write-Host '  $env:FORCE_LAB_NOW="1"; dotnet run --project src\UnifiedOrchestrator' -ForegroundColor White
    exit 0
} else {
    Write-Host "❌ Some critical checks failed." -ForegroundColor Red
    Write-Host ""
    Write-Host "To fix missing data files:" -ForegroundColor Cyan
    Write-Host "  python fetch-and-save-historical-data.py" -ForegroundColor White
    Write-Host ""
    Write-Host "For more help, see:" -ForegroundColor Cyan
    Write-Host "  LAB_MODE_STARTUP_TROUBLESHOOTING.md" -ForegroundColor White
    exit 1
}
